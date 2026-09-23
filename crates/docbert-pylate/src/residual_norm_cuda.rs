//! BF16 residual addition and zero-bias LayerNorm with Candle's reduction order.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp3,
    D,
    DType,
    Layout,
    Result,
    Shape,
    Tensor,
    backend::BackendStorage,
    cuda_backend::{
        CudaStorageSlice,
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
};
use candle_nn::LayerNorm;

struct ResidualNorm {
    eps: f32,
}

#[cfg(test)]
thread_local! {
    pub(crate) static DISPATCH_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl CustomOp3 for ResidualNorm {
    fn name(&self) -> &'static str {
        "docbert_residual_norm_bf16"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("residual norm CUDA operation received CPU storage")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        y: &CudaStorage,
        yl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (
            CudaStorageSlice::BF16(xs),
            CudaStorageSlice::BF16(ys),
            CudaStorageSlice::BF16(weights),
        ) = (&x.slice, &y.slice, &weight.slice)
        else {
            candle_core::bail!("fused residual norm requires BF16")
        };
        let cols = xl.shape().dim(D::Minus1)?;
        let count = xl.shape().elem_count();
        let rows = count / cols;
        let device = x.device();
        let function = device.get_or_load_custom_func(
            "add_and_normalize_bf16",
            "docbert_residual_norm",
            include_str!(concat!(env!("OUT_DIR"), "/residual_norm.ptx")),
        )?;
        let x = xs.slice(xl.start_offset()..xl.start_offset() + count);
        let y = ys.slice(yl.start_offset()..yl.start_offset() + count);
        let weight = weights.slice(wl.start_offset()..wl.start_offset() + cols);
        // The launch writes both halves before either output tensor is read.
        let mut output = unsafe { device.alloc(2 * count)? };
        let (count, cols) = (count as u32, cols as i32);
        let mut launch = function.builder();
        launch
            .arg(&x)
            .arg(&y)
            .arg(&weight)
            .arg(&mut output)
            .arg(&count)
            .arg(&cols)
            .arg(&self.eps);
        // The caller validates shapes, contiguous BF16 storage and index bounds.
        // Scalar loads need only BF16 alignment, including odd storage offsets.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (if cols < 1024 { 32 } else { 1024 }, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        #[cfg(test)]
        DISPATCH_COUNT.set(DISPATCH_COUNT.get() + 1);
        let mut dims = vec![2];
        dims.extend_from_slice(xl.dims());
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: device.clone(),
            },
            Shape::from(dims),
        ))
    }
}

/// The norm is constructed with a persistent positive-zero bias on CUDA.
pub(crate) fn add_and_normalize(
    x: &Tensor,
    y: &Tensor,
    norm: &LayerNorm,
) -> Result<(Tensor, Tensor)> {
    let cols = x.dim(D::Minus1)?;
    let weight = norm.weight();
    if x.device().is_cuda()
        && (1..=1024).contains(&cols)
        && x.elem_count() > 0
        && x.elem_count() <= u32::MAX as usize / 2
        && x.dims() == y.dims()
        && weight.dims() == [cols]
        && norm.remove_mean()
        && norm.bias().is_some()
        && [x, y, weight].iter().all(|tensor| {
            tensor.is_contiguous()
                && tensor.dtype() == DType::BF16
                && tensor.device().same_device(x.device())
        })
    {
        let output = x.apply_op3_no_bwd(
            y,
            weight,
            &ResidualNorm {
                eps: norm.eps() as f32,
            },
        )?;
        Ok((output.get(0)?, output.get(1)?))
    } else {
        let residual = (x + y)?;
        let normalized = residual.apply(norm)?;
        Ok((residual, normalized))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Storage};

    use super::*;

    fn read_bits(tensor: &Tensor) -> Result<Vec<u32>> {
        let tensor = tensor.flatten_all()?.to_device(&Device::Cpu)?;
        let (storage, layout) = tensor.storage_and_layout();
        let range =
            layout.start_offset()..layout.start_offset() + tensor.elem_count();
        match &*storage {
            Storage::Cpu(CpuStorage::BF16(values)) => Ok(values[range]
                .iter()
                .map(|value| u32::from(value.to_bits()))
                .collect()),
            Storage::Cpu(CpuStorage::F16(values)) => Ok(values[range]
                .iter()
                .map(|value| u32::from(value.to_bits()))
                .collect()),
            Storage::Cpu(CpuStorage::F32(values)) => {
                Ok(values[range].iter().map(|value| value.to_bits()).collect())
            }
            _ => panic!("unsupported residual norm test dtype"),
        }
    }

    fn assert_matches_candle(
        x: &Tensor,
        y: &Tensor,
        weight: &Tensor,
        eps: f64,
        dispatches: usize,
    ) -> Result<()> {
        let norm = LayerNorm::new(weight.clone(), weight.zeros_like()?, eps);
        let residual = (x + y)?;
        let normalized = residual.apply(&norm)?;
        let before = DISPATCH_COUNT.get();
        let (actual_residual, actual_normalized) =
            add_and_normalize(x, y, &norm)?;
        assert_eq!(
            DISPATCH_COUNT.get() - before,
            dispatches,
            "unexpected residual norm dispatch"
        );
        for (name, actual, expected) in [
            ("residual", actual_residual, residual),
            ("normalized", actual_normalized, normalized),
        ] {
            assert_eq!(actual.dims(), expected.dims());
            for (index, (actual, expected)) in read_bits(&actual)?
                .iter()
                .zip(read_bits(&expected)?)
                .enumerate()
            {
                assert_eq!(
                    *actual,
                    expected,
                    "{name} {:?} eps={eps}, element {index}",
                    x.dims()
                );
            }
        }
        Ok(())
    }

    fn build_tensor(
        values: &[f32],
        cols: usize,
        offset: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut storage = vec![0.0; offset];
        storage.extend_from_slice(values);
        let len = storage.len();
        Tensor::from_vec(storage, len, device)?
            .to_dtype(DType::BF16)?
            .narrow(0, offset, values.len())?
            .reshape((values.len() / cols, cols))
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn residual_norm_matches_candle_bits() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for cols in [1, 7, 31, 32, 33, 64, 128, 768, 1023, 1024] {
            // Odd row counts and widths exercise partial warps and offset views.
            let rows = 37;
            let values: Vec<_> = (0..rows * cols)
                .map(|i| match i / cols {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1.0,
                    3 => -17.0,
                    row => {
                        let scale = [1.0, 1e-20, 3e4, 1e18, 3e38][row % 5];
                        ((i * 37 % 4093) as f32 - 2046.0) / 2046.0 * scale
                    }
                })
                .collect();
            let cancellation: Vec<_> = values
                .iter()
                .enumerate()
                .map(
                    |(i, value)| {
                        if i % 3 == 0 { -*value } else { -*value * 0.996 }
                    },
                )
                .collect();
            let weights: Vec<_> = (0..cols)
                .map(|i| ((i * 13 % 257) as f32 - 128.0) / 97.0)
                .collect();
            for offset in [0, 1, 8] {
                let x = build_tensor(&values, cols, offset, &device)?;
                let y = build_tensor(&cancellation, cols, offset + 2, &device)?;
                let weight = build_tensor(&weights, cols, offset + 3, &device)?
                    .flatten_all()?;
                for eps in [1e-5, 1e-12, 0.1] {
                    assert_matches_candle(&x, &y, &weight, eps, 1)?;
                    assert_matches_candle(&x, &x, &weight, eps, 1)?;
                }
            }
        }
        let values: Vec<_> = (0u32..=65535)
            .filter(|value| value & 0x7f80 != 0x7f80)
            .map(|value| f32::from_bits(value << 16))
            .collect();
        assert_eq!(values.len(), 85 * 768);
        let x = build_tensor(&values, 768, 1, &device)?;
        let reversed: Vec<_> = values.iter().copied().rev().collect();
        let y = build_tensor(&reversed, 768, 3, &device)?;
        let weight = Tensor::ones(768, DType::BF16, &device)?;
        for eps in [1e-5, 1e-12, 0.1] {
            assert_matches_candle(&x, &y, &weight, eps, 1)?;
            assert_matches_candle(&x, &x, &weight, eps, 1)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn residual_norm_preserves_layouts_and_fallbacks() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values: Vec<_> =
            (0..6 * 768).map(|i| (i as f32 - 2304.0) / 97.0).collect();
        let x = build_tensor(&values, 768, 0, &device)?;
        let weight = Tensor::ones(768, DType::BF16, &device)?;
        let query = x.reshape((2, 3, 768))?;
        assert_matches_candle(&query, &query, &weight, 1e-5, 1)?;
        for dtype in [DType::F16, DType::F32] {
            let x = x.to_dtype(dtype)?;
            assert_matches_candle(&x, &x, &weight.to_dtype(dtype)?, 1e-5, 0)?;
        }
        let strided = x.narrow(1, 0, 128)?;
        let narrow_weight = weight.narrow(0, 0, 128)?;
        assert_matches_candle(&strided, &strided, &narrow_weight, 1e-5, 0)?;
        let wide = x.reshape((3, 1536))?;
        let wide_weight = Tensor::ones(1536, DType::BF16, &device)?;
        assert_matches_candle(&wide, &wide, &wide_weight, 1e-5, 0)?;
        let cpu = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let cpu_weight =
            weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        assert_matches_candle(&cpu, &cpu, &cpu_weight, 1e-5, 0)?;
        Ok(())
    }
}

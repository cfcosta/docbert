//! BF16 Q/K rotary embedding with one launch and one output allocation.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp3,
    DType,
    Layout,
    Result,
    Shape,
    Storage,
    Tensor,
    backend::BackendStorage,
    cuda_backend::{
        CudaStorageSlice,
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    },
};

struct RotaryPair {
    sin: Tensor,
}

#[cfg(test)]
thread_local! {
    static DISPATCH_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl CustomOp3 for RotaryPair {
    fn name(&self) -> &'static str {
        "docbert_rotary_pair_bf16"
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
        candle_core::bail!("paired rotary CUDA operation received CPU storage")
    }

    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        q_layout: &Layout,
        k: &CudaStorage,
        k_layout: &Layout,
        cos: &CudaStorage,
        cos_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (sin_storage, sin_layout) = self.sin.storage_and_layout();
        let Storage::Cuda(sin) = &*sin_storage else {
            candle_core::bail!("paired rotary sine table requires CUDA")
        };
        let (
            CudaStorageSlice::BF16(q_values),
            CudaStorageSlice::BF16(k_values),
            CudaStorageSlice::BF16(cos_values),
            CudaStorageSlice::BF16(sin_values),
        ) = (&q.slice, &k.slice, &cos.slice, &sin.slice)
        else {
            candle_core::bail!("paired rotary requires BF16")
        };
        let (_, tokens, heads, dim) = q_layout.shape().dims4()?;
        let count = q_layout.shape().elem_count();
        let table_count = tokens * (dim / 2);
        let device = q.device();
        let function = device.get_or_load_custom_func(
            "apply_rotary_pair_bf16",
            "docbert_rope_pair",
            include_str!(concat!(env!("OUT_DIR"), "/rope_pair.ptx")),
        )?;
        let q = q_values
            .slice(q_layout.start_offset()..q_layout.start_offset() + count);
        let k = k_values
            .slice(k_layout.start_offset()..k_layout.start_offset() + count);
        let cos = cos_values.slice(
            cos_layout.start_offset()..cos_layout.start_offset() + table_count,
        );
        let sin = sin_values.slice(
            sin_layout.start_offset()..sin_layout.start_offset() + table_count,
        );
        // The launch writes both halves before either output tensor is read.
        let mut output = unsafe { device.alloc(2 * count)? };
        let (count, tokens, heads, dim) =
            (count as u32, tokens as u32, heads as u32, dim as u32);
        let mut launch = function.builder();
        launch
            .arg(&q)
            .arg(&k)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&count)
            .arg(&tokens)
            .arg(&heads)
            .arg(&dim);
        // The caller validates shapes, contiguous BF16 storage and index bounds.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((count / 2).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        #[cfg(test)]
        DISPATCH_COUNT.set(DISPATCH_COUNT.get() + 1);
        let mut dims = vec![2];
        dims.extend_from_slice(q_layout.dims());
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: device.clone(),
            },
            Shape::from(dims),
        ))
    }
}

pub(crate) fn apply_rotary_pair(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, tokens, heads, dim) = q.dims4()?;
    // Dispatch covers FlashAttention head widths and aligned storage views.
    // The scalar kernel itself accepts any positive even head width.
    if q.device().is_cuda()
        && batch > 0
        && tokens > 0
        && heads > 0
        && (8..=512).contains(&dim)
        && dim.is_multiple_of(8)
        && q.elem_count() <= u32::MAX as usize / 2
        && q.dims() == k.dims()
        && matches!(cos.dims(), &[rows, cols] if rows >= tokens && cols == dim / 2)
        && cos.dims() == sin.dims()
        && [q, k, cos, sin].iter().all(|tensor| {
            tensor.is_contiguous()
                && tensor.layout().start_offset().is_multiple_of(2)
                && tensor.dtype() == DType::BF16
                && tensor.device().same_device(q.device())
        })
    {
        let output =
            q.apply_op3_no_bwd(k, cos, &RotaryPair { sin: sin.clone() })?;
        // Both views stay contiguous, including K's nonzero storage offset.
        // FlashAttention consumes their strides and offsets without a copy.
        Ok((output.get(0)?, output.get(1)?))
    } else {
        let cos = cos.contiguous()?;
        let sin = sin.contiguous()?;
        Ok((
            candle_nn::rotary_emb::rope_thd(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope_thd(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn build_tensor(
        values: &[f32],
        dims: &[usize],
        offset: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut storage = vec![0.0; offset];
        storage.extend_from_slice(values);
        let len = storage.len();
        Tensor::from_vec(storage, len, device)?
            .to_dtype(DType::BF16)?
            .narrow(0, offset, values.len())?
            .reshape(dims)
    }

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
            _ => panic!("unsupported rotary test dtype"),
        }
    }

    fn assert_matches_candle(
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        dispatches: usize,
    ) -> Result<()> {
        let before = DISPATCH_COUNT.get();
        let (actual_q, actual_k) = apply_rotary_pair(q, k, cos, sin)?;
        assert_eq!(DISPATCH_COUNT.get() - before, dispatches);
        if dispatches == 1 {
            assert_eq!(actual_q.layout().start_offset(), 0);
            assert_eq!(actual_k.layout().start_offset(), q.elem_count());
            let (q_storage, _) = actual_q.storage_and_layout();
            let (k_storage, _) = actual_k.storage_and_layout();
            assert!(std::ptr::eq(&*q_storage, &*k_storage));
        }
        for (input, actual) in [(q, actual_q), (k, actual_k)] {
            let expected = candle_nn::rotary_emb::rope_thd(
                &input.contiguous()?,
                &cos.contiguous()?,
                &sin.contiguous()?,
            )?;
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(actual.dtype(), expected.dtype());
            assert!(actual.is_contiguous());
            assert_eq!(actual.contiguous()?.id(), actual.id());
            let actual = read_bits(&actual)?;
            let expected = read_bits(&expected)?;
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in
                actual.iter().zip(&expected).enumerate()
            {
                assert_eq!(
                    actual,
                    expected,
                    "shape {:?}, element {index}",
                    input.dims()
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn rotary_matches_candle_bits() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for (batch, tokens, heads, dim) in [
            (1, 1, 1, 16),
            (1, 7, 3, 80),
            (1, 129, 12, 64),
            (1, 85, 12, 64),
            (3, 7, 3, 80),
            (2, 33, 12, 64),
            (1, 7, 3, 20),
        ] {
            let count = batch * tokens * heads * dim;
            let values: Vec<f32> = if tokens == 85 {
                (0u32..=65535)
                    .filter(|bits| bits & 0x7f80 != 0x7f80)
                    .map(|bits| f32::from_bits(bits << 16))
                    .collect()
            } else {
                (0..count)
                    .map(|index| ((index * 37 % 4093) as f32 - 2046.0) / 127.0)
                    .collect()
            };
            let reversed: Vec<_> = values.iter().copied().rev().collect();
            // Extra rows match the query path's full position tables.
            let table_rows = if batch == 1 { tokens } else { tokens + 5 };
            let angles: Vec<_> = (0..table_rows * dim / 2)
                .map(|index| index as f32 / 19.0)
                .collect();
            let cos_values: Vec<_> = angles.iter().map(|v| v.cos()).collect();
            let sin_values: Vec<_> = angles.iter().map(|v| v.sin()).collect();
            for offsets in
                [[0, 0, 0, 0], [8, 16, 24, 32], [1, 0, 0, 0], [0, 0, 1, 0]]
            {
                let dims = [batch, tokens, heads, dim];
                let q = build_tensor(&values, &dims, offsets[0], &device)?;
                let k = build_tensor(&reversed, &dims, offsets[1], &device)?;
                let cos = build_tensor(
                    &cos_values,
                    &[table_rows, dim / 2],
                    offsets[2],
                    &device,
                )?;
                let sin = build_tensor(
                    &sin_values,
                    &[table_rows, dim / 2],
                    offsets[3],
                    &device,
                )?;
                let dispatches = usize::from(
                    dim != 20
                        && offsets
                            .iter()
                            .all(|offset| offset.is_multiple_of(2)),
                );
                assert_matches_candle(&q, &k, &cos, &sin, dispatches)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn matches_candle_for_special_values_and_fallbacks() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let specials = [
            0x0000u32, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0xffc1,
            0x7f81, 0xff81, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
        ];
        let values: Vec<_> = specials
            .iter()
            .flat_map(|&bits| [f32::from_bits(bits << 16); 16])
            .collect();
        let tables: Vec<_> = specials
            .iter()
            .cycle()
            .take(128)
            .map(|&bits| f32::from_bits(bits << 16))
            .collect();
        let q = build_tensor(&values, &[1, 16, 1, 16], 8, &device)?;
        let k = build_tensor(
            &values.iter().copied().rev().collect::<Vec<_>>(),
            &[1, 16, 1, 16],
            16,
            &device,
        )?;
        let cos = build_tensor(&tables, &[16, 8], 24, &device)?;
        let sin = build_tensor(
            &tables.iter().copied().rev().collect::<Vec<_>>(),
            &[16, 8],
            32,
            &device,
        )?;
        assert_matches_candle(&q, &k, &cos, &sin, 1)?;
        // Transposing equal-sized axes changes strides without changing shapes.
        let strided_q = q.transpose(1, 3)?;
        assert!(!strided_q.is_contiguous());
        assert_matches_candle(&strided_q, &k, &cos, &sin, 0)?;
        let strided_cos = cos.t()?.contiguous()?.t()?;
        assert!(!strided_cos.is_contiguous());
        assert_matches_candle(&q, &k, &strided_cos, &sin, 0)?;
        for dtype in [DType::F16, DType::F32] {
            assert_matches_candle(
                &q.to_dtype(dtype)?,
                &k.to_dtype(dtype)?,
                &cos.to_dtype(dtype)?,
                &sin.to_dtype(dtype)?,
                0,
            )?;
        }
        assert_matches_candle(
            &q.to_device(&Device::Cpu)?,
            &k.to_device(&Device::Cpu)?,
            &cos.to_device(&Device::Cpu)?,
            &sin.to_device(&Device::Cpu)?,
            0,
        )?;
        Ok(())
    }
}

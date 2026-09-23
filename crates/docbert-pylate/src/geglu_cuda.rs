//! BF16 CUDA GeGLU reading both halves of the fused input projection.
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp1,
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

struct GeGlu;

#[cfg(test)]
thread_local! {
    static DISPATCH_COUNTS: std::cell::Cell<[usize; 2]> = const {
        std::cell::Cell::new([0; 2])
    };
}

impl CustomOp1 for GeGlu {
    fn name(&self) -> &'static str {
        "docbert_geglu_bf16"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GeGLU CUDA operation received CPU storage")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let CudaStorageSlice::BF16(input) = &storage.slice else {
            candle_core::bail!("fused GeGLU requires BF16 input")
        };
        let width = layout.shape().dim(D::Minus1)?;
        let input_count = layout.shape().elem_count();
        if !layout.is_contiguous()
            || width == 0
            || !width.is_multiple_of(2)
            || input_count == 0
            || input_count > u32::MAX as usize
        {
            candle_core::bail!("unsupported fused GeGLU layout")
        }
        let inter = (width / 2) as u32;
        let count = (input_count / 2) as u32;
        let is_paired =
            inter.is_multiple_of(2) && layout.start_offset().is_multiple_of(2);
        let name = if is_paired {
            "apply_geglu_bf16_pair"
        } else {
            "apply_geglu_bf16"
        };
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/geglu.ptx"));
        let device = storage.device();
        let function =
            device.get_or_load_custom_func(name, "docbert_geglu", ptx)?;
        let input = input
            .slice(layout.start_offset()..layout.start_offset() + input_count);
        // The launch writes every output element before the tensor is read.
        let mut output = unsafe { device.alloc(count as usize)? };
        let mut launch = function.builder();
        launch.arg(&input).arg(&mut output).arg(&count).arg(&inter);
        let threads = if is_paired { count / 2 } else { count };
        let config = LaunchConfig {
            grid_dim: (threads.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // The contiguous input holds two inter-wide halves per output row.
        // Pairwise reads require even widths and offsets; indices fit in u32.
        unsafe { launch.launch(config) }.w()?;
        #[cfg(test)]
        DISPATCH_COUNTS.set({
            let mut counts = DISPATCH_COUNTS.get();
            counts[usize::from(is_paired)] += 1;
            counts
        });
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().expect("GeGLU input has a last dimension") =
            inter as usize;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: device.clone(),
            },
            Shape::from(dims),
        ))
    }
}

pub(crate) fn apply_geglu(input: &Tensor) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    if input.device().is_cuda()
        && input.dtype() == DType::BF16
        && input.is_contiguous()
        && width > 0
        && width.is_multiple_of(2)
        && input.elem_count() > 0
        && input.elem_count() <= u32::MAX as usize
    {
        input.apply_op1_no_bwd(&GeGlu)
    } else {
        let halves = input.chunk(2, D::Minus1)?;
        halves[0].gelu_erf()? * &halves[1]
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Storage};

    use super::*;

    fn build_input(
        activations: &[u16],
        gates: &[u16],
        inter: usize,
        offset: usize,
        device: &Device,
    ) -> Result<Tensor> {
        assert_eq!(activations.len(), gates.len());
        assert!(activations.len().is_multiple_of(inter));
        let mut bits = vec![0x7fc1u16; offset];
        for (activation, gate) in
            activations.chunks(inter).zip(gates.chunks(inter))
        {
            bits.extend_from_slice(activation);
            bits.extend_from_slice(gate);
        }
        bits.extend_from_slice(&[0x7fc1; 3]);
        let bytes: Vec<_> =
            bits.iter().flat_map(|value| value.to_le_bytes()).collect();
        Tensor::from_raw_buffer(&bytes, DType::BF16, &[bits.len()], device)?
            .narrow(0, offset, activations.len() * 2)?
            .reshape((activations.len() / inter, inter * 2))
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
            _ => panic!("unsupported exactness-test dtype"),
        }
    }

    fn assert_matches_candle(
        input: &Tensor,
        dispatches: [usize; 2],
    ) -> Result<()> {
        let halves = input.chunk(2, D::Minus1)?;
        let expected = (halves[0].gelu_erf()? * &halves[1])?;
        let counts = DISPATCH_COUNTS.get();
        let actual = apply_geglu(input)?;
        assert_eq!(
            DISPATCH_COUNTS.get(),
            [counts[0] + dispatches[0], counts[1] + dispatches[1]],
            "unexpected scalar/pairwise dispatch counts"
        );
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        assert!(actual.is_contiguous());
        let expected = read_bits(&expected)?;
        let actual = read_bits(&actual)?;
        assert_eq!(actual.len(), expected.len());
        let mismatches: Vec<_> = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (actual, expected))| actual != expected)
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "first differing output bits: {mismatches:?}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn matches_candle_for_every_finite_bf16_value() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values: Vec<_> = (0..=u16::MAX)
            .filter(|bits| bits & 0x7f80 != 0x7f80)
            .collect();
        for (inter, dispatches) in [(256, [0, 1]), (255, [1, 0])] {
            // Every finite activation and gate occurs in each shifted sweep.
            for shift in [0, 1, 127, 8191, 32767] {
                let gates: Vec<_> = (0..values.len())
                    .map(|index| values[(index + shift) % values.len()])
                    .collect();
                let input = build_input(&values, &gates, inter, 0, &device)?;
                assert_matches_candle(&input, dispatches)?;
            }
            for gate in [
                0x0000, 0x8000, 0x3f80, 0xbf80, 0x3f00, 0x7f80, 0xff80, 0x7fc0,
                0x0001, 0x8001,
            ] {
                let gates = vec![gate; values.len()];
                let input = build_input(&values, &gates, inter, 0, &device)?;
                assert_matches_candle(&input, dispatches)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn matches_candle_for_special_bf16_values() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let specials = [
            0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0xffc1,
            0x7f81, 0xff81, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
        ];
        let activations: Vec<_> =
            specials.iter().flat_map(|&value| [value; 16]).collect();
        let gates = specials.repeat(specials.len());
        for (inter, offset, dispatches) in
            [(16, 0, [0, 1]), (16, 1, [1, 0]), (1, 0, [1, 0])]
        {
            let input =
                build_input(&activations, &gates, inter, offset, &device)?;
            assert_matches_candle(&input, dispatches)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn matches_candle_for_offsets_tails_and_fallbacks() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for (rows, inter) in [(1, 1), (3, 257), (3, 258), (2, 1152)] {
            let activations: Vec<_> = (0..rows * inter)
                .map(|index| 0xbb00 + (index % 2048) as u16)
                .collect();
            let gates: Vec<_> = (0..rows * inter)
                .map(|index| 0x3b00 + (index * 127 % 2048) as u16)
                .collect();
            for offset in [0, 1, 2, 7, 8] {
                let input =
                    build_input(&activations, &gates, inter, offset, &device)?;
                assert_eq!(input.layout().start_offset(), offset);
                let dispatches =
                    if inter.is_multiple_of(2) && offset.is_multiple_of(2) {
                        [0, 1]
                    } else {
                        [1, 0]
                    };
                assert_matches_candle(&input, dispatches)?;
            }
        }
        let input = build_input(&[0xbf80; 24], &[0x3f80; 24], 4, 2, &device)?;
        assert_matches_candle(&input.reshape((2, 3, 8))?, [0, 1])?;
        assert_matches_candle(&input.narrow(0, 0, 1)?.reshape(8)?, [0, 1])?;
        let strided = input.t()?;
        assert!(!strided.is_contiguous());
        assert_matches_candle(&strided, [0, 0])?;
        let strided = input.narrow(1, 1, 6)?;
        assert!(!strided.is_contiguous());
        assert_matches_candle(&strided, [0, 0])?;
        for dtype in [DType::F16, DType::F32] {
            assert_matches_candle(&input.to_dtype(dtype)?, [0, 0])?;
        }
        assert_matches_candle(&input.to_device(&Device::Cpu)?, [0, 0])?;
        let unsupported = input.narrow(0, 0, 1)?.narrow(1, 0, 7)?;
        let counts = DISPATCH_COUNTS.get();
        assert!(apply_geglu(&unsupported).is_err());
        assert_eq!(DISPATCH_COUNTS.get(), counts);
        Ok(())
    }

    #[test]
    #[ignore = "requires an Ampere or newer CUDA GPU"]
    fn matches_candle_for_model_mlp_dimensions() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let rows = 4133;
        let inter = 1152;
        let mut state = 0xc0ffeeu32;
        let values: Vec<_> = (0..rows * inter * 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32 - 0.5) * 16.0
            })
            .collect();
        let input = Tensor::from_vec(values, (rows, inter * 2), &device)?
            .to_dtype(DType::BF16)?;
        assert_matches_candle(&input, [0, 1])
    }
}

//! Rounded dual GEMM using views of ModernBERT's fused Wi weight.
use std::ffi::c_void;

use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp2,
    DType,
    Layout,
    Result,
    Shape,
    Tensor,
    backend::BackendStorage,
    cuda_backend::{
        CudaStorageSlice,
        DeviceId,
        WrapErr,
        cudarc::{
            cublaslt::sys::cublasLtGetVersion,
            driver::{DevicePtr, DevicePtrMut, sys::CUdevice_attribute},
        },
    },
};

thread_local! {
    static DEVICE_SUPPORT: std::cell::Cell<Option<(DeviceId, bool)>> = const {
        std::cell::Cell::new(None)
    };
}

fn is_supported_device(input: &Tensor) -> bool {
    let Ok(device) = input.device().as_cuda_device() else {
        return false;
    };
    if let Some((id, is_supported)) = DEVICE_SUPPORT.get() {
        if id == device.id() {
            return is_supported;
        }
    }
    let stream = device.cuda_stream();
    let context = stream.context();
    let is_supported = (|| -> Result<bool> {
        context.bind_to_thread().w()?;
        let major = context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).w()?;
        let minor = context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).w()?;
        // This query takes no pointers and does not mutate cuBLAS state.
        let version = unsafe { cublasLtGetVersion() };
        Ok(context.name().w()? == "NVIDIA GeForce RTX 3080 Ti"
            && major == 8 && minor == 6
            && version == 120901)
    })().unwrap_or(false);
    DEVICE_SUPPORT.set(Some((device.id(), is_supported)));
    is_supported
}

fn can_apply_dual_geglu(input: &Tensor, weight: &Tensor) -> bool {
    input.device().is_cuda()
        && input.device().same_device(weight.device())
        && input.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
        && input.rank() == 2
        && (2048..=16384).contains(&input.dims()[0])
        && input.dims()[1] == 768
        && weight.dims() == [2304, 768]
        && input.is_contiguous()
        && weight.is_contiguous()
        && input.layout().start_offset().is_multiple_of(8)
        && weight.layout().start_offset().is_multiple_of(8)
        && !candle_core::cuda_backend::gemm_reduced_precision_bf16()
        && is_supported_device(input)
}

unsafe extern "C" {
    fn apply_docbert_dual_geglu(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        inter: i32,
        width: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(test)]
thread_local! {
    static DISPATCH_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SHOULD_COMPARE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SHOULD_USE_CANDLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static COMPARISONS: std::cell::RefCell<Vec<(usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn project_geglu(
    input: &Tensor,
    projection: &candle_nn::Linear,
) -> Result<Tensor> {
    #[cfg(test)]
    if SHOULD_COMPARE.get() || SHOULD_USE_CANDLE.get() {
        let expected =
            crate::geglu_cuda::apply_geglu(&input.apply(projection)?)?;
        compare_real_activations(input, projection.weight(), &expected)?;
        return Ok(expected);
    }
    if projection.bias().is_none()
        && can_apply_dual_geglu(input, projection.weight())
    {
        apply_dual_geglu(input, projection.weight())
    } else {
        crate::geglu_cuda::apply_geglu(&input.apply(projection)?)
    }
}

fn apply_dual_geglu(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    input.apply_op2_no_bwd(weight, &DualGeGlu)
}

struct DualGeGlu;

impl CustomOp2 for DualGeGlu {
    fn name(&self) -> &'static str {
        "docbert_dual_geglu_bf16"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("dual GeGLU CUDA operation received CPU storage")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (rows, width) = input_layout.shape().dims2()?;
        if width != 768
            || weight_layout.dims() != [2304, 768]
            || rows == 0
            || rows > 16384
            || input.device().id() != weight.device().id()
            || !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !input_layout.start_offset().is_multiple_of(8)
            || !weight_layout.start_offset().is_multiple_of(8)
        {
            candle_core::bail!("unsupported dual GeGLU layout")
        }
        let (CudaStorageSlice::BF16(inputs), CudaStorageSlice::BF16(weights)) =
            (&input.slice, &weight.slice)
        else {
            candle_core::bail!("dual GeGLU requires BF16 input and weight")
        };
        let inter = 1152;
        let device = input.device();
        let stream = device.cuda_stream();
        let inputs = inputs.slice(
            input_layout.start_offset()
                ..input_layout.start_offset() + rows * width,
        );
        let weights = weights.slice(
            weight_layout.start_offset()
                ..weight_layout.start_offset() + 2 * inter * width,
        );
        // The kernel writes every output element before the tensor is read.
        let mut output = unsafe { device.alloc(rows * inter)? };
        let (input_pointer, input_guard) = inputs.device_ptr(&stream);
        let (weight_pointer, weight_guard) = weights.device_ptr(&stream);
        let (output_pointer, output_guard) = output.device_ptr_mut(&stream);
        stream.context().bind_to_thread().w()?;
        // Aligned contiguous BF16 matrices match the checked dimensions.
        // Pointer guards order reads and writes on the owning CUDA stream.
        let status = unsafe {
            apply_docbert_dual_geglu(
                input_pointer as *const _,
                weight_pointer as *const _,
                output_pointer as *mut _,
                rows as i32,
                inter as i32,
                width as i32,
                stream.cu_stream().cast(),
            )
        };
        if status != 0 {
            candle_core::bail!("CUTLASS dual GeGLU returned status {status}")
        }
        drop((input_guard, weight_guard, output_guard));
        #[cfg(test)]
        DISPATCH_COUNT.set(DISPATCH_COUNT.get() + 1);
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: device.clone(),
            },
            (rows, inter).into(),
        ))
    }
}

#[cfg(test)]
fn compare_real_activations(
    input: &Tensor,
    weight: &Tensor,
    expected: &Tensor,
) -> Result<()> {
    if !SHOULD_COMPARE.get() {
        return Ok(());
    }
    let before = DISPATCH_COUNT.get();
    let actual = apply_dual_geglu(input, weight)?;
    assert_eq!(DISPATCH_COUNT.get(), before + 1);
    let differences = tests::count_differences(&actual, expected)?;
    COMPARISONS.with_borrow_mut(|comparisons| {
        comparisons.push((input.dims()[0], differences))
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    pub(super) fn count_differences(
        actual: &Tensor,
        expected: &Tensor,
    ) -> Result<usize> {
        assert_eq!(actual.dims(), expected.dims());
        let bits = |tensor: &Tensor| -> Result<Vec<u32>> {
            Ok(tensor
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        Ok(bits(actual)?
            .iter()
            .zip(bits(expected)?)
            .filter(|(a, b)| **a != *b)
            .count())
    }

    fn print_device(device: &Device) -> Result<()> {
        let context = device.as_cuda_device()?.cuda_stream().context().clone();
        // This query takes no pointers and does not mutate cuBLAS state.
        let version = unsafe {
            candle_core::cuda_backend::cudarc::cublaslt::sys::cublasLtGetVersion(
            )
        };
        eprintln!(
            "DUAL_DEVICE name={} cublas_lt={version} reduced_bf16={}",
            context.name().w()?,
            candle_core::cuda_backend::gemm_reduced_precision_bf16()
        );
        Ok(())
    }

    fn assert_projection_matches(
        input: &Tensor,
        weight: &Tensor,
        dispatches: usize,
    ) -> Result<()> {
        let projection = candle_nn::Linear::new(weight.clone(), None);
        let expected =
            crate::geglu_cuda::apply_geglu(&input.apply(&projection)?)?;
        let before = DISPATCH_COUNT.get();
        let actual = project_geglu(input, &projection)?;
        assert_eq!(DISPATCH_COUNT.get() - before, dispatches);
        assert_eq!(count_differences(&actual, &expected)?, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires RTX 3080 Ti CUDA; run alone"]
    fn dual_dispatch_preserves_offsets_and_fallbacks() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let build_tensor = |rows, columns, offset| -> Result<Tensor> {
            let values: Vec<_> = (0..rows * columns + offset)
                .map(|index| (index % 253) as f32 / 128.0 - 1.0)
                .collect();
            Tensor::from_vec(values, rows * columns + offset, &device)?
                .to_dtype(DType::BF16)?
                .narrow(0, offset, rows * columns)?
                .reshape((rows, columns))
        };
        let input = build_tensor(2048, 768, 0)?;
        let weight = build_tensor(2304, 768, 0)?;
        assert!(can_apply_dual_geglu(&input, &weight));
        assert_projection_matches(&input, &weight, 1)?;
        for offset in [1, 7, 8, 16] {
            let dispatches = usize::from(offset % 8 == 0);
            assert_projection_matches(
                &build_tensor(2048, 768, offset)?,
                &weight,
                dispatches,
            )?;
            assert_projection_matches(
                &input,
                &build_tensor(2304, 768, offset)?,
                dispatches,
            )?;
        }
        assert_projection_matches(&input.t()?.contiguous()?.t()?, &weight, 0)?;
        assert_projection_matches(&input, &weight.t()?.contiguous()?.t()?, 0)?;
        for dtype in [DType::F16, DType::F32] {
            assert_projection_matches(
                &input.to_dtype(dtype)?,
                &weight.to_dtype(dtype)?,
                0,
            )?;
        }
        let small = input.narrow(0, 0, 3)?;
        assert_projection_matches(
            &small.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
            &weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
            0,
        )?;
        let cpu_input = small.to_device(&Device::Cpu)?;
        let cpu_projection =
            candle_nn::Linear::new(weight.to_device(&Device::Cpu)?, None);
        let before = DISPATCH_COUNT.get();
        assert_eq!(
            project_geglu(&cpu_input, &cpu_projection)
                .unwrap_err()
                .to_string(),
            cpu_input.apply(&cpu_projection).unwrap_err().to_string()
        );
        assert_eq!(DISPATCH_COUNT.get(), before);
        assert_projection_matches(&input.narrow(0, 0, 2047)?, &weight, 0)?;
        assert_projection_matches(&build_tensor(16385, 768, 0)?, &weight, 0)?;
        assert_projection_matches(&input.reshape((2, 1024, 768))?, &weight, 0)?;
        assert_projection_matches(
            &build_tensor(2048, 256, 0)?,
            &build_tensor(512, 256, 0)?,
            0,
        )?;
        let reduced_precision =
            candle_core::cuda_backend::gemm_reduced_precision_bf16();
        candle_core::cuda_backend::set_gemm_reduced_precision_bf16(true);
        let can_dispatch = can_apply_dual_geglu(&input, &weight);
        let comparison = assert_projection_matches(&input, &weight, 0);
        candle_core::cuda_backend::set_gemm_reduced_precision_bf16(
            reduced_precision,
        );
        assert!(!can_dispatch);
        comparison?;
        eprintln!(
            "DUAL_GUARDS offsets=5 noncontiguous=2 non_bf16=2 cpu=1 row_boundaries=2 rank3=1 unsupported_width=1 reduced_precision=1 differences=0"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and cached GTE-ModernColBERT-v1; run alone"]
    fn dual_dispatch_preserves_document_and_query_bits() -> anyhow::Result<()> {
        let mut model: crate::ColBERT =
            crate::ColBERT::from("lightonai/GTE-ModernColBERT-v1")
                .with_device(Device::new_cuda(0)?)
                .with_dtype(DType::BF16)
                .try_into()?;
        let documents: Vec<_> = (0..64).map(|index| format!("Document {index}. {}", "The retrieval pipeline stores embeddings and compares document tokens with query vectors. ".repeat(3 + index % 15))).collect();
        let queries: Vec<_> = (0..32)
            .map(|index| {
                format!("Which retrieval pipeline indexed document {index}?")
            })
            .collect();
        for batch_size in [16, 64] {
            model.batch_size = batch_size;
            for (is_query, texts) in [(false, &documents), (true, &queries)] {
                SHOULD_USE_CANDLE.set(true);
                let reference = model.encode(texts, is_query);
                SHOULD_USE_CANDLE.set(false);
                let reference = reference?;
                let before = DISPATCH_COUNT.get();
                let actual = model.encode(texts, is_query)?;
                let dispatches = DISPATCH_COUNT.get() - before;
                if is_query {
                    assert_eq!(dispatches, 0);
                } else {
                    assert!(dispatches > 0);
                }
                let differences = count_differences(&actual, &reference)?;
                assert_eq!(differences, 0);
                eprintln!(
                    "DUAL_ENCODER batch={batch_size} queries={is_query} outputs={} dispatches={dispatches} differences={differences}",
                    actual.elem_count()
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and cached GTE-ModernColBERT-v1; run alone"]
    fn measure_dual_geglu_document_batches() -> anyhow::Result<()> {
        let mut model: crate::ColBERT =
            crate::ColBERT::from("lightonai/GTE-ModernColBERT-v1")
                .with_device(Device::new_cuda(0)?)
                .with_dtype(DType::BF16)
                .try_into()?;
        model.document_length = 180;
        let documents: Vec<_> = (0..64).map(|index| format!("Document {index}. {}", "The retrieval pipeline stores embeddings and compares document tokens with query vectors. ".repeat(3 + index % 15))).collect();
        let mut encode =
            |batch_size, should_use_candle| -> anyhow::Result<(Tensor, f64)> {
                model.batch_size = batch_size;
                SHOULD_USE_CANDLE.set(should_use_candle);
                let before = DISPATCH_COUNT.get();
                let measured = (|| -> anyhow::Result<_> {
                    model.device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = model.encode(&documents, false)?;
                    model.device.synchronize()?;
                    Ok((output, start.elapsed().as_secs_f64() * 1000.0))
                })();
                SHOULD_USE_CANDLE.set(false);
                if should_use_candle {
                    assert_eq!(DISPATCH_COUNT.get(), before);
                } else {
                    assert!(DISPATCH_COUNT.get() > before);
                }
                measured
            };
        for repeat in 1..=2 {
            for batch_size in [16, 64] {
                for _ in 0..3 {
                    encode(batch_size, true)?;
                    encode(batch_size, false)?;
                }
                let mut timings = Vec::new();
                for pair in 0..40 {
                    let (reference, fused) = if pair % 2 == 0 {
                        (encode(batch_size, true)?, encode(batch_size, false)?)
                    } else {
                        let fused = encode(batch_size, false)?;
                        (encode(batch_size, true)?, fused)
                    };
                    assert_eq!(count_differences(&reference.0, &fused.0)?, 0);
                    timings.push((reference.1, fused.1));
                }
                let mut changes: Vec<_> = timings
                    .iter()
                    .map(|&(reference, fused)| {
                        100.0 * (fused / reference - 1.0)
                    })
                    .collect();
                changes.sort_by(f64::total_cmp);
                let median = (changes[19] + changes[20]) / 2.0;
                let faster = timings
                    .iter()
                    .filter(|(reference, fused)| fused < reference)
                    .count();
                eprintln!(
                    "DUAL_TIMING repeat={repeat} batch={batch_size} pairs=40 median_change_percent={median:.3} faster={faster} differences=0"
                );
                eprintln!(
                    "DUAL_TIMING_SAMPLES repeat={repeat} batch={batch_size} reference_fused_ms={timings:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires RTX 3080 Ti CUDA; run alone"]
    fn dual_products_match_candle() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        print_device(&device)?;
        let mut seed = 917u32;
        let mut build_tensor = |rows, columns| -> Result<Tensor> {
            let values: Vec<f32> = (0..rows * columns)
                .map(|index| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / 16777216.0 - 0.5)
                        * [0.001, 0.1, 1.0, 8.0, 60.0][index % 5]
                })
                .collect();
            Tensor::from_vec(values, (rows, columns), &device)?
                .to_dtype(DType::BF16)
        };
        let row_counts = [
            1, 31, 57, 129, 256, 516, 768, 1032, 1200, 1536, 2047, 2048, 2049,
            2304, 3072, 4095, 4096, 4097, 5400, 8191, 8192, 8193, 12288, 16383,
            16384,
        ];
        let mut outputs = 0usize;
        let mut failures = Vec::new();
        let before = DISPATCH_COUNT.get();
        for weights in 0..3 {
            let weight = build_tensor(2304, 768)?;
            let projection = candle_nn::Linear::new(weight.clone(), None);
            for rows in row_counts {
                let input = build_tensor(rows, 768)?;
                assert_eq!(can_apply_dual_geglu(&input, &weight), rows >= 2048);
                for (sign, input) in [(1, input.clone()), (-1, input.neg()?)] {
                    let expected = crate::geglu_cuda::apply_geglu(
                        &input.apply(&projection)?,
                    )?;
                    let actual = apply_dual_geglu(&input, &weight)?;
                    let differences = count_differences(&actual, &expected)?;
                    outputs += rows * 1152;
                    eprintln!(
                        "DUAL_PRODUCT weights={weights} rows={rows} sign={sign} outputs={} differences={differences}",
                        rows * 1152
                    );
                    if differences > 0 {
                        failures.push((weights, rows, sign, differences));
                    }
                }
            }
        }
        let products = DISPATCH_COUNT.get() - before;
        assert_eq!(products, row_counts.len() * 6);
        eprintln!(
            "DUAL_SYNTHETIC row_counts={} products={products} outputs={outputs} failures={failures:?}",
            row_counts.len()
        );
        assert!(failures.iter().all(|&(_, rows, _, _)| rows < 2048));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and cached GTE-ModernColBERT-v1; run alone"]
    fn dual_matches_candle_on_real_activations() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        print_device(&device)?;
        let mut model: crate::ColBERT =
            crate::ColBERT::from("lightonai/GTE-ModernColBERT-v1")
                .with_device(device)
                .with_dtype(DType::BF16)
                .with_batch_size(128)
                .try_into()?;
        model.document_length = 180;
        let paragraphs = [
            "The indexing pipeline compares content hashes and updates changed documents. Unchanged files are skipped. ",
            "A clinical trial compares treatment outcomes with the baseline, including glucose, inflammation, and cardiovascular markers. ",
            "Invoice 2026-09: customer paid 137.42 USD. Record the exchange rate and reconcile the account balance. ",
        ];
        COMPARISONS.with_borrow_mut(Vec::clear);
        SHOULD_COMPARE.set(true);
        for repeats in [1, 2, 3, 4, 6, 9] {
            for batch in [1, 3, 5, 8, 11, 12, 13, 16, 17, 24, 32, 48, 64, 90] {
                let documents: Vec<_> = (0..batch)
                    .map(|index| {
                        format!(
                            "Document {index}. {}",
                            paragraphs[index % paragraphs.len()]
                                .repeat(repeats + index % 2)
                        )
                    })
                    .collect();
                let tokens = model.tokenize(&documents, false)?;
                let crate::BaseModel::ModernBert(encoder) = &model.model else {
                    panic!("expected ModernBERT");
                };
                let before = DISPATCH_COUNT.get();
                encoder.forward_varlen_padded(
                    &tokens.token_ids,
                    &tokens.valid_lens,
                )?;
                assert_eq!(DISPATCH_COUNT.get() - before, 22);
                let rows: usize = tokens.valid_lens.iter().sum();
                let differences: usize =
                    COMPARISONS.with_borrow(|comparisons| {
                        comparisons
                            .iter()
                            .rev()
                            .take(22)
                            .map(|&(_, differences)| differences)
                            .sum()
                    });
                eprintln!(
                    "DUAL_WORKLOAD repeats={repeats} batch={batch} rows={rows} layers=22 differences={differences}"
                );
            }
        }
        // Truncate real tokenized documents to exercise exact packed-row boundaries.
        for rows in [
            1535usize, 1536, 1537, 2046, 2047, 2048, 2049, 2050, 2303, 2304,
            2305, 4095, 4096, 4097, 8191, 8192, 8193, 12287, 12288, 12289,
            16383, 16384,
        ] {
            let batch = rows.div_ceil(180);
            let documents: Vec<_> = (0..batch)
                .map(|index| {
                    format!(
                        "Document {index}. {}",
                        paragraphs[index % paragraphs.len()].repeat(16)
                    )
                })
                .collect();
            let tokens = model.tokenize(&documents, false)?;
            assert!(tokens.valid_lens.iter().all(|&length| length == 180));
            let lengths: Vec<_> = (0..batch)
                .map(|index| rows / batch + usize::from(index < rows % batch))
                .collect();
            assert_eq!(lengths.iter().sum::<usize>(), rows);
            let crate::BaseModel::ModernBert(encoder) = &model.model else {
                panic!("expected ModernBERT");
            };
            let before = DISPATCH_COUNT.get();
            encoder.forward_varlen_padded(&tokens.token_ids, &lengths)?;
            assert_eq!(DISPATCH_COUNT.get() - before, 22);
            let differences: usize = COMPARISONS.with_borrow(|comparisons| {
                comparisons
                    .iter()
                    .rev()
                    .take(22)
                    .map(|&(_, differences)| differences)
                    .sum()
            });
            eprintln!(
                "DUAL_BOUNDARY rows={rows} layers=22 differences={differences}"
            );
        }
        SHOULD_COMPARE.set(false);
        let comparisons = COMPARISONS.take();
        let mut row_counts: Vec<_> =
            comparisons.iter().map(|&(rows, _)| rows).collect();
        row_counts.sort_unstable();
        row_counts.dedup();
        let failures: Vec<_> = comparisons
            .iter()
            .enumerate()
            .filter(|(_, (_, differences))| *differences > 0)
            .collect();
        let outputs: usize =
            comparisons.iter().map(|&(rows, _)| rows * 1152).sum();
        let eligible_layers = comparisons
            .iter()
            .filter(|&&(rows, _)| rows >= 2048)
            .count();
        let eligible_outputs: usize = comparisons
            .iter()
            .filter(|&&(rows, _)| rows >= 2048)
            .map(|&(rows, _)| rows * 1152)
            .sum();
        let mut failed_rows: Vec<_> =
            failures.iter().map(|(_, (rows, _))| *rows).collect();
        failed_rows.sort_unstable();
        failed_rows.dedup();
        eprintln!(
            "DUAL_REAL row_counts={} layers={} outputs={outputs} eligible_layers={eligible_layers} eligible_outputs={eligible_outputs} mismatched_layers={} failed_rows={failed_rows:?}",
            row_counts.len(),
            comparisons.len(),
            failures.len()
        );
        eprintln!("DUAL_REAL_ROWS {row_counts:?}");
        assert!(failures.iter().all(|(_, (rows, _))| *rows < 2048));
        Ok(())
    }
}

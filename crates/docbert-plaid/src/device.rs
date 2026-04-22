//! Device selection for the candle-backed inner loops.
//!
//! Picks CUDA when the crate was built with the `cuda` feature *and* a
//! GPU is actually present at runtime; falls back to CPU otherwise. The
//! choice is made once on first call and cached for the rest of the
//! process — every subsequent tensor allocation lands on the same
//! device, so there are no surprise host↔device transfers in the
//! middle of a tight loop.
//!
//! Callers shouldn't need to think about devices in normal use:
//! `default_device()` returns whatever's right for this build.

use std::sync::OnceLock;

use candle_core::Device;

static DEVICE: OnceLock<Device> = OnceLock::new();

/// Return the device docbert-plaid uses for all tensor compute.
///
/// Cached on first call. The choice is:
///
/// - With the `cuda` feature **and** a usable CUDA device → `Device::Cuda(0)`.
/// - Otherwise → `Device::Cpu`.
pub fn default_device() -> &'static Device {
    DEVICE.get_or_init(select_device)
}

fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            return dev;
        }
    }
    Device::Cpu
}

/// Return `(free, total)` device-memory bytes for the selected
/// device, or `None` when we're not on CUDA.
///
/// Thin wrapper around `cuMemGetInfo` via cudarc; exposed so the
/// PLAID builder can annotate its progress output with the headroom
/// the CUDA mempool has left before a big allocation.
pub fn device_memory_info() -> Option<(usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = default_device() {
            use candle_core::cuda_backend::cudarc::driver::result;
            if let Ok((free, total)) = result::mem_get_info() {
                return Some((free, total));
            }
        }
    }
    None
}

/// Release cached but currently-unused device memory back to the
/// driver.
///
/// Candle hands every dropped `Tensor` back to CUDA's async memory
/// pool (via `cuMemFreeAsync`). The pool keeps the bytes around for
/// fast reuse, which is normally what you want — but when the encoder
/// model finishes embedding, its ~2 GB of ModernBert per-batch
/// caches stay committed to the pool even after the callers drop the
/// model. That's enough to block a subsequent 3.47 GB `Tensor`
/// allocation on a 12 GB card and surface as `CUDA_ERROR_OUT_OF_MEMORY`.
///
/// This function asks CUDA to trim the default mempool to zero
/// retained bytes, handing the freed blocks back to the driver so
/// the next large allocation can grow into them. Without the `cuda`
/// feature (or when running on CPU) it's a no-op.
///
/// Safe wrapper around `cuDeviceGetDefaultMemPool` +
/// `cuMemPoolTrimTo`. Only trims device 0 — docbert currently only
/// ever uses the default CUDA device.
pub fn release_cached_device_memory() -> Result<(), candle_core::Error> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = default_device() {
            use candle_core::cuda_backend::cudarc::driver::result;
            // Safety: `result::device::get` is a safe wrapper that
            // returns a valid device handle; `get_default_mem_pool`
            // and `trim_to` are unsafe because they take raw CUDA
            // handles, but we only ever pass handles produced by
            // cudarc in the same call — the documented preconditions
            // ("valid device", "valid pool") hold. Trimming the pool
            // while no outstanding allocations reference it is
            // always-safe per the CUDA docs.
            unsafe {
                let dev = result::device::get(0).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "release_cached_device_memory: cuDeviceGet(0) failed: {e}"
                    ))
                })?;
                let pool =
                    result::device::get_default_mem_pool(dev).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "release_cached_device_memory: cuDeviceGetDefaultMemPool failed: {e}"
                        ))
                    })?;
                result::mem_pool::trim_to(pool, 0).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "release_cached_device_memory: cuMemPoolTrimTo failed: {e}"
                    ))
                })?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_is_stable_across_calls() {
        let a = default_device();
        let b = default_device();
        // Same backing static, so the references should compare as
        // pointing at the same value.
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn default_device_is_cpu_without_cuda_feature() {
        // When the cuda feature is off, the choice is unconditional.
        // When the feature is on but no GPU is present (CI without
        // CUDA), select_device() also falls back to CPU. So this test
        // only asserts the no-cuda behaviour to stay portable.
        #[cfg(not(feature = "cuda"))]
        {
            assert!(matches!(default_device(), Device::Cpu));
        }
    }
}

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

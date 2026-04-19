//! Test-only helpers shared across the crate's `#[cfg(test)] mod tests`
//! blocks. Only compiled under `cfg(test)`.

use candle_core::Device;

/// Returns the device unit tests should build tensors on.
///
/// Mirrors how `ModelManager::select_device` picks an accelerator in
/// production: CUDA if the `cuda` feature is enabled **and** a device is
/// actually present, then Metal under the same guard, otherwise CPU. Runtime
/// fall-back to CPU means a `cuda`-enabled build on a machine without a GPU
/// (typical CI) still runs the tests.
pub(crate) fn test_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

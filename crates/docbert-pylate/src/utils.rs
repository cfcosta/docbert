use candle_core::Tensor;

use crate::error::ColbertError;

/// Normalizes a tensor using L2 normalization along the last dimension.
///
/// A small epsilon is applied to avoid NaNs when a row is entirely zero.
pub fn normalize_l2(v: &Tensor) -> Result<Tensor, ColbertError> {
    let inv_norm_l2 =
        ((v.sqr()?.sum_keepdim(v.rank() - 1)?.sqrt()? + 1e-12f64)?).recip()?;
    v.broadcast_mul(&inv_norm_l2).map_err(ColbertError::from)
}

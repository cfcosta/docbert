pub mod builder;
#[cfg(feature = "cuda")]
mod cutlass_geglu;
pub mod error;
#[cfg(feature = "cuda")]
mod geglu_cuda;
pub mod model;
pub mod modernbert;
pub mod pooling;
#[cfg(feature = "cuda")]
mod residual_norm_cuda;
#[cfg(feature = "cuda")]
mod rope_cuda;
pub mod types;
pub mod utils;

pub use builder::ColbertBuilder;
pub use error::ColbertError;
pub use model::{BaseModel, ColBERT};
pub use pooling::hierarchical_pooling;
pub use types::{
    EncodeInput,
    EncodeOutput,
    RawSimilarityOutput,
    Similarities,
    SimilarityInput,
};
pub use utils::normalize_l2;

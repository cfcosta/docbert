pub mod builder;
pub mod error;
pub mod model;
pub mod modernbert;
pub mod pooling;
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

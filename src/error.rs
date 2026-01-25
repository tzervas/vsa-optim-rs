//! Error types for VSA training optimization.

use thiserror::Error;

/// Result type alias for VSA optimization operations.
pub type Result<T> = std::result::Result<T, OptimError>;

/// Errors that can occur during VSA optimization operations.
#[derive(Debug, Error)]
pub enum OptimError {
    /// Invalid configuration parameter.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Shape mismatch in tensor operations.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Empty input where non-empty was required.
    #[error("empty input: {0}")]
    EmptyInput(String),

    /// Compression error.
    #[error("compression error: {0}")]
    Compression(String),

    /// Prediction error.
    #[error("prediction error: {0}")]
    Prediction(String),

    /// Candle tensor operation error.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Ternary operation error.
    #[error("ternary error: {0}")]
    Ternary(#[from] trit_vsa::TernaryError),
}

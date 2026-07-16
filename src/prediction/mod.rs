//! Gradient prediction for training acceleration.
//!
//! Instead of computing full gradients for every step, we can predict
//! gradients based on history and apply corrections periodically. This enables:
//! 1. Faster training by reducing compute per step
//! 2. Similar convergence via correction cycles
//! 3. Memory efficiency through compressed gradient history
//!
//! The key insight is that gradients in consecutive steps are highly correlated.
//! We can exploit this temporal redundancy by predicting future gradients from
//! past gradients and only computing full gradients periodically for correction.
//!
//! # Prediction Modes
//!
//! - **Deterministic** (recommended): Uses weighted least squares to fit a linear
//!   model to gradient evolution. Predictions are fully reproducible given the
//!   same history. Includes residual tracking for drift correction.
//!
//! - **Momentum**: Simple extrapolation using momentum-based formula. Faster
//!   but less accurate and not fully deterministic under certain conditions.
//!
//! # References
//!
//! - Gradient Prediction (ICLR 2019): Predicting gradients for faster training
//! - Lookahead Optimizer: Using slow/fast weight updates

mod deterministic;
mod predictor;

pub use deterministic::{
    DeterministicPredictionConfig, DeterministicPredictor, PredictorStatistics,
};
pub use predictor::{GradientPredictor, PredictorStats};

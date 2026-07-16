//! # vsa-optim-rs
//!
//! Deterministic training optimization using Vector Symbolic Architecture (VSA),
//! ternary quantization, and closed-form gradient prediction.
//!
//! This crate enables efficient large model fine-tuning on consumer hardware through
//! mathematically principled gradient compression and prediction with guaranteed
//! reproducibility.
//!
//! ## Key Properties
//!
//! - **Deterministic**: Identical inputs produce identical outputs
//! - **Closed-form**: Weighted least squares with Cramer's rule—no iterative optimization
//! - **Memory-efficient**: ~90% gradient storage reduction via VSA compression
//! - **Compute-efficient**: ~80% backward pass reduction via gradient prediction
//!
//! ## Quick Start
//!
//! The recommended entry point is [`DeterministicPhaseTrainer`], which orchestrates
//! training through four phases: WARMUP → FULL → PREDICT → CORRECT.
//!
//! ```ignore
//! use vsa_optim_rs::{DeterministicPhaseTrainer, DeterministicPhaseConfig, DeterministicPhase};
//! use candle_core::Device;
//!
//! let shapes = vec![
//!     ("layer1.weight".into(), vec![768, 768]),
//!     ("layer2.weight".into(), vec![768, 3072]),
//! ];
//!
//! let config = DeterministicPhaseConfig::default();
//! let mut trainer = DeterministicPhaseTrainer::new(&shapes, config, &Device::Cpu)?;
//!
//! for step in 0..100 {
//!     let info = trainer.begin_step()?;
//!     
//!     if trainer.should_compute_full() {
//!         // Compute gradients via backpropagation
//!         trainer.record_full_gradients(&gradients)?;
//!     } else {
//!         // Use deterministically predicted gradients
//!         let predicted = trainer.get_predicted_gradients()?;
//!     }
//!     
//!     trainer.end_step(loss)?;
//! }
//! # Ok::<(), vsa_optim_rs::error::OptimError>(())
//! ```
//!
//! ## Modules
//!
//! - [`config`]: Configuration types for all components
//! - [`error`]: Error types and result aliases  
//! - [`phase`]: Phase-based training orchestration (deterministic and legacy)
//! - [`prediction`]: Gradient prediction (deterministic least squares and momentum)
//! - [`ternary`]: Ternary `{-1, 0, +1}` gradient accumulation
//! - [`vsa`]: VSA gradient compression with bind/bundle/unbind operations
//!
//! ## Deterministic Gradient Prediction
//!
//! The core algorithm fits a linear gradient model using weighted least squares:
//!
//! ```text
//! g(t) = baseline + velocity × t + residual
//! ```
//!
//! - **baseline**: Weighted mean of historical gradients
//! - **velocity**: Gradient change rate (fitted via Cramer's rule)
//! - **residual**: Exponentially-averaged prediction error for drift correction
//!
//! ## References
//!
//! - Kanerva, P. (2009). Hyperdimensional Computing
//! - Johnson, W. & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings
//! - Ma, S. et al. (2024). The Era of 1-bit LLMs

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod error;
pub mod phase;
pub mod prediction;
pub mod ternary;
pub mod vsa;

// Re-export main types at crate root for convenience
pub use config::{PhaseConfig, PredictionConfig, TernaryConfig, VSAConfig};
pub use error::{OptimError, Result};
pub use phase::{PhaseTrainer, TrainingPhase};
pub use prediction::GradientPredictor;
pub use ternary::{TernaryGradientAccumulator, TernaryOptimizerWrapper};
pub use vsa::VSAGradientCompressor;

// Re-export deterministic training types (recommended for production)
pub use phase::{
    DeterministicPhase, DeterministicPhaseConfig, DeterministicPhaseTrainer, DeterministicStepInfo,
    DeterministicTrainerStats, LossAnomaly, LossHistory, LossHistoryConfig, LossMeasurement,
    LossStatistics,
};
pub use prediction::{DeterministicPredictionConfig, DeterministicPredictor, PredictorStatistics};

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::*;

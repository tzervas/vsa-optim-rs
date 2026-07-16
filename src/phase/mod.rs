//! Phase-based training with prediction and correction cycles.
//!
//! Full gradient computation is expensive. By alternating between:
//! 1. Full training phases (accurate gradients)
//! 2. Predicted phases (fast, approximate gradients)
//! 3. Correction phases (fix accumulated errors)
//!
//! We can achieve similar convergence with significantly reduced compute.
//!
//! # Training Modes
//!
//! ## Deterministic Mode (Recommended)
//!
//! Uses [`DeterministicPhaseTrainer`] with weighted least-squares gradient model.
//! Guarantees reproducible predictions and includes residual tracking.
//!
//! ```text
//! WARMUP (W steps) → FULL (N steps) → PREDICT (M steps) → CORRECT → repeat
//! ```
//!
//! ## Legacy Mode
//!
//! Uses [`PhaseTrainer`] with momentum-based extrapolation. Faster but less
//! accurate and not fully deterministic.
//!
//! # Determinism Guarantees
//!
//! With `DeterministicPhaseTrainer`:
//! - Same seed + same data = identical training trajectory
//! - No stochastic operations in prediction
//! - Residuals ensure convergence to actual gradients

mod deterministic_trainer;
mod loss_history;
mod trainer;

pub use deterministic_trainer::{
    DeterministicPhase, DeterministicPhaseConfig, DeterministicPhaseTrainer, DeterministicStepInfo,
    DeterministicTrainerStats,
};
pub use loss_history::{
    LossAnomaly, LossHistory, LossHistoryConfig, LossMeasurement, LossStatistics,
};
pub use trainer::{PhaseTrainer, StepInfo, TrainerStats, TrainingPhase};

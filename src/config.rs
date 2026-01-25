//! Configuration types for VSA training optimization.
//!
//! This module provides configuration structs for all optimization components:
//! - [`VSAConfig`]: VSA gradient compression settings
//! - [`TernaryConfig`]: Ternary gradient accumulation settings
//! - [`PredictionConfig`]: Gradient prediction settings
//! - [`PhaseConfig`]: Phase-based training orchestration settings

use serde::{Deserialize, Serialize};

/// Configuration for VSA gradient compression.
///
/// # Example
///
/// ```
/// use vsa_optim_rs::VSAConfig;
///
/// let config = VSAConfig::default()
///     .with_compression_ratio(0.1)
///     .with_ternary(true);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VSAConfig {
    /// Hypervector dimension for compression.
    pub dimension: usize,

    /// Target compression ratio (0.0 to 1.0).
    /// A ratio of 0.1 means 90% compression.
    pub compression_ratio: f32,

    /// Whether to use ternary quantization on compressed gradients.
    pub use_ternary: bool,

    /// Random seed for reproducible projections.
    pub seed: u64,
}

impl Default for VSAConfig {
    fn default() -> Self {
        Self {
            dimension: 8192,
            compression_ratio: 0.1,
            use_ternary: true,
            seed: 42,
        }
    }
}

impl VSAConfig {
    /// Set the compression ratio.
    #[must_use]
    pub const fn with_compression_ratio(mut self, ratio: f32) -> Self {
        self.compression_ratio = ratio;
        self
    }

    /// Set whether to use ternary quantization.
    #[must_use]
    pub const fn with_ternary(mut self, use_ternary: bool) -> Self {
        self.use_ternary = use_ternary;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the hypervector dimension.
    #[must_use]
    pub const fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }
}

/// Configuration for ternary gradient accumulation.
///
/// # Example
///
/// ```
/// use vsa_optim_rs::TernaryConfig;
///
/// let config = TernaryConfig::default()
///     .with_accumulation_steps(8)
///     .with_stochastic_rounding(true);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryConfig {
    /// Number of gradient accumulation steps before optimizer update.
    pub accumulation_steps: usize,

    /// Threshold for ternary quantization (relative to mean abs).
    pub ternary_threshold: f32,

    /// Learning rate for scale parameters.
    pub scale_learning_rate: f32,

    /// Whether to use stochastic rounding (unbiased) or deterministic.
    pub use_stochastic_rounding: bool,
}

impl Default for TernaryConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 8,
            ternary_threshold: 0.5,
            scale_learning_rate: 0.01,
            use_stochastic_rounding: true,
        }
    }
}

impl TernaryConfig {
    /// Set the number of accumulation steps.
    #[must_use]
    pub const fn with_accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps;
        self
    }

    /// Set whether to use stochastic rounding.
    #[must_use]
    pub const fn with_stochastic_rounding(mut self, stochastic: bool) -> Self {
        self.use_stochastic_rounding = stochastic;
        self
    }

    /// Set the ternary threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.ternary_threshold = threshold;
        self
    }
}

/// Configuration for gradient prediction.
///
/// # Example
///
/// ```
/// use vsa_optim_rs::PredictionConfig;
///
/// let config = PredictionConfig::default()
///     .with_history_size(5)
///     .with_prediction_steps(4);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Number of past gradients to keep in history.
    pub history_size: usize,

    /// Number of steps to predict before computing full gradients.
    pub prediction_steps: usize,

    /// Momentum factor for gradient extrapolation.
    pub momentum: f32,

    /// Weight applied to correction terms.
    pub correction_weight: f32,

    /// Minimum correlation threshold for using prediction.
    pub min_correlation: f32,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            history_size: 5,
            prediction_steps: 4,
            momentum: 0.9,
            correction_weight: 0.5,
            min_correlation: 0.8,
        }
    }
}

impl PredictionConfig {
    /// Set the history size.
    #[must_use]
    pub const fn with_history_size(mut self, size: usize) -> Self {
        self.history_size = size;
        self
    }

    /// Set the number of prediction steps.
    #[must_use]
    pub const fn with_prediction_steps(mut self, steps: usize) -> Self {
        self.prediction_steps = steps;
        self
    }

    /// Set the momentum factor.
    #[must_use]
    pub const fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the correction weight.
    #[must_use]
    pub const fn with_correction_weight(mut self, weight: f32) -> Self {
        self.correction_weight = weight;
        self
    }
}

/// Configuration for phase-based training.
///
/// The training cycle is: FULL → PREDICT → CORRECT → repeat
///
/// # Example
///
/// ```
/// use vsa_optim_rs::PhaseConfig;
///
/// let config = PhaseConfig::default()
///     .with_full_steps(10)
///     .with_predict_steps(40);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Number of full gradient computation steps per cycle.
    pub full_steps: usize,

    /// Number of predicted gradient steps per cycle.
    pub predict_steps: usize,

    /// Frequency of correction steps during predict phase.
    pub correct_every: usize,

    /// Sub-configuration for gradient prediction.
    pub prediction_config: PredictionConfig,

    /// Sub-configuration for ternary optimization.
    pub ternary_config: TernaryConfig,

    /// Sub-configuration for VSA compression.
    pub vsa_config: VSAConfig,

    /// Gradient accumulation steps.
    pub gradient_accumulation: usize,

    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f32,

    /// Whether to adaptively adjust phase lengths based on loss.
    pub adaptive_phases: bool,

    /// Loss increase threshold for triggering more full steps.
    pub loss_threshold: f32,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            full_steps: 10,
            predict_steps: 40,
            correct_every: 10,
            prediction_config: PredictionConfig::default(),
            ternary_config: TernaryConfig::default(),
            vsa_config: VSAConfig::default(),
            gradient_accumulation: 1,
            max_grad_norm: 1.0,
            adaptive_phases: true,
            loss_threshold: 0.1,
        }
    }
}

impl PhaseConfig {
    /// Set the number of full training steps.
    #[must_use]
    pub const fn with_full_steps(mut self, steps: usize) -> Self {
        self.full_steps = steps;
        self
    }

    /// Set the number of prediction steps.
    #[must_use]
    pub const fn with_predict_steps(mut self, steps: usize) -> Self {
        self.predict_steps = steps;
        self
    }

    /// Set the correction frequency.
    #[must_use]
    pub const fn with_correct_every(mut self, every: usize) -> Self {
        self.correct_every = every;
        self
    }

    /// Set the maximum gradient norm for clipping.
    #[must_use]
    pub const fn with_max_grad_norm(mut self, norm: f32) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set whether to use adaptive phase scheduling.
    #[must_use]
    pub const fn with_adaptive_phases(mut self, adaptive: bool) -> Self {
        self.adaptive_phases = adaptive;
        self
    }

    /// Set the prediction sub-configuration.
    #[must_use]
    pub fn with_prediction_config(mut self, config: PredictionConfig) -> Self {
        self.prediction_config = config;
        self
    }

    /// Set the ternary sub-configuration.
    #[must_use]
    pub fn with_ternary_config(mut self, config: TernaryConfig) -> Self {
        self.ternary_config = config;
        self
    }

    /// Set the VSA sub-configuration.
    #[must_use]
    pub fn with_vsa_config(mut self, config: VSAConfig) -> Self {
        self.vsa_config = config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsa_config_defaults() {
        let config = VSAConfig::default();
        assert_eq!(config.dimension, 8192);
        assert!((config.compression_ratio - 0.1).abs() < 0.001);
        assert!(config.use_ternary);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_vsa_config_builder() {
        let config = VSAConfig::default()
            .with_compression_ratio(0.2)
            .with_ternary(false)
            .with_seed(123);

        assert!((config.compression_ratio - 0.2).abs() < 0.001);
        assert!(!config.use_ternary);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_ternary_config_defaults() {
        let config = TernaryConfig::default();
        assert_eq!(config.accumulation_steps, 8);
        assert!(config.use_stochastic_rounding);
    }

    #[test]
    fn test_prediction_config_defaults() {
        let config = PredictionConfig::default();
        assert_eq!(config.history_size, 5);
        assert_eq!(config.prediction_steps, 4);
        assert!((config.momentum - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_phase_config_defaults() {
        let config = PhaseConfig::default();
        assert_eq!(config.full_steps, 10);
        assert_eq!(config.predict_steps, 40);
        assert_eq!(config.correct_every, 10);
        assert!(config.adaptive_phases);
    }

    #[test]
    fn test_phase_config_builder() {
        let config = PhaseConfig::default()
            .with_full_steps(5)
            .with_predict_steps(20)
            .with_correct_every(5)
            .with_adaptive_phases(false);

        assert_eq!(config.full_steps, 5);
        assert_eq!(config.predict_steps, 20);
        assert_eq!(config.correct_every, 5);
        assert!(!config.adaptive_phases);
    }
}

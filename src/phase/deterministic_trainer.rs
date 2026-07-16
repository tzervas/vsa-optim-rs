//! Deterministic phase trainer implementation.
//!
//! Orchestrates phase-based training using deterministic gradient prediction.
//! This trainer guarantees reproducible training outcomes by:
//!
//! 1. Using deterministic least-squares gradient model fitting
//! 2. Tracking residuals for drift correction
//! 3. Requiring warmup before prediction begins
//!
//! # Training Phases
//!
//! ```text
//! WARMUP ──► FULL ──► PREDICT ──► CORRECT ──► FULL ──► ...
//!   │                    │            │
//!   │                    │            └─► Extract residual, refit model
//!   │                    └─► Use predicted gradients
//!   └─► Build gradient history for model fitting
//! ```
//!
//! # Determinism Guarantees
//!
//! - Same random seed + same data order = identical training trajectory
//! - No stochastic operations in prediction
//! - Residuals ensure predictions converge to actual gradients over time

use std::collections::{HashMap, VecDeque};

use candle_core::{Device, Tensor};

use crate::error::{OptimError, Result};
use crate::prediction::{DeterministicPredictionConfig, DeterministicPredictor};

use super::loss_history::{LossHistory, LossHistoryConfig};

fn warn_cpu_fallback(device: &Device) {
    static WARN_ONCE: std::sync::Once = std::sync::Once::new();
    if matches!(device, Device::Cpu) {
        WARN_ONCE.call_once(|| {
            eprintln!(
                "vsa-optim-rs: CPU device in use. CUDA is the intended default; use Device::cuda_if_available(0) when possible."
            );
        });
    }
}

/// Configuration for deterministic phase training.
#[derive(Debug, Clone)]
pub struct DeterministicPhaseConfig {
    /// Warmup steps before prediction begins.
    pub warmup_steps: usize,

    /// Full gradient steps per cycle (after warmup).
    pub full_steps: usize,

    /// Prediction steps per cycle.
    pub predict_steps: usize,

    /// Correction frequency during prediction phase.
    pub correct_every: usize,

    /// History window for model fitting.
    pub history_window: usize,

    /// Whether to adaptively adjust phase lengths.
    pub adaptive_phases: bool,

    /// Loss threshold for triggering more full steps.
    pub loss_threshold: f32,

    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f32,

    /// Whether to enable loss history tracking.
    pub track_loss_history: bool,

    /// Configuration for loss history (if enabled).
    pub loss_history_config: LossHistoryConfig,
}

impl Default for DeterministicPhaseConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 10,
            full_steps: 5,
            predict_steps: 20,
            correct_every: 5,
            history_window: 8,
            adaptive_phases: true,
            loss_threshold: 0.1,
            max_grad_norm: 1.0,
            track_loss_history: true,
            loss_history_config: LossHistoryConfig::default(),
        }
    }
}

impl DeterministicPhaseConfig {
    /// Builder: Set warmup steps.
    #[must_use]
    pub const fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Builder: Set full steps per cycle.
    #[must_use]
    pub const fn with_full_steps(mut self, steps: usize) -> Self {
        self.full_steps = steps;
        self
    }

    /// Builder: Set prediction steps per cycle.
    #[must_use]
    pub const fn with_predict_steps(mut self, steps: usize) -> Self {
        self.predict_steps = steps;
        self
    }

    /// Builder: Set correction frequency.
    #[must_use]
    pub const fn with_correct_every(mut self, every: usize) -> Self {
        self.correct_every = every;
        self
    }

    /// Builder: Enable or disable loss history tracking.
    #[must_use]
    pub const fn with_loss_tracking(mut self, enabled: bool) -> Self {
        self.track_loss_history = enabled;
        self
    }

    /// Builder: Set loss history configuration.
    #[must_use]
    pub fn with_loss_history_config(mut self, config: LossHistoryConfig) -> Self {
        self.loss_history_config = config;
        self
    }
}

/// Training phase for deterministic phase trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeterministicPhase {
    /// Initial warmup phase - always compute full gradients.
    Warmup,
    /// Full gradient computation phase.
    Full,
    /// Prediction phase - use predicted gradients.
    Predict,
    /// Correction phase - compute full gradients and update residuals.
    Correct,
}

impl std::fmt::Display for DeterministicPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Warmup => write!(f, "WARMUP"),
            Self::Full => write!(f, "FULL"),
            Self::Predict => write!(f, "PREDICT"),
            Self::Correct => write!(f, "CORRECT"),
        }
    }
}

/// Step information from the phase trainer.
#[derive(Debug, Clone)]
pub struct DeterministicStepInfo {
    /// Current phase.
    pub phase: DeterministicPhase,
    /// Step within current phase.
    pub phase_step: usize,
    /// Total training steps.
    pub total_step: usize,
    /// Training cycle count (after warmup).
    pub cycle: usize,
    /// Whether phase changed this step.
    pub phase_changed: bool,
    /// Whether backward pass is needed.
    pub needs_backward: bool,
}

/// Training statistics.
#[derive(Debug, Clone)]
pub struct DeterministicTrainerStats {
    /// Total steps taken.
    pub total_steps: usize,
    /// Warmup steps taken.
    pub warmup_steps: usize,
    /// Full gradient steps taken.
    pub full_steps: usize,
    /// Prediction steps taken.
    pub predict_steps: usize,
    /// Correction steps taken.
    pub correct_steps: usize,
    /// Training cycles completed.
    pub cycles: usize,
    /// Effective speedup ratio.
    pub speedup: f32,
    /// Mean prediction error.
    pub mean_prediction_error: f32,
    /// Current loss (most recent).
    pub current_loss: f32,
}

impl std::fmt::Display for DeterministicTrainerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {} | Cycles: {} | Speedup: {:.2}x | Warmup: {} | Full: {} | Predict: {} | Correct: {}",
            self.total_steps,
            self.cycles,
            self.speedup,
            self.warmup_steps,
            self.full_steps,
            self.predict_steps,
            self.correct_steps
        )
    }
}

/// Deterministic phase-based trainer.
///
/// Orchestrates training with guaranteed deterministic outcomes.
/// Uses warmup → full → predict → correct cycle with residual tracking.
pub struct DeterministicPhaseTrainer {
    config: DeterministicPhaseConfig,
    device: Device,

    /// Deterministic gradient predictor.
    predictor: DeterministicPredictor,

    /// Current phase.
    current_phase: DeterministicPhase,

    /// Step within current phase.
    phase_step: usize,

    /// Total training steps.
    total_step: usize,

    /// Training cycles (full → predict → correct sequences).
    cycle_count: usize,

    /// Steps taken per phase.
    warmup_steps_taken: usize,
    full_steps_taken: usize,
    predict_steps_taken: usize,
    correct_steps_taken: usize,

    /// Recent losses for adaptive scheduling.
    recent_losses: VecDeque<f32>,

    /// Last recorded loss.
    last_loss: f32,

    /// Whether warmup is complete.
    warmup_complete: bool,

    /// Effective full steps per cycle (may adapt).
    effective_full_steps: usize,

    /// Effective predict steps per cycle (may adapt).
    effective_predict_steps: usize,

    /// Loss history tracker (optional).
    loss_history: Option<LossHistory>,
}

impl DeterministicPhaseTrainer {
    /// Create a new deterministic phase trainer.
    ///
    /// # Arguments
    ///
    /// * `param_shapes` - List of (name, shape) tuples for parameters
    /// * `config` - Phase training configuration
    /// * `device` - Device for tensor storage
    ///
    /// # Errors
    ///
    /// Returns error if predictor initialization fails.
    pub fn new(
        param_shapes: &[(String, Vec<usize>)],
        config: DeterministicPhaseConfig,
        device: &Device,
    ) -> Result<Self> {
        warn_cpu_fallback(device);
        let prediction_config = DeterministicPredictionConfig {
            warmup_steps: config.warmup_steps,
            history_window: config.history_window,
            prediction_horizon: config.predict_steps,
            history_decay: 0.95,
            residual_threshold: 0.5,
        };

        let predictor = DeterministicPredictor::new(param_shapes, prediction_config, device)?;

        let loss_history = if config.track_loss_history {
            Some(LossHistory::with_config(config.loss_history_config.clone()))
        } else {
            None
        };

        Ok(Self {
            effective_full_steps: config.full_steps,
            effective_predict_steps: config.predict_steps,
            config,
            device: device.clone(),
            predictor,
            current_phase: DeterministicPhase::Warmup,
            phase_step: 0,
            total_step: 0,
            cycle_count: 0,
            warmup_steps_taken: 0,
            full_steps_taken: 0,
            predict_steps_taken: 0,
            correct_steps_taken: 0,
            recent_losses: VecDeque::with_capacity(100),
            last_loss: 0.0,
            warmup_complete: false,
            loss_history,
        })
    }

    /// Begin a training step.
    ///
    /// Returns information about the current phase and whether
    /// backward pass (full gradient computation) is needed.
    pub fn begin_step(&mut self) -> Result<DeterministicStepInfo> {
        // Check for phase transitions
        let (next_phase, phase_changed) = self.compute_next_phase();
        if phase_changed {
            self.transition_to(next_phase);
        }

        // Determine if backward is needed
        let needs_backward = matches!(
            self.current_phase,
            DeterministicPhase::Warmup | DeterministicPhase::Full | DeterministicPhase::Correct
        );

        Ok(DeterministicStepInfo {
            phase: self.current_phase,
            phase_step: self.phase_step,
            total_step: self.total_step,
            cycle: self.cycle_count,
            phase_changed,
            needs_backward,
        })
    }

    /// Compute the next phase based on current state.
    fn compute_next_phase(&self) -> (DeterministicPhase, bool) {
        match self.current_phase {
            DeterministicPhase::Warmup => {
                if self.predictor.is_ready() {
                    (DeterministicPhase::Full, true)
                } else {
                    (DeterministicPhase::Warmup, false)
                }
            }
            DeterministicPhase::Full => {
                if self.phase_step >= self.effective_full_steps {
                    (DeterministicPhase::Predict, true)
                } else {
                    (DeterministicPhase::Full, false)
                }
            }
            DeterministicPhase::Predict => {
                // Check for correction
                if self.phase_step > 0 && self.phase_step % self.config.correct_every == 0 {
                    return (DeterministicPhase::Correct, true);
                }
                // Check for residual-triggered correction
                if self.predictor.needs_correction() {
                    return (DeterministicPhase::Correct, true);
                }
                // Check for cycle completion
                if self.phase_step >= self.effective_predict_steps {
                    return (DeterministicPhase::Full, true);
                }
                (DeterministicPhase::Predict, false)
            }
            DeterministicPhase::Correct => {
                // After correction, continue predict or start new cycle
                let remaining = self.effective_predict_steps.saturating_sub(self.phase_step);
                if remaining > 0 {
                    (DeterministicPhase::Predict, true)
                } else {
                    (DeterministicPhase::Full, true)
                }
            }
        }
    }

    /// Handle phase transition.
    fn transition_to(&mut self, new_phase: DeterministicPhase) {
        let old_phase = self.current_phase;
        self.current_phase = new_phase;

        match new_phase {
            DeterministicPhase::Warmup => {
                // Shouldn't happen - warmup only at start
            }
            DeterministicPhase::Full => {
                // Starting new cycle
                if old_phase != DeterministicPhase::Warmup {
                    self.cycle_count += 1;
                }
                self.phase_step = 0;
                self.warmup_complete = true;

                // Adaptive phase adjustment
                if self.config.adaptive_phases {
                    self.adjust_phase_lengths();
                }
            }
            DeterministicPhase::Predict => {
                if old_phase == DeterministicPhase::Full {
                    self.phase_step = 0;
                }
                // Don't reset phase_step when returning from correction
            }
            DeterministicPhase::Correct => {
                // Don't reset phase_step - we continue prediction count
            }
        }
    }

    /// Adjust phase lengths based on training dynamics.
    fn adjust_phase_lengths(&mut self) {
        if self.recent_losses.len() < 20 {
            return;
        }

        let losses: Vec<f32> = self.recent_losses.iter().copied().collect();
        let early: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let late: f32 = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;

        if late > early * (1.0 + self.config.loss_threshold) {
            // Loss increasing: more full training, less prediction
            self.effective_full_steps = (self.effective_full_steps + 2).min(30);
            self.effective_predict_steps = self.effective_predict_steps.saturating_sub(5).max(5);
        } else if late < early * 0.95 {
            // Loss decreasing well: can use more prediction
            self.effective_full_steps = self.effective_full_steps.saturating_sub(1).max(3);
            self.effective_predict_steps = (self.effective_predict_steps + 3).min(50);
        }
    }

    /// Check if backward pass is needed for current step.
    #[must_use]
    pub fn needs_backward(&self) -> bool {
        matches!(
            self.current_phase,
            DeterministicPhase::Warmup | DeterministicPhase::Full | DeterministicPhase::Correct
        )
    }

    /// Record full gradients after backward pass.
    ///
    /// Called during WARMUP, FULL, or CORRECT phases after computing
    /// gradients via backpropagation.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Map of parameter names to gradient tensors
    pub fn record_full_gradients(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        let is_correction = self.current_phase == DeterministicPhase::Correct;
        self.predictor.record_gradient(gradients, is_correction)?;
        Ok(())
    }

    /// Get predicted gradients for current step.
    ///
    /// Called during PREDICT phase to get deterministic gradient predictions.
    ///
    /// # Returns
    ///
    /// Map of parameter names to predicted gradient tensors.
    pub fn get_predicted_gradients(&mut self) -> Result<HashMap<String, Tensor>> {
        if !self.warmup_complete {
            return Err(OptimError::Prediction(
                "Cannot predict during warmup phase".to_string(),
            ));
        }
        self.predictor.predict_gradient()
    }

    /// End the current training step.
    ///
    /// Updates internal state and statistics.
    ///
    /// # Arguments
    ///
    /// * `loss` - Loss value for this step
    #[allow(clippy::cast_precision_loss)]
    pub fn end_step(&mut self, loss: f32) -> Result<()> {
        // Track loss
        if self.recent_losses.len() >= 100 {
            self.recent_losses.pop_front();
        }
        self.recent_losses.push_back(loss);
        self.last_loss = loss;

        // Record to loss history if enabled
        if let Some(history) = &mut self.loss_history {
            history.record(loss, self.current_phase);
        }

        // Update phase-specific counters
        match self.current_phase {
            DeterministicPhase::Warmup => self.warmup_steps_taken += 1,
            DeterministicPhase::Full => self.full_steps_taken += 1,
            DeterministicPhase::Predict => self.predict_steps_taken += 1,
            DeterministicPhase::Correct => self.correct_steps_taken += 1,
        }

        // Update step counters
        self.phase_step += 1;
        self.total_step += 1;

        Ok(())
    }

    /// Get current training phase.
    #[must_use]
    pub const fn current_phase(&self) -> DeterministicPhase {
        self.current_phase
    }

    /// Check if warmup is complete.
    #[must_use]
    pub const fn warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    /// Get training statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn get_stats(&self) -> DeterministicTrainerStats {
        // Calculate speedup: total forward steps / backward steps
        let total_forward = self.total_step as f32;
        let total_backward =
            (self.warmup_steps_taken + self.full_steps_taken + self.correct_steps_taken).max(1)
                as f32;
        let speedup = total_forward / total_backward;

        DeterministicTrainerStats {
            total_steps: self.total_step,
            warmup_steps: self.warmup_steps_taken,
            full_steps: self.full_steps_taken,
            predict_steps: self.predict_steps_taken,
            correct_steps: self.correct_steps_taken,
            cycles: self.cycle_count,
            speedup,
            mean_prediction_error: self.predictor.get_stats().mean_abs_error,
            current_loss: self.last_loss,
        }
    }

    /// Reset trainer state.
    pub fn reset(&mut self) -> Result<()> {
        self.predictor.reset()?;
        self.current_phase = DeterministicPhase::Warmup;
        self.phase_step = 0;
        self.total_step = 0;
        self.cycle_count = 0;
        self.warmup_steps_taken = 0;
        self.full_steps_taken = 0;
        self.predict_steps_taken = 0;
        self.correct_steps_taken = 0;
        self.recent_losses.clear();
        self.last_loss = 0.0;
        self.warmup_complete = false;
        self.effective_full_steps = self.config.full_steps;
        self.effective_predict_steps = self.config.predict_steps;
        if let Some(history) = &mut self.loss_history {
            history.clear();
        }
        Ok(())
    }

    /// Get access to the loss history tracker.
    ///
    /// Returns `None` if loss tracking is disabled.
    #[must_use]
    pub fn loss_history(&self) -> Option<&LossHistory> {
        self.loss_history.as_ref()
    }

    /// Get mutable access to the loss history tracker.
    ///
    /// Returns `None` if loss tracking is disabled.
    #[must_use]
    pub fn loss_history_mut(&mut self) -> Option<&mut LossHistory> {
        self.loss_history.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_shapes() -> Vec<(String, Vec<usize>)> {
        vec![
            ("layer.weight".to_string(), vec![16, 32]),
            ("layer.bias".to_string(), vec![16]),
        ]
    }

    fn create_mock_gradients(device: &Device, scale: f32) -> HashMap<String, Tensor> {
        let mut grads = HashMap::new();
        grads.insert(
            "layer.weight".to_string(),
            Tensor::ones((16, 32), candle_core::DType::F32, device)
                .unwrap()
                .affine(scale as f64, 0.0)
                .unwrap(),
        );
        grads.insert(
            "layer.bias".to_string(),
            Tensor::ones(16, candle_core::DType::F32, device)
                .unwrap()
                .affine(scale as f64, 0.0)
                .unwrap(),
        );
        grads
    }

    #[test]
    fn test_warmup_to_full_transition() {
        let config = DeterministicPhaseConfig::default()
            .with_warmup_steps(5)
            .with_full_steps(3);

        let mut trainer =
            DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

        // Should start in warmup
        let info = trainer.begin_step().unwrap();
        assert_eq!(info.phase, DeterministicPhase::Warmup);
        assert!(info.needs_backward);

        // Run through warmup
        for i in 0..5 {
            let grads = create_mock_gradients(&Device::Cpu, 1.0 + i as f32 * 0.1);
            trainer.record_full_gradients(&grads).unwrap();
            trainer.end_step(1.0 - i as f32 * 0.1).unwrap();
            trainer.begin_step().unwrap();
        }

        // Should now be in FULL phase
        assert!(trainer.warmup_complete());
        assert_eq!(trainer.current_phase(), DeterministicPhase::Full);
    }

    #[test]
    fn test_full_cycle() {
        let config = DeterministicPhaseConfig::default()
            .with_warmup_steps(3)
            .with_full_steps(2)
            .with_predict_steps(4)
            .with_correct_every(2);

        let mut trainer =
            DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

        let mut phases_seen = Vec::new();

        // Run 20 steps
        for i in 0..20 {
            let info = trainer.begin_step().unwrap();
            phases_seen.push(info.phase);

            if info.needs_backward {
                let grads = create_mock_gradients(&Device::Cpu, 1.0 + i as f32 * 0.05);
                trainer.record_full_gradients(&grads).unwrap();
            } else {
                let _predicted = trainer.get_predicted_gradients().unwrap();
            }

            trainer.end_step(1.0 / (i + 1) as f32).unwrap();
        }

        // Should have seen all phase types
        assert!(phases_seen.contains(&DeterministicPhase::Warmup));
        assert!(phases_seen.contains(&DeterministicPhase::Full));
        assert!(phases_seen.contains(&DeterministicPhase::Predict));
        // Correction may or may not trigger depending on residuals
    }

    #[test]
    fn test_deterministic_stats() {
        let config = DeterministicPhaseConfig::default()
            .with_warmup_steps(5)
            .with_full_steps(2)
            .with_predict_steps(8);

        let mut trainer =
            DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

        // Run some steps
        for i in 0..15 {
            let info = trainer.begin_step().unwrap();
            if info.needs_backward {
                let grads = create_mock_gradients(&Device::Cpu, 1.0);
                trainer.record_full_gradients(&grads).unwrap();
            } else {
                let _ = trainer.get_predicted_gradients();
            }
            trainer.end_step(0.5).unwrap();
        }

        let stats = trainer.get_stats();
        assert_eq!(stats.total_steps, 15);
        assert!(stats.speedup >= 1.0);
        assert!(stats.warmup_steps > 0);
    }
}

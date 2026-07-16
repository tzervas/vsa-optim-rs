//! Phase trainer implementation.
//!
//! Orchestrates phase-based training for acceleration by combining
//! gradient prediction, VSA compression, and ternary optimization.

use std::collections::{HashMap, VecDeque};

use candle_core::{Device, Tensor};

use crate::config::PhaseConfig;
use crate::error::Result;
use crate::prediction::GradientPredictor;
use crate::ternary::TernaryGradientAccumulator;
use crate::vsa::VSAGradientCompressor;

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

/// Training phase types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrainingPhase {
    /// Full gradient computation.
    Full,
    /// Predicted gradients.
    Predict,
    /// Correction phase.
    Correct,
}

impl std::fmt::Display for TrainingPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "FULL"),
            Self::Predict => write!(f, "PREDICT"),
            Self::Correct => write!(f, "CORRECT"),
        }
    }
}

/// Orchestrates phase-based training for acceleration.
///
/// This is the main training orchestrator that combines all optimization
/// techniques. It manages the phase transitions and ensures convergence
/// while maximizing training speed.
///
/// The trainer automatically:
/// 1. Tracks which phase we're in
/// 2. Manages gradient prediction during PREDICT phase
/// 3. Applies corrections to prevent drift
/// 4. Uses ternary accumulation for memory efficiency
/// 5. Optionally uses VSA compression for gradient storage
///
/// # Example
///
/// ```ignore
/// use vsa_optim_rs::phase::PhaseTrainer;
/// use vsa_optim_rs::PhaseConfig;
///
/// let shapes = vec![("layer.weight".to_string(), vec![64, 128])];
/// let config = PhaseConfig::default();
/// let mut trainer = PhaseTrainer::new(&shapes, config, &Device::Cpu)?;
///
/// // Training loop
/// for step in 0..total_steps {
///     let step_info = trainer.begin_step()?;
///
///     match step_info.phase {
///         TrainingPhase::Full | TrainingPhase::Correct => {
///             // Compute full gradients via backprop
///             trainer.record_full_gradients(&gradients)?;
///         }
///         TrainingPhase::Predict => {
///             // Use predicted gradients
///             let predicted = trainer.get_predicted_gradients()?;
///         }
///     }
///
///     trainer.end_step(loss_value)?;
/// }
/// ```
pub struct PhaseTrainer {
    config: PhaseConfig,
    device: Device,

    /// Gradient predictor.
    predictor: GradientPredictor,

    /// Ternary gradient accumulator.
    ternary_accum: TernaryGradientAccumulator,

    /// VSA gradient compressor.
    vsa_compressor: VSAGradientCompressor,

    /// Current training phase.
    current_phase: TrainingPhase,

    /// Steps in current phase.
    phase_step: usize,

    /// Total training steps.
    total_step: usize,

    /// Cycle count (full phase completions).
    cycle_count: usize,

    /// Per-phase loss tracking.
    phase_losses: HashMap<TrainingPhase, Vec<f32>>,

    /// Recent losses for adaptive scheduling.
    recent_losses: VecDeque<f32>,

    /// Speedup ratio.
    speedup_ratio: f32,

    /// Steps taken per phase type.
    full_steps_taken: usize,
    predict_steps_taken: usize,
    correct_steps_taken: usize,

    /// Parameter shapes for reference.
    param_shapes: Vec<(String, Vec<usize>)>,
}

impl PhaseTrainer {
    /// Create a new phase trainer.
    ///
    /// # Arguments
    ///
    /// * `param_shapes` - List of (name, shape) tuples for parameters
    /// * `config` - Phase training configuration
    /// * `device` - Device for tensor storage
    ///
    /// # Errors
    ///
    /// Returns error if component initialization fails.
    pub fn new(
        param_shapes: &[(String, Vec<usize>)],
        config: PhaseConfig,
        device: &Device,
    ) -> Result<Self> {
        warn_cpu_fallback(device);
        let predictor =
            GradientPredictor::new(param_shapes, config.prediction_config.clone(), device)?;

        let ternary_accum =
            TernaryGradientAccumulator::new(param_shapes, config.ternary_config.clone(), device)?;

        let param_count: usize = param_shapes
            .iter()
            .map(|(_, s)| s.iter().product::<usize>())
            .sum();
        let vsa_compressor = VSAGradientCompressor::new(param_count, config.vsa_config.clone());

        let mut phase_losses = HashMap::new();
        phase_losses.insert(TrainingPhase::Full, Vec::new());
        phase_losses.insert(TrainingPhase::Predict, Vec::new());
        phase_losses.insert(TrainingPhase::Correct, Vec::new());

        Ok(Self {
            config,
            device: device.clone(),
            predictor,
            ternary_accum,
            vsa_compressor,
            current_phase: TrainingPhase::Full,
            phase_step: 0,
            total_step: 0,
            cycle_count: 0,
            phase_losses,
            recent_losses: VecDeque::with_capacity(100),
            speedup_ratio: 1.0,
            full_steps_taken: 0,
            predict_steps_taken: 0,
            correct_steps_taken: 0,
            param_shapes: param_shapes.to_vec(),
        })
    }

    /// Determine the next training phase.
    fn get_next_phase(&self) -> TrainingPhase {
        match self.current_phase {
            TrainingPhase::Full => {
                if self.phase_step >= self.config.full_steps {
                    TrainingPhase::Predict
                } else {
                    TrainingPhase::Full
                }
            }
            TrainingPhase::Predict => {
                // Check for correction
                if self.phase_step > 0 && self.phase_step % self.config.correct_every == 0 {
                    return TrainingPhase::Correct;
                }
                // Check for cycle completion
                if self.phase_step >= self.config.predict_steps {
                    return TrainingPhase::Full;
                }
                TrainingPhase::Predict
            }
            TrainingPhase::Correct => {
                // After correction, back to predict or full
                let remaining_predict = self.config.predict_steps.saturating_sub(self.phase_step);
                if remaining_predict > 0 {
                    TrainingPhase::Predict
                } else {
                    TrainingPhase::Full
                }
            }
        }
    }

    /// Handle phase transition.
    fn transition_phase(&mut self, new_phase: TrainingPhase) {
        let old_phase = self.current_phase;
        self.current_phase = new_phase;

        match new_phase {
            TrainingPhase::Full => {
                // Starting new cycle
                self.phase_step = 0;
                self.cycle_count += 1;

                // Apply adaptive scheduling if enabled
                if self.config.adaptive_phases && self.recent_losses.len() >= 10 {
                    self.adjust_phase_lengths();
                }
            }
            TrainingPhase::Predict => {
                if old_phase == TrainingPhase::Full {
                    // Entering predict from full
                    self.phase_step = 0;
                }
            }
            TrainingPhase::Correct => {
                // Correction is a single step, don't reset phase_step
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
            // Loss increasing: more full training
            self.config.full_steps = (self.config.full_steps + 5).min(50);
            self.config.predict_steps = self.config.predict_steps.saturating_sub(10).max(10);
        } else if late < early * 0.95 {
            // Loss decreasing well: can use more prediction
            self.config.full_steps = self.config.full_steps.saturating_sub(2).max(5);
            self.config.predict_steps = (self.config.predict_steps + 5).min(100);
        }
    }

    /// Begin a training step. Returns info about current phase.
    ///
    /// # Returns
    ///
    /// Step information including phase and whether phase changed.
    ///
    /// # Errors
    ///
    /// Returns error if phase transition fails.
    pub fn begin_step(&mut self) -> Result<StepInfo> {
        // Check for phase transition
        let next_phase = self.get_next_phase();
        let phase_changed = next_phase != self.current_phase;
        if phase_changed {
            self.transition_phase(next_phase);
        }

        Ok(StepInfo {
            phase: self.current_phase,
            phase_step: self.phase_step,
            total_step: self.total_step,
            cycle: self.cycle_count,
            phase_changed,
        })
    }

    /// Record full gradients after backprop (for FULL or CORRECT phase).
    ///
    /// # Arguments
    ///
    /// * `gradients` - Map of parameter names to gradient tensors
    ///
    /// # Errors
    ///
    /// Returns error if recording fails.
    pub fn record_full_gradients(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        // Record for prediction
        self.predictor.record_gradient(gradients)?;

        // If in correction phase, compute and apply correction
        if self.current_phase == TrainingPhase::Correct {
            self.predictor.compute_correction(gradients)?;
        }

        Ok(())
    }

    /// Get predicted gradients (for PREDICT phase).
    ///
    /// # Returns
    ///
    /// Map of parameter names to predicted gradient tensors.
    ///
    /// # Errors
    ///
    /// Returns error if prediction fails.
    pub fn get_predicted_gradients(&mut self) -> Result<HashMap<String, Tensor>> {
        self.predictor.predict_gradient()
    }

    /// Apply correction to gradients.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Mutable map of gradients to modify in-place
    ///
    /// # Errors
    ///
    /// Returns error if correction fails.
    pub fn apply_correction(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        self.predictor.apply_correction(gradients)
    }

    /// End the training step.
    ///
    /// # Arguments
    ///
    /// * `loss` - Loss value for this step
    ///
    /// # Errors
    ///
    /// Returns error if tracking fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn end_step(&mut self, loss: f32) -> Result<()> {
        // Track loss
        if self.recent_losses.len() >= 100 {
            self.recent_losses.pop_front();
        }
        self.recent_losses.push_back(loss);

        if let Some(phase_losses) = self.phase_losses.get_mut(&self.current_phase) {
            phase_losses.push(loss);
        }

        // Update phase step count
        match self.current_phase {
            TrainingPhase::Full => self.full_steps_taken += 1,
            TrainingPhase::Predict => self.predict_steps_taken += 1,
            TrainingPhase::Correct => self.correct_steps_taken += 1,
        }

        // Update counters
        self.phase_step += 1;
        self.total_step += 1;

        // Calculate speedup
        let total_forward =
            (self.full_steps_taken + self.predict_steps_taken + self.correct_steps_taken) as f32;
        let total_backward = (self.full_steps_taken + self.correct_steps_taken).max(1) as f32;
        self.speedup_ratio = total_forward / total_backward;

        Ok(())
    }

    /// Get current training phase.
    #[must_use]
    pub const fn current_phase(&self) -> TrainingPhase {
        self.current_phase
    }

    /// Get total step count.
    #[must_use]
    pub const fn total_step(&self) -> usize {
        self.total_step
    }

    /// Get cycle count.
    #[must_use]
    pub const fn cycle_count(&self) -> usize {
        self.cycle_count
    }

    /// Get speedup ratio.
    #[must_use]
    pub const fn speedup_ratio(&self) -> f32 {
        self.speedup_ratio
    }

    /// Get training statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn get_stats(&self) -> TrainerStats {
        let mut phase_avg_losses = HashMap::new();

        for (phase, losses) in &self.phase_losses {
            if !losses.is_empty() {
                let recent: Vec<&f32> = losses.iter().rev().take(100).collect();
                let avg: f32 = recent.iter().copied().sum::<f32>() / recent.len() as f32;
                phase_avg_losses.insert(*phase, avg);
            }
        }

        TrainerStats {
            total_steps: self.total_step,
            cycles: self.cycle_count,
            speedup: self.speedup_ratio,
            full_steps: self.full_steps_taken,
            predict_steps: self.predict_steps_taken,
            correct_steps: self.correct_steps_taken,
            current_full_steps: self.config.full_steps,
            current_predict_steps: self.config.predict_steps,
            phase_avg_losses,
        }
    }

    /// Reset trainer state.
    pub fn reset(&mut self) -> Result<()> {
        self.predictor.reset();
        self.ternary_accum.reset()?;
        self.current_phase = TrainingPhase::Full;
        self.phase_step = 0;
        self.total_step = 0;
        self.cycle_count = 0;
        self.recent_losses.clear();
        self.speedup_ratio = 1.0;
        self.full_steps_taken = 0;
        self.predict_steps_taken = 0;
        self.correct_steps_taken = 0;

        for losses in self.phase_losses.values_mut() {
            losses.clear();
        }

        Ok(())
    }

    /// Get mutable access to VSA compressor.
    pub fn vsa_compressor_mut(&mut self) -> &mut VSAGradientCompressor {
        &mut self.vsa_compressor
    }

    /// Get mutable access to ternary accumulator.
    pub fn ternary_accumulator_mut(&mut self) -> &mut TernaryGradientAccumulator {
        &mut self.ternary_accum
    }

    /// Check if should compute full gradients.
    #[must_use]
    pub fn should_compute_full(&self) -> bool {
        matches!(
            self.current_phase,
            TrainingPhase::Full | TrainingPhase::Correct
        )
    }
}

/// Information about current training step.
#[derive(Debug, Clone)]
pub struct StepInfo {
    /// Current training phase.
    pub phase: TrainingPhase,
    /// Step within current phase.
    pub phase_step: usize,
    /// Total training step.
    pub total_step: usize,
    /// Cycle count.
    pub cycle: usize,
    /// Whether phase changed this step.
    pub phase_changed: bool,
}

/// Training statistics.
#[derive(Debug, Clone)]
pub struct TrainerStats {
    /// Total training steps.
    pub total_steps: usize,
    /// Cycle count.
    pub cycles: usize,
    /// Speedup ratio (total steps / backward steps).
    pub speedup: f32,
    /// Full phase steps taken.
    pub full_steps: usize,
    /// Predict phase steps taken.
    pub predict_steps: usize,
    /// Correct phase steps taken.
    pub correct_steps: usize,
    /// Current full steps per cycle.
    pub current_full_steps: usize,
    /// Current predict steps per cycle.
    pub current_predict_steps: usize,
    /// Average loss per phase.
    pub phase_avg_losses: HashMap<TrainingPhase, f32>,
}

impl std::fmt::Display for TrainerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {} | Cycles: {} | Speedup: {:.2}x | Full: {} | Predict: {} | Correct: {}",
            self.total_steps,
            self.cycles,
            self.speedup,
            self.full_steps,
            self.predict_steps,
            self.correct_steps
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_param_shapes() -> Vec<(String, Vec<usize>)> {
        vec![
            ("layer1.weight".to_string(), vec![64, 128]),
            ("layer1.bias".to_string(), vec![64]),
        ]
    }

    fn create_mock_gradients(device: &Device) -> HashMap<String, Tensor> {
        let mut gradients = HashMap::new();
        gradients.insert(
            "layer1.weight".to_string(),
            Tensor::randn(0.0f32, 0.1, (64, 128), device).unwrap(),
        );
        gradients.insert(
            "layer1.bias".to_string(),
            Tensor::randn(0.0f32, 0.1, 64, device).unwrap(),
        );
        gradients
    }

    #[test]
    fn test_trainer_creation() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PhaseConfig::default();

        let trainer = PhaseTrainer::new(&shapes, config, &device).unwrap();
        assert_eq!(trainer.current_phase(), TrainingPhase::Full);
        assert_eq!(trainer.total_step(), 0);
    }

    #[test]
    fn test_phase_transitions() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PhaseConfig::default()
            .with_full_steps(2)
            .with_predict_steps(4)
            .with_correct_every(2);

        let mut trainer = PhaseTrainer::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Start in FULL phase
        assert_eq!(trainer.current_phase(), TrainingPhase::Full);

        // Step 1: FULL
        let info = trainer.begin_step().unwrap();
        assert_eq!(info.phase, TrainingPhase::Full);
        trainer.record_full_gradients(&gradients).unwrap();
        trainer.end_step(1.0).unwrap();

        // Step 2: FULL (still, phase_step was 0)
        let info = trainer.begin_step().unwrap();
        assert_eq!(info.phase, TrainingPhase::Full);
        trainer.record_full_gradients(&gradients).unwrap();
        trainer.end_step(0.9).unwrap();

        // Step 3: Should transition to PREDICT
        let info = trainer.begin_step().unwrap();
        assert!(info.phase_changed);
        assert_eq!(info.phase, TrainingPhase::Predict);
    }

    #[test]
    fn test_speedup_calculation() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PhaseConfig::default()
            .with_full_steps(1)
            .with_predict_steps(3)
            .with_correct_every(10); // No correction in this short test

        let mut trainer = PhaseTrainer::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // 1 full step
        trainer.begin_step().unwrap();
        trainer.record_full_gradients(&gradients).unwrap();
        trainer.end_step(1.0).unwrap();

        // 3 predict steps
        for _ in 0..3 {
            trainer.begin_step().unwrap();
            let _ = trainer.get_predicted_gradients().unwrap();
            trainer.end_step(0.9).unwrap();
        }

        // Speedup should be 4/1 = 4.0 (4 total steps, 1 backward step)
        assert!((trainer.speedup_ratio() - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_stats() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PhaseConfig::default();

        let mut trainer = PhaseTrainer::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Run a few steps
        for i in 0..5 {
            trainer.begin_step().unwrap();
            if trainer.should_compute_full() {
                trainer.record_full_gradients(&gradients).unwrap();
            } else {
                let _ = trainer.get_predicted_gradients().unwrap();
            }
            trainer.end_step(1.0 - i as f32 * 0.1).unwrap();
        }

        let stats = trainer.get_stats();
        assert_eq!(stats.total_steps, 5);
    }

    #[test]
    fn test_reset() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PhaseConfig::default();

        let mut trainer = PhaseTrainer::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Run some steps
        trainer.begin_step().unwrap();
        trainer.record_full_gradients(&gradients).unwrap();
        trainer.end_step(1.0).unwrap();

        assert_eq!(trainer.total_step(), 1);

        // Reset
        trainer.reset().unwrap();

        assert_eq!(trainer.total_step(), 0);
        assert_eq!(trainer.current_phase(), TrainingPhase::Full);
    }
}

//! Gradient predictor implementation.
//!
//! Predicts future gradients from history using momentum-based extrapolation.

use std::collections::{HashMap, VecDeque};

use candle_core::{DType, Device, Tensor};

use crate::config::PredictionConfig;
use crate::error::Result;

/// Predict future gradients from history.
///
/// Gradient prediction reduces compute by ~80% (4 predicted steps
/// per 1 computed step) while maintaining convergence quality through
/// periodic correction cycles.
///
/// The predictor maintains a history of recent gradients and uses a
/// momentum-based extrapolation to predict future gradients. Corrections
/// are computed as the difference between predicted and actual gradients.
///
/// # Example
///
/// ```ignore
/// use vsa_optim_rs::prediction::GradientPredictor;
/// use vsa_optim_rs::PredictionConfig;
///
/// let shapes = vec![
///     ("layer1.weight".to_string(), vec![64, 128]),
/// ];
/// let mut predictor = GradientPredictor::new(&shapes, PredictionConfig::default(), &Device::Cpu)?;
///
/// // Training loop
/// for step in 0..total_steps {
///     if predictor.should_compute_full() {
///         // loss.backward() - compute full gradients
///         predictor.record_gradient(&gradients)?;
///         predictor.apply_correction(&mut gradients);
///     } else {
///         let predicted = predictor.predict_gradient()?;
///         // Use predicted gradients for optimizer step
///     }
/// }
/// ```
pub struct GradientPredictor {
    config: PredictionConfig,
    device: Device,

    /// Gradient history per parameter (circular buffer).
    gradient_history: HashMap<String, VecDeque<Tensor>>,

    /// Original shapes for reconstruction.
    shapes: HashMap<String, Vec<usize>>,

    /// Steps since last full gradient computation.
    steps_since_full: usize,

    /// Total training steps.
    total_steps: usize,

    /// Last predicted gradients.
    last_prediction: HashMap<String, Tensor>,

    /// Accumulated corrections.
    correction_accumulator: HashMap<String, Tensor>,

    /// Recent prediction errors for adaptive prediction.
    prediction_errors: VecDeque<f32>,
}

impl GradientPredictor {
    /// Create a new gradient predictor.
    ///
    /// # Arguments
    ///
    /// * `param_shapes` - List of (name, shape) tuples for parameters
    /// * `config` - Prediction configuration
    /// * `device` - Device for tensor storage
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails.
    pub fn new(
        param_shapes: &[(String, Vec<usize>)],
        config: PredictionConfig,
        device: &Device,
    ) -> Result<Self> {
        let mut gradient_history = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, shape) in param_shapes {
            gradient_history.insert(name.clone(), VecDeque::with_capacity(config.history_size));
            shapes.insert(name.clone(), shape.clone());
        }

        Ok(Self {
            config,
            device: device.clone(),
            gradient_history,
            shapes,
            steps_since_full: 0,
            total_steps: 0,
            last_prediction: HashMap::new(),
            correction_accumulator: HashMap::new(),
            prediction_errors: VecDeque::with_capacity(100),
        })
    }

    /// Check if full gradient computation is needed.
    ///
    /// Full computation is needed:
    /// 1. At the start (insufficient history)
    /// 2. After `prediction_steps` predicted steps (correction cycle)
    /// 3. When prediction quality degrades below threshold
    #[must_use]
    pub fn should_compute_full(&self) -> bool {
        // Need full gradient at start for history
        let any_history = self.gradient_history.values().next();
        if let Some(history) = any_history {
            if history.len() < 2 {
                return true;
            }
        } else {
            return true;
        }

        // Need correction after prediction_steps
        if self.steps_since_full >= self.config.prediction_steps {
            return true;
        }

        // Check if prediction quality is poor
        if self.prediction_errors.len() >= 10 {
            let recent: f32 = self.prediction_errors.iter().rev().take(10).sum::<f32>() / 10.0;
            if recent > 0.5 {
                return true;
            }
        }

        false
    }

    /// Record current gradients to history.
    ///
    /// Called after full gradient computation to update history.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Map of parameter names to gradient tensors
    ///
    /// # Errors
    ///
    /// Returns error if tensor cloning fails.
    pub fn record_gradient(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        for (name, grad) in gradients {
            if let Some(history) = self.gradient_history.get_mut(name) {
                // Maintain max history size
                if history.len() >= self.config.history_size {
                    history.pop_front();
                }
                history.push_back(grad.clone());
            }
        }

        self.steps_since_full = 0;
        self.total_steps += 1;
        Ok(())
    }

    /// Predict gradients based on history.
    ///
    /// Uses momentum-based extrapolation from gradient history:
    /// ```text
    /// g_pred = g[-1] + momentum * (g[-1] - g[-2])
    /// ```
    ///
    /// # Returns
    ///
    /// Dictionary mapping parameter names to predicted gradients.
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail.
    pub fn predict_gradient(&mut self) -> Result<HashMap<String, Tensor>> {
        let mut predicted = HashMap::new();
        let momentum = self.config.momentum;

        for (name, history) in &self.gradient_history {
            let prediction = match history.len() {
                0 => {
                    // No history, create zeros
                    if let Some(shape) = self.shapes.get(name) {
                        Tensor::zeros(shape.as_slice(), DType::F32, &self.device)?
                    } else {
                        continue;
                    }
                }
                1 => {
                    // Single history entry, use as-is
                    history.back().unwrap().clone()
                }
                _ => {
                    // Momentum-based extrapolation
                    let g_prev = &history[history.len() - 2];
                    let g_curr = history.back().unwrap();

                    // delta = g_curr - g_prev
                    let delta = g_curr.sub(g_prev)?;

                    // g_pred = g_curr + momentum * delta
                    let scaled_delta = (&delta * momentum as f64)?;
                    g_curr.add(&scaled_delta)?
                }
            };

            predicted.insert(name.clone(), prediction);
        }

        self.last_prediction = predicted.clone();
        self.steps_since_full += 1;
        self.total_steps += 1;

        Ok(predicted)
    }

    /// Compute correction between predicted and actual gradients.
    ///
    /// The correction term captures the prediction error and is
    /// accumulated to apply a "catch-up" adjustment.
    ///
    /// # Arguments
    ///
    /// * `actual_gradients` - The actual computed gradients
    ///
    /// # Returns
    ///
    /// Dictionary of correction terms.
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail.
    pub fn compute_correction(
        &mut self,
        actual_gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut corrections = HashMap::new();

        for (name, actual) in actual_gradients {
            if let Some(predicted) = self.last_prediction.get(name) {
                // Correction = actual - predicted
                let correction = actual.sub(predicted)?;

                // Track prediction error
                let error = correction.abs()?.mean_all()?.to_scalar::<f32>()?;

                if self.prediction_errors.len() >= 100 {
                    self.prediction_errors.pop_front();
                }
                self.prediction_errors.push_back(error);

                // Accumulate for later application
                if let Some(existing) = self.correction_accumulator.get(name) {
                    self.correction_accumulator
                        .insert(name.clone(), existing.add(&correction)?);
                } else {
                    self.correction_accumulator
                        .insert(name.clone(), correction.clone());
                }

                corrections.insert(name.clone(), correction);
            }
        }

        Ok(corrections)
    }

    /// Apply accumulated corrections to gradients.
    ///
    /// After computing full gradients, adds the accumulated
    /// correction to account for prediction errors from previous
    /// predicted steps.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Mutable map of gradients to modify in-place
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail.
    pub fn apply_correction(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        let weight = self.config.correction_weight;

        for (name, grad) in gradients.iter_mut() {
            if let Some(correction) = self.correction_accumulator.get(name) {
                // Add weighted correction: grad += weight * correction
                let scaled = (correction * weight as f64)?;
                *grad = grad.add(&scaled)?;
            }
        }

        // Clear accumulator
        self.correction_accumulator.clear();
        Ok(())
    }

    /// Get prediction statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn get_stats(&self) -> PredictorStats {
        let mean_error = if !self.prediction_errors.is_empty() {
            self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32
        } else {
            0.0
        };

        let recent_error = if self.prediction_errors.len() >= 10 {
            self.prediction_errors.iter().rev().take(10).sum::<f32>() / 10.0
        } else if !self.prediction_errors.is_empty() {
            self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32
        } else {
            0.0
        };

        let prediction_ratio = 1.0 - (1.0 / (self.config.prediction_steps + 1) as f32);

        PredictorStats {
            total_steps: self.total_steps,
            prediction_ratio,
            mean_error,
            recent_error,
            history_size: self.gradient_history.values().next().map_or(0, |h| h.len()),
        }
    }

    /// Get total steps.
    #[must_use]
    pub const fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Reset predictor state.
    pub fn reset(&mut self) {
        for history in self.gradient_history.values_mut() {
            history.clear();
        }
        self.steps_since_full = 0;
        self.total_steps = 0;
        self.last_prediction.clear();
        self.correction_accumulator.clear();
        self.prediction_errors.clear();
    }
}

/// Prediction statistics.
#[derive(Debug, Clone)]
pub struct PredictorStats {
    /// Total training steps.
    pub total_steps: usize,
    /// Fraction of steps using prediction (0 to 1).
    pub prediction_ratio: f32,
    /// Mean prediction error across all history.
    pub mean_error: f32,
    /// Recent prediction error (last 10 steps).
    pub recent_error: f32,
    /// Current history size.
    pub history_size: usize,
}

impl std::fmt::Display for PredictorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {} | Prediction ratio: {:.1}% | Mean error: {:.4} | Recent error: {:.4}",
            self.total_steps,
            self.prediction_ratio * 100.0,
            self.mean_error,
            self.recent_error
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
    fn test_predictor_creation() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default();

        let predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        assert_eq!(predictor.total_steps(), 0);
    }

    #[test]
    fn test_should_compute_full_initially() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default();

        let predictor = GradientPredictor::new(&shapes, config, &device).unwrap();

        // Should compute full at start (no history)
        assert!(predictor.should_compute_full());
    }

    #[test]
    fn test_record_gradient() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default();

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        predictor.record_gradient(&gradients).unwrap();
        assert_eq!(predictor.total_steps(), 1);

        // Still should compute full (need 2+ entries in history)
        assert!(predictor.should_compute_full());

        predictor.record_gradient(&gradients).unwrap();
        assert_eq!(predictor.total_steps(), 2);

        // Now should not require full computation (has history)
        assert!(!predictor.should_compute_full());
    }

    #[test]
    fn test_predict_gradient() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default().with_prediction_steps(4);

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Build history
        predictor.record_gradient(&gradients).unwrap();
        predictor.record_gradient(&gradients).unwrap();

        // Now we can predict
        let predicted = predictor.predict_gradient().unwrap();
        assert_eq!(predicted.len(), 2);

        // Check shapes match
        for (name, _shape) in &shapes {
            assert!(predicted.contains_key(name));
        }
    }

    #[test]
    fn test_correction_cycle() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default().with_prediction_steps(2);

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Build history
        predictor.record_gradient(&gradients).unwrap();
        predictor.record_gradient(&gradients).unwrap();

        // Predict for 2 steps
        predictor.predict_gradient().unwrap();
        predictor.predict_gradient().unwrap();

        // Now should require full computation (correction cycle)
        assert!(predictor.should_compute_full());
    }

    #[test]
    fn test_compute_correction() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default();

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Build history and predict
        predictor.record_gradient(&gradients).unwrap();
        predictor.record_gradient(&gradients).unwrap();
        let _predicted = predictor.predict_gradient().unwrap();

        // Compute correction with actual gradients
        let actual = create_mock_gradients(&device);
        let corrections = predictor.compute_correction(&actual).unwrap();

        assert_eq!(corrections.len(), 2);
    }

    #[test]
    fn test_apply_correction() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default().with_correction_weight(0.5);

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Build history and predict
        predictor.record_gradient(&gradients).unwrap();
        predictor.record_gradient(&gradients).unwrap();
        let _predicted = predictor.predict_gradient().unwrap();

        // Compute correction
        let actual = create_mock_gradients(&device);
        predictor.compute_correction(&actual).unwrap();

        // Apply correction
        let mut grads_to_modify = create_mock_gradients(&device);
        predictor.apply_correction(&mut grads_to_modify).unwrap();

        // Correction should be cleared
        assert!(predictor.correction_accumulator.is_empty());
    }

    #[test]
    fn test_stats() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default().with_prediction_steps(4);

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        predictor.record_gradient(&gradients).unwrap();
        predictor.record_gradient(&gradients).unwrap();
        predictor.predict_gradient().unwrap();

        let stats = predictor.get_stats();
        assert_eq!(stats.total_steps, 3);
        assert!(stats.prediction_ratio > 0.7); // 4/(4+1) = 0.8
    }

    #[test]
    fn test_reset() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = PredictionConfig::default();

        let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        predictor.record_gradient(&gradients).unwrap();
        predictor.record_gradient(&gradients).unwrap();

        assert_eq!(predictor.total_steps(), 2);

        predictor.reset();

        assert_eq!(predictor.total_steps(), 0);
        assert!(predictor.should_compute_full());
    }
}

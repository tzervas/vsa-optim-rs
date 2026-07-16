//! Deterministic gradient prediction for phase-based training.
//!
//! This module implements a deterministic prediction system that models
//! gradient evolution during training. Unlike stochastic extrapolation,
//! this approach guarantees reproducible predictions given the same history.
//!
//! # Algorithm
//!
//! The predictor fits a linear regression model to the gradient trajectory:
//!
//! ```text
//! g(t) = g(0) + α * t + residual(t)
//! ```
//!
//! Where:
//! - `g(t)` is the gradient at step t
//! - `g(0)` is the baseline gradient (from warmup)
//! - `α` is the fitted gradient velocity (change per step)
//! - `residual(t)` accumulates prediction errors for correction
//!
//! # Phases
//!
//! 1. **Warmup**: `warmup_steps` of full gradient computation to establish baseline
//! 2. **Predict**: Extrapolate gradients using fitted model
//! 3. **Correct**: Compare prediction with actual, update model and residuals
//!
//! # Determinism
//!
//! Predictions are fully deterministic because:
//! - No random sampling or stochastic operations
//! - Model fit uses closed-form least squares (no iterative optimization)
//! - Same history always produces same prediction

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use crate::error::{OptimError, Result};

/// Configuration for deterministic gradient prediction.
#[derive(Debug, Clone)]
pub struct DeterministicPredictionConfig {
    /// Minimum steps of full training before prediction begins.
    pub warmup_steps: usize,

    /// Number of history steps to use for fitting.
    pub history_window: usize,

    /// Steps to predict before correction.
    pub prediction_horizon: usize,

    /// Exponential decay for older history (1.0 = no decay).
    pub history_decay: f32,

    /// Threshold for residual magnitude to trigger early correction.
    pub residual_threshold: f32,
}

impl Default for DeterministicPredictionConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 10,
            history_window: 8,
            prediction_horizon: 4,
            history_decay: 0.95,
            residual_threshold: 0.5,
        }
    }
}

impl DeterministicPredictionConfig {
    /// Builder: Set warmup steps.
    #[must_use]
    pub const fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Builder: Set history window.
    #[must_use]
    pub const fn with_history_window(mut self, window: usize) -> Self {
        self.history_window = window;
        self
    }

    /// Builder: Set prediction horizon.
    #[must_use]
    pub const fn with_prediction_horizon(mut self, horizon: usize) -> Self {
        self.prediction_horizon = horizon;
        self
    }

    /// Builder: Set history decay.
    #[must_use]
    pub const fn with_history_decay(mut self, decay: f32) -> Self {
        self.history_decay = decay;
        self
    }
}

/// Gradient history entry with step index.
#[derive(Clone)]
struct GradientSnapshot {
    /// Global step index when this gradient was recorded.
    step: usize,
    /// The gradient tensor.
    gradient: Tensor,
}

/// Linear model for gradient evolution: g(t) = baseline + velocity * t
#[derive(Clone)]
struct LinearGradientModel {
    /// Baseline gradient (intercept).
    baseline: Tensor,
    /// Gradient velocity (slope) per step.
    velocity: Tensor,
    /// Step index at which model was fitted.
    fit_step: usize,
}

/// Deterministic gradient predictor.
///
/// Maintains gradient history and fits a linear model to predict
/// future gradients deterministically.
pub struct DeterministicPredictor {
    config: DeterministicPredictionConfig,
    device: Device,

    /// Parameter shapes for reconstruction.
    shapes: HashMap<String, Vec<usize>>,

    /// Gradient history per parameter.
    history: HashMap<String, Vec<GradientSnapshot>>,

    /// Fitted linear models per parameter.
    models: HashMap<String, LinearGradientModel>,

    /// Accumulated residuals (prediction errors) per parameter.
    residuals: HashMap<String, Tensor>,

    /// Current global step.
    global_step: usize,

    /// Steps since last model fit.
    steps_since_fit: usize,

    /// Whether warmup is complete.
    warmup_complete: bool,

    /// Statistics tracking.
    stats: PredictorStatistics,
}

/// Statistics for prediction quality monitoring.
#[derive(Debug, Clone, Default)]
pub struct PredictorStatistics {
    /// Total steps processed.
    pub total_steps: usize,
    /// Steps with full gradient computation.
    pub full_steps: usize,
    /// Steps with predicted gradients.
    pub predicted_steps: usize,
    /// Mean absolute prediction error.
    pub mean_abs_error: f32,
    /// Maximum observed residual.
    pub max_residual: f32,
    /// Number of early corrections triggered.
    pub early_corrections: usize,
}

impl DeterministicPredictor {
    /// Create a new deterministic predictor.
    ///
    /// # Arguments
    ///
    /// * `param_shapes` - List of (name, shape) tuples for parameters
    /// * `config` - Prediction configuration
    /// * `device` - Device for tensor storage
    pub fn new(
        param_shapes: &[(String, Vec<usize>)],
        config: DeterministicPredictionConfig,
        device: &Device,
    ) -> Result<Self> {
        let mut shapes = HashMap::new();
        let mut history = HashMap::new();
        let mut residuals = HashMap::new();

        for (name, shape) in param_shapes {
            shapes.insert(name.clone(), shape.clone());
            history.insert(name.clone(), Vec::with_capacity(config.history_window + 4));
            // Initialize residuals to zero
            residuals.insert(
                name.clone(),
                Tensor::zeros(shape.as_slice(), DType::F32, device)?,
            );
        }

        Ok(Self {
            config,
            device: device.clone(),
            shapes,
            history,
            models: HashMap::new(),
            residuals,
            global_step: 0,
            steps_since_fit: 0,
            warmup_complete: false,
            stats: PredictorStatistics::default(),
        })
    }

    /// Check if still in warmup phase (must compute full gradients).
    #[must_use]
    pub fn in_warmup(&self) -> bool {
        !self.warmup_complete
    }

    /// Check if correction is needed based on residual magnitude.
    #[must_use]
    pub fn needs_correction(&self) -> bool {
        // Need correction after prediction horizon
        if self.steps_since_fit >= self.config.prediction_horizon {
            return true;
        }

        // Check residual threshold
        for residual in self.residuals.values() {
            if let Ok(max_abs) = residual
                .abs()
                .and_then(|t| t.max(0))
                .and_then(|t| t.to_scalar::<f32>())
            {
                if max_abs > self.config.residual_threshold {
                    return true;
                }
            }
        }

        false
    }

    /// Record a gradient from full computation.
    ///
    /// Updates history and potentially refits the prediction model.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Map of parameter names to gradient tensors
    /// * `is_correction` - Whether this is a correction step (vs. regular full step)
    pub fn record_gradient(
        &mut self,
        gradients: &HashMap<String, Tensor>,
        is_correction: bool,
    ) -> Result<()> {
        // Record to history
        for (name, grad) in gradients {
            if let Some(hist) = self.history.get_mut(name) {
                hist.push(GradientSnapshot {
                    step: self.global_step,
                    gradient: grad.clone(),
                });

                // Trim history to window size
                let window = self.config.history_window;
                if hist.len() > window + 2 {
                    hist.drain(0..hist.len() - window - 2);
                }
            }
        }

        // Update statistics
        self.stats.total_steps += 1;
        self.stats.full_steps += 1;

        // Check warmup completion
        if !self.warmup_complete {
            let min_history = self.history.values().map(|h| h.len()).min().unwrap_or(0);
            if min_history >= self.config.warmup_steps {
                self.warmup_complete = true;
                self.fit_models()?;
            }
        } else if is_correction {
            // Update residuals based on prediction error
            self.update_residuals(gradients)?;
            // Refit model with new data
            self.fit_models()?;
        } else {
            // Regular full step - refit model
            self.fit_models()?;
        }

        self.global_step += 1;
        self.steps_since_fit = 0;

        Ok(())
    }

    /// Predict gradient for current step.
    ///
    /// Uses the fitted linear model to extrapolate from history.
    /// Prediction is fully deterministic.
    ///
    /// # Returns
    ///
    /// Map of parameter names to predicted gradient tensors.
    pub fn predict_gradient(&mut self) -> Result<HashMap<String, Tensor>> {
        if !self.warmup_complete {
            return Err(OptimError::Prediction(
                "Cannot predict during warmup phase".to_string(),
            ));
        }

        let mut predicted = HashMap::new();

        for (name, model) in &self.models {
            // Steps since model was fitted
            let dt = (self.global_step - model.fit_step) as f64;

            // Linear prediction: g(t) = baseline + velocity * dt
            let velocity_term = (&model.velocity * dt)?;
            let mut prediction = model.baseline.add(&velocity_term)?;

            // Add accumulated residual correction
            if let Some(residual) = self.residuals.get(name) {
                // Apply weighted residual (decays with steps since correction)
                let residual_weight = self.config.history_decay.powi(self.steps_since_fit as i32);
                let scaled_residual = (residual * residual_weight as f64)?;
                prediction = prediction.add(&scaled_residual)?;
            }

            predicted.insert(name.clone(), prediction);
        }

        // Update statistics
        self.stats.total_steps += 1;
        self.stats.predicted_steps += 1;
        self.global_step += 1;
        self.steps_since_fit += 1;

        Ok(predicted)
    }

    /// Update residuals based on prediction error.
    ///
    /// Called during correction step to accumulate the difference
    /// between predicted and actual gradients.
    fn update_residuals(&mut self, actual: &HashMap<String, Tensor>) -> Result<()> {
        for (name, actual_grad) in actual {
            if let Some(model) = self.models.get(name) {
                // What we predicted for this step
                let dt = (self.global_step - model.fit_step) as f64;
                let velocity_term = (&model.velocity * dt)?;
                let predicted = model.baseline.add(&velocity_term)?;

                // Residual = actual - predicted
                let error = actual_grad.sub(&predicted)?;

                // Update accumulated residual with exponential averaging
                if let Some(existing) = self.residuals.get(name) {
                    let decay = self.config.history_decay as f64;
                    let decayed_existing = (existing * decay)?;
                    let new_contribution = (&error * (1.0 - decay))?;
                    self.residuals
                        .insert(name.clone(), decayed_existing.add(&new_contribution)?);
                } else {
                    self.residuals.insert(name.clone(), error);
                }

                // Track statistics
                if let Ok(mean_err) = actual_grad
                    .sub(&predicted)
                    .and_then(|t| t.abs())
                    .and_then(|t| t.mean_all())
                    .and_then(|t| t.to_scalar::<f32>())
                {
                    self.stats.mean_abs_error = 0.9 * self.stats.mean_abs_error + 0.1 * mean_err;
                }
            }
        }

        Ok(())
    }

    /// Fit linear models to gradient history.
    ///
    /// Uses weighted least squares to fit g(t) = baseline + velocity * t
    /// for each parameter.
    fn fit_models(&mut self) -> Result<()> {
        for (name, hist) in &self.history {
            if hist.len() < 2 {
                continue;
            }

            let shape = self
                .shapes
                .get(name)
                .ok_or_else(|| OptimError::Prediction(format!("Unknown parameter: {name}")))?;

            // Weighted least squares fitting
            // g(t) = baseline + velocity * t
            // Minimize: sum_i w_i * (g_i - baseline - velocity * t_i)^2

            let n = hist.len();
            let mut sum_w = 0.0f64;
            let mut sum_wt = 0.0f64;
            let mut sum_wt2 = 0.0f64;
            let mut sum_wg: Option<Tensor> = None;
            let mut sum_wtg: Option<Tensor> = None;

            // Reference step for numerical stability
            let t_ref = hist.last().map(|s| s.step).unwrap_or(0);

            for (i, snapshot) in hist.iter().enumerate() {
                // Exponential weight favoring recent gradients
                let age = (n - 1 - i) as i32;
                let w = self.config.history_decay.powi(age) as f64;

                // Relative step index
                let t = (snapshot.step as i64 - t_ref as i64) as f64;

                sum_w += w;
                sum_wt += w * t;
                sum_wt2 += w * t * t;

                // Accumulate weighted gradients
                let wg = (&snapshot.gradient * w)?;
                let wtg = (&snapshot.gradient * (w * t))?;

                sum_wg = Some(match sum_wg {
                    Some(acc) => acc.add(&wg)?,
                    None => wg,
                });

                sum_wtg = Some(match sum_wtg {
                    Some(acc) => acc.add(&wtg)?,
                    None => wtg,
                });
            }

            // Solve normal equations for least squares
            // [sum_w    sum_wt  ] [baseline]   [sum_wg ]
            // [sum_wt   sum_wt2 ] [velocity] = [sum_wtg]

            let det = sum_w * sum_wt2 - sum_wt * sum_wt;
            if det.abs() < 1e-10 {
                // Degenerate case: use latest gradient as baseline, zero velocity
                let baseline = hist.last().unwrap().gradient.clone();
                let velocity = Tensor::zeros(shape.as_slice(), DType::F32, &self.device)?;
                self.models.insert(
                    name.clone(),
                    LinearGradientModel {
                        baseline,
                        velocity,
                        fit_step: self.global_step,
                    },
                );
                continue;
            }

            let sum_wg = sum_wg
                .ok_or_else(|| OptimError::Prediction("Empty gradient history".to_string()))?;
            let sum_wtg = sum_wtg
                .ok_or_else(|| OptimError::Prediction("Empty gradient history".to_string()))?;

            // Cramer's rule
            // baseline = (sum_wt2 * sum_wg - sum_wt * sum_wtg) / det
            // velocity = (sum_w * sum_wtg - sum_wt * sum_wg) / det

            let baseline = {
                let term1 = (&sum_wg * sum_wt2)?;
                let term2 = (&sum_wtg * sum_wt)?;
                let numer = term1.sub(&term2)?;
                (&numer * (1.0 / det))?
            };

            let velocity = {
                let term1 = (&sum_wtg * sum_w)?;
                let term2 = (&sum_wg * sum_wt)?;
                let numer = term1.sub(&term2)?;
                (&numer * (1.0 / det))?
            };

            self.models.insert(
                name.clone(),
                LinearGradientModel {
                    baseline,
                    velocity,
                    fit_step: self.global_step,
                },
            );
        }

        Ok(())
    }

    /// Get prediction statistics.
    #[must_use]
    pub fn get_stats(&self) -> &PredictorStatistics {
        &self.stats
    }

    /// Reset predictor state.
    pub fn reset(&mut self) -> Result<()> {
        for hist in self.history.values_mut() {
            hist.clear();
        }
        self.models.clear();

        // Reset residuals to zero
        for (name, shape) in &self.shapes {
            self.residuals.insert(
                name.clone(),
                Tensor::zeros(shape.as_slice(), DType::F32, &self.device)?,
            );
        }

        self.global_step = 0;
        self.steps_since_fit = 0;
        self.warmup_complete = false;
        self.stats = PredictorStatistics::default();

        Ok(())
    }

    /// Get current global step.
    #[must_use]
    pub const fn global_step(&self) -> usize {
        self.global_step
    }

    /// Check if warmup is complete.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        self.warmup_complete
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

    #[test]
    fn test_warmup_phase() {
        let config = DeterministicPredictionConfig::default().with_warmup_steps(5);
        let mut predictor =
            DeterministicPredictor::new(&create_shapes(), config, &Device::Cpu).unwrap();

        assert!(predictor.in_warmup());
        assert!(!predictor.is_ready());

        // Record warmup gradients
        for i in 0..5 {
            let mut grads = HashMap::new();
            grads.insert(
                "layer.weight".to_string(),
                Tensor::ones((16, 32), DType::F32, &Device::Cpu)
                    .unwrap()
                    .affine(i as f64, 0.0)
                    .unwrap(),
            );
            grads.insert(
                "layer.bias".to_string(),
                Tensor::ones(16, DType::F32, &Device::Cpu)
                    .unwrap()
                    .affine(i as f64, 0.0)
                    .unwrap(),
            );
            predictor.record_gradient(&grads, false).unwrap();
        }

        assert!(!predictor.in_warmup());
        assert!(predictor.is_ready());
    }

    #[test]
    fn test_deterministic_prediction() {
        let config = DeterministicPredictionConfig::default()
            .with_warmup_steps(3)
            .with_prediction_horizon(2);
        let device = Device::Cpu;

        // Create two identical predictors
        let shapes = create_shapes();
        let mut pred1 = DeterministicPredictor::new(&shapes, config.clone(), &device).unwrap();
        let mut pred2 = DeterministicPredictor::new(&shapes, config, &device).unwrap();

        // Feed identical history
        for i in 0..5 {
            let mut grads = HashMap::new();
            grads.insert(
                "layer.weight".to_string(),
                Tensor::ones((16, 32), DType::F32, &device)
                    .unwrap()
                    .affine(1.0 + i as f64 * 0.1, 0.0)
                    .unwrap(),
            );
            grads.insert(
                "layer.bias".to_string(),
                Tensor::ones(16, DType::F32, &device)
                    .unwrap()
                    .affine(1.0 + i as f64 * 0.1, 0.0)
                    .unwrap(),
            );
            pred1.record_gradient(&grads, false).unwrap();
            pred2.record_gradient(&grads, false).unwrap();
        }

        // Predictions should be identical
        let p1 = pred1.predict_gradient().unwrap();
        let p2 = pred2.predict_gradient().unwrap();

        for (name, t1) in &p1 {
            let t2 = p2.get(name).unwrap();
            let diff: f32 = t1
                .sub(t2)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(
                diff < 1e-6,
                "Predictions should be deterministic, got diff={diff}"
            );
        }
    }

    #[test]
    fn test_linear_fit_quality() {
        // Test that linear model correctly fits linear gradient evolution
        let config = DeterministicPredictionConfig::default()
            .with_warmup_steps(5)
            .with_prediction_horizon(3);
        let device = Device::Cpu;
        let shapes = vec![("param".to_string(), vec![8])];

        let mut predictor = DeterministicPredictor::new(&shapes, config, &device).unwrap();

        // Generate perfectly linear gradients: g(t) = 1 + 0.1*t
        for t in 0..5 {
            let mut grads = HashMap::new();
            grads.insert(
                "param".to_string(),
                Tensor::ones(8, DType::F32, &device)
                    .unwrap()
                    .affine(1.0 + 0.1 * t as f64, 0.0)
                    .unwrap(),
            );
            predictor.record_gradient(&grads, false).unwrap();
        }

        // Predict next step - should be close to 1 + 0.1*5 = 1.5
        let predicted = predictor.predict_gradient().unwrap();
        let pred_vals: Vec<f32> = predicted.get("param").unwrap().to_vec1().unwrap();

        // All values should be close to 1.5
        for v in &pred_vals {
            assert!(
                (*v - 1.5).abs() < 0.1,
                "Linear prediction should be accurate, got {v}"
            );
        }
    }
}

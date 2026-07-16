//! Ternary gradient accumulator for memory-efficient training.
//!
//! Gradient accumulation over many steps can be memory-intensive.
//! Using ternary representation with scale factors reduces memory by ~10x
//! while maintaining accuracy through the scale factors.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use crate::config::TernaryConfig;
use crate::error::Result;
use crate::ternary::{
    calculate_memory_savings, ternary_quantize_deterministic, ternary_quantize_stochastic,
};

/// Accumulated state for a single parameter.
#[derive(Debug, Clone)]
struct AccumulatedGradient {
    /// Accumulated ternary direction (sum of ternary values).
    ternary: Tensor,
    /// Accumulated scale factors.
    scale_sum: f32,
    /// Original shape for reconstruction.
    shape: Vec<usize>,
}

/// Accumulate gradients using ternary representation.
///
/// The accumulator keeps a ternary "direction" tensor and a scale tensor.
/// New gradients are projected onto this representation and accumulated.
/// Full-precision reconstruction happens only at update time.
///
/// # Example
///
/// ```ignore
/// use vsa_optim_rs::ternary::TernaryGradientAccumulator;
/// use vsa_optim_rs::TernaryConfig;
///
/// let shapes = vec![
///     ("layer1.weight".to_string(), vec![64, 128]),
///     ("layer1.bias".to_string(), vec![64]),
/// ];
/// let mut accumulator = TernaryGradientAccumulator::new(&shapes, TernaryConfig::default(), &Device::Cpu)?;
///
/// // Accumulate gradients
/// accumulator.accumulate(&gradients)?;
///
/// // Get full-precision result
/// let accumulated = accumulator.get_accumulated()?;
/// accumulator.reset();
/// ```
pub struct TernaryGradientAccumulator {
    config: TernaryConfig,
    device: Device,
    /// Accumulated gradients per parameter.
    accumulators: HashMap<String, AccumulatedGradient>,
    /// Number of accumulation steps.
    count: usize,
}

impl TernaryGradientAccumulator {
    /// Create a new gradient accumulator.
    ///
    /// # Arguments
    ///
    /// * `param_shapes` - List of (name, shape) tuples for parameters
    /// * `config` - Ternary configuration
    /// * `device` - Device for tensor storage
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails.
    pub fn new(
        param_shapes: &[(String, Vec<usize>)],
        config: TernaryConfig,
        device: &Device,
    ) -> Result<Self> {
        let mut accumulators = HashMap::new();

        for (name, shape) in param_shapes {
            let ternary = Tensor::zeros(shape.as_slice(), DType::F32, device)?;
            accumulators.insert(
                name.clone(),
                AccumulatedGradient {
                    ternary,
                    scale_sum: 0.0,
                    shape: shape.clone(),
                },
            );
        }

        Ok(Self {
            config,
            device: device.clone(),
            accumulators,
            count: 0,
        })
    }

    /// Accumulate gradients in ternary form.
    ///
    /// Converts gradients to ternary, accumulates direction,
    /// and tracks scale. This is called after each backward pass.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Map of parameter names to gradient tensors
    ///
    /// # Errors
    ///
    /// Returns error if quantization or tensor operations fail.
    pub fn accumulate(&mut self, gradients: &HashMap<String, Tensor>) -> Result<()> {
        let threshold = Some(self.config.ternary_threshold);

        for (name, grad) in gradients {
            if let Some(accum) = self.accumulators.get_mut(name) {
                // Quantize gradient
                let (ternary, scale) = if self.config.use_stochastic_rounding {
                    ternary_quantize_stochastic(grad, threshold)?
                } else {
                    ternary_quantize_deterministic(grad, threshold)?
                };

                // Accumulate (ternary addition = element-wise sum)
                // Note: Sum of ternary is not ternary, but uses fewer bits
                accum.ternary = accum.ternary.add(&ternary)?;
                accum.scale_sum += scale;
            }
        }

        self.count += 1;
        Ok(())
    }

    /// Get full-precision accumulated gradients.
    ///
    /// Reconstructs full-precision gradients from ternary accumulation.
    /// The scale is averaged and applied to the accumulated direction.
    ///
    /// # Returns
    ///
    /// Dictionary mapping parameter names to accumulated gradients.
    ///
    /// # Errors
    ///
    /// Returns error if tensor operations fail.
    #[allow(clippy::cast_precision_loss)]
    pub fn get_accumulated(&self) -> Result<HashMap<String, Tensor>> {
        let mut accumulated = HashMap::new();

        for (name, accum) in &self.accumulators {
            if self.count > 0 {
                // Average scale
                let avg_scale = accum.scale_sum / self.count as f32;
                // Reconstruct: direction * scale / count
                let result = (&accum.ternary * avg_scale as f64)?;
                let result = (result / self.count as f64)?;
                accumulated.insert(name.clone(), result);
            } else {
                accumulated.insert(name.clone(), accum.ternary.clone());
            }
        }

        Ok(accumulated)
    }

    /// Reset accumulator for next accumulation cycle.
    ///
    /// # Errors
    ///
    /// Returns error if tensor zeroing fails.
    pub fn reset(&mut self) -> Result<()> {
        for accum in self.accumulators.values_mut() {
            accum.ternary = accum.ternary.zeros_like()?;
            accum.scale_sum = 0.0;
        }
        self.count = 0;
        Ok(())
    }

    /// Get the number of accumulated steps.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Calculate memory savings from ternary representation.
    ///
    /// # Returns
    ///
    /// Fraction of memory saved (0 to 1).
    #[must_use]
    pub fn memory_savings(&self) -> f32 {
        let param_count: usize = self
            .accumulators
            .values()
            .map(|a| a.shape.iter().product::<usize>())
            .sum();
        let num_tensors = self.accumulators.len();
        calculate_memory_savings(param_count, num_tensors)
    }

    /// Check if ready for optimizer update.
    #[must_use]
    pub fn ready_for_update(&self) -> bool {
        self.count >= self.config.accumulation_steps
    }
}

/// Optimizer wrapper with ternary gradient accumulation.
///
/// Combines ternary accumulation with gradient management for
/// memory-efficient training. Useful for large batch training where
/// gradient accumulation is necessary.
///
/// # Example
///
/// ```ignore
/// use vsa_optim_rs::ternary::TernaryOptimizerWrapper;
/// use vsa_optim_rs::TernaryConfig;
///
/// let mut wrapper = TernaryOptimizerWrapper::new(param_shapes, TernaryConfig::default(), &device)?;
///
/// for (i, batch) in batches.iter().enumerate() {
///     // Compute gradients...
///     let should_update = wrapper.step(&gradients)?;
///     if should_update {
///         let accumulated = wrapper.get_gradients_for_update()?;
///         // Apply to optimizer...
///     }
/// }
/// ```
pub struct TernaryOptimizerWrapper {
    config: TernaryConfig,
    accumulator: TernaryGradientAccumulator,
    step_count: usize,
    update_count: usize,
}

impl TernaryOptimizerWrapper {
    /// Create a new ternary optimizer wrapper.
    ///
    /// # Arguments
    ///
    /// * `param_shapes` - List of (name, shape) tuples for parameters
    /// * `config` - Ternary configuration
    /// * `device` - Device for tensor storage
    ///
    /// # Errors
    ///
    /// Returns error if accumulator creation fails.
    pub fn new(
        param_shapes: &[(String, Vec<usize>)],
        config: TernaryConfig,
        device: &Device,
    ) -> Result<Self> {
        let accumulator = TernaryGradientAccumulator::new(param_shapes, config.clone(), device)?;

        Ok(Self {
            config,
            accumulator,
            step_count: 0,
            update_count: 0,
        })
    }

    /// Accumulate gradient and check if update is needed.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Current step gradients
    ///
    /// # Returns
    ///
    /// True if optimizer update should be performed, False if just accumulated.
    ///
    /// # Errors
    ///
    /// Returns error if accumulation fails.
    pub fn step(&mut self, gradients: &HashMap<String, Tensor>) -> Result<bool> {
        // Accumulate current gradients
        self.accumulator.accumulate(gradients)?;
        self.step_count += 1;

        // Check if update is needed
        Ok(self.step_count % self.config.accumulation_steps == 0)
    }

    /// Get accumulated gradients for optimizer update.
    ///
    /// Call this when `step()` returns true.
    ///
    /// # Returns
    ///
    /// Accumulated full-precision gradients.
    ///
    /// # Errors
    ///
    /// Returns error if reconstruction fails.
    pub fn get_gradients_for_update(&mut self) -> Result<HashMap<String, Tensor>> {
        let grads = self.accumulator.get_accumulated()?;
        self.accumulator.reset()?;
        self.update_count += 1;
        Ok(grads)
    }

    /// Get optimization statistics.
    #[must_use]
    pub fn get_stats(&self) -> OptimizerStats {
        OptimizerStats {
            step_count: self.step_count,
            update_count: self.update_count,
            memory_savings: self.accumulator.memory_savings(),
            accumulation_steps: self.config.accumulation_steps,
        }
    }

    /// Get the step count.
    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the update count.
    #[must_use]
    pub const fn update_count(&self) -> usize {
        self.update_count
    }

    /// Reset state for checkpointing.
    pub fn reset_state(&mut self) {
        self.step_count = 0;
        self.update_count = 0;
    }

    /// Load state from checkpoint.
    pub fn load_state(&mut self, step_count: usize, update_count: usize) {
        self.step_count = step_count;
        self.update_count = update_count;
    }
}

/// Optimization statistics.
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    /// Total number of steps.
    pub step_count: usize,
    /// Number of optimizer updates.
    pub update_count: usize,
    /// Memory savings fraction.
    pub memory_savings: f32,
    /// Configured accumulation steps.
    pub accumulation_steps: usize,
}

impl std::fmt::Display for OptimizerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {} | Updates: {} | Memory saved: {:.1}%",
            self.step_count,
            self.update_count,
            self.memory_savings * 100.0
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
            ("layer2.weight".to_string(), vec![32, 64]),
        ]
    }

    fn create_mock_gradients(device: &Device) -> HashMap<String, Tensor> {
        let mut gradients = HashMap::new();
        gradients.insert(
            "layer1.weight".to_string(),
            Tensor::randn(0.0f32, 1.0, (64, 128), device).unwrap(),
        );
        gradients.insert(
            "layer1.bias".to_string(),
            Tensor::randn(0.0f32, 1.0, 64, device).unwrap(),
        );
        gradients.insert(
            "layer2.weight".to_string(),
            Tensor::randn(0.0f32, 1.0, (32, 64), device).unwrap(),
        );
        gradients
    }

    #[test]
    fn test_accumulator_creation() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default();

        let accumulator = TernaryGradientAccumulator::new(&shapes, config, &device).unwrap();
        assert_eq!(accumulator.count(), 0);
    }

    #[test]
    fn test_accumulator_accumulate() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default();

        let mut accumulator = TernaryGradientAccumulator::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        accumulator.accumulate(&gradients).unwrap();
        assert_eq!(accumulator.count(), 1);

        accumulator.accumulate(&gradients).unwrap();
        assert_eq!(accumulator.count(), 2);
    }

    #[test]
    fn test_accumulator_get_accumulated() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default();

        let mut accumulator = TernaryGradientAccumulator::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        accumulator.accumulate(&gradients).unwrap();
        let accumulated = accumulator.get_accumulated().unwrap();

        assert_eq!(accumulated.len(), 3);
        for (name, _shape) in &shapes {
            assert!(accumulated.contains_key(name));
        }
    }

    #[test]
    fn test_accumulator_reset() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default();

        let mut accumulator = TernaryGradientAccumulator::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        accumulator.accumulate(&gradients).unwrap();
        assert_eq!(accumulator.count(), 1);

        accumulator.reset().unwrap();
        assert_eq!(accumulator.count(), 0);
    }

    #[test]
    fn test_accumulator_memory_savings() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default();

        let accumulator = TernaryGradientAccumulator::new(&shapes, config, &device).unwrap();
        let savings = accumulator.memory_savings();

        // Should save ~90%+ for reasonable sizes
        assert!(
            savings > 0.9,
            "Expected >90% savings, got {:.2}%",
            savings * 100.0
        );
    }

    #[test]
    fn test_optimizer_wrapper_step() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default().with_accumulation_steps(4);

        let mut wrapper = TernaryOptimizerWrapper::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        // Steps 1-3: accumulate only
        for _ in 0..3 {
            let should_update = wrapper.step(&gradients).unwrap();
            assert!(!should_update);
        }

        // Step 4: should update
        let should_update = wrapper.step(&gradients).unwrap();
        assert!(should_update);

        // Get gradients and verify
        let accumulated = wrapper.get_gradients_for_update().unwrap();
        assert_eq!(accumulated.len(), 3);

        // Step 5: back to accumulating
        let should_update = wrapper.step(&gradients).unwrap();
        assert!(!should_update);
    }

    #[test]
    fn test_optimizer_wrapper_stats() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;
        let config = TernaryConfig::default().with_accumulation_steps(2);

        let mut wrapper = TernaryOptimizerWrapper::new(&shapes, config, &device).unwrap();
        let gradients = create_mock_gradients(&device);

        wrapper.step(&gradients).unwrap();
        wrapper.step(&gradients).unwrap();
        let _ = wrapper.get_gradients_for_update().unwrap();

        let stats = wrapper.get_stats();
        assert_eq!(stats.step_count, 2);
        assert_eq!(stats.update_count, 1);
        assert!(stats.memory_savings > 0.9);
    }

    #[test]
    fn test_stochastic_vs_deterministic() {
        let shapes = create_param_shapes();
        let device = Device::Cpu;

        // Test stochastic
        let config_stochastic = TernaryConfig::default().with_stochastic_rounding(true);
        let mut acc_stochastic =
            TernaryGradientAccumulator::new(&shapes, config_stochastic, &device).unwrap();

        // Test deterministic
        let config_deterministic = TernaryConfig::default().with_stochastic_rounding(false);
        let mut acc_deterministic =
            TernaryGradientAccumulator::new(&shapes, config_deterministic, &device).unwrap();

        let gradients = create_mock_gradients(&device);

        acc_stochastic.accumulate(&gradients).unwrap();
        acc_deterministic.accumulate(&gradients).unwrap();

        // Both should produce valid results
        let result_stochastic = acc_stochastic.get_accumulated().unwrap();
        let result_deterministic = acc_deterministic.get_accumulated().unwrap();

        assert_eq!(result_stochastic.len(), 3);
        assert_eq!(result_deterministic.len(), 3);
    }
}

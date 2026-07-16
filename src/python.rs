//! Python bindings for vsa-optim-rs using PyO3.
//!
//! This module provides Python-accessible wrappers for the main training
//! optimization components, enabling seamless integration with PyTorch
//! training pipelines.
//!
//! # Usage from Python
//!
//! ```python
//! from vsa_optim_rs import PhaseTrainer, PhaseConfig
//!
//! # Create configuration
//! config = PhaseConfig(full_steps=10, predict_steps=40)
//!
//! # Create trainer
//! shapes = [("layer.weight", [64, 128]), ("layer.bias", [64])]
//! trainer = PhaseTrainer(shapes, config)
//!
//! # Training loop
//! for step in range(total_steps):
//!     step_info = trainer.begin_step()
//!
//!     if trainer.should_compute_full():
//!         # Compute full gradients
//!         trainer.record_full_gradients(gradients)
//!     else:
//!         predicted = trainer.get_predicted_gradients()
//!
//!     trainer.end_step(loss_value)
//! ```

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use candle_core::{Device, Tensor};

use crate::config::{PhaseConfig, PredictionConfig, TernaryConfig, VSAConfig};
use crate::phase::{PhaseTrainer as RustPhaseTrainer, TrainingPhase};
use crate::prediction::GradientPredictor as RustGradientPredictor;
use crate::ternary::TernaryGradientAccumulator as RustTernaryAccumulator;
use crate::vsa::VSAGradientCompressor as RustVSACompressor;

fn resolve_device(use_cuda: Option<bool>) -> PyResult<Device> {
    let force_cpu = std::env::var("VSA_OPTIM_FORCE_CPU")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if force_cpu {
        eprintln!(
            "vsa-optim-rs: CPU mode forced via VSA_OPTIM_FORCE_CPU=1. GPU is the intended default."
        );
        return Ok(Device::Cpu);
    }

    if use_cuda == Some(false) {
        eprintln!("vsa-optim-rs: CPU mode selected; GPU is the intended default.");
        return Ok(Device::Cpu);
    }

    let cuda_device = std::env::var("VSA_OPTIM_CUDA_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    match Device::cuda_if_available(cuda_device) {
        Ok(device @ Device::Cuda(_)) => Ok(device),
        Ok(_) => {
            eprintln!("vsa-optim-rs: CUDA not available; falling back to CPU. This is a compatibility path only.");
            Ok(Device::Cpu)
        }
        Err(err) => {
            eprintln!("vsa-optim-rs: CUDA init failed ({err}); falling back to CPU. This is a compatibility path only.");
            Ok(Device::Cpu)
        }
    }
}

// ============================================================================
// Configuration Wrappers
// ============================================================================

/// Python-accessible VSA configuration.
#[pyclass(name = "VSAConfig")]
#[derive(Clone)]
pub struct PyVSAConfig {
    inner: VSAConfig,
}

#[pymethods]
impl PyVSAConfig {
    /// Create a new VSA configuration.
    ///
    /// Args:
    ///     dimension: Hypervector dimension (default: 8192)
    ///     compression_ratio: Target compression ratio, 0.1 = 90% compression (default: 0.1)
    ///     use_ternary: Whether to use ternary quantization (default: True)
    ///     seed: Random seed for reproducibility (default: 42)
    #[new]
    #[pyo3(signature = (dimension=8192, compression_ratio=0.1, use_ternary=true, seed=42))]
    fn new(dimension: usize, compression_ratio: f32, use_ternary: bool, seed: u64) -> Self {
        Self {
            inner: VSAConfig {
                dimension,
                compression_ratio,
                use_ternary,
                seed,
            },
        }
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio
    }

    #[getter]
    fn use_ternary(&self) -> bool {
        self.inner.use_ternary
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "VSAConfig(dimension={}, compression_ratio={}, use_ternary={}, seed={})",
            self.inner.dimension,
            self.inner.compression_ratio,
            self.inner.use_ternary,
            self.inner.seed
        )
    }
}

/// Python-accessible ternary configuration.
#[pyclass(name = "TernaryConfig")]
#[derive(Clone)]
pub struct PyTernaryConfig {
    inner: TernaryConfig,
}

#[pymethods]
impl PyTernaryConfig {
    /// Create a new ternary configuration.
    ///
    /// Args:
    ///     accumulation_steps: Steps before optimizer update (default: 8)
    ///     ternary_threshold: Quantization threshold relative to mean abs (default: 0.5)
    ///     scale_learning_rate: Learning rate for scale parameters (default: 0.01)
    ///     use_stochastic_rounding: Use unbiased stochastic rounding (default: True)
    #[new]
    #[pyo3(signature = (accumulation_steps=8, ternary_threshold=0.5, scale_learning_rate=0.01, use_stochastic_rounding=true))]
    fn new(
        accumulation_steps: usize,
        ternary_threshold: f32,
        scale_learning_rate: f32,
        use_stochastic_rounding: bool,
    ) -> Self {
        Self {
            inner: TernaryConfig {
                accumulation_steps,
                ternary_threshold,
                scale_learning_rate,
                use_stochastic_rounding,
            },
        }
    }

    #[getter]
    fn accumulation_steps(&self) -> usize {
        self.inner.accumulation_steps
    }

    #[getter]
    fn ternary_threshold(&self) -> f32 {
        self.inner.ternary_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "TernaryConfig(accumulation_steps={}, ternary_threshold={}, stochastic={})",
            self.inner.accumulation_steps,
            self.inner.ternary_threshold,
            self.inner.use_stochastic_rounding
        )
    }
}

/// Python-accessible prediction configuration.
#[pyclass(name = "PredictionConfig")]
#[derive(Clone)]
pub struct PyPredictionConfig {
    inner: PredictionConfig,
}

#[pymethods]
impl PyPredictionConfig {
    /// Create a new prediction configuration.
    ///
    /// Args:
    ///     history_size: Number of past gradients to keep (default: 5)
    ///     prediction_steps: Steps to predict before full compute (default: 4)
    ///     momentum: Momentum factor for extrapolation (default: 0.9)
    ///     correction_weight: Weight for correction terms (default: 0.5)
    ///     min_correlation: Minimum correlation to use prediction (default: 0.8)
    #[new]
    #[pyo3(signature = (history_size=5, prediction_steps=4, momentum=0.9, correction_weight=0.5, min_correlation=0.8))]
    fn new(
        history_size: usize,
        prediction_steps: usize,
        momentum: f32,
        correction_weight: f32,
        min_correlation: f32,
    ) -> Self {
        Self {
            inner: PredictionConfig {
                history_size,
                prediction_steps,
                momentum,
                correction_weight,
                min_correlation,
            },
        }
    }

    #[getter]
    fn history_size(&self) -> usize {
        self.inner.history_size
    }

    #[getter]
    fn prediction_steps(&self) -> usize {
        self.inner.prediction_steps
    }

    #[getter]
    fn momentum(&self) -> f32 {
        self.inner.momentum
    }

    fn __repr__(&self) -> String {
        format!(
            "PredictionConfig(history_size={}, prediction_steps={}, momentum={})",
            self.inner.history_size, self.inner.prediction_steps, self.inner.momentum
        )
    }
}

/// Python-accessible phase training configuration.
#[pyclass(name = "PhaseConfig")]
#[derive(Clone)]
pub struct PyPhaseConfig {
    inner: PhaseConfig,
}

#[pymethods]
impl PyPhaseConfig {
    /// Create a new phase training configuration.
    ///
    /// Args:
    ///     full_steps: Full gradient computation steps per cycle (default: 10)
    ///     predict_steps: Predicted gradient steps per cycle (default: 40)
    ///     correct_every: Correction frequency during predict phase (default: 10)
    ///     gradient_accumulation: Gradient accumulation steps (default: 1)
    ///     max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    ///     adaptive_phases: Adaptively adjust phase lengths (default: True)
    ///     loss_threshold: Loss increase threshold for more full steps (default: 0.1)
    #[new]
    #[pyo3(signature = (
        full_steps=10,
        predict_steps=40,
        correct_every=10,
        gradient_accumulation=1,
        max_grad_norm=1.0,
        adaptive_phases=true,
        loss_threshold=0.1,
        prediction_config=None,
        ternary_config=None,
        vsa_config=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        full_steps: usize,
        predict_steps: usize,
        correct_every: usize,
        gradient_accumulation: usize,
        max_grad_norm: f32,
        adaptive_phases: bool,
        loss_threshold: f32,
        prediction_config: Option<PyPredictionConfig>,
        ternary_config: Option<PyTernaryConfig>,
        vsa_config: Option<PyVSAConfig>,
    ) -> Self {
        Self {
            inner: PhaseConfig {
                full_steps,
                predict_steps,
                correct_every,
                prediction_config: prediction_config.map(|c| c.inner).unwrap_or_default(),
                ternary_config: ternary_config.map(|c| c.inner).unwrap_or_default(),
                vsa_config: vsa_config.map(|c| c.inner).unwrap_or_default(),
                gradient_accumulation,
                max_grad_norm,
                adaptive_phases,
                loss_threshold,
            },
        }
    }

    #[getter]
    fn full_steps(&self) -> usize {
        self.inner.full_steps
    }

    #[getter]
    fn predict_steps(&self) -> usize {
        self.inner.predict_steps
    }

    #[getter]
    fn correct_every(&self) -> usize {
        self.inner.correct_every
    }

    fn __repr__(&self) -> String {
        format!(
            "PhaseConfig(full_steps={}, predict_steps={}, correct_every={})",
            self.inner.full_steps, self.inner.predict_steps, self.inner.correct_every
        )
    }
}

// ============================================================================
// Stats Wrappers
// ============================================================================

/// Training statistics returned by `PhaseTrainer.get_stats()`.
#[pyclass(name = "TrainerStats")]
pub struct PyTrainerStats {
    /// Total training steps completed.
    #[pyo3(get)]
    pub total_steps: usize,
    /// Steps with full gradient computation.
    #[pyo3(get)]
    pub full_steps: usize,
    /// Steps using predicted gradients.
    #[pyo3(get)]
    pub predicted_steps: usize,
    /// Steps with correction applied.
    #[pyo3(get)]
    pub correction_steps: usize,
    /// Estimated speedup factor.
    #[pyo3(get)]
    pub speedup: f32,
    /// Average loss across training.
    #[pyo3(get)]
    pub avg_loss: f32,
}

#[pymethods]
impl PyTrainerStats {
    fn __repr__(&self) -> String {
        format!(
            "TrainerStats(total={}, full={}, predicted={}, speedup={:.2}x)",
            self.total_steps, self.full_steps, self.predicted_steps, self.speedup
        )
    }
}

/// Step information returned by `PhaseTrainer.begin_step()`.
#[pyclass(name = "StepInfo")]
pub struct PyStepInfo {
    /// Current training step.
    #[pyo3(get)]
    pub step: usize,
    /// Current phase ("FULL", "PREDICT", or "CORRECT").
    #[pyo3(get)]
    pub phase: String,
    /// Whether full gradients should be computed.
    #[pyo3(get)]
    pub compute_full: bool,
}

#[pymethods]
impl PyStepInfo {
    fn __repr__(&self) -> String {
        format!(
            "StepInfo(step={}, phase={}, compute_full={})",
            self.step, self.phase, self.compute_full
        )
    }
}

// ============================================================================
// Main Component Wrappers
// ============================================================================

/// Convert Python dict of {name: list[float]} to HashMap<String, Tensor>.
fn dict_to_gradients(
    gradients: HashMap<String, Vec<f32>>,
    shapes: &[(String, Vec<usize>)],
    device: &Device,
) -> PyResult<HashMap<String, Tensor>> {
    let mut result = HashMap::new();

    for (name, data) in gradients {
        // Find the shape for this parameter
        let shape = shapes
            .iter()
            .find(|(n, _)| n == &name)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| PyValueError::new_err(format!("Unknown parameter: {name}")))?;

        let tensor = Tensor::from_vec(data, shape.as_slice(), device)
            .map_err(|e| PyValueError::new_err(format!("Tensor creation failed: {e}")))?;
        result.insert(name, tensor);
    }

    Ok(result)
}

/// Convert HashMap<String, Tensor> to Python dict of {name: list[float]}.
fn gradients_to_dict(gradients: &HashMap<String, Tensor>) -> PyResult<HashMap<String, Vec<f32>>> {
    let mut result = HashMap::new();

    for (name, tensor) in gradients {
        let data: Vec<f32> = tensor
            .flatten_all()
            .map_err(|e| PyValueError::new_err(format!("Flatten failed: {e}")))?
            .to_vec1()
            .map_err(|e| PyValueError::new_err(format!("Conversion failed: {e}")))?;
        result.insert(name.clone(), data);
    }

    Ok(result)
}

/// Phase-based training orchestrator.
///
/// Combines VSA compression, ternary accumulation, and gradient prediction
/// to accelerate training by ~5x while maintaining convergence.
///
/// Example:
///     >>> trainer = PhaseTrainer([("fc.weight", [64, 128])], PhaseConfig())
///     >>> for step in range(1000):
///     ...     info = trainer.begin_step()
///     ...     if trainer.should_compute_full():
///     ...         trainer.record_full_gradients(grads)
///     ...     else:
///     ...         predicted = trainer.get_predicted_gradients()
///     ...     trainer.end_step(loss)
#[pyclass(name = "PhaseTrainer")]
pub struct PyPhaseTrainer {
    inner: RustPhaseTrainer,
    shapes: Vec<(String, Vec<usize>)>,
    device: Device,
}

#[pymethods]
impl PyPhaseTrainer {
    /// Create a new phase trainer.
    ///
    /// Args:
    ///     shapes: List of (name, shape) tuples for model parameters
    ///     config: Phase training configuration
    ///     use_cuda: Optional bool, None to auto-detect CUDA (default: None)
    #[new]
    #[pyo3(signature = (shapes, config=None, use_cuda=None))]
    fn new(
        shapes: Vec<(String, Vec<usize>)>,
        config: Option<PyPhaseConfig>,
        use_cuda: Option<bool>,
    ) -> PyResult<Self> {
        let device = resolve_device(use_cuda)?;

        let config = config.map(|c| c.inner).unwrap_or_default();
        let inner = RustPhaseTrainer::new(&shapes, config, &device)
            .map_err(|e| PyValueError::new_err(format!("Trainer creation failed: {e}")))?;

        Ok(Self {
            inner,
            shapes,
            device,
        })
    }

    /// Begin a training step.
    ///
    /// Returns:
    ///     StepInfo with current step, phase, and whether to compute full gradients.
    fn begin_step(&mut self) -> PyResult<PyStepInfo> {
        let info = self
            .inner
            .begin_step()
            .map_err(|e| PyValueError::new_err(format!("begin_step failed: {e}")))?;

        Ok(PyStepInfo {
            step: info.total_step,
            phase: info.phase.to_string(),
            compute_full: matches!(info.phase, TrainingPhase::Full | TrainingPhase::Correct),
        })
    }

    /// Check if full gradients should be computed this step.
    fn should_compute_full(&self) -> bool {
        self.inner.should_compute_full()
    }

    /// Record full gradients computed via backpropagation.
    ///
    /// Args:
    ///     gradients: Dict mapping parameter names to flat gradient arrays.
    fn record_full_gradients(&mut self, gradients: HashMap<String, Vec<f32>>) -> PyResult<()> {
        let grads = dict_to_gradients(gradients, &self.shapes, &self.device)?;
        self.inner
            .record_full_gradients(&grads)
            .map_err(|e| PyValueError::new_err(format!("record_full_gradients failed: {e}")))
    }

    /// Get predicted gradients for this step.
    ///
    /// Returns:
    ///     Dict mapping parameter names to flat gradient arrays.
    fn get_predicted_gradients(&mut self) -> PyResult<HashMap<String, Vec<f32>>> {
        let predicted = self
            .inner
            .get_predicted_gradients()
            .map_err(|e| PyValueError::new_err(format!("get_predicted_gradients failed: {e}")))?;
        gradients_to_dict(&predicted)
    }

    /// End the current training step.
    ///
    /// Args:
    ///     loss: The loss value for this step.
    fn end_step(&mut self, loss: f32) -> PyResult<()> {
        self.inner
            .end_step(loss)
            .map_err(|e| PyValueError::new_err(format!("end_step failed: {e}")))
    }

    /// Get current training statistics.
    fn get_stats(&self) -> PyTrainerStats {
        let stats = self.inner.get_stats();
        // Compute average loss from phase losses
        let avg_loss = if stats.phase_avg_losses.is_empty() {
            0.0
        } else {
            stats.phase_avg_losses.values().sum::<f32>() / stats.phase_avg_losses.len() as f32
        };
        PyTrainerStats {
            total_steps: stats.total_steps,
            full_steps: stats.full_steps,
            predicted_steps: stats.predict_steps,
            correction_steps: stats.correct_steps,
            speedup: stats.speedup,
            avg_loss,
        }
    }

    /// Get the current training phase.
    fn current_phase(&self) -> String {
        self.inner.current_phase().to_string()
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.get_stats();
        format!(
            "PhaseTrainer(step={}, phase={}, speedup={:.2}x)",
            stats.total_steps,
            self.inner.current_phase(),
            stats.speedup
        )
    }
}

/// VSA gradient compressor using hyperdimensional computing.
///
/// Compresses gradients using Vector Symbolic Architecture with proper
/// bind/bundle/unbind operations for memory-efficient storage.
///
/// Example:
///     >>> compressor = VSAGradientCompressor(1000000, VSAConfig(dimension=8192))
///     >>> compressed, metadata = compressor.compress(gradients)
///     >>> reconstructed = compressor.decompress(compressed, metadata)
#[pyclass(name = "VSAGradientCompressor")]
pub struct PyVSAGradientCompressor {
    inner: RustVSACompressor,
    param_count: usize,
}

#[pymethods]
impl PyVSAGradientCompressor {
    /// Create a new VSA gradient compressor.
    ///
    /// Args:
    ///     param_count: Total number of model parameters.
    ///     config: VSA configuration.
    #[new]
    #[pyo3(signature = (param_count, config=None))]
    fn new(param_count: usize, config: Option<PyVSAConfig>) -> Self {
        let config = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: RustVSACompressor::new(param_count, config),
            param_count,
        }
    }

    /// Get the compressed dimension.
    #[getter]
    fn compressed_dim(&self) -> usize {
        self.inner.compressed_dim()
    }

    /// Get compression statistics.
    fn get_compression_stats(&self) -> HashMap<String, f32> {
        let stats = self.inner.get_compression_stats();
        let mut result = HashMap::new();
        result.insert("original_params".to_string(), stats.original_params as f32);
        result.insert("compressed_dim".to_string(), stats.compressed_dim as f32);
        result.insert("memory_saving".to_string(), stats.memory_saving);
        result.insert("compression_ratio".to_string(), stats.compression_ratio);
        result
    }

    fn __repr__(&self) -> String {
        format!(
            "VSAGradientCompressor(params={}, dim={})",
            self.param_count,
            self.inner.compressed_dim()
        )
    }
}

/// Ternary gradient accumulator for memory-efficient training.
///
/// Accumulates gradients in ternary format {-1, 0, +1} achieving ~10x
/// memory reduction compared to full precision.
#[pyclass(name = "TernaryGradientAccumulator")]
pub struct PyTernaryAccumulator {
    inner: RustTernaryAccumulator,
    shapes: Vec<(String, Vec<usize>)>,
    device: Device,
}

#[pymethods]
impl PyTernaryAccumulator {
    /// Create a new ternary gradient accumulator.
    ///
    /// Args:
    ///     shapes: List of (name, shape) tuples for model parameters.
    ///     config: Ternary configuration.
    ///     use_cuda: Optional bool, None to auto-detect CUDA (default: None)
    #[new]
    #[pyo3(signature = (shapes, config=None, use_cuda=None))]
    fn new(
        shapes: Vec<(String, Vec<usize>)>,
        config: Option<PyTernaryConfig>,
        use_cuda: Option<bool>,
    ) -> PyResult<Self> {
        let device = resolve_device(use_cuda)?;

        let config = config.map(|c| c.inner).unwrap_or_default();
        let inner = RustTernaryAccumulator::new(&shapes, config, &device)
            .map_err(|e| PyValueError::new_err(format!("Accumulator creation failed: {e}")))?;

        Ok(Self {
            inner,
            shapes,
            device,
        })
    }

    /// Accumulate gradients.
    ///
    /// Args:
    ///     gradients: Dict mapping parameter names to flat gradient arrays.
    fn accumulate(&mut self, gradients: HashMap<String, Vec<f32>>) -> PyResult<()> {
        let grads = dict_to_gradients(gradients, &self.shapes, &self.device)?;
        self.inner
            .accumulate(&grads)
            .map_err(|e| PyValueError::new_err(format!("accumulate failed: {e}")))
    }

    /// Get accumulated gradients.
    ///
    /// Returns:
    ///     Dict mapping parameter names to flat gradient arrays.
    fn get_accumulated(&self) -> PyResult<HashMap<String, Vec<f32>>> {
        let accumulated = self
            .inner
            .get_accumulated()
            .map_err(|e| PyValueError::new_err(format!("get_accumulated failed: {e}")))?;
        gradients_to_dict(&accumulated)
    }

    /// Reset the accumulator.
    fn reset(&mut self) -> PyResult<()> {
        self.inner
            .reset()
            .map_err(|e| PyValueError::new_err(format!("reset failed: {e}")))
    }

    /// Get current accumulation step count.
    fn current_step(&self) -> usize {
        self.inner.count()
    }

    fn __repr__(&self) -> String {
        format!(
            "TernaryGradientAccumulator(step={}/{})",
            self.inner.count(),
            self.shapes.len()
        )
    }
}

/// Gradient predictor for training acceleration.
///
/// Predicts gradients based on history to reduce compute by ~80%
/// while maintaining convergence through periodic correction.
#[pyclass(name = "GradientPredictor")]
pub struct PyGradientPredictor {
    inner: RustGradientPredictor,
    shapes: Vec<(String, Vec<usize>)>,
    device: Device,
}

#[pymethods]
impl PyGradientPredictor {
    /// Create a new gradient predictor.
    ///
    /// Args:
    ///     shapes: List of (name, shape) tuples for model parameters.
    ///     config: Prediction configuration.
    ///     use_cuda: Optional bool, None to auto-detect CUDA (default: None)
    #[new]
    #[pyo3(signature = (shapes, config=None, use_cuda=None))]
    fn new(
        shapes: Vec<(String, Vec<usize>)>,
        config: Option<PyPredictionConfig>,
        use_cuda: Option<bool>,
    ) -> PyResult<Self> {
        let device = resolve_device(use_cuda)?;

        let config = config.map(|c| c.inner).unwrap_or_default();
        let inner = RustGradientPredictor::new(&shapes, config, &device)
            .map_err(|e| PyValueError::new_err(format!("Predictor creation failed: {e}")))?;

        Ok(Self {
            inner,
            shapes,
            device,
        })
    }

    /// Check if full gradient should be computed.
    fn should_compute_full(&self) -> bool {
        self.inner.should_compute_full()
    }

    /// Record a gradient for history.
    ///
    /// Args:
    ///     gradients: Dict mapping parameter names to flat gradient arrays.
    fn record_gradient(&mut self, gradients: HashMap<String, Vec<f32>>) -> PyResult<()> {
        let grads = dict_to_gradients(gradients, &self.shapes, &self.device)?;
        self.inner
            .record_gradient(&grads)
            .map_err(|e| PyValueError::new_err(format!("record_gradient failed: {e}")))
    }

    /// Predict gradient for current step.
    ///
    /// Returns:
    ///     Dict mapping parameter names to flat gradient arrays.
    fn predict_gradient(&mut self) -> PyResult<HashMap<String, Vec<f32>>> {
        let predicted = self
            .inner
            .predict_gradient()
            .map_err(|e| PyValueError::new_err(format!("predict_gradient failed: {e}")))?;
        gradients_to_dict(&predicted)
    }

    /// Get predictor statistics.
    fn get_stats(&self) -> HashMap<String, f32> {
        let stats = self.inner.get_stats();
        let mut result = HashMap::new();
        result.insert("history_size".to_string(), stats.history_size as f32);
        result.insert("prediction_ratio".to_string(), stats.prediction_ratio);
        result.insert("total_steps".to_string(), stats.total_steps as f32);
        result.insert("mean_error".to_string(), stats.mean_error);
        result
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.get_stats();
        format!(
            "GradientPredictor(history={}, ratio={:.1}%)",
            stats.history_size,
            stats.prediction_ratio * 100.0
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Python module for VSA training optimization.
#[pymodule]
#[pyo3(name = "vsa_optim_rs")]
pub fn vsa_optim_rs_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configurations
    m.add_class::<PyVSAConfig>()?;
    m.add_class::<PyTernaryConfig>()?;
    m.add_class::<PyPredictionConfig>()?;
    m.add_class::<PyPhaseConfig>()?;

    // Stats
    m.add_class::<PyTrainerStats>()?;
    m.add_class::<PyStepInfo>()?;

    // Main components
    m.add_class::<PyPhaseTrainer>()?;
    m.add_class::<PyVSAGradientCompressor>()?;
    m.add_class::<PyTernaryAccumulator>()?;
    m.add_class::<PyGradientPredictor>()?;

    Ok(())
}

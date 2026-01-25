# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Loss History Tracking** (`LossHistory`)
  - Per-step loss recording with phase and timestamp information
  - Statistical analysis: mean, variance, std dev, min, max, median
  - Rolling statistics over configurable windows
  - Convergence detection with configurable thresholds
  - Anomaly detection (spikes and divergence)
  - Per-phase statistics for comparing Warmup, Full, Predict, and Correct phases
  - JSON export for persistence and external visualization
  - Time tracking for performance analysis
  - 16 unit tests + 9 integration tests
  - Comprehensive documentation and examples
- `LossHistoryConfig` for configurable tracking behavior
  - `max_history`: Maximum measurements to retain (default: 1000)
  - `rolling_window`: Window for rolling statistics (default: 20)
  - `spike_threshold`: Sensitivity for spike detection (default: 3.0σ)
  - `divergence_threshold`: Threshold for divergence detection (default: 1.2×)
- `DeterministicPhaseConfig` extensions:
  - `track_loss_history`: Enable/disable tracking (default: true)
  - `loss_history_config`: Configure loss tracking behavior
  - `with_loss_tracking()`: Builder method
  - `with_loss_history_config()`: Builder method
- `DeterministicPhaseTrainer` methods:
  - `loss_history()`: Access loss history tracker
  - `loss_history_mut()`: Mutable access to tracker
  - Automatic loss recording in `end_step()`
  - Loss history reset in `reset()`
- Example program `examples/loss_tracking_demo.rs`
- Documentation `LOSS_TRACKING_GUIDE.md` with comprehensive usage examples

### Changed
- Loss tracking is now enabled by default in `DeterministicPhaseConfig`
- Integration tests now cover loss history functionality

## [0.1.1] - 2026-01-24

### Added
- CUDA-first device selection with explicit CPU fallback warnings
- Python bindings now auto-detect CUDA (with `VSA_OPTIM_FORCE_CPU` override)

### Changed
- Bumped minimum Rust version to 1.92

## [0.1.0] - 2026-01-24

### Added

- **Deterministic Gradient Prediction** (`DeterministicPredictor`)
  - Weighted least squares model fitting with closed-form Cramer's rule solution
  - Linear gradient model: `g(t) = baseline + velocity × t + residual`
  - Exponential decay weighting (0.95) favoring recent gradients
  - Residual tracking with exponential averaging for drift correction
  - Configurable warmup, history window, and prediction horizon

- **Deterministic Phase Training** (`DeterministicPhaseTrainer`)
  - Four-phase training cycle: WARMUP → FULL → PREDICT → CORRECT
  - Automatic phase transitions based on step counts
  - Adaptive phase adjustment on loss increase
  - Complete statistics tracking (speedup, full steps, predicted steps)
  - Guaranteed reproducibility: identical inputs = identical outputs

- **VSA Gradient Compression** (`VSAGradientCompressor`)
  - Hyperdimensional computing with bind/bundle/unbind operations
  - Johnson-Lindenstrauss random projection for dimensionality reduction
  - Quasi-orthogonal key vectors for gradient binding
  - Majority-vote bundling for compressed representation
  - ~90% gradient storage reduction with configurable reconstruction accuracy

- **Ternary Gradient Accumulation** (`TernaryGradientAccumulator`)
  - Balanced ternary `{-1, 0, +1}` representation via `trit-vsa` crate
  - Stochastic and deterministic rounding modes
  - ~93% memory reduction during gradient accumulation
  - Optimizer wrapper for seamless integration

- **Legacy Phase Training** (`PhaseTrainer`)
  - Momentum-based gradient extrapolation
  - Three-phase cycle: FULL → PREDICT → CORRECT
  - Configurable phase lengths and correction frequency

- **Configuration System**
  - `DeterministicPhaseConfig` for deterministic training
  - `PhaseConfig` for legacy phase training
  - `VSAConfig` for compression parameters
  - `TernaryConfig` for accumulation settings
  - `PredictionConfig` for gradient prediction

- **Integration Support**
  - Full integration with `axolotl-rs` via `VSAAccelerator`
  - Optional Python bindings via PyO3 (`python` feature)
  - Workspace-level dependency management

### Technical Details

- Pure Rust implementation with no unsafe code
- Built on `candle-core` 0.9+ for tensor operations
- Leverages `trit-vsa` for ternary arithmetic
- Comprehensive test suite (53 unit tests, 7 integration tests)
- Criterion benchmarks for performance validation

### Performance

| Metric | Value |
|--------|-------|
| Gradient storage reduction | ~90% |
| Backward pass reduction | ~80% |
| Accumulation memory savings | ~93% |
| Theoretical speedup | 2-3x |

## [Unreleased]

### Planned

- GPU acceleration via CubeCL backend
- Adaptive history window sizing
- Per-layer prediction models
- Quantization-aware prediction

---

[0.1.0]: https://github.com/tzervas/vsa-optim-rs/releases/tag/v0.1.0
[Unreleased]: https://github.com/tzervas/vsa-optim-rs/compare/v0.1.0...HEAD

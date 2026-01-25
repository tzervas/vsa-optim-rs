# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

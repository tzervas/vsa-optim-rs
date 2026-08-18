# vsa-optim-rs Development Guide

## Project Overview

Training optimization toolkit using Vector Symbolic Architecture (VSA), ternary
quantization, and **deterministic** closed-form gradient prediction. Enables
efficient large model fine-tuning on consumer hardware.

## Architecture

```
src/
├── lib.rs              # Public API exports
├── config.rs           # Configuration types
├── error.rs            # Error handling
├── phase/              # Training phase management
│   ├── mod.rs
│   ├── trainer.rs           # Legacy PhaseTrainer (momentum-based)
│   └── deterministic_trainer.rs  # DeterministicPhaseTrainer (least squares)
├── prediction/         # Gradient prediction
│   ├── mod.rs
│   ├── predictor.rs         # Legacy momentum predictor
│   └── deterministic.rs     # Deterministic least squares predictor
├── ternary/            # Ternary gradient operations
│   ├── mod.rs
│   └── accumulator.rs       # TernaryGradientAccumulator
└── vsa/                # VSA compression
    ├── mod.rs
    └── compressor.rs        # VSAGradientCompressor
```

## Key Components

### DeterministicPredictor (Primary)

**Location**: `src/prediction/deterministic.rs`

Predicts gradients using weighted least squares with closed-form solution:

```
g(t) = baseline + velocity × t + residual
```

Key properties:
- **Deterministic**: No stochastic operations
- **Closed-form**: Cramer's rule for normal equations
- **Drift correction**: Residual tracking with exponential decay

### DeterministicPhaseTrainer (Primary)

**Location**: `src/phase/deterministic_trainer.rs`

Orchestrates training through four phases:
1. **WARMUP**: Collect initial gradients for predictor initialization
2. **FULL**: Standard backprop with gradient recording
3. **PREDICT**: Use predicted gradients (no backward pass)
4. **CORRECT**: Compute actual gradient, update residuals

### VSAGradientCompressor

**Location**: `src/vsa/compressor.rs`

Compresses gradients using hyperdimensional computing:
- Random projection to high-dimensional space
- Bind gradients with unique keys
- Bundle via majority voting
- Unbind and inverse project to decompress

### TernaryGradientAccumulator

**Location**: `src/ternary/accumulator.rs`

Memory-efficient gradient accumulation in `{-1, 0, +1}`:
- Stochastic or deterministic rounding
- ~93% memory savings
- Unbiased gradient estimates

## Build Commands

```bash
# Check compilation
cargo check -p vsa-optim-rs

# Run tests
cargo test -p vsa-optim-rs

# Run specific test
cargo test -p vsa-optim-rs test_deterministic_prediction

# Integration tests
cargo test -p vsa-optim-rs --test integration_tests

# Benchmarks
cargo bench -p vsa-optim-rs

# Build with Python bindings
cargo build -p vsa-optim-rs --features python

# Generate docs
cargo doc -p vsa-optim-rs --open
```

## Testing

- **Unit tests**: 53 tests covering all modules
- **Integration tests**: 7 tests in `tests/integration_tests.rs`
- **Determinism tests**: Verify identical inputs = identical outputs

Key test files:
- `src/prediction/deterministic.rs` (inline tests)
- `src/phase/deterministic_trainer.rs` (inline tests)
- `tests/integration_tests.rs`

## Dependencies

- `candle-core`: Tensor operations
- `trit-vsa`: Balanced ternary arithmetic
- `thiserror`: Error handling
- `serde`: Configuration serialization
- `rand/rand_chacha`: Reproducible random generation

## Integration Points

### axolotl-rs Integration

The `VSAAccelerator` in axolotl-rs wraps `DeterministicPhaseTrainer`:

```rust
// axolotl-rs/src/vsa_accel.rs
use vsa_optim_rs::{DeterministicPhaseTrainer, DeterministicPhaseConfig};

pub struct VSAAccelerator {
    trainer: DeterministicPhaseTrainer,
    config: VSAAcceleratorConfig,
}
```

## Code Style

- Follow Rust 2021 idioms
- Use `thiserror` for error types
- Prefer `Result<T>` over panics
- Document public APIs with examples
- Keep functions focused and small

## Performance Considerations

- Prediction overhead: O(history_window × parameters)
- Memory: ~7% of baseline gradient storage
- Speedup: ~2.9x with default configuration (35 backward passes per 100 steps)

## Release Checklist

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Run full test suite: `cargo test -p vsa-optim-rs`
4. Check documentation: `cargo doc -p vsa-optim-rs`
5. Verify examples compile
6. Tag release: `git tag v0.1.0`
7. Publish: `cargo publish -p vsa-optim-rs`

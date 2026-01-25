# Loss History Tracking Guide

This guide explains how to use the loss history tracking feature in vsa-optim-rs to monitor training dynamics, detect convergence, and identify anomalies during prediction-based training.

## Overview

The loss history tracking system provides:

- **Per-step loss recording** with phase and timestamp information
- **Statistical analysis** including mean, variance, median, and rolling statistics
- **Convergence detection** to identify when training is improving
- **Anomaly detection** for loss spikes and divergence
- **Phase-based analysis** to compare loss across warmup, full, predict, and correct phases
- **JSON export** for persistence and visualization

## Quick Start

### Enable Loss Tracking

```rust
use vsa_optim_rs::{DeterministicPhaseConfig, DeterministicPhaseTrainer};
use candle_core::Device;

let config = DeterministicPhaseConfig::default()
    .with_warmup_steps(10)
    .with_loss_tracking(true);  // Enable tracking

let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
let mut trainer = DeterministicPhaseTrainer::new(&param_shapes, config, &device)?;
```

### Record Losses During Training

```rust
for step in 0..num_steps {
    let info = trainer.begin_step()?;

    if info.needs_backward {
        // Compute gradients via backpropagation
        trainer.record_full_gradients(&gradients)?;
    } else {
        // Use predicted gradients
        let predicted = trainer.get_predicted_gradients()?;
    }

    // Loss is automatically recorded when you call end_step
    trainer.end_step(loss_value)?;
}
```

### Analyze Loss History

```rust
let history = trainer.loss_history().expect("Tracking enabled");

// Get current loss
let current = history.current_loss();

// Compute statistics
let stats = history.compute_statistics(None).unwrap();
println!("Mean loss: {}, Std dev: {}", stats.mean, stats.std_dev);

// Check for convergence
if history.is_converging(50, 0.05) {
    println!("Training is converging!");
}

// Detect anomalies
let anomalies = history.detect_anomalies(20);
for anomaly in anomalies {
    match anomaly {
        LossAnomaly::Spike { step, magnitude } => {
            println!("Loss spike at step {}: {}σ", step, magnitude);
        }
        LossAnomaly::Divergence { step, rate } => {
            println!("Loss divergence at step {}: {}x", step, rate);
        }
    }
}
```

## Configuration

### LossHistoryConfig

Control loss tracking behavior:

```rust
use vsa_optim_rs::LossHistoryConfig;

let loss_config = LossHistoryConfig::default()
    .with_max_history(1000)        // Keep last 1000 measurements
    .with_rolling_window(20)        // 20-step rolling statistics
    .with_spike_threshold(3.0);     // Detect spikes > 3 std devs

let trainer_config = DeterministicPhaseConfig::default()
    .with_loss_tracking(true)
    .with_loss_history_config(loss_config);
```

Configuration options:

- `max_history`: Maximum number of measurements to retain (default: 1000)
- `rolling_window`: Window size for rolling statistics (default: 20)
- `spike_threshold`: Multiples of std dev for spike detection (default: 3.0)
- `divergence_threshold`: Loss increase ratio for divergence (default: 1.2)
- `convergence_window`: Window for convergence detection (default: 50)
- `convergence_threshold`: Min improvement rate for convergence (default: 0.01)

## Features

### 1. Basic Statistics

Compute statistics over any window:

```rust
// All measurements
let stats = history.compute_statistics(None).unwrap();

// Last 50 measurements
let recent_stats = history.compute_statistics(Some(50)).unwrap();

println!("Mean: {}", stats.mean);
println!("Variance: {}", stats.variance);
println!("Std Dev: {}", stats.std_dev);
println!("Min: {}", stats.min);
println!("Max: {}", stats.max);
println!("Median: {}", stats.median);
```

### 2. Rolling Statistics

Track recent training dynamics:

```rust
// Rolling average over last 20 steps
let avg = history.rolling_average(20).unwrap();

// Rolling variance
let var = history.rolling_variance(20).unwrap();

// Rolling standard deviation
let std = history.rolling_std_dev(20).unwrap();
```

### 3. Convergence Detection

Determine if training is improving:

```rust
// Check if loss is decreasing over a 50-step window
// Requires at least 5% relative improvement
let is_converging = history.is_converging(50, 0.05);

if is_converging {
    println!("Training is converging - consider reducing full steps");
}

// Get improvement rate (negative = improving)
let rate = history.improvement_rate(50).unwrap();
println!("Loss change: {:.2}%", rate * 100.0);
```

### 4. Anomaly Detection

Identify unusual training behavior:

```rust
let anomalies = history.detect_anomalies(20);

for anomaly in anomalies {
    match anomaly {
        LossAnomaly::Spike { step, magnitude } => {
            eprintln!("WARNING: Loss spike at step {}", step);
            eprintln!("  Magnitude: {:.2} standard deviations", magnitude);
            // Consider triggering correction or reducing learning rate
        }
        LossAnomaly::Divergence { step, rate } => {
            eprintln!("WARNING: Loss divergence at step {}", step);
            eprintln!("  Rate: {:.2}x increase", rate);
            // Consider stopping or adjusting hyperparameters
        }
    }
}
```

### 5. Phase-Based Analysis

Compare loss across different training phases:

```rust
// Get measurements for specific phase
let warmup_losses = history.measurements_for_phase(DeterministicPhase::Warmup);
let predict_losses = history.measurements_for_phase(DeterministicPhase::Predict);

// Get statistics per phase
let phase_summary = history.phase_summary();
for (phase, stats) in phase_summary {
    println!("{}: mean={:.4}, count={}", phase, stats.mean, stats.count);
}

// Example output:
// WARMUP: mean=1.2345, count=10
// FULL: mean=0.9876, count=15
// PREDICT: mean=1.0234, count=62
// CORRECT: mean=0.9543, count=13
```

This helps identify if predicted gradients maintain training quality.

### 6. JSON Export

Persist loss history for analysis or visualization:

```rust
// Export to JSON string
let json = history.to_json()?;

// Write to file
use std::fs;
fs::write("loss_history.json", json)?;

// JSON format:
// [
//   {
//     "step": 0,
//     "loss": 1.234,
//     "phase": "WARMUP"
//   },
//   ...
// ]
```

This can be imported into Python, plotted with matplotlib, or analyzed with pandas.

### 7. Time Tracking

Monitor wall-clock training time:

```rust
let elapsed = history.elapsed_time();
println!("Training time: {:?}", elapsed);

// Access individual measurement timestamps
for measurement in history.measurements() {
    println!("Step {}: loss={}, time={:?}",
        measurement.step,
        measurement.loss,
        measurement.timestamp
    );
}
```

## Advanced Usage

### Adaptive Phase Adjustment

Use loss history to dynamically adjust phase lengths:

```rust
let config = DeterministicPhaseConfig::default()
    .with_adaptive_phases(true)  // Enable built-in adaptation
    .with_loss_tracking(true);

// The trainer automatically adjusts phase lengths based on loss
```

Or implement custom logic:

```rust
// Manual adjustment based on loss analysis
let history = trainer.loss_history().unwrap();

if let Some(rate) = history.improvement_rate(20) {
    if rate > 0.0 {  // Loss increasing
        // Need more full training
        config = config.with_full_steps(config.full_steps + 2);
        config = config.with_predict_steps(config.predict_steps - 5);
    } else if rate < -0.1 {  // Loss decreasing well
        // Can use more prediction
        config = config.with_full_steps(config.full_steps - 1);
        config = config.with_predict_steps(config.predict_steps + 3);
    }
}
```

### Custom Analysis

Access raw measurements for custom analytics:

```rust
let measurements = history.measurements();

// Custom windowed analysis
let window_size = 10;
for window in measurements.chunks(window_size) {
    let mean: f32 = window.iter().map(|m| m.loss).sum::<f32>() / window_size as f32;
    println!("Window mean: {}", mean);
}

// Phase transition analysis
let mut prev_phase = None;
for m in measurements {
    if Some(m.phase) != prev_phase {
        println!("Phase change at step {}: {:?}", m.step, m.phase);
        prev_phase = Some(m.phase);
    }
}
```

## Integration Examples

### With TensorBoard

Export to TensorBoard format (requires additional dependencies):

```rust
// Pseudo-code - requires tensorboard-rs or similar
use tensorboard_rs::summary_writer::SummaryWriter;

let writer = SummaryWriter::new("./runs/experiment1")?;

for measurement in history.measurements() {
    writer.add_scalar(
        "loss/overall",
        measurement.loss,
        measurement.step
    )?;
    writer.add_scalar(
        &format!("loss/{}", measurement.phase),
        measurement.loss,
        measurement.step
    )?;
}
```

### With Plotters

Visualize loss curves:

```rust
use plotters::prelude::*;

let root = BitMapBackend::new("loss_curve.png", (800, 600))
    .into_drawing_area();
root.fill(&WHITE)?;

let measurements = history.measurements();
let data: Vec<(usize, f32)> = measurements
    .iter()
    .map(|m| (m.step, m.loss))
    .collect();

let mut chart = ChartBuilder::on(&root)
    .caption("Training Loss", ("sans-serif", 50))
    .build_cartesian_2d(0..100, 0.0..2.0)?;

chart.draw_series(LineSeries::new(data, &RED))?;
```

### With Metrics Logging

Log to structured logging systems:

```rust
use tracing::{info, warn};

// Log statistics periodically
if step % 100 == 0 {
    let stats = history.compute_statistics(Some(20)).unwrap();
    info!(
        step = step,
        loss_mean = stats.mean,
        loss_std = stats.std_dev,
        "Loss statistics"
    );
}

// Log anomalies
let anomalies = history.detect_anomalies(20);
for anomaly in anomalies {
    warn!(?anomaly, "Loss anomaly detected");
}
```

## Performance Considerations

Loss history tracking has minimal overhead:

- **Memory**: ~40 bytes per measurement (loss + phase + step + timestamp)
  - Default 1000 measurements = ~40KB
- **CPU**: O(1) for recording, O(n) for statistics computation
- **Recommended**: Compute statistics every N steps rather than every step

To disable tracking for production:

```rust
let config = DeterministicPhaseConfig::default()
    .with_loss_tracking(false);  // Disable for maximum performance
```

## Troubleshooting

### Q: Loss history is None

A: Ensure you enabled tracking:

```rust
let config = config.with_loss_tracking(true);
```

### Q: Statistics return None

A: Need enough measurements for the requested window:

```rust
if history.len() >= window_size {
    let stats = history.compute_statistics(Some(window_size));
}
```

### Q: JSON export too large

A: Reduce max_history:

```rust
let config = LossHistoryConfig::default()
    .with_max_history(100);  // Only keep last 100
```

### Q: No anomalies detected despite visible spikes

A: Adjust threshold:

```rust
let config = LossHistoryConfig::default()
    .with_spike_threshold(2.0);  // More sensitive (default: 3.0)
```

## Best Practices

1. **Enable tracking during development/tuning**, disable in production
2. **Use rolling statistics** (last N steps) rather than all-time averages
3. **Monitor convergence** to decide when to stop or adjust hyperparameters
4. **Export history** periodically for post-training analysis
5. **Set appropriate thresholds** based on your domain (NLP vs vision vs RL)
6. **Check phase statistics** to ensure predicted steps maintain quality

## Examples

See `/examples/loss_tracking_demo.rs` for a complete working example.

## API Reference

Full API documentation: `cargo doc --open`

Key types:
- `LossHistory`: Main tracker struct
- `LossHistoryConfig`: Configuration
- `LossMeasurement`: Single measurement
- `LossStatistics`: Computed statistics
- `LossAnomaly`: Anomaly types (Spike/Divergence)

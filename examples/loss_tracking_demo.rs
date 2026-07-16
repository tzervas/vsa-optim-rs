//! Example demonstrating loss history tracking in VSA-based training.
//!
//! This example shows how to:
//! - Enable loss history tracking
//! - Access and analyze loss statistics
//! - Detect convergence and anomalies
//! - Export loss data to JSON
//!
//! Run with:
//! ```bash
//! cargo run --example loss_tracking_demo
//! ```

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use vsa_optim_rs::{
    DeterministicPhase, DeterministicPhaseConfig, DeterministicPhaseTrainer, LossHistoryConfig,
};

fn create_parameter_shapes() -> Vec<(String, Vec<usize>)> {
    vec![
        ("transformer.layer1.weight".to_string(), vec![128, 256]),
        ("transformer.layer1.bias".to_string(), vec![128]),
        ("transformer.layer2.weight".to_string(), vec![256, 512]),
        ("transformer.layer2.bias".to_string(), vec![256]),
    ]
}

fn mock_gradients(device: &Device, step: usize) -> HashMap<String, Tensor> {
    let mut grads = HashMap::new();

    // Simulate realistic gradient patterns
    let scale = 1.0 / (step + 1) as f32; // Gradients decrease over time

    grads.insert(
        "transformer.layer1.weight".to_string(),
        Tensor::ones((128, 256), DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    grads.insert(
        "transformer.layer1.bias".to_string(),
        Tensor::ones(128, DType::F32, device)
            .unwrap()
            .affine(scale as f64 * 0.5, 0.0)
            .unwrap(),
    );
    grads.insert(
        "transformer.layer2.weight".to_string(),
        Tensor::ones((256, 512), DType::F32, device)
            .unwrap()
            .affine(scale as f64 * 0.8, 0.0)
            .unwrap(),
    );
    grads.insert(
        "transformer.layer2.bias".to_string(),
        Tensor::ones(256, DType::F32, device)
            .unwrap()
            .affine(scale as f64 * 0.3, 0.0)
            .unwrap(),
    );

    grads
}

fn simulate_loss(step: usize) -> f32 {
    // Simulate a realistic loss curve with noise
    let base_loss = 2.0 * (-0.05 * step as f32).exp();
    let noise = (step as f32 * 0.1).sin() * 0.05;
    base_loss + noise
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VSA-Optim Loss History Tracking Demo ===\n");

    // Configure loss history tracking
    let loss_config = LossHistoryConfig::default()
        .with_max_history(500) // Keep last 500 measurements
        .with_rolling_window(20) // 20-step rolling statistics
        .with_spike_threshold(3.0); // Detect spikes > 3 std devs

    // Configure deterministic phase trainer with loss tracking enabled
    let trainer_config = DeterministicPhaseConfig::default()
        .with_warmup_steps(10)
        .with_full_steps(5)
        .with_predict_steps(15)
        .with_correct_every(5)
        .with_loss_tracking(true)
        .with_loss_history_config(loss_config);

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}\n", device);

    let shapes = create_parameter_shapes();
    let mut trainer = DeterministicPhaseTrainer::new(&shapes, trainer_config, &device)?;

    println!("Training for 100 steps with loss tracking enabled...\n");

    // Training loop
    for step in 0..100 {
        let info = trainer.begin_step()?;

        // Determine if we need to compute full gradients
        if info.needs_backward {
            let grads = mock_gradients(&device, step);
            trainer.record_full_gradients(&grads)?;
        } else {
            let _predicted = trainer.get_predicted_gradients()?;
        }

        // Compute and record loss
        let loss = simulate_loss(step);
        trainer.end_step(loss)?;

        // Print progress every 10 steps
        if (step + 1) % 10 == 0 {
            println!("Step {}: Phase={}, Loss={:.4}", step + 1, info.phase, loss);
        }
    }

    println!("\n=== Loss History Analysis ===\n");

    // Get loss history
    let history = trainer
        .loss_history()
        .expect("Loss tracking should be enabled");

    // Overall statistics
    println!("Total measurements: {}", history.len());
    println!("Training time: {:?}\n", history.elapsed_time());

    // Compute statistics over all measurements
    if let Some(stats) = history.compute_statistics(None) {
        println!("Overall Statistics:");
        println!("  Mean loss:     {:.4}", stats.mean);
        println!("  Std dev:       {:.4}", stats.std_dev);
        println!("  Min loss:      {:.4}", stats.min);
        println!("  Max loss:      {:.4}", stats.max);
        println!("  Median loss:   {:.4}\n", stats.median);
    }

    // Rolling statistics (last 20 steps)
    println!("Rolling Statistics (last 20 steps):");
    if let Some(avg) = history.rolling_average(20) {
        println!("  Rolling average: {:.4}", avg);
    }
    if let Some(std) = history.rolling_std_dev(20) {
        println!("  Rolling std dev: {:.4}", std);
    }
    if let Some(rate) = history.improvement_rate(20) {
        println!("  Improvement rate: {:.2}%\n", rate * 100.0);
    }

    // Convergence analysis
    println!("Convergence Analysis:");
    let is_converging = history.is_converging(50, 0.05);
    println!("  Is converging (50-step window): {}", is_converging);

    if let Some(current) = history.current_loss() {
        println!("  Current loss: {:.4}\n", current);
    }

    // Per-phase statistics
    println!("Statistics by Phase:");
    for (phase, stats) in history.phase_summary() {
        println!("  {}:", phase);
        println!("    Count:       {}", stats.count);
        println!("    Mean:        {:.4}", stats.mean);
        println!("    Std dev:     {:.4}", stats.std_dev);
    }
    println!();

    // Anomaly detection
    println!("Anomaly Detection:");
    let anomalies = history.detect_anomalies(20);
    if anomalies.is_empty() {
        println!("  No anomalies detected");
    } else {
        for anomaly in &anomalies {
            match anomaly {
                vsa_optim_rs::LossAnomaly::Spike { step, magnitude } => {
                    println!("  SPIKE at step {}: {:.2}σ above mean", step, magnitude);
                }
                vsa_optim_rs::LossAnomaly::Divergence { step, rate } => {
                    println!("  DIVERGENCE at step {}: {:.2}x increase", step, rate);
                }
            }
        }
    }
    println!();

    // Export to JSON (optional)
    println!("Exporting loss history to JSON...");
    match history.to_json() {
        Ok(json) => {
            // In a real application, you would write this to a file
            println!("  JSON export successful ({} bytes)", json.len());
            println!(
                "  Sample (first 200 chars): {}",
                &json[..200.min(json.len())]
            );
        }
        Err(e) => println!("  JSON export failed: {}", e),
    }
    println!();

    // Trainer statistics
    let trainer_stats = trainer.get_stats();
    println!("=== Training Statistics ===\n");
    println!("{}", trainer_stats);
    println!();

    // Compare prediction efficiency
    let backward_ratio = (trainer_stats.warmup_steps
        + trainer_stats.full_steps
        + trainer_stats.correct_steps) as f32
        / trainer_stats.total_steps as f32;
    let predict_ratio = trainer_stats.predict_steps as f32 / trainer_stats.total_steps as f32;

    println!("Efficiency Breakdown:");
    println!(
        "  Backward passes:  {:.1}% of steps",
        backward_ratio * 100.0
    );
    println!("  Predicted steps:  {:.1}% of steps", predict_ratio * 100.0);
    println!("  Overall speedup:  {:.2}x", trainer_stats.speedup);

    Ok(())
}

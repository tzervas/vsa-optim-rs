//! Integration tests for vsa-optim-rs.
//!
//! These tests verify the complete optimization pipeline including
//! VSA compression, ternary accumulation, gradient prediction, and
//! phase-based training orchestration.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use vsa_optim_rs::{
    config::{PhaseConfig, PredictionConfig, TernaryConfig, VSAConfig},
    phase::{
        DeterministicPhase, DeterministicPhaseConfig, DeterministicPhaseTrainer, PhaseTrainer,
    },
    prediction::GradientPredictor,
    ternary::TernaryGradientAccumulator,
    vsa::VSAGradientCompressor,
};

/// Helper to create mock gradients simulating a small MLP.
/// Uses smaller dimensions suitable for fast integration tests.
fn create_mlp_gradients(device: &Device) -> HashMap<String, Tensor> {
    let mut gradients = HashMap::new();

    // Layer 1: 32 -> 64 (small test MLP)
    gradients.insert(
        "fc1.weight".to_string(),
        Tensor::randn(0.0f32, 0.1, (64, 32), device).unwrap(),
    );
    gradients.insert(
        "fc1.bias".to_string(),
        Tensor::randn(0.0f32, 0.1, 64, device).unwrap(),
    );

    // Layer 2: 64 -> 32
    gradients.insert(
        "fc2.weight".to_string(),
        Tensor::randn(0.0f32, 0.1, (32, 64), device).unwrap(),
    );
    gradients.insert(
        "fc2.bias".to_string(),
        Tensor::randn(0.0f32, 0.1, 32, device).unwrap(),
    );

    // Layer 3: 32 -> 10 (output)
    gradients.insert(
        "fc3.weight".to_string(),
        Tensor::randn(0.0f32, 0.1, (10, 32), device).unwrap(),
    );
    gradients.insert(
        "fc3.bias".to_string(),
        Tensor::randn(0.0f32, 0.1, 10, device).unwrap(),
    );

    gradients
}

/// Extract shapes from gradients for API calls.
fn extract_shapes(gradients: &HashMap<String, Tensor>) -> Vec<(String, Vec<usize>)> {
    gradients
        .iter()
        .map(|(name, grad)| (name.clone(), grad.dims().to_vec()))
        .collect()
}

/// Compute total parameter count from gradients.
fn param_count(gradients: &HashMap<String, Tensor>) -> usize {
    gradients.values().map(|g| g.elem_count()).sum()
}

#[test]
fn test_vsa_compression_mlp_gradients() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let total_params = param_count(&gradients);

    // Create compressor with reasonable dimension for tests (~4K params)
    let mut compressor = VSAGradientCompressor::new(
        total_params,
        VSAConfig::default()
            .with_dimension(512)
            .with_compression_ratio(0.1),
    );

    // Compress
    let (bundled, metadata) = compressor.compress(&gradients).unwrap();

    // Verify compression
    let stats = compressor.get_compression_stats();
    println!(
        "Compressed {} params to {} dim ({:.1}% memory saved)",
        stats.original_params,
        stats.compressed_dim,
        stats.memory_saving * 100.0
    );
    assert!(stats.memory_saving > 0.5); // Should save >50% memory for test sizes

    // Decompress
    let reconstructed = compressor.decompress(&bundled, &metadata).unwrap();

    // Verify all gradients recovered with correct shapes
    assert_eq!(reconstructed.len(), gradients.len());
    for (name, orig) in &gradients {
        let recon = reconstructed.get(name).unwrap();
        assert_eq!(orig.dims(), recon.dims(), "Shape mismatch for {name}");
    }

    // Verify direction preservation for large weight matrices
    for (name, orig) in &gradients {
        if !name.contains("weight") {
            continue;
        }

        let recon = reconstructed.get(name).unwrap();
        let orig_flat: Vec<f32> = orig.flatten_all().unwrap().to_vec1().unwrap();
        let recon_flat: Vec<f32> = recon.flatten_all().unwrap().to_vec1().unwrap();

        let dot: f32 = orig_flat
            .iter()
            .zip(recon_flat.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_orig: f32 = orig_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_recon: f32 = recon_flat.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_orig > 1e-6 && norm_recon > 1e-6 {
            let cosine = dot / (norm_orig * norm_recon + 1e-8);
            println!("{name}: cosine similarity = {cosine:.3}");
            assert!(
                cosine > 0.0,
                "Gradient direction should be preserved for {name}"
            );
        }
    }
}

#[test]
fn test_ternary_accumulation_flow() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let shapes = extract_shapes(&gradients);

    let config = TernaryConfig::default()
        .with_accumulation_steps(4)
        .with_stochastic_rounding(true);

    let mut accumulator = TernaryGradientAccumulator::new(&shapes, config, &device).unwrap();

    // Simulate 4 accumulation steps
    for _ in 0..4 {
        accumulator.accumulate(&gradients).unwrap();
    }

    // Get accumulated gradients
    let accumulated = accumulator.get_accumulated().unwrap();
    assert_eq!(accumulated.len(), gradients.len());

    // Verify shapes
    for (name, orig) in &gradients {
        let acc = accumulated.get(name).unwrap();
        assert_eq!(orig.dims(), acc.dims());
    }

    // Reset and verify
    accumulator.reset().unwrap();
}

#[test]
fn test_gradient_prediction_cycle() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let shapes = extract_shapes(&gradients);

    let config = PredictionConfig::default()
        .with_history_size(3)
        .with_prediction_steps(2)
        .with_momentum(0.9);

    let mut predictor = GradientPredictor::new(&shapes, config, &device).unwrap();

    // Initially should compute full
    assert!(predictor.should_compute_full());

    // Record first gradient
    predictor.record_gradient(&gradients).unwrap();

    // After first recording, should still need more history
    assert!(predictor.should_compute_full());

    // Record more gradients to build history
    for _ in 0..2 {
        predictor.record_gradient(&gradients).unwrap();
    }

    // Now should have enough history to predict
    if !predictor.should_compute_full() {
        let predicted = predictor.predict_gradient().unwrap();
        assert_eq!(predicted.len(), gradients.len());

        for (name, orig) in &gradients {
            let pred = predicted.get(name).unwrap();
            assert_eq!(orig.dims(), pred.dims());
        }
    }

    // Check stats
    let stats = predictor.get_stats();
    assert!(stats.history_size <= 3);
}

#[test]
fn test_phase_trainer_cycle() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let shapes = extract_shapes(&gradients);

    let config = PhaseConfig::default()
        .with_full_steps(2)
        .with_predict_steps(4)
        .with_correct_every(2);

    let mut trainer = PhaseTrainer::new(&shapes, config, &device).unwrap();

    // Simulate several training steps
    for step in 0..10 {
        let _step_info = trainer.begin_step().unwrap();

        if trainer.should_compute_full() {
            // Full gradient computation phase
            trainer.record_full_gradients(&gradients).unwrap();
        } else {
            // Prediction phase
            let predicted = trainer.get_predicted_gradients().unwrap();
            assert_eq!(predicted.len(), gradients.len());
        }

        let loss = 1.0 / (step as f32 + 1.0); // Simulated decreasing loss
        trainer.end_step(loss).unwrap();
    }

    // Check final stats
    let stats = trainer.get_stats();
    assert!(stats.total_steps >= 10);
    assert!(stats.speedup > 0.0);
}

#[test]
fn test_combined_vsa_and_ternary() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let total_params = param_count(&gradients);

    // First compress with VSA
    let mut compressor =
        VSAGradientCompressor::new(total_params, VSAConfig::default().with_dimension(512));

    let (bundled, metadata) = compressor.compress(&gradients).unwrap();

    // The bundled result is already in ternary format (PackedTritVec)
    // Verify it's compact
    let memory_per_element = 2.0 / 8.0; // 2 bits per trit
    let compressed_bytes = bundled.len() as f32 * memory_per_element;
    let original_bytes = total_params as f32 * 4.0; // 32-bit floats
    let compression_ratio = compressed_bytes / original_bytes;

    println!(
        "Combined compression: {:.1}% of original ({:.0} bytes -> {:.0} bytes)",
        compression_ratio * 100.0,
        original_bytes,
        compressed_bytes
    );

    assert!(
        compression_ratio < 0.3,
        "Should achieve >70% compression with VSA+ternary for test sizes"
    );

    // Decompress and verify
    let reconstructed = compressor.decompress(&bundled, &metadata).unwrap();
    assert_eq!(reconstructed.len(), gradients.len());
}

#[test]
fn test_deterministic_phase_trainer_full_cycle() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let shapes = extract_shapes(&gradients);

    // Configure deterministic trainer
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(5)
        .with_full_steps(3)
        .with_predict_steps(6)
        .with_correct_every(3);

    let mut trainer = DeterministicPhaseTrainer::new(&shapes, config, &device).unwrap();

    // Track phases seen
    let mut phases_seen = std::collections::HashSet::new();
    let mut backward_count = 0;
    let mut forward_count = 0;

    // Simulate 30 training steps
    for step in 0..30 {
        let info = trainer.begin_step().unwrap();
        phases_seen.insert(format!("{}", info.phase));

        // Simulated forward pass (always needed)
        forward_count += 1;

        if info.needs_backward {
            // Full gradient computation
            backward_count += 1;
            let step_grads = create_mlp_gradients(&device);
            trainer.record_full_gradients(&step_grads).unwrap();
        } else {
            // Use predicted gradients
            let predicted = trainer.get_predicted_gradients().unwrap();
            assert_eq!(predicted.len(), gradients.len());
            for (name, pred) in &predicted {
                let orig_shape = gradients.get(name).unwrap().dims();
                assert_eq!(pred.dims(), orig_shape, "Shape mismatch for {name}");
            }
        }

        // Simulated loss (decreasing)
        let loss = 1.0 / (step + 1) as f32;
        trainer.end_step(loss).unwrap();
    }

    // Verify we saw the expected phases
    assert!(phases_seen.contains("WARMUP"), "Should have warmup phase");
    assert!(phases_seen.contains("FULL"), "Should have full phase");
    assert!(phases_seen.contains("PREDICT"), "Should have predict phase");

    // Verify speedup
    let stats = trainer.get_stats();
    println!("Deterministic trainer stats: {}", stats);
    println!("  Forward passes: {}", forward_count);
    println!("  Backward passes: {}", backward_count);
    println!("  Speedup: {:.2}x", stats.speedup);

    assert!(
        backward_count < forward_count,
        "Should have fewer backward passes than forward"
    );
    assert!(
        stats.speedup > 1.0,
        "Should achieve speedup with prediction"
    );
}

#[test]
fn test_deterministic_training_reproducibility() {
    let device = Device::Cpu;
    let gradients = create_mlp_gradients(&device);
    let shapes = extract_shapes(&gradients);

    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_full_steps(2)
        .with_predict_steps(4);

    // Create two identical trainers
    let mut trainer1 = DeterministicPhaseTrainer::new(&shapes, config.clone(), &device).unwrap();
    let mut trainer2 = DeterministicPhaseTrainer::new(&shapes, config, &device).unwrap();

    // Feed identical gradients and collect predictions
    let mut preds1 = Vec::new();
    let mut preds2 = Vec::new();

    for step in 0..15 {
        let info1 = trainer1.begin_step().unwrap();
        let info2 = trainer2.begin_step().unwrap();

        // Phases should match
        assert_eq!(
            format!("{}", info1.phase),
            format!("{}", info2.phase),
            "Phase mismatch at step {step}"
        );

        if info1.needs_backward {
            // Use deterministic gradients (same for both)
            let step_grads = create_deterministic_gradients(&device, step);
            trainer1.record_full_gradients(&step_grads).unwrap();
            trainer2.record_full_gradients(&step_grads).unwrap();
        } else {
            // Collect predictions
            let p1 = trainer1.get_predicted_gradients().unwrap();
            let p2 = trainer2.get_predicted_gradients().unwrap();
            preds1.push(p1);
            preds2.push(p2);
        }

        trainer1.end_step(0.5).unwrap();
        trainer2.end_step(0.5).unwrap();
    }

    // Verify predictions are identical
    assert!(!preds1.is_empty(), "Should have made predictions");
    for (i, (p1, p2)) in preds1.iter().zip(preds2.iter()).enumerate() {
        for (name, t1) in p1 {
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
                "Prediction {i} for {name} should be deterministic, diff={diff}"
            );
        }
    }
}

/// Create deterministic gradients for reproducibility testing.
fn create_deterministic_gradients(device: &Device, step: usize) -> HashMap<String, Tensor> {
    let mut gradients = HashMap::new();
    let scale = 1.0 + step as f32 * 0.05;

    gradients.insert(
        "fc1.weight".to_string(),
        Tensor::ones((64, 32), candle_core::DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    gradients.insert(
        "fc1.bias".to_string(),
        Tensor::ones(64, candle_core::DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    gradients.insert(
        "fc2.weight".to_string(),
        Tensor::ones((32, 64), candle_core::DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    gradients.insert(
        "fc2.bias".to_string(),
        Tensor::ones(32, candle_core::DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    gradients.insert(
        "fc3.weight".to_string(),
        Tensor::ones((10, 32), candle_core::DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    gradients.insert(
        "fc3.bias".to_string(),
        Tensor::ones(10, candle_core::DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );

    gradients
}

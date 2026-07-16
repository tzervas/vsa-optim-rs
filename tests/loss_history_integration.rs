//! Integration tests for loss history tracking in prediction training.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use vsa_optim_rs::{
    DeterministicPhase, DeterministicPhaseConfig, DeterministicPhaseTrainer, LossHistoryConfig,
};

fn create_shapes() -> Vec<(String, Vec<usize>)> {
    vec![
        ("layer.weight".to_string(), vec![16, 32]),
        ("layer.bias".to_string(), vec![16]),
    ]
}

fn create_mock_gradients(device: &Device, scale: f32) -> HashMap<String, Tensor> {
    let mut grads = HashMap::new();
    grads.insert(
        "layer.weight".to_string(),
        Tensor::ones((16, 32), DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    grads.insert(
        "layer.bias".to_string(),
        Tensor::ones(16, DType::F32, device)
            .unwrap()
            .affine(scale as f64, 0.0)
            .unwrap(),
    );
    grads
}

#[test]
fn test_loss_history_basic_integration() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(5)
        .with_full_steps(3)
        .with_predict_steps(6)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // Run training for 20 steps
    for i in 0..20 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0 + i as f32 * 0.05);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        // Decreasing loss to simulate convergence
        let loss = 1.0 / (i + 1) as f32;
        trainer.end_step(loss).unwrap();
    }

    // Verify loss history was recorded
    let history = trainer.loss_history().unwrap();
    assert_eq!(history.len(), 20);

    // Check that we have measurements from all phases
    let phase_summary = history.phase_summary();
    assert!(!phase_summary.is_empty());

    // Verify convergence detection
    assert!(history.is_converging(15, 0.05));

    // Verify statistics
    let stats = history.compute_statistics(None).unwrap();
    assert_eq!(stats.count, 20);
    assert!(stats.mean > 0.0);
    assert!(stats.std_dev >= 0.0);
}

#[test]
fn test_loss_history_phase_tracking() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_full_steps(2)
        .with_predict_steps(4)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // Track which phases we see
    let mut phases_seen = Vec::new();

    for i in 0..15 {
        let info = trainer.begin_step().unwrap();
        phases_seen.push(info.phase);

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        trainer.end_step(0.5).unwrap();
    }

    // Verify loss history recorded all phases
    let history = trainer.loss_history().unwrap();
    let warmup_losses = history.measurements_for_phase(DeterministicPhase::Warmup);
    let full_losses = history.measurements_for_phase(DeterministicPhase::Full);
    let predict_losses = history.measurements_for_phase(DeterministicPhase::Predict);

    assert!(!warmup_losses.is_empty(), "Should have warmup losses");
    assert!(!full_losses.is_empty(), "Should have full losses");
    assert!(!predict_losses.is_empty(), "Should have predict losses");
}

#[test]
fn test_loss_history_rolling_statistics() {
    let loss_config = LossHistoryConfig::default().with_rolling_window(5);

    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_loss_tracking(true)
        .with_loss_history_config(loss_config);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // Run warmup with varying loss
    for i in 0..10 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        let loss = 1.0 + i as f32 * 0.1;
        trainer.end_step(loss).unwrap();
    }

    let history = trainer.loss_history().unwrap();

    // Test rolling statistics
    let rolling_avg = history.rolling_average(5);
    assert!(rolling_avg.is_some());

    let rolling_std = history.rolling_std_dev(5);
    assert!(rolling_std.is_some());

    let improvement = history.improvement_rate(5);
    assert!(improvement.is_some());
}

#[test]
fn test_loss_history_anomaly_detection() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(5)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // First, establish a baseline with stable loss
    for i in 0..20 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        trainer.end_step(0.5).unwrap();
    }

    // Add a spike
    {
        let info = trainer.begin_step().unwrap();
        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }
        trainer.end_step(10.0).unwrap(); // Big spike
    }

    let history = trainer.loss_history().unwrap();
    let anomalies = history.detect_anomalies(20);

    // Should detect the spike
    assert!(!anomalies.is_empty(), "Should detect loss spike");
}

#[test]
fn test_loss_history_disabled() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_loss_tracking(false); // Disable tracking

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    for i in 0..5 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        trainer.end_step(0.5).unwrap();
    }

    // Loss history should be None when disabled
    assert!(trainer.loss_history().is_none());
}

#[test]
fn test_loss_history_json_export() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    for i in 0..5 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        }

        trainer.end_step(1.0 - i as f32 * 0.1).unwrap();
    }

    let history = trainer.loss_history().unwrap();
    let json = history.to_json().unwrap();

    // Verify JSON is valid and contains expected data
    assert!(json.contains("step"));
    assert!(json.contains("loss"));
    assert!(json.contains("phase"));
}

#[test]
fn test_loss_history_reset() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // Record some losses
    for i in 0..5 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        }

        trainer.end_step(0.5).unwrap();
    }

    assert_eq!(trainer.loss_history().unwrap().len(), 5);

    // Reset trainer
    trainer.reset().unwrap();

    // Loss history should be cleared
    assert_eq!(trainer.loss_history().unwrap().len(), 0);
}

#[test]
fn test_loss_history_convergence_detection() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // Simulate improving loss (converging)
    for i in 0..30 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        // Exponentially decreasing loss
        let loss = 1.0 * 0.95_f32.powi(i as i32);
        trainer.end_step(loss).unwrap();
    }

    let history = trainer.loss_history().unwrap();

    // Should detect convergence
    assert!(
        history.is_converging(20, 0.05),
        "Should detect converging loss"
    );

    // Improvement rate should be negative (improving)
    let improvement = history.improvement_rate(20).unwrap();
    assert!(improvement < 0.0, "Loss should be improving");
}

#[test]
fn test_loss_history_phase_statistics() {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(3)
        .with_full_steps(2)
        .with_predict_steps(5)
        .with_loss_tracking(true);

    let mut trainer =
        DeterministicPhaseTrainer::new(&create_shapes(), config, &Device::Cpu).unwrap();

    // Run a full cycle
    for i in 0..15 {
        let info = trainer.begin_step().unwrap();

        if info.needs_backward {
            let grads = create_mock_gradients(&Device::Cpu, 1.0);
            trainer.record_full_gradients(&grads).unwrap();
        } else {
            let _predicted = trainer.get_predicted_gradients().unwrap();
        }

        // Different loss values per phase type
        let loss = match info.phase {
            DeterministicPhase::Warmup => 1.0,
            DeterministicPhase::Full => 0.8,
            DeterministicPhase::Predict => 0.9,
            DeterministicPhase::Correct => 0.85,
        };
        trainer.end_step(loss).unwrap();
    }

    let history = trainer.loss_history().unwrap();
    let phase_summary = history.phase_summary();

    // Should have statistics for multiple phases
    assert!(
        phase_summary.len() >= 2,
        "Should have stats for multiple phases"
    );

    // Check that each phase has valid statistics
    for (phase, stats) in &phase_summary {
        assert!(
            stats.count > 0,
            "Phase {:?} should have measurements",
            phase
        );
        assert!(
            stats.mean > 0.0,
            "Phase {:?} should have positive mean",
            phase
        );
    }
}

//! Loss history tracking for prediction training.
//!
//! This module provides comprehensive loss tracking with statistical analysis
//! to monitor training dynamics, detect convergence, and identify anomalies.
//!
//! # Example
//!
//! ```ignore
//! use vsa_optim_rs::phase::LossHistory;
//! use vsa_optim_rs::phase::DeterministicPhase;
//!
//! let mut history = LossHistory::new(100);
//!
//! history.record(0.8, DeterministicPhase::Warmup);
//! history.record(0.5, DeterministicPhase::Full);
//!
//! let stats = history.compute_statistics(10);
//! println!("Mean loss: {}", stats.mean);
//! println!("Is converging: {}", history.is_converging(20, 0.01));
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::DeterministicPhase;

/// A single loss measurement with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossMeasurement {
    /// Step index when loss was recorded.
    pub step: usize,
    /// Loss value.
    pub loss: f32,
    /// Training phase at this step.
    pub phase: DeterministicPhase,
    /// Timestamp when recorded (duration since history creation).
    #[serde(skip)]
    pub timestamp: Duration,
}

/// Statistical summary of loss history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossStatistics {
    /// Number of measurements in window.
    pub count: usize,
    /// Mean loss value.
    pub mean: f32,
    /// Variance of loss values.
    pub variance: f32,
    /// Standard deviation of loss values.
    pub std_dev: f32,
    /// Minimum loss value.
    pub min: f32,
    /// Maximum loss value.
    pub max: f32,
    /// Median loss value.
    pub median: f32,
}

/// Anomaly detection result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossAnomaly {
    /// Loss spike detected (sudden increase).
    Spike {
        /// Step where spike occurred.
        step: usize,
        /// Magnitude of spike (multiples of std dev).
        magnitude: f32,
    },
    /// Loss divergence (sustained increase).
    Divergence {
        /// Step where divergence started.
        step: usize,
        /// Rate of increase.
        rate: f32,
    },
}

/// Configuration for loss history tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossHistoryConfig {
    /// Maximum number of measurements to keep.
    pub max_history: usize,
    /// Window size for rolling statistics.
    pub rolling_window: usize,
    /// Threshold for spike detection (multiples of std dev).
    pub spike_threshold: f32,
    /// Threshold for divergence detection (loss increase ratio).
    pub divergence_threshold: f32,
    /// Window for convergence detection.
    pub convergence_window: usize,
    /// Threshold for convergence (max loss decrease rate).
    pub convergence_threshold: f32,
}

impl Default for LossHistoryConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            rolling_window: 20,
            spike_threshold: 3.0,
            divergence_threshold: 1.2,
            convergence_window: 50,
            convergence_threshold: 0.01,
        }
    }
}

impl LossHistoryConfig {
    /// Builder: Set maximum history size.
    #[must_use]
    pub const fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Builder: Set rolling window size.
    #[must_use]
    pub const fn with_rolling_window(mut self, window: usize) -> Self {
        self.rolling_window = window;
        self
    }

    /// Builder: Set spike detection threshold.
    #[must_use]
    pub const fn with_spike_threshold(mut self, threshold: f32) -> Self {
        self.spike_threshold = threshold;
        self
    }
}

/// Loss history tracker with analysis capabilities.
///
/// Maintains a rolling window of loss measurements with timestamps and phase
/// information, providing statistical analysis and anomaly detection.
pub struct LossHistory {
    config: LossHistoryConfig,
    measurements: VecDeque<LossMeasurement>,
    start_time: Instant,
    current_step: usize,
}

impl LossHistory {
    /// Create a new loss history tracker.
    ///
    /// # Arguments
    ///
    /// * `max_history` - Maximum number of measurements to keep
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self::with_config(LossHistoryConfig {
            max_history,
            ..Default::default()
        })
    }

    /// Create a new loss history tracker with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for loss tracking
    #[must_use]
    pub fn with_config(config: LossHistoryConfig) -> Self {
        Self {
            measurements: VecDeque::with_capacity(config.max_history),
            config,
            start_time: Instant::now(),
            current_step: 0,
        }
    }

    /// Record a loss measurement.
    ///
    /// # Arguments
    ///
    /// * `loss` - Loss value to record
    /// * `phase` - Current training phase
    pub fn record(&mut self, loss: f32, phase: DeterministicPhase) {
        let measurement = LossMeasurement {
            step: self.current_step,
            loss,
            phase,
            timestamp: self.start_time.elapsed(),
        };

        self.measurements.push_back(measurement);

        // Maintain max history size
        while self.measurements.len() > self.config.max_history {
            self.measurements.pop_front();
        }

        self.current_step += 1;
    }

    /// Get the most recent loss value.
    #[must_use]
    pub fn current_loss(&self) -> Option<f32> {
        self.measurements.back().map(|m| m.loss)
    }

    /// Get all measurements.
    #[must_use]
    pub fn measurements(&self) -> &VecDeque<LossMeasurement> {
        &self.measurements
    }

    /// Get measurements for a specific phase.
    #[must_use]
    pub fn measurements_for_phase(&self, phase: DeterministicPhase) -> Vec<&LossMeasurement> {
        self.measurements
            .iter()
            .filter(|m| m.phase == phase)
            .collect()
    }

    /// Compute statistics over the most recent window.
    ///
    /// # Arguments
    ///
    /// * `window` - Number of recent measurements to include (None = all)
    #[must_use]
    pub fn compute_statistics(&self, window: Option<usize>) -> Option<LossStatistics> {
        if self.measurements.is_empty() {
            return None;
        }

        let window_size = window.unwrap_or(self.measurements.len());
        let start = self.measurements.len().saturating_sub(window_size);
        let values: Vec<f32> = self
            .measurements
            .iter()
            .skip(start)
            .map(|m| m.loss)
            .collect();

        if values.is_empty() {
            return None;
        }

        Some(compute_stats(&values))
    }

    /// Get rolling average over recent measurements.
    ///
    /// # Arguments
    ///
    /// * `window` - Window size for rolling average
    #[must_use]
    pub fn rolling_average(&self, window: usize) -> Option<f32> {
        if self.measurements.len() < window {
            return None;
        }

        let sum: f32 = self
            .measurements
            .iter()
            .rev()
            .take(window)
            .map(|m| m.loss)
            .sum();

        Some(sum / window as f32)
    }

    /// Get rolling variance over recent measurements.
    ///
    /// # Arguments
    ///
    /// * `window` - Window size for rolling variance
    #[must_use]
    pub fn rolling_variance(&self, window: usize) -> Option<f32> {
        let stats = self.compute_statistics(Some(window))?;
        Some(stats.variance)
    }

    /// Get rolling standard deviation over recent measurements.
    ///
    /// # Arguments
    ///
    /// * `window` - Window size for rolling std dev
    #[must_use]
    pub fn rolling_std_dev(&self, window: usize) -> Option<f32> {
        let stats = self.compute_statistics(Some(window))?;
        Some(stats.std_dev)
    }

    /// Check if loss is converging (decreasing trend).
    ///
    /// Compares early and late portions of the window to detect improvement.
    ///
    /// # Arguments
    ///
    /// * `window` - Window size to analyze
    /// * `threshold` - Minimum relative improvement to consider converging
    #[must_use]
    pub fn is_converging(&self, window: usize, threshold: f32) -> bool {
        if self.measurements.len() < window {
            return false;
        }

        let half = window / 2;
        let values: Vec<f32> = self
            .measurements
            .iter()
            .rev()
            .take(window)
            .map(|m| m.loss)
            .collect();

        if values.len() < window {
            return false;
        }

        // Compare first half to second half
        let early_mean: f32 = values[half..].iter().sum::<f32>() / half as f32;
        let late_mean: f32 = values[..half].iter().sum::<f32>() / half as f32;

        // Converging if late loss is significantly lower
        early_mean > late_mean && (early_mean - late_mean) / early_mean > threshold
    }

    /// Detect loss anomalies (spikes or divergence).
    ///
    /// # Arguments
    ///
    /// * `window` - Window size for baseline calculation
    #[must_use]
    pub fn detect_anomalies(&self, window: usize) -> Vec<LossAnomaly> {
        let mut anomalies = Vec::new();

        if self.measurements.len() < window + 1 {
            return anomalies;
        }

        // Detect spikes
        let stats = match self.compute_statistics(Some(window)) {
            Some(s) => s,
            None => return anomalies,
        };

        // Check most recent measurement for spike
        if let Some(latest) = self.measurements.back() {
            let z_score = (latest.loss - stats.mean) / stats.std_dev;
            if z_score > self.config.spike_threshold {
                anomalies.push(LossAnomaly::Spike {
                    step: latest.step,
                    magnitude: z_score,
                });
            }
        }

        // Detect divergence (sustained increase)
        if self.measurements.len() >= window * 2 {
            let early_window = window;
            let late_window = window;

            let early_start = self.measurements.len().saturating_sub(window * 2);
            let early_values: Vec<f32> = self
                .measurements
                .iter()
                .skip(early_start)
                .take(early_window)
                .map(|m| m.loss)
                .collect();

            let late_values: Vec<f32> = self
                .measurements
                .iter()
                .rev()
                .take(late_window)
                .map(|m| m.loss)
                .collect();

            if !early_values.is_empty() && !late_values.is_empty() {
                let early_mean: f32 = early_values.iter().sum::<f32>() / early_values.len() as f32;
                let late_mean: f32 = late_values.iter().sum::<f32>() / late_values.len() as f32;

                if late_mean > early_mean * self.config.divergence_threshold {
                    if let Some(latest) = self.measurements.back() {
                        anomalies.push(LossAnomaly::Divergence {
                            step: latest.step,
                            rate: late_mean / early_mean,
                        });
                    }
                }
            }
        }

        anomalies
    }

    /// Get the loss improvement rate over a window.
    ///
    /// Returns the relative change in loss (negative = improvement).
    ///
    /// # Arguments
    ///
    /// * `window` - Window size to analyze
    #[must_use]
    pub fn improvement_rate(&self, window: usize) -> Option<f32> {
        if self.measurements.len() < window {
            return None;
        }

        let values: Vec<f32> = self
            .measurements
            .iter()
            .rev()
            .take(window)
            .map(|m| m.loss)
            .collect();

        if values.is_empty() {
            return None;
        }

        let first = values[values.len() - 1];
        let last = values[0];

        if first == 0.0 {
            return None;
        }

        Some((last - first) / first)
    }

    /// Get loss per phase summary.
    #[must_use]
    pub fn phase_summary(&self) -> Vec<(DeterministicPhase, LossStatistics)> {
        let phases = [
            DeterministicPhase::Warmup,
            DeterministicPhase::Full,
            DeterministicPhase::Predict,
            DeterministicPhase::Correct,
        ];

        phases
            .iter()
            .filter_map(|&phase| {
                let values: Vec<f32> = self
                    .measurements_for_phase(phase)
                    .iter()
                    .map(|m| m.loss)
                    .collect();

                if values.is_empty() {
                    None
                } else {
                    Some((phase, compute_stats(&values)))
                }
            })
            .collect()
    }

    /// Clear all measurements.
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.start_time = Instant::now();
        self.current_step = 0;
    }

    /// Get total number of measurements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Check if history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Export measurements to JSON.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.measurements)
    }

    /// Get the elapsed training time.
    #[must_use]
    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Helper function to compute statistics from a slice of values.
fn compute_stats(values: &[f32]) -> LossStatistics {
    let count = values.len();

    let mean = values.iter().sum::<f32>() / count as f32;

    let variance = values
        .iter()
        .map(|&v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f32>()
        / count as f32;

    let std_dev = variance.sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };

    LossStatistics {
        count,
        mean,
        variance,
        std_dev,
        min,
        max,
        median,
    }
}

// Make DeterministicPhase serializable for JSON export
impl Serialize for DeterministicPhase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for DeterministicPhase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "WARMUP" => Ok(DeterministicPhase::Warmup),
            "FULL" => Ok(DeterministicPhase::Full),
            "PREDICT" => Ok(DeterministicPhase::Predict),
            "CORRECT" => Ok(DeterministicPhase::Correct),
            _ => Err(serde::de::Error::custom(format!("Unknown phase: {s}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_recording() {
        let mut history = LossHistory::new(100);
        history.record(1.0, DeterministicPhase::Warmup);
        history.record(0.8, DeterministicPhase::Full);
        history.record(0.6, DeterministicPhase::Predict);

        assert_eq!(history.len(), 3);
        assert_eq!(history.current_loss(), Some(0.6));
    }

    #[test]
    fn test_statistics() {
        let mut history = LossHistory::new(100);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for (i, &v) in values.iter().enumerate() {
            history.record(v, DeterministicPhase::Full);
        }

        let stats = history.compute_statistics(None).unwrap();
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_rolling_average() {
        let mut history = LossHistory::new(100);
        for i in 1..=10 {
            history.record(i as f32, DeterministicPhase::Full);
        }

        let avg = history.rolling_average(5).unwrap();
        // Last 5 values are 6, 7, 8, 9, 10; average = 8.0
        assert!((avg - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_convergence_detection() {
        let mut history = LossHistory::new(100);

        // Decreasing loss (converging)
        for i in (1..=20).rev() {
            history.record(i as f32, DeterministicPhase::Full);
        }

        assert!(history.is_converging(20, 0.1));
    }

    #[test]
    fn test_spike_detection() {
        let mut history = LossHistory::new(100);

        // Normal values around 1.0
        for _ in 0..20 {
            history.record(1.0, DeterministicPhase::Full);
        }

        // Add a spike
        history.record(10.0, DeterministicPhase::Predict);

        let anomalies = history.detect_anomalies(20);
        assert!(!anomalies.is_empty());

        if let LossAnomaly::Spike { magnitude, .. } = anomalies[0] {
            assert!(magnitude > 3.0);
        } else {
            panic!("Expected spike anomaly");
        }
    }

    #[test]
    fn test_divergence_detection() {
        let mut history = LossHistory::new(100);

        // First window: losses around 1.0
        for _ in 0..20 {
            history.record(1.0, DeterministicPhase::Full);
        }

        // Second window: losses around 2.0 (diverging)
        for _ in 0..20 {
            history.record(2.0, DeterministicPhase::Predict);
        }

        let anomalies = history.detect_anomalies(20);
        assert!(!anomalies.is_empty());

        let has_divergence = anomalies
            .iter()
            .any(|a| matches!(a, LossAnomaly::Divergence { .. }));
        assert!(has_divergence);
    }

    #[test]
    fn test_phase_filtering() {
        let mut history = LossHistory::new(100);

        history.record(1.0, DeterministicPhase::Warmup);
        history.record(2.0, DeterministicPhase::Full);
        history.record(3.0, DeterministicPhase::Predict);
        history.record(4.0, DeterministicPhase::Full);

        let full_phase = history.measurements_for_phase(DeterministicPhase::Full);
        assert_eq!(full_phase.len(), 2);
        assert_eq!(full_phase[0].loss, 2.0);
        assert_eq!(full_phase[1].loss, 4.0);
    }

    #[test]
    fn test_improvement_rate() {
        let mut history = LossHistory::new(100);

        // Decreasing from 10.0 to 5.0 over 10 steps
        for i in 0..10 {
            let loss = 10.0 - i as f32 * 0.5;
            history.record(loss, DeterministicPhase::Full);
        }

        let rate = history.improvement_rate(10).unwrap();
        // First = 10.0, Last = 5.5, rate = (5.5-10.0)/10.0 = -0.45
        assert!(rate < 0.0); // Negative = improving
        assert!((rate + 0.45).abs() < 0.01);
    }

    #[test]
    fn test_max_history_limit() {
        let mut history = LossHistory::new(10);

        // Add more than max
        for i in 0..20 {
            history.record(i as f32, DeterministicPhase::Full);
        }

        // Should only keep last 10
        assert_eq!(history.len(), 10);
        assert_eq!(history.current_loss(), Some(19.0));
    }

    #[test]
    fn test_json_export() {
        let mut history = LossHistory::new(100);
        history.record(1.0, DeterministicPhase::Warmup);
        history.record(0.5, DeterministicPhase::Full);

        let json = history.to_json().unwrap();
        assert!(json.contains("WARMUP"));
        assert!(json.contains("FULL"));
    }
}

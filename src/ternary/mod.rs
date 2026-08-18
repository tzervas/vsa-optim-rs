//! Ternary math acceleration for gradient operations.
//!
//! This module provides memory-efficient gradient accumulation using ternary
//! representation {-1, 0, +1}:
//! - Multiplications become sign flips or zeros
//! - Additions remain additions
//! - Memory reduced by ~10x (2 bits vs 32 bits)
//!
//! # Key Insight
//!
//! While model weights can be ternary, gradients need more precision for
//! convergence. However, we can use ternary representations for intermediate
//! computations and restore precision for updates.
//!
//! # Example
//!
//! ```ignore
//! use vsa_optim_rs::ternary::{TernaryGradientAccumulator, ternary_quantize_stochastic};
//! use vsa_optim_rs::TernaryConfig;
//!
//! let mut accumulator = TernaryGradientAccumulator::new(&param_shapes, TernaryConfig::default());
//!
//! // Accumulate gradients
//! accumulator.accumulate(&gradients)?;
//!
//! // Get full-precision result
//! let accumulated = accumulator.get_accumulated()?;
//! ```

mod accumulator;

pub use accumulator::{OptimizerStats, TernaryGradientAccumulator, TernaryOptimizerWrapper};

use candle_core::{DType, Tensor};
use rand::Rng;
use trit_vsa::{PackedTritVec, Trit};

use crate::error::Result;

/// Quantize tensor to ternary using stochastic rounding.
///
/// Stochastic rounding preserves gradient information in expectation.
/// A value like 0.3 has 30% chance of being +1 and 70% chance of being 0,
/// so the expected value equals the input. This enables unbiased gradient
/// accumulation even with ternary storage.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `threshold` - Quantization threshold (uses mean abs if None)
///
/// # Returns
///
/// Tuple of (ternary tensor, scale factor).
#[allow(clippy::cast_possible_truncation)]
pub fn ternary_quantize_stochastic(x: &Tensor, threshold: Option<f32>) -> Result<(Tensor, f32)> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let flat = x_f32.flatten_all()?;
    let data: Vec<f32> = flat.to_vec1()?;

    // Compute threshold if not provided
    let threshold = threshold.unwrap_or_else(|| {
        if data.is_empty() {
            0.0
        } else {
            data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
        }
    });

    // Compute scale
    let scale = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if scale == 0.0 {
        return Ok((x.zeros_like()?, 0.0));
    }

    let mut rng = rand::rng();

    // Stochastic quantization
    let quantized: Vec<f32> = data
        .iter()
        .map(|&v| {
            let normalized = v / scale;
            let abs_norm = normalized.abs();

            // P(+1) = max(0, normalized), P(-1) = max(0, -normalized), P(0) = 1 - |normalized|
            let rand_val: f32 = rng.random();

            if rand_val < abs_norm {
                if normalized > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            }
        })
        .collect();

    let result = Tensor::from_vec(quantized, x.shape(), x.device())?;
    Ok((result, scale))
}

/// Quantize tensor to ternary using deterministic thresholding.
///
/// Deterministic quantization is faster and reproducible but introduces bias.
/// Used when speed matters more than unbiasedness.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `threshold` - Quantization threshold (uses mean abs if None)
///
/// # Returns
///
/// Tuple of (ternary tensor, scale factor).
#[allow(clippy::cast_possible_truncation)]
pub fn ternary_quantize_deterministic(x: &Tensor, threshold: Option<f32>) -> Result<(Tensor, f32)> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let flat = x_f32.flatten_all()?;
    let data: Vec<f32> = flat.to_vec1()?;

    // Compute threshold if not provided
    let threshold = threshold.unwrap_or_else(|| {
        if data.is_empty() {
            0.0
        } else {
            data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
        }
    });

    // Compute scale
    let scale = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if scale == 0.0 {
        return Ok((x.zeros_like()?, 0.0));
    }

    // Simple thresholding
    let quantized: Vec<f32> = data
        .iter()
        .map(|&v| {
            if v > threshold {
                1.0
            } else if v < -threshold {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    let result = Tensor::from_vec(quantized, x.shape(), x.device())?;
    Ok((result, scale))
}

/// Quantize tensor to packed ternary using stochastic rounding.
#[allow(clippy::cast_possible_truncation)]
pub fn ternary_quantize_stochastic_packed(
    x: &Tensor,
    _threshold: Option<f32>,
) -> Result<(PackedTritVec, f32)> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let flat = x_f32.flatten_all()?;
    let data: Vec<f32> = flat.to_vec1()?;

    // Compute scale
    let scale = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if scale == 0.0 {
        return Ok((PackedTritVec::new(data.len()), 0.0));
    }

    let mut rng = rand::rng();
    let mut packed = PackedTritVec::new(data.len());

    // Stochastic quantization
    for (i, &v) in data.iter().enumerate() {
        let normalized = v / scale;
        let abs_norm = normalized.abs();
        let rand_val: f32 = rng.random();

        let trit = if rand_val < abs_norm {
            if normalized > 0.0 {
                Trit::P
            } else {
                Trit::N
            }
        } else {
            Trit::Z
        };
        packed.set(i, trit);
    }

    Ok((packed, scale))
}

/// Calculate memory savings from ternary representation.
///
/// Full precision: 32 bits per element
/// Ternary: 2 bits per element + 32 bits scale per tensor
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn calculate_memory_savings(param_count: usize, num_tensors: usize) -> f32 {
    let full_bits = param_count * 32;
    let ternary_bits = param_count * 2 + num_tensors * 32;
    1.0 - (ternary_bits as f32 / full_bits as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ternary_quantize_stochastic() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, 1000, &device).unwrap();

        let (quantized, scale) = ternary_quantize_stochastic(&x, None).unwrap();
        assert!(scale > 0.0);

        let values: Vec<f32> = quantized.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(v == -1.0 || v == 0.0 || v == 1.0, "Unexpected value: {v}");
        }
    }

    #[test]
    fn test_ternary_quantize_deterministic() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, 1000, &device).unwrap();

        let (quantized, scale) = ternary_quantize_deterministic(&x, None).unwrap();
        assert!(scale > 0.0);

        let values: Vec<f32> = quantized.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(v == -1.0 || v == 0.0 || v == 1.0, "Unexpected value: {v}");
        }
    }

    #[test]
    fn test_ternary_quantize_zeros() {
        let device = Device::Cpu;
        let x = Tensor::zeros(100, DType::F32, &device).unwrap();

        let (quantized, scale) = ternary_quantize_deterministic(&x, None).unwrap();
        assert_eq!(scale, 0.0);

        let values: Vec<f32> = quantized.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_memory_savings() {
        let savings = calculate_memory_savings(1_000_000, 10);
        // Should save ~93.75% (1 - 2/32 ≈ 0.9375)
        assert!(
            savings > 0.9,
            "Expected > 90% savings, got {:.2}%",
            savings * 100.0
        );
    }

    #[test]
    fn test_stochastic_packed() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, 1000, &device).unwrap();

        let (packed, scale) = ternary_quantize_stochastic_packed(&x, None).unwrap();
        assert_eq!(packed.len(), 1000);
        assert!(scale > 0.0);
    }
}

//! VSA (Vector Symbolic Architecture) gradient compression.
//!
//! This module provides hyperdimensional computing operations for gradient compression:
//! - Random projection using Johnson-Lindenstrauss lemma
//! - Ternary quantization for extreme compression
//! - Bind/bundle operations for structured representations
//!
//! # Why VSA?
//!
//! Hyperdimensional computing enables efficient gradient approximation:
//! - Near-orthogonality: Random vectors are almost orthogonal in high dimensions
//! - Distributed representation: Information spread across all dimensions
//! - Noise tolerance: Robust to errors and approximation
//! - Efficient operations: Binding, bundling work element-wise
//!
//! # Example
//!
//! ```ignore
//! use vsa_optim_rs::vsa::{VSAGradientCompressor, hyperdimensional_bind, hyperdimensional_bundle};
//! use vsa_optim_rs::VSAConfig;
//!
//! // Create compressor
//! let compressor = VSAGradientCompressor::new(1_000_000, VSAConfig::default());
//!
//! // Compress gradients
//! let (compressed, metadata) = compressor.compress(&gradients)?;
//!
//! // Decompress
//! let reconstructed = compressor.decompress(&compressed, &metadata, &shapes)?;
//! ```
//!
//! # References
//!
//! - Kanerva (2009): Hyperdimensional Computing
//! - Rahimi et al. (2016): High-Dimensional Computing as a Nanoscalable Paradigm

mod compressor;

pub use compressor::{CompressionStats, GradientMetadata, VSAGradientCompressor};

use candle_core::Tensor;
use trit_vsa::{PackedTritVec, Trit};

use crate::error::{OptimError, Result};

/// Bind two hypervectors using element-wise multiplication.
///
/// Binding creates a new vector that's dissimilar to both inputs but can be
/// unbound to retrieve either one. This is the key operation for creating
/// structured representations in hyperdimensional space.
///
/// For ternary vectors, binding uses trit-vsa's bind operation.
///
/// # Arguments
///
/// * `a` - First hypervector
/// * `b` - Second hypervector
///
/// # Returns
///
/// Bound hypervector with same dimension as inputs.
///
/// # Errors
///
/// Returns error if vectors have different dimensions.
pub fn hyperdimensional_bind(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dims() != b.dims() {
        return Err(OptimError::DimensionMismatch {
            expected: a.dims().iter().product(),
            actual: b.dims().iter().product(),
        });
    }
    Ok(a.mul(b)?)
}

/// Bind two packed ternary vectors.
///
/// Uses trit-vsa's native bind operation for efficiency.
pub fn hyperdimensional_bind_ternary(
    a: &PackedTritVec,
    b: &PackedTritVec,
) -> Result<PackedTritVec> {
    if a.len() != b.len() {
        return Err(OptimError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(trit_vsa::vsa::bind(a, b))
}

/// Bundle multiple hypervectors into one superposition.
///
/// Bundling creates a superposition that's similar to all inputs. The result
/// can be queried to retrieve any bundled vector. This enables storing multiple
/// gradient directions in a single compressed vector.
///
/// # Arguments
///
/// * `vectors` - Slice of hypervectors to bundle
/// * `weights` - Optional importance weights for each vector
///
/// # Returns
///
/// Bundled hypervector (same dimension as inputs).
///
/// # Errors
///
/// Returns error if vectors is empty or vectors have different dimensions.
pub fn hyperdimensional_bundle(vectors: &[Tensor], weights: Option<&[f32]>) -> Result<Tensor> {
    if vectors.is_empty() {
        return Err(OptimError::EmptyInput(
            "Cannot bundle empty list".to_string(),
        ));
    }

    let dim = vectors[0].dims();
    for v in vectors.iter().skip(1) {
        if v.dims() != dim {
            return Err(OptimError::ShapeMismatch {
                expected: dim.to_vec(),
                actual: v.dims().to_vec(),
            });
        }
    }

    let n = vectors.len();
    let weights: Vec<f32> = weights.map(|w| w.to_vec()).unwrap_or_else(|| vec![1.0; n]);

    if weights.len() != n {
        return Err(OptimError::DimensionMismatch {
            expected: n,
            actual: weights.len(),
        });
    }

    // Weighted sum
    let mut result = vectors[0].zeros_like()?;
    for (v, &w) in vectors.iter().zip(weights.iter()) {
        let weighted = (v * w as f64)?;
        result = result.add(&weighted)?;
    }

    // Normalize
    Ok((result / n as f64)?)
}

/// Bundle multiple packed ternary vectors.
///
/// Uses trit-vsa's native bundle operation (majority voting).
pub fn hyperdimensional_bundle_ternary(vectors: &[&PackedTritVec]) -> Result<PackedTritVec> {
    if vectors.is_empty() {
        return Err(OptimError::EmptyInput(
            "Cannot bundle empty list".to_string(),
        ));
    }
    Ok(trit_vsa::vsa::bundle_many(vectors))
}

/// Quantize tensor to ternary {-1, 0, +1}.
///
/// Ternary representation enables extremely fast operations using only
/// additions/subtractions (no multiplications). The scale factor preserves
/// magnitude information for accurate reconstruction.
///
/// # Arguments
///
/// * `x` - Input tensor
/// * `scale` - Optional scale factor (computed as mean abs if None)
///
/// # Returns
///
/// Tuple of (quantized tensor, scale factor).
#[allow(clippy::cast_possible_truncation)]
pub fn ternary_quantize(x: &Tensor, scale: Option<f32>) -> Result<(Tensor, f32)> {
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let flat = x_f32.flatten_all()?;
    let data: Vec<f32> = flat.to_vec1()?;

    // Compute scale if not provided
    let scale = scale.unwrap_or_else(|| {
        if data.is_empty() {
            0.0
        } else {
            data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
        }
    });

    if scale == 0.0 {
        return Ok((x.zeros_like()?, 0.0));
    }

    // Quantize to {-1, 0, +1}
    let quantized: Vec<f32> = data
        .iter()
        .map(|&v| {
            if v > scale {
                1.0
            } else if v < -scale {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    let result = Tensor::from_vec(quantized, x.shape(), x.device())?;
    Ok((result, scale))
}

/// Quantize tensor to packed ternary vector.
///
/// More memory-efficient than tensor representation.
#[allow(clippy::cast_possible_truncation)]
pub fn ternary_quantize_packed(x: &Tensor, scale: Option<f32>) -> Result<(PackedTritVec, f32)> {
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let flat = x_f32.flatten_all()?;
    let data: Vec<f32> = flat.to_vec1()?;

    // Compute scale if not provided
    let scale = scale.unwrap_or_else(|| {
        if data.is_empty() {
            0.0
        } else {
            data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
        }
    });

    let mut packed = PackedTritVec::new(data.len());

    if scale == 0.0 {
        return Ok((packed, 0.0));
    }

    // Quantize to {-1, 0, +1}
    for (i, &v) in data.iter().enumerate() {
        let trit = if v > scale {
            Trit::P
        } else if v < -scale {
            Trit::N
        } else {
            Trit::Z
        };
        packed.set(i, trit);
    }

    Ok((packed, scale))
}

/// Dequantize packed ternary vector back to tensor.
#[allow(clippy::cast_precision_loss)]
pub fn ternary_dequantize_packed(
    packed: &PackedTritVec,
    scale: f32,
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<Tensor> {
    let data: Vec<f32> = (0..packed.len())
        .map(|i| packed.get(i).value() as f32 * scale)
        .collect();

    Ok(Tensor::from_vec(data, shape, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_hyperdimensional_bind_shape() {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, 100, &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, 100, &device).unwrap();

        let result = hyperdimensional_bind(&a, &b).unwrap();
        assert_eq!(result.dims(), a.dims());
    }

    #[test]
    fn test_hyperdimensional_bind_dimension_mismatch() {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, 100, &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, 50, &device).unwrap();

        assert!(hyperdimensional_bind(&a, &b).is_err());
    }

    #[test]
    fn test_hyperdimensional_bundle_shape() {
        let device = Device::Cpu;
        let vectors: Vec<Tensor> = (0..5)
            .map(|_| Tensor::randn(0.0f32, 1.0, 100, &device).unwrap())
            .collect();

        let result = hyperdimensional_bundle(&vectors, None).unwrap();
        assert_eq!(result.dims(), vectors[0].dims());
    }

    #[test]
    fn test_hyperdimensional_bundle_empty() {
        let result = hyperdimensional_bundle(&[], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ternary_quantize_values() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, 1000, &device).unwrap();

        let (quantized, scale) = ternary_quantize(&x, None).unwrap();
        assert!(scale > 0.0);

        let values: Vec<f32> = quantized.flatten_all().unwrap().to_vec1().unwrap();
        for v in values {
            assert!(v == -1.0 || v == 0.0 || v == 1.0, "Unexpected value: {v}");
        }
    }

    #[test]
    fn test_ternary_quantize_zeros() {
        let device = Device::Cpu;
        let x = Tensor::zeros(100, candle_core::DType::F32, &device).unwrap();

        let (quantized, scale) = ternary_quantize(&x, None).unwrap();
        assert_eq!(scale, 0.0);

        let values: Vec<f32> = quantized.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ternary_quantize_packed_roundtrip() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (10, 10), &device).unwrap();

        let (packed, scale) = ternary_quantize_packed(&x, None).unwrap();
        assert_eq!(packed.len(), 100);

        let restored = ternary_dequantize_packed(&packed, scale, &[10, 10], &device).unwrap();
        assert_eq!(restored.dims(), x.dims());
    }

    #[test]
    fn test_bind_ternary() {
        let mut a = PackedTritVec::new(100);
        let mut b = PackedTritVec::new(100);

        for i in 0..100 {
            a.set(i, if i % 3 == 0 { Trit::P } else { Trit::Z });
            b.set(i, if i % 2 == 0 { Trit::N } else { Trit::P });
        }

        let bound = hyperdimensional_bind_ternary(&a, &b).unwrap();
        assert_eq!(bound.len(), 100);
    }
}

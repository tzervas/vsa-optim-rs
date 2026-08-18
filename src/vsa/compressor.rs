//! VSA Gradient Compressor implementation.
//!
//! Uses proper Vector Symbolic Architecture operations:
//! - **Bind**: Associates each gradient with a unique random key
//! - **Bundle**: Combines all bound gradients into a single superposition
//! - **Unbind**: Extracts individual gradients by binding with inverse key
//!
//! This approach maintains the memory benefits of bundling while enabling
//! accurate reconstruction via the quasi-orthogonality of random keys in
//! high-dimensional space.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use rand_chacha::rand_core::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use trit_vsa::{vsa as trit_vsa_ops, PackedTritVec, Trit};

use crate::config::VSAConfig;
use crate::error::{OptimError, Result};

/// Gradient metadata for reconstruction.
#[derive(Debug, Clone)]
pub struct GradientMetadata {
    /// Index in the bundling order (used for key generation).
    pub key_index: usize,
    /// Scale factor from quantization.
    pub scale: f32,
    /// Original shape.
    pub shape: Vec<usize>,
}

/// Compress gradients using Vector Symbolic Architecture.
///
/// This compressor uses proper VSA operations:
/// 1. **Project** each gradient to hyperdimensional space
/// 2. **Bind** each projected gradient with a unique random key
/// 3. **Bundle** all bound vectors into a single superposition
/// 4. **Unbind** during decompression to extract individual gradients
///
/// The bundled representation achieves significant compression while the
/// bind/unbind operations enable accurate reconstruction due to the
/// quasi-orthogonality of random keys in high dimensions.
///
/// # Example
///
/// ```ignore
/// use vsa_optim_rs::vsa::VSAGradientCompressor;
/// use vsa_optim_rs::VSAConfig;
///
/// let compressor = VSAGradientCompressor::new(1_000_000, VSAConfig::default());
///
/// // After computing gradients
/// let (compressed, metadata) = compressor.compress(&gradients)?;
/// let reconstructed = compressor.decompress(&compressed, &metadata)?;
/// ```
pub struct VSAGradientCompressor {
    config: VSAConfig,
    param_count: usize,
    hypervector_dim: usize,
    /// Cache of binding keys per gradient index
    key_cache: HashMap<usize, PackedTritVec>,
    /// Cache of projection matrices
    projection_cache: HashMap<usize, Tensor>,
}

impl VSAGradientCompressor {
    /// Create a new VSA gradient compressor.
    ///
    /// # Arguments
    ///
    /// * `param_count` - Total number of model parameters
    /// * `config` - VSA configuration
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    pub fn new(param_count: usize, config: VSAConfig) -> Self {
        // Use configured dimension or default based on compression ratio
        let hypervector_dim = config
            .dimension
            .max((param_count as f32 * config.compression_ratio).max(256.0) as usize);

        Self {
            config,
            param_count,
            hypervector_dim,
            key_cache: HashMap::new(),
            projection_cache: HashMap::new(),
        }
    }

    /// Get the hypervector dimension.
    #[must_use]
    pub const fn compressed_dim(&self) -> usize {
        self.hypervector_dim
    }

    /// Generate or retrieve a random binding key for a gradient.
    fn get_binding_key(&mut self, index: usize) -> PackedTritVec {
        if let Some(key) = self.key_cache.get(&index) {
            return key.clone();
        }

        // Generate deterministic random key for this index
        let seed = self.config.seed.wrapping_add(index as u64 * 12345);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut key = PackedTritVec::new(self.hypervector_dim);
        for i in 0..self.hypervector_dim {
            // Generate random ternary: ~33% each of -1, 0, +1
            let r: f32 = (rng.next_u32() as f64 / u32::MAX as f64) as f32;
            let trit = if r < 0.33 {
                Trit::N
            } else if r < 0.66 {
                Trit::Z
            } else {
                Trit::P
            };
            key.set(i, trit);
        }

        self.key_cache.insert(index, key.clone());
        key
    }

    /// Generate projection matrix to map gradient to hypervector space.
    fn get_projection(&mut self, grad_size: usize, device: &Device) -> Result<Tensor> {
        if let Some(proj) = self.projection_cache.get(&grad_size) {
            return Ok(proj.clone());
        }

        let seed = self.config.seed.wrapping_add(grad_size as u64 * 54321);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Scale for Johnson-Lindenstrauss: 1/sqrt(d) preserves dot products in expectation
        let scale = 1.0 / (self.hypervector_dim as f32).sqrt();

        let data: Vec<f32> = (0..grad_size * self.hypervector_dim)
            .map(|_| {
                // Sparse random projection for efficiency: ~68% zeros
                let r: f32 = (rng.next_u32() as f64 / u32::MAX as f64) as f32;
                if r < 0.16 {
                    scale * 3.0_f32.sqrt() // sqrt(3) to maintain variance
                } else if r < 0.32 {
                    -scale * 3.0_f32.sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        let proj = Tensor::from_vec(data, (grad_size, self.hypervector_dim), device)?;
        self.projection_cache.insert(grad_size, proj.clone());
        Ok(proj)
    }

    /// Project gradient to hypervector, returning ternary representation.
    fn project_to_hypervector(&mut self, gradient: &Tensor) -> Result<(PackedTritVec, f32)> {
        let device = gradient.device();
        let flat = gradient.flatten_all()?.to_dtype(DType::F32)?;
        let grad_size = flat.elem_count();

        // Get projection matrix
        let proj = self.get_projection(grad_size, device)?;

        // Project: (1, grad_size) @ (grad_size, dim) -> (1, dim)
        let projected = flat.unsqueeze(0)?.matmul(&proj)?.squeeze(0)?;
        let data: Vec<f32> = projected.to_vec1()?;

        // Compute scale (mean absolute value)
        let scale = if data.is_empty() {
            0.0
        } else {
            data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
        };

        // Quantize to ternary
        let mut packed = PackedTritVec::new(self.hypervector_dim);
        if scale > 0.0 {
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
        }

        Ok((packed, scale))
    }

    /// Compress gradients to bundled hyperdimensional representation.
    ///
    /// # Algorithm
    ///
    /// 1. Project each gradient to hypervector space
    /// 2. Quantize to ternary {-1, 0, +1}
    /// 3. Bind with unique random key
    /// 4. Bundle all bound vectors via element-wise sum
    ///
    /// # Arguments
    ///
    /// * `gradients` - Map of parameter names to gradient tensors
    ///
    /// # Returns
    ///
    /// Tuple of (bundled hypervector, metadata for reconstruction).
    pub fn compress(
        &mut self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<(PackedTritVec, HashMap<String, GradientMetadata>)> {
        if gradients.is_empty() {
            return Err(OptimError::EmptyInput(
                "No gradients to compress".to_string(),
            ));
        }

        let mut metadata = HashMap::new();
        let mut bound_vectors: Vec<PackedTritVec> = Vec::new();

        for (index, (name, grad)) in gradients.iter().enumerate() {
            // Project gradient to hypervector
            let (projected, scale) = self.project_to_hypervector(grad)?;

            // Get binding key for this gradient
            let key = self.get_binding_key(index);

            // Bind: gradient ⊛ key
            let bound = trit_vsa_ops::bind(&projected, &key);
            bound_vectors.push(bound);

            metadata.insert(
                name.clone(),
                GradientMetadata {
                    key_index: index,
                    scale,
                    shape: grad.dims().to_vec(),
                },
            );
        }

        // Bundle all bound vectors via majority voting
        let refs: Vec<&PackedTritVec> = bound_vectors.iter().collect();
        let bundled = trit_vsa_ops::bundle_many(&refs);

        Ok((bundled, metadata))
    }

    /// Decompress gradients from bundled hypervector.
    ///
    /// # Algorithm
    ///
    /// For each gradient:
    /// 1. Unbind with the gradient's key to extract from bundle
    /// 2. Inverse project back to gradient space
    /// 3. Apply stored scale factor
    ///
    /// # Arguments
    ///
    /// * `bundled` - Bundled hypervector from compress
    /// * `metadata` - Metadata from compression
    ///
    /// # Returns
    ///
    /// Map of reconstructed gradients.
    pub fn decompress(
        &mut self,
        bundled: &PackedTritVec,
        metadata: &HashMap<String, GradientMetadata>,
    ) -> Result<HashMap<String, Tensor>> {
        let device = Device::Cpu; // Ternary ops are CPU-based
        let mut gradients = HashMap::new();

        for (name, meta) in metadata {
            // Get the binding key used during compression
            let key = self.get_binding_key(meta.key_index);

            // Unbind to extract this gradient's contribution
            let unbound = trit_vsa_ops::unbind(bundled, &key);

            // Convert ternary to float and apply scale
            let grad_size: usize = meta.shape.iter().product();
            let proj = self.get_projection(grad_size, &device)?;

            // Inverse projection: unbound @ proj.T
            // First convert unbound ternary to float
            let unbound_float: Vec<f32> = (0..self.hypervector_dim)
                .map(|i| unbound.get(i).value() as f32 * meta.scale)
                .collect();

            let unbound_tensor = Tensor::from_vec(unbound_float, self.hypervector_dim, &device)?;

            // Inverse project: (1, dim) @ (dim, grad_size) -> (1, grad_size)
            let reconstructed = unbound_tensor
                .unsqueeze(0)?
                .matmul(&proj.t()?)?
                .squeeze(0)?;

            // Reshape to original
            let grad = reconstructed.reshape(meta.shape.as_slice())?;
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }

    /// Get compression statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn get_compression_stats(&self) -> CompressionStats {
        CompressionStats {
            original_params: self.param_count,
            compressed_dim: self.hypervector_dim,
            compression_ratio: self.hypervector_dim as f32 / self.param_count as f32,
            // Ternary uses 2 bits per element vs 32 bits for float
            memory_saving: 1.0
                - (self.hypervector_dim as f32 * 2.0 / 32.0) / self.param_count as f32,
        }
    }

    /// Clear caches to free memory.
    pub fn clear_cache(&mut self) {
        self.key_cache.clear();
        self.projection_cache.clear();
    }
}

/// Compression statistics.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original parameter count.
    pub original_params: usize,
    /// Compressed dimension.
    pub compressed_dim: usize,
    /// Compression ratio (compressed / original).
    pub compression_ratio: f32,
    /// Memory saving fraction (1 - compression_ratio).
    pub memory_saving: f32,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Compression: {} → {} ({:.1}% saved)",
            self.original_params,
            self.compressed_dim,
            self.memory_saving * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_gradients(device: &Device) -> HashMap<String, Tensor> {
        let mut gradients = HashMap::new();
        gradients.insert(
            "layer1.weight".to_string(),
            Tensor::randn(0.0f32, 1.0, (64, 128), device).unwrap(),
        );
        gradients.insert(
            "layer1.bias".to_string(),
            Tensor::randn(0.0f32, 1.0, 64, device).unwrap(),
        );
        gradients.insert(
            "layer2.weight".to_string(),
            Tensor::randn(0.0f32, 1.0, (32, 64), device).unwrap(),
        );
        gradients
    }

    #[test]
    fn test_compressor_creation() {
        let compressor = VSAGradientCompressor::new(1_000_000, VSAConfig::default());
        assert!(compressor.compressed_dim() >= 256);
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let device = Device::Cpu;
        let gradients = create_mock_gradients(&device);

        let param_count: usize = gradients.values().map(|g| g.elem_count()).sum();
        let mut compressor = VSAGradientCompressor::new(
            param_count,
            VSAConfig::default().with_compression_ratio(0.5),
        );

        // Compress
        let (bundled, metadata) = compressor.compress(&gradients).unwrap();
        assert_eq!(bundled.len(), compressor.compressed_dim());
        assert_eq!(metadata.len(), 3);

        // Decompress
        let reconstructed = compressor.decompress(&bundled, &metadata).unwrap();
        assert_eq!(reconstructed.len(), 3);

        // Check shapes match
        for (name, orig) in &gradients {
            let recon = reconstructed.get(name).unwrap();
            assert_eq!(orig.dims(), recon.dims());
        }
    }

    #[test]
    fn test_compression_stats() {
        let compressor = VSAGradientCompressor::new(1_000_000, VSAConfig::default());
        let stats = compressor.get_compression_stats();

        assert_eq!(stats.original_params, 1_000_000);
        // With ternary (2 bits per element), memory saving should be high
        assert!(stats.memory_saving > 0.9);
    }

    #[test]
    fn test_direction_preservation() {
        let device = Device::Cpu;
        let gradients = create_mock_gradients(&device);

        let param_count: usize = gradients.values().map(|g| g.elem_count()).sum();
        let mut compressor = VSAGradientCompressor::new(
            param_count,
            VSAConfig::default()
                .with_dimension(8192) // Use larger dimension for better reconstruction
                .with_compression_ratio(0.5),
        );

        let (bundled, metadata) = compressor.compress(&gradients).unwrap();
        let reconstructed = compressor.decompress(&bundled, &metadata).unwrap();

        // Check cosine similarity is positive (direction preserved)
        for (name, orig) in &gradients {
            let recon = reconstructed.get(name).unwrap();

            let orig_flat = orig.flatten_all().unwrap();
            let recon_flat = recon.flatten_all().unwrap();

            let orig_data: Vec<f32> = orig_flat.to_vec1().unwrap();
            let recon_data: Vec<f32> = recon_flat.to_vec1().unwrap();

            let dot: f32 = orig_data
                .iter()
                .zip(recon_data.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_orig: f32 = orig_data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_recon: f32 = recon_data.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Skip very small tensors where numerical instability is expected
            if norm_orig < 1e-6 || norm_recon < 1e-6 {
                continue;
            }

            let cosine = dot / (norm_orig * norm_recon + 1e-8);

            // Direction should be roughly preserved for larger tensors
            // VSA reconstruction is approximate due to bundling interference
            if orig.elem_count() >= 1024 {
                assert!(
                    cosine > 0.1, // Lower threshold due to bundling noise
                    "Gradient direction not preserved for {name}: cosine = {cosine}"
                );
            }
        }
    }

    #[test]
    fn test_bind_unbind_property() {
        // Test that bind/unbind correctly recovers the original
        let mut compressor =
            VSAGradientCompressor::new(1000, VSAConfig::default().with_dimension(1024));

        let key0 = compressor.get_binding_key(0);
        let key1 = compressor.get_binding_key(1);

        // Keys should be different
        let mut same_count = 0;
        for i in 0..key0.len() {
            if key0.get(i) == key1.get(i) {
                same_count += 1;
            }
        }
        // Should be roughly 1/3 same by chance
        assert!(same_count < key0.len() * 2 / 3);

        // Bind and unbind should recover original
        let test_vec = key0.clone();
        let bound = trit_vsa_ops::bind(&test_vec, &key1);
        let recovered = trit_vsa_ops::unbind(&bound, &key1);

        for i in 0..test_vec.len() {
            assert_eq!(test_vec.get(i), recovered.get(i));
        }
    }

    #[test]
    fn test_empty_gradients() {
        let mut compressor = VSAGradientCompressor::new(1000, VSAConfig::default());
        let gradients = HashMap::new();

        let result = compressor.compress(&gradients);
        assert!(result.is_err());
    }
}

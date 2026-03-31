//! Rotational Quantization (RoQ) for embeddings
//!
//! 8-bit per-sample quantization with FWHT-based rotation.
//! Stores (codes, scale, offset, norm_sq) per vector for fast distance estimation.

use crate::quantization::fwht::FastWalshHadamard;
use wide::f32x8;

/// Configuration for Rotational Quantization
#[derive(Clone, Debug)]
pub struct RoQConfig {
    /// Input dimension
    pub dim: usize,
    /// Number of FWHT rotation rounds (default: 3)
    pub num_rounds: usize,
    /// FWHT block size, must be power of 2 (default: 256)
    pub block_size: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of bits per dimension (1 or 8, default: 8)
    pub num_bits: usize,
}

impl Default for RoQConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            num_rounds: 3,
            block_size: 256,
            seed: 42,
            num_bits: 8,
        }
    }
}

/// Quantized vector representation
#[derive(Clone)]
pub struct QuantizedVector {
    /// Quantized codes (padded_dim bytes for 8-bit, padded_dim/8 for 1-bit)
    pub codes: Vec<u8>,
    /// Sum of codes (precomputed for fast distance, 8-bit only)
    pub code_sum: u32,
    /// Scale factor (delta, 8-bit only)
    pub scale: f32,
    /// Offset (l, minimum value, 8-bit only)
    pub offset: f32,
    /// Squared L2 norm of original vector
    pub norm_sq: f32,
}

/// Rotational Quantizer with 8-bit or 1-bit per-sample quantization
pub struct RotationalQuantizer {
    config: RoQConfig,
    fwht: FastWalshHadamard,
    padded_dim: usize,
}

impl RotationalQuantizer {
    /// Create a new rotational quantizer
    pub fn new(config: RoQConfig) -> Self {
        assert!(config.num_bits == 1 || config.num_bits == 4 || config.num_bits == 8, "Only 1, 4, or 8-bit supported");
        let fwht = FastWalshHadamard::new(
            config.dim,
            config.num_rounds,
            config.block_size,
            config.seed,
        );
        let padded_dim = fwht.padded_dim();
        
        Self {
            config,
            fwht,
            padded_dim,
        }
    }
    
    /// Quantize a batch of vectors
    ///
    /// # Arguments
    /// * `vectors` - Flat array of vectors (N * dim)
    /// * `n` - Number of vectors
    ///
    /// # Returns
    /// Vector of quantized representations
    pub fn quantize(&self, vectors: &[f32], n: usize) -> Vec<QuantizedVector> {
        assert_eq!(vectors.len(), n * self.config.dim);
        
        // Compute original norms
        let norms_sq: Vec<f32> = (0..n)
            .map(|i| {
                let v = &vectors[i * self.config.dim..(i + 1) * self.config.dim];
                v.iter().map(|x| x * x).sum()
            })
            .collect();
        
        // Rotate
        let rotated = self.fwht.rotate(vectors, n);
        
        // Quantize each vector
        let mut result = Vec::with_capacity(n);
        
        for i in 0..n {
            let vec = &rotated[i * self.padded_dim..(i + 1) * self.padded_dim];
            
            if self.config.num_bits == 1 {
                // Binary Quantization
                // 1 if x >= 0, 0 otherwise
                // Pack 8 bits into u8
                let n_bytes = (self.padded_dim + 7) / 8;
                let mut codes = vec![0u8; n_bytes];
                
                for j in 0..self.padded_dim {
                    if vec[j] >= 0.0 {
                        let byte_idx = j / 8;
                        let bit_idx = 7 - (j % 8); // Big-endian packing matches numpy packbits
                        codes[byte_idx] |= 1 << bit_idx;
                    }
                }
                
                result.push(QuantizedVector {
                    codes,
                    code_sum: 0,
                    scale: 1.0, 
                    offset: 0.0,
                    norm_sq: norms_sq[i],
                });
            } else if self.config.num_bits == 4 {
                // 4-bit Quantization [0, 15]
                // Pack 2 vals per byte: (High << 4) | Low
                
                let (min_val, max_val) = vec.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
                     (min.min(x), max.max(x))
                });
                
                let range = max_val - min_val;
                let scale = if range > 1e-8 { range / 15.0 } else { 1.0 };
                let offset = min_val;
                
                // Quantize to 0..15 vals
                let quantized: Vec<u8> = vec.iter()
                    .map(|&x| {
                        let q = ((x - offset) / scale).round();
                        q.clamp(0.0, 15.0) as u8
                    })
                    .collect();
                
                let code_sum: u32 = quantized.iter().map(|&c| c as u32).sum();
                
                // Pack
                // Output size: ceil(dim / 2)
                // If dim is odd, last byte has only high nibble? or low?
                // Logic: High=2k, Low=2k+1.
                let n_packed = (self.padded_dim + 1) / 2;
                let mut codes = Vec::with_capacity(n_packed);
                
                for chunk in quantized.chunks(2) {
                    let high = chunk[0];
                    let low = if chunk.len() > 1 { chunk[1] } else { 0 };
                    codes.push((high << 4) | low);
                }
                
                result.push(QuantizedVector {
                    codes,
                    code_sum,
                    scale,
                    offset,
                    norm_sq: norms_sq[i],
                });
            } else {
                // 8-bit Quantization
                // Find min/max
                let (min_val, max_val) = vec.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
                    (min.min(x), max.max(x))
                });
                
                let range = max_val - min_val;
                let scale = if range > 1e-8 { range / 255.0 } else { 1.0 };
                let offset = min_val;
                
                // Quantize to u8
                let codes: Vec<u8> = vec.iter()
                    .map(|&x| {
                        let q = ((x - offset) / scale).round();
                        q.clamp(0.0, 255.0) as u8
                    })
                    .collect();
                
                let code_sum: u32 = codes.iter().map(|&c| c as u32).sum();
                
                result.push(QuantizedVector {
                    codes,
                    code_sum,
                    scale,
                    offset,
                    norm_sq: norms_sq[i],
                });
            }
        }
        
        result
    }
    
    /// Compute squared L2 distance between two quantized vectors
    #[inline]
    pub fn distance_sq(&self, a: &QuantizedVector, b: &QuantizedVector) -> f32 {
        if self.config.num_bits == 1 {
            // Hamming distance for binary
            // Typically returned as integer count, but returning f32 for API consistency
            self.hamming_distance(a, b) as f32
        } else {
            // Symmetric distance for 8-bit
            let inner = self.inner_product_estimate(a, b);
            let dist_sq = a.norm_sq + b.norm_sq - 2.0 * inner;
            dist_sq.max(0.0) // Clamp negative due to approximation error
        }
    }
    
    /// Compute Hamming distance (number of differing bits)
    #[inline]
    pub fn hamming_distance(&self, a: &QuantizedVector, b: &QuantizedVector) -> u32 {
        debug_assert_eq!(a.codes.len(), b.codes.len());
        
        let mut dist = 0;
        let n = a.codes.len();
        
        // Handle u64 chunks for speed (if possible, simplified byte loop here)
        // Compiler usually optimizes this loop to SIMD/popcount instructions
        for i in 0..n {
            let xor = a.codes[i] ^ b.codes[i];
            dist += xor.count_ones();
        }
        
        dist
    }
    
    #[inline]
    pub fn inner_product_estimate(&self, a: &QuantizedVector, b: &QuantizedVector) -> f32 {
        let d = self.padded_dim as f32;
        
        // Term 1: D * l_a * l_b
        let t1 = d * a.offset * b.offset;
        
        if self.config.num_bits == 8 {
             // 8-bit logic
             let t2 = a.offset * b.scale * (b.code_sum as f32);
             let t3 = b.offset * a.scale * (a.code_sum as f32);
             
             // Compute <c_a, c_b>
             // Use u32 accumulation to avoid expensive int->float conversions per element
             let mut dot_sum: u32 = 0;
             let n = a.codes.len();
             
             // Unroll 8x manually
             let mut i = 0;
             while i + 8 <= n {
                 dot_sum += (a.codes[i] as u32) * (b.codes[i] as u32);
                 dot_sum += (a.codes[i+1] as u32) * (b.codes[i+1] as u32);
                 dot_sum += (a.codes[i+2] as u32) * (b.codes[i+2] as u32);
                 dot_sum += (a.codes[i+3] as u32) * (b.codes[i+3] as u32);
                 dot_sum += (a.codes[i+4] as u32) * (b.codes[i+4] as u32);
                 dot_sum += (a.codes[i+5] as u32) * (b.codes[i+5] as u32);
                 dot_sum += (a.codes[i+6] as u32) * (b.codes[i+6] as u32);
                 dot_sum += (a.codes[i+7] as u32) * (b.codes[i+7] as u32);
                 i += 8;
             }
             
             while i < n {
                 dot_sum += (a.codes[i] as u32) * (b.codes[i] as u32);
                 i += 1;
             }
             
             let dot_prod = dot_sum as f32;
             let t4 = a.scale * b.scale * dot_prod;
             
             t1 + t2 + t3 + t4
             
        } else if self.config.num_bits == 4 {
             // 4-bit logic
             let t2 = a.offset * b.scale * (b.code_sum as f32);
             let t3 = b.offset * a.scale * (a.code_sum as f32);
             
             // Compute <c_a, c_b> (Unpacked dot product)
             let mut dot_sum: u32 = 0; // sum of u8 products fits in u32
             let n = a.codes.len(); // packed bytes
             
             for i in 0..n {
                 let ca = a.codes[i];
                 let cb = b.codes[i];
                 
                 let ca_h = (ca >> 4);
                 let ca_l = (ca & 0xF);
                 let cb_h = (cb >> 4);
                 let cb_l = (cb & 0xF);
                 
                 dot_sum += (ca_h as u32) * (cb_h as u32) + (ca_l as u32) * (cb_l as u32);
             }
             
             let dot_prod = dot_sum as f32;
             let t4 = a.scale * b.scale * dot_prod;
             
             t1 + t2 + t3 + t4
             
        } else {
             0.0 // 1-bit uses hamming
        }
    }
        

    
    /// SIMD-optimized u8 dot product
    #[inline]
    fn dot_u8(&self, a: &[u8], b: &[u8]) -> u32 {
        debug_assert_eq!(a.len(), b.len());
        
        let n = a.len();
        let mut sum: u32 = 0;
        
        // Process 8 elements at a time using u32 accumulation
        let mut i = 0;
        while i + 8 <= n {
            // Accumulate in u32 to avoid overflow
            for j in 0..8 {
                sum += (a[i + j] as u32) * (b[i + j] as u32);
            }
            i += 8;
        }
        
        // Remainder
        while i < n {
            sum += (a[i] as u32) * (b[i] as u32);
            i += 1;
        }
        
        sum
    }
    
    /// Find k nearest neighbors using quantized distances
    pub fn search(&self, query: &QuantizedVector, index: &[QuantizedVector], k: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = index.iter()
            .enumerate()
            .map(|(i, v)| (i, self.distance_sq(query, v)))
            .collect();
        
        // Partial sort for top-k
        let k = k.min(distances.len());
        distances.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        distances.truncate(k);
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        distances
    }
    
    /// Get padded dimension
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Compute MaxSim score between a set of query tokens and a set of document tokens
    /// 
    /// MaxSim(Q, D) = sum_i(max_j(Sim(q_i, d_j)))
    ///
    /// # Arguments
    /// * `query_tokens` - Vector of quantized tokens for the query
    /// * `doc_tokens` - Vector of quantized tokens for the document
    ///
    /// # Returns
    /// MaxSim score
    pub fn max_sim_score(&self, query_tokens: &[QuantizedVector], doc_tokens: &[QuantizedVector]) -> f32 {
        let mut total_score = 0.0;

        for q in query_tokens {
            let mut max_sim = f32::NEG_INFINITY;
            
            for d in doc_tokens {
                // Determine similarity.
                // For distance_sq: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
                // So <x,y> = (||x||^2 + ||y||^2 - distance_sq) / 2
                // We want to MAXIMIZE <x,y>.
                // This is equivalent to MINIMIZING distance_sq IF norms are constant?
                // No, norms vary per token.
                // So we must compute dot product estimate.
                
                let sim = if self.config.num_bits == 1 {
                    // 1-Bit Similarity Proxy: dim - 2 * HammingDist
                    // Hamming = 0 => Sim = dim
                    // Hamming = dim => Sim = -dim
                    let dist = self.hamming_distance(q, d);
                    (self.padded_dim as f32) - 2.0 * (dist as f32)
                } else {
                    // 8-Bit Inner Product Estimate
                    self.inner_product_estimate(q, d)
                };
                
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            
            total_score += max_sim;
        }
        
        total_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    
    #[test]
    fn test_quantize_roundtrip() {
        let config = RoQConfig {
            dim: 128,
            num_rounds: 3,
            block_size: 64,
            seed: 42,
            num_bits: 8,
        };
        let roq = RotationalQuantizer::new(config);
        
        // Random vector
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let v: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
        
        let qv = roq.quantize(&v, 1);
        
        assert_eq!(qv.len(), 1);
        assert_eq!(qv[0].codes.len(), roq.padded_dim());
    }

    #[test]
    fn test_binary_quantize() {
        let config = RoQConfig {
            dim: 128,
            num_rounds: 3,
            block_size: 64,
            seed: 42,
            num_bits: 1,
        };
        let roq = RotationalQuantizer::new(config); // padded_dim = 128
        
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let v: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
        
        let qv = roq.quantize(&v, 1);
        
        // 128 bits = 16 bytes
        assert_eq!(qv[0].codes.len(), 16);
        
        // Test packing correctness (simple case)
        // If vector all positive -> all 1s
        let pos: Vec<f32> = vec![1.0; 128];
        let q_pos = roq.quantize(&pos, 1);
        // FWHT of [1,1...] is [val, 0...]? 
        // With random signs, it spreads.
        // Can't easily predict codes without running FWHT logic.
        // But we can check length.
        assert_eq!(q_pos[0].codes.len(), 16);
    }
    
    #[test]
    fn test_recall() {
        let dim = 128;
        let n = 500;
        let n_queries = 20;
        let k = 10;
        
        let config = RoQConfig {
            dim,
            num_rounds: 3,
            block_size: 64,
            seed: 42,
            num_bits: 8,
        };
        let roq = RotationalQuantizer::new(config);
        
        // Generate random normalized data
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let mut data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        
        // Normalize
        for i in 0..n {
            let v = &mut data[i * dim..(i + 1) * dim];
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= norm; }
        }
        
        // Generate queries
        let mut queries: Vec<f32> = (0..n_queries * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        for i in 0..n_queries {
            let v = &mut queries[i * dim..(i + 1) * dim];
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= norm; }
        }
        
        // Quantize
        let index = roq.quantize(&data, n);
        let q_queries = roq.quantize(&queries, n_queries);
        
        // Compute recall
        let mut total_recall = 0.0;
        
        for qi in 0..n_queries {
            // Ground truth (brute force exact)
            let query = &queries[qi * dim..(qi + 1) * dim];
            let mut exact: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    let dist: f32 = query.iter().zip(v.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i, dist)
                })
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let truth: std::collections::HashSet<usize> = exact.iter().take(k).map(|(i, _)| *i).collect();
            
            // Quantized search
            let results = roq.search(&q_queries[qi], &index, k);
            let found: std::collections::HashSet<usize> = results.iter().map(|(i, _)| *i).collect();
            
            let recall = truth.intersection(&found).count() as f32 / k as f32;
            total_recall += recall;
        }
        
        let avg_recall = total_recall / n_queries as f32;
        println!("Average Recall@{}: {:.4}", k, avg_recall);
        
        assert!(avg_recall > 0.90, "Recall too low: {}", avg_recall);
    }

    #[test]
    fn test_binary_recall_correlated() {
        // Test binary recall with HIGH correlation data
        let dim = 128;
        let n = 200;
        let n_queries = 20;
        let k = 5;
        
        let config = RoQConfig {
            dim,
            num_rounds: 3,
            block_size: 64,
            seed: 42,
            num_bits: 1, // Binary
        };
        let roq = RotationalQuantizer::new(config);
        
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let mut data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        
        // Normalize data
        for i in 0..n {
            let v = &mut data[i * dim..(i + 1) * dim];
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= norm; }
        }
        
        // Generate queries = data[idx] + noise
        // Ensure high correlation
        let mut queries = Vec::with_capacity(n_queries * dim);
        let mut truth_indices = Vec::new();
        
        for _ in 0..n_queries {
            let idx = rng.gen_range(0..n);
            truth_indices.push(idx);
            
            let v = &data[idx * dim..(idx + 1) * dim];
            for &x in v {
                // Mix 90% signal, 10% noise
                let noise = rng.gen::<f32>() - 0.5;
                queries.push(0.9 * x + 0.1 * noise);
            }
        }
        
        // Normalize queries
        for i in 0..n_queries {
            let v = &mut queries[i * dim..(i + 1) * dim];
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= norm; }
        }
        
        let index = roq.quantize(&data, n);
        let q_queries = roq.quantize(&queries, n_queries);
        
        let mut hits = 0;
        for i in 0..n_queries {
            // We just check if top-1 is the source vector (since correlation is high)
            let results = roq.search(&q_queries[i], &index, k);
            if results.iter().any(|(idx, _)| *idx == truth_indices[i]) {
                hits += 1;
            }
        }
        
        let accuracy = hits as f32 / n_queries as f32;
        println!("Binary Top-{} Recall: {:.4}", k, accuracy);
        assert!(accuracy > 0.80, "Binary accuracy too low: {}", accuracy);
    }
}

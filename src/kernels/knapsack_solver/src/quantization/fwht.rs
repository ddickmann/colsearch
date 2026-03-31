//! Fast Walsh-Hadamard Transform (FWHT) with SIMD optimization
//!
//! Provides O(D log D) pseudorandom rotation for vector quantization.
//! Uses blocked FWHT with random sign flips and permutations.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use wide::f32x8;

/// Fast Walsh-Hadamard Transform for pseudorandom rotation
pub struct FastWalshHadamard {
    dim: usize,
    padded_dim: usize,
    block_size: usize,
    num_rounds: usize,
    /// Random signs for each round: Vec<Vec<f32>> where inner is padded_dim
    signs: Vec<Vec<f32>>,
    /// Random permutations for each round
    permutations: Vec<Vec<usize>>,
}

impl FastWalshHadamard {
    /// Create a new FWHT rotator
    ///
    /// # Arguments
    /// * `dim` - Input vector dimension
    /// * `num_rounds` - Number of rotation rounds (default 3)
    /// * `block_size` - FWHT block size, must be power of 2 (default 256)
    /// * `seed` - Random seed for reproducibility
    pub fn new(dim: usize, num_rounds: usize, block_size: usize, seed: u64) -> Self {
        assert!(block_size.is_power_of_two(), "block_size must be power of 2");
        
        let padded_dim = ((dim + block_size - 1) / block_size) * block_size;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        
        let mut signs = Vec::with_capacity(num_rounds);
        let mut permutations = Vec::with_capacity(num_rounds);
        
        for _ in 0..num_rounds {
            // Random signs {-1, 1}
            let s: Vec<f32> = (0..padded_dim)
                .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
                .collect();
            signs.push(s);
            
            // Random permutation
            let mut perm: Vec<usize> = (0..padded_dim).collect();
            perm.shuffle(&mut rng);
            permutations.push(perm);
        }
        
        Self {
            dim,
            padded_dim,
            block_size,
            num_rounds,
            signs,
            permutations,
        }
    }
    
    /// Apply FWHT to a single block (power of 2 size)
    #[inline]
    fn fwht_block(&self, block: &mut [f32]) {
        let n = block.len();
        debug_assert!(n.is_power_of_two());
        
        let mut h = 1;
        while h < n {
            let mut i = 0;
            while i < n {
                for j in i..(i + h) {
                    let u = block[j];
                    let v = block[j + h];
                    block[j] = u + v;
                    block[j + h] = u - v;
                }
                i += h * 2;
            }
            h *= 2;
        }
        
        // Normalize
        let scale = 1.0 / (n as f32).sqrt();
        for x in block.iter_mut() {
            *x *= scale;
        }
    }
    
    /// Apply FWHT to a single block using SIMD (f32x8)
    #[inline]
    fn fwht_block_simd(&self, block: &mut [f32]) {
        let n = block.len();
        debug_assert!(n.is_power_of_two());
        debug_assert!(n >= 8);
        
        let mut h = 1;
        while h < n {
            if h >= 4 {
                // Use SIMD for larger strides
                let mut i = 0;
                while i < n {
                    let mut j = i;
                    while j + 8 <= i + h {
                        let u = f32x8::from(&block[j..j+8]);
                        let v = f32x8::from(&block[j + h..j + h + 8]);
                        let sum = u + v;
                        let diff = u - v;
                        let sum_arr = sum.to_array();
                        let diff_arr = diff.to_array();
                        block[j..j+8].copy_from_slice(&sum_arr);
                        block[j + h..j + h + 8].copy_from_slice(&diff_arr);
                        j += 8;
                    }
                    // Handle remainder
                    while j < i + h {
                        let u = block[j];
                        let v = block[j + h];
                        block[j] = u + v;
                        block[j + h] = u - v;
                        j += 1;
                    }
                    i += h * 2;
                }
            } else {
                // Scalar for small strides
                let mut i = 0;
                while i < n {
                    for j in i..(i + h) {
                        let u = block[j];
                        let v = block[j + h];
                        block[j] = u + v;
                        block[j + h] = u - v;
                    }
                    i += h * 2;
                }
            }
            h *= 2;
        }
        
        // SIMD normalization
        let scale = f32x8::splat(1.0 / (n as f32).sqrt());
        let mut i = 0;
        while i + 8 <= n {
            let v = f32x8::from(&block[i..i+8]);
            let result = (v * scale).to_array();
            block[i..i+8].copy_from_slice(&result);
            i += 8;
        }
        // Remainder
        let scale_scalar = 1.0 / (n as f32).sqrt();
        while i < n {
            block[i] *= scale_scalar;
            i += 1;
        }
    }
    
    /// Rotate a batch of vectors
    ///
    /// # Arguments
    /// * `vectors` - Flat array of vectors (N * dim)
    /// * `n` - Number of vectors
    ///
    /// # Returns
    /// Rotated vectors (N * padded_dim)
    pub fn rotate(&self, vectors: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(vectors.len(), n * self.dim);
        
        // Allocate output with padding
        let mut output = vec![0.0f32; n * self.padded_dim];
        
        // Copy input with padding
        for i in 0..n {
            let src = &vectors[i * self.dim..(i + 1) * self.dim];
            let dst = &mut output[i * self.padded_dim..i * self.padded_dim + self.dim];
            dst.copy_from_slice(src);
        }
        
        // Apply rotation rounds
        for round in 0..self.num_rounds {
            let signs = &self.signs[round];
            
            // Apply random signs (SIMD)
            for i in 0..n {
                let vec = &mut output[i * self.padded_dim..(i + 1) * self.padded_dim];
                let mut j = 0;
                while j + 8 <= self.padded_dim {
                    let v = f32x8::from(&vec[j..j+8]);
                    let s = f32x8::from(&signs[j..j+8]);
                    let result = (v * s).to_array();
                    vec[j..j+8].copy_from_slice(&result);
                    j += 8;
                }
                while j < self.padded_dim {
                    vec[j] *= signs[j];
                    j += 1;
                }
            }
            
            // Apply blocked FWHT
            for i in 0..n {
                let vec = &mut output[i * self.padded_dim..(i + 1) * self.padded_dim];
                for block_start in (0..self.padded_dim).step_by(self.block_size) {
                    let block = &mut vec[block_start..block_start + self.block_size];
                    if self.block_size >= 64 {
                        self.fwht_block_simd(block);
                    } else {
                        self.fwht_block(block);
                    }
                }
            }
            
            // Apply permutation (between rounds, not after last)
            if round < self.num_rounds - 1 {
                let perm = &self.permutations[round];
                let mut temp = vec![0.0f32; self.padded_dim];
                for i in 0..n {
                    let vec = &mut output[i * self.padded_dim..(i + 1) * self.padded_dim];
                    for (dst_idx, &src_idx) in perm.iter().enumerate() {
                        temp[dst_idx] = vec[src_idx];
                    }
                    vec.copy_from_slice(&temp);
                }
            }
        }
        
        output
    }
    
    /// Get the padded dimension
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fwht_orthogonality() {
        let dim = 128;
        let fwht = FastWalshHadamard::new(dim, 1, 128, 42);
        
        // Create unit vector
        let mut input = vec![0.0f32; dim];
        input[0] = 1.0;
        
        let output = fwht.rotate(&input, 1);
        
        // Norm should be preserved
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Norm not preserved: {}", norm);
    }
    
    #[test]
    fn test_fwht_distance_preservation() {
        let dim = 128;
        let fwht = FastWalshHadamard::new(dim, 3, 64, 42);
        
        // Two random unit vectors
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let mut v1: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let mut v2: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        
        // Normalize
        let n1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v1.iter_mut() { *x /= n1; }
        for x in v2.iter_mut() { *x /= n2; }
        
        // Original distance
        let dist_orig: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        
        // Rotate
        let r1 = fwht.rotate(&v1, 1);
        let r2 = fwht.rotate(&v2, 1);
        
        // Rotated distance (use only first padded_dim elements)
        let dist_rot: f32 = r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        
        assert!((dist_orig - dist_rot).abs() < 1e-4, 
            "Distance not preserved: {} vs {}", dist_orig, dist_rot);
    }
}

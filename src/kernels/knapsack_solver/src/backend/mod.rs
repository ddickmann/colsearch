//! Backend abstraction for tensor operations used by the CPU reference solver.
//!
//! The open-source default is the Rust CPU backend. In-tree wgpu/CUDA backends
//! remain experimental fallbacks and are not treated as the premium product path.

mod cuda;
pub mod simd;

pub use cuda::CudaBackend;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::env;

/// Backend type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU backend using ndarray + rayon + explicit SIMD
    Cpu,
    /// GPU backend using wgpu (Vulkan/Metal/DX12)
    Gpu,
    /// NVIDIA CUDA backend
    Cuda,
    /// Automatic selection. In OSS this resolves to CPU unless experimental
    /// backends are selected explicitly by the caller.
    Auto,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::Auto
    }
}

/// Static objective tensors reused across batch evaluations.
pub struct ObjectiveBatchContext<'a> {
    pub num_candidates: usize,
    pub relevance: &'a [f32],
    pub density: &'a [f32],
    pub centrality: &'a [f32],
    pub recency: &'a [f32],
    pub auxiliary: &'a [f32],
    pub similarity_matrix: &'a [f32],
    pub coverage_matrix: Option<&'a [f32]>,
    pub query_token_weights: Option<&'a [f32]>,
    pub num_query_tokens: usize,
    pub weights: ObjectiveWeights,
}

/// Backend trait for tensor operations
pub trait Backend: Send + Sync {
    /// Compute dot product of two vectors
    fn dot(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// Compute matrix-vector multiplication
    fn matmul_vec(&self, matrix: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32>;
    
    /// Compute matrix-matrix multiplication
    fn matmul(&self, a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize) -> Vec<f32>;
    
    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// Compute pairwise cosine similarity matrix
    fn cosine_similarity_matrix(&self, embeddings: &[f32], n: usize, dim: usize) -> Vec<f32>;
    
    /// Element-wise multiply and sum (dot product with selection mask)
    fn masked_sum(&self, values: &[f32], mask: &[bool]) -> f32;

    /// Prepare reusable objective tensors for repeated batch evaluation.
    fn prepare_objective_context(&self, _context: &ObjectiveBatchContext<'_>, _max_batch_size: usize) {}
    
    /// Batch objective computation
    fn compute_objectives_batch(
        &self,
        selections: &[bool],  // Flattened (batch_size, n)
        batch_size: usize,
        context: &ObjectiveBatchContext<'_>,
    ) -> Vec<f32>;

    /// Batch objective computation for move neighborhoods against a current mask.
    fn compute_move_objectives_batch(
        &self,
        current_selection: &[bool],
        swap_ins: &[usize],
        swap_outs: &[usize],
        context: &ObjectiveBatchContext<'_>,
    ) -> Vec<f32> {
        let batch_size = swap_ins.len();
        let n = context.num_candidates;
        let mut flattened = Vec::with_capacity(batch_size * n);
        for (&swap_in, &swap_out) in swap_ins.iter().zip(swap_outs.iter()) {
            let mut selection = current_selection.to_vec();
            if swap_out < n {
                selection[swap_out] = false;
            }
            selection[swap_in] = true;
            flattened.extend_from_slice(&selection);
        }
        self.compute_objectives_batch(&flattened, batch_size, context)
    }
    
    /// Get backend type
    fn backend_type(&self) -> BackendType;
    
    /// Check if this backend is GPU-accelerated
    fn is_gpu(&self) -> bool {
        matches!(self.backend_type(), BackendType::Gpu | BackendType::Cuda)
    }
}

/// Objective function weights
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveWeights {
    pub alpha: f32,  // Relevance
    pub beta: f32,   // Density
    pub gamma: f32,  // Centrality
    pub delta: f32,  // Recency
    pub epsilon: f32, // Auxiliary
    pub mu: f32, // Fulfilment
    pub lambda: f32, // Redundancy penalty
    pub support_secondary_discount: f32,
    pub support_quorum_bonus: f32,
    pub support_quorum_threshold: f32,
    pub support_quorum_cap: f32,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.3,
            gamma: 0.2,
            delta: 0.1,
            epsilon: 0.0,
            mu: 1.0,
            lambda: 0.5,
            support_secondary_discount: 0.35,
            support_quorum_bonus: 0.18,
            support_quorum_threshold: 0.55,
            support_quorum_cap: 4.0,
        }
    }
}

/// CPU backend implementation using SIMD and rayon
pub struct CpuBackend {
    #[allow(dead_code)]
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }
    
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        simd::dot_simd(a, b)
    }
    
    fn matmul_vec(&self, matrix: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
        debug_assert_eq!(matrix.len(), rows * cols);
        debug_assert_eq!(vec.len(), cols);
        
        // Parallelize over rows
        (0..rows).into_par_iter().map(|i| {
            let row_start = i * cols;
            simd::dot_simd(&matrix[row_start..row_start+cols], vec)
        }).collect()
    }
    
    fn matmul(&self, a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), a_rows * a_cols);
        debug_assert_eq!(b.len(), a_cols * b_cols);
        
        let mut result = vec![0.0; a_rows * b_cols];
        
        // Naive blocked matmul could be better, but we mostly need SimMatrix
        // which has a specialized kernel. For generic matmul, using ndarray 
        // without BLAS is slow anyway.
        // Let's rely on standard loops for now as this isn't the critical path
        // for the solver (only Similarity Matrix is).
        
        // Actually, we can just use loops but dot_simd helps?
        // Transposing B helps vectorization.
        let mat_a = ArrayView2::from_shape((a_rows, a_cols), a).unwrap();
        let mat_b = ArrayView2::from_shape((a_cols, b_cols), b).unwrap();
        mat_a.dot(&mat_b).into_raw_vec()
    }
    
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        simd::cosine_similarity_simd(a, b)
    }
    
    fn cosine_similarity_matrix(&self, embeddings: &[f32], n: usize, dim: usize) -> Vec<f32> {
        simd::cosine_similarity_matrix_simd(embeddings, n, dim)
    }
    
    fn masked_sum(&self, values: &[f32], mask: &[bool]) -> f32 {
        debug_assert_eq!(values.len(), mask.len());
        
        values
            .iter()
            .zip(mask.iter())
            .filter_map(|(v, &m)| if m { Some(*v) } else { None })
            .sum()
    }
    
    fn compute_objectives_batch(
        &self,
        selections: &[bool],
        batch_size: usize,
        context: &ObjectiveBatchContext<'_>,
    ) -> Vec<f32> {
        let n = context.num_candidates;
        (0..batch_size)
            .into_par_iter()
            .map(|batch_idx| {
                let sel_start = batch_idx * n;
                let selection = &selections[sel_start..sel_start + n];
                
                // Linear terms
                let rel_term: f32 = selection
                    .iter()
                    .zip(context.relevance.iter())
                    .filter_map(|(&s, &r)| if s { Some(r) } else { None })
                    .sum();
                
                let den_term: f32 = selection
                    .iter()
                    .zip(context.density.iter())
                    .filter_map(|(&s, &d)| if s { Some(d) } else { None })
                    .sum();
                
                let cen_term: f32 = selection
                    .iter()
                    .zip(context.centrality.iter())
                    .filter_map(|(&s, &c)| if s { Some(c) } else { None })
                    .sum();
                
                let rec_term: f32 = selection
                    .iter()
                    .zip(context.recency.iter())
                    .filter_map(|(&s, &r)| if s { Some(r) } else { None })
                    .sum();
                
                let aux_term: f32 = selection
                    .iter()
                    .zip(context.auxiliary.iter())
                    .filter_map(|(&s, &a)| if s { Some(a) } else { None })
                    .sum();

                let mut fulfilment = 0.0f32;
                if let Some(coverage_matrix) = context.coverage_matrix {
                    for token_idx in 0..context.num_query_tokens {
                        let row = &coverage_matrix[token_idx * n..(token_idx + 1) * n];
                        let mut best = 0.0f32;
                        let mut second = 0.0f32;
                        let mut third = 0.0f32;
                        let mut fourth = 0.0f32;
                        let mut quorum_count = 0u32;
                        for i in 0..n {
                            if selection[i] {
                                let score = row[i];
                                if score >= context.weights.support_quorum_threshold {
                                    quorum_count += 1;
                                }
                                if score >= best {
                                    fourth = third;
                                    third = second;
                                    second = best;
                                    best = score;
                                } else if score > second {
                                    fourth = third;
                                    third = second;
                                    second = score;
                                } else if score > third {
                                    fourth = third;
                                    third = score;
                                } else if score > fourth {
                                    fourth = score;
                                }
                            }
                        }
                        let weight = context
                            .query_token_weights
                            .and_then(|weights| weights.get(token_idx))
                            .copied()
                            .unwrap_or(1.0);
                        let quorum_cap = context.weights.support_quorum_cap.max(2.0) as u32;
                        let third_mass = if quorum_cap >= 3 && quorum_count >= 3 {
                            context.weights.support_quorum_bonus * third.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let fourth_mass = if quorum_cap >= 4 && quorum_count >= 4 {
                            0.5 * context.weights.support_quorum_bonus * fourth.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let supported = best.clamp(0.0, 1.0)
                            + context.weights.support_secondary_discount * second.clamp(0.0, 1.0)
                            + third_mass
                            + fourth_mass;
                        fulfilment += weight * supported;
                    }
                }
                
                // Quadratic redundancy term: x^T S x
                let mut redundancy = 0.0f32;
                for i in 0..n {
                    if selection[i] {
                        for j in 0..n {
                            if selection[j] && i != j {
                                redundancy += context.similarity_matrix[i * n + j];
                            }
                        }
                    }
                }
                redundancy *= 0.5; // Avoid double counting
                
                context.weights.alpha * rel_term
                    + context.weights.beta * den_term
                    + context.weights.gamma * cen_term
                    + context.weights.delta * rec_term
                    + context.weights.epsilon * aux_term
                    + context.weights.mu * fulfilment
                    - context.weights.lambda * redundancy
            })
            .collect()
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }
}

/// Create a backend based on the specified type
pub fn create_backend(backend_type: BackendType) -> Box<dyn Backend> {
    fn strict_gpu_required() -> bool {
        env::var("LATENCE_SOLVER_STRICT_GPU")
            .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    }

    fn forbid_fallback(requested: &str, reason: &str) -> ! {
        panic!(
            "Strict GPU mode forbids fallback from {} backend: {}",
            requested, reason
        )
    }

    match backend_type {
        BackendType::Cpu => Box::new(CpuBackend::new()),
        
        #[cfg(feature = "gpu")]
        BackendType::Gpu => {
            if strict_gpu_required() {
                forbid_fallback("wgpu", "experimental OSS wgpu backend is not implemented");
            }
            tracing::warn!("Experimental wgpu backend is not implemented; using CPU reference backend");
            Box::new(CpuBackend::new())
        }
        
        #[cfg(not(feature = "gpu"))]
        BackendType::Gpu => {
            if strict_gpu_required() {
                forbid_fallback("wgpu", "gpu feature is not enabled");
            }
            tracing::warn!("Experimental wgpu feature not enabled; using CPU reference backend");
            Box::new(CpuBackend::new())
        }
        
        BackendType::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match CudaBackend::default_device() {
                    Ok(backend) => {
                        if let Err(err) = backend.startup_self_test() {
                            if strict_gpu_required() {
                                forbid_fallback(
                                    "cuda",
                                    &format!("experimental CUDA startup self-test failed: {:?}", err),
                                );
                            }
                            tracing::warn!(
                                "Experimental CUDA backend failed startup self-test: {:?}; falling back to CPU",
                                err
                            );
                            return Box::new(CpuBackend::new());
                        }
                        tracing::warn!("Using experimental in-tree CUDA backend");
                        return Box::new(backend);
                    }
                    Err(e) => {
                        if strict_gpu_required() {
                            forbid_fallback("cuda", &format!("failed to initialize experimental CUDA backend: {:?}", e));
                        }
                        tracing::warn!("Failed to initialize experimental CUDA backend: {:?}; falling back to CPU", e);
                    }
                }
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                if strict_gpu_required() {
                    forbid_fallback("cuda", "cuda feature is not enabled");
                }
                tracing::warn!("Experimental CUDA feature not enabled; using CPU reference backend");
            }
            
            Box::new(CpuBackend::new())
        }
        
        BackendType::Auto => {
            tracing::debug!("Auto backend resolves to CPU reference backend in OSS mode");
            Box::new(CpuBackend::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_dot() {
        let backend = CpuBackend::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = backend.dot(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_cpu_cosine_similarity() {
        let backend = CpuBackend::new();
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        
        let result = backend.cosine_similarity(&a, &b);
        assert!((result - 1.0).abs() < 1e-6);
        
        let c = vec![0.0, 1.0];
        let result2 = backend.cosine_similarity(&a, &c);
        assert!(result2.abs() < 1e-6);
    }
    
    #[test]
    fn test_cpu_similarity_matrix() {
        let backend = CpuBackend::new();
        
        // 3 embeddings of dimension 2
        let embeddings = vec![
            1.0, 0.0,  // [1, 0]
            0.0, 1.0,  // [0, 1]
            1.0, 1.0,  // [1, 1]
        ];
        
        let matrix = backend.cosine_similarity_matrix(&embeddings, 3, 2);
        
        // Diagonal should be 1.0
        assert!((matrix[0] - 1.0).abs() < 1e-6);
        assert!((matrix[4] - 1.0).abs() < 1e-6);
        assert!((matrix[8] - 1.0).abs() < 1e-6);
        
        // [1,0] and [0,1] should be orthogonal
        assert!(matrix[1].abs() < 1e-6);
    }
    
    #[test]
    fn test_matmul() {
        let backend = CpuBackend::new();
        
        // 2x3 matrix
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 3x2 matrix
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let result = backend.matmul(&a, 2, 3, &b, 2);
        
        // Expected: [[22, 28], [49, 64]]
        assert!((result[0] - 22.0).abs() < 1e-6);
        assert!((result[1] - 28.0).abs() < 1e-6);
        assert!((result[2] - 49.0).abs() < 1e-6);
        assert!((result[3] - 64.0).abs() < 1e-6);
    }
}

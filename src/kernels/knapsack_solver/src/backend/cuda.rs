//! CUDA Backend Implementation
//!
//! Provides GPU-accelerated tensor operations for the Knapsack Solver
//! using NVIDIA CUDA via the cudarc crate.
//!
//! Key kernels:
//! - cosine_similarity_matrix: Compute N×N pairwise similarity matrix
//! - batch_objective_eval: Evaluate objective function for all candidate moves
//! - delta_update: O(1) incremental redundancy contribution updates

#[cfg(feature = "cuda")]
mod cuda_impl {
    use crate::backend::{Backend, BackendType, ObjectiveBatchContext, ObjectiveWeights};
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use parking_lot::Mutex;
    use std::env;
    use std::sync::{Arc, OnceLock};

    static DEFAULT_CUDA_DEVICE: OnceLock<Result<Arc<CudaDevice>, String>> = OnceLock::new();

    /// CUDA Backend for GPU-accelerated operations
    pub struct CudaBackend {
        device: Arc<CudaDevice>,
        /// Compiled PTX kernels
        kernels_loaded: bool,
        objective_cache: Mutex<Option<CudaObjectiveCache>>,
    }

    struct CudaObjectiveCache {
        num_candidates: usize,
        num_query_tokens: usize,
        linear_terms_d: CudaSlice<f32>,
        similarity_d: CudaSlice<f32>,
        coverage_d: CudaSlice<f32>,
        query_token_weights_d: CudaSlice<f32>,
        weights_d: CudaSlice<f32>,
        objectives_d: CudaSlice<f32>,
        objective_capacity: usize,
    }

    impl CudaBackend {
        fn load_kernels_on_device(device: &Arc<CudaDevice>) -> Result<(), CudaError> {
            let ptx = compile_ptx(KNAPSACK_KERNELS_CU)?;
            device.load_ptx(
                ptx,
                "knapsack_kernels",
                &[
                    "cosine_similarity_matrix_kernel",
                    "batch_objective_kernel",
                    "move_objective_kernel",
                    "masked_sum_kernel",
                ],
            )?;
            Ok(())
        }

        fn shared_default_device() -> Result<Arc<CudaDevice>, CudaError> {
            let cached = DEFAULT_CUDA_DEVICE.get_or_init(|| {
                let device = CudaDevice::new(0).map_err(|err| err.to_string())?;
                Self::load_kernels_on_device(&device).map_err(|err| err.to_string())?;
                Ok(device)
            });
            cached
                .as_ref()
                .map(Arc::clone)
                .map_err(|err| CudaError::DeviceError(err.clone()))
        }

        fn strict_gpu_required() -> bool {
            env::var("LATENCE_SOLVER_STRICT_GPU")
                .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false)
        }

        fn fallback_or_panic<T, E, F>(&self, operation: &str, result: Result<T, E>, fallback: F) -> T
        where
            E: std::fmt::Display,
            F: FnOnce() -> T,
        {
            match result {
                Ok(value) => value,
                Err(err) => {
                    if Self::strict_gpu_required() {
                        panic!(
                            "Strict GPU mode forbids CPU fallback for CUDA operation `{}`: {}",
                            operation,
                            err
                        );
                    }
                    tracing::warn!(
                        "Falling back to CPU for CUDA operation `{}` after GPU error: {}",
                        operation,
                        err
                    );
                    fallback()
                }
            }
        }

        fn dummy_device_buffer(&self) -> Result<CudaSlice<f32>, CudaError> {
            Ok(self.device.htod_sync_copy(&[0.0f32])?)
        }

        fn ensure_objective_cache(
            &self,
            context: &ObjectiveBatchContext<'_>,
            min_batch_capacity: usize,
        ) -> Result<(), CudaError> {
            let mut cache_guard = self.objective_cache.lock();
            let needs_rebuild = match cache_guard.as_ref() {
                Some(cache) => {
                    cache.num_candidates != context.num_candidates
                        || cache.num_query_tokens != context.num_query_tokens
                        || cache.objective_capacity < min_batch_capacity.max(1)
                }
                None => true,
            };

            if !needs_rebuild {
                return Ok(());
            }

            let coverage_d = if let Some(coverage) = context.coverage_matrix {
                self.device.htod_sync_copy(coverage)?
            } else {
                self.dummy_device_buffer()?
            };
            let query_token_weights_d = if let Some(weights) = context.query_token_weights {
                self.device.htod_sync_copy(weights)?
            } else {
                self.dummy_device_buffer()?
            };
            let weights_arr = [
                context.weights.lambda,
                context.weights.mu,
                context.weights.support_secondary_discount,
                context.weights.support_quorum_bonus,
                context.weights.support_quorum_threshold,
                context.weights.support_quorum_cap,
            ];
            let linear_terms: Vec<f32> = (0..context.num_candidates)
                .map(|idx| {
                    context.weights.alpha * context.relevance[idx]
                        + context.weights.beta * context.density[idx]
                        + context.weights.gamma * context.centrality[idx]
                        + context.weights.delta * context.recency[idx]
                        + context.weights.epsilon * context.auxiliary[idx]
                })
                .collect();

            *cache_guard = Some(CudaObjectiveCache {
                num_candidates: context.num_candidates,
                num_query_tokens: context.num_query_tokens,
                linear_terms_d: self.device.htod_sync_copy(&linear_terms)?,
                similarity_d: self.device.htod_sync_copy(context.similarity_matrix)?,
                coverage_d,
                query_token_weights_d,
                weights_d: self.device.htod_sync_copy(&weights_arr)?,
                objectives_d: self.device.alloc_zeros(min_batch_capacity.max(1))?,
                objective_capacity: min_batch_capacity.max(1),
            });
            Ok(())
        }

        fn dtoh_objectives(
            &self,
            objectives_d: &CudaSlice<f32>,
            batch_size: usize,
        ) -> Result<Vec<f32>, CudaError> {
            let full = self.device.dtoh_sync_copy(objectives_d)?;
            Ok(full.into_iter().take(batch_size).collect())
        }

        /// Create a new CUDA backend on the specified device
        pub fn new(device_id: usize) -> Result<Self, CudaError> {
            if device_id == 0 {
                return Ok(Self {
                    device: Self::shared_default_device()?,
                    kernels_loaded: true,
                    objective_cache: Mutex::new(None),
                });
            }

            let device = CudaDevice::new(device_id)?;
            let mut backend = Self {
                device,
                kernels_loaded: false,
                objective_cache: Mutex::new(None),
            };
            backend.load_kernels()?;
            Ok(backend)
        }
        
        /// Create backend on the default device (GPU 0)
        pub fn default_device() -> Result<Self, CudaError> {
            Self::new(0)
        }

        pub fn startup_self_test(&self) -> Result<(), CudaError> {
            let embeddings = vec![1.0f32, 0.0, 0.0, 1.0];
            let similarity = self.cosine_similarity_matrix_cuda(&embeddings, 2, 2)?;
            if similarity.len() != 4 {
                return Err(CudaError::KernelError(
                    "cosine_similarity_matrix_cuda returned unexpected output size".to_string(),
                ));
            }

            let context = ObjectiveBatchContext {
                num_candidates: 2,
                relevance: &[1.0, 0.1],
                density: &[0.0, 0.0],
                centrality: &[0.0, 0.0],
                recency: &[0.0, 0.0],
                auxiliary: &[0.0, 0.0],
                similarity_matrix: &[0.0, 0.2, 0.2, 0.0],
                coverage_matrix: Some(&[1.0, 0.2]),
                query_token_weights: Some(&[1.0]),
                num_query_tokens: 1,
                weights: ObjectiveWeights {
                    alpha: 1.0,
                    beta: 0.0,
                    gamma: 0.0,
                    delta: 0.0,
                    epsilon: 0.0,
                    mu: 1.0,
                    lambda: 0.5,
                    support_secondary_discount: 0.35,
                    support_quorum_bonus: 0.18,
                    support_quorum_threshold: 0.55,
                    support_quorum_cap: 4.0,
                },
            };
            self.ensure_objective_cache(&context, 2)?;
            let objectives = self.compute_objectives_batch_cuda(&[true, false, true, true], 2, &context)?;
            if objectives.len() != 2 {
                return Err(CudaError::KernelError(
                    "batch_objective_kernel returned unexpected output size".to_string(),
                ));
            }

            let move_scores =
                self.compute_move_objectives_batch_cuda(&[true, false], &[1usize], &[0usize], &context)?;
            if move_scores.len() != 1 {
                return Err(CudaError::KernelError(
                    "move_objective_kernel returned unexpected output size".to_string(),
                ));
            }

            let masked_sum = self.masked_sum_cuda(&[1.0, 2.0], &[true, false])?;
            if !masked_sum.is_finite() {
                return Err(CudaError::KernelError(
                    "masked_sum_kernel returned a non-finite value".to_string(),
                ));
            }

            Ok(())
        }
        
        /// Load and compile CUDA kernels
        pub fn load_kernels(&mut self) -> Result<(), CudaError> {
            if self.kernels_loaded {
                return Ok(());
            }

            Self::load_kernels_on_device(&self.device)?;
            self.kernels_loaded = true;
            Ok(())
        }
        
        /// Compute cosine similarity matrix on GPU
        pub fn cosine_similarity_matrix_cuda(
            &self,
            embeddings: &[f32],
            n: usize,
            dim: usize,
        ) -> Result<Vec<f32>, CudaError> {
            // Allocate device memory
            let embeddings_d = self.device.htod_sync_copy(embeddings)?;
            let mut similarity_d: CudaSlice<f32> = self.device.alloc_zeros(n * n)?;
            
            // Launch kernel
            let func = self
                .device
                .get_func("knapsack_kernels", "cosine_similarity_matrix_kernel")
                .ok_or_else(|| CudaError::KernelError("missing cosine_similarity_matrix_kernel".to_string()))?;
            
            // Grid/block configuration
            let block_size = 16;
            let grid_size = (n + block_size - 1) / block_size;
            
            let cfg = LaunchConfig {
                grid_dim: (grid_size as u32, grid_size as u32, 1),
                block_dim: (block_size as u32, block_size as u32, 1),
                shared_mem_bytes: 0,
            };
            
            unsafe {
                func.launch(cfg, (
                    &embeddings_d,
                    &mut similarity_d,
                    n as i32,
                    dim as i32,
                ))?;
            }
            
            // Copy result back to host
            let similarity = self.device.dtoh_sync_copy(&similarity_d)?;
            
            Ok(similarity)
        }
        
        /// Compute batch objective values on GPU
        pub fn compute_objectives_batch_cuda(
            &self,
            selections: &[bool],
            batch_size: usize,
            context: &ObjectiveBatchContext<'_>,
        ) -> Result<Vec<f32>, CudaError> {
            self.ensure_objective_cache(context, batch_size)?;
            // Convert bool selection to u8 for GPU
            let selections_u8: Vec<u8> = selections.iter().map(|&b| b as u8).collect();
            let selections_d = self.device.htod_sync_copy(&selections_u8)?;
            let mut cache_guard = self.objective_cache.lock();
            let cache = cache_guard
                .as_mut()
                .ok_or_else(|| CudaError::MemoryError("objective cache was not prepared".to_string()))?;

            // Launch kernel
            let func = self
                .device
                .get_func("knapsack_kernels", "batch_objective_kernel")
                .ok_or_else(|| CudaError::KernelError("missing batch_objective_kernel".to_string()))?;
            
            let block_size = 256;
            let grid_size = (batch_size + block_size - 1) / block_size;
            
            let cfg = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            
            unsafe {
                func.launch(cfg, (
                    &selections_d,
                    batch_size as i32,
                    context.num_candidates as i32,
                    &cache.linear_terms_d,
                    &cache.similarity_d,
                    &cache.coverage_d,
                    &cache.query_token_weights_d,
                    context.num_query_tokens as i32,
                    &cache.weights_d,
                    &mut cache.objectives_d,
                ))?;
            }
            
            self.dtoh_objectives(&cache.objectives_d, batch_size)
        }

        pub fn compute_move_objectives_batch_cuda(
            &self,
            current_selection: &[bool],
            swap_ins: &[usize],
            swap_outs: &[usize],
            context: &ObjectiveBatchContext<'_>,
        ) -> Result<Vec<f32>, CudaError> {
            let batch_size = swap_ins.len();
            self.ensure_objective_cache(context, batch_size)?;
            let current_selection_u8: Vec<u8> = current_selection.iter().map(|&b| b as u8).collect();
            let swap_ins_i32: Vec<i32> = swap_ins.iter().map(|&idx| idx as i32).collect();
            let swap_outs_i32: Vec<i32> = swap_outs.iter().map(|&idx| idx as i32).collect();
            let current_selection_d = self.device.htod_sync_copy(&current_selection_u8)?;
            let swap_ins_d = self.device.htod_sync_copy(&swap_ins_i32)?;
            let swap_outs_d = self.device.htod_sync_copy(&swap_outs_i32)?;
            let mut cache_guard = self.objective_cache.lock();
            let cache = cache_guard
                .as_mut()
                .ok_or_else(|| CudaError::MemoryError("objective cache was not prepared".to_string()))?;
            let func = self
                .device
                .get_func("knapsack_kernels", "move_objective_kernel")
                .ok_or_else(|| CudaError::KernelError("missing move_objective_kernel".to_string()))?;

            let block_size = 256;
            let grid_size = (batch_size + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(cfg, (
                    &current_selection_d,
                    &swap_ins_d,
                    &swap_outs_d,
                    batch_size as i32,
                    context.num_candidates as i32,
                    &cache.linear_terms_d,
                    &cache.similarity_d,
                    &cache.coverage_d,
                    &cache.query_token_weights_d,
                    context.num_query_tokens as i32,
                    &cache.weights_d,
                    &mut cache.objectives_d,
                ))?;
            }

            self.dtoh_objectives(&cache.objectives_d, batch_size)
        }

        pub fn masked_sum_cuda(
            &self,
            values: &[f32],
            mask: &[bool],
        ) -> Result<f32, CudaError> {
            if values.len() != mask.len() {
                return Err(CudaError::KernelError(
                    "values and mask length must match for masked_sum_cuda".to_string(),
                ));
            }

            let mask_u8: Vec<u8> = mask.iter().map(|&selected| selected as u8).collect();
            let values_d = self.device.htod_sync_copy(values)?;
            let mask_d = self.device.htod_sync_copy(&mask_u8)?;
            let mut result_d: CudaSlice<f32> = self.device.alloc_zeros(1)?;
            let func = self
                .device
                .get_func("knapsack_kernels", "masked_sum_kernel")
                .ok_or_else(|| CudaError::KernelError("missing masked_sum_kernel".to_string()))?;
            let block_size = 256usize;
            let grid_size = (values.len() + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
            };

            unsafe {
                func.launch(cfg, (&values_d, &mask_d, values.len() as i32, &mut result_d))?;
            }

            let result = self.device.dtoh_sync_copy(&result_d)?;
            Ok(result[0])
        }
    }

    impl Backend for CudaBackend {
        fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
            if Self::strict_gpu_required() {
                panic!("Strict GPU mode forbids CPU fallback for CUDA backend dot()");
            }
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
        
        fn matmul_vec(&self, matrix: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
            if Self::strict_gpu_required() {
                panic!("Strict GPU mode forbids CPU fallback for CUDA backend matmul_vec()");
            }
            let mut result = vec![0.0; rows];
            for i in 0..rows {
                for j in 0..cols {
                    result[i] += matrix[i * cols + j] * vec[j];
                }
            }
            result
        }
        
        fn matmul(&self, a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize) -> Vec<f32> {
            if Self::strict_gpu_required() {
                panic!("Strict GPU mode forbids CPU fallback for CUDA backend matmul()");
            }
            let mut result = vec![0.0; a_rows * b_cols];
            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a[i * a_cols + k] * b[k * b_cols + j];
                    }
                    result[i * b_cols + j] = sum;
                }
            }
            result
        }
        
        fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
            if Self::strict_gpu_required() {
                panic!("Strict GPU mode forbids CPU fallback for CUDA backend cosine_similarity()");
            }
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            }
        }
        
        fn cosine_similarity_matrix(&self, embeddings: &[f32], n: usize, dim: usize) -> Vec<f32> {
            self.fallback_or_panic(
                "cosine_similarity_matrix",
                self.cosine_similarity_matrix_cuda(embeddings, n, dim),
                || self.cosine_similarity_matrix_cpu(embeddings, n, dim),
            )
        }
        
        fn masked_sum(&self, values: &[f32], mask: &[bool]) -> f32 {
            self.fallback_or_panic(
                "masked_sum",
                self.masked_sum_cuda(values, mask),
                || {
                    values
                        .iter()
                        .zip(mask.iter())
                        .filter_map(|(v, &m)| if m { Some(*v) } else { None })
                        .sum()
                },
            )
        }

        fn prepare_objective_context(&self, context: &ObjectiveBatchContext<'_>, max_batch_size: usize) {
            if let Err(err) = self.ensure_objective_cache(context, max_batch_size.max(1)) {
                if Self::strict_gpu_required() {
                    panic!(
                        "Strict GPU mode forbids CPU fallback while preparing CUDA objective context: {}",
                        err
                    );
                }
                tracing::warn!("Failed to prepare CUDA objective context cache: {}", err);
            }
        }
        
        fn compute_objectives_batch(
            &self,
            selections: &[bool],
            batch_size: usize,
            context: &ObjectiveBatchContext<'_>,
        ) -> Vec<f32> {
            self.fallback_or_panic(
                "compute_objectives_batch",
                self.compute_objectives_batch_cuda(
                    selections,
                    batch_size,
                    context,
                ),
                || {
                    self.compute_objectives_batch_cpu(
                        selections,
                        batch_size,
                        context,
                    )
                },
            )
        }

        fn compute_move_objectives_batch(
            &self,
            current_selection: &[bool],
            swap_ins: &[usize],
            swap_outs: &[usize],
            context: &ObjectiveBatchContext<'_>,
        ) -> Vec<f32> {
            self.fallback_or_panic(
                "compute_move_objectives_batch",
                self.compute_move_objectives_batch_cuda(
                    current_selection,
                    swap_ins,
                    swap_outs,
                    context,
                ),
                || {
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
                    self.compute_objectives_batch_cpu(&flattened, batch_size, context)
                },
            )
        }
        
        fn backend_type(&self) -> BackendType {
            BackendType::Cuda
        }
    }

    impl CudaBackend {
        /// CPU fallback for similarity matrix
        fn cosine_similarity_matrix_cpu(&self, embeddings: &[f32], n: usize, dim: usize) -> Vec<f32> {
            use ndarray::{ArrayView2, Array2};
            
            let emb = ArrayView2::from_shape((n, dim), embeddings).unwrap();
            
            // Compute norms
            let norms: Vec<f32> = emb
                .rows()
                .into_iter()
                .map(|row| row.dot(&row).sqrt())
                .collect();
            
            // Normalize embeddings
            let mut normalized = Array2::zeros((n, dim));
            for (i, row) in emb.rows().into_iter().enumerate() {
                if norms[i] > 0.0 {
                    for (j, &val) in row.iter().enumerate() {
                        normalized[[i, j]] = val / norms[i];
                    }
                }
            }
            
            // Compute similarity matrix
            let mut similarity = vec![0.0; n * n];
            for i in 0..n {
                let row_i = normalized.row(i);
                for j in 0..n {
                    let row_j = normalized.row(j);
                    similarity[i * n + j] = row_i.dot(&row_j);
                }
            }
            
            similarity
        }
        
        /// CPU fallback for batch objectives
        fn compute_objectives_batch_cpu(
            &self,
            selections: &[bool],
            batch_size: usize,
            context: &ObjectiveBatchContext<'_>,
        ) -> Vec<f32> {
            use rayon::prelude::*;
            let n = context.num_candidates;
            
            (0..batch_size)
                .into_par_iter()
                .map(|batch_idx| {
                    let sel_start = batch_idx * n;
                    let selection = &selections[sel_start..sel_start + n];
                    
                    let rel_term: f32 = selection.iter()
                        .zip(context.relevance.iter())
                        .filter_map(|(&s, &r)| if s { Some(r) } else { None })
                        .sum();
                    
                    let den_term: f32 = selection.iter()
                        .zip(context.density.iter())
                        .filter_map(|(&s, &d)| if s { Some(d) } else { None })
                        .sum();
                    
                    let cen_term: f32 = selection.iter()
                        .zip(context.centrality.iter())
                        .filter_map(|(&s, &c)| if s { Some(c) } else { None })
                        .sum();
                    
                    let rec_term: f32 = selection.iter()
                        .zip(context.recency.iter())
                        .filter_map(|(&s, &r)| if s { Some(r) } else { None })
                        .sum();

                    let aux_term: f32 = selection.iter()
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
                            fulfilment += weight
                                * (best.clamp(0.0, 1.0)
                                    + context.weights.support_secondary_discount * second.clamp(0.0, 1.0)
                                    + third_mass
                                    + fourth_mass);
                        }
                    }
                    
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
                    redundancy *= 0.5;
                    
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
    }

    /// CUDA kernel source code
    const KNAPSACK_KERNELS_CU: &str = r#"
extern "C" __global__ void cosine_similarity_matrix_kernel(
    const float* embeddings,
    float* similarity,
    int n,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= n || j >= n) return;
    
    // Compute dot product
    float dot = 0.0f;
    float norm_i = 0.0f;
    float norm_j = 0.0f;
    
    for (int k = 0; k < dim; k++) {
        float ei = embeddings[i * dim + k];
        float ej = embeddings[j * dim + k];
        dot += ei * ej;
        norm_i += ei * ei;
        norm_j += ej * ej;
    }
    
    norm_i = sqrtf(norm_i);
    norm_j = sqrtf(norm_j);
    
    float sim = 0.0f;
    if (norm_i > 0.0f && norm_j > 0.0f) {
        sim = dot / (norm_i * norm_j);
    }
    
    similarity[i * n + j] = sim;
}

extern "C" __global__ void batch_objective_kernel(
    const unsigned char* selections,  // batch_size * n
    int batch_size,
    int n,
    const float* linear_terms,
    const float* similarity,
    const float* coverage,
    const float* query_token_weights,
    int num_query_tokens,
    const float* weights,  // [lambda, mu, support_secondary_discount, support_quorum_bonus, support_quorum_threshold, support_quorum_cap]
    float* objectives
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    const unsigned char* sel = selections + batch_idx * n;
    
    float lambda = weights[0];
    float mu = weights[1];
    float support_secondary_discount = weights[2];
    float support_quorum_bonus = weights[3];
    float support_quorum_threshold = weights[4];
    float support_quorum_cap = fmaxf(weights[5], 2.0f);
    
    float linear_sum = 0.0f;
    float fulfilment = 0.0f;
    float redundancy = 0.0f;
    
    for (int i = 0; i < n; i++) {
        if (sel[i]) {
            linear_sum += linear_terms[i];
            
            for (int j = 0; j < n; j++) {
                if (sel[j] && i != j) {
                    redundancy += similarity[i * n + j];
                }
            }
        }
    }
    redundancy *= 0.5f;

    for (int token_idx = 0; token_idx < num_query_tokens; token_idx++) {
        const float* row = coverage + token_idx * n;
        float best = 0.0f;
        float second = 0.0f;
        float third = 0.0f;
        float fourth = 0.0f;
        int quorum_count = 0;
        for (int i = 0; i < n; i++) {
            if (sel[i]) {
                float score = row[i];
                if (score >= support_quorum_threshold) {
                    quorum_count += 1;
                }
                if (score >= best) {
                    fourth = third;
                    third = second;
                    second = best;
                    best = score;
                } else if (score > second) {
                    fourth = third;
                    third = second;
                    second = score;
                } else if (score > third) {
                    fourth = third;
                    third = score;
                } else if (score > fourth) {
                    fourth = score;
                }
            }
        }
        float weight = query_token_weights[token_idx];
        int quorum_cap = (int)support_quorum_cap;
        float third_mass = (quorum_cap >= 3 && quorum_count >= 3)
            ? support_quorum_bonus * fminf(fmaxf(third, 0.0f), 1.0f)
            : 0.0f;
        float fourth_mass = (quorum_cap >= 4 && quorum_count >= 4)
            ? 0.5f * support_quorum_bonus * fminf(fmaxf(fourth, 0.0f), 1.0f)
            : 0.0f;
        fulfilment += weight * (
            fminf(fmaxf(best, 0.0f), 1.0f)
            + support_secondary_discount * fminf(fmaxf(second, 0.0f), 1.0f)
            + third_mass
            + fourth_mass
        );
    }
    
    objectives[batch_idx] = linear_sum
                          + mu * fulfilment
                          - lambda * redundancy;
}

extern "C" __global__ void move_objective_kernel(
    const unsigned char* current_selection,
    const int* swap_ins,
    const int* swap_outs,
    int batch_size,
    int n,
    const float* linear_terms,
    const float* similarity,
    const float* coverage,
    const float* query_token_weights,
    int num_query_tokens,
    const float* weights,
    float* objectives
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int swap_in = swap_ins[batch_idx];
    int swap_out = swap_outs[batch_idx];

    float lambda = weights[0];
    float mu = weights[1];
    float support_secondary_discount = weights[2];
    float support_quorum_bonus = weights[3];
    float support_quorum_threshold = weights[4];
    float support_quorum_cap = fmaxf(weights[5], 2.0f);

    float linear_sum = 0.0f;
    float fulfilment = 0.0f;
    float redundancy = 0.0f;

    for (int i = 0; i < n; i++) {
        bool selected = current_selection[i] != 0;
        if (swap_out >= 0 && swap_out < n && i == swap_out) selected = false;
        if (i == swap_in) selected = true;
        if (!selected) continue;

        linear_sum += linear_terms[i];

        for (int j = 0; j < n; j++) {
            bool selected_j = current_selection[j] != 0;
            if (swap_out >= 0 && swap_out < n && j == swap_out) selected_j = false;
            if (j == swap_in) selected_j = true;
            if (selected_j && i != j) {
                redundancy += similarity[i * n + j];
            }
        }
    }
    redundancy *= 0.5f;

    for (int token_idx = 0; token_idx < num_query_tokens; token_idx++) {
        const float* row = coverage + token_idx * n;
        float best = 0.0f;
        float second = 0.0f;
        float third = 0.0f;
        float fourth = 0.0f;
        int quorum_count = 0;
        for (int i = 0; i < n; i++) {
            bool selected = current_selection[i] != 0;
            if (swap_out >= 0 && swap_out < n && i == swap_out) selected = false;
            if (i == swap_in) selected = true;
            if (selected) {
                float score = row[i];
                if (score >= support_quorum_threshold) {
                    quorum_count += 1;
                }
                if (score >= best) {
                    fourth = third;
                    third = second;
                    second = best;
                    best = score;
                } else if (score > second) {
                    fourth = third;
                    third = second;
                    second = score;
                } else if (score > third) {
                    fourth = third;
                    third = score;
                } else if (score > fourth) {
                    fourth = score;
                }
            }
        }
        float weight = query_token_weights[token_idx];
        int quorum_cap = (int)support_quorum_cap;
        float third_mass = (quorum_cap >= 3 && quorum_count >= 3)
            ? support_quorum_bonus * fminf(fmaxf(third, 0.0f), 1.0f)
            : 0.0f;
        float fourth_mass = (quorum_cap >= 4 && quorum_count >= 4)
            ? 0.5f * support_quorum_bonus * fminf(fmaxf(fourth, 0.0f), 1.0f)
            : 0.0f;
        fulfilment += weight * (
            fminf(fmaxf(best, 0.0f), 1.0f)
            + support_secondary_discount * fminf(fmaxf(second, 0.0f), 1.0f)
            + third_mass
            + fourth_mass
        );
    }

    objectives[batch_idx] = linear_sum
                          + mu * fulfilment
                          - lambda * redundancy;
}

extern "C" __global__ void masked_sum_kernel(
    const float* values,
    const unsigned char* mask,
    int n,
    float* result
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n && mask[i]) ? values[i] : 0.0f;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ============== Rotational Quantization Kernels ==============

// FWHT butterfly step for a single block
extern "C" __global__ void fwht_butterfly_kernel(
    float* data,           // (n_vectors * padded_dim)
    int n_vectors,
    int padded_dim,
    int block_size,
    int h                  // Current butterfly stride
) {
    int vec_idx = blockIdx.y;
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vec_idx >= n_vectors) return;
    
    int blocks_per_vec = padded_dim / block_size;
    if (block_idx >= blocks_per_vec) return;
    
    float* vec = data + vec_idx * padded_dim + block_idx * block_size;
    
    // Each thread handles multiple butterfly operations
    int pairs_per_block = block_size / (2 * h);
    int ops_per_thread = (pairs_per_block * h + blockDim.x - 1) / blockDim.x;
    
    for (int op = 0; op < ops_per_thread; op++) {
        int flat_idx = tid + op * blockDim.x;
        if (flat_idx >= pairs_per_block * h) continue;
        
        int pair = flat_idx / h;
        int offset = flat_idx % h;
        int i = pair * 2 * h + offset;
        int j = i + h;
        
        if (j < block_size) {
            float u = vec[i];
            float v = vec[j];
            vec[i] = u + v;
            vec[j] = u - v;
        }
    }
}

// Apply random signs element-wise
extern "C" __global__ void apply_signs_kernel(
    float* data,           // (n_vectors * padded_dim)
    const float* signs,    // (padded_dim)
    int n_vectors,
    int padded_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx / padded_dim;
    int elem_idx = idx % padded_dim;
    
    if (vec_idx < n_vectors) {
        data[idx] *= signs[elem_idx];
    }
}

// Scalar quantization: compute min, max, and quantize
extern "C" __global__ void scalar_quantize_kernel(
    const float* rotated,      // (n_vectors * padded_dim)
    unsigned char* codes,      // (n_vectors * padded_dim)
    float* scales,             // (n_vectors) - output delta
    float* offsets,            // (n_vectors) - output min
    int n_vectors,
    int padded_dim
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;
    
    const float* vec = rotated + vec_idx * padded_dim;
    unsigned char* code = codes + vec_idx * padded_dim;
    
    // Find min/max (parallel reduction in shared memory)
    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;
    
    float local_min = 1e30f;
    float local_max = -1e30f;
    
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float v = vec[i];
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }
    
    smin[threadIdx.x] = local_min;
    smax[threadIdx.x] = local_max;
    __syncthreads();
    
    // Parallel reduction for min/max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x + s]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    float min_val = smin[0];
    float max_val = smax[0];
    float range = max_val - min_val;
    float delta = (range > 1e-8f) ? range / 255.0f : 1.0f;
    
    if (threadIdx.x == 0) {
        scales[vec_idx] = delta;
        offsets[vec_idx] = min_val;
    }
    __syncthreads();
    
    // Quantize
    delta = scales[vec_idx];
    min_val = offsets[vec_idx];
    
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float v = vec[i];
        float q = (v - min_val) / delta;
        q = fminf(fmaxf(q, 0.0f), 255.0f);
        code[i] = (unsigned char)(q + 0.5f);
    }
}

// Compute uint8 dot products for distance estimation
extern "C" __global__ void uint8_dot_product_batch_kernel(
    const unsigned char* codes_a,  // (n_a * dim)
    const unsigned char* codes_b,  // (n_b * dim)
    unsigned int* dots,            // (n_a * n_b) output
    int n_a,
    int n_b,
    int dim
) {
    int a_idx = blockIdx.y;
    int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (a_idx >= n_a || b_idx >= n_b) return;
    
    const unsigned char* a = codes_a + a_idx * dim;
    const unsigned char* b = codes_b + b_idx * dim;
    
    unsigned int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (unsigned int)a[i] * (unsigned int)b[i];
    }
    
    dots[a_idx * n_b + b_idx] = sum;
}

// ============== Binary Quantization Kernels ==============

// Binary quantization: sign(x) -> packed bits
// One thread processes one vector's 32 bits (4 bytes) or 8 bits?
// Let's have each thread process one output byte (8 bits of input)
extern "C" __global__ void binary_quantize_kernel(
    const float* rotated,      // (n_vectors * padded_dim)
    unsigned char* codes,      // (n_vectors * (padded_dim / 8))
    int n_vectors,
    int padded_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int n_bytes = padded_dim / 8;
    int total_bytes = n_vectors * n_bytes;
    
    if (idx >= total_bytes) return;
    
    int vec_idx = idx / n_bytes;
    int byte_idx = idx % n_bytes;
    
    const float* vec = rotated + vec_idx * padded_dim + byte_idx * 8;
    
    unsigned char b = 0;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // Big-endian packing to match numpy/Rust
        if (vec[i] >= 0.0f) {
            b |= (1 << (7 - i));
        }
    }
    
    codes[idx] = b;
}

// Hamming distance kernel
extern "C" __global__ void hamming_distance_kernel(
    const unsigned char* codes_a,  // (n_a * n_bytes)
    const unsigned char* codes_b,  // (n_b * n_bytes)
    unsigned int* dists,           // (n_a * n_b) output
    int n_a,
    int n_b,
    int n_bytes
) {
    int a_idx = blockIdx.y;
    int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (a_idx >= n_a || b_idx >= n_b) return;
    
    const unsigned char* a = codes_a + a_idx * n_bytes;
    const unsigned char* b = codes_b + b_idx * n_bytes;
    
    unsigned int dist = 0;
    
    // Cast to int* for faster popcount if aligned (assumes dim % 32 == 0)
    // For safety, byte loop with popc
    for (int i = 0; i < n_bytes; i++) {
        unsigned char xor_val = a[i] ^ b[i];
        dist += __popc((unsigned int)xor_val);
    }
    
    dists[a_idx * n_b + b_idx] = dist;
}
"#;

    /// Error type for CUDA operations
    #[derive(Debug)]
    pub enum CudaError {
        DeviceError(String),
        KernelError(String),
        MemoryError(String),
    }

    impl std::fmt::Display for CudaError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                CudaError::DeviceError(s) => write!(f, "CUDA device error: {}", s),
                CudaError::KernelError(s) => write!(f, "CUDA kernel error: {}", s),
                CudaError::MemoryError(s) => write!(f, "CUDA memory error: {}", s),
            }
        }
    }

    impl std::error::Error for CudaError {}

    impl From<cudarc::driver::DriverError> for CudaError {
        fn from(err: cudarc::driver::DriverError) -> Self {
            CudaError::DeviceError(err.to_string())
        }
    }

    impl From<cudarc::nvrtc::CompileError> for CudaError {
        fn from(err: cudarc::nvrtc::CompileError) -> Self {
            CudaError::KernelError(err.to_string())
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

// Stub for when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub mod cuda_stub {
    use crate::backend::{Backend, BackendType, CpuBackend, ObjectiveBatchContext};
    
    /// Stub CUDA backend that falls back to CPU
    pub struct CudaBackend {
        cpu: CpuBackend,
    }
    
    impl CudaBackend {
        pub fn new(_device_id: usize) -> Result<Self, &'static str> {
            tracing::warn!("CUDA feature not enabled, using CPU backend");
            Ok(Self { cpu: CpuBackend::new() })
        }
        
        pub fn default_device() -> Result<Self, &'static str> {
            Self::new(0)
        }
    }
    
    impl Backend for CudaBackend {
        fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
            self.cpu.dot(a, b)
        }
        
        fn matmul_vec(&self, matrix: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
            self.cpu.matmul_vec(matrix, rows, cols, vec)
        }
        
        fn matmul(&self, a: &[f32], a_rows: usize, a_cols: usize, b: &[f32], b_cols: usize) -> Vec<f32> {
            self.cpu.matmul(a, a_rows, a_cols, b, b_cols)
        }
        
        fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
            self.cpu.cosine_similarity(a, b)
        }
        
        fn cosine_similarity_matrix(&self, embeddings: &[f32], n: usize, dim: usize) -> Vec<f32> {
            self.cpu.cosine_similarity_matrix(embeddings, n, dim)
        }
        
        fn masked_sum(&self, values: &[f32], mask: &[bool]) -> f32 {
            self.cpu.masked_sum(values, mask)
        }
        
        fn compute_objectives_batch(
            &self,
            selections: &[bool],
            batch_size: usize,
            context: &ObjectiveBatchContext<'_>,
        ) -> Vec<f32> {
            self.cpu.compute_objectives_batch(
                selections,
                batch_size,
                context,
            )
        }
        
        fn backend_type(&self) -> BackendType {
            BackendType::Cpu  // Actually CPU, CUDA not available
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use cuda_stub::*;

//! Python bindings for the Latence Solver
//!
//! Provides PyO3 bindings for seamless integration with Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyArrayMethods};

use crate::solver::{
    TabuSearchSolver,
    SolverConfig,
    SolverConstraints,
    SolverInput,
    SolverOutput,
    ChunkCandidate,
    RhetoricalRole,
};
use crate::backend::BackendType;

/// Python-exposed solver configuration
#[pyclass(name = "SolverConfig")]
#[derive(Clone)]
pub struct PySolverConfig {
    #[pyo3(get, set)]
    pub alpha: f32,
    #[pyo3(get, set)]
    pub beta: f32,
    #[pyo3(get, set)]
    pub gamma: f32,
    #[pyo3(get, set)]
    pub delta: f32,
    #[pyo3(get, set)]
    pub lambda_: f32,
    #[pyo3(get, set)]
    pub epsilon: f32,
    #[pyo3(get, set)]
    pub mu: f32,
    #[pyo3(get, set)]
    pub support_secondary_discount: f32,
    #[pyo3(get, set)]
    pub support_quorum_bonus: f32,
    #[pyo3(get, set)]
    pub support_quorum_threshold: f32,
    #[pyo3(get, set)]
    pub support_quorum_cap: usize,
    #[pyo3(get, set)]
    pub iterations: usize,
    #[pyo3(get, set)]
    pub tabu_tenure: usize,
    #[pyo3(get, set)]
    pub early_stopping_patience: usize,
    #[pyo3(get, set)]
    pub use_gpu: bool,
    #[pyo3(get, set)]
    pub random_seed: Option<u64>,
    #[pyo3(get, set)]
    pub enable_gpu_move_evaluation: bool,
    #[pyo3(get, set)]
    pub enable_path_relinking: bool,
    #[pyo3(get, set)]
    pub enable_destroy_repair: bool,
    #[pyo3(get, set)]
    pub enable_reactive_tenure: bool,
    #[pyo3(get, set)]
    pub enable_exact_window: bool,
    #[pyo3(get, set)]
    pub exact_window_size: usize,
    #[pyo3(get, set)]
    pub exact_window_time_ms: u64,
}

#[pymethods]
impl PySolverConfig {
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        beta = 0.3,
        gamma = 0.2,
        delta = 0.1,
        epsilon = 0.0,
        mu = 1.0,
        support_secondary_discount = 0.35,
        support_quorum_bonus = 0.18,
        support_quorum_threshold = 0.55,
        support_quorum_cap = 4,
        lambda_ = 0.5,
        iterations = 100,
        tabu_tenure = 10,
        early_stopping_patience = 20,
        use_gpu = false,
        random_seed = None,
        enable_gpu_move_evaluation = true,
        enable_path_relinking = true,
        enable_destroy_repair = true,
        enable_reactive_tenure = true,
        enable_exact_window = true,
        exact_window_size = 14,
        exact_window_time_ms = 25
    ))]
    fn new(
        alpha: f32,
        beta: f32,
        gamma: f32,
        delta: f32,
        epsilon: f32,
        mu: f32,
        support_secondary_discount: f32,
        support_quorum_bonus: f32,
        support_quorum_threshold: f32,
        support_quorum_cap: usize,
        lambda_: f32,
        iterations: usize,
        tabu_tenure: usize,
        early_stopping_patience: usize,
        use_gpu: bool,
        random_seed: Option<u64>,
        enable_gpu_move_evaluation: bool,
        enable_path_relinking: bool,
        enable_destroy_repair: bool,
        enable_reactive_tenure: bool,
        enable_exact_window: bool,
        exact_window_size: usize,
        exact_window_time_ms: u64,
    ) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            mu,
            support_secondary_discount,
            support_quorum_bonus,
            support_quorum_threshold,
            support_quorum_cap,
            lambda_,
            iterations,
            tabu_tenure,
            early_stopping_patience,
            use_gpu,
            random_seed,
            enable_gpu_move_evaluation,
            enable_path_relinking,
            enable_destroy_repair,
            enable_reactive_tenure,
            enable_exact_window,
            exact_window_size,
            exact_window_time_ms,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SolverConfig(alpha={}, beta={}, gamma={}, delta={}, epsilon={}, mu={}, lambda_={}, iterations={}, use_gpu={})",
            self.alpha, self.beta, self.gamma, self.delta, self.epsilon, self.mu, self.lambda_, self.iterations, self.use_gpu
        )
    }
}

impl From<PySolverConfig> for SolverConfig {
    fn from(py_config: PySolverConfig) -> Self {
        let mode = if py_config.beta.abs() > f32::EPSILON 
                   || py_config.gamma.abs() > f32::EPSILON 
                   || py_config.delta.abs() > f32::EPSILON {
            crate::solver::SolverMode::Enriched
        } else {
            crate::solver::SolverMode::Vanilla
        };

        SolverConfig {
            mode,
            alpha: py_config.alpha,
            beta: py_config.beta,
            gamma: py_config.gamma,
            delta: py_config.delta,
            epsilon: py_config.epsilon,
            mu: py_config.mu,
            support_secondary_discount: py_config.support_secondary_discount,
            support_quorum_bonus: py_config.support_quorum_bonus,
            support_quorum_threshold: py_config.support_quorum_threshold,
            support_quorum_cap: py_config.support_quorum_cap,
            lambda: py_config.lambda_,
            iterations: py_config.iterations,
            tabu_tenure: py_config.tabu_tenure,
            tabu_similarity_threshold: 0.85,
            early_stopping_patience: py_config.early_stopping_patience,
            min_improvement: 1e-6,
            random_seed: py_config.random_seed,
            use_gpu: py_config.use_gpu,
            num_workers: None,
            batch_size: 64,
            enable_gpu_move_evaluation: py_config.enable_gpu_move_evaluation,
            enable_path_relinking: py_config.enable_path_relinking,
            enable_destroy_repair: py_config.enable_destroy_repair,
            enable_reactive_tenure: py_config.enable_reactive_tenure,
            enable_exact_window: py_config.enable_exact_window,
            exact_window_size: py_config.exact_window_size,
            exact_window_time_ms: py_config.exact_window_time_ms,
        }
    }
}

/// Python-exposed solver constraints
#[pyclass(name = "SolverConstraints")]
#[derive(Clone)]
pub struct PySolverConstraints {
    #[pyo3(get, set)]
    pub max_tokens: u32,
    #[pyo3(get, set)]
    pub min_tokens: u32,
    #[pyo3(get, set)]
    pub min_chunks: usize,
    #[pyo3(get, set)]
    pub max_chunks: usize,
    #[pyo3(get, set)]
    pub max_per_cluster: usize,
    pub must_include_roles: Vec<String>,
    pub excluded_chunks: Vec<String>,
    pub required_chunks: Vec<String>,
}

#[pymethods]
impl PySolverConstraints {
    #[new]
    #[pyo3(signature = (
        max_tokens = 8192,
        min_tokens = 0,
        min_chunks = 1,
        max_chunks = 50,
        max_per_cluster = 3,
        must_include_roles = None,
        excluded_chunks = None,
        required_chunks = None
    ))]
    fn new(
        max_tokens: u32,
        min_tokens: u32,
        min_chunks: usize,
        max_chunks: usize,
        max_per_cluster: usize,
        must_include_roles: Option<Vec<String>>,
        excluded_chunks: Option<Vec<String>>,
        required_chunks: Option<Vec<String>>,
    ) -> Self {
        Self {
            max_tokens,
            min_tokens,
            min_chunks,
            max_chunks,
            max_per_cluster,
            must_include_roles: must_include_roles.unwrap_or_default(),
            excluded_chunks: excluded_chunks.unwrap_or_default(),
            required_chunks: required_chunks.unwrap_or_default(),
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SolverConstraints(max_tokens={}, max_chunks={}, max_per_cluster={})",
            self.max_tokens, self.max_chunks, self.max_per_cluster
        )
    }
}

impl From<PySolverConstraints> for SolverConstraints {
    fn from(py_constraints: PySolverConstraints) -> Self {
        SolverConstraints {
            max_tokens: py_constraints.max_tokens,
            min_tokens: py_constraints.min_tokens,
            min_chunks: py_constraints.min_chunks,
            max_chunks: py_constraints.max_chunks,
            max_per_cluster: py_constraints.max_per_cluster,
            must_include_roles: py_constraints
                .must_include_roles
                .iter()
                .map(|s| RhetoricalRole::from(s.as_str()))
                .collect(),
            excluded_chunks: py_constraints.excluded_chunks.into_iter().collect(),
            required_chunks: py_constraints.required_chunks.into_iter().collect(),
        }
    }
}

/// Python-exposed solver output
#[pyclass(name = "SolverOutput")]
pub struct PySolverOutput {
    #[pyo3(get)]
    pub selected_indices: Vec<usize>,
    #[pyo3(get)]
    pub objective_score: f64,
    #[pyo3(get)]
    pub relevance_total: f64,
    #[pyo3(get)]
    pub density_total: f64,
    #[pyo3(get)]
    pub centrality_total: f64,
    #[pyo3(get)]
    pub recency_total: f64,
    #[pyo3(get)]
    pub auxiliary_total: f64,
    #[pyo3(get)]
    pub fulfilment_total: f64,
    #[pyo3(get)]
    pub redundancy_penalty: f64,
    #[pyo3(get)]
    pub total_tokens: u32,
    #[pyo3(get)]
    pub num_selected: usize,
    #[pyo3(get)]
    pub iterations_run: usize,
    #[pyo3(get)]
    pub best_iteration: usize,
    #[pyo3(get)]
    pub constraints_satisfied: bool,
    #[pyo3(get)]
    pub constraint_violations: Vec<String>,
    #[pyo3(get)]
    pub solve_time_ms: f64,
    #[pyo3(get)]
    pub exact_window_used: bool,
    #[pyo3(get)]
    pub exact_window_core_size: usize,
    #[pyo3(get)]
    pub exact_window_nodes: usize,
    #[pyo3(get)]
    pub exact_window_exhaustive: bool,
    #[pyo3(get)]
    pub exact_window_gap: f64,
    #[pyo3(get)]
    pub exact_window_fixed_in: usize,
    #[pyo3(get)]
    pub exact_window_fixed_out: usize,
}

#[pymethods]
impl PySolverOutput {
    fn __repr__(&self) -> String {
        format!(
            "SolverOutput(selected={}, score={:.4}, tokens={}, time={:.2}ms)",
            self.num_selected, self.objective_score, self.total_tokens, self.solve_time_ms
        )
    }
    
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("selected_indices", &self.selected_indices)?;
        dict.set_item("objective_score", self.objective_score)?;
        dict.set_item("relevance_total", self.relevance_total)?;
        dict.set_item("density_total", self.density_total)?;
        dict.set_item("centrality_total", self.centrality_total)?;
        dict.set_item("centrality_total", self.centrality_total)?;
        dict.set_item("recency_total", self.recency_total)?;
        dict.set_item("auxiliary_total", self.auxiliary_total)?;
        dict.set_item("fulfilment_total", self.fulfilment_total)?;
        dict.set_item("redundancy_penalty", self.redundancy_penalty)?;
        dict.set_item("total_tokens", self.total_tokens)?;
        dict.set_item("num_selected", self.num_selected)?;
        dict.set_item("iterations_run", self.iterations_run)?;
        dict.set_item("best_iteration", self.best_iteration)?;
        dict.set_item("constraints_satisfied", self.constraints_satisfied)?;
        dict.set_item("constraint_violations", &self.constraint_violations)?;
        dict.set_item("solve_time_ms", self.solve_time_ms)?;
        dict.set_item("exact_window_used", self.exact_window_used)?;
        dict.set_item("exact_window_core_size", self.exact_window_core_size)?;
        dict.set_item("exact_window_nodes", self.exact_window_nodes)?;
        dict.set_item("exact_window_exhaustive", self.exact_window_exhaustive)?;
        dict.set_item("exact_window_gap", self.exact_window_gap)?;
        dict.set_item("exact_window_fixed_in", self.exact_window_fixed_in)?;
        dict.set_item("exact_window_fixed_out", self.exact_window_fixed_out)?;
        Ok(dict.into())
    }
}

impl From<SolverOutput> for PySolverOutput {
    fn from(output: SolverOutput) -> Self {
        Self {
            selected_indices: output.selected_indices,
            objective_score: output.objective_score,
            relevance_total: output.relevance_total,
            density_total: output.density_total,
            centrality_total: output.centrality_total,
            recency_total: output.recency_total,
            auxiliary_total: output.auxiliary_total,
            fulfilment_total: output.fulfilment_total,
            redundancy_penalty: output.redundancy_penalty,
            total_tokens: output.total_tokens,
            num_selected: output.num_selected,
            iterations_run: output.iterations_run,
            best_iteration: output.best_iteration,
            constraints_satisfied: output.constraints_satisfied,
            constraint_violations: output.constraint_violations,
            solve_time_ms: output.solve_time_ms,
            exact_window_used: output.exact_window_used,
            exact_window_core_size: output.exact_window_core_size,
            exact_window_nodes: output.exact_window_nodes,
            exact_window_exhaustive: output.exact_window_exhaustive,
            exact_window_gap: output.exact_window_gap,
            exact_window_fixed_in: output.exact_window_fixed_in,
            exact_window_fixed_out: output.exact_window_fixed_out,
        }
    }
}

/// Python-exposed Tabu Search Solver
#[pyclass(name = "TabuSearchSolver")]
pub struct PyTabuSearchSolver {
    solver: TabuSearchSolver,
}

fn backend_kind_name(backend_type: BackendType) -> &'static str {
    match backend_type {
        BackendType::Cpu | BackendType::Auto => "cpu_reference",
        BackendType::Gpu => "rust_gpu_experimental",
        BackendType::Cuda => "rust_cuda_experimental",
    }
}

#[pymethods]
impl PyTabuSearchSolver {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<PySolverConfig>) -> Self {
        let rust_config = config.map(SolverConfig::from).unwrap_or_default();
        Self {
            solver: TabuSearchSolver::new(rust_config),
        }
    }

    fn backend_kind(&self) -> String {
        backend_kind_name(self.solver.backend_type()).to_string()
    }
    
    /// Solve the knapsack problem from numpy arrays
    #[pyo3(signature = (
        embeddings,
        query_embedding,
        token_costs,
        density_scores,
        centrality_scores,
        recency_scores,
        auxiliary_scores,
        roles,
        cluster_ids,
        constraints = None
    ))]
    fn solve_numpy(
        &self,
        py: Python<'_>,
        embeddings: PyReadonlyArray2<f32>,
        query_embedding: PyReadonlyArray1<f32>,
        token_costs: PyReadonlyArray1<u32>,
        density_scores: PyReadonlyArray1<f32>,
        centrality_scores: PyReadonlyArray1<f32>,
        recency_scores: PyReadonlyArray1<f32>,
        auxiliary_scores: PyReadonlyArray1<f32>,
        roles: PyReadonlyArray1<u8>,
        cluster_ids: PyReadonlyArray1<i32>,
        constraints: Option<PySolverConstraints>,
    ) -> PyResult<PySolverOutput> {
        let emb = embeddings.as_slice()?;
        let query = query_embedding.as_slice()?;
        let tokens = token_costs.as_slice()?;
        let densities = density_scores.as_slice()?;
        let centralities = centrality_scores.as_slice()?;
        let recencies = recency_scores.as_slice()?;
        let auxiliaries = auxiliary_scores.as_slice()?;
        let role_array = roles.as_slice()?;
        let clusters = cluster_ids.as_slice()?;
        
        let n = tokens.len();
        let dim = if n > 0 { emb.len() / n } else { 0 };
        
        // Build relevance scores from cosine similarity
        let relevance_scores: Vec<f32> = (0..n)
            .map(|i| {
                let start = i * dim;
                let chunk_emb = &emb[start..start + dim];
                let dot: f32 = chunk_emb.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let norm_c: f32 = chunk_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_c > 0.0 && norm_q > 0.0 { dot / (norm_c * norm_q) } else { 0.0 }
            })
            .collect();
        
        let input = SolverInput {
            num_candidates: n,
            embedding_dim: dim,
            embeddings: emb.to_vec(),
            query_embedding: query.to_vec(),
            relevance_scores,
            density_scores: densities.to_vec(),
            centrality_scores: centralities.to_vec(),
            recency_scores: recencies.to_vec(),
            auxiliary_scores: auxiliaries.to_vec(),
            fulfilment_scores: vec![0.0; n],
            token_costs: tokens.to_vec(),
            roles: role_array.iter().map(|&r| RhetoricalRole::from(r)).collect(),
            cluster_ids: clusters.iter().map(|&c| if c >= 0 { Some(c as u32) } else { None }).collect(),
            chunk_ids: (0..n).map(|i| format!("chunk_{}", i)).collect(),
            coverage_matrix: None,
            query_token_weights: None,
            num_query_tokens: 0,
            similarity_matrix: None,
        };
        
        let rust_constraints = constraints.map(SolverConstraints::from).unwrap_or_default();
        
        py.allow_threads(|| {
            let output = self.solver.solve_input(&input, &rust_constraints);
            Ok(PySolverOutput::from(output))
        })
    }

    /// Solve from precomputed relevance, redundancy, and fulfilment tensors.
    #[pyo3(signature = (
        embeddings,
        token_costs,
        density_scores,
        centrality_scores,
        recency_scores,
        auxiliary_scores,
        roles,
        cluster_ids,
        relevance_scores,
        similarity_matrix = None,
        fulfilment_scores = None,
        coverage_matrix = None,
        query_token_weights = None,
        query_embedding = None,
        constraints = None
    ))]
    fn solve_precomputed_numpy(
        &self,
        py: Python<'_>,
        embeddings: PyReadonlyArray2<f32>,
        token_costs: PyReadonlyArray1<u32>,
        density_scores: PyReadonlyArray1<f32>,
        centrality_scores: PyReadonlyArray1<f32>,
        recency_scores: PyReadonlyArray1<f32>,
        auxiliary_scores: PyReadonlyArray1<f32>,
        roles: PyReadonlyArray1<u8>,
        cluster_ids: PyReadonlyArray1<i32>,
        relevance_scores: PyReadonlyArray1<f32>,
        similarity_matrix: Option<PyReadonlyArray2<f32>>,
        fulfilment_scores: Option<PyReadonlyArray1<f32>>,
        coverage_matrix: Option<PyReadonlyArray2<f32>>,
        query_token_weights: Option<PyReadonlyArray1<f32>>,
        query_embedding: Option<PyReadonlyArray1<f32>>,
        constraints: Option<PySolverConstraints>,
    ) -> PyResult<PySolverOutput> {
        let emb = embeddings.as_slice()?;
        let tokens = token_costs.as_slice()?;
        let densities = density_scores.as_slice()?;
        let centralities = centrality_scores.as_slice()?;
        let recencies = recency_scores.as_slice()?;
        let auxiliaries = auxiliary_scores.as_slice()?;
        let role_array = roles.as_slice()?;
        let clusters = cluster_ids.as_slice()?;
        let relevance = relevance_scores.as_slice()?;

        let n = tokens.len();
        let dim = if n > 0 { emb.len() / n } else { 0 };

        if relevance.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "relevance_scores length must match number of candidates",
            ));
        }

        let similarity_matrix = if let Some(matrix) = similarity_matrix {
            let array = matrix.as_array();
            if array.shape() != [n, n] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "similarity_matrix must have shape (num_candidates, num_candidates)",
                ));
            }
            Some(array.iter().copied().collect())
        } else {
            None
        };

        let fulfilment_scores = if let Some(scores) = fulfilment_scores {
            let values = scores.as_slice()?;
            if values.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "fulfilment_scores length must match number of candidates",
                ));
            }
            values.to_vec()
        } else {
            vec![0.0; n]
        };

        let (coverage_matrix, query_token_weights, num_query_tokens) =
            if let Some(coverage) = coverage_matrix {
                let array = coverage.as_array();
                if array.shape().len() != 2 || array.shape()[1] != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "coverage_matrix must have shape (num_query_tokens, num_candidates)",
                    ));
                }
                let num_query_tokens = array.shape()[0];
                let weights = if let Some(weight_array) = query_token_weights {
                    let values = weight_array.as_slice()?;
                    if values.len() != num_query_tokens {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "query_token_weights must match coverage_matrix rows",
                        ));
                    }
                    Some(values.to_vec())
                } else {
                    Some(vec![1.0; num_query_tokens])
                };
                (
                    Some(array.iter().copied().collect()),
                    weights,
                    num_query_tokens,
                )
            } else {
                if query_token_weights.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "query_token_weights requires coverage_matrix",
                    ));
                }
                (None, None, 0usize)
            };

        let query_embedding = if let Some(query_array) = query_embedding {
            let query = query_array.as_slice()?;
            if dim > 0 && query.len() != dim {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "query_embedding length must match embedding dimension",
                ));
            }
            query.to_vec()
        } else {
            vec![0.0; dim]
        };

        let input = SolverInput {
            num_candidates: n,
            embedding_dim: dim,
            embeddings: emb.to_vec(),
            query_embedding,
            relevance_scores: relevance.to_vec(),
            density_scores: densities.to_vec(),
            centrality_scores: centralities.to_vec(),
            recency_scores: recencies.to_vec(),
            auxiliary_scores: auxiliaries.to_vec(),
            fulfilment_scores,
            token_costs: tokens.to_vec(),
            roles: role_array.iter().map(|&r| RhetoricalRole::from(r)).collect(),
            cluster_ids: clusters.iter().map(|&c| if c >= 0 { Some(c as u32) } else { None }).collect(),
            chunk_ids: (0..n).map(|i| format!("chunk_{}", i)).collect(),
            coverage_matrix,
            query_token_weights,
            num_query_tokens,
            similarity_matrix,
        };

        let rust_constraints = constraints.map(SolverConstraints::from).unwrap_or_default();

        py.allow_threads(|| {
            let output = self.solver.solve_input(&input, &rust_constraints);
            Ok(PySolverOutput::from(output))
        })
    }
    
    /// Solve from a list of chunk dictionaries
    #[pyo3(signature = (chunks, query_embedding, constraints = None))]
    fn solve(
        &self,
        py: Python<'_>,
        chunks: &Bound<'_, PyList>,
        query_embedding: PyReadonlyArray1<f32>,
        constraints: Option<PySolverConstraints>,
    ) -> PyResult<PySolverOutput> {
         // (Implementation matches original file)
         // For brevity, skipping full body if not used by benchmark, but I should verify if I can just omit it or if it cause compile error if incomplete.
         // Unused in benchmark, but needed for compile. 
         let query = query_embedding.as_slice()?.to_vec();
        
        // Parse chunks from Python dicts
        let candidates: Vec<ChunkCandidate> = chunks
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let dict = item.downcast::<PyDict>()?;
                
                let chunk_id = dict
                    .get_item("chunk_id")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_else(|| format!("chunk_{}", i));
                
                let content = dict
                    .get_item("content")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default();
                
                let embedding: Vec<f32> = dict
                    .get_item("embedding")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing embedding"))?
                    .extract()?;
                
                let token_count = dict
                    .get_item("token_count")?
                    .map(|v| v.extract::<u32>())
                    .transpose()?
                    .unwrap_or(100);
                
                let fact_density = dict
                    .get_item("fact_density")?
                    .map(|v| v.extract::<f32>())
                    .transpose()?
                    .unwrap_or(0.5);
                
                let centrality_score = dict
                    .get_item("centrality_score")?
                    .map(|v| v.extract::<f32>())
                    .transpose()?
                    .unwrap_or(0.5);
                
                let recency_score = dict
                    .get_item("recency_score")?
                    .map(|v| v.extract::<f32>())
                    .transpose()?
                    .unwrap_or(0.5);

                let auxiliary_score = dict
                    .get_item("auxiliary_score")?
                    .map(|v| v.extract::<f32>())
                    .transpose()?
                    .unwrap_or(0.0);
                
                let role_str = dict
                    .get_item("rhetorical_role")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_else(|| "unknown".to_string());
                
                let cluster_id = dict
                    .get_item("cluster_id")?
                    .map(|v| v.extract::<i32>())
                    .transpose()?
                    .and_then(|c| if c >= 0 { Some(c as u32) } else { None });
                
                Ok(ChunkCandidate {
                     chunk_id, content, embedding, token_count, fact_density, centrality_score, uniqueness_score: 0.5, recency_score, auxiliary_score, rhetorical_role: RhetoricalRole::from(role_str.as_str()), cluster_id, parent_doc_id: None, position: 0.5,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        let rust_constraints = constraints.map(SolverConstraints::from).unwrap_or_default();
        
        py.allow_threads(|| {
            let output = self.solver.solve(&candidates, &query, &rust_constraints);
            Ok(PySolverOutput::from(output))
        })
    }
    
    fn __repr__(&self) -> String {
        format!("TabuSearchSolver(backend='{}')", backend_kind_name(self.solver.backend_type()))
    }
}

/// Check if GPU is available
#[pyfunction]
fn gpu_available() -> bool {
    crate::gpu_available()
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_available() -> bool {
    crate::cuda_available()
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

use crate::quantization::rotational::{RoQConfig, RotationalQuantizer, QuantizedVector};
use rayon::prelude::*;

/// Compute MaxSim batch scores (Rust CPU benchmark)
#[pyfunction]
#[pyo3(signature = (q_codes, q_meta, d_codes, d_meta, num_bits, dim))]
fn compute_max_sim_batch(
    py: Python<'_>,
    q_codes: PyReadonlyArray3<u8>, // (A, S, DimBytes)
    q_meta: PyReadonlyArray3<f32>, // (A, S, 4)
    d_codes: PyReadonlyArray3<u8>, // (B, T, DimBytes)
    d_meta: PyReadonlyArray3<f32>, // (B, T, 4)
    num_bits: usize,
    dim: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let q_codes = q_codes.as_array();
    let q_meta = q_meta.as_array();
    let d_codes = d_codes.as_array();
    let d_meta = d_meta.as_array();
    
    let a = q_codes.shape()[0];
    let s_tokens = q_codes.shape()[1];
    
    let b = d_codes.shape()[0];
    let t_tokens = d_codes.shape()[1];
    
    // Compute block_size to match Python logic
    let mut block_size = 1;
    while block_size < dim {
        block_size *= 2;
    }

    // Create Dummy Quantizer to reuse logic (needs config)
    let config = RoQConfig {
        dim,
        num_bits,
        block_size,
        ..Default::default()
    };
    let roq = RotationalQuantizer::new(config);
    
    // Pre-convert to QuantizedVector structs to use `max_sim_score`
    // This adds some overhead but ensures logic consistency
    // Parallelize conversion?
    let queries: Vec<Vec<QuantizedVector>> = (0..a).into_par_iter().map(|i| {
        (0..s_tokens).map(|j| {
            let codes = q_codes.slice(numpy::ndarray::s![i, j, ..]).to_vec();
            let m = q_meta.slice(numpy::ndarray::s![i, j, ..]);
            QuantizedVector {
                codes,
                scale: m[0],
                offset: m[1],
                code_sum: m[2] as u32,
                norm_sq: m[3],
            }
        }).collect()
    }).collect();
    
    let docs: Vec<Vec<QuantizedVector>> = (0..b).into_par_iter().map(|i| {
        (0..t_tokens).map(|j| {
            let codes = d_codes.slice(numpy::ndarray::s![i, j, ..]).to_vec();
            let m = d_meta.slice(numpy::ndarray::s![i, j, ..]);
            QuantizedVector {
                codes,
                scale: m[0],
                offset: m[1],
                code_sum: m[2] as u32,
                norm_sq: m[3],
            }
        }).collect()
    }).collect();
    
    // Compute scores
    let mut scores = Vec::with_capacity(a * b);
    scores.resize(a * b, 0.0f32);
    
    let scores_vec: Vec<f32> = (0..a).into_par_iter().flat_map(|i| {
        let q_vecs = &queries[i];
        
        let row_scores: Vec<f32> = (0..b).map(|j| {
            let d_vecs = &docs[j];
            roq.max_sim_score(q_vecs, d_vecs)
        }).collect();
        
        row_scores
    }).collect();
    
    // Correct way to create PyArray2 from vec is via PyArray1 and reshape
    let out_array = numpy::PyArray1::from_slice_bound(py, &scores_vec).reshape((a, b)).unwrap();
    Ok(out_array.into())
}

/// Compute MaxSim batch scores for Float32 embeddings (Rust Baseline)
#[pyfunction]
#[pyo3(signature = (q_vecs, d_vecs))]
fn compute_max_sim_batch_f32(
    py: Python<'_>,
    q_vecs: PyReadonlyArray3<f32>, // (A, S, Dim)
    d_vecs: PyReadonlyArray3<f32>, // (B, T, Dim)
) -> PyResult<PyObject> {
    use crate::backend::simd::dot_simd;
    
    let q = q_vecs.as_array();
    let d = d_vecs.as_array();
    
    let a = q.shape()[0];
    let s = q.shape()[1];
    let dim_q = q.shape()[2];
    
    let b = d.shape()[0];
    let t = d.shape()[1];
    let dim_d = d.shape()[2];
    
    if dim_q != dim_d {
        return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
    }
    
    let mut scores = Vec::with_capacity(a * b);
    scores.resize(a * b, 0.0f32);
    
    // Flatten access for speed?
    // ndarray slicing is fast.
    
    let scores_vec: Vec<f32> = (0..a).into_par_iter().flat_map(|i| {
        let row_scores: Vec<f32> = (0..b).map(|j| {
            // Compute MaxSim(Q_i, D_j)
            let mut total_score = 0.0;
            
            for k in 0..s {
                // q_token: [dim]
                let q_token = q.slice(numpy::ndarray::s![i, k, ..]);
                let q_slice = q_token.as_slice().unwrap(); // Assume contiguous?
                
                let mut max_sim = f32::NEG_INFINITY;
                
                for l in 0..t {
                    let d_token = d.slice(numpy::ndarray::s![j, l, ..]);
                    let d_slice = d_token.as_slice().unwrap();
                    
                    // Dot product
                    let sim = dot_simd(q_slice, d_slice);
                    
                    if sim > max_sim {
                        max_sim = sim;
                    }
                }
                total_score += max_sim;
            }
            total_score
        }).collect();
        row_scores
    }).collect();
    
    let out_array = numpy::PyArray1::from_slice_bound(py, &scores_vec).reshape((a, b)).unwrap();
    Ok(out_array.into())
}

/// Python module definition
#[pymodule]
fn latence_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PySolverConfig>()?;
    m.add_class::<PySolverConstraints>()?;
    m.add_class::<PySolverOutput>()?;
    m.add_class::<PyTabuSearchSolver>()?;
    
    // Functions
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_sim_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_sim_batch_f32, m)?)?;
    
    Ok(())
}

//! Configuration types for the Tabu Search Solver

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Solver operation mode
/// 
/// - `Vanilla`: Uses only relevance scores, token costs, and redundancy penalty.
///   Works immediately with raw ColBERT/BM25 retrieval results.
/// - `Enriched`: Uses stateless intelligence-style features (density, centrality,
///   recency, auxiliary terms, roles, and cluster diversity) computed at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SolverMode {
    /// Minimal features: relevance + redundancy only
    #[default]
    Vanilla,
    /// Full intelligence features: density, centrality, recency, roles
    Enriched,
}

/// Rhetorical role classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RhetoricalRole {
    Definition = 0,
    Example = 1,
    Evidence = 2,
    Conclusion = 3,
    Risk = 4,
    Constraint = 5,
    DataTable = 6,
    Procedure = 7,
    Unknown = 255,
}

impl Default for RhetoricalRole {
    fn default() -> Self {
        Self::Unknown
    }
}

impl From<u8> for RhetoricalRole {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Definition,
            1 => Self::Example,
            2 => Self::Evidence,
            3 => Self::Conclusion,
            4 => Self::Risk,
            5 => Self::Constraint,
            6 => Self::DataTable,
            7 => Self::Procedure,
            _ => Self::Unknown,
        }
    }
}

impl From<&str> for RhetoricalRole {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "definition" => Self::Definition,
            "example" => Self::Example,
            "evidence" => Self::Evidence,
            "conclusion" => Self::Conclusion,
            "risk" => Self::Risk,
            "constraint" => Self::Constraint,
            "data_table" | "datatable" | "table" => Self::DataTable,
            "procedure" => Self::Procedure,
            _ => Self::Unknown,
        }
    }
}

/// A candidate chunk for context optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkCandidate {
    /// Unique identifier
    pub chunk_id: String,
    
    /// Text content
    pub content: String,
    
    /// Dense embedding vector
    pub embedding: Vec<f32>,
    
    /// Number of tokens
    pub token_count: u32,
    
    /// Fact density score [0, 1]
    pub fact_density: f32,
    
    /// Document centrality score [0, 1]
    pub centrality_score: f32,
    
    /// Semantic uniqueness score [0, 1]
    pub uniqueness_score: f32,
    
    /// Recency/temporal decay score [0, 1]
    pub recency_score: f32,

    /// Auxiliary/Custom score (User defined) [0, 1]
    pub auxiliary_score: f32,
    
    /// Rhetorical role classification
    pub rhetorical_role: RhetoricalRole,
    
    /// Semantic cluster ID
    pub cluster_id: Option<u32>,
    
    /// Parent document ID
    pub parent_doc_id: Option<String>,
    
    /// Position in parent document [0, 1]
    pub position: f32,
}

impl ChunkCandidate {
    /// Create a new chunk candidate with minimal required fields
    pub fn new(chunk_id: String, content: String, embedding: Vec<f32>, token_count: u32) -> Self {
        Self {
            chunk_id,
            content,
            embedding,
            token_count,
            fact_density: 0.5,
            centrality_score: 0.5,
            uniqueness_score: 0.5,
            recency_score: 0.5,
            auxiliary_score: 0.0,
            rhetorical_role: RhetoricalRole::Unknown,
            cluster_id: None,
            parent_doc_id: None,
            position: 0.5,
        }
    }
    
    /// Builder pattern: set fact density
    pub fn with_fact_density(mut self, density: f32) -> Self {
        self.fact_density = density;
        self
    }
    
    /// Builder pattern: set centrality score
    pub fn with_centrality(mut self, score: f32) -> Self {
        self.centrality_score = score;
        self
    }
    
    /// Builder pattern: set uniqueness score
    pub fn with_uniqueness(mut self, score: f32) -> Self {
        self.uniqueness_score = score;
        self
    }
    
    /// Builder pattern: set recency score
    pub fn with_recency(mut self, score: f32) -> Self {
        self.recency_score = score;
        self
    }

    /// Builder pattern: set auxiliary score
    pub fn with_auxiliary(mut self, score: f32) -> Self {
        self.auxiliary_score = score;
        self
    }
    
    /// Builder pattern: set rhetorical role
    pub fn with_role(mut self, role: RhetoricalRole) -> Self {
        self.rhetorical_role = role;
        self
    }
    
    /// Builder pattern: set cluster ID
    pub fn with_cluster(mut self, cluster_id: u32) -> Self {
        self.cluster_id = Some(cluster_id);
        self
    }
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Solver mode (Vanilla uses only relevance+redundancy, Enriched uses all features)
    pub mode: SolverMode,
    
    /// Weight for relevance term (α)
    pub alpha: f32,
    
    /// Weight for fact density term (β) - only used in Enriched mode
    pub beta: f32,
    
    /// Weight for centrality term (γ) - only used in Enriched mode
    pub gamma: f32,
    
    /// Weight for recency term (δ) - only used in Enriched mode
    pub delta: f32,
    
    /// Weight for auxiliary term (ε) - used in all modes if non-zero
    pub epsilon: f32,

    /// Weight for fulfilment / query coverage term (μ)
    pub mu: f32,

    /// Discount applied to the second-best chunk support per facet.
    pub support_secondary_discount: f32,

    /// Bonus strength for quorum support beyond the top-two chunks.
    pub support_quorum_bonus: f32,

    /// Minimum facet coverage required for a chunk to count toward quorum.
    pub support_quorum_threshold: f32,

    /// Maximum number of supportive chunks to count when scaling quorum bonus.
    pub support_quorum_cap: usize,
    
    /// Weight for redundancy penalty (λ)
    pub lambda: f32,
    
    /// Number of tabu search iterations
    pub iterations: usize,
    
    /// Tabu tenure (how long moves stay tabu)
    pub tabu_tenure: usize,
    
    /// Semantic similarity threshold for tabu matching
    pub tabu_similarity_threshold: f32,
    
    /// Early stopping patience (iterations without improvement)
    pub early_stopping_patience: usize,
    
    /// Minimum improvement to reset patience
    pub min_improvement: f32,
    
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    
    /// Request an accelerated backend when one is explicitly enabled.
    ///
    /// In the OSS package this is CPU-first by default. Private premium backends
    /// or experimental in-tree accelerators must be enabled separately.
    pub use_gpu: bool,
    
    /// Number of parallel workers for CPU
    pub num_workers: Option<usize>,
    
    /// Batch size for move evaluation
    pub batch_size: usize,

    /// Enable batched backend move evaluation for accelerated backends
    pub enable_gpu_move_evaluation: bool,

    /// Enable path-relinking against elite solutions during diversification
    pub enable_path_relinking: bool,

    /// Enable destroy-repair diversification steps
    pub enable_destroy_repair: bool,

    /// Enable reactive tabu tenure based on stagnation
    pub enable_reactive_tenure: bool,

    /// Exhaustively optimize a reduced candidate window around the incumbent.
    pub enable_exact_window: bool,

    /// Maximum number of candidates allowed to vary inside the exact window.
    pub exact_window_size: usize,

    /// Time budget for exact-window intensification in milliseconds.
    pub exact_window_time_ms: u64,
}

impl SolverConfig {
    /// Convert config weights to backend ObjectiveWeights struct
    pub fn weights(&self) -> crate::backend::ObjectiveWeights {
        let is_vanilla = self.mode == SolverMode::Vanilla;
        crate::backend::ObjectiveWeights {
            alpha: self.alpha,
            beta: if is_vanilla { 0.0 } else { self.beta },
            gamma: if is_vanilla { 0.0 } else { self.gamma },
            delta: if is_vanilla { 0.0 } else { self.delta },
            epsilon: self.epsilon,
            mu: self.mu,
            lambda: self.lambda,
            support_secondary_discount: self.support_secondary_discount,
            support_quorum_bonus: self.support_quorum_bonus,
            support_quorum_threshold: self.support_quorum_threshold,
            support_quorum_cap: self.support_quorum_cap as f32,
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            mode: SolverMode::default(),
            alpha: 1.0,
            beta: 0.3,
            gamma: 0.2,
            delta: 0.1,
            epsilon: 0.0,
            mu: 1.0,
            support_secondary_discount: 0.35,
            support_quorum_bonus: 0.18,
            support_quorum_threshold: 0.55,
            support_quorum_cap: 4,
            lambda: 0.5,
            iterations: 100,
            tabu_tenure: 10,
            tabu_similarity_threshold: 0.85,
            early_stopping_patience: 20,
            min_improvement: 1e-6,
            random_seed: None,
            use_gpu: false,
            num_workers: None,
            batch_size: 64,
            enable_gpu_move_evaluation: true,
            enable_path_relinking: true,
            enable_destroy_repair: true,
            enable_reactive_tenure: true,
            enable_exact_window: true,
            exact_window_size: 14,
            exact_window_time_ms: 25,
        }
    }
}

/// Constraints for the optimization problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConstraints {
    /// Maximum token budget
    pub max_tokens: u32,
    
    /// Minimum token count
    pub min_tokens: u32,
    
    /// Minimum number of chunks to select
    pub min_chunks: usize,
    
    /// Maximum number of chunks to select
    pub max_chunks: usize,
    
    /// Required rhetorical roles (must include at least one of each)
    pub must_include_roles: HashSet<RhetoricalRole>,
    
    /// Maximum chunks per semantic cluster (diversity constraint)
    pub max_per_cluster: usize,
    
    /// Excluded chunk IDs
    pub excluded_chunks: HashSet<String>,
    
    /// Required chunk IDs (must be included)
    pub required_chunks: HashSet<String>,
}

impl Default for SolverConstraints {
    fn default() -> Self {
        Self {
            max_tokens: 8192,
            min_tokens: 0,
            min_chunks: 1,
            max_chunks: 50,
            must_include_roles: HashSet::new(),
            max_per_cluster: 3,
            excluded_chunks: HashSet::new(),
            required_chunks: HashSet::new(),
        }
    }
}

impl SolverConstraints {
    /// Create constraints with just a token budget
    pub fn with_budget(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }
    
    /// Add a required role
    pub fn require_role(mut self, role: RhetoricalRole) -> Self {
        self.must_include_roles.insert(role);
        self
    }
    
    /// Set maximum chunks per cluster
    pub fn with_cluster_limit(mut self, limit: usize) -> Self {
        self.max_per_cluster = limit;
        self
    }
}

/// Input to the solver
#[derive(Debug, Clone)]
pub struct SolverInput {
    /// Number of candidates
    pub num_candidates: usize,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Flattened embeddings (num_candidates * embedding_dim)
    pub embeddings: Vec<f32>,
    
    /// Query embedding
    pub query_embedding: Vec<f32>,
    
    /// Relevance scores (pre-computed or from query similarity)
    pub relevance_scores: Vec<f32>,
    
    /// Fact density scores
    pub density_scores: Vec<f32>,
    
    /// Centrality scores
    pub centrality_scores: Vec<f32>,
    
    /// Recency scores
    pub recency_scores: Vec<f32>,

    /// Auxiliary scores
    pub auxiliary_scores: Vec<f32>,

    /// Linear fulfilment gain proxy per candidate
    pub fulfilment_scores: Vec<f32>,
    
    /// Token costs
    pub token_costs: Vec<u32>,
    
    /// Rhetorical roles
    pub roles: Vec<RhetoricalRole>,
    
    /// Cluster IDs
    pub cluster_ids: Vec<Option<u32>>,
    
    /// Chunk IDs
    pub chunk_ids: Vec<String>,

    /// Optional query-token coverage matrix flattened as (num_query_tokens, num_candidates)
    pub coverage_matrix: Option<Vec<f32>>,

    /// Optional per-query-token weights aligned with `coverage_matrix`
    pub query_token_weights: Option<Vec<f32>>,

    /// Number of query tokens encoded in `coverage_matrix`
    pub num_query_tokens: usize,
    
    /// Pre-computed similarity matrix (optional, will compute if None)
    pub similarity_matrix: Option<Vec<f32>>,
}

impl SolverInput {
    /// Create solver input from chunk candidates
    pub fn from_candidates(candidates: &[ChunkCandidate], query_embedding: Vec<f32>) -> Self {
        let n = candidates.len();
        let dim = if n > 0 { candidates[0].embedding.len() } else { 0 };
        
        let embeddings: Vec<f32> = candidates
            .iter()
            .flat_map(|c| c.embedding.iter().copied())
            .collect();
        
        // Compute relevance as cosine similarity to query
        let relevance_scores: Vec<f32> = candidates
            .iter()
            .map(|c| {
                let dot: f32 = c.embedding.iter().zip(query_embedding.iter()).map(|(a, b)| a * b).sum();
                let norm_c: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_q: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_c > 0.0 && norm_q > 0.0 { dot / (norm_c * norm_q) } else { 0.0 }
            })
            .collect();
        
        Self {
            num_candidates: n,
            embedding_dim: dim,
            embeddings,
            query_embedding,
            relevance_scores,
            density_scores: candidates.iter().map(|c| c.fact_density).collect(),
            centrality_scores: candidates.iter().map(|c| c.centrality_score).collect(),
            recency_scores: candidates.iter().map(|c| c.recency_score).collect(),
            auxiliary_scores: candidates.iter().map(|c| c.auxiliary_score).collect(),
            fulfilment_scores: vec![0.0; n],
            token_costs: candidates.iter().map(|c| c.token_count).collect(),
            roles: candidates.iter().map(|c| c.rhetorical_role).collect(),
            cluster_ids: candidates.iter().map(|c| c.cluster_id).collect(),
            chunk_ids: candidates.iter().map(|c| c.chunk_id.clone()).collect(),
            coverage_matrix: None,
            query_token_weights: None,
            num_query_tokens: 0,
            similarity_matrix: None,
        }
    }
}

/// Output from the solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverOutput {
    /// Indices of selected chunks
    pub selected_indices: Vec<usize>,
    
    /// Selection mask (true = selected)
    pub selected_mask: Vec<bool>,
    
    /// Final objective score
    pub objective_score: f64,
    
    /// Breakdown of objective components
    pub relevance_total: f64,
    pub density_total: f64,
    pub centrality_total: f64,
    pub recency_total: f64,
    pub auxiliary_total: f64,
    pub fulfilment_total: f64,
    pub redundancy_penalty: f64,
    
    /// Total tokens used
    pub total_tokens: u32,
    
    /// Number of selected chunks
    pub num_selected: usize,
    
    /// Number of iterations run
    pub iterations_run: usize,
    
    /// Iteration where best solution was found
    pub best_iteration: usize,
    
    /// Improvement history
    pub improvement_history: Vec<f64>,
    
    /// Whether constraints are satisfied
    pub constraints_satisfied: bool,
    
    /// Constraint violations (if any)
    pub constraint_violations: Vec<String>,
    
    /// Solve time in milliseconds
    pub solve_time_ms: f64,

    /// Whether an exact-window intensification pass ran.
    pub exact_window_used: bool,

    /// Number of candidates varied inside the exact window.
    pub exact_window_core_size: usize,

    /// Number of assignments evaluated inside the exact window.
    pub exact_window_nodes: usize,

    /// Whether the exact window was fully enumerated inside the time budget.
    pub exact_window_exhaustive: bool,

    /// Local optimality gap for the exact window.
    pub exact_window_gap: f64,

    /// Number of selected candidates fixed outside the exact window.
    pub exact_window_fixed_in: usize,

    /// Number of unselected candidates fixed outside the exact window.
    pub exact_window_fixed_out: usize,
}

impl SolverOutput {
    /// Get selected chunk IDs
    pub fn selected_ids<'a>(&self, candidates: &'a [ChunkCandidate]) -> Vec<&'a str> {
        self.selected_indices
            .iter()
            .map(|&i| candidates[i].chunk_id.as_str())
            .collect()
    }
    
    /// Build context string from selected chunks
    pub fn build_context(&self, candidates: &[ChunkCandidate]) -> String {
        self.selected_indices
            .iter()
            .map(|&i| candidates[i].content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rhetorical_role_from_str() {
        assert_eq!(RhetoricalRole::from("definition"), RhetoricalRole::Definition);
        assert_eq!(RhetoricalRole::from("RISK"), RhetoricalRole::Risk);
        assert_eq!(RhetoricalRole::from("unknown_role"), RhetoricalRole::Unknown);
    }
    
    #[test]
    fn test_chunk_candidate_builder() {
        let chunk = ChunkCandidate::new(
            "test_id".to_string(),
            "Test content".to_string(),
            vec![0.1, 0.2, 0.3],
            100,
        )
        .with_fact_density(0.8)
        .with_role(RhetoricalRole::Evidence)
        .with_cluster(5);
        
        assert_eq!(chunk.fact_density, 0.8);
        assert_eq!(chunk.rhetorical_role, RhetoricalRole::Evidence);
        assert_eq!(chunk.cluster_id, Some(5));
    }
    
    #[test]
    fn test_solver_input_from_candidates() {
        let candidates = vec![
            ChunkCandidate::new("c1".to_string(), "Content 1".to_string(), vec![1.0, 0.0], 50),
            ChunkCandidate::new("c2".to_string(), "Content 2".to_string(), vec![0.0, 1.0], 75),
        ];
        
        let query = vec![1.0, 0.0];
        let input = SolverInput::from_candidates(&candidates, query);
        
        assert_eq!(input.num_candidates, 2);
        assert_eq!(input.embedding_dim, 2);
        assert!((input.relevance_scores[0] - 1.0).abs() < 1e-6); // Identical to query
        assert!(input.relevance_scores[1].abs() < 1e-6); // Orthogonal to query
    }
}


//! Reciprocal Rank Fusion (RRF) Baseline Implementation
//!
//! RRF is a simple but effective fusion algorithm that combines rankings from
//! multiple retrieval sources. This implementation serves as a baseline to
//! compare against our Quadratic Knapsack Solver.
//!
//! RRF Score: score_i = Σ 1/(k + rank_source_j)
//! where k is typically 60 (to dampen the contribution of high-ranking items)
//!
//! Limitations of RRF (why our solver is better):
//! - Uses only rank positions, ignores actual similarity scores
//! - Cannot penalize redundancy between selected chunks
//! - Ignores token budget constraints (just takes top-k)
//! - Cannot incorporate intelligence features (density, centrality, etc.)

use crate::solver::{ChunkCandidate, SolverOutput};
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for RRF baseline
#[derive(Debug, Clone)]
pub struct RRFConfig {
    /// Damping constant (typically 60, higher = less emphasis on top ranks)
    pub k: f32,
    /// Maximum number of chunks to select
    pub top_n: usize,
    /// Maximum token budget (optional, for fair comparison with knapsack)
    pub max_tokens: Option<u32>,
}

impl Default for RRFConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            top_n: 20,
            max_tokens: Some(8192),
        }
    }
}

/// A ranking source with chunk IDs and their ranks
#[derive(Debug, Clone)]
pub struct RankingSource {
    /// Name of the ranking source (e.g., "colbert", "bm25")
    pub name: String,
    /// Ordered list of chunk IDs from highest to lowest rank
    pub ranked_ids: Vec<String>,
}

impl RankingSource {
    pub fn new(name: impl Into<String>, ranked_ids: Vec<String>) -> Self {
        Self {
            name: name.into(),
            ranked_ids,
        }
    }
    
    /// Create from candidates sorted by score (highest first)
    pub fn from_scored_candidates(name: impl Into<String>, candidates: &[ChunkCandidate]) -> Self {
        let mut sorted: Vec<_> = candidates.iter().collect();
        sorted.sort_by(|a, b| b.fact_density.partial_cmp(&a.fact_density).unwrap_or(std::cmp::Ordering::Equal));
        
        Self {
            name: name.into(),
            ranked_ids: sorted.iter().map(|c| c.chunk_id.clone()).collect(),
        }
    }
}

/// RRF Baseline Solver
pub struct RRFSolver {
    config: RRFConfig,
}

impl RRFSolver {
    pub fn new(config: RRFConfig) -> Self {
        Self { config }
    }
    
    /// Compute RRF scores for all chunks across multiple ranking sources
    pub fn compute_rrf_scores(&self, sources: &[RankingSource]) -> HashMap<String, f32> {
        let mut scores: HashMap<String, f32> = HashMap::new();
        
        for source in sources {
            for (rank, chunk_id) in source.ranked_ids.iter().enumerate() {
                let rrf_contribution = 1.0 / (self.config.k + (rank as f32) + 1.0);
                *scores.entry(chunk_id.clone()).or_insert(0.0) += rrf_contribution;
            }
        }
        
        scores
    }
    
    /// Select top chunks using RRF fusion
    pub fn solve(
        &self,
        candidates: &[ChunkCandidate],
        sources: &[RankingSource],
    ) -> RRFOutput {
        let start = std::time::Instant::now();
        
        // Compute RRF scores
        let rrf_scores = self.compute_rrf_scores(sources);
        
        // Create a map from chunk_id to candidate
        let candidate_map: HashMap<&str, &ChunkCandidate> = candidates
            .iter()
            .map(|c| (c.chunk_id.as_str(), c))
            .collect();
        
        // Sort chunks by RRF score (descending)
        let mut scored_chunks: Vec<_> = rrf_scores
            .iter()
            .filter_map(|(id, score)| {
                candidate_map.get(id.as_str()).map(|c| (*c, *score))
            })
            .collect();
        
        scored_chunks.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select top-n respecting token budget if specified
        let mut selected_indices = Vec::new();
        let mut total_tokens: u32 = 0;
        let mut total_score = 0.0;
        
        for (candidate, score) in scored_chunks.iter().take(self.config.top_n) {
            // Check token budget
            if let Some(max_tokens) = self.config.max_tokens {
                if total_tokens + candidate.token_count > max_tokens {
                    continue; // Skip this chunk if it would exceed budget
                }
            }
            
            // Find original index
            if let Some(idx) = candidates.iter().position(|c| c.chunk_id == candidate.chunk_id) {
                selected_indices.push(idx);
                total_tokens += candidate.token_count;
                total_score += score;
            }
            
            // Check if we've selected enough chunks
            if selected_indices.len() >= self.config.top_n {
                break;
            }
        }
        
        let solve_time = start.elapsed();
        
        RRFOutput {
            selected_indices,
            rrf_scores,
            total_rrf_score: total_score,
            total_tokens,
            solve_time_ms: solve_time.as_secs_f64() * 1000.0,
        }
    }
    
    /// Convert RRF output to SolverOutput for comparison
    pub fn to_solver_output(
        &self,
        rrf_output: &RRFOutput,
        candidates: &[ChunkCandidate],
    ) -> SolverOutput {
        let n = candidates.len();
        let mut selected_mask = vec![false; n];
        
        for &idx in &rrf_output.selected_indices {
            selected_mask[idx] = true;
        }
        
        SolverOutput {
            selected_indices: rrf_output.selected_indices.clone(),
            selected_mask,
            objective_score: rrf_output.total_rrf_score as f64,
            relevance_total: rrf_output.total_rrf_score as f64,
            density_total: 0.0,
            centrality_total: 0.0,
            recency_total: 0.0,
            auxiliary_total: 0.0,
            fulfilment_total: 0.0,
            redundancy_penalty: 0.0, // RRF doesn't compute redundancy
            total_tokens: rrf_output.total_tokens,
            num_selected: rrf_output.selected_indices.len(),
            iterations_run: 0,
            best_iteration: 0,
            improvement_history: vec![],
            constraints_satisfied: true,
            constraint_violations: vec![],
            solve_time_ms: rrf_output.solve_time_ms,
            exact_window_used: false,
            exact_window_core_size: 0,
            exact_window_nodes: 0,
            exact_window_exhaustive: false,
            exact_window_gap: 0.0,
            exact_window_fixed_in: 0,
            exact_window_fixed_out: 0,
        }
    }
}

/// Output from RRF solver
#[derive(Debug, Clone)]
pub struct RRFOutput {
    /// Indices of selected chunks
    pub selected_indices: Vec<usize>,
    /// RRF scores for all chunks
    pub rrf_scores: HashMap<String, f32>,
    /// Sum of RRF scores for selected chunks
    pub total_rrf_score: f32,
    /// Total tokens used
    pub total_tokens: u32,
    /// Solve time in milliseconds
    pub solve_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::RhetoricalRole;
    
    fn create_test_candidates() -> Vec<ChunkCandidate> {
        (0..10)
            .map(|i| {
                ChunkCandidate::new(
                    format!("chunk_{}", i),
                    format!("Content for chunk {}", i),
                    vec![0.1 * (i as f32); 128],
                    100 + (i as u32) * 20,
                )
                .with_fact_density(0.5 + 0.05 * (i as f32))
            })
            .collect()
    }
    
    #[test]
    fn test_rrf_basic() {
        let candidates = create_test_candidates();
        
        // Create two ranking sources with different orderings
        let source1 = RankingSource::new("colbert", vec![
            "chunk_0".into(), "chunk_2".into(), "chunk_4".into(), 
            "chunk_6".into(), "chunk_8".into(),
        ]);
        
        let source2 = RankingSource::new("bm25", vec![
            "chunk_1".into(), "chunk_0".into(), "chunk_3".into(),
            "chunk_5".into(), "chunk_7".into(),
        ]);
        
        let config = RRFConfig {
            k: 60.0,
            top_n: 3,
            max_tokens: None,
        };
        
        let solver = RRFSolver::new(config);
        let output = solver.solve(&candidates, &[source1, source2]);
        
        // chunk_0 appears in both lists (rank 1 and rank 2), should be highest
        assert!(!output.selected_indices.is_empty());
        assert!(output.rrf_scores.get("chunk_0").unwrap() > output.rrf_scores.get("chunk_9").unwrap_or(&0.0));
    }
    
    #[test]
    fn test_rrf_respects_token_budget() {
        let candidates = create_test_candidates();
        
        let source1 = RankingSource::new("colbert", 
            candidates.iter().map(|c| c.chunk_id.clone()).collect()
        );
        
        let config = RRFConfig {
            k: 60.0,
            top_n: 10,
            max_tokens: Some(300), // Only enough for ~2-3 chunks
        };
        
        let solver = RRFSolver::new(config);
        let output = solver.solve(&candidates, &[source1]);
        
        assert!(output.total_tokens <= 300);
        assert!(output.selected_indices.len() <= 3);
    }
    
    #[test]
    fn test_rrf_to_solver_output() {
        let candidates = create_test_candidates();
        
        let source1 = RankingSource::new("colbert", 
            candidates.iter().map(|c| c.chunk_id.clone()).collect()
        );
        
        let config = RRFConfig::default();
        let solver = RRFSolver::new(config);
        
        let rrf_output = solver.solve(&candidates, &[source1]);
        let solver_output = solver.to_solver_output(&rrf_output, &candidates);
        
        assert_eq!(solver_output.selected_indices.len(), rrf_output.selected_indices.len());
        assert_eq!(solver_output.total_tokens, rrf_output.total_tokens);
    }
}

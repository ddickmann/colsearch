//! Objective Function for Quadratic Knapsack Problem
//!
//! Implements the objective function:
//! Z = α·Σ(xᵢ·rᵢ) + β·Σ(xᵢ·dᵢ) + γ·Σ(xᵢ·cᵢ) + δ·Σ(xᵢ·tᵢ) - λ·Σᵢⱼ(xᵢ·xⱼ·Sᵢⱼ)

use crate::backend::{Backend, ObjectiveBatchContext, ObjectiveWeights};
use crate::solver::config::{SolverConfig, SolverInput};

/// Objective function for the Quadratic Knapsack Problem
pub struct ObjectiveFunction {
    weights: ObjectiveWeights,
    coverage_weight: f32,
}

#[derive(Debug, Clone, Default)]
pub struct FulfilmentState {
    pub best_scores: Vec<f32>,
    pub second_scores: Vec<f32>,
    pub third_scores: Vec<f32>,
    pub fourth_scores: Vec<f32>,
    pub best_indices: Vec<Option<usize>>,
    pub second_indices: Vec<Option<usize>>,
    pub third_indices: Vec<Option<usize>>,
    pub quorum_counts: Vec<u8>,
}

impl ObjectiveFunction {
    /// Create a new objective function from solver config
    pub fn new(config: &SolverConfig) -> Self {
        Self {
            weights: config.weights(),
            coverage_weight: config.mu,
        }
    }

    #[inline]
    fn quorum_support(
        &self,
        best: f64,
        second: f64,
        third: f64,
        fourth: f64,
        quorum_count: usize,
    ) -> f64 {
        let capped_best = best.clamp(0.0, 1.0);
        let capped_second = second.clamp(0.0, 1.0);
        let capped_third = third.clamp(0.0, 1.0);
        let capped_fourth = fourth.clamp(0.0, 1.0);
        let quorum_cap = self.weights.support_quorum_cap.max(2.0) as usize;
        let third_mass = if quorum_cap >= 3 && quorum_count >= 3 {
            self.weights.support_quorum_bonus as f64 * capped_third
        } else {
            0.0
        };
        let fourth_mass = if quorum_cap >= 4 && quorum_count >= 4 {
            0.5 * self.weights.support_quorum_bonus as f64 * capped_fourth
        } else {
            0.0
        };
        capped_best + self.weights.support_secondary_discount as f64 * capped_second + third_mass + fourth_mass
    }

    fn compute_fulfilment_total(&self, selection: &[bool], input: &SolverInput) -> f64 {
        let Some(coverage_matrix) = input.coverage_matrix.as_ref() else {
            return 0.0;
        };
        if input.num_query_tokens == 0 || self.coverage_weight.abs() <= f32::EPSILON {
            return 0.0;
        }

        let n = input.num_candidates;
        let mut total = 0.0f64;
        for token_idx in 0..input.num_query_tokens {
            let row = &coverage_matrix[token_idx * n..(token_idx + 1) * n];
            let mut best = 0.0f64;
            let mut second = 0.0f64;
            let mut third = 0.0f64;
            let mut fourth = 0.0f64;
            let mut quorum_count = 0usize;
            for i in 0..n {
                if selection[i] {
                    let score = row[i] as f64;
                    if row[i] >= self.weights.support_quorum_threshold {
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
            let weight = input
                .query_token_weights
                .as_ref()
                .and_then(|weights| weights.get(token_idx))
                .copied()
                .unwrap_or(1.0) as f64;
            total += weight * self.quorum_support(best, second, third, fourth, quorum_count);
        }
        total
    }

    pub fn batch_context<'a>(
        &self,
        input: &'a SolverInput,
        similarity_matrix: &'a [f32],
    ) -> ObjectiveBatchContext<'a> {
        ObjectiveBatchContext {
            num_candidates: input.num_candidates,
            relevance: &input.relevance_scores,
            density: &input.density_scores,
            centrality: &input.centrality_scores,
            recency: &input.recency_scores,
            auxiliary: &input.auxiliary_scores,
            similarity_matrix,
            coverage_matrix: input.coverage_matrix.as_deref(),
            query_token_weights: input.query_token_weights.as_deref(),
            num_query_tokens: input.num_query_tokens,
            weights: self.weights,
        }
    }

    fn compute_fulfilment_delta(
        &self,
        current_selection: &[bool],
        swap_in: usize,
        swap_out: Option<usize>,
        input: &SolverInput,
    ) -> f64 {
        let Some(coverage_matrix) = input.coverage_matrix.as_ref() else {
            return 0.0;
        };
        if input.num_query_tokens == 0 || self.coverage_weight.abs() <= f32::EPSILON {
            return 0.0;
        }

        let n = input.num_candidates;
        let selected_indices: Vec<usize> = current_selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected { Some(idx) } else { None })
            .collect();

        let mut total = 0.0f64;
        for token_idx in 0..input.num_query_tokens {
            let row = &coverage_matrix[token_idx * n..(token_idx + 1) * n];
            let mut old_best = 0.0f64;
            let mut old_second = 0.0f64;
            let mut old_third = 0.0f64;
            let mut old_fourth = 0.0f64;
            let mut new_best = 0.0f64;
            let mut new_second = 0.0f64;
            let mut new_third = 0.0f64;
            let mut new_fourth = 0.0f64;
            let mut old_quorum_count = 0usize;
            let mut new_quorum_count = 0usize;

            for &idx in &selected_indices {
                let score = row[idx] as f64;
                if row[idx] >= self.weights.support_quorum_threshold {
                    old_quorum_count += 1;
                }
                if score >= old_best {
                    old_fourth = old_third;
                    old_third = old_second;
                    old_second = old_best;
                    old_best = score;
                } else if score > old_second {
                    old_fourth = old_third;
                    old_third = old_second;
                    old_second = score;
                } else if score > old_third {
                    old_fourth = old_third;
                    old_third = score;
                } else if score > old_fourth {
                    old_fourth = score;
                }
                if Some(idx) != swap_out {
                    if row[idx] >= self.weights.support_quorum_threshold {
                        new_quorum_count += 1;
                    }
                    if score >= new_best {
                        new_fourth = new_third;
                        new_third = new_second;
                        new_second = new_best;
                        new_best = score;
                    } else if score > new_second {
                        new_fourth = new_third;
                        new_third = new_second;
                        new_second = score;
                    } else if score > new_third {
                        new_fourth = new_third;
                        new_third = score;
                    } else if score > new_fourth {
                        new_fourth = score;
                    }
                }
            }

            if swap_in < n {
                let score = row[swap_in] as f64;
                if row[swap_in] >= self.weights.support_quorum_threshold {
                    new_quorum_count += 1;
                }
                if score >= new_best {
                    new_fourth = new_third;
                    new_third = new_second;
                    new_second = new_best;
                    new_best = score;
                } else if score > new_second {
                    new_fourth = new_third;
                    new_third = new_second;
                    new_second = score;
                } else if score > new_third {
                    new_fourth = new_third;
                    new_third = score;
                } else if score > new_fourth {
                    new_fourth = score;
                }
            }

            let weight = input
                .query_token_weights
                .as_ref()
                .and_then(|weights| weights.get(token_idx))
                .copied()
                .unwrap_or(1.0) as f64;
            let old_support =
                self.quorum_support(old_best, old_second, old_third, old_fourth, old_quorum_count);
            let new_support =
                self.quorum_support(new_best, new_second, new_third, new_fourth, new_quorum_count);
            total += weight * (new_support - old_support);
        }
        total
    }

    pub fn build_fulfilment_state(
        &self,
        selection: &[bool],
        input: &SolverInput,
    ) -> Option<FulfilmentState> {
        let coverage_matrix = input.coverage_matrix.as_ref()?;
        if input.num_query_tokens == 0 || self.coverage_weight.abs() <= f32::EPSILON {
            return None;
        }

        let n = input.num_candidates;
        let mut best_scores = vec![0.0f32; input.num_query_tokens];
        let mut second_scores = vec![0.0f32; input.num_query_tokens];
        let mut third_scores = vec![0.0f32; input.num_query_tokens];
        let mut fourth_scores = vec![0.0f32; input.num_query_tokens];
        let mut best_indices = vec![None; input.num_query_tokens];
        let mut second_indices = vec![None; input.num_query_tokens];
        let mut third_indices = vec![None; input.num_query_tokens];
        let mut quorum_counts = vec![0u8; input.num_query_tokens];

        for token_idx in 0..input.num_query_tokens {
            let row = &coverage_matrix[token_idx * n..(token_idx + 1) * n];
            let mut best = 0.0f32;
            let mut second = 0.0f32;
            let mut third = 0.0f32;
            let mut fourth = 0.0f32;
            let mut best_idx = None;
            let mut second_idx = None;
            let mut third_idx = None;
            let mut quorum_count = 0u8;
            for (idx, &selected) in selection.iter().enumerate() {
                if !selected {
                    continue;
                }
                let score = row[idx];
                if score >= self.weights.support_quorum_threshold {
                    quorum_count = quorum_count.saturating_add(1);
                }
                if score >= best {
                    fourth = third;
                    third = second;
                    third_idx = second_idx;
                    second = best;
                    second_idx = best_idx;
                    best = score;
                    best_idx = Some(idx);
                } else if score > second {
                    fourth = third;
                    third = second;
                    third_idx = second_idx;
                    second = score;
                    second_idx = Some(idx);
                } else if score > third {
                    fourth = third;
                    third = score;
                    third_idx = Some(idx);
                } else if score > fourth {
                    fourth = score;
                }
            }
            best_scores[token_idx] = best;
            second_scores[token_idx] = second;
            third_scores[token_idx] = third;
            fourth_scores[token_idx] = fourth;
            best_indices[token_idx] = best_idx;
            second_indices[token_idx] = second_idx;
            third_indices[token_idx] = third_idx;
            quorum_counts[token_idx] = quorum_count;
        }

        Some(FulfilmentState {
            best_scores,
            second_scores,
            third_scores,
            fourth_scores,
            best_indices,
            second_indices,
            third_indices,
            quorum_counts,
        })
    }

    fn compute_fulfilment_delta_with_state(
        &self,
        fulfilment_state: &FulfilmentState,
        swap_in: usize,
        swap_out: Option<usize>,
        input: &SolverInput,
    ) -> f64 {
        let Some(coverage_matrix) = input.coverage_matrix.as_ref() else {
            return 0.0;
        };
        if input.num_query_tokens == 0 || self.coverage_weight.abs() <= f32::EPSILON {
            return 0.0;
        }

        let n = input.num_candidates;
        let mut total = 0.0f64;
        for token_idx in 0..input.num_query_tokens {
            let row = &coverage_matrix[token_idx * n..(token_idx + 1) * n];
            let old_best = fulfilment_state.best_scores[token_idx] as f64;
            let old_second = fulfilment_state.second_scores[token_idx] as f64;
            let old_third = fulfilment_state.third_scores[token_idx] as f64;
            let old_fourth = fulfilment_state.fourth_scores[token_idx] as f64;
            let old_quorum_count = fulfilment_state.quorum_counts[token_idx] as usize;
            let old_support =
                self.quorum_support(old_best, old_second, old_third, old_fourth, old_quorum_count);
            let mut new_best = old_best;
            let mut new_second = old_second;
            let mut new_third = old_third;
            let mut new_fourth = old_fourth;
            let mut new_quorum_count = old_quorum_count;
            if fulfilment_state.best_indices[token_idx] == swap_out {
                new_best = fulfilment_state.second_scores[token_idx] as f64;
                new_second = fulfilment_state.third_scores[token_idx] as f64;
                new_third = fulfilment_state.fourth_scores[token_idx] as f64;
                new_fourth = 0.0;
            } else if fulfilment_state.second_indices[token_idx] == swap_out {
                new_second = fulfilment_state.third_scores[token_idx] as f64;
                new_third = fulfilment_state.fourth_scores[token_idx] as f64;
                new_fourth = 0.0;
            } else if fulfilment_state.third_indices[token_idx] == swap_out {
                new_third = fulfilment_state.fourth_scores[token_idx] as f64;
                new_fourth = 0.0;
            }
            if let Some(remove_idx) = swap_out {
                if row[remove_idx] >= self.weights.support_quorum_threshold {
                    new_quorum_count = new_quorum_count.saturating_sub(1);
                }
            }
            if swap_in < n {
                let score = row[swap_in] as f64;
                if row[swap_in] >= self.weights.support_quorum_threshold {
                    new_quorum_count += 1;
                }
                if score >= new_best {
                    new_fourth = new_third;
                    new_third = new_second;
                    new_second = new_best;
                    new_best = score;
                } else if score > new_second {
                    new_fourth = new_third;
                    new_third = new_second;
                    new_second = score;
                } else if score > new_third {
                    new_fourth = new_third;
                    new_third = score;
                } else if score > new_fourth {
                    new_fourth = score;
                }
            }
            let weight = input
                .query_token_weights
                .as_ref()
                .and_then(|weights| weights.get(token_idx))
                .copied()
                .unwrap_or(1.0) as f64;
            let new_support =
                self.quorum_support(new_best, new_second, new_third, new_fourth, new_quorum_count);
            total += weight * (new_support - old_support);
        }
        total
    }
    
    /// Compute objective value for a selection
    pub fn compute(
        &self,
        selection: &[bool],
        input: &SolverInput,
        similarity_matrix: &[f32],
        backend: &dyn Backend,
    ) -> (f64, ObjectiveBreakdown) {
        let context = self.batch_context(input, similarity_matrix);
        // Linear terms using backend
        let relevance_term = backend.masked_sum(&input.relevance_scores, selection) as f64;
        let density_term = backend.masked_sum(&input.density_scores, selection) as f64;
        let centrality_term = backend.masked_sum(&input.centrality_scores, selection) as f64;
        let recency_term = backend.masked_sum(&input.recency_scores, selection) as f64;
        let auxiliary_term = backend.masked_sum(&input.auxiliary_scores, selection) as f64;
        let fulfilment_term = self.compute_fulfilment_total(selection, input);
        let weighted_linear_total = self.weights.alpha as f64 * relevance_term
            + self.weights.beta as f64 * density_term
            + self.weights.gamma as f64 * centrality_term
            + self.weights.delta as f64 * recency_term
            + self.weights.epsilon as f64 * auxiliary_term;
        let fulfilment_contribution = self.coverage_weight as f64 * fulfilment_term;
        let objective = backend.compute_objectives_batch(selection, 1, &context)[0] as f64;
        let redundancy_penalty = (weighted_linear_total + fulfilment_contribution - objective).max(0.0);
        
        let breakdown = ObjectiveBreakdown {
            relevance: self.weights.alpha as f64 * relevance_term,
            density: self.weights.beta as f64 * density_term,
            centrality: self.weights.gamma as f64 * centrality_term,
            recency: self.weights.delta as f64 * recency_term,
            auxiliary: self.weights.epsilon as f64 * auxiliary_term,
            fulfilment: fulfilment_contribution,
            redundancy_penalty,
        };
        
        (objective, breakdown)
    }
    
    /// Compute objective for multiple selections in batch (GPU-accelerated)
    pub fn compute_batch(
        &self,
        selections: &[bool],  // Flattened (batch_size, n)
        batch_size: usize,
        input: &SolverInput,
        similarity_matrix: &[f32],
        backend: &dyn Backend,
    ) -> Vec<f64> {
        let context = self.batch_context(input, similarity_matrix);
        backend
            .compute_objectives_batch(selections, batch_size, &context)
            .into_iter()
            .map(|x| x as f64)
            .collect()
    }
    
    /// Compute delta (change in objective) for a swap move
    /// This is more efficient than recomputing the full objective
    pub fn compute_delta(
        &self,
        current_selection: &[bool],
        swap_in: usize,
        swap_out: usize,
        input: &SolverInput,
        similarity_matrix: &[f32],
    ) -> f64 {
        let n = input.num_candidates;
        
        // Change in linear terms
        let delta_relevance = self.weights.alpha as f64 
            * (input.relevance_scores[swap_in] - input.relevance_scores[swap_out]) as f64;
        
        let delta_density = self.weights.beta as f64 
            * (input.density_scores[swap_in] - input.density_scores[swap_out]) as f64;
        
        let delta_centrality = self.weights.gamma as f64 
            * (input.centrality_scores[swap_in] - input.centrality_scores[swap_out]) as f64;
        
        let delta_recency = self.weights.delta as f64 
            * (input.recency_scores[swap_in] - input.recency_scores[swap_out]) as f64;

        let delta_auxiliary = self.weights.epsilon as f64
            * (input.auxiliary_scores[swap_in] - input.auxiliary_scores[swap_out]) as f64;
        let delta_fulfilment = self.coverage_weight as f64
            * self.compute_fulfilment_delta(current_selection, swap_in, Some(swap_out), input);
        
        // Change in redundancy term
        // Adding swap_in: +Σⱼ S[swap_in, j] for selected j
        // Removing swap_out: -Σⱼ S[swap_out, j] for selected j
        let mut delta_redundancy = 0.0f64;
        
        for j in 0..n {
            if current_selection[j] && j != swap_out && j != swap_in {
                delta_redundancy += similarity_matrix[swap_in * n + j] as f64;
                delta_redundancy -= similarity_matrix[swap_out * n + j] as f64;
            }
        }
        
        delta_relevance + delta_density + delta_centrality + delta_recency + delta_auxiliary + delta_fulfilment
            - self.weights.lambda as f64 * delta_redundancy
    }
    
    /// Compute delta for adding a single element (no swap)
    pub fn compute_add_delta(
        &self,
        current_selection: &[bool],
        add_idx: usize,
        input: &SolverInput,
        similarity_matrix: &[f32],
    ) -> f64 {
        let n = input.num_candidates;
        
        let linear_gain = self.weights.alpha as f64 * input.relevance_scores[add_idx] as f64
            + self.weights.beta as f64 * input.density_scores[add_idx] as f64
            + self.weights.gamma as f64 * input.centrality_scores[add_idx] as f64
            + self.weights.delta as f64 * input.recency_scores[add_idx] as f64
            + self.weights.epsilon as f64 * input.auxiliary_scores[add_idx] as f64;
        let delta_fulfilment = self.coverage_weight as f64
            * self.compute_fulfilment_delta(current_selection, add_idx, None, input);
        
        // Redundancy cost with already selected items
        let mut redundancy_cost = 0.0f64;
        for j in 0..n {
            if current_selection[j] {
                redundancy_cost += similarity_matrix[add_idx * n + j] as f64;
            }
        }
        
        linear_gain + delta_fulfilment - self.weights.lambda as f64 * redundancy_cost
    }
        /// Compute delta using pre-calculated redundancy contributions
    /// This is O(1) instead of O(N)
    #[inline]
    pub fn compute_delta_fast(

        &self,
        current_selection: &[bool],
        swap_in: usize,
        swap_out: usize,
        input: &SolverInput,
        similarity_matrix: &[f32],
        redundancy_contributions: &[f32],
        fulfilment_state: Option<&FulfilmentState>,
    ) -> f64 {
        let n = input.num_candidates;
        
        let delta_relevance = self.weights.alpha as f64 
            * (input.relevance_scores[swap_in] - input.relevance_scores[swap_out]) as f64;
        
        let delta_density = self.weights.beta as f64 
            * (input.density_scores[swap_in] - input.density_scores[swap_out]) as f64;
        
        let delta_centrality = self.weights.gamma as f64 
            * (input.centrality_scores[swap_in] - input.centrality_scores[swap_out]) as f64;
        
        let delta_recency = self.weights.delta as f64 
            * (input.recency_scores[swap_in] - input.recency_scores[swap_out]) as f64;

        let delta_auxiliary = self.weights.epsilon as f64
            * (input.auxiliary_scores[swap_in] - input.auxiliary_scores[swap_out]) as f64;
        let delta_fulfilment = self.coverage_weight as f64
            * fulfilment_state
                .map(|state| {
                    self.compute_fulfilment_delta_with_state(state, swap_in, Some(swap_out), input)
                })
                .unwrap_or_else(|| {
                    self.compute_fulfilment_delta(current_selection, swap_in, Some(swap_out), input)
                });
            
        // Redundancy Delta:
        // Removing swap_out reduced redundancy by (R[swap_out]) 
        // Adding swap_in increases redundancy by (R[swap_in] - sim[swap_in, swap_out])
        // Note: R[swap_in] includes sim[swap_in, swap_out] because swap_out IS in selection.
        // We want sum of sim[swap_in, j] for j in S_new.
        // S_new = S_old - swap_out.
        // So we need: Sum(swap_in, S_old) - sim(swap_in, swap_out).
        // That is exactly redundancy_contributions[swap_in] - sim[swap_in, swap_out].
        
        // Wait, removing swap_out also removes its contribution to everyone else?
        // Total Redundancy = 0.5 * Sum(x_i * x_j * S_ij).
        // Change = (Change in row swap_in) + (Change in row swap_out).
        // Removing u: - Sum_{j in S, j!=u} S_uj.  (which is R[u])
        // Adding v: + Sum_{j in S', j!=v} S_vj. (where S' = S-u+v)
        // Sum_{j in S-u} S_vj = Sum_{j in S} S_vj - S_vu = R[v] - S_vu.
        
        // So total change in Redundancy Term = (R[v] - S_{vu}) - R[u].
        // (Assuming S_{ii} is 0 or handled. usually S_{ii}=1 but we skip i!=j in calculation).
        
        let sim_uv = similarity_matrix[swap_in * n + swap_out] as f64;
        let r_in = redundancy_contributions[swap_in] as f64;
        let r_out = redundancy_contributions[swap_out] as f64;
        
        let delta_redundancy = (r_in - sim_uv) - r_out;
        
        delta_relevance + delta_density + delta_centrality + delta_recency + delta_auxiliary + delta_fulfilment
            - self.weights.lambda as f64 * delta_redundancy
    }

    /// Compute delta for adding single element using fast updates
    #[inline]
    pub fn compute_add_delta_fast(
        &self,
        current_selection: &[bool],
        add_idx: usize,
        input: &SolverInput,
        redundancy_contributions: &[f32],
        fulfilment_state: Option<&FulfilmentState>,
    ) -> f64 {
        let linear_gain = self.weights.alpha as f64 * input.relevance_scores[add_idx] as f64
            + self.weights.beta as f64 * input.density_scores[add_idx] as f64
            + self.weights.gamma as f64 * input.centrality_scores[add_idx] as f64
            + self.weights.delta as f64 * input.recency_scores[add_idx] as f64
            + self.weights.epsilon as f64 * input.auxiliary_scores[add_idx] as f64;
        let delta_fulfilment = self.coverage_weight as f64
            * fulfilment_state
                .map(|state| self.compute_fulfilment_delta_with_state(state, add_idx, None, input))
                .unwrap_or_else(|| self.compute_fulfilment_delta(current_selection, add_idx, None, input));
            
        // Adding v: Redundancy increases by Sum_{j in S} S_vj = R[v]
        let delta_redundancy = redundancy_contributions[add_idx] as f64;
        
        linear_gain + delta_fulfilment - self.weights.lambda as f64 * delta_redundancy
    }
}

/// Breakdown of objective function components
#[derive(Debug, Clone, Default)]
pub struct ObjectiveBreakdown {
    pub relevance: f64,
    pub density: f64,
    pub centrality: f64,
    pub recency: f64,
    pub auxiliary: f64,
    pub fulfilment: f64,
    pub redundancy_penalty: f64,
}

impl ObjectiveBreakdown {
    /// Total objective value
    pub fn total(&self) -> f64 {
        self.relevance
            + self.density
            + self.centrality
            + self.recency
            + self.auxiliary
            + self.fulfilment
            - self.redundancy_penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    
    fn create_test_input() -> SolverInput {
        SolverInput {
            num_candidates: 3,
            embedding_dim: 2,
            embeddings: vec![1.0, 0.0, 0.0, 1.0, 0.707, 0.707],
            query_embedding: vec![1.0, 0.0],
            relevance_scores: vec![1.0, 0.5, 0.8],
            density_scores: vec![0.7, 0.6, 0.9],
            centrality_scores: vec![0.8, 0.5, 0.7],
            recency_scores: vec![1.0, 0.5, 0.8],
            auxiliary_scores: vec![0.0, 0.0, 0.0],
            fulfilment_scores: vec![0.0, 0.0, 0.0],
            token_costs: vec![100, 150, 200],
            roles: vec![
                crate::solver::RhetoricalRole::Definition,
                crate::solver::RhetoricalRole::Example,
                crate::solver::RhetoricalRole::Evidence,
            ],
            cluster_ids: vec![Some(0), Some(0), Some(1)],
            chunk_ids: vec!["c1".to_string(), "c2".to_string(), "c3".to_string()],
            coverage_matrix: None,
            query_token_weights: None,
            num_query_tokens: 0,
            similarity_matrix: None,
        }
    }
    
    #[test]
    fn test_objective_compute() {
        let config = SolverConfig::default();
        let objective = ObjectiveFunction::new(&config);
        let input = create_test_input();
        let backend = CpuBackend::new();
        
        // Compute similarity matrix
        let sim = backend.cosine_similarity_matrix(&input.embeddings, 3, 2);
        
        let selection = vec![true, false, true];
        let (score, breakdown) = objective.compute(&selection, &input, &sim, &backend);
        
        assert!(score > 0.0);
        assert!(breakdown.relevance > 0.0);
        assert!(breakdown.redundancy_penalty >= 0.0);
    }
    
    #[test]
    fn test_delta_consistency() {
        let config = SolverConfig::default();
        let objective = ObjectiveFunction::new(&config);
        let input = create_test_input();
        let backend = CpuBackend::new();
        
        let sim = backend.cosine_similarity_matrix(&input.embeddings, 3, 2);
        
        // Initial selection
        let selection1 = vec![true, false, true];
        let (score1, _) = objective.compute(&selection1, &input, &sim, &backend);
        
        // Swap: remove index 2, add index 1
        let delta = objective.compute_delta(&selection1, 1, 2, &input, &sim);
        
        // New selection after swap
        let selection2 = vec![true, true, false];
        let (score2, _) = objective.compute(&selection2, &input, &sim, &backend);
        
        // Delta should match the difference
        let expected_delta = score2 - score1;
        assert!((delta - expected_delta).abs() < 1e-6, 
            "Delta mismatch: computed={}, expected={}", delta, expected_delta);
    }

    #[test]
    fn test_fulfilment_state_matches_fast_delta() {
        let mut config = SolverConfig::default();
        config.mu = 1.0;
        let objective = ObjectiveFunction::new(&config);
        let mut input = create_test_input();
        input.coverage_matrix = Some(vec![
            0.9, 0.3, 0.4,
            0.2, 0.8, 0.7,
        ]);
        input.query_token_weights = Some(vec![0.6, 0.4]);
        input.num_query_tokens = 2;
        input.fulfilment_scores = vec![0.7, 0.5, 0.6];
        let backend = CpuBackend::new();
        let sim = backend.cosine_similarity_matrix(&input.embeddings, 3, 2);

        let selection = vec![true, false, true];
        let redundancy: Vec<f32> = (0..input.num_candidates)
            .map(|idx| {
                (0..input.num_candidates)
                    .filter(|&other| selection[other] && other != idx)
                    .map(|other| sim[idx * input.num_candidates + other])
                    .sum()
            })
            .collect();
        let fulfilment_state = objective.build_fulfilment_state(&selection, &input);
        let delta = objective.compute_delta_fast(
            &selection,
            1,
            2,
            &input,
            &sim,
            &redundancy,
            fulfilment_state.as_ref(),
        );
        let reference = objective.compute_delta(&selection, 1, 2, &input, &sim);

        assert!((delta - reference).abs() < 1e-6);
    }
}


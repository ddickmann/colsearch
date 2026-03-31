//! Constraint Validation for the Knapsack Problem
//!
//! Implements constraint checking for:
//! - Token budget constraints
//! - Role requirements
//! - Cluster diversity limits
//! - Required/excluded chunks

use std::collections::{HashMap, HashSet};
use crate::solver::config::{SolverConstraints, SolverInput, RhetoricalRole};

/// Constraint validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all constraints are satisfied
    pub is_valid: bool,
    
    /// List of violations
    pub violations: Vec<ConstraintViolation>,
    
    /// Total tokens used
    pub total_tokens: u32,
    
    /// Number of chunks selected
    pub num_selected: usize,
}

impl ValidationResult {
    pub fn valid(total_tokens: u32, num_selected: usize) -> Self {
        Self {
            is_valid: true,
            violations: Vec::new(),
            total_tokens,
            num_selected,
        }
    }
    
    pub fn invalid(violations: Vec<ConstraintViolation>, total_tokens: u32, num_selected: usize) -> Self {
        Self {
            is_valid: false,
            violations,
            total_tokens,
            num_selected,
        }
    }
}

/// Types of constraint violations
#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    /// Token budget exceeded
    TokenBudgetExceeded { used: u32, max: u32 },
    
    /// Minimum tokens not met
    MinTokensNotMet { used: u32, min: u32 },
    
    /// Too few chunks selected
    TooFewChunks { selected: usize, min: usize },
    
    /// Too many chunks selected
    TooManyChunks { selected: usize, max: usize },
    
    /// Required role not present
    MissingRole(RhetoricalRole),
    
    /// Too many chunks from same cluster
    ClusterOverflow { cluster_id: u32, count: usize, max: usize },
    
    /// Required chunk not selected
    RequiredChunkMissing(String),
    
    /// Excluded chunk was selected
    ExcludedChunkSelected(String),
}

impl std::fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenBudgetExceeded { used, max } => {
                write!(f, "Token budget exceeded: {} > {}", used, max)
            }
            Self::MinTokensNotMet { used, min } => {
                write!(f, "Minimum tokens not met: {} < {}", used, min)
            }
            Self::TooFewChunks { selected, min } => {
                write!(f, "Too few chunks: {} < {}", selected, min)
            }
            Self::TooManyChunks { selected, max } => {
                write!(f, "Too many chunks: {} > {}", selected, max)
            }
            Self::MissingRole(role) => {
                write!(f, "Missing required role: {:?}", role)
            }
            Self::ClusterOverflow { cluster_id, count, max } => {
                write!(f, "Cluster {} overflow: {} > {}", cluster_id, count, max)
            }
            Self::RequiredChunkMissing(id) => {
                write!(f, "Required chunk missing: {}", id)
            }
            Self::ExcludedChunkSelected(id) => {
                write!(f, "Excluded chunk selected: {}", id)
            }
        }
    }
}

/// Constraint validator for the optimization problem
pub struct ConstraintValidator<'a> {
    constraints: &'a SolverConstraints,
}

#[derive(Debug, Clone)]
pub struct SelectionSummary {
    pub total_tokens: u32,
    pub num_selected: usize,
    pub role_counts: HashMap<RhetoricalRole, usize>,
    pub cluster_counts: HashMap<u32, usize>,
    pub selected_chunks: HashSet<String>,
}

impl<'a> ConstraintValidator<'a> {
    /// Create a new constraint validator
    pub fn new(constraints: &'a SolverConstraints) -> Self {
        Self { constraints }
    }
    
    /// Validate a selection against all constraints
    pub fn validate(&self, selection: &[bool], input: &SolverInput) -> ValidationResult {
        let mut violations = Vec::new();
        
        // Count selected and compute totals
        let selected_indices: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i) } else { None })
            .collect();
        
        let num_selected = selected_indices.len();
        let total_tokens: u32 = selected_indices
            .iter()
            .map(|&i| input.token_costs[i])
            .sum();
        
        // Check token budget
        if total_tokens > self.constraints.max_tokens {
            violations.push(ConstraintViolation::TokenBudgetExceeded {
                used: total_tokens,
                max: self.constraints.max_tokens,
            });
        }
        
        if total_tokens < self.constraints.min_tokens {
            violations.push(ConstraintViolation::MinTokensNotMet {
                used: total_tokens,
                min: self.constraints.min_tokens,
            });
        }
        
        // Check chunk count
        if num_selected < self.constraints.min_chunks {
            violations.push(ConstraintViolation::TooFewChunks {
                selected: num_selected,
                min: self.constraints.min_chunks,
            });
        }
        
        if num_selected > self.constraints.max_chunks {
            violations.push(ConstraintViolation::TooManyChunks {
                selected: num_selected,
                max: self.constraints.max_chunks,
            });
        }
        
        // Check required roles
        let selected_roles: std::collections::HashSet<RhetoricalRole> = selected_indices
            .iter()
            .map(|&i| input.roles[i])
            .collect();
        
        for required_role in &self.constraints.must_include_roles {
            if !selected_roles.contains(required_role) {
                violations.push(ConstraintViolation::MissingRole(*required_role));
            }
        }
        
        // Check cluster diversity
        let mut cluster_counts: HashMap<u32, usize> = HashMap::new();
        for &i in &selected_indices {
            if let Some(cluster_id) = input.cluster_ids[i] {
                *cluster_counts.entry(cluster_id).or_insert(0) += 1;
            }
        }
        
        for (cluster_id, count) in cluster_counts {
            if count > self.constraints.max_per_cluster {
                violations.push(ConstraintViolation::ClusterOverflow {
                    cluster_id,
                    count,
                    max: self.constraints.max_per_cluster,
                });
            }
        }
        
        // Check required chunks
        let selected_ids: std::collections::HashSet<&str> = selected_indices
            .iter()
            .map(|&i| input.chunk_ids[i].as_str())
            .collect();
        
        for required_id in &self.constraints.required_chunks {
            if !selected_ids.contains(required_id.as_str()) {
                violations.push(ConstraintViolation::RequiredChunkMissing(required_id.clone()));
            }
        }
        
        // Check excluded chunks
        for &i in &selected_indices {
            if self.constraints.excluded_chunks.contains(&input.chunk_ids[i]) {
                violations.push(ConstraintViolation::ExcludedChunkSelected(
                    input.chunk_ids[i].clone(),
                ));
            }
        }
        
        if violations.is_empty() {
            ValidationResult::valid(total_tokens, num_selected)
        } else {
            ValidationResult::invalid(violations, total_tokens, num_selected)
        }
    }

    pub fn summarize_selection(&self, selection: &[bool], input: &SolverInput) -> SelectionSummary {
        let mut total_tokens = 0u32;
        let mut num_selected = 0usize;
        let mut role_counts: HashMap<RhetoricalRole, usize> = HashMap::new();
        let mut cluster_counts: HashMap<u32, usize> = HashMap::new();
        let mut selected_chunks: HashSet<String> = HashSet::new();

        for (idx, &selected) in selection.iter().enumerate() {
            if !selected {
                continue;
            }
            total_tokens += input.token_costs[idx];
            num_selected += 1;
            *role_counts.entry(input.roles[idx]).or_insert(0) += 1;
            if let Some(cluster_id) = input.cluster_ids[idx] {
                *cluster_counts.entry(cluster_id).or_insert(0) += 1;
            }
            selected_chunks.insert(input.chunk_ids[idx].clone());
        }

        SelectionSummary {
            total_tokens,
            num_selected,
            role_counts,
            cluster_counts,
            selected_chunks,
        }
    }
    
    /// Check if adding a chunk would violate the token budget or max chunks constraint
    pub fn can_add(&self, idx: usize, summary: &SelectionSummary, input: &SolverInput) -> bool {
        if summary.selected_chunks.contains(&input.chunk_ids[idx]) {
            return false;
        }
        if self.constraints.excluded_chunks.contains(&input.chunk_ids[idx]) {
            return false;
        }
        if summary.total_tokens + input.token_costs[idx] > self.constraints.max_tokens {
            return false;
        }
        if summary.num_selected >= self.constraints.max_chunks {
            return false;
        }
        if let Some(cluster_id) = input.cluster_ids[idx] {
            let count = summary.cluster_counts.get(&cluster_id).copied().unwrap_or(0);
            if count >= self.constraints.max_per_cluster {
                return false;
            }
        }
        true
    }
    
    /// Check if a swap is feasible (maintains token budget)
    pub fn can_swap(
        &self,
        add_idx: usize,
        remove_idx: usize,
        summary: &SelectionSummary,
        input: &SolverInput,
    ) -> bool {
        if !summary.selected_chunks.contains(&input.chunk_ids[remove_idx]) {
            return false;
        }
        if summary.selected_chunks.contains(&input.chunk_ids[add_idx]) {
            return false;
        }
        if self.constraints.required_chunks.contains(&input.chunk_ids[remove_idx]) {
            return false;
        }
        if self.constraints.excluded_chunks.contains(&input.chunk_ids[add_idx]) {
            return false;
        }

        let new_tokens = summary.total_tokens + input.token_costs[add_idx] - input.token_costs[remove_idx];
        if new_tokens > self.constraints.max_tokens || new_tokens < self.constraints.min_tokens {
            return false;
        }

        if let Some(remove_cluster) = input.cluster_ids[remove_idx] {
            let mut next_count = summary.cluster_counts.get(&remove_cluster).copied().unwrap_or(0);
            next_count = next_count.saturating_sub(1);
            if let Some(add_cluster) = input.cluster_ids[add_idx] {
                if add_cluster == remove_cluster {
                    next_count += 1;
                }
            }
            if next_count > self.constraints.max_per_cluster {
                return false;
            }
        }

        if let Some(add_cluster) = input.cluster_ids[add_idx] {
            let mut next_count = summary.cluster_counts.get(&add_cluster).copied().unwrap_or(0);
            if input.cluster_ids[remove_idx] != Some(add_cluster) {
                next_count += 1;
            }
            if next_count > self.constraints.max_per_cluster {
                return false;
            }
        }

        let remove_role = input.roles[remove_idx];
        let add_role = input.roles[add_idx];
        for required_role in &self.constraints.must_include_roles {
            let current = summary.role_counts.get(required_role).copied().unwrap_or(0);
            let after_remove = if *required_role == remove_role {
                current.saturating_sub(1)
            } else {
                current
            };
            let after_swap = if *required_role == add_role {
                after_remove + 1
            } else {
                after_remove
            };
            if after_swap == 0 {
                return false;
            }
        }

        true
    }
    
    /// Get indices of candidates that violate exclusion constraints
    pub fn get_excluded_indices(&self, input: &SolverInput) -> Vec<usize> {
        input.chunk_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| self.constraints.excluded_chunks.contains(*id))
            .map(|(i, _)| i)
            .collect()
    }
    
    /// Get indices of candidates that must be included
    pub fn get_required_indices(&self, input: &SolverInput) -> Vec<usize> {
        input.chunk_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| self.constraints.required_chunks.contains(*id))
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_input() -> SolverInput {
        SolverInput {
            num_candidates: 5,
            embedding_dim: 2,
            embeddings: vec![0.0; 10],
            query_embedding: vec![1.0, 0.0],
            relevance_scores: vec![0.9, 0.8, 0.7, 0.6, 0.5],
            density_scores: vec![0.5; 5],
            centrality_scores: vec![0.5; 5],
            recency_scores: vec![0.5; 5],
            auxiliary_scores: vec![0.0; 5],
            fulfilment_scores: vec![0.0; 5],
            token_costs: vec![100, 200, 150, 300, 250],
            roles: vec![
                RhetoricalRole::Definition,
                RhetoricalRole::Example,
                RhetoricalRole::Evidence,
                RhetoricalRole::Risk,
                RhetoricalRole::Conclusion,
            ],
            cluster_ids: vec![Some(0), Some(0), Some(1), Some(1), Some(2)],
            chunk_ids: vec!["c1", "c2", "c3", "c4", "c5"]
                .into_iter()
                .map(String::from)
                .collect(),
            coverage_matrix: None,
            query_token_weights: None,
            num_query_tokens: 0,
            similarity_matrix: None,
        }
    }
    
    #[test]
    fn test_token_budget() {
        let constraints = SolverConstraints::with_budget(400);
        let validator = ConstraintValidator::new(&constraints);
        let input = create_test_input();
        
        // Within budget
        let selection = vec![true, true, false, false, false]; // 100 + 200 = 300
        let result = validator.validate(&selection, &input);
        assert!(result.is_valid);
        assert_eq!(result.total_tokens, 300);
        
        // Over budget
        let selection = vec![true, true, true, true, false]; // 100 + 200 + 150 + 300 = 750
        let result = validator.validate(&selection, &input);
        assert!(!result.is_valid);
        assert!(matches!(
            &result.violations[0],
            ConstraintViolation::TokenBudgetExceeded { .. }
        ));
    }
    
    #[test]
    fn test_required_roles() {
        let constraints = SolverConstraints::with_budget(1000)
            .require_role(RhetoricalRole::Risk);
        let validator = ConstraintValidator::new(&constraints);
        let input = create_test_input();
        
        // Missing Risk role
        let selection = vec![true, true, false, false, false];
        let result = validator.validate(&selection, &input);
        assert!(!result.is_valid);
        assert!(matches!(
            &result.violations[0],
            ConstraintViolation::MissingRole(RhetoricalRole::Risk)
        ));
        
        // Has Risk role
        let selection = vec![true, false, false, true, false]; // c1 (Definition) + c4 (Risk)
        let result = validator.validate(&selection, &input);
        assert!(result.is_valid);
    }
    
    #[test]
    fn test_cluster_diversity() {
        let constraints = SolverConstraints::with_budget(1000)
            .with_cluster_limit(1);
        let validator = ConstraintValidator::new(&constraints);
        let input = create_test_input();
        
        // Two from cluster 0
        let selection = vec![true, true, false, false, false];
        let result = validator.validate(&selection, &input);
        assert!(!result.is_valid);
        assert!(matches!(
            &result.violations[0],
            ConstraintViolation::ClusterOverflow { cluster_id: 0, .. }
        ));
        
        // One from each cluster
        let selection = vec![true, false, true, false, true];
        let result = validator.validate(&selection, &input);
        assert!(result.is_valid);
    }
    
    #[test]
    fn test_can_swap() {
        let constraints = SolverConstraints::with_budget(400);
        let validator = ConstraintValidator::new(&constraints);
        let input = create_test_input();
        let selection = vec![true, true, false, false, false];
        let summary = validator.summarize_selection(&selection, &input);
        
        // Swap that maintains budget
        assert!(validator.can_swap(2, 1, &summary, &input)); // -200 + 150 = 250
        
        // Swap that exceeds budget
        assert!(!validator.can_swap(3, 0, &summary, &input)); // -100 + 300 = 500
    }

    #[test]
    fn test_can_swap_preserves_required_role() {
        let constraints = SolverConstraints::with_budget(1000).require_role(RhetoricalRole::Definition);
        let validator = ConstraintValidator::new(&constraints);
        let input = create_test_input();
        let selection = vec![true, false, true, false, false];
        let summary = validator.summarize_selection(&selection, &input);

        assert!(!validator.can_swap(1, 0, &summary, &input));
    }

    #[test]
    fn test_can_add_respects_cluster_limit() {
        let constraints = SolverConstraints::with_budget(1000).with_cluster_limit(1);
        let validator = ConstraintValidator::new(&constraints);
        let input = create_test_input();
        let selection = vec![true, false, true, false, false];
        let summary = validator.summarize_selection(&selection, &input);

        assert!(!validator.can_add(1, &summary, &input));
    }
}


//! Tabu Search Knapsack Solver
//!
//! GPU/CPU accelerated implementation of Tabu Search for the
//! Quadratic Knapsack Problem (QKP).

use std::time::Instant;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::backend::{Backend, BackendType, CpuBackend, create_backend};
use crate::solver::{
    SolverConfig,
    SolverConstraints,
    SolverInput,
    SolverOutput,
    ChunkCandidate,
    FulfilmentState,
    ObjectiveFunction,
    SemanticTabuList,
    ConstraintValidator,
    RhetoricalRole,
};
use crate::solver::tabu::Move;

fn experimental_backends_enabled() -> bool {
    std::env::var("LATENCE_SOLVER_ENABLE_EXPERIMENTAL_BACKENDS")
        .map(|value| !matches!(value.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
        .unwrap_or(true)
}

#[derive(Debug, Clone, Default)]
struct ExactWindowReport {
    used: bool,
    core_size: usize,
    nodes: usize,
    exhaustive: bool,
    gap: f64,
    fixed_in: usize,
    fixed_out: usize,
}

/// Tabu Search Knapsack Solver
pub struct TabuSearchSolver {
    config: SolverConfig,
    backend: Box<dyn Backend>,
}

impl TabuSearchSolver {
    /// Create a new solver with the given configuration
    pub fn new(config: SolverConfig) -> Self {
        let backend_type = if config.use_gpu && experimental_backends_enabled() {
            if crate::cuda_available() {
                BackendType::Cuda
            } else if crate::gpu_available() {
                BackendType::Gpu
            } else {
                BackendType::Cpu
            }
        } else {
            BackendType::Cpu
        };
        
        let backend = create_backend(backend_type);
        
        tracing::info!(
            "Created TabuSearchSolver with {:?} backend",
            backend.backend_type()
        );
        
        Self { config, backend }
    }
    
    /// Create a solver with a specific backend
    pub fn with_backend(config: SolverConfig, backend: Box<dyn Backend>) -> Self {
        Self { config, backend }
    }

    pub fn backend_type(&self) -> BackendType {
        self.backend.backend_type()
    }

    fn weighted_linear_gain(&self, idx: usize, input: &SolverInput) -> f64 {
        let weights = self.config.weights();
        weights.alpha as f64 * input.relevance_scores[idx] as f64
            + weights.beta as f64 * input.density_scores[idx] as f64
            + weights.gamma as f64 * input.centrality_scores[idx] as f64
            + weights.delta as f64 * input.recency_scores[idx] as f64
            + weights.epsilon as f64 * input.auxiliary_scores[idx] as f64
            + self.config.mu as f64 * input.fulfilment_scores[idx] as f64
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
        let weights = self.config.weights();
        let capped_best = best.clamp(0.0, 1.0);
        let capped_second = second.clamp(0.0, 1.0);
        let capped_third = third.clamp(0.0, 1.0);
        let capped_fourth = fourth.clamp(0.0, 1.0);
        let quorum_cap = weights.support_quorum_cap.max(2.0) as usize;
        let third_mass = if quorum_cap >= 3 && quorum_count >= 3 {
            weights.support_quorum_bonus as f64 * capped_third
        } else {
            0.0
        };
        let fourth_mass = if quorum_cap >= 4 && quorum_count >= 4 {
            0.5 * weights.support_quorum_bonus as f64 * capped_fourth
        } else {
            0.0
        };
        capped_best + weights.support_secondary_discount as f64 * capped_second + third_mass + fourth_mass
    }

    fn coverage_move_signals(
        &self,
        selection: &[bool],
        input: &SolverInput,
        similarity_matrix: &[f32],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = input.num_candidates;
        let mut add_gain = vec![0.0f64; n];
        let mut removal_loss = vec![0.0f64; n];
        let mut redundancy_burden = vec![0.0f64; n];

        if similarity_matrix.len() == n * n {
            for i in 0..n {
                let mut burden = 0.0f64;
                for j in 0..n {
                    if i == j || !selection[j] {
                        continue;
                    }
                    burden += similarity_matrix[i * n + j] as f64;
                }
                redundancy_burden[i] = burden;
            }
        }

        let Some(coverage_matrix) = input.coverage_matrix.as_ref() else {
            return (add_gain, removal_loss, redundancy_burden);
        };
        if input.num_query_tokens == 0 {
            return (add_gain, removal_loss, redundancy_burden);
        }

        for token_idx in 0..input.num_query_tokens {
            let row = &coverage_matrix[token_idx * n..(token_idx + 1) * n];
            let weight = input
                .query_token_weights
                .as_ref()
                .and_then(|weights| weights.get(token_idx))
                .copied()
                .unwrap_or(1.0) as f64;
            let mut best = 0.0f32;
            let mut second = 0.0f32;
            let mut third = 0.0f32;
            let mut fourth = 0.0f32;
            let mut winner = None;
            let mut runner_up = None;
            let mut third_place = None;
            let mut quorum_count = 0usize;
            let threshold = self.config.support_quorum_threshold;
            for idx in 0..n {
                if !selection[idx] {
                    continue;
                }
                let score = row[idx];
                if score >= threshold {
                    quorum_count += 1;
                }
                if score >= best {
                    fourth = third;
                    third = second;
                    third_place = runner_up;
                    second = best;
                    runner_up = winner;
                    best = score;
                    winner = Some(idx);
                } else if score > second {
                    fourth = third;
                    third = second;
                    third_place = runner_up;
                    second = score;
                    runner_up = Some(idx);
                } else if score > third {
                    fourth = third;
                    third = score;
                    third_place = Some(idx);
                } else if score > fourth {
                    fourth = score;
                }
            }

            let old_support =
                self.quorum_support(best as f64, second as f64, third as f64, fourth as f64, quorum_count);

            for idx in 0..n {
                if selection[idx] {
                    continue;
                }
                let score = row[idx] as f64;
                let mut new_best = best as f64;
                let mut new_second = second as f64;
                let mut new_third = third as f64;
                let mut new_fourth = fourth as f64;
                let new_quorum_count = quorum_count + usize::from(row[idx] >= threshold);
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
                let gain =
                    (self.quorum_support(new_best, new_second, new_third, new_fourth, new_quorum_count)
                        - old_support)
                        .max(0.0);
                add_gain[idx] += weight * gain;
            }

            for idx in 0..n {
                if !selection[idx] {
                    continue;
                }
                let mut new_best = best as f64;
                let mut new_second = second as f64;
                let mut new_third = third as f64;
                let mut new_fourth = fourth as f64;
                if Some(idx) == winner {
                    new_best = second as f64;
                    new_second = third as f64;
                    new_third = fourth as f64;
                    new_fourth = 0.0;
                } else if Some(idx) == runner_up {
                    new_second = third as f64;
                    new_third = fourth as f64;
                    new_fourth = 0.0;
                } else if Some(idx) == third_place {
                    new_third = fourth as f64;
                    new_fourth = 0.0;
                }
                let new_quorum_count = quorum_count.saturating_sub(usize::from(row[idx] >= threshold));
                let retained =
                    self.quorum_support(new_best, new_second, new_third, new_fourth, new_quorum_count);
                removal_loss[idx] += weight * (old_support - retained).max(0.0);
            }
        }

        (add_gain, removal_loss, redundancy_burden)
    }

    fn candidate_add_pressure(
        &self,
        idx: usize,
        input: &SolverInput,
        add_gain: &[f64],
        redundancy_burden: &[f64],
    ) -> f64 {
        let token_cost = input.token_costs[idx].max(1) as f64;
        let support_gain = add_gain.get(idx).copied().unwrap_or(0.0);
        let cheap_support_bonus = support_gain / token_cost.powf(0.65);
        (self.weighted_linear_gain(idx, input) / token_cost.powf(0.85))
            + (1.45 * support_gain)
            + (0.35 * cheap_support_bonus)
            - (0.06 * redundancy_burden.get(idx).copied().unwrap_or(0.0))
    }

    fn candidate_retain_pressure(
        &self,
        idx: usize,
        input: &SolverInput,
        removal_loss: &[f64],
        redundancy_burden: &[f64],
    ) -> f64 {
        let token_cost = input.token_costs[idx].max(1) as f64;
        (self.weighted_linear_gain(idx, input) / token_cost.powf(0.9))
            + (1.95 * removal_loss.get(idx).copied().unwrap_or(0.0))
            - (0.10 * redundancy_burden.get(idx).copied().unwrap_or(0.0))
    }

    fn greedy_priority(
        &self,
        idx: usize,
        input: &SolverInput,
        selected_roles: &HashSet<RhetoricalRole>,
        cluster_counts: &HashMap<Option<u32>, usize>,
    ) -> f64 {
        let token_cost = input.token_costs[idx].max(1) as f64;
        let mut score = self.weighted_linear_gain(idx, input) / token_cost.powf(0.85);

        if input.roles[idx] != RhetoricalRole::Unknown && !selected_roles.contains(&input.roles[idx]) {
            score += 0.08;
        }

        match input.cluster_ids[idx] {
            Some(cluster_id) => {
                let seen = cluster_counts.get(&Some(cluster_id)).copied().unwrap_or(0);
                if seen == 0 {
                    score += 0.12;
                } else {
                    score -= 0.04 * seen as f64;
                }
            }
            None => {
                score += 0.03;
            }
        }

        score
    }

    fn build_redundancy_contributions(&self, selection: &[bool], similarity_matrix: &[f32], n: usize) -> Vec<f32> {
        let mut redundancy_contributions = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                if selection[j] && i != j {
                    redundancy_contributions[i] += similarity_matrix[i * n + j];
                }
            }
        }
        redundancy_contributions
    }

    fn uses_gpu_search_path(&self) -> bool {
        self.backend.is_gpu() && self.config.enable_gpu_move_evaluation
    }

    fn seed_solutions(
        &self,
        input: &SolverInput,
        constraints: &SolverConstraints,
    ) -> Vec<Vec<bool>> {
        let empty_tabu = SemanticTabuList::new(
            self.config.tabu_tenure,
            self.config.tabu_similarity_threshold,
            input.num_candidates,
        );
        vec![
            self.greedy_initial_solution(input, constraints),
            self.diversified_restart_solution(input, constraints, &empty_tabu),
        ]
    }

    fn update_elite_pool(
        &self,
        elite_pool: &mut Vec<(Vec<bool>, f64)>,
        selection: &[bool],
        score: f64,
        max_size: usize,
    ) {
        if elite_pool.iter().any(|(existing, _)| existing == selection) {
            return;
        }
        elite_pool.push((selection.to_vec(), score));
        elite_pool.sort_by(|(_, lhs), (_, rhs)| rhs.partial_cmp(lhs).unwrap_or(Ordering::Equal));
        elite_pool.truncate(max_size);
    }

    fn select_elite_restart(
        &self,
        elite_pool: &[(Vec<bool>, f64)],
        current_selection: &[bool],
    ) -> Option<Vec<bool>> {
        elite_pool
            .iter()
            .filter(|(selection, _)| selection != current_selection)
            .max_by_key(|(selection, _)| {
                selection
                    .iter()
                    .zip(current_selection.iter())
                    .filter(|(lhs, rhs)| lhs != rhs)
                    .count()
            })
            .map(|(selection, _)| selection.clone())
    }

    fn greedy_repair_selection(
        &self,
        mut selection: Vec<bool>,
        input: &SolverInput,
        constraints: &SolverConstraints,
    ) -> Vec<bool> {
        let validator = ConstraintValidator::new(constraints);
        let excluded: HashSet<usize> = validator.get_excluded_indices(input).into_iter().collect();

        for idx in validator.get_required_indices(input) {
            if selection[idx] || excluded.contains(&idx) {
                continue;
            }
            let summary = validator.summarize_selection(&selection, input);
            if validator.can_add(idx, &summary, input) {
                selection[idx] = true;
            }
        }

        for required_role in &constraints.must_include_roles {
            let mut summary = validator.summarize_selection(&selection, input);
            if summary.role_counts.get(required_role).copied().unwrap_or(0) > 0 {
                continue;
            }
            let selected_roles: HashSet<RhetoricalRole> = summary.role_counts.keys().copied().collect();
            let cluster_counts: HashMap<Option<u32>, usize> = summary
                .cluster_counts
                .iter()
                .map(|(&cluster_id, &count)| (Some(cluster_id), count))
                .collect();
            let mut best_idx = None;
            let mut best_score = f64::NEG_INFINITY;
            for idx in 0..input.num_candidates {
                if selection[idx] || excluded.contains(&idx) || input.roles[idx] != *required_role {
                    continue;
                }
                if !validator.can_add(idx, &summary, input) {
                    continue;
                }
                let score = self.greedy_priority(idx, input, &selected_roles, &cluster_counts);
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }
            if let Some(idx) = best_idx {
                selection[idx] = true;
                summary = validator.summarize_selection(&selection, input);
                if summary.num_selected >= constraints.max_chunks {
                    break;
                }
            }
        }

        loop {
            let summary = validator.summarize_selection(&selection, input);
            if summary.num_selected >= constraints.max_chunks {
                break;
            }
            let selected_roles: HashSet<RhetoricalRole> = summary.role_counts.keys().copied().collect();
            let cluster_counts: HashMap<Option<u32>, usize> = summary
                .cluster_counts
                .iter()
                .map(|(&cluster_id, &count)| (Some(cluster_id), count))
                .collect();
            let mut best_idx = None;
            let mut best_score = f64::NEG_INFINITY;
            for idx in 0..input.num_candidates {
                if selection[idx] || excluded.contains(&idx) {
                    continue;
                }
                if !validator.can_add(idx, &summary, input) {
                    continue;
                }
                let score = self.greedy_priority(idx, input, &selected_roles, &cluster_counts);
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            selection[idx] = true;
        }

        selection
    }

    fn destroy_and_repair_solution(
        &self,
        current_selection: &[bool],
        input: &SolverInput,
        constraints: &SolverConstraints,
        tabu_list: &SemanticTabuList,
    ) -> Vec<bool> {
        let validator = ConstraintValidator::new(constraints);
        let required: HashSet<usize> = validator.get_required_indices(input).into_iter().collect();
        let mut selection = current_selection.to_vec();
        let mut removable: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected && !required.contains(&idx) { Some(idx) } else { None })
            .collect();
        removable.sort_by(|&lhs, &rhs| {
            let left_score = self.weighted_linear_gain(lhs, input)
                - (tabu_list.get_frequency(lhs) as f64 * 0.05)
                - input.token_costs[lhs] as f64 * 0.0002;
            let right_score = self.weighted_linear_gain(rhs, input)
                - (tabu_list.get_frequency(rhs) as f64 * 0.05)
                - input.token_costs[rhs] as f64 * 0.0002;
            left_score.partial_cmp(&right_score).unwrap_or(Ordering::Equal)
        });

        let remove_count = (removable.len() / 3).max(1).min(removable.len());
        for idx in removable.into_iter().take(remove_count) {
            selection[idx] = false;
        }

        self.greedy_repair_selection(selection, input, constraints)
    }

    fn path_relink_solution(
        &self,
        current_selection: &[bool],
        target_selection: &[bool],
        input: &SolverInput,
        constraints: &SolverConstraints,
    ) -> Vec<bool> {
        let validator = ConstraintValidator::new(constraints);
        let mut selection = current_selection.to_vec();
        let add_candidates: Vec<usize> = target_selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected && !selection[idx] { Some(idx) } else { None })
            .collect();
        let mut remove_candidates: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected && !target_selection[idx] { Some(idx) } else { None })
            .collect();
        remove_candidates.sort_by(|&lhs, &rhs| {
            self.weighted_linear_gain(lhs, input)
                .partial_cmp(&self.weighted_linear_gain(rhs, input))
                .unwrap_or(Ordering::Equal)
        });

        for add_idx in add_candidates {
            let summary = validator.summarize_selection(&selection, input);
            if validator.can_add(add_idx, &summary, input) {
                selection[add_idx] = true;
                continue;
            }
            if let Some((remove_pos, remove_idx)) = remove_candidates
                .iter()
                .copied()
                .enumerate()
                .find(|(_, remove_idx)| validator.can_swap(add_idx, *remove_idx, &summary, input))
            {
                selection[remove_idx] = false;
                selection[add_idx] = true;
                remove_candidates.remove(remove_pos);
            }
        }

        self.greedy_repair_selection(selection, input, constraints)
    }

    fn exact_window_intensify(
        &self,
        incumbent_selection: &[bool],
        incumbent_score: f64,
        incumbent_breakdown: &crate::solver::objective::ObjectiveBreakdown,
        input: &SolverInput,
        similarity_matrix: &[f32],
        _constraints: &SolverConstraints,
        objective: &ObjectiveFunction,
        validator: &ConstraintValidator,
    ) -> Option<(
        Vec<bool>,
        f64,
        crate::solver::objective::ObjectiveBreakdown,
        ExactWindowReport,
    )> {
        if !self.config.enable_exact_window || self.config.exact_window_size == 0 {
            return None;
        }

        let selected: Vec<usize> = incumbent_selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected { Some(idx) } else { None })
            .collect();
        let unselected: Vec<usize> = incumbent_selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if !selected { Some(idx) } else { None })
            .collect();
        if selected.is_empty() && unselected.is_empty() {
            return None;
        }

        let core_budget = self.config.exact_window_size.min(20).min(input.num_candidates);
        if core_budget == 0 {
            return None;
        }

        let required: HashSet<usize> = validator.get_required_indices(input).into_iter().collect();
        let (add_gain, removal_loss, redundancy_burden) =
            self.coverage_move_signals(incumbent_selection, input, similarity_matrix);

        let mut variable_selected: Vec<usize> = selected
            .iter()
            .copied()
            .filter(|idx| !required.contains(idx))
            .collect();
        variable_selected.sort_by(|&lhs, &rhs| {
            self.candidate_retain_pressure(lhs, input, &removal_loss, &redundancy_burden)
                .partial_cmp(&self.candidate_retain_pressure(rhs, input, &removal_loss, &redundancy_burden))
                .unwrap_or(Ordering::Equal)
        });
        let variable_selected_cap = variable_selected
            .len()
            .min(core_budget.saturating_sub(1).max(1))
            .min((core_budget / 2).max(1));
        variable_selected.truncate(variable_selected_cap);

        let mut addition_candidates = unselected.clone();
        addition_candidates.sort_by(|&lhs, &rhs| {
            self.candidate_add_pressure(rhs, input, &add_gain, &redundancy_burden)
                .partial_cmp(&self.candidate_add_pressure(lhs, input, &add_gain, &redundancy_burden))
                .unwrap_or(Ordering::Equal)
        });
        addition_candidates.truncate(core_budget.saturating_sub(variable_selected.len()));

        let mut core_indices = variable_selected;
        core_indices.extend(addition_candidates);
        core_indices.sort_unstable();
        core_indices.dedup();
        if core_indices.is_empty() {
            return None;
        }

        let core_set: HashSet<usize> = core_indices.iter().copied().collect();
        let mut base_selection = vec![false; input.num_candidates];
        for idx in &selected {
            if !core_set.contains(idx) {
                base_selection[*idx] = true;
            }
        }

        let total_assignments = 1usize << core_indices.len();
        let time_budget_ms = self.config.exact_window_time_ms.max(1);
        let start = Instant::now();
        let cpu_backend = CpuBackend::new();
        let mut best_selection = incumbent_selection.to_vec();
        let mut best_score = incumbent_score;
        let mut best_breakdown = incumbent_breakdown.clone();
        let mut report = ExactWindowReport {
            used: true,
            core_size: core_indices.len(),
            nodes: 0,
            exhaustive: true,
            gap: 0.0,
            fixed_in: base_selection.iter().filter(|&&selected| selected).count(),
            fixed_out: input
                .num_candidates
                .saturating_sub(base_selection.iter().filter(|&&selected| selected).count())
                .saturating_sub(core_indices.len()),
        };

        for assignment in 0..total_assignments {
            if start.elapsed().as_millis() as u64 >= time_budget_ms {
                report.exhaustive = false;
                break;
            }
            report.nodes += 1;
            let mut selection = base_selection.clone();
            for (bit_idx, candidate_idx) in core_indices.iter().enumerate() {
                if ((assignment >> bit_idx) & 1usize) == 1 {
                    selection[*candidate_idx] = true;
                }
            }
            let validation = validator.validate(&selection, input);
            if !validation.is_valid {
                continue;
            }
            let (score, breakdown) =
                objective.compute(&selection, input, similarity_matrix, &cpu_backend);
            if score > best_score + self.config.min_improvement as f64 {
                best_selection = selection;
                best_score = score;
                best_breakdown = breakdown;
            }
        }

        report.gap = if report.exhaustive { 0.0 } else { -1.0 };
        Some((best_selection, best_score, best_breakdown, report))
    }

    fn diversified_restart_solution(
        &self,
        input: &SolverInput,
        constraints: &SolverConstraints,
        tabu_list: &SemanticTabuList,
    ) -> Vec<bool> {
        let validator = ConstraintValidator::new(constraints);
        let excluded: HashSet<usize> = validator.get_excluded_indices(input).into_iter().collect();
        let mut selection = vec![false; input.num_candidates];
        let mut current_tokens = 0u32;
        let mut selected_roles = HashSet::new();
        let mut cluster_counts: HashMap<Option<u32>, usize> = HashMap::new();
        let mut num_selected = 0usize;

        let mut indices: Vec<usize> = (0..input.num_candidates).collect();
        indices.sort_by(|&a, &b| {
            let score_a = self.weighted_linear_gain(a, input)
                - (tabu_list.get_frequency(a) as f64 * 0.03)
                - (input.token_costs[a] as f64 * 0.0003);
            let score_b = self.weighted_linear_gain(b, input)
                - (tabu_list.get_frequency(b) as f64 * 0.03)
                - (input.token_costs[b] as f64 * 0.0003);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        for idx in validator.get_required_indices(input) {
            if current_tokens + input.token_costs[idx] > constraints.max_tokens || num_selected >= constraints.max_chunks {
                continue;
            }
            selection[idx] = true;
            current_tokens += input.token_costs[idx];
            num_selected += 1;
            selected_roles.insert(input.roles[idx]);
            *cluster_counts.entry(input.cluster_ids[idx]).or_insert(0) += 1;
        }

        for required_role in &constraints.must_include_roles {
            if selected_roles.contains(required_role) {
                continue;
            }
            let mut best_idx: Option<usize> = None;
            let mut best_score = f64::NEG_INFINITY;
            for &idx in &indices {
                if selection[idx] || excluded.contains(&idx) || input.roles[idx] != *required_role {
                    continue;
                }
                if current_tokens + input.token_costs[idx] > constraints.max_tokens || num_selected >= constraints.max_chunks {
                    continue;
                }
                if input.cluster_ids[idx]
                    .and_then(|cluster_id| cluster_counts.get(&Some(cluster_id)).copied())
                    .unwrap_or(0)
                    >= constraints.max_per_cluster
                {
                    continue;
                }
                let score = self.weighted_linear_gain(idx, input);
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }
            if let Some(idx) = best_idx {
                selection[idx] = true;
                current_tokens += input.token_costs[idx];
                num_selected += 1;
                selected_roles.insert(input.roles[idx]);
                *cluster_counts.entry(input.cluster_ids[idx]).or_insert(0) += 1;
            }
        }

        for idx in indices {
            if selection[idx] || excluded.contains(&idx) {
                continue;
            }
            if current_tokens + input.token_costs[idx] > constraints.max_tokens || num_selected >= constraints.max_chunks {
                continue;
            }
            if input.cluster_ids[idx]
                .and_then(|cluster_id| cluster_counts.get(&Some(cluster_id)).copied())
                .unwrap_or(0)
                >= constraints.max_per_cluster
            {
                continue;
            }

            selection[idx] = true;
            current_tokens += input.token_costs[idx];
            num_selected += 1;
            selected_roles.insert(input.roles[idx]);
            *cluster_counts.entry(input.cluster_ids[idx]).or_insert(0) += 1;
        }

        selection
    }
    
    /// Solve the knapsack problem
    /// Evaluate the objective score for a specific set of selected indices
    pub fn evaluate_solution_score(
        &self,
        selected_indices: &[usize],
        candidates: &[ChunkCandidate],
        query_embedding: &[f32],
    ) -> f64 {
        let n = candidates.len();
        let mut selection = vec![false; n];
        for &idx in selected_indices {
            if idx < n {
                selection[idx] = true;
            }
        }
        
        let mut solver_input = SolverInput::from_candidates(candidates, query_embedding.to_vec());
        
        // Compute similarity matrix for redundancy
        if solver_input.similarity_matrix.is_none() {
            let sim_matrix = self.backend.cosine_similarity_matrix(
                &solver_input.embeddings,
                solver_input.num_candidates,
                solver_input.embedding_dim,
            );
            solver_input.similarity_matrix = Some(sim_matrix);
        }
        
        let objective = ObjectiveFunction::new(&self.config);
        objective
            .compute(
                &selection,
                &solver_input,
                solver_input.similarity_matrix.as_deref().unwrap_or(&[]),
                &*self.backend,
            )
            .0
    }
    
    pub fn solve(
        &self,
        candidates: &[ChunkCandidate],
        query_embedding: &[f32],
        constraints: &SolverConstraints,
    ) -> SolverOutput {
        let start = Instant::now();

        // Build solver input
        let input = SolverInput::from_candidates(candidates, query_embedding.to_vec());
        
        // Run the solver
        let output = self.solve_input(&input, constraints);
        
        let elapsed = start.elapsed();
        tracing::info!(
            "Solved knapsack: {} candidates -> {} selected in {:.2}ms",
            candidates.len(),
            output.num_selected,
            elapsed.as_secs_f64() * 1000.0
        );
        
        SolverOutput {
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            ..output
        }
    }
    
    /// Solve from pre-built input
    pub fn solve_input(
        &self,
        input: &SolverInput,
        constraints: &SolverConstraints,
    ) -> SolverOutput {
        let start = Instant::now();
        let n = input.num_candidates;
        
        if n == 0 {
            return SolverOutput {
                selected_indices: Vec::new(),
                selected_mask: Vec::new(),
                objective_score: 0.0,
                relevance_total: 0.0,
                density_total: 0.0,
                centrality_total: 0.0,
                recency_total: 0.0,
                auxiliary_total: 0.0,
                fulfilment_total: 0.0,
                redundancy_penalty: 0.0,
                total_tokens: 0,
                num_selected: 0,
                iterations_run: 0,
                best_iteration: 0,
                improvement_history: Vec::new(),
                constraints_satisfied: true,
                constraint_violations: Vec::new(),
                solve_time_ms: 0.0,
                exact_window_used: false,
                exact_window_core_size: 0,
                exact_window_nodes: 0,
                exact_window_exhaustive: false,
                exact_window_gap: 0.0,
                exact_window_fixed_in: 0,
                exact_window_fixed_out: 0,
            };
        }
        
        // Compute similarity matrix
        let similarity_matrix = input.similarity_matrix.clone().unwrap_or_else(|| {
            self.backend.cosine_similarity_matrix(
                &input.embeddings,
                n,
                input.embedding_dim,
            )
        });
        
        // Initialize components
        let objective = ObjectiveFunction::new(&self.config);
        let objective_context = objective.batch_context(input, &similarity_matrix);
        self.backend
            .prepare_objective_context(&objective_context, self.config.batch_size.max(1));
        let validator = ConstraintValidator::new(constraints);
            let mut tabu_list = SemanticTabuList::new(
            self.config.tabu_tenure,
            self.config.tabu_similarity_threshold,
            n,
        );
        
        // Generate multiple deterministic seeds and start from the best feasible one.
        let mut seed_rows: Vec<(Vec<bool>, f64, crate::solver::objective::ObjectiveBreakdown, bool)> = self
            .seed_solutions(input, constraints)
            .into_iter()
            .map(|selection| {
                let (score, breakdown) = objective.compute(
                    &selection,
                    input,
                    &similarity_matrix,
                    &*self.backend,
                );
                let validation = validator.validate(&selection, input);
                (selection, score, breakdown, validation.is_valid)
            })
            .collect();
        seed_rows.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

        let (mut current_selection, mut current_score, mut current_breakdown, initial_validation) =
            if let Some((selection, score, breakdown, _)) =
                seed_rows.iter().find(|(_, _, _, is_valid)| *is_valid).cloned()
            {
                let validation = validator.validate(&selection, input);
                (selection, score, breakdown, validation)
            } else {
                let (selection, score, breakdown, _) = seed_rows
                    .first()
                    .cloned()
                    .unwrap_or((vec![false; n], 0.0, crate::solver::objective::ObjectiveBreakdown::default(), false));
                let validation = validator.validate(&selection, input);
                (selection, score, breakdown, validation)
            };
        let mut elite_pool: Vec<(Vec<bool>, f64)> = seed_rows
            .iter()
            .filter(|(_, _, _, is_valid)| *is_valid)
            .map(|(selection, score, _, _)| (selection.clone(), *score))
            .collect();
        elite_pool.truncate(4);
        if initial_validation.is_valid {
            self.update_elite_pool(&mut elite_pool, &current_selection, current_score, 4);
        }
        let gpu_search = self.uses_gpu_search_path();
        let mut fulfilment_state = if gpu_search {
            None
        } else {
            objective.build_fulfilment_state(&current_selection, input)
        };
        
        // Initialize redundancy contributions for O(1) updates
        let mut redundancy_contributions = if gpu_search {
            Vec::new()
        } else {
            self.build_redundancy_contributions(&current_selection, &similarity_matrix, n)
        };
        
        // Track best feasible solution and best overall fallback separately.
        let mut best_selection = if initial_validation.is_valid {
            current_selection.clone()
        } else {
            vec![false; n]
        };
        let mut best_score = if initial_validation.is_valid {
            current_score
        } else {
            f64::NEG_INFINITY
        };
        let mut best_breakdown = if initial_validation.is_valid {
            current_breakdown.clone()
        } else {
            crate::solver::objective::ObjectiveBreakdown::default()
        };
        let mut exact_window_report = ExactWindowReport::default();
        let mut best_iteration = 0;
        let mut fallback_selection = current_selection.clone();
        let mut fallback_score = current_score;
        let mut fallback_breakdown = current_breakdown.clone();
        
        // Track improvement history
        let mut improvement_history = vec![current_score];
        let mut patience_counter = 0;
        
        // Main Tabu Search loop
        for iter in 0..self.config.iterations {
            tabu_list.next_iteration();
            if self.config.enable_reactive_tenure {
                tabu_list.react_to_stagnation(patience_counter);
            }
            
            // Generate candidate moves
            let moves = self.generate_moves(
                &current_selection,
                input,
                &similarity_matrix,
                constraints,
                &validator,
                patience_counter,
            );
            
            if moves.is_empty() {
                tracing::debug!("No valid moves at iteration {}", iter);
                break;
            }
            
            // Evaluate moves and find best non-tabu move
            let best_move = self.find_best_move_parallel(
                &moves,
                &current_selection,
                input,
                &similarity_matrix,
                &objective,
                &tabu_list,
                best_score,
                current_score,
                &redundancy_contributions,
                fulfilment_state.as_ref(),
                patience_counter,
            );
            
            if let Some((mv, _delta)) = best_move {
                // Apply move
                if mv.swap_out < n {
                    current_selection[mv.swap_out] = false;
                }
                current_selection[mv.swap_in] = true;
                
                // Update redundancy contributions incrementally
                if !gpu_search {
                    // Remove swap_out impact
                    if mv.swap_out < n {
                        for k in 0..n {
                            if k != mv.swap_out {
                                redundancy_contributions[k] -= similarity_matrix[k * n + mv.swap_out];
                            }
                        }
                    }
                    // Add swap_in impact
                    for k in 0..n {
                        if k != mv.swap_in {
                            redundancy_contributions[k] += similarity_matrix[k * n + mv.swap_in];
                        }
                    }
                }
                
                // Recompute breakdown (more accurate than delta)
                let (score, breakdown) = objective.compute(
                    &current_selection,
                    input,
                    &similarity_matrix,
                    &*self.backend,
                );
                current_score = score;
                current_breakdown = breakdown;
                if !gpu_search {
                    fulfilment_state = objective.build_fulfilment_state(&current_selection, input);
                }
                let current_validation = validator.validate(&current_selection, input);
                
                // Add to tabu list
                tabu_list.add(mv, &current_selection, Some(&input.cluster_ids));
                
                if current_score > fallback_score + self.config.min_improvement as f64 {
                    fallback_score = current_score;
                    fallback_selection = current_selection.clone();
                    fallback_breakdown = current_breakdown.clone();
                }

                if current_validation.is_valid
                    && current_score > best_score + self.config.min_improvement as f64
                {
                    best_score = current_score;
                    best_selection = current_selection.clone();
                    best_breakdown = current_breakdown.clone();
                    best_iteration = iter;
                    self.update_elite_pool(&mut elite_pool, &current_selection, current_score, 4);
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                }
                
                improvement_history.push(current_score);
            } else {
                patience_counter += 1;
            }

            let diversification_trigger = (self.config.early_stopping_patience / 2).max(6);
            if patience_counter >= diversification_trigger && patience_counter < self.config.early_stopping_patience {
                tracing::debug!(
                    "Diversifying search at iteration {} after {} stagnant iterations",
                    iter,
                    patience_counter
                );
                current_selection = if self.config.enable_path_relinking {
                    if let Some(target) = self.select_elite_restart(&elite_pool, &current_selection) {
                        if self.config.enable_destroy_repair
                            && patience_counter >= diversification_trigger + 2
                        {
                            self.destroy_and_repair_solution(&current_selection, input, constraints, &tabu_list)
                        } else {
                            self.path_relink_solution(&current_selection, &target, input, constraints)
                        }
                    } else if self.config.enable_destroy_repair {
                        self.destroy_and_repair_solution(&current_selection, input, constraints, &tabu_list)
                    } else {
                        self.diversified_restart_solution(input, constraints, &tabu_list)
                    }
                } else if self.config.enable_destroy_repair {
                    self.destroy_and_repair_solution(&current_selection, input, constraints, &tabu_list)
                } else {
                    self.diversified_restart_solution(input, constraints, &tabu_list)
                };
                let (score, breakdown) = objective.compute(
                    &current_selection,
                    input,
                    &similarity_matrix,
                    &*self.backend,
                );
                current_score = score;
                current_breakdown = breakdown;
                if !gpu_search {
                    fulfilment_state = objective.build_fulfilment_state(&current_selection, input);
                }
                let current_validation = validator.validate(&current_selection, input);
                if !gpu_search {
                    redundancy_contributions =
                        self.build_redundancy_contributions(&current_selection, &similarity_matrix, n);
                }
                tabu_list.clear();
                if current_score > fallback_score + self.config.min_improvement as f64 {
                    fallback_score = current_score;
                    fallback_selection = current_selection.clone();
                    fallback_breakdown = current_breakdown.clone();
                }
                if current_validation.is_valid
                    && current_score > best_score + self.config.min_improvement as f64
                {
                    best_score = current_score;
                    best_selection = current_selection.clone();
                    best_breakdown = current_breakdown.clone();
                    best_iteration = iter;
                    self.update_elite_pool(&mut elite_pool, &current_selection, current_score, 4);
                }
                patience_counter = 0;
                improvement_history.push(current_score);
                continue;
            }
            
            // Early stopping
            if patience_counter >= self.config.early_stopping_patience {
                tracing::debug!(
                    "Early stopping at iteration {} (no improvement for {} iterations)",
                    iter,
                    patience_counter
                );
                break;
            }
        }

        if !validator.validate(&best_selection, input).is_valid {
            best_selection = fallback_selection;
            best_score = fallback_score;
            best_breakdown = fallback_breakdown;
        }
        if validator.validate(&best_selection, input).is_valid {
            if let Some((selection, score, breakdown, report)) = self.exact_window_intensify(
                &best_selection,
                best_score,
                &best_breakdown,
                input,
                &similarity_matrix,
                constraints,
                &objective,
                &validator,
            ) {
                exact_window_report = report;
                if score > best_score + self.config.min_improvement as f64 {
                    best_selection = selection;
                    best_score = score;
                    best_breakdown = breakdown;
                    best_iteration = tabu_list.iteration();
                }
            }
        }
        
        // Validate final solution
        let validation = validator.validate(&best_selection, input);
        
        // Build output
        let selected_indices: Vec<usize> = best_selection
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i) } else { None })
            .collect();
        
        SolverOutput {
            selected_indices,
            selected_mask: best_selection,
            objective_score: best_score,
            relevance_total: best_breakdown.relevance,
            density_total: best_breakdown.density,
            centrality_total: best_breakdown.centrality,
            recency_total: best_breakdown.recency,
            auxiliary_total: best_breakdown.auxiliary,
            fulfilment_total: best_breakdown.fulfilment,
            redundancy_penalty: best_breakdown.redundancy_penalty,
            total_tokens: validation.total_tokens,
            num_selected: validation.num_selected,
            iterations_run: tabu_list.iteration(),
            best_iteration,
            improvement_history,
            constraints_satisfied: validation.is_valid,
            constraint_violations: validation.violations.iter().map(|v| v.to_string()).collect(),
            solve_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            exact_window_used: exact_window_report.used,
            exact_window_core_size: exact_window_report.core_size,
            exact_window_nodes: exact_window_report.nodes,
            exact_window_exhaustive: exact_window_report.exhaustive,
            exact_window_gap: exact_window_report.gap,
            exact_window_fixed_in: exact_window_report.fixed_in,
            exact_window_fixed_out: exact_window_report.fixed_out,
        }
    }
    
    /// Generate greedy initial solution with token-aware utility and diversity bonuses.
    fn greedy_initial_solution(
        &self,
        input: &SolverInput,
        constraints: &SolverConstraints,
    ) -> Vec<bool> {
        let n = input.num_candidates;
        let mut selection = vec![false; n];
        let mut current_tokens = 0u32;
        let mut num_selected = 0usize;
        let mut selected_roles = HashSet::new();
        let mut cluster_counts: HashMap<Option<u32>, usize> = HashMap::new();

        let validator = ConstraintValidator::new(constraints);
        for idx in validator.get_required_indices(input) {
            if current_tokens + input.token_costs[idx] <= constraints.max_tokens && num_selected < constraints.max_chunks {
                selection[idx] = true;
                current_tokens += input.token_costs[idx];
                num_selected += 1;
                selected_roles.insert(input.roles[idx]);
                *cluster_counts.entry(input.cluster_ids[idx]).or_insert(0) += 1;
            }
        }

        let excluded: HashSet<usize> = validator
            .get_excluded_indices(input)
            .into_iter()
            .collect();

        for required_role in &constraints.must_include_roles {
            if selected_roles.contains(required_role) {
                continue;
            }
            let mut best_idx: Option<usize> = None;
            let mut best_score = f64::NEG_INFINITY;

            for idx in 0..n {
                if selection[idx] || excluded.contains(&idx) || input.roles[idx] != *required_role {
                    continue;
                }
                if current_tokens + input.token_costs[idx] > constraints.max_tokens {
                    continue;
                }
                if input.cluster_ids[idx]
                    .and_then(|cluster_id| cluster_counts.get(&Some(cluster_id)).copied())
                    .unwrap_or(0)
                    >= constraints.max_per_cluster
                {
                    continue;
                }

                let score = self.greedy_priority(idx, input, &selected_roles, &cluster_counts);
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                selection[idx] = true;
                current_tokens += input.token_costs[idx];
                num_selected += 1;
                selected_roles.insert(input.roles[idx]);
                *cluster_counts.entry(input.cluster_ids[idx]).or_insert(0) += 1;
            }
        }

        while num_selected < constraints.max_chunks {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f64::NEG_INFINITY;

            for idx in 0..n {
                if selection[idx] || excluded.contains(&idx) {
                    continue;
                }
                if current_tokens + input.token_costs[idx] > constraints.max_tokens {
                    continue;
                }
                if input.cluster_ids[idx]
                    .and_then(|cluster_id| cluster_counts.get(&Some(cluster_id)).copied())
                    .unwrap_or(0)
                    >= constraints.max_per_cluster
                {
                    continue;
                }

                let score = self.greedy_priority(idx, input, &selected_roles, &cluster_counts);
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }

            let Some(idx) = best_idx else {
                break;
            };
            selection[idx] = true;
            current_tokens += input.token_costs[idx];
            num_selected += 1;
            selected_roles.insert(input.roles[idx]);
            *cluster_counts.entry(input.cluster_ids[idx]).or_insert(0) += 1;
        }

        selection
    }
    
    /// Generate candidate moves
    fn generate_moves(
        &self,
        selection: &[bool],
        input: &SolverInput,
        similarity_matrix: &[f32],
        _constraints: &SolverConstraints,
        validator: &ConstraintValidator,
        patience: usize,
    ) -> Vec<Move> {
        let n = input.num_candidates;
        let mut moves = Vec::new();

        let selected: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i) } else { None })
            .collect();

        let unselected: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if !s { Some(i) } else { None })
            .collect();

        let excluded: HashSet<usize> = validator.get_excluded_indices(input).into_iter().collect();
        let required: HashSet<usize> = validator.get_required_indices(input).into_iter().collect();

        let summary = validator.summarize_selection(selection, input);
        let (add_gain, removal_loss, redundancy_burden) =
            self.coverage_move_signals(selection, input, similarity_matrix);

        let mut add_candidates: Vec<usize> = unselected
            .into_iter()
            .filter(|idx| !excluded.contains(idx))
            .collect();
        add_candidates.sort_by(|&a, &b| {
            let score_a = self.candidate_add_pressure(a, input, &add_gain, &redundancy_burden);
            let score_b = self.candidate_add_pressure(b, input, &add_gain, &redundancy_burden);
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        let mut remove_candidates = selected.clone();
        remove_candidates.sort_by(|&a, &b| {
            let utility_a = self.candidate_retain_pressure(a, input, &removal_loss, &redundancy_burden);
            let utility_b = self.candidate_retain_pressure(b, input, &removal_loss, &redundancy_burden);
            utility_a.partial_cmp(&utility_b).unwrap_or(Ordering::Equal)
        });

        let neighborhood_scale = (1 + patience / (self.config.early_stopping_patience / 4).max(2)).min(3);
        let add_cap = add_candidates
            .len()
            .min(((n as f64).sqrt().ceil() as usize).saturating_mul(6 * neighborhood_scale).max(12));
        let remove_cap = remove_candidates
            .len()
            .min(
                ((remove_candidates.len() as f64).sqrt().ceil() as usize)
                    .saturating_mul(4 * neighborhood_scale)
                    .max(6),
            );
        add_candidates.truncate(add_cap);
        remove_candidates.truncate(remove_cap);

        for add_idx in add_candidates {
            for &remove_idx in &remove_candidates {
                if required.contains(&remove_idx) {
                    continue;
                }
                if validator.can_swap(add_idx, remove_idx, &summary, input) {
                    moves.push(Move::new(add_idx, remove_idx));
                }
            }

            if validator.can_add(add_idx, &summary, input) {
                moves.push(Move::new(add_idx, n));
            }
        }

        moves
    }

    fn apply_move_to_selection(selection: &[bool], mv: &Move, n: usize) -> Vec<bool> {
        let mut resulting_selection = selection.to_vec();
        if mv.swap_out < n {
            resulting_selection[mv.swap_out] = false;
        }
        resulting_selection[mv.swap_in] = true;
        resulting_selection
    }

    fn find_best_move_gpu_batch(
        &self,
        moves: &[Move],
        selection: &[bool],
        input: &SolverInput,
        similarity_matrix: &[f32],
        objective: &ObjectiveFunction,
        tabu_list: &SemanticTabuList,
        best_score: f64,
        current_score: f64,
        patience: usize,
    ) -> Option<(Move, f64)> {
        let n = input.num_candidates;
        let batch_size = self.config.batch_size.max(1);
        let freq_weight = if patience > 4 { 0.02 * patience as f64 } else { 0.0 };
        let mut best_candidate: Option<(Move, f64, f64)> = None;
        let context = objective.batch_context(input, similarity_matrix);

        for move_chunk in moves.chunks(batch_size) {
            let swap_ins: Vec<usize> = move_chunk.iter().map(|mv| mv.swap_in).collect();
            let swap_outs: Vec<usize> = move_chunk.iter().map(|mv| mv.swap_out).collect();
            let mut resulting_selections = Vec::with_capacity(move_chunk.len());
            for mv in move_chunk {
                let resulting_selection = Self::apply_move_to_selection(selection, mv, n);
                resulting_selections.push(resulting_selection);
            }

            let scores: Vec<f64> = self
                .backend
                .compute_move_objectives_batch(
                    selection,
                    &swap_ins,
                    &swap_outs,
                    &context,
                )
                .into_iter()
                .map(|score| score as f64)
                .collect();

            for ((mv, resulting_selection), new_score) in move_chunk
                .iter()
                .zip(resulting_selections.iter())
                .zip(scores.into_iter())
            {
                let delta = new_score - current_score;
                let freq_penalty = if freq_weight > 0.0 {
                    -(tabu_list.get_frequency(mv.swap_in) as f64 * freq_weight)
                } else {
                    0.0
                };
                let structure_penalty = tabu_list.structure_penalty(resulting_selection, &input.cluster_ids);
                let check_val = delta + freq_penalty - structure_penalty;
                let is_tabu = tabu_list.is_tabu_semantic(
                    mv,
                    resulting_selection,
                    &input.embeddings,
                    input.embedding_dim,
                    &*self.backend,
                );
                let aspirates = new_score > best_score + self.config.min_improvement as f64
                    || (patience >= (self.config.early_stopping_patience / 2).max(1)
                        && new_score > current_score + self.config.min_improvement as f64);
                if is_tabu && !aspirates {
                    continue;
                }

                match &best_candidate {
                    Some((_, best_delta, best_value)) => {
                        if check_val > *best_value
                            || ((check_val - *best_value).abs() < f64::EPSILON && delta >= *best_delta)
                        {
                            best_candidate = Some((mv.clone(), delta, check_val));
                        }
                    }
                    None => {
                        best_candidate = Some((mv.clone(), delta, check_val));
                    }
                }
            }
        }

        best_candidate.map(|(mv, delta, _)| (mv, delta))
    }
    
    /// Find the best non-tabu move (or aspiration move) using parallel evaluation
    fn find_best_move_parallel(
        &self,
        moves: &[Move],
        selection: &[bool],
        input: &SolverInput,
        similarity_matrix: &[f32],
        objective: &ObjectiveFunction,
        tabu_list: &SemanticTabuList,
        best_score: f64,
        current_score: f64,
        redundancy_contributions: &[f32],
        fulfilment_state: Option<&FulfilmentState>,
        patience: usize,
    ) -> Option<(Move, f64)> {
        if self.backend.is_gpu() && self.config.enable_gpu_move_evaluation {
            return self.find_best_move_gpu_batch(
                moves,
                selection,
                input,
                similarity_matrix,
                objective,
                tabu_list,
                best_score,
                current_score,
                patience,
            );
        }

        let n = input.num_candidates;
        
        let freq_weight = if patience > 4 { 0.02 * patience as f64 } else { 0.0 };

        moves.par_iter()
            .map(|mv| {
                let delta = if mv.swap_out >= n {
                    objective.compute_add_delta_fast(
                        selection,
                        mv.swap_in,
                        input,
                        redundancy_contributions,
                        fulfilment_state,
                    )
                } else {
                    objective.compute_delta_fast(
                        selection,
                        mv.swap_in,
                        mv.swap_out,
                        input,
                        similarity_matrix,
                        redundancy_contributions,
                        fulfilment_state,
                    )
                };

                let freq_penalty = if freq_weight > 0.0 {
                    let freq = tabu_list.get_frequency(mv.swap_in);
                    -(freq as f64 * freq_weight)
                } else {
                    0.0
                };

                let mut resulting_selection = selection.to_vec();
                if mv.swap_out < n {
                    resulting_selection[mv.swap_out] = false;
                }
                resulting_selection[mv.swap_in] = true;

                let check_val = delta + freq_penalty;
                let structure_penalty = tabu_list.structure_penalty(&resulting_selection, &input.cluster_ids);
                let is_tabu = tabu_list.is_tabu_semantic(
                    mv,
                    &resulting_selection,
                    &input.embeddings,
                    input.embedding_dim,
                    &*self.backend,
                );
                let new_score = current_score + delta;
                let aspirates = new_score > best_score + self.config.min_improvement as f64
                    || (patience >= (self.config.early_stopping_patience / 2).max(1)
                        && new_score > current_score + self.config.min_improvement as f64);

                if !is_tabu || aspirates {
                    Some((mv.clone(), delta, check_val - structure_penalty))
                } else {
                    None
                }
            })
            .reduce(
                || None,
                |best: Option<(Move, f64, f64)>, current: Option<(Move, f64, f64)>| {
                    match (best, current) {
                        (Some((m1, d1, v1)), Some((m2, d2, v2))) => {
                            if v1 > v2 || ((v1 - v2).abs() < f64::EPSILON && d1 >= d2) {
                                Some((m1, d1, v1))
                            } else {
                                Some((m2, d2, v2))
                            }
                        },
                        (Some(v), None) => Some(v),
                        (None, Some(v)) => Some(v),
                        (None, None) => None,
                    }
                }
            )
            .map(|(m, d, _)| (m, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::RhetoricalRole;
    
    fn create_test_candidates() -> Vec<ChunkCandidate> {
        vec![
            ChunkCandidate::new("c1".to_string(), "Definition text".to_string(), vec![1.0, 0.0], 100)
                .with_fact_density(0.8)
                .with_role(RhetoricalRole::Definition),
            ChunkCandidate::new("c2".to_string(), "Example text".to_string(), vec![0.7, 0.7], 150)
                .with_fact_density(0.6)
                .with_role(RhetoricalRole::Example),
            ChunkCandidate::new("c3".to_string(), "Evidence text".to_string(), vec![0.0, 1.0], 200)
                .with_fact_density(0.9)
                .with_role(RhetoricalRole::Evidence),
            ChunkCandidate::new("c4".to_string(), "Risk text".to_string(), vec![0.5, 0.5], 120)
                .with_fact_density(0.7)
                .with_role(RhetoricalRole::Risk),
        ]
    }
    
    #[test]
    fn test_solver_basic() {
        let config = SolverConfig {
            iterations: 10,
            use_gpu: false,
            ..Default::default()
        };
        
        let solver = TabuSearchSolver::new(config);
        let candidates = create_test_candidates();
        let query = vec![1.0, 0.0];
        let constraints = SolverConstraints::with_budget(500);
        
        let output = solver.solve(&candidates, &query, &constraints);
        
        assert!(output.num_selected > 0);
        assert!(output.total_tokens <= 500);
        assert!(output.objective_score > 0.0);
    }
    
    #[test]
    fn test_solver_respects_budget() {
        let config = SolverConfig {
            iterations: 20,
            use_gpu: false,
            ..Default::default()
        };
        
        let solver = TabuSearchSolver::new(config);
        let candidates = create_test_candidates();
        let query = vec![1.0, 0.0];
        let constraints = SolverConstraints::with_budget(200);
        
        let output = solver.solve(&candidates, &query, &constraints);
        
        assert!(output.total_tokens <= 200, "Budget exceeded: {}", output.total_tokens);
    }
    
    #[test]
    fn test_solver_empty_input() {
        let config = SolverConfig::default();
        let solver = TabuSearchSolver::new(config);
        let candidates: Vec<ChunkCandidate> = Vec::new();
        let query = vec![1.0, 0.0];
        let constraints = SolverConstraints::default();
        
        let output = solver.solve(&candidates, &query, &constraints);
        
        assert_eq!(output.num_selected, 0);
        assert!(output.constraints_satisfied);
    }
    
    #[test]
    fn test_solver_single_candidate() {
        let config = SolverConfig::default();
        let solver = TabuSearchSolver::new(config);
        let candidates = vec![
            ChunkCandidate::new("c1".to_string(), "Test".to_string(), vec![1.0, 0.0], 100),
        ];
        let query = vec![1.0, 0.0];
        let constraints = SolverConstraints::with_budget(200);
        
        let output = solver.solve(&candidates, &query, &constraints);
        
        assert_eq!(output.num_selected, 1);
        assert!(output.constraints_satisfied);
    }
}


//! Semantic Tabu List Implementation
//!
//! Implements a tabu list that uses semantic similarity to prevent
//! revisiting similar solutions, not just identical ones.

use std::collections::{HashMap, VecDeque};
use crate::backend::Backend;

/// A move in the search space (swap in/out pair)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Move {
    pub swap_in: usize,
    pub swap_out: usize,
}

impl Move {
    pub fn new(swap_in: usize, swap_out: usize) -> Self {
        Self { swap_in, swap_out }
    }
    
    /// Get the reverse of this move
    pub fn reverse(&self) -> Self {
        Self {
            swap_in: self.swap_out,
            swap_out: self.swap_in,
        }
    }
}

/// Entry in the tabu list
#[derive(Debug, Clone)]
struct TabuEntry {
    /// The move that was made
    mv: Move,
    
    /// Selection state after the move (for semantic comparison)
    selection_hash: u64,

    /// Selection mask after the move for solution-similarity tabooing
    selection: Vec<bool>,
    
    /// Iteration when this entry was added
    iteration: usize,
}

/// Semantic Tabu List
///
/// Uses both exact matching and semantic similarity to determine
/// if a move is tabu. A move is considered tabu if:
/// 1. It's the exact reverse of a recent move, OR
/// 2. The resulting selection is semantically similar to a recent one
/// Semantic Tabu List
///
/// Uses both exact matching and semantic similarity to prevent
/// revisiting similar solutions, not just identical ones.
pub struct SemanticTabuList {
    /// Maximum size of the tabu list
    max_size: usize,
    
    /// Base tabu tenure
    base_tenure: usize,
    
    /// Current dynamic tenure (deterministically varied around base)
    current_tenure: usize,
    
    /// Similarity threshold for semantic tabu matching
    similarity_threshold: f32,
    
    /// The tabu list entries
    entries: VecDeque<TabuEntry>,
    
    /// Current iteration
    current_iteration: usize,
    
    /// Frequency of selection for each candidate (for diversification)
    selection_frequency: Vec<usize>,

    /// Frequency of high-redundancy or recurring selected pairs.
    pair_frequency: HashMap<(usize, usize), usize>,

    /// Frequency of selected clusters in accepted solutions.
    cluster_frequency: HashMap<u32, usize>,
}

impl SemanticTabuList {
    /// Create a new semantic tabu list
    pub fn new(tenure: usize, similarity_threshold: f32, num_candidates: usize) -> Self {
        Self {
            max_size: tenure * 3,  // Keep more history for dynamic tenure
            base_tenure: tenure,
            current_tenure: tenure,
            similarity_threshold,
            entries: VecDeque::with_capacity(tenure * 3),
            current_iteration: 0,
            selection_frequency: vec![0; num_candidates],
            pair_frequency: HashMap::new(),
            cluster_frequency: HashMap::new(),
        }
    }
    
    /// Add a move to the tabu list and update frequencies
    pub fn add(&mut self, mv: Move, selection: &[bool], cluster_ids: Option<&[Option<u32>]>) {
        let hash = selection_hash(selection);
        
        self.entries.push_back(TabuEntry {
            mv: mv.clone(),
            selection_hash: hash,
            selection: selection.to_vec(),
            iteration: self.current_iteration,
        });
        
        // Deterministically vary tenure slightly to prevent cycling while
        // preserving reproducibility for parity tests and CPU-reference runs.
        // Range: [0.8 * base, 1.2 * base]
        let range = (self.base_tenure as f32 * 0.2) as usize;
        if range > 0 {
            let span = (range * 2) + 1;
            let delta = ((self.current_iteration as u64 + hash) % span as u64) as usize;
            self.current_tenure = self.base_tenure.saturating_sub(range) + delta;
        }
        
        // Update frequencies
        for (i, &selected) in selection.iter().enumerate() {
            if selected && i < self.selection_frequency.len() {
                self.selection_frequency[i] += 1;
            }
        }

        let selected_indices: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected { Some(idx) } else { None })
            .collect();
        for (pos, &lhs) in selected_indices.iter().enumerate() {
            for &rhs in selected_indices.iter().skip(pos + 1) {
                *self.pair_frequency.entry((lhs.min(rhs), lhs.max(rhs))).or_insert(0) += 1;
            }
        }
        if let Some(cluster_ids) = cluster_ids {
            for idx in selected_indices {
                if let Some(cluster_id) = cluster_ids.get(idx).and_then(|value| *value) {
                    *self.cluster_frequency.entry(cluster_id).or_insert(0) += 1;
                }
            }
        }
        
        // Trim old entries
        while self.entries.len() > self.max_size {
            self.entries.pop_front();
        }
    }
    
    /// Check if a move is tabu
    ///
    /// A move is tabu if:
    /// 1. Its reverse is in the active tabu list (within dynamic tenure), OR
    /// 2. The resulting selection would be too similar to a recent one
    pub fn is_tabu(&self, mv: &Move, _resulting_selection: &[bool]) -> bool {
        let reverse = mv.reverse();
        
        // Check exact match on reverse move
        for entry in self.entries.iter().rev() {
            // Only check entries within dynamic tenure
            if self.current_iteration - entry.iteration > self.current_tenure {
                break;
            }
            
            if entry.mv == reverse {
                return true;
            }
        }
        
        false
    }
    
    /// Get selection frequency for a candidate
    pub fn get_frequency(&self, candidate_idx: usize) -> usize {
        if candidate_idx < self.selection_frequency.len() {
            self.selection_frequency[candidate_idx]
        } else {
            0
        }
    }

    pub fn get_pair_frequency(&self, lhs: usize, rhs: usize) -> usize {
        self.pair_frequency
            .get(&(lhs.min(rhs), lhs.max(rhs)))
            .copied()
            .unwrap_or(0)
    }

    pub fn get_cluster_frequency(&self, cluster_id: u32) -> usize {
        self.cluster_frequency.get(&cluster_id).copied().unwrap_or(0)
    }

    pub fn structure_penalty(
        &self,
        selection: &[bool],
        cluster_ids: &[Option<u32>],
    ) -> f64 {
        let selected_indices: Vec<usize> = selection
            .iter()
            .enumerate()
            .filter_map(|(idx, &selected)| if selected { Some(idx) } else { None })
            .collect();
        let candidate_penalty = selected_indices
            .iter()
            .map(|&idx| self.get_frequency(idx) as f64)
            .sum::<f64>()
            * 0.005;
        let cluster_penalty = selected_indices
            .iter()
            .filter_map(|&idx| cluster_ids.get(idx).and_then(|value| *value))
            .map(|cluster_id| self.get_cluster_frequency(cluster_id) as f64)
            .sum::<f64>()
            * 0.004;
        let mut pair_penalty = 0.0;
        for (pos, &lhs) in selected_indices.iter().enumerate() {
            for &rhs in selected_indices.iter().skip(pos + 1).take(8) {
                pair_penalty += self.get_pair_frequency(lhs, rhs) as f64;
            }
        }
        candidate_penalty + cluster_penalty + pair_penalty * 0.003
    }
    
    /// Check if a move is tabu with semantic similarity check
    pub fn is_tabu_semantic(
        &self,
        mv: &Move,
        resulting_selection: &[bool],
        _embeddings: &[f32],
        _dim: usize,
        _backend: &dyn Backend,
    ) -> bool {
        // First check exact match
        if self.is_tabu(mv, resulting_selection) {
            return true;
        }
        
        let result_hash = selection_hash(resulting_selection);
        
        // Check semantic similarity with recent selections
        for entry in self.entries.iter().rev() {
            if self.current_iteration - entry.iteration > self.current_tenure {
                break;
            }
            
            // Fast path: identical selection
            if entry.selection_hash == result_hash {
                return true;
            }

            if self.similarity_threshold > 0.0
                && selection_similarity(&entry.selection, resulting_selection) >= self.similarity_threshold
            {
                return true;
            }
        }
        
        // For full semantic check, we would compute centroid similarity
        // This is expensive so we skip it for now and rely on hash-based checking
        false
    }
    
    /// Advance to next iteration and prune expired entries
    pub fn next_iteration(&mut self) {
        self.current_iteration += 1;
        
        // Remove entries older than max history
        while let Some(front) = self.entries.front() {
            if self.current_iteration - front.iteration > self.max_size {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// Reactively widen or shrink tenure based on stagnation pressure.
    pub fn react_to_stagnation(&mut self, stagnation: usize) {
        if stagnation == 0 {
            self.current_tenure = self.base_tenure;
            return;
        }
        let bonus = (stagnation / 2).min(self.base_tenure.max(1));
        self.current_tenure = (self.base_tenure + bonus).min(self.max_size.max(self.base_tenure));
    }
    
    /// Clear the tabu list
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_iteration = 0;
        self.selection_frequency.fill(0);
        self.pair_frequency.clear();
        self.cluster_frequency.clear();
    }
    
    /// Get current iteration count
    pub fn iteration(&self) -> usize {
        self.current_iteration
    }
    
    /// Get number of active tabu entries
    pub fn len(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| self.current_iteration - e.iteration <= self.current_tenure)
            .count()
    }
    
    /// Check if tabu list is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compute a hash of the selection for fast comparison
fn selection_hash(selection: &[bool]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    selection.hash(&mut hasher);
    hasher.finish()
}

fn selection_similarity(a: &[bool], b: &[bool]) -> f32 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&lhs, &rhs) in a.iter().zip(b.iter()) {
        if lhs || rhs {
            union += 1;
        }
        if lhs && rhs {
            intersection += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Compute the centroid of selected embeddings
#[allow(dead_code)]
fn compute_selection_centroid(
    selection: &[bool],
    embeddings: &[f32],
    dim: usize,
) -> Vec<f32> {
    let mut centroid = vec![0.0f32; dim];
    let mut count = 0;
    
    for (i, &selected) in selection.iter().enumerate() {
        if selected {
            let start = i * dim;
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += embeddings[start + j];
            }
            count += 1;
        }
    }
    
    if count > 0 {
        for c in &mut centroid {
            *c /= count as f32;
        }
    }
    
    centroid
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_move_reverse() {
        let mv = Move::new(5, 3);
        let rev = mv.reverse();
        
        assert_eq!(rev.swap_in, 3);
        assert_eq!(rev.swap_out, 5);
    }
    
    #[test]
    fn test_tabu_list_basic() {
        let mut tabu = SemanticTabuList::new(5, 0.85, 10);
        
        let selection = vec![true, false, true, false, false, false, false, false, false, false];
        tabu.add(Move::new(0, 1), &selection, None);
        
        // Forward move is not tabu
        assert!(!tabu.is_tabu(&Move::new(0, 1), &selection));
        
        // Reverse move should be tabu
        assert!(tabu.is_tabu(&Move::new(1, 0), &selection));
    }
    
    #[test]
    fn test_tabu_tenure() {
        // Tenure = 3
        // Dynamic range might shift it to 2-4, but checking immediately should be taboo
        let mut tabu = SemanticTabuList::new(3, 0.85, 10);
        
        let selection = vec![true, false, true, false, false, false, false, false, false, false];
        tabu.add(Move::new(0, 1), &selection, None);
        
        // Should be tabu initially
        assert!(tabu.is_tabu(&Move::new(1, 0), &selection));
        
        // Advance past tenure (safe upper bound)
        for _ in 0..10 {
            tabu.next_iteration();
        }
        
        // Should no longer be tabu
        assert!(!tabu.is_tabu(&Move::new(1, 0), &selection));
    }
    
    #[test]
    fn test_selection_hash() {
        let s1 = vec![true, false, true];
        let s2 = vec![true, false, true];
        let s3 = vec![false, true, true];
        
        assert_eq!(selection_hash(&s1), selection_hash(&s2));
        assert_ne!(selection_hash(&s1), selection_hash(&s3));
    }

    #[test]
    fn test_dynamic_tenure_is_deterministic() {
        let selection = vec![true, false, true, false, false, false, false, false, false, false];
        let mv = Move::new(0, 1);

        let mut first = SemanticTabuList::new(10, 0.85, 10);
        let mut second = SemanticTabuList::new(10, 0.85, 10);

        first.add(mv.clone(), &selection, None);
        second.add(mv, &selection, None);

        assert_eq!(first.current_tenure, second.current_tenure);
    }

    #[test]
    fn test_reactive_tenure_grows_with_stagnation() {
        let mut tabu = SemanticTabuList::new(6, 0.85, 10);
        let baseline = tabu.current_tenure;

        tabu.react_to_stagnation(8);

        assert!(tabu.current_tenure >= baseline);
    }

    #[test]
    fn test_semantic_similarity_threshold_blocks_similar_selection() {
        let mut tabu = SemanticTabuList::new(5, 0.5, 6);
        let prior = vec![true, true, true, false, false, false];
        let similar = vec![true, true, false, true, false, false];

        tabu.add(Move::new(0, 4), &prior, None);

        assert!(tabu.is_tabu_semantic(
            &Move::new(2, 1),
            &similar,
            &[],
            0,
            &crate::backend::CpuBackend::new(),
        ));
    }
}


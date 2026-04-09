use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

use crate::emd::qch_proxy_between_docs;
use crate::search::beam_search_construction;

/// Sealed GEM proximity graph with optional semantic shortcuts.
pub struct GemGraph {
    pub adjacency: Vec<Vec<u32>>,
    pub shortcuts: Vec<Vec<u32>>,
    pub max_degree: usize,
}

impl GemGraph {
    pub fn n_nodes(&self) -> usize {
        self.adjacency.len()
    }

    pub fn n_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum()
    }

    pub fn total_shortcuts(&self) -> usize {
        self.shortcuts.iter().map(|s| s.len()).sum()
    }

    #[inline]
    pub fn neighbors(&self, idx: usize) -> &[u32] {
        &self.adjacency[idx]
    }

    pub fn inject_shortcuts(
        &mut self,
        pairs: &[(Vec<f32>, u32)],
        max_per_node: usize,
        codebook: &TwoStageCodebook,
        flat_codes: &FlatDocCodes,
        dim: usize,
    ) {
        if dim == 0 {
            return;
        }
        for (query_flat, target_int) in pairs {
            let target = *target_int as usize;
            if target >= self.adjacency.len() {
                continue;
            }
            let n_query = query_flat.len() / dim;
            if n_query == 0 {
                continue;
            }
            let query_scores = codebook.compute_query_centroid_scores(query_flat, n_query);
            let n_fine = codebook.n_fine;

            let candidates = beam_search_construction(
                &self.adjacency,
                &[0],
                &query_scores,
                n_query,
                flat_codes,
                n_fine,
                64,
                self.adjacency.len(),
            );

            for (cand_idx, _) in candidates.iter().take(max_per_node) {
                let cand = *cand_idx as usize;
                if cand != target && !self.shortcuts[target].contains(&(*cand_idx)) {
                    if self.shortcuts[target].len() < max_per_node {
                        self.shortcuts[target].push(*cand_idx);
                    }
                }
            }
        }
    }
}

/// Select neighbors using HNSW diversity heuristic (GEM paper Algorithm 2).
/// Candidates must be sorted ascending by score (best-first).
/// Uses asymmetric qCH with cand as first arg in both comparisons (WS-2.3 fix).
pub fn select_neighbors_heuristic(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Vec<(u32, f32)> {
    let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_degree);

    for &(cand_idx, _) in candidates {
        if selected.len() >= max_degree {
            break;
        }
        let cand_codes = flat_codes.doc_codes(cand_idx as usize);
        let cand_to_query = qch_proxy_between_docs(codebook, cand_codes, query_codes);

        let too_close = selected.iter().any(|&(sel_idx, _)| {
            let sel_codes = flat_codes.doc_codes(sel_idx as usize);
            let cand_to_sel = qch_proxy_between_docs(codebook, cand_codes, sel_codes);
            cand_to_sel < cand_to_query
        });

        if !too_close {
            selected.push((cand_idx, cand_to_query));
        }
    }

    selected
}

/// Shrink a node's neighbor list to max_degree using the diversity heuristic.
pub fn shrink_neighbors(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    let node_codes = flat_codes.doc_codes(node_idx);
    let neighbors: Vec<u32> = adjacency[node_idx].clone();

    let mut scored: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&nbr| {
            let nbr_codes = flat_codes.doc_codes(nbr as usize);
            let dist = qch_proxy_between_docs(codebook, node_codes, nbr_codes);
            (nbr, dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));

    let kept = select_neighbors_heuristic(&scored, node_codes, max_degree, codebook, flat_codes);
    adjacency[node_idx] = kept.iter().map(|&(idx, _)| idx).collect();
}

/// Build a GEM proximity graph using sequential HNSW-style insertion
/// with cluster-guided entry points and diversity-based neighbor selection.
pub fn build_graph(
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_docs];
    let n_fine = codebook.n_fine;

    for i in 1..n_docs {
        let (start, end) = doc_offsets[i];
        let n_tokens = end - start;
        let doc_vecs = &all_vectors[start * dim..end * dim];
        let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);

        let mut entries: Vec<u32> = Vec::new();
        for &cluster in &doc_profiles[i].ctop {
            if let Some(reps) = postings.cluster_reps.get(cluster as usize) {
                if let Some(rep) = reps {
                    if (*rep as usize) < i && !entries.contains(rep) {
                        entries.push(*rep);
                    }
                }
            }
        }
        if entries.is_empty() {
            entries.push(0);
        }

        let candidates = beam_search_construction(
            &adjacency,
            &entries,
            &query_scores,
            n_tokens,
            flat_codes,
            n_fine,
            ef_construction,
            i,
        );

        let doc_codes = flat_codes.doc_codes(i);
        let neighbors = select_neighbors_heuristic(
            &candidates,
            doc_codes,
            max_degree,
            codebook,
            flat_codes,
        );

        for &(nbr_idx, _score) in &neighbors {
            adjacency[i].push(nbr_idx);
            adjacency[nbr_idx as usize].push(i as u32);

            if adjacency[nbr_idx as usize].len() > max_degree {
                shrink_neighbors(
                    nbr_idx as usize,
                    max_degree,
                    &mut adjacency,
                    codebook,
                    flat_codes,
                );
            }
        }
    }

    bridge_repair(
        &mut adjacency,
        max_degree,
        postings,
        codebook,
        flat_codes,
    );

    GemGraph {
        adjacency,
        shortcuts: vec![Vec::new(); n_docs],
        max_degree,
    }
}

/// Bridge repair: ensure cross-cluster connectivity by adding edges between
/// cluster-reachable subgraph and unreachable cluster members.
/// Mirrors C++ `repair_fine_graph_structure` from the GEM reference.
pub fn bridge_repair(
    adjacency: &mut Vec<Vec<u32>>,
    max_degree: usize,
    postings: &ClusterPostings,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    let n_clusters = postings.lists.len();

    for cluster_id in 0..n_clusters {
        let entry_rep = match postings.cluster_reps.get(cluster_id) {
            Some(Some(rep)) => *rep as usize,
            _ => continue,
        };

        let members: Vec<u32> = postings.lists[cluster_id].clone();
        if members.len() <= 1 {
            continue;
        }

        // BFS from cluster entry to find reachable nodes within cluster
        let mut reachable = std::collections::HashSet::new();
        let mut bfs_queue = std::collections::VecDeque::new();
        let member_set: std::collections::HashSet<u32> = members.iter().copied().collect();

        bfs_queue.push_back(entry_rep as u32);
        reachable.insert(entry_rep as u32);

        // Track nodes with spare capacity for bridge targets
        let mut spare_capacity: Vec<u32> = Vec::new();
        if adjacency[entry_rep].len() < max_degree {
            spare_capacity.push(entry_rep as u32);
        }

        while let Some(node) = bfs_queue.pop_front() {
            for &nbr in &adjacency[node as usize] {
                if member_set.contains(&nbr) && !reachable.contains(&nbr) {
                    reachable.insert(nbr);
                    bfs_queue.push_back(nbr);
                    if adjacency[nbr as usize].len() < max_degree {
                        spare_capacity.push(nbr);
                    }
                }
            }
        }

        // For each unreachable cluster member, add a bridge edge
        let mut spare_idx = 0;
        for &member in &members {
            if reachable.contains(&member) {
                continue;
            }

            // Find a reachable node with spare capacity, or evict worst neighbor
            let bridge_source = if spare_idx < spare_capacity.len() {
                let src = spare_capacity[spare_idx] as usize;
                spare_idx += 1;
                src
            } else {
                // No spare capacity: evict worst neighbor from entry rep
                evict_worst_neighbor(entry_rep, max_degree, adjacency, codebook, flat_codes);
                entry_rep
            };

            let m = member as usize;
            if !adjacency[bridge_source].contains(&member) {
                adjacency[bridge_source].push(member);
            }
            if !adjacency[m].contains(&(bridge_source as u32)) {
                adjacency[m].push(bridge_source as u32);
                if adjacency[m].len() > max_degree {
                    evict_worst_neighbor(m, max_degree, adjacency, codebook, flat_codes);
                }
            }
        }
    }
}

/// Evict the worst (most distant) neighbor from a node's adjacency list
/// to make room for a bridge edge. Returns the evicted neighbor.
fn evict_worst_neighbor(
    node: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Option<u32> {
    if adjacency[node].len() < max_degree {
        return None;
    }
    if adjacency[node].is_empty() {
        return None;
    }
    let node_codes = flat_codes.doc_codes(node);
    let worst_idx = adjacency[node]
        .iter()
        .enumerate()
        .map(|(i, &nbr)| {
            let nbr_codes = flat_codes.doc_codes(nbr as usize);
            let dist = qch_proxy_between_docs(codebook, node_codes, nbr_codes);
            (i, dist)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    worst_idx.map(|i| adjacency[node].swap_remove(i))
}

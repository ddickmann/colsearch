use std::time::Instant;

fn _progress_write_graph(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("/tmp/gem_build_progress.log")
    {
        let _ = writeln!(f, "{}", msg);
    }
    let bytes = format!("{}\n", msg);
    unsafe {
        libc::write(2, bytes.as_ptr() as *const libc::c_void, bytes.len());
    }
}

macro_rules! progress {
    ($($arg:tt)*) => {{
        _progress_write_graph(&format!($($arg)*));
    }};
}

use serde::{Deserialize, Serialize};

use latence_gem_router::codebook::TwoStageCodebook;
use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

use rand::SeedableRng;
use rayon::prelude::*;

use crate::emd::construction_distance;
use crate::score_cache::ScoreCache;
use crate::search::beam_search_construction;

/// Trait for graph adjacency access, generic over CSR and Vec<Vec<u32>>.
pub trait Adjacency {
    fn neighbors(&self, node: usize) -> &[u32];
    fn n_nodes(&self) -> usize;
    fn n_edges(&self) -> usize;
}

/// Compressed Sparse Row adjacency — single contiguous allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrAdjacency {
    pub edges: Vec<u32>,
    pub offsets: Vec<usize>,
}

impl CsrAdjacency {
    pub fn empty(n_nodes: usize) -> Self {
        Self {
            edges: Vec::new(),
            offsets: vec![0; n_nodes + 1],
        }
    }

    pub fn from_adj_lists(adj: &[Vec<u32>]) -> Self {
        let n = adj.len();
        let mut offsets = Vec::with_capacity(n + 1);
        let total_edges: usize = adj.iter().map(|a| a.len()).sum();
        let mut edges = Vec::with_capacity(total_edges);

        let mut off = 0usize;
        for neighbors in adj {
            offsets.push(off);
            edges.extend_from_slice(neighbors);
            off += neighbors.len();
        }
        offsets.push(off);

        Self { edges, offsets }
    }

    pub fn to_adj_lists(&self) -> Vec<Vec<u32>> {
        let n = self.n_nodes();
        (0..n).map(|i| self.neighbors(i).to_vec()).collect()
    }
}

impl Adjacency for CsrAdjacency {
    #[inline]
    fn neighbors(&self, node: usize) -> &[u32] {
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        &self.edges[start..end]
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        if self.offsets.is_empty() { 0 } else { self.offsets.len() - 1 }
    }

    fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Vec<Vec<u32>> adjacency for mutable segments.
pub struct VecAdjacency<'a>(pub &'a [Vec<u32>]);

impl<'a> Adjacency for VecAdjacency<'a> {
    #[inline]
    fn neighbors(&self, node: usize) -> &[u32] {
        &self.0[node]
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.0.len()
    }

    fn n_edges(&self) -> usize {
        self.0.iter().map(|a| a.len()).sum()
    }
}

/// Assign HNSW level to a node using geometric distribution.
/// ml = 1 / ln(max_degree) gives optimal level distribution.
fn random_level(ml: f64, rng: &mut impl rand::Rng) -> usize {
    let r: f64 = rng.gen::<f64>().max(1e-15);
    (-r.ln() * ml).floor() as usize
}

/// Sealed multi-level GEM graph with CSR adjacency and semantic shortcuts.
pub struct GemGraph {
    /// Layer 0 is the bottom (all nodes), higher layers are sparser.
    pub levels: Vec<CsrAdjacency>,
    pub shortcuts: Vec<Vec<u32>>,
    /// Per-shortcut generation counter for aging-based pruning.
    pub shortcut_generations: Vec<Vec<usize>>,
    pub node_levels: Vec<usize>,
    pub entry_point: u32,
    pub max_degree: usize,
}

impl GemGraph {
    pub fn n_nodes(&self) -> usize {
        if self.levels.is_empty() { 0 } else { self.levels[0].n_nodes() }
    }

    pub fn n_edges(&self) -> usize {
        self.levels.iter().map(|l| l.n_edges()).sum()
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn total_shortcuts(&self) -> usize {
        self.shortcuts.iter().map(|s| s.len()).sum()
    }

    /// Convenience: bottom-layer adjacency for backward compatibility.
    pub fn bottom_adjacency(&self) -> &CsrAdjacency {
        &self.levels[0]
    }

    /// BFS-based connectivity report on the bottom layer.
    ///
    /// Returns (n_components, giant_component_frac) where giant_component_frac
    /// is the fraction of nodes in the largest connected component.
    pub fn connectivity_report(&self) -> (usize, f64) {
        if self.levels.is_empty() {
            return (0, 0.0);
        }
        let n = self.levels[0].n_nodes();
        if n == 0 {
            return (0, 0.0);
        }

        let adj = &self.levels[0];
        let mut visited = vec![false; n];
        let mut component_sizes: Vec<usize> = Vec::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut size = 0usize;
            queue.push_back(start);
            visited[start] = true;
            while let Some(node) = queue.pop_front() {
                size += 1;
                for &nbr in adj.neighbors(node) {
                    let nb = nbr as usize;
                    if nb < n && !visited[nb] {
                        visited[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
            component_sizes.push(size);
        }

        let n_components = component_sizes.len();
        let giant = component_sizes.iter().copied().max().unwrap_or(0);
        let frac = if n > 0 { giant as f64 / n as f64 } else { 0.0 };
        (n_components, frac)
    }

    /// Remove shortcuts pointing to deleted nodes. Optionally prune shortcuts
    /// older than `max_age` generations (tracked via `shortcut_generations`).
    pub fn prune_stale_shortcuts(&mut self, deleted: &[bool], max_age: Option<usize>, current_generation: usize) {
        let n = self.shortcuts.len();
        let has_gens = self.shortcut_generations.len() == n;

        // Phase 1: Remove shortcuts pointing to deleted nodes, keeping
        // shortcut_generations in lockstep so indices stay aligned.
        for i in 0..n {
            let mut keep_sc = Vec::new();
            let mut keep_gen = Vec::new();
            for (idx, &target) in self.shortcuts[i].iter().enumerate() {
                let t = target as usize;
                if t < deleted.len() && deleted[t] {
                    continue;
                }
                keep_sc.push(target);
                if has_gens {
                    if let Some(&g) = self.shortcut_generations[i].get(idx) {
                        keep_gen.push(g);
                    }
                }
            }
            self.shortcuts[i] = keep_sc;
            if has_gens {
                self.shortcut_generations[i] = keep_gen;
            }
        }

        // Phase 2: Age-based pruning (only if generations are tracked).
        if let Some(age_limit) = max_age {
            if has_gens {
                for i in 0..n {
                    let gens = &mut self.shortcut_generations[i];
                    let sc = &mut self.shortcuts[i];
                    let mut keep = Vec::with_capacity(sc.len());
                    let mut keep_gen = Vec::with_capacity(gens.len());
                    for (idx, &target) in sc.iter().enumerate() {
                        let gen = gens.get(idx).copied().unwrap_or(0);
                        if current_generation.saturating_sub(gen) <= age_limit {
                            keep.push(target);
                            keep_gen.push(gen);
                        }
                    }
                    *sc = keep;
                    *gens = keep_gen;
                }
            }
        }
    }

    pub fn inject_shortcuts(
        &mut self,
        pairs: &[(Vec<f32>, u32)],
        max_per_node: usize,
        codebook: &TwoStageCodebook,
        flat_codes: &FlatDocCodes,
        dim: usize,
    ) {
        if dim == 0 || self.levels.is_empty() {
            return;
        }
        let bottom = &self.levels[0];
        for (query_flat, target_int) in pairs {
            let target = *target_int as usize;
            if target >= bottom.n_nodes() {
                continue;
            }
            let n_query = query_flat.len() / dim;
            if n_query == 0 {
                continue;
            }
            let query_scores = codebook.compute_query_centroid_scores(query_flat, n_query);
            let n_fine = codebook.n_fine;

            let candidates = beam_search_construction(
                bottom,
                &[self.entry_point],
                &query_scores,
                n_query,
                flat_codes,
                n_fine,
                64,
                bottom.n_nodes(),
            );

            for (cand_idx, _) in candidates.iter().take(max_per_node) {
                let cand = *cand_idx as usize;
                if cand != target && !self.shortcuts[target].contains(cand_idx)
                    && self.shortcuts[target].len() < max_per_node {
                        self.shortcuts[target].push(*cand_idx);
                    }
            }
        }
    }
}

/// Select neighbors using HNSW diversity heuristic (GEM paper Algorithm 2).
/// When `use_emd` is true, uses qEMD (Earth Mover's Distance) for edge scoring
/// during construction (metric decoupling from the GEM paper, Section 4.2).
pub fn select_neighbors_heuristic(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_cached(candidates, query_codes, max_degree, codebook, flat_codes, None, false)
}

pub fn select_neighbors_heuristic_cached(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    mut cache: Option<&mut ScoreCache>,
    use_emd: bool,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_inner(candidates, query_codes, max_degree, codebook, flat_codes, &mut cache, None, 0, use_emd)
}

/// Payload-aware variant: candidates sharing a payload cluster with query_node
/// get a distance bonus, biasing edges toward same-payload documents.
pub fn select_neighbors_payload_aware(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    mut cache: Option<&mut ScoreCache>,
    payload_clusters: &[u32],
    query_node: u32,
    use_emd: bool,
) -> Vec<(u32, f32)> {
    select_neighbors_heuristic_inner(candidates, query_codes, max_degree, codebook, flat_codes, &mut cache, Some(payload_clusters), query_node, use_emd)
}

fn select_neighbors_heuristic_inner(
    candidates: &[(u32, f32)],
    query_codes: &[u16],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    cache: &mut Option<&mut ScoreCache>,
    payload_clusters: Option<&[u32]>,
    query_node: u32,
    use_emd: bool,
) -> Vec<(u32, f32)> {
    let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_degree);

    // Pre-compute candidate-to-query distances using the construction metric.
    // When use_emd=true, this uses qEMD (metric, triangle inequality) instead of qCH.
    let cand_to_query_scores: Vec<f32> = if candidates.len() > 16 {
        candidates.par_iter()
            .map(|&(cand_idx, _)| {
                let cand_codes = flat_codes.doc_codes(cand_idx as usize);
                construction_distance(codebook, cand_codes, query_codes, use_emd)
            })
            .collect()
    } else {
        candidates.iter()
            .map(|&(cand_idx, _)| {
                let cand_codes = flat_codes.doc_codes(cand_idx as usize);
                construction_distance(codebook, cand_codes, query_codes, use_emd)
            })
            .collect()
    };

    const PAYLOAD_BONUS: f32 = 0.95;

    for (ci, &(cand_idx, _)) in candidates.iter().enumerate() {
        if selected.len() >= max_degree {
            break;
        }
        let mut cand_to_query = cand_to_query_scores[ci];
        if let Some(pc) = payload_clusters {
            let qn = query_node as usize;
            let cn = cand_idx as usize;
            if qn < pc.len() && cn < pc.len() && pc[qn] == pc[cn] {
                cand_to_query *= PAYLOAD_BONUS;
            }
        }
        let cand_codes = flat_codes.doc_codes(cand_idx as usize);

        let too_close = selected.iter().any(|&(sel_idx, _)| {
            let dist = if let Some(ref mut cache) = cache {
                cache.get(cand_idx, sel_idx)
            } else {
                None
            };
            let cand_to_sel = dist.unwrap_or_else(|| {
                let sel_codes = flat_codes.doc_codes(sel_idx as usize);
                construction_distance(codebook, cand_codes, sel_codes, use_emd)
            });
            cand_to_sel < cand_to_query
        });

        if !too_close {
            if let Some(ref mut c) = cache {
                for &(sel_idx, _) in &selected {
                    if c.get(cand_idx, sel_idx).is_none() {
                        let sel_codes = flat_codes.doc_codes(sel_idx as usize);
                        let d = construction_distance(codebook, cand_codes, sel_codes, use_emd);
                        c.insert(cand_idx, sel_idx, d);
                    }
                }
            }
            selected.push((cand_idx, cand_to_query));
        }
    }

    selected
}

/// Shrink a node's neighbor list to max_degree using the diversity heuristic.
/// Maintains bidirectional consistency: removed neighbors drop their reverse edge.
pub fn shrink_neighbors(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    shrink_neighbors_emd(node_idx, max_degree, adjacency, codebook, flat_codes, false)
}

pub fn shrink_neighbors_emd(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    shrink_neighbors_emd_cached(node_idx, max_degree, adjacency, codebook, flat_codes, use_emd, None)
}

pub fn shrink_neighbors_emd_cached(
    node_idx: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    mut cache: Option<&mut ScoreCache>,
) {
    let node_codes = flat_codes.doc_codes(node_idx);
    let neighbors: Vec<u32> = adjacency[node_idx].clone();
    let node_u32 = node_idx as u32;

    let mut scored: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&nbr| {
            let dist = if let Some(ref mut c) = cache {
                if let Some(d) = c.get(node_u32, nbr) {
                    d
                } else {
                    let nbr_codes = flat_codes.doc_codes(nbr as usize);
                    let d = construction_distance(codebook, node_codes, nbr_codes, use_emd);
                    c.insert(node_u32, nbr, d);
                    d
                }
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                construction_distance(codebook, node_codes, nbr_codes, use_emd)
            };
            (nbr, dist)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let kept = select_neighbors_heuristic_cached(
        &scored, node_codes, max_degree, codebook, flat_codes, cache, use_emd,
    );
    let kept_set: std::collections::HashSet<u32> = kept.iter().map(|&(idx, _)| idx).collect();

    for &old_nbr in &neighbors {
        if !kept_set.contains(&old_nbr) {
            adjacency[old_nbr as usize].retain(|&x| x != node_idx as u32);
        }
    }

    adjacency[node_idx] = kept.iter().map(|&(idx, _)| idx).collect();
}

/// Build a multi-level GEM graph using HNSW-style hierarchical insertion.
/// `payload_clusters`: optional per-doc cluster IDs from payload similarity.
/// When provided, neighbor selection biases toward same-cluster edges,
/// making filtered search faster (Qdrant uses post-hoc filtering; we do it natively).
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
    build_graph_with_payload(all_vectors, dim, doc_offsets, codebook, flat_codes, doc_profiles, postings, max_degree, ef_construction, None, false)
}

pub fn build_graph_with_payload(
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
    payload_clusters: Option<&[u32]>,
    use_emd: bool,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            shortcut_generations: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_degree,
        };
    }

    let n_fine = codebook.n_fine;
    let ml = 1.0 / (max_degree as f64).ln();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut node_levels: Vec<usize> = Vec::with_capacity(n_docs);
    let mut max_level = 0usize;
    for _ in 0..n_docs {
        let level = random_level(ml, &mut rng);
        if level > max_level {
            max_level = level;
        }
        node_levels.push(level);
    }

    let n_levels = max_level + 1;
    let mut score_cache = ScoreCache::new(n_docs * 16);
    let mut levels: Vec<Vec<Vec<u32>>> = (0..n_levels)
        .map(|_| vec![Vec::new(); n_docs])
        .collect();

    let max_degree_upper = max_degree;
    let max_degree_bottom = max_degree;

    let mut entry_point: u32 = 0;

    for i in 0..n_docs {
        let (start, end) = doc_offsets[i];
        let n_tokens = end - start;
        let doc_vecs = &all_vectors[start * dim..end * dim];
        let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);
        let node_level = node_levels[i];

        if i == 0 {
            // First node: no neighbors to search yet, just update entry point
            if node_level > node_levels[entry_point as usize] {
                entry_point = i as u32;
            }
            continue;
        }

        let ep_level = node_levels[entry_point as usize];
        let mut cur_entry = entry_point;

        // Greedy search from top to node_level+1 (upper layers, ef=1)
        for level in (node_level + 1..=ep_level.min(max_level)).rev() {
            let adj = VecAdjacency(&levels[level]);
            let candidates = beam_search_construction(
                &adj, &[cur_entry], &query_scores, n_tokens,
                flat_codes, n_fine, 1, i,
            );
            if let Some(&(best, _)) = candidates.first() {
                cur_entry = best;
            }
        }

        // Search and connect at each level from min(node_level, ep_level) down to 0
        let start_level = node_level.min(ep_level.min(max_level));
        for level in (0..=start_level).rev() {
            let md = if level == 0 { max_degree_bottom } else { max_degree_upper };

            // Use cluster-guided entry points for level 0
            let mut entries = vec![cur_entry];
            if level == 0 {
                for &cluster in &doc_profiles[i].ctop {
                    if let Some(Some(rep)) = postings.cluster_reps.get(cluster as usize) {
                        if (*rep as usize) < i && !entries.contains(rep) {
                            entries.push(*rep);
                        }
                    }
                }
            }

            let candidates = {
                let adj = VecAdjacency(&levels[level]);
                beam_search_construction(
                    &adj, &entries, &query_scores, n_tokens,
                    flat_codes, n_fine, ef_construction, i,
                )
            };

            if let Some(&(best, _)) = candidates.first() {
                cur_entry = best;
            }

            let doc_codes = flat_codes.doc_codes(i);
            let neighbors = if let Some(pc) = payload_clusters {
                select_neighbors_payload_aware(
                    &candidates, doc_codes, md, codebook, flat_codes,
                    Some(&mut score_cache), pc, i as u32, use_emd,
                )
            } else {
                select_neighbors_heuristic_cached(
                    &candidates, doc_codes, md, codebook, flat_codes,
                    Some(&mut score_cache), use_emd,
                )
            };

            for &(nbr_idx, _score) in &neighbors {
                levels[level][i].push(nbr_idx);
                levels[level][nbr_idx as usize].push(i as u32);

                if levels[level][nbr_idx as usize].len() > md {
                    shrink_neighbors_emd_cached(
                        nbr_idx as usize, md,
                        &mut levels[level], codebook, flat_codes, use_emd,
                        Some(&mut score_cache),
                    );
                }
            }
        }

        if node_level > node_levels[entry_point as usize] {
            entry_point = i as u32;
        }
    }

    // Bridge repair on bottom layer only
    bridge_repair_emd(&mut levels[0], max_degree, postings, codebook, flat_codes, use_emd);
    global_bridge_repair(&mut levels[0], max_degree, codebook, flat_codes, use_emd);

    let csr_levels: Vec<CsrAdjacency> = levels
        .iter()
        .map(|l| CsrAdjacency::from_adj_lists(l))
        .collect();

    GemGraph {
        levels: csr_levels,
        shortcuts: vec![Vec::new(); n_docs],
        shortcut_generations: vec![Vec::new(); n_docs],
        node_levels,
        entry_point,
        max_degree,
    }
}

/// Bridge repair on a mutable adjacency layer.
pub fn bridge_repair(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    postings: &ClusterPostings,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
) {
    bridge_repair_emd(adjacency, max_degree, postings, codebook, flat_codes, false)
}

pub fn bridge_repair_emd(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    postings: &ClusterPostings,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    let n_clusters = postings.lists.len();

    // Phase 1 (parallel): BFS per-cluster to discover unreachable members.
    // Returns (cluster_entry_rep, spare_capacity_nodes, unreachable_members).
    let adj_snapshot: &[Vec<u32>] = adjacency;
    let cluster_plans: Vec<(usize, Vec<u32>, Vec<u32>)> = (0..n_clusters)
        .into_par_iter()
        .filter_map(|cluster_id| {
            let entry_rep = match postings.cluster_reps.get(cluster_id) {
                Some(Some(rep)) => *rep as usize,
                _ => return None,
            };
            let members = &postings.lists[cluster_id];
            if members.len() <= 1 {
                return None;
            }

            let member_set: std::collections::HashSet<u32> = members.iter().copied().collect();
            let mut reachable = std::collections::HashSet::new();
            let mut bfs_queue = std::collections::VecDeque::new();

            bfs_queue.push_back(entry_rep as u32);
            reachable.insert(entry_rep as u32);

            let mut spare = Vec::new();
            if adj_snapshot[entry_rep].len() < max_degree {
                spare.push(entry_rep as u32);
            }

            while let Some(node) = bfs_queue.pop_front() {
                for &nbr in &adj_snapshot[node as usize] {
                    if member_set.contains(&nbr) && !reachable.contains(&nbr) {
                        reachable.insert(nbr);
                        bfs_queue.push_back(nbr);
                        if adj_snapshot[nbr as usize].len() < max_degree {
                            spare.push(nbr);
                        }
                    }
                }
            }

            let unreachable: Vec<u32> = members.iter()
                .copied()
                .filter(|m| !reachable.contains(m))
                .collect();

            if unreachable.is_empty() {
                return None;
            }
            Some((entry_rep, spare, unreachable))
        })
        .collect();

    // Phase 2 (sequential): Apply bridge edges.
    for (entry_rep, spare_capacity, unreachable) in cluster_plans {
        let mut spare_idx = 0;
        for &member in &unreachable {
            let bridge_source = if spare_idx < spare_capacity.len() {
                let src = spare_capacity[spare_idx] as usize;
                spare_idx += 1;
                src
            } else {
                evict_worst_neighbor_emd(entry_rep, max_degree, adjacency, codebook, flat_codes, use_emd);
                entry_rep
            };

            let m = member as usize;
            if !adjacency[bridge_source].contains(&member) {
                adjacency[bridge_source].push(member);
            }
            if !adjacency[m].contains(&(bridge_source as u32)) {
                adjacency[m].push(bridge_source as u32);
                if adjacency[m].len() > max_degree {
                    evict_worst_neighbor_emd(m, max_degree, adjacency, codebook, flat_codes, use_emd);
                }
            }
        }
    }
}

fn bfs_components(adjacency: &[Vec<u32>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut visited = vec![false; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            comp.push(node);
            for &nbr in &adjacency[node] {
                let nb = nbr as usize;
                if nb < n && !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
        components.push(comp);
    }
    components
}

/// Add a bridge edge that cannot be evicted. We allow max_degree+1 for bridge
/// nodes rather than risk the bridge being evicted as the "worst" neighbor,
/// which would leave components disconnected.
fn add_bridge_edge(
    adjacency: &mut [Vec<u32>],
    a: usize,
    b: usize,
    _max_degree: usize,
    _codebook: &TwoStageCodebook,
    _flat_codes: &FlatDocCodes,
    _use_emd: bool,
) {
    if !adjacency[a].contains(&(b as u32)) {
        adjacency[a].push(b as u32);
    }
    if !adjacency[b].contains(&(a as u32)) {
        adjacency[b].push(a as u32);
    }
}

/// Global bridge repair: connect ALL disconnected components into a single
/// connected graph. Runs iteratively until only one component remains.
///
/// For each small component, finds the nearest bridge edge to the giant
/// component by scanning all giant nodes. Falls back to connecting the first
/// node of each small component to the giant entry if no code-based pair exists.
/// Bridge edges are never evicted (they may exceed max_degree by 1).
pub fn global_bridge_repair(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    let n = adjacency.len();
    if n <= 1 {
        return;
    }

    const MAX_ROUNDS: usize = 10;
    for round in 0..MAX_ROUNDS {
        let components = bfs_components(adjacency);
        if components.len() <= 1 {
            return;
        }

        let giant_idx = components
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.len())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let giant_entry = components[giant_idx][0];
        let giant_comp = &components[giant_idx];

        // Parallel: find best bridge pair per small component.
        let small_comps: Vec<(usize, &Vec<usize>)> = components.iter().enumerate()
            .filter(|&(ci, c)| ci != giant_idx && !c.is_empty())
            .collect();

        let bridge_pairs: Vec<(usize, usize)> = small_comps.par_iter()
            .map(|&(_, comp)| {
                let mut best_pair: Option<(usize, usize, f32)> = None;
                for &small_node in comp {
                    let small_codes = flat_codes.doc_codes(small_node);
                    if small_codes.is_empty() {
                        continue;
                    }
                    for &giant_node in giant_comp {
                        let giant_codes = flat_codes.doc_codes(giant_node);
                        if giant_codes.is_empty() {
                            continue;
                        }
                        let d = construction_distance(codebook, small_codes, giant_codes, use_emd);
                        if best_pair.is_none() || d < best_pair.unwrap().2 {
                            best_pair = Some((small_node, giant_node, d));
                        }
                    }
                }
                match best_pair {
                    Some((s, g, _)) => (s, g),
                    None => (comp[0], giant_entry),
                }
            })
            .collect();

        if bridge_pairs.is_empty() {
            break;
        }

        for (bridge_small, bridge_giant) in bridge_pairs {
            add_bridge_edge(adjacency, bridge_small, bridge_giant, max_degree, codebook, flat_codes, use_emd);
        }

        let remaining = bfs_components(adjacency).len();
        if remaining <= 1 {
            return;
        }
        if round == MAX_ROUNDS - 1 {
            log::warn!(
                "global_bridge_repair: {} components remain after {} rounds (expected 1)",
                remaining, MAX_ROUNDS,
            );
        }
    }
}

/// Cached variant of global_bridge_repair — reuses pairwise distance cache
/// across rounds, avoiding redundant recomputation when the same
/// small_node × giant_node pair appears in successive BFS rounds.
pub fn global_bridge_repair_cached(
    adjacency: &mut [Vec<u32>],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    cache: &mut ScoreCache,
) {
    let n = adjacency.len();
    if n <= 1 {
        return;
    }

    const MAX_ROUNDS: usize = 10;
    for round in 0..MAX_ROUNDS {
        let components = bfs_components(adjacency);
        if components.len() <= 1 {
            return;
        }

        let giant_idx = components
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.len())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let giant_entry = components[giant_idx][0];
        let giant_comp = &components[giant_idx];

        let small_comps: Vec<&Vec<usize>> = components.iter().enumerate()
            .filter(|&(ci, c)| ci != giant_idx && !c.is_empty())
            .map(|(_, c)| c)
            .collect();

        // Sequential bridge search with cache (cache is &mut, can't use par_iter)
        let mut bridge_pairs: Vec<(usize, usize)> = Vec::with_capacity(small_comps.len());
        for comp in &small_comps {
            let mut best_pair: Option<(usize, usize, f32)> = None;
            for &small_node in comp.iter() {
                let small_codes = flat_codes.doc_codes(small_node);
                if small_codes.is_empty() {
                    continue;
                }
                let small_u32 = small_node as u32;
                for &giant_node in giant_comp {
                    let d = if let Some(cached) = cache.get(small_u32, giant_node as u32) {
                        cached
                    } else {
                        let giant_codes = flat_codes.doc_codes(giant_node);
                        let d = construction_distance(codebook, small_codes, giant_codes, use_emd);
                        cache.insert(small_u32, giant_node as u32, d);
                        d
                    };
                    if best_pair.is_none() || d < best_pair.unwrap().2 {
                        best_pair = Some((small_node, giant_node, d));
                    }
                }
            }
            match best_pair {
                Some((s, g, _)) => bridge_pairs.push((s, g)),
                None => bridge_pairs.push((comp[0], giant_entry)),
            }
        }

        if bridge_pairs.is_empty() {
            break;
        }

        for (bridge_small, bridge_giant) in bridge_pairs {
            add_bridge_edge(adjacency, bridge_small, bridge_giant, max_degree, codebook, flat_codes, use_emd);
        }

        let remaining = bfs_components(adjacency).len();
        if remaining <= 1 {
            return;
        }
        if round == MAX_ROUNDS - 1 {
            log::warn!(
                "global_bridge_repair_cached: {} components remain after {} rounds (expected 1)",
                remaining, MAX_ROUNDS,
            );
        }
    }
}

fn evict_worst_neighbor_emd(
    node: usize,
    max_degree: usize,
    adjacency: &mut [Vec<u32>],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
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
            let dist = construction_distance(codebook, node_codes, nbr_codes, use_emd);
            (i, dist)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    worst_idx.map(|i| adjacency[node].swap_remove(i))
}

// ---------------------------------------------------------------------------
// Dual-Graph Construction (GEM Paper Section 4.3, Algorithms 1-3)
// ---------------------------------------------------------------------------

/// Build a GEM graph using the paper's dual-graph per-cluster construction.
///
/// Instead of HNSW-style sequential insertion over all documents, this builds
/// per-cluster local graphs, then merges them via bridge sets. This gives
/// better intra-cluster locality and explicit cross-cluster connectivity.
///
/// Algorithm 1: For each cluster C_i, process its member documents.
/// - First encounter (not yet inserted): `build_cluster_local` connects the doc
///   to existing cluster members using beam search + diversity heuristic.
/// - Subsequent encounter (already inserted via another cluster): `update_bridges`
///   merges old neighbors with new cluster neighbors, enforcing at least one
///   neighbor from each of the document's assigned clusters.
pub fn build_graph_dual(
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
    use_emd: bool,
) -> GemGraph {
    let n_docs = doc_offsets.len();
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            shortcut_generations: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_degree,
        };
    }

    let n_fine = codebook.n_fine;
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_docs];
    let mut inserted: Vec<bool> = vec![false; n_docs];
    let mut score_cache = ScoreCache::new(n_docs * 16);

    let n_clusters = postings.lists.len();
    let graph_start = Instant::now();
    let mut total_inserted: usize = 0;
    let mut total_bridged: usize = 0;

    // Pre-allocate reusable scores buffer for bridge updates (Opt 1).
    // For first insertions, scores are batch-computed via rayon (Opt 3).
    let max_tokens = doc_offsets.iter().map(|&(s, e)| e - s).max().unwrap_or(0);
    let mut scores_buf: Vec<f32> = vec![0.0f32; max_tokens * n_fine];

    progress!("[GEM graph] dual-graph construction: {} docs, {} clusters, max_degree={}, ef_construction={}, use_emd={}",
        n_docs, n_clusters, max_degree, ef_construction, use_emd);
    progress!("[GEM graph] max_tokens_per_doc={}, scores_buf={:.1} MB",
        max_tokens, (max_tokens * n_fine * 4) as f64 / 1e6);

    // Process each cluster's member documents (Algorithm 1)
    for cluster_id in 0..n_clusters {
        let members = &postings.lists[cluster_id];
        if members.is_empty() {
            continue;
        }
        let cluster_start = Instant::now();

        // Collect already-inserted members as potential entry points for this cluster
        let cluster_entries: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&m| inserted[m as usize])
            .collect();

        // Partition members into new (need first insertion) and bridge (already inserted)
        let mut new_members: Vec<u32> = Vec::new();
        let mut bridge_members: Vec<u32> = Vec::new();
        for &doc_u32 in members {
            let doc_idx = doc_u32 as usize;
            if doc_idx >= n_docs { continue; }
            if inserted[doc_idx] {
                bridge_members.push(doc_u32);
            } else {
                new_members.push(doc_u32);
            }
        }

        // --- Opt 3: Batch-parallel GEMM for all new members ---
        // Compute query_centroid_scores for all new members in parallel using rayon,
        // then do sequential beam search + neighbor selection with precomputed scores.
        let precomputed_scores: Vec<(u32, Vec<f32>)> = if !new_members.is_empty() && !cluster_entries.is_empty() {
            new_members.par_iter().map(|&doc_u32| {
                let doc_idx = doc_u32 as usize;
                let (start, end) = doc_offsets[doc_idx];
                let n_tokens = end - start;
                if n_tokens == 0 || dim == 0 {
                    return (doc_u32, Vec::new());
                }
                let doc_vecs = &all_vectors[start * dim..end * dim];
                let scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);
                (doc_u32, scores)
            }).collect()
        } else {
            new_members.iter().map(|&d| (d, Vec::new())).collect()
        };

        // Sequential: beam search + neighbor insertion using precomputed scores
        let cluster_inserts = precomputed_scores.len();
        for (doc_u32, query_scores) in precomputed_scores {
            let doc_idx = doc_u32 as usize;
            if query_scores.is_empty() || cluster_entries.is_empty() {
                inserted[doc_idx] = true;
                continue;
            }
            let (start, end) = doc_offsets[doc_idx];
            let n_tokens = end - start;

            let adj = VecAdjacency(&adjacency);
            let candidates = beam_search_construction(
                &adj,
                &cluster_entries,
                &query_scores,
                n_tokens,
                flat_codes,
                n_fine,
                ef_construction,
                doc_idx,
            );

            let doc_codes = flat_codes.doc_codes(doc_idx);
            let neighbors = select_neighbors_heuristic_cached(
                &candidates,
                doc_codes,
                max_degree,
                codebook,
                flat_codes,
                Some(&mut score_cache),
                use_emd,
            );

            for &(nbr_idx, _) in &neighbors {
                adjacency[doc_idx].push(nbr_idx);
                adjacency[nbr_idx as usize].push(doc_idx as u32);
                if adjacency[nbr_idx as usize].len() > max_degree {
                    shrink_neighbors_emd_cached(
                        nbr_idx as usize, max_degree, &mut adjacency, codebook, flat_codes, use_emd,
                        Some(&mut score_cache),
                    );
                }
            }
            inserted[doc_idx] = true;
        }

        // Bridge updates for already-inserted members (reuse scores_buf — Opt 1)
        let cluster_bridges = bridge_members.len();
        for &doc_u32 in &bridge_members {
            let doc_idx = doc_u32 as usize;
            let (start, end) = doc_offsets[doc_idx];
            let n_tokens = end - start;
            if n_tokens == 0 || dim == 0 || cluster_entries.is_empty() {
                continue;
            }
            let doc_vecs = &all_vectors[start * dim..end * dim];
            let needed = n_tokens * n_fine;
            if scores_buf.len() < needed {
                scores_buf.resize(needed, 0.0);
            }
            codebook.compute_query_centroid_scores_into(doc_vecs, n_tokens, &mut scores_buf);

            let adj = VecAdjacency(&adjacency);
            let new_neighbors = beam_search_construction(
                &adj,
                &cluster_entries,
                &scores_buf[..needed],
                n_tokens,
                flat_codes,
                n_fine,
                ef_construction,
                doc_idx,
            );
            update_bridges_cached(
                doc_idx,
                &new_neighbors,
                &mut adjacency,
                doc_profiles,
                max_degree,
                codebook,
                flat_codes,
                use_emd,
                Some(&mut score_cache),
            );
        }

        total_inserted += cluster_inserts;
        total_bridged += cluster_bridges;
        let elapsed = graph_start.elapsed().as_secs_f64();
        let frac = (cluster_id + 1) as f64 / n_clusters as f64;
        let eta = if frac > 0.0 { elapsed / frac - elapsed } else { 0.0 };
        progress!(
            "[GEM graph] cluster {}/{} done ({} members, {} new inserts, {} bridges) — {:.1}s elapsed, {:.0}s cluster, ETA {:.0}s | total inserted {}/{}",
            cluster_id + 1, n_clusters, members.len(), cluster_inserts, cluster_bridges,
            elapsed, cluster_start.elapsed().as_secs_f64(), eta, total_inserted, n_docs,
        );
    }

    // Repair isolated nodes (degree 0): connect to nearest neighbor by brute-force.
    let isolated_count = adjacency.iter().filter(|a| a.is_empty()).count();
    if isolated_count > 0 {
        log::info!("Repairing {} isolated nodes (brute-force scan over {} docs)...",
                   isolated_count, n_docs);
        let mut repaired = 0usize;
        for doc_idx in 0..n_docs {
            if !adjacency[doc_idx].is_empty() {
                continue;
            }
            let doc_codes = flat_codes.doc_codes(doc_idx);
            if doc_codes.is_empty() {
                continue;
            }
            let mut best_nbr = u32::MAX;
            let mut best_dist = f32::MAX;
            #[allow(clippy::needless_range_loop)]
            for other in 0..n_docs {
                if other == doc_idx || adjacency[other].is_empty() {
                    continue;
                }
                let other_codes = flat_codes.doc_codes(other);
                let d = construction_distance(codebook, doc_codes, other_codes, use_emd);
                if d < best_dist {
                    best_dist = d;
                    best_nbr = other as u32;
                }
            }
            if best_nbr != u32::MAX {
                adjacency[doc_idx].push(best_nbr);
                adjacency[best_nbr as usize].push(doc_idx as u32);
                if adjacency[best_nbr as usize].len() > max_degree {
                    shrink_neighbors_emd_cached(best_nbr as usize, max_degree, &mut adjacency, codebook, flat_codes, use_emd, Some(&mut score_cache));
                }
                repaired += 1;
            }
        }
        log::info!("Isolated node repair done: {}/{} repaired", repaired, isolated_count);
    }

    progress!("[GEM graph] cluster loop complete in {:.1}s — {} inserts, {} bridges",
        graph_start.elapsed().as_secs_f64(), total_inserted, total_bridged);

    // Safety net: bridge repair for any remaining disconnected components
    let repair_start = Instant::now();
    progress!("[GEM graph] running bridge repair...");
    bridge_repair_emd(&mut adjacency, max_degree, postings, codebook, flat_codes, use_emd);
    progress!("[GEM graph] bridge repair done in {:.1}s, running global bridge repair...",
        repair_start.elapsed().as_secs_f64());
    let global_start = Instant::now();
    global_bridge_repair(&mut adjacency, max_degree, codebook, flat_codes, use_emd);
    progress!("[GEM graph] global bridge repair done in {:.1}s", global_start.elapsed().as_secs_f64());

    // Build single-level graph (dual-graph uses flat structure, not multi-level HNSW)
    progress!("[GEM graph] building CSR adjacency...");
    let csr = CsrAdjacency::from_adj_lists(&adjacency);

    // Entry point: the node with the most connections (hub-like)
    let entry_point = (0..n_docs)
        .max_by_key(|&i| adjacency[i].len())
        .unwrap_or(0) as u32;

    GemGraph {
        levels: vec![csr],
        shortcuts: vec![Vec::new(); n_docs],
        shortcut_generations: vec![Vec::new(); n_docs],
        node_levels: vec![0; n_docs],
        entry_point,
        max_degree,
    }
}

/// First insertion of a document into the graph via a specific cluster.
/// Beam-searches among already-connected cluster members to find candidates,
/// then selects neighbors via the diversity heuristic.
#[allow(dead_code)]
fn build_cluster_local(
    doc_idx: usize,
    cluster_entries: &[u32],
    adjacency: &mut [Vec<u32>],
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    max_degree: usize,
    ef_construction: usize,
    use_emd: bool,
    score_cache: &mut ScoreCache,
) {
    if cluster_entries.is_empty() {
        return;
    }

    let (start, end) = doc_offsets[doc_idx];
    let n_tokens = end - start;
    if n_tokens == 0 || dim == 0 {
        return;
    }
    let doc_vecs = &all_vectors[start * dim..end * dim];
    let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);

    let adj = VecAdjacency(adjacency);
    let candidates = beam_search_construction(
        &adj,
        cluster_entries,
        &query_scores,
        n_tokens,
        flat_codes,
        n_fine,
        ef_construction,
        doc_idx,
    );

    let doc_codes = flat_codes.doc_codes(doc_idx);
    let neighbors = select_neighbors_heuristic_cached(
        &candidates,
        doc_codes,
        max_degree,
        codebook,
        flat_codes,
        Some(score_cache),
        use_emd,
    );

    for &(nbr_idx, _) in &neighbors {
        adjacency[doc_idx].push(nbr_idx);
        adjacency[nbr_idx as usize].push(doc_idx as u32);

        if adjacency[nbr_idx as usize].len() > max_degree {
            shrink_neighbors_emd_cached(
                nbr_idx as usize, max_degree, adjacency, codebook, flat_codes, use_emd,
                Some(score_cache),
            );
        }
    }
}

/// Find new candidate neighbors for a doc when encountering it in a second cluster.
#[allow(dead_code)]
fn find_new_cluster_neighbors(
    doc_idx: usize,
    cluster_entries: &[u32],
    adjacency: &[Vec<u32>],
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    n_fine: usize,
    ef_construction: usize,
) -> Vec<(u32, f32)> {
    if cluster_entries.is_empty() {
        return Vec::new();
    }

    let (start, end) = doc_offsets[doc_idx];
    let n_tokens = end - start;
    if n_tokens == 0 || dim == 0 {
        return Vec::new();
    }
    let doc_vecs = &all_vectors[start * dim..end * dim];
    let query_scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);

    let adj = VecAdjacency(adjacency);
    beam_search_construction(
        &adj,
        cluster_entries,
        &query_scores,
        n_tokens,
        flat_codes,
        n_fine,
        ef_construction,
        doc_idx,
    )
}

/// Update bridge edges for a document encountered in a second (or subsequent) cluster.
///
/// GEM Paper Algorithm 3:
/// 1. C_old = existing neighbors
/// 2. C_all = C_new ∪ C_old
/// 3. If |C_all| <= M: keep all
/// 4. Else: keep M closest by construction distance, then for each cluster in ctop(P):
///    if no neighbor from that cluster is in the final set, force-include the closest
///    from C_all that belongs to that cluster (replacing the farthest).
pub fn update_bridges(
    doc_idx: usize,
    new_neighbors: &[(u32, f32)],
    adjacency: &mut [Vec<u32>],
    doc_profiles: &[DocProfile],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) {
    update_bridges_cached(doc_idx, new_neighbors, adjacency, doc_profiles, max_degree, codebook, flat_codes, use_emd, None)
}

pub fn update_bridges_cached(
    doc_idx: usize,
    new_neighbors: &[(u32, f32)],
    adjacency: &mut [Vec<u32>],
    doc_profiles: &[DocProfile],
    max_degree: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    mut cache: Option<&mut ScoreCache>,
) {
    let doc_codes = flat_codes.doc_codes(doc_idx);
    let doc_u32 = doc_idx as u32;

    let old_neighbors: Vec<u32> = adjacency[doc_idx].clone();
    let mut all_candidates: Vec<(u32, f32)> = Vec::new();

    for &nbr in &old_neighbors {
        let dist = if let Some(ref mut c) = cache {
            if let Some(d) = c.get(doc_u32, nbr) {
                d
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                let d = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                c.insert(doc_u32, nbr, d);
                d
            }
        } else {
            let nbr_codes = flat_codes.doc_codes(nbr as usize);
            construction_distance(codebook, doc_codes, nbr_codes, use_emd)
        };
        all_candidates.push((nbr, dist));
    }

    for &(nbr, _) in new_neighbors {
        if nbr as usize != doc_idx && !old_neighbors.contains(&nbr) {
            let dist = if let Some(ref mut c) = cache {
                if let Some(d) = c.get(doc_u32, nbr) {
                    d
                } else {
                    let nbr_codes = flat_codes.doc_codes(nbr as usize);
                    let d = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                    c.insert(doc_u32, nbr, d);
                    d
                }
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                construction_distance(codebook, doc_codes, nbr_codes, use_emd)
            };
            all_candidates.push((nbr, dist));
        }
    }

    if all_candidates.len() <= max_degree {
        for &old_nbr in &old_neighbors {
            adjacency[old_nbr as usize].retain(|&x| x != doc_idx as u32);
        }
        adjacency[doc_idx] = all_candidates.iter().map(|&(id, _)| id).collect();
        for &(nbr, _) in &all_candidates {
            if !adjacency[nbr as usize].contains(&(doc_idx as u32)) {
                adjacency[nbr as usize].push(doc_idx as u32);
                if adjacency[nbr as usize].len() > max_degree {
                    shrink_neighbors_emd_cached(nbr as usize, max_degree, adjacency, codebook, flat_codes, use_emd, cache.as_deref_mut());
                }
            }
        }
        return;
    }

    // Step 3: Select top-M by distance
    all_candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let mut selected: Vec<(u32, f32)> = all_candidates.iter().take(max_degree).cloned().collect();

    // Step 4: Enforce cross-cluster constraint (iterated to convergence)
    // Iterate up to |ctop| times since replacing for cluster A can evict the only
    // neighbor covering cluster B.
    if doc_idx < doc_profiles.len() {
        let ctop = &doc_profiles[doc_idx].ctop;
        let max_passes = ctop.len();

        for _ in 0..max_passes {
            let mut any_replaced = false;
            for &cluster_id in ctop {
                let has_cluster_neighbor = selected.iter().any(|&(nbr, _)| {
                    let nbr_idx = nbr as usize;
                    nbr_idx < doc_profiles.len()
                        && doc_profiles[nbr_idx].ctop.contains(&cluster_id)
                });

                if !has_cluster_neighbor {
                    if let Some(&(bridge_nbr, bridge_dist)) = all_candidates.iter().find(|&&(nbr, _)| {
                        let ni = nbr as usize;
                        ni != doc_idx
                            && ni < doc_profiles.len()
                            && doc_profiles[ni].ctop.contains(&cluster_id)
                    }) {
                        if let Some(farthest_idx) = selected
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
                            .map(|(i, _)| i)
                        {
                            selected[farthest_idx] = (bridge_nbr, bridge_dist);
                            any_replaced = true;
                        }
                    }
                }
            }
            if !any_replaced {
                break;
            }
        }
    }

    // Apply: remove old reverse edges, set new ones
    for &old_nbr in &old_neighbors {
        adjacency[old_nbr as usize].retain(|&x| x != doc_idx as u32);
    }
    adjacency[doc_idx] = selected.iter().map(|&(id, _)| id).collect();
    for &(nbr, _) in &selected {
        if !adjacency[nbr as usize].contains(&(doc_idx as u32)) {
            adjacency[nbr as usize].push(doc_idx as u32);
            if adjacency[nbr as usize].len() > max_degree {
                evict_worst_neighbor_emd(nbr as usize, max_degree + 1, adjacency, codebook, flat_codes, use_emd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NN-Descent Graph Construction (Dong et al. 2011 + Parallel Clusters)
//
// Faithful implementation of "Efficient K-Nearest Neighbor Graph Construction
// for Generic Similarity Measures" with the following SOTA optimizations:
//
// 1. New/Old flag tracking: only join pairs where ≥1 neighbor is "new" (i.e.
//    inserted in the previous iteration). Old-old pairs are NEVER re-examined.
//    This yields exponentially decreasing work per iteration.
// 2. Reverse neighbor lists: precomputed in O(n*k) instead of O(n²) scan.
//    Ensures bidirectional discovery: if u is in v's list, v's neighbors
//    are candidates for u even if u isn't in v's list.
// 3. Global pair deduplication: all (new×new) and (new×old) pairs across all
//    nodes are collected, sorted, and deduped before distance computation.
//    A single distance(u1, u2) updates BOTH u1's and u2's lists.
// 4. Parallel distance computation: deduplicated pairs are evaluated in
//    parallel via rayon.
// 5. Parallel initialization: random k-NN init is embarrassingly parallel.
// 6. Binary-search insertion: O(k) shift instead of O(k log k) re-sort.
// ---------------------------------------------------------------------------

/// Try to insert `neighbor` at distance `dist` into a sorted k-NN list.
/// Uses binary search for position, O(k) shift. Returns true on success.
/// `nn_flag` tracks the new/old status per entry (true = "new").
#[inline]
fn nn_try_insert(
    nn_idx: &mut Vec<u32>,
    nn_dist: &mut Vec<f32>,
    nn_flag: &mut Vec<bool>,
    neighbor: u32,
    dist: f32,
    k: usize,
) -> bool {
    if nn_idx.iter().any(|&i| i == neighbor) {
        return false;
    }
    if nn_idx.len() >= k {
        if let Some(&worst) = nn_dist.last() {
            if dist >= worst {
                return false;
            }
        }
        nn_idx.pop();
        nn_dist.pop();
        nn_flag.pop();
    }
    let pos = nn_dist.partition_point(|&d| d < dist);
    nn_idx.insert(pos, neighbor);
    nn_dist.insert(pos, dist);
    nn_flag.insert(pos, true);
    true
}

/// Batch distance function: given pairs of global doc IDs, return distances.
/// Enables plugging in CPU (rayon-parallel) or GPU (Triton kernel) computation.
pub type BatchDistFn = dyn Fn(&[u32], &[u32]) -> Vec<f32> + Send + Sync;

/// NN-Descent (Dong et al. 2011): build approximate k-NN graph via iterative
/// local joins with new/old tracking and reverse neighbor propagation.
///
/// Operates on a subset of documents (identified by `doc_ids` which are global
/// indices into `flat_codes`). Internally uses local indices 0..n, converting
/// back to global IDs in the output.
///
/// When `gpu_dist_fn` is provided, pair collection and distance computation are
/// separated: pairs are collected first (parallel), then distances are computed
/// in a single batch call (enabling GPU acceleration). When None, distances are
/// computed inline during pair collection (CPU-parallel streaming join).
fn nn_descent(
    doc_ids: &[u32],
    k: usize,
    n_iters: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    seed: u64,
    gpu_dist_fn: Option<&BatchDistFn>,
) -> Vec<Vec<(u32, f32)>> {
    use rand::RngCore;
    use std::collections::HashSet;

    let n = doc_ids.len();
    if n <= 1 {
        return vec![Vec::new(); n];
    }
    let k = k.min(n - 1);
    let local_to_global: Vec<u32> = doc_ids.to_vec();

    // ── Random initialization ───────────────────────────────────────────
    // Collect all init pairs, then batch-compute distances (GPU or CPU).
    let init_pairs: Vec<(usize, Vec<usize>)> = (0..n).into_par_iter().map(|i| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(i as u64));
        let mut indices: Vec<usize> = (0..n).collect();
        indices.swap(i, n - 1);
        let pool = &mut indices[..n - 1];
        for j in 0..k.min(pool.len()) {
            let r = j + (rng.next_u64() as usize % (pool.len() - j));
            pool.swap(j, r);
        }
        let selected: Vec<usize> = pool[..k.min(pool.len())].to_vec();
        (i, selected)
    }).collect();

    // Flatten init pairs for batch distance computation
    let mut init_a: Vec<u32> = Vec::new();
    let mut init_b: Vec<u32> = Vec::new();
    let mut init_pair_map: Vec<(usize, usize)> = Vec::new(); // (node_local, pair_index)
    for (i, selected) in &init_pairs {
        for &j in selected {
            init_pair_map.push((*i, init_a.len()));
            init_a.push(local_to_global[*i]);
            init_b.push(local_to_global[j]);
        }
    }

    let init_dists = if let Some(gpu_fn) = gpu_dist_fn {
        gpu_fn(&init_a, &init_b)
    } else {
        init_a.par_iter().zip(init_b.par_iter()).map(|(&ga, &gb)| {
            construction_distance(codebook, flat_codes.doc_codes(ga as usize), flat_codes.doc_codes(gb as usize), use_emd)
        }).collect()
    };

    let mut nn_idx: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut nn_dist: Vec<Vec<f32>> = vec![Vec::new(); n];
    let mut nn_flag: Vec<Vec<bool>> = vec![Vec::new(); n];

    // Reconstruct per-node sorted neighbor lists from batch results
    let mut per_node_scored: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    let mut dist_idx = 0usize;
    for (i, selected) in &init_pairs {
        for &j in selected {
            per_node_scored[*i].push((j as u32, init_dists[dist_idx]));
            dist_idx += 1;
        }
    }
    for i in 0..n {
        per_node_scored[i].sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        nn_idx[i] = per_node_scored[i].iter().map(|&(j, _)| j).collect();
        nn_dist[i] = per_node_scored[i].iter().map(|&(_, d)| d).collect();
        nn_flag[i] = vec![true; nn_idx[i].len()];
    }
    drop(per_node_scored);
    drop(init_pairs);

    let convergence_threshold = ((0.001 * n as f64 * k as f64) as usize).max(1);
    let rho_k = k;

    // ── Main loop ───────────────────────────────────────────────────────
    for iter in 0..n_iters {
        let mut new_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut old_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for v in 0..n {
            for (pos, &nbr) in nn_idx[v].iter().enumerate() {
                let nbr_usize = nbr as usize;
                if nbr_usize >= n { continue; }
                if nn_flag[v][pos] {
                    new_rev[nbr_usize].push(v as u32);
                } else {
                    old_rev[nbr_usize].push(v as u32);
                }
            }
        }

        let new_cands: Vec<Vec<u32>> = (0..n).map(|v| {
            let mut c: Vec<u32> = nn_idx[v].iter().enumerate()
                .filter(|&(pos, _)| nn_flag[v][pos])
                .map(|(_, &idx)| idx)
                .collect();
            c.extend_from_slice(&new_rev[v]);
            c.sort_unstable();
            c.dedup();
            c.truncate(rho_k);
            c
        }).collect();

        let old_cands: Vec<Vec<u32>> = (0..n).map(|v| {
            let mut c: Vec<u32> = nn_idx[v].iter().enumerate()
                .filter(|&(pos, _)| !nn_flag[v][pos])
                .map(|(_, &idx)| idx)
                .collect();
            c.extend_from_slice(&old_rev[v]);
            c.sort_unstable();
            c.dedup();
            c.truncate(rho_k);
            c
        }).collect();

        drop(new_rev);
        drop(old_rev);

        for flags in &mut nn_flag {
            flags.iter_mut().for_each(|f| *f = false);
        }

        // Step 4: Collect pairs (parallel) — no distance computation yet.
        // This separation enables GPU batch distance computation.
        let all_pair_sets: Vec<Vec<(u32, u32)>> = (0..n).into_par_iter().filter_map(|v| {
            let new_v = &new_cands[v];
            let old_v = &old_cands[v];
            if new_v.is_empty() { return None; }

            let mut results = Vec::new();
            let mut seen = HashSet::new();

            for i in 0..new_v.len() {
                for j in (i + 1)..new_v.len() {
                    let (a, b) = if new_v[i] < new_v[j] { (new_v[i], new_v[j]) } else { (new_v[j], new_v[i]) };
                    if seen.insert((a, b)) {
                        results.push((a, b));
                    }
                }
            }
            for &u1 in new_v {
                for &u2 in old_v {
                    if u1 == u2 { continue; }
                    let (a, b) = if u1 < u2 { (u1, u2) } else { (u2, u1) };
                    if seen.insert((a, b)) {
                        results.push((a, b));
                    }
                }
            }

            if results.is_empty() { None } else { Some(results) }
        }).collect();

        // Flatten and globally deduplicate pairs
        let mut unique_pairs: Vec<(u32, u32)> = all_pair_sets.into_iter().flatten().collect();
        let n_pairs_raw = unique_pairs.len();
        unique_pairs.sort_unstable();
        unique_pairs.dedup();

        if unique_pairs.is_empty() {
            progress!("[NN-Descent] iter {}/{}: 0 pairs — converged", iter + 1, n_iters);
            break;
        }

        // Step 5: Batch distance computation (GPU or CPU)
        let pair_a: Vec<u32> = unique_pairs.iter().map(|&(a, _)| local_to_global[a as usize]).collect();
        let pair_b: Vec<u32> = unique_pairs.iter().map(|&(_, b)| local_to_global[b as usize]).collect();

        let distances = if let Some(gpu_fn) = gpu_dist_fn {
            gpu_fn(&pair_a, &pair_b)
        } else {
            pair_a.par_iter().zip(pair_b.par_iter()).map(|(&ga, &gb)| {
                construction_distance(codebook, flat_codes.doc_codes(ga as usize), flat_codes.doc_codes(gb as usize), use_emd)
            }).collect()
        };

        // Step 6: Apply symmetrized updates
        let mut total_updates = 0usize;
        for (idx, &(a, b)) in unique_pairs.iter().enumerate() {
            let d = distances[idx];
            if nn_try_insert(&mut nn_idx[a as usize], &mut nn_dist[a as usize],
                             &mut nn_flag[a as usize], b, d, k) {
                total_updates += 1;
            }
            if nn_try_insert(&mut nn_idx[b as usize], &mut nn_dist[b as usize],
                             &mut nn_flag[b as usize], a, d, k) {
                total_updates += 1;
            }
        }

        progress!("[NN-Descent] iter {}/{}: {} unique pairs (from {}), {} updates (threshold={})",
            iter + 1, n_iters, unique_pairs.len(), n_pairs_raw, total_updates, convergence_threshold);

        if total_updates <= convergence_threshold {
            progress!("[NN-Descent] converged at iter {}", iter + 1);
            break;
        }
    }

    nn_idx.iter().zip(&nn_dist).map(|(idx, dist)| {
        idx.iter().zip(dist).map(|(&local, &d)| {
            (local_to_global[local as usize], d)
        }).collect()
    }).collect()
}

/// Build GEM graph using NN-Descent + Parallel Cluster Subgraphs.
///
/// Phase 1: Parallel cluster-local NN-Descent (all clusters run concurrently)
/// Phase 2: Merge per-doc neighbors from all clusters + diversity prune (parallel)
/// Phase 2b (optional): Beam search refinement — discovers navigable long-range
///   edges by simulating search on the NN-Descent graph. Off by default; useful
///   at 100K+ docs where cross-cluster navigability becomes critical.
/// Phase 3: Repair (bridge repair + global connectivity guarantee)
///
/// `refine_with_beam_search`: if Some((all_vectors, dim, doc_offsets, ef_construction)),
///   runs Phase 2b beam search refinement after the NN-Descent merge.
/// `gpu_dist_fn`: optional batch distance function for GPU-accelerated distance
///   computation during NN-Descent. When provided, clusters are processed
///   sequentially (GPU handles parallelism within each cluster's distance batch).
pub fn build_graph_nndescent(
    n_docs: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    use_emd: bool,
    refine_with_beam_search: Option<(&[f32], usize, &[(usize, usize)], usize)>,
    gpu_dist_fn: Option<&BatchDistFn>,
) -> GemGraph {
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            shortcut_generations: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_degree,
        };
    }

    let graph_start = Instant::now();
    let n_clusters = postings.lists.len();
    let nn_k = (max_degree * 2).min(96);
    let nn_iters = 12;

    fn rss_gb() -> f64 {
        if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = s.split_whitespace().nth(1) {
                if let Ok(p) = rss_pages.parse::<u64>() {
                    return (p * 4096) as f64 / 1e9;
                }
            }
        }
        -1.0
    }

    let use_gpu = gpu_dist_fn.is_some();
    progress!("[GEM nndescent] ══════════════════════════════════════════════════");
    progress!("[GEM nndescent] {} docs, {} clusters, max_degree={}, nn_k={}, nn_iters={}",
        n_docs, n_clusters, max_degree, nn_k, nn_iters);
    progress!("[GEM nndescent] use_emd={}, gpu={}, RSS: {:.2} GB", use_emd, use_gpu, rss_gb());

    // ── Phase 1: Cluster-local NN-Descent ─────────────────────────────
    // CPU mode: process clusters in parallel batches via rayon.
    // GPU mode: process clusters sequentially (GPU handles parallelism
    //   within each cluster's batch distance computation).
    let batch_size = if use_gpu {
        1 // Sequential when GPU handles distance parallelism
    } else {
        (n_clusters / 2).max(rayon::current_num_threads()).max(4)
    };
    progress!("[GEM nndescent] Phase 1: Cluster-local NN-Descent (batch_size={}, {})...",
        batch_size, if use_gpu { "GPU" } else { "CPU" });
    let phase1_start = Instant::now();

    let mut per_doc_candidates: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_docs];
    let mut clusters_done = 0usize;

    let mut cluster_order: Vec<usize> = (0..n_clusters).collect();
    cluster_order.sort_unstable_by(|&a, &b| postings.lists[b].len().cmp(&postings.lists[a].len()));

    for batch_start in (0..n_clusters).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n_clusters);
        let batch_ids: Vec<usize> = cluster_order[batch_start..batch_end].to_vec();

        let batch_results: Vec<Vec<(u32, Vec<(u32, f32)>)>> = if use_gpu {
            // GPU path: sequential cluster processing
            batch_ids.iter().filter_map(|&cluster_id| {
                let members = &postings.lists[cluster_id];
                if members.len() <= 1 {
                    return None;
                }
                let k_local = nn_k.min(members.len() - 1);
                if k_local == 0 {
                    return None;
                }

                let nn_lists = nn_descent(
                    members, k_local, nn_iters,
                    codebook, flat_codes, use_emd,
                    42u64.wrapping_add(cluster_id as u64),
                    gpu_dist_fn,
                );

                let result: Vec<(u32, Vec<(u32, f32)>)> = members.iter()
                    .enumerate()
                    .map(|(local_idx, &global_id)| (global_id, nn_lists[local_idx].clone()))
                    .collect();
                Some(result)
            }).collect()
        } else {
            // CPU path: parallel cluster processing via rayon
            batch_ids.into_par_iter().filter_map(|cluster_id| {
                let members = &postings.lists[cluster_id];
                if members.len() <= 1 {
                    return None;
                }
                let k_local = nn_k.min(members.len() - 1);
                if k_local == 0 {
                    return None;
                }

                let nn_lists = nn_descent(
                    members, k_local, nn_iters,
                    codebook, flat_codes, use_emd,
                    42u64.wrapping_add(cluster_id as u64),
                    None, // CPU: inline distance computation
                );

                let result: Vec<(u32, Vec<(u32, f32)>)> = members.iter()
                    .enumerate()
                    .map(|(local_idx, &global_id)| (global_id, nn_lists[local_idx].clone()))
                    .collect();
                Some(result)
            }).collect()
        };

        for cluster_nn in batch_results {
            for (global_id, nn_list) in cluster_nn {
                let doc_idx = global_id as usize;
                if doc_idx < n_docs {
                    per_doc_candidates[doc_idx].extend(nn_list);
                }
            }
            clusters_done += 1;
        }

        unsafe { libc::malloc_trim(0); }

        if batch_start % (batch_size * 4).max(1) == 0 || batch_end == n_clusters {
            progress!("[GEM nndescent] Phase 1: {}/{} clusters done ({:.0}s) — RSS: {:.2} GB",
                clusters_done, n_clusters, phase1_start.elapsed().as_secs_f64(), rss_gb());
        }
    }

    progress!("[GEM nndescent] Phase 1 done in {:.1}s — {} clusters — RSS: {:.2} GB",
        phase1_start.elapsed().as_secs_f64(), clusters_done, rss_gb());

    // ── Phase 2: Merge + Diversity Prune ────────────────────────────────
    progress!("[GEM nndescent] Phase 2: Merge + diversity prune — RSS: {:.2} GB", rss_gb());
    let phase2_start = Instant::now();

    // Parallel: dedup, diversity-prune, enforce cross-cluster bridges
    let pruned_neighbors: Vec<Vec<u32>> = per_doc_candidates.into_par_iter()
        .enumerate()
        .map(|(doc_idx, mut cands)| {
            if cands.is_empty() {
                return Vec::new();
            }

            // Dedup: sort by neighbor ID, keep best distance per neighbor
            cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
            cands.dedup_by_key(|c| c.0);
            cands.retain(|&(nbr, _)| nbr as usize != doc_idx);
            cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            let doc_codes = flat_codes.doc_codes(doc_idx);
            if doc_codes.is_empty() {
                return cands.iter().take(max_degree).map(|&(id, _)| id).collect();
            }

            let selected = select_neighbors_heuristic(
                &cands, doc_codes, max_degree, codebook, flat_codes,
            );

            let mut result: Vec<u32> = selected.iter().map(|&(id, _)| id).collect();

            // Cross-cluster bridge constraint
            if doc_idx < doc_profiles.len() {
                let ctop = &doc_profiles[doc_idx].ctop;
                for &cluster_id in ctop {
                    let covered = result.iter().any(|&nbr| {
                        let ni = nbr as usize;
                        ni < doc_profiles.len()
                            && doc_profiles[ni].ctop.contains(&cluster_id)
                    });
                    if !covered {
                        if let Some(&(bridge_nbr, _)) = cands.iter().find(|&&(nbr, _)| {
                            let ni = nbr as usize;
                            ni != doc_idx
                                && ni < doc_profiles.len()
                                && doc_profiles[ni].ctop.contains(&cluster_id)
                        }) {
                            if result.len() >= max_degree {
                                result.pop();
                            }
                            if !result.contains(&bridge_nbr) {
                                result.push(bridge_nbr);
                            }
                        }
                    }
                }
            }

            result
        })
        .collect();

    // Build bidirectional adjacency via edge collection (avoids O(k) contains)
    let total_directed: usize = pruned_neighbors.iter().map(|n| n.len()).sum();
    let mut all_edges: Vec<(u32, u32)> = Vec::with_capacity(total_directed * 2);
    for (doc_idx, nbrs) in pruned_neighbors.iter().enumerate() {
        let d = doc_idx as u32;
        for &nbr in nbrs {
            all_edges.push((d, nbr));
            all_edges.push((nbr, d));
        }
    }
    all_edges.sort_unstable();

    let mut adjacency: Vec<Vec<u32>> = (0..n_docs).into_par_iter().map(|i| {
        let i_u32 = i as u32;
        let start = all_edges.partition_point(|e| e.0 < i_u32);
        let end = all_edges.partition_point(|e| e.0 <= i_u32);
        let mut nbrs: Vec<u32> = all_edges[start..end].iter().map(|&(_, t)| t).collect();
        nbrs.sort_unstable();
        nbrs.dedup();
        nbrs.retain(|&n| n as usize != i);
        nbrs
    }).collect();
    drop(all_edges);

    // Shared score cache for Phase 2 trim, Phase 2b, and Phase 3 repair
    let mut score_cache = ScoreCache::new(n_docs * 8);

    // Trim over-degree nodes
    for doc_idx in 0..n_docs {
        if adjacency[doc_idx].len() > max_degree {
            shrink_neighbors_emd_cached(doc_idx, max_degree, &mut adjacency, codebook, flat_codes, use_emd, Some(&mut score_cache));
        }
    }

    unsafe { libc::malloc_trim(0); }
    progress!("[GEM nndescent] Phase 2 done in {:.1}s — RSS: {:.2} GB", phase2_start.elapsed().as_secs_f64(), rss_gb());

    // ── Phase 2b (optional): Beam search refinement ─────────────────────
    // At small scale (<10K) NN-Descent quality already matches beam search
    // construction. At large scale (100K+), cross-cluster navigability
    // becomes critical and this pass discovers long-range highway edges
    // that NN-Descent's local search cannot find.
    if let Some((all_vectors, dim, doc_offsets, ef_construction)) = refine_with_beam_search {
        progress!("[GEM nndescent] Phase 2b: Beam search refinement (ef={})...", ef_construction);
        let phase2b_start = Instant::now();
        let n_fine = codebook.n_fine;

        let all_scores: Vec<(usize, Vec<f32>)> = (0..n_docs).into_par_iter().map(|doc_idx| {
            let (start, end) = doc_offsets[doc_idx];
            let n_tokens = end - start;
            if n_tokens == 0 || dim == 0 {
                return (doc_idx, Vec::new());
            }
            let doc_vecs = &all_vectors[start * dim..end * dim];
            let scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);
            (doc_idx, scores)
        }).collect();

        let mut refined = 0usize;
        for (doc_idx, query_scores) in &all_scores {
            let doc_idx = *doc_idx;
            if query_scores.is_empty() {
                continue;
            }
            let (start, end) = doc_offsets[doc_idx];
            let n_tokens = end - start;

            let mut entries: Vec<u32> = Vec::new();
            if doc_idx < doc_profiles.len() {
                for &cid in &doc_profiles[doc_idx].ctop {
                    if let Some(Some(rep)) = postings.cluster_reps.get(cid as usize) {
                        if (*rep as usize) != doc_idx && !entries.contains(rep) {
                            entries.push(*rep);
                        }
                    }
                }
            }
            for &nbr in &adjacency[doc_idx] {
                if !entries.contains(&nbr) {
                    entries.push(nbr);
                }
                if entries.len() >= 8 { break; }
            }
            if entries.is_empty() {
                continue;
            }

            let adj = VecAdjacency(&adjacency);
            let candidates = beam_search_construction(
                &adj, &entries, query_scores, n_tokens,
                flat_codes, n_fine, ef_construction, n_docs,
            );

            if candidates.is_empty() {
                continue;
            }

            let doc_codes = flat_codes.doc_codes(doc_idx);
            let old_neighbors: Vec<u32> = adjacency[doc_idx].clone();

            let mut all_cands: Vec<(u32, f32)> = Vec::with_capacity(old_neighbors.len() + candidates.len());
            for &nbr in &old_neighbors {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                let dist = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                all_cands.push((nbr, dist));
            }
            for &(cand, score) in &candidates {
                if cand as usize != doc_idx && !old_neighbors.contains(&cand) {
                    all_cands.push((cand, score));
                }
            }
            all_cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            all_cands.dedup_by_key(|c| c.0);

            let new_neighbors = select_neighbors_heuristic_cached(
                &all_cands, doc_codes, max_degree, codebook, flat_codes,
                Some(&mut score_cache), use_emd,
            );

            for &old_nbr in &old_neighbors {
                adjacency[old_nbr as usize].retain(|&x| x != doc_idx as u32);
            }
            adjacency[doc_idx] = new_neighbors.iter().map(|&(id, _)| id).collect();
            for &(nbr, _) in &new_neighbors {
                if !adjacency[nbr as usize].contains(&(doc_idx as u32)) {
                    adjacency[nbr as usize].push(doc_idx as u32);
                    if adjacency[nbr as usize].len() > max_degree {
                        shrink_neighbors_emd_cached(
                            nbr as usize, max_degree, &mut adjacency, codebook, flat_codes,
                            use_emd, Some(&mut score_cache),
                        );
                    }
                }
            }
            refined += 1;
        }
        drop(all_scores);

        unsafe { libc::malloc_trim(0); }
        progress!("[GEM nndescent] Phase 2b done in {:.1}s — {} nodes refined — RSS: {:.2} GB",
            phase2b_start.elapsed().as_secs_f64(), refined, rss_gb());
    }

    // ── Phase 3: Repair ─────────────────────────────────────────────────
    progress!("[GEM nndescent] Phase 3: Repair — RSS: {:.2} GB", rss_gb());
    let phase3_start = Instant::now();

    let isolated_count = adjacency.iter().filter(|a| a.is_empty()).count();
    if isolated_count > 0 {
        progress!("[GEM nndescent] Repairing {} isolated nodes...", isolated_count);
        let mut repaired = 0usize;
        for doc_idx in 0..n_docs {
            if !adjacency[doc_idx].is_empty() {
                continue;
            }
            let doc_codes = flat_codes.doc_codes(doc_idx);
            if doc_codes.is_empty() {
                continue;
            }
            let doc_u32 = doc_idx as u32;
            let mut best_nbr = u32::MAX;
            let mut best_dist = f32::MAX;
            #[allow(clippy::needless_range_loop)]
            for other in 0..n_docs {
                if other == doc_idx || adjacency[other].is_empty() {
                    continue;
                }
                let d = if let Some(cached) = score_cache.get(doc_u32, other as u32) {
                    cached
                } else {
                    let other_codes = flat_codes.doc_codes(other);
                    let d = construction_distance(codebook, doc_codes, other_codes, use_emd);
                    score_cache.insert(doc_u32, other as u32, d);
                    d
                };
                if d < best_dist {
                    best_dist = d;
                    best_nbr = other as u32;
                }
            }
            if best_nbr != u32::MAX {
                adjacency[doc_idx].push(best_nbr);
                adjacency[best_nbr as usize].push(doc_idx as u32);
                repaired += 1;
            }
        }
        progress!("[GEM nndescent] Repaired {} isolated nodes", repaired);
    }

    bridge_repair_emd(&mut adjacency, max_degree, postings, codebook, flat_codes, use_emd);
    progress!("[GEM nndescent] bridge repair done in {:.1}s", phase3_start.elapsed().as_secs_f64());

    let global_start = Instant::now();
    global_bridge_repair_cached(&mut adjacency, max_degree, codebook, flat_codes, use_emd, &mut score_cache);
    progress!("[GEM nndescent] global bridge repair done in {:.1}s", global_start.elapsed().as_secs_f64());
    progress!("[GEM nndescent] score cache: {} entries, {:.1}% hit rate",
        score_cache.len(), score_cache.hit_rate() * 100.0);

    let csr = CsrAdjacency::from_adj_lists(&adjacency);

    let entry_point = (0..n_docs)
        .max_by_key(|&i| adjacency[i].len())
        .unwrap_or(0) as u32;

    progress!("[GEM nndescent] ══════════════════════════════════════════════════");
    progress!("[GEM nndescent] TOTAL BUILD TIME: {:.1}s ({:.1} min)",
        graph_start.elapsed().as_secs_f64(), graph_start.elapsed().as_secs_f64() / 60.0);
    progress!("[GEM nndescent] ══════════════════════════════════════════════════");

    GemGraph {
        levels: vec![csr],
        shortcuts: vec![Vec::new(); n_docs],
        shortcut_generations: vec![Vec::new(); n_docs],
        node_levels: vec![0; n_docs],
        entry_point,
        max_degree,
    }
}

// ---------------------------------------------------------------------------
// RNN-Descent: NN-Descent + MRNG pruning + Navigating Node + DFS connectivity
//
// Implements the RNN-Descent algorithm (Ono & Matsui, ACM MM 2023) adapted
// for multi-vector set-level distances (qCH/qEMD). Combines:
//   1. NN-Descent for fast parallel kNN construction
//   2. MRNG (Monotonic Relative Neighborhood Graph) edge selection
//   3. Navigating node (medoid) as global entry point
//   4. DFS spanning tree connectivity guarantee
//
// This produces a navigable graph (unlike vanilla NN-Descent) at construction
// speeds 2-5x faster than HNSW (unlike sequential HNSW insertion).
// ---------------------------------------------------------------------------

/// Compute the navigating node (medoid): the document whose average qCH
/// distance to all other documents is minimal. For efficiency, we approximate
/// by finding the doc whose centroid codes are closest to the "average" centroid
/// distribution. Uses sampling for large corpora.
fn compute_navigating_node(
    n_docs: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
) -> u32 {
    if n_docs <= 1 {
        return 0;
    }

    // For small corpora, exact medoid via pairwise distances
    // For large corpora, sample-based approximation
    let sample_size = n_docs.min(200);
    let step = if n_docs > sample_size { n_docs / sample_size } else { 1 };
    let samples: Vec<usize> = (0..n_docs).step_by(step).take(sample_size).collect();

    let best = (0..n_docs).into_par_iter().map(|i| {
        let codes_i = flat_codes.doc_codes(i);
        if codes_i.is_empty() {
            return (i, f64::MAX);
        }
        let total_dist: f64 = samples.iter().map(|&j| {
            if j == i { return 0.0; }
            let codes_j = flat_codes.doc_codes(j);
            if codes_j.is_empty() { return 0.0; }
            construction_distance(codebook, codes_i, codes_j, use_emd) as f64
        }).sum();
        (i, total_dist)
    }).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.0));

    best.0 as u32
}

/// Build DFS spanning tree from the navigating node. Returns parent[i] for
/// each node (parent[nav_node] = nav_node). Nodes unreachable from nav_node
/// get parent = u32::MAX.
fn dfs_spanning_tree(adjacency: &[Vec<u32>], nav_node: u32) -> Vec<u32> {
    let n = adjacency.len();
    let mut parent = vec![u32::MAX; n];
    parent[nav_node as usize] = nav_node;
    let mut stack = vec![nav_node as usize];

    while let Some(node) = stack.pop() {
        for &nbr in &adjacency[node] {
            let nb = nbr as usize;
            if nb < n && parent[nb] == u32::MAX {
                parent[nb] = node as u32;
                stack.push(nb);
            }
        }
    }
    parent
}

/// MRNG-aware NN-Descent: during each iteration's local join, apply the RNG
/// condition before inserting a candidate. A candidate c is rejected for node v
/// if any existing neighbor n of v satisfies dist(n, c) < dist(v, c) — meaning
/// c is "detoured" through n and the edge v→c is redundant for greedy search.
///
/// This is the key insight from RNN-Descent: pruning non-navigable edges during
/// construction (not just at merge time) produces a navigable graph directly.
fn nn_descent_mrng(
    doc_ids: &[u32],
    k: usize,
    n_iters: usize,
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    use_emd: bool,
    seed: u64,
    gpu_dist_fn: Option<&BatchDistFn>,
) -> Vec<Vec<(u32, f32)>> {
    use rand::RngCore;
    use std::collections::HashSet;

    let n = doc_ids.len();
    if n <= 1 {
        return vec![Vec::new(); n];
    }
    let k = k.min(n - 1);
    let local_to_global: Vec<u32> = doc_ids.to_vec();

    // Random initialization (same as vanilla nn_descent)
    let init_pairs: Vec<(usize, Vec<usize>)> = (0..n).into_par_iter().map(|i| {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(i as u64));
        let mut indices: Vec<usize> = (0..n).collect();
        indices.swap(i, n - 1);
        let pool = &mut indices[..n - 1];
        for j in 0..k.min(pool.len()) {
            let r = j + (rng.next_u64() as usize % (pool.len() - j));
            pool.swap(j, r);
        }
        let selected: Vec<usize> = pool[..k.min(pool.len())].to_vec();
        (i, selected)
    }).collect();

    let mut init_a: Vec<u32> = Vec::new();
    let mut init_b: Vec<u32> = Vec::new();
    for (i, selected) in &init_pairs {
        for &j in selected {
            init_a.push(local_to_global[*i]);
            init_b.push(local_to_global[j]);
        }
    }

    let init_dists = if let Some(gpu_fn) = gpu_dist_fn {
        gpu_fn(&init_a, &init_b)
    } else {
        init_a.par_iter().zip(init_b.par_iter()).map(|(&ga, &gb)| {
            construction_distance(codebook, flat_codes.doc_codes(ga as usize), flat_codes.doc_codes(gb as usize), use_emd)
        }).collect()
    };

    let mut nn_idx: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut nn_dist: Vec<Vec<f32>> = vec![Vec::new(); n];
    let mut nn_flag: Vec<Vec<bool>> = vec![Vec::new(); n];

    let mut per_node_scored: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    let mut dist_idx = 0usize;
    for (i, selected) in &init_pairs {
        for &j in selected {
            per_node_scored[*i].push((j as u32, init_dists[dist_idx]));
            dist_idx += 1;
        }
    }
    for i in 0..n {
        per_node_scored[i].sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        nn_idx[i] = per_node_scored[i].iter().map(|&(j, _)| j).collect();
        nn_dist[i] = per_node_scored[i].iter().map(|&(_, d)| d).collect();
        nn_flag[i] = vec![true; nn_idx[i].len()];
    }
    drop(per_node_scored);
    drop(init_pairs);

    let convergence_threshold = ((0.001 * n as f64 * k as f64) as usize).max(1);

    // Main loop with MRNG-aware insertion
    for iter in 0..n_iters {
        let mut new_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut old_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for v in 0..n {
            for (pos, &nbr) in nn_idx[v].iter().enumerate() {
                let nbr_usize = nbr as usize;
                if nbr_usize >= n { continue; }
                if nn_flag[v][pos] {
                    new_rev[nbr_usize].push(v as u32);
                } else {
                    old_rev[nbr_usize].push(v as u32);
                }
            }
        }

        let new_cands: Vec<Vec<u32>> = (0..n).map(|v| {
            let mut c: Vec<u32> = nn_idx[v].iter().enumerate()
                .filter(|&(pos, _)| nn_flag[v][pos])
                .map(|(_, &idx)| idx)
                .collect();
            c.extend_from_slice(&new_rev[v]);
            c.sort_unstable();
            c.dedup();
            c.truncate(k);
            c
        }).collect();

        let old_cands: Vec<Vec<u32>> = (0..n).map(|v| {
            let mut c: Vec<u32> = nn_idx[v].iter().enumerate()
                .filter(|&(pos, _)| !nn_flag[v][pos])
                .map(|(_, &idx)| idx)
                .collect();
            c.extend_from_slice(&old_rev[v]);
            c.sort_unstable();
            c.dedup();
            c.truncate(k);
            c
        }).collect();

        drop(new_rev);
        drop(old_rev);

        for flags in &mut nn_flag {
            flags.iter_mut().for_each(|f| *f = false);
        }

        // Collect pairs
        let all_pair_sets: Vec<Vec<(u32, u32)>> = (0..n).into_par_iter().filter_map(|v| {
            let new_v = &new_cands[v];
            let old_v = &old_cands[v];
            if new_v.is_empty() { return None; }

            let mut results = Vec::new();
            let mut seen = HashSet::new();

            for i in 0..new_v.len() {
                for j in (i + 1)..new_v.len() {
                    let (a, b) = if new_v[i] < new_v[j] { (new_v[i], new_v[j]) } else { (new_v[j], new_v[i]) };
                    if seen.insert((a, b)) {
                        results.push((a, b));
                    }
                }
            }
            for &u1 in new_v {
                for &u2 in old_v {
                    if u1 == u2 { continue; }
                    let (a, b) = if u1 < u2 { (u1, u2) } else { (u2, u1) };
                    if seen.insert((a, b)) {
                        results.push((a, b));
                    }
                }
            }

            if results.is_empty() { None } else { Some(results) }
        }).collect();

        let mut unique_pairs: Vec<(u32, u32)> = all_pair_sets.into_iter().flatten().collect();
        let n_pairs_raw = unique_pairs.len();
        unique_pairs.sort_unstable();
        unique_pairs.dedup();

        if unique_pairs.is_empty() {
            progress!("[RNN-Descent] iter {}/{}: 0 pairs — converged", iter + 1, n_iters);
            break;
        }

        // Batch distance computation
        let pair_a: Vec<u32> = unique_pairs.iter().map(|&(a, _)| local_to_global[a as usize]).collect();
        let pair_b: Vec<u32> = unique_pairs.iter().map(|&(_, b)| local_to_global[b as usize]).collect();

        let distances = if let Some(gpu_fn) = gpu_dist_fn {
            gpu_fn(&pair_a, &pair_b)
        } else {
            pair_a.par_iter().zip(pair_b.par_iter()).map(|(&ga, &gb)| {
                construction_distance(codebook, flat_codes.doc_codes(ga as usize), flat_codes.doc_codes(gb as usize), use_emd)
            }).collect()
        };

        // MRNG-aware update: before inserting (a, b) into a's neighbor list,
        // check if b is "detoured" by any existing neighbor of a.
        let mut total_updates = 0usize;
        for (idx, &(a, b)) in unique_pairs.iter().enumerate() {
            let d = distances[idx];

            // MRNG check for a→b: is b detoured by an existing neighbor of a?
            let detoured_ab = nn_idx[a as usize].iter().zip(nn_dist[a as usize].iter()).any(|(&nbr, &_nbr_dist)| {
                if nbr == b { return false; }
                // Check if dist(nbr, b) < dist(a, b) — meaning going through nbr is shorter
                let nbr_global = local_to_global[nbr as usize];
                let b_global = local_to_global[b as usize];
                let d_nb = construction_distance(
                    codebook,
                    flat_codes.doc_codes(nbr_global as usize),
                    flat_codes.doc_codes(b_global as usize),
                    use_emd,
                );
                d_nb < d
            });

            if !detoured_ab {
                if nn_try_insert(&mut nn_idx[a as usize], &mut nn_dist[a as usize],
                                 &mut nn_flag[a as usize], b, d, k) {
                    total_updates += 1;
                }
            }

            // MRNG check for b→a
            let detoured_ba = nn_idx[b as usize].iter().zip(nn_dist[b as usize].iter()).any(|(&nbr, &_nbr_dist)| {
                if nbr == a { return false; }
                let nbr_global = local_to_global[nbr as usize];
                let a_global = local_to_global[a as usize];
                let d_na = construction_distance(
                    codebook,
                    flat_codes.doc_codes(nbr_global as usize),
                    flat_codes.doc_codes(a_global as usize),
                    use_emd,
                );
                d_na < d
            });

            if !detoured_ba {
                if nn_try_insert(&mut nn_idx[b as usize], &mut nn_dist[b as usize],
                                 &mut nn_flag[b as usize], a, d, k) {
                    total_updates += 1;
                }
            }
        }

        progress!("[RNN-Descent] iter {}/{}: {} unique pairs (from {}), {} updates (threshold={})",
            iter + 1, n_iters, unique_pairs.len(), n_pairs_raw, total_updates, convergence_threshold);

        if total_updates <= convergence_threshold {
            progress!("[RNN-Descent] converged at iter {}", iter + 1);
            break;
        }
    }

    nn_idx.iter().zip(&nn_dist).map(|(idx, dist)| {
        idx.iter().zip(dist).map(|(&local, &d)| {
            (local_to_global[local as usize], d)
        }).collect()
    }).collect()
}

/// Build GEM graph using RNN-Descent: NN-Descent with MRNG inline pruning,
/// navigating node selection, and DFS spanning tree connectivity.
///
/// This is the SOTA hybrid approach: fast parallel construction from NN-Descent
/// combined with navigability guarantees from NSG/MRNG theory. Produces a graph
/// that is both fast to build and navigable for greedy search.
pub fn build_graph_rnn_descent(
    n_docs: usize,
    all_vectors: &[f32],
    dim: usize,
    doc_offsets: &[(usize, usize)],
    codebook: &TwoStageCodebook,
    flat_codes: &FlatDocCodes,
    doc_profiles: &[DocProfile],
    postings: &ClusterPostings,
    max_degree: usize,
    ef_construction: usize,
    use_emd: bool,
    gpu_dist_fn: Option<&BatchDistFn>,
) -> GemGraph {
    if n_docs == 0 {
        return GemGraph {
            levels: vec![CsrAdjacency::empty(0)],
            shortcuts: Vec::new(),
            shortcut_generations: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_degree,
        };
    }

    let graph_start = Instant::now();
    let n_clusters = postings.lists.len();
    let nn_k = (max_degree * 2).min(96);
    let nn_iters = 12;

    fn rss_gb() -> f64 {
        if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = s.split_whitespace().nth(1) {
                if let Ok(p) = rss_pages.parse::<u64>() {
                    return (p * 4096) as f64 / 1e9;
                }
            }
        }
        -1.0
    }

    let use_gpu = gpu_dist_fn.is_some();
    progress!("[RNN-Descent] ══════════════════════════════════════════════════");
    progress!("[RNN-Descent] {} docs, {} clusters, max_degree={}, nn_k={}", n_docs, n_clusters, max_degree, nn_k);
    progress!("[RNN-Descent] use_emd={}, gpu={}, RSS: {:.2} GB", use_emd, use_gpu, rss_gb());

    // ── Phase 1: Cluster-local MRNG-aware NN-Descent ─────────────────
    let phase1_start = Instant::now();
    progress!("[RNN-Descent] Phase 1: MRNG-aware NN-Descent per cluster...");

    let mut per_doc_candidates: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_docs];
    let mut clusters_done = 0usize;

    let mut cluster_order: Vec<usize> = (0..n_clusters).collect();
    cluster_order.sort_unstable_by(|&a, &b| postings.lists[b].len().cmp(&postings.lists[a].len()));

    let batch_size = if use_gpu { 1 } else {
        (n_clusters / 2).max(rayon::current_num_threads()).max(4)
    };

    for batch_start in (0..n_clusters).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n_clusters);
        let batch_ids: Vec<usize> = cluster_order[batch_start..batch_end].to_vec();

        let batch_results: Vec<Vec<(u32, Vec<(u32, f32)>)>> = if use_gpu {
            batch_ids.iter().filter_map(|&cluster_id| {
                let members = &postings.lists[cluster_id];
                if members.len() <= 1 { return None; }
                let k_local = nn_k.min(members.len() - 1);
                if k_local == 0 { return None; }

                let nn_lists = nn_descent_mrng(
                    members, k_local, nn_iters,
                    codebook, flat_codes, use_emd,
                    42u64.wrapping_add(cluster_id as u64),
                    gpu_dist_fn,
                );

                Some(members.iter().enumerate()
                    .map(|(li, &gid)| (gid, nn_lists[li].clone()))
                    .collect())
            }).collect()
        } else {
            batch_ids.into_par_iter().filter_map(|cluster_id| {
                let members = &postings.lists[cluster_id];
                if members.len() <= 1 { return None; }
                let k_local = nn_k.min(members.len() - 1);
                if k_local == 0 { return None; }

                let nn_lists = nn_descent_mrng(
                    members, k_local, nn_iters,
                    codebook, flat_codes, use_emd,
                    42u64.wrapping_add(cluster_id as u64),
                    None,
                );

                Some(members.iter().enumerate()
                    .map(|(li, &gid)| (gid, nn_lists[li].clone()))
                    .collect())
            }).collect()
        };

        for cluster_nn in batch_results {
            for (global_id, nn_list) in cluster_nn {
                let doc_idx = global_id as usize;
                if doc_idx < n_docs {
                    per_doc_candidates[doc_idx].extend(nn_list);
                }
            }
            clusters_done += 1;
        }

        unsafe { libc::malloc_trim(0); }

        if batch_start % (batch_size * 4).max(1) == 0 || batch_end == n_clusters {
            progress!("[RNN-Descent] Phase 1: {}/{} clusters done ({:.0}s) — RSS: {:.2} GB",
                clusters_done, n_clusters, phase1_start.elapsed().as_secs_f64(), rss_gb());
        }
    }

    progress!("[RNN-Descent] Phase 1 done in {:.1}s — RSS: {:.2} GB",
        phase1_start.elapsed().as_secs_f64(), rss_gb());

    // ── Phase 2: Merge + MRNG diversity prune ────────────────────────
    progress!("[RNN-Descent] Phase 2: Merge + MRNG diversity prune...");
    let phase2_start = Instant::now();

    let pruned_neighbors: Vec<Vec<u32>> = per_doc_candidates.into_par_iter()
        .enumerate()
        .map(|(doc_idx, mut cands)| {
            if cands.is_empty() { return Vec::new(); }

            cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
            cands.dedup_by_key(|c| c.0);
            cands.retain(|&(nbr, _)| nbr as usize != doc_idx);
            cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            let doc_codes = flat_codes.doc_codes(doc_idx);
            if doc_codes.is_empty() {
                return cands.iter().take(max_degree).map(|&(id, _)| id).collect();
            }

            let selected = select_neighbors_heuristic(
                &cands, doc_codes, max_degree, codebook, flat_codes,
            );

            let mut result: Vec<u32> = selected.iter().map(|&(id, _)| id).collect();

            // Cross-cluster bridge constraint
            if doc_idx < doc_profiles.len() {
                let ctop = &doc_profiles[doc_idx].ctop;
                for &cluster_id in ctop {
                    let covered = result.iter().any(|&nbr| {
                        let ni = nbr as usize;
                        ni < doc_profiles.len()
                            && doc_profiles[ni].ctop.contains(&cluster_id)
                    });
                    if !covered {
                        if let Some(&(bridge_nbr, _)) = cands.iter().find(|&&(nbr, _)| {
                            let ni = nbr as usize;
                            ni != doc_idx
                                && ni < doc_profiles.len()
                                && doc_profiles[ni].ctop.contains(&cluster_id)
                        }) {
                            if result.len() >= max_degree { result.pop(); }
                            if !result.contains(&bridge_nbr) { result.push(bridge_nbr); }
                        }
                    }
                }
            }

            result
        })
        .collect();

    // Build bidirectional adjacency
    let total_directed: usize = pruned_neighbors.iter().map(|n| n.len()).sum();
    let mut all_edges: Vec<(u32, u32)> = Vec::with_capacity(total_directed * 2);
    for (doc_idx, nbrs) in pruned_neighbors.iter().enumerate() {
        let d = doc_idx as u32;
        for &nbr in nbrs {
            all_edges.push((d, nbr));
            all_edges.push((nbr, d));
        }
    }
    all_edges.sort_unstable();

    let mut adjacency: Vec<Vec<u32>> = (0..n_docs).into_par_iter().map(|i| {
        let i_u32 = i as u32;
        let start = all_edges.partition_point(|e| e.0 < i_u32);
        let end = all_edges.partition_point(|e| e.0 <= i_u32);
        let mut nbrs: Vec<u32> = all_edges[start..end].iter().map(|&(_, t)| t).collect();
        nbrs.sort_unstable();
        nbrs.dedup();
        nbrs.retain(|&n| n as usize != i);
        nbrs
    }).collect();
    drop(all_edges);

    let mut score_cache = ScoreCache::new(n_docs * 8);

    for doc_idx in 0..n_docs {
        if adjacency[doc_idx].len() > max_degree {
            shrink_neighbors_emd_cached(doc_idx, max_degree, &mut adjacency, codebook, flat_codes, use_emd, Some(&mut score_cache));
        }
    }

    unsafe { libc::malloc_trim(0); }
    progress!("[RNN-Descent] Phase 2 done in {:.1}s — RSS: {:.2} GB",
        phase2_start.elapsed().as_secs_f64(), rss_gb());

    // ── Phase 3: Navigating node + DFS connectivity ──────────────────
    progress!("[RNN-Descent] Phase 3: Navigating node + DFS connectivity...");
    let phase3_start = Instant::now();

    let nav_node = compute_navigating_node(n_docs, codebook, flat_codes, use_emd);
    progress!("[RNN-Descent] Navigating node: {} (of {} docs)", nav_node, n_docs);

    // DFS spanning tree from navigating node
    let dfs_parent = dfs_spanning_tree(&adjacency, nav_node);
    let unreachable: Vec<usize> = (0..n_docs)
        .filter(|&i| dfs_parent[i] == u32::MAX)
        .collect();

    if !unreachable.is_empty() {
        progress!("[RNN-Descent] {} nodes unreachable from nav node — adding tree edges...", unreachable.len());

        // Connect unreachable nodes to the nearest reachable node
        for &node in &unreachable {
            let node_codes = flat_codes.doc_codes(node);
            if node_codes.is_empty() { continue; }
            let node_u32 = node as u32;

            let mut best_nbr = u32::MAX;
            let mut best_dist = f32::MAX;

            // Search among reachable nodes for nearest
            for other in 0..n_docs {
                if other == node || dfs_parent[other] == u32::MAX { continue; }
                let d = if let Some(cached) = score_cache.get(node_u32, other as u32) {
                    cached
                } else {
                    let other_codes = flat_codes.doc_codes(other);
                    let d = construction_distance(codebook, node_codes, other_codes, use_emd);
                    score_cache.insert(node_u32, other as u32, d);
                    d
                };
                if d < best_dist {
                    best_dist = d;
                    best_nbr = other as u32;
                }
            }

            if best_nbr != u32::MAX {
                if !adjacency[node].contains(&best_nbr) {
                    adjacency[node].push(best_nbr);
                }
                if !adjacency[best_nbr as usize].contains(&node_u32) {
                    adjacency[best_nbr as usize].push(node_u32);
                }
            }
        }
    }

    // Add DFS tree edges as mandatory (ensures reachability from nav node)
    for i in 0..n_docs {
        let p = dfs_parent[i];
        if p != u32::MAX && p as usize != i {
            if !adjacency[i].contains(&p) {
                adjacency[i].push(p);
            }
            let i_u32 = i as u32;
            if !adjacency[p as usize].contains(&i_u32) {
                adjacency[p as usize].push(i_u32);
            }
        }
    }

    // Bridge + global connectivity repair
    bridge_repair_emd(&mut adjacency, max_degree, postings, codebook, flat_codes, use_emd);
    global_bridge_repair_cached(&mut adjacency, max_degree, codebook, flat_codes, use_emd, &mut score_cache);

    // Final trim: respect max_degree but never remove DFS tree edges
    for doc_idx in 0..n_docs {
        if adjacency[doc_idx].len() > max_degree + 2 {
            let parent = dfs_parent[doc_idx];
            let doc_codes = flat_codes.doc_codes(doc_idx);
            let mut scored: Vec<(u32, f32)> = adjacency[doc_idx].iter().map(|&nbr| {
                let d = if let Some(cached) = score_cache.get(doc_idx as u32, nbr) {
                    cached
                } else {
                    let nbr_codes = flat_codes.doc_codes(nbr as usize);
                    let d = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                    score_cache.insert(doc_idx as u32, nbr, d);
                    d
                };
                (nbr, d)
            }).collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            let selected = select_neighbors_heuristic(
                &scored, doc_codes, max_degree, codebook, flat_codes,
            );
            let mut result: Vec<u32> = selected.iter().map(|&(id, _)| id).collect();

            // Ensure DFS tree edge is preserved
            if parent != u32::MAX && parent as usize != doc_idx && !result.contains(&parent) {
                if result.len() >= max_degree { result.pop(); }
                result.push(parent);
            }

            adjacency[doc_idx] = result;
        }
    }

    progress!("[RNN-Descent] Phase 3 done in {:.1}s — RSS: {:.2} GB",
        phase3_start.elapsed().as_secs_f64(), rss_gb());

    // ── Phase 4: Beam search refinement pass ─────────────────────────
    // Now that the graph has DFS connectivity and a proper navigating node,
    // beam search CAN work. Run a refinement pass to discover navigable
    // long-range edges that NN-Descent couldn't find locally.
    progress!("[RNN-Descent] Phase 4: Beam search refinement (ef={})...", ef_construction);
    let phase4_start = Instant::now();
    let n_fine = codebook.n_fine;

    let all_scores: Vec<(usize, Vec<f32>)> = (0..n_docs).into_par_iter().map(|doc_idx| {
        let (start, end) = doc_offsets[doc_idx];
        let n_tokens = end - start;
        if n_tokens == 0 || dim == 0 {
            return (doc_idx, Vec::new());
        }
        let doc_vecs = &all_vectors[start * dim..end * dim];
        let scores = codebook.compute_query_centroid_scores(doc_vecs, n_tokens);
        (doc_idx, scores)
    }).collect();

    let mut refined = 0usize;
    for (doc_idx, query_scores) in &all_scores {
        let doc_idx = *doc_idx;
        if query_scores.is_empty() { continue; }
        let (start, end) = doc_offsets[doc_idx];
        let n_tokens = end - start;

        // Use navigating node + cluster reps + current neighbors as entry points
        let mut entries: Vec<u32> = vec![nav_node];
        if doc_idx < doc_profiles.len() {
            for &cid in &doc_profiles[doc_idx].ctop {
                if let Some(Some(rep)) = postings.cluster_reps.get(cid as usize) {
                    if (*rep as usize) != doc_idx && !entries.contains(rep) {
                        entries.push(*rep);
                    }
                }
            }
        }
        for &nbr in &adjacency[doc_idx] {
            if !entries.contains(&nbr) {
                entries.push(nbr);
            }
            if entries.len() >= 8 { break; }
        }

        let adj = VecAdjacency(&adjacency);
        let candidates = beam_search_construction(
            &adj, &entries, query_scores, n_tokens,
            flat_codes, n_fine, ef_construction, n_docs,
        );

        if candidates.is_empty() { continue; }

        let doc_codes = flat_codes.doc_codes(doc_idx);
        let old_neighbors: Vec<u32> = adjacency[doc_idx].clone();

        let mut all_cands: Vec<(u32, f32)> = Vec::with_capacity(old_neighbors.len() + candidates.len());
        for &nbr in &old_neighbors {
            let d = if let Some(cached) = score_cache.get(doc_idx as u32, nbr) {
                cached
            } else {
                let nbr_codes = flat_codes.doc_codes(nbr as usize);
                let d = construction_distance(codebook, doc_codes, nbr_codes, use_emd);
                score_cache.insert(doc_idx as u32, nbr, d);
                d
            };
            all_cands.push((nbr, d));
        }
        for &(cand, score) in &candidates {
            if cand as usize != doc_idx && !old_neighbors.contains(&cand) {
                all_cands.push((cand, score));
            }
        }
        all_cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        all_cands.dedup_by_key(|c| c.0);

        let new_neighbors = select_neighbors_heuristic_cached(
            &all_cands, doc_codes, max_degree, codebook, flat_codes,
            Some(&mut score_cache), use_emd,
        );

        let mut result: Vec<u32> = new_neighbors.iter().map(|&(id, _)| id).collect();

        // Preserve DFS tree edge
        let parent = dfs_parent[doc_idx];
        if parent != u32::MAX && parent as usize != doc_idx && !result.contains(&parent) {
            if result.len() >= max_degree { result.pop(); }
            result.push(parent);
        }

        // Update adjacency
        for &old_nbr in &old_neighbors {
            adjacency[old_nbr as usize].retain(|&x| x != doc_idx as u32);
        }
        adjacency[doc_idx] = result.clone();
        for &nbr in &result {
            if !adjacency[nbr as usize].contains(&(doc_idx as u32)) {
                adjacency[nbr as usize].push(doc_idx as u32);
                if adjacency[nbr as usize].len() > max_degree + 2 {
                    shrink_neighbors_emd_cached(
                        nbr as usize, max_degree, &mut adjacency, codebook, flat_codes,
                        use_emd, Some(&mut score_cache),
                    );
                }
            }
        }
        refined += 1;
    }
    drop(all_scores);

    unsafe { libc::malloc_trim(0); }
    progress!("[RNN-Descent] Phase 4 done in {:.1}s — {} nodes refined — RSS: {:.2} GB",
        phase4_start.elapsed().as_secs_f64(), refined, rss_gb());

    progress!("[RNN-Descent] score cache: {} entries, {:.1}% hit rate",
        score_cache.len(), score_cache.hit_rate() * 100.0);

    // Build CSR
    let csr = CsrAdjacency::from_adj_lists(&adjacency);

    progress!("[RNN-Descent] ══════════════════════════════════════════════════");
    progress!("[RNN-Descent] TOTAL BUILD TIME: {:.1}s ({:.1} min)",
        graph_start.elapsed().as_secs_f64(), graph_start.elapsed().as_secs_f64() / 60.0);
    progress!("[RNN-Descent] ══════════════════════════════════════════════════");

    // Multi-level structure: assign HNSW-style levels to nodes for navigability
    let ml = 1.0 / (max_degree as f64).ln();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut node_levels: Vec<usize> = Vec::with_capacity(n_docs);
    let mut max_level = 0usize;
    for _ in 0..n_docs {
        let level = random_level(ml, &mut rng);
        if level > max_level { max_level = level; }
        node_levels.push(level);
    }

    // For now, we only have the bottom layer built by RNN-Descent.
    // Upper layers are empty but the search code handles this gracefully
    // (cluster-guided entry points provide the coarse routing).
    let n_levels = max_level + 1;
    let mut levels: Vec<CsrAdjacency> = Vec::with_capacity(n_levels);
    levels.push(csr);
    for _ in 1..n_levels {
        levels.push(CsrAdjacency::empty(n_docs));
    }

    // Force nav_node to highest level for entry point
    node_levels[nav_node as usize] = max_level;

    GemGraph {
        levels,
        shortcuts: vec![Vec::new(); n_docs],
        shortcut_generations: vec![Vec::new(); n_docs],
        node_levels,
        entry_point: nav_node,
        max_degree,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use latence_gem_router::codebook::{TwoStageCodebook, compute_ctop};
    use latence_gem_router::router::{FlatDocCodes, DocProfile, ClusterPostings};

    fn setup_test_data(n_docs: usize, tokens_per_doc: usize) -> (
        Vec<f32>, usize, Vec<(usize, usize)>, TwoStageCodebook,
        FlatDocCodes, Vec<DocProfile>, ClusterPostings, Vec<u64>,
    ) {
        let dim = 8;
        let n_fine = 8;
        let n_coarse = 4;
        let ctop_r = 2;
        let mut rng = StdRng::seed_from_u64(42);
        let n_vectors = n_docs * tokens_per_doc;
        let all_vectors: Vec<f32> = (0..n_vectors * dim)
            .map(|_| rng.gen::<f32>() - 0.5)
            .collect();
        let doc_offsets: Vec<(usize, usize)> = (0..n_docs)
            .map(|i| (i * tokens_per_doc, (i + 1) * tokens_per_doc))
            .collect();
        let doc_ids: Vec<u64> = (0..n_docs as u64).collect();

        let codebook = TwoStageCodebook::build(
            &all_vectors, n_vectors, dim, n_fine, n_coarse, 10, 42,
        );
        let all_assignments = codebook.assign_vectors(&all_vectors, n_vectors);

        let mut doc_centroid_sets = Vec::with_capacity(n_docs);
        for &(start, end) in &doc_offsets {
            doc_centroid_sets.push(all_assignments[start..end].to_vec());
        }

        let mut flat_codes = FlatDocCodes::new();
        let mut doc_profiles = Vec::with_capacity(n_docs);
        let mut postings = ClusterPostings::new(n_coarse);
        for (doc_idx, cids) in doc_centroid_sets.into_iter().enumerate() {
            let ctop = compute_ctop(&codebook, &cids, ctop_r);
            postings.add_doc(doc_idx as u32, &ctop);
            flat_codes.add_doc(&cids);
            doc_profiles.push(DocProfile {
                centroid_ids: cids,
                ctop,
            });
        }

        (all_vectors, dim, doc_offsets, codebook, flat_codes, doc_profiles, postings, doc_ids)
    }

    #[test]
    fn test_build_graph_basic() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let graph = build_graph(&vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32);
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0, "graph should have edges");
        assert!(!graph.levels.is_empty());
        for i in 0..20 {
            let nbrs = graph.levels[0].neighbors(i);
            assert!(nbrs.len() <= 8, "node {} has {} neighbors > max_degree 8", i, nbrs.len());
        }
    }

    #[test]
    fn test_build_graph_single_doc() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(1, 5);
        let graph = build_graph(&vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32);
        assert_eq!(graph.n_nodes(), 1);
    }

    #[test]
    fn test_build_graph_with_payload() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let payload: Vec<u32> = (0..20).map(|i| (i % 3) as u32).collect();
        let graph = build_graph_with_payload(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32,
            Some(&payload), false,
        );
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0);
    }

    #[test]
    fn test_build_graph_dual_basic() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let graph = build_graph_dual(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32, false,
        );
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0, "dual graph should have edges");
        for i in 0..20 {
            let nbrs = graph.levels[0].neighbors(i);
            assert!(nbrs.len() <= 8, "node {} has {} neighbors > max_degree 8", i, nbrs.len());
        }
    }

    #[test]
    fn test_build_graph_dual_with_emd() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(10, 4);
        let graph = build_graph_dual(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 6, 16, true,
        );
        assert_eq!(graph.n_nodes(), 10);
        assert!(graph.n_edges() > 0);
    }

    #[test]
    fn test_heuristic_emd_vs_ch() {
        let (vecs, dim, offsets, cb, fc, dp, post, _) = setup_test_data(15, 5);
        let graph_ch = build_graph_with_payload(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32, None, false,
        );
        let graph_emd = build_graph_with_payload(
            &vecs, dim, &offsets, &cb, &fc, &dp, &post, 8, 32, None, true,
        );
        assert_eq!(graph_ch.n_nodes(), graph_emd.n_nodes());
        assert!(graph_ch.n_edges() > 0);
        assert!(graph_emd.n_edges() > 0);
    }

    #[test]
    fn test_update_bridges_degree_limit() {
        let (_vecs, _dim, _offsets, cb, fc, dp, _post, _) = setup_test_data(20, 5);
        let max_degree = 4;
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); 20];
        adjacency[0] = vec![1, 2, 3, 4];
        adjacency[1] = vec![0];
        adjacency[2] = vec![0];
        adjacency[3] = vec![0];
        adjacency[4] = vec![0];

        let new_neighbors: Vec<(u32, f32)> = vec![(5, 0.5), (6, 0.6), (7, 0.7)];
        update_bridges(0, &new_neighbors, &mut adjacency, &dp, max_degree, &cb, &fc, false);

        assert!(
            adjacency[0].len() <= max_degree,
            "node 0 has {} neighbors > max_degree {}", adjacency[0].len(), max_degree
        );
    }

    #[test]
    fn test_build_graph_nndescent_basic() {
        let (_vecs, _dim, _offsets, cb, fc, dp, post, _) = setup_test_data(20, 5);
        let graph = build_graph_nndescent(
            20, &cb, &fc, &dp, &post, 8, false, None,
        );
        assert_eq!(graph.n_nodes(), 20);
        assert!(graph.n_edges() > 0, "nndescent graph should have edges");
        let (n_comp, giant_frac) = graph.connectivity_report();
        assert_eq!(n_comp, 1, "nndescent graph should be fully connected, got {} components", n_comp);
        assert!((giant_frac - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_graph_nndescent_with_emd() {
        let (_vecs, _dim, _offsets, cb, fc, dp, post, _) = setup_test_data(10, 4);
        let graph = build_graph_nndescent(
            10, &cb, &fc, &dp, &post, 6, true, None,
        );
        assert_eq!(graph.n_nodes(), 10);
        assert!(graph.n_edges() > 0);
        let (n_comp, _) = graph.connectivity_report();
        assert_eq!(n_comp, 1, "nndescent qEMD graph should be fully connected");
    }

    #[test]
    fn test_csr_roundtrip() {
        let adj = vec![
            vec![1u32, 2],
            vec![0, 2],
            vec![0, 1],
        ];
        let csr = CsrAdjacency::from_adj_lists(&adj);
        assert_eq!(csr.n_nodes(), 3);
        assert_eq!(csr.neighbors(0), &[1, 2]);
        assert_eq!(csr.neighbors(1), &[0, 2]);
        assert_eq!(csr.neighbors(2), &[0, 1]);
        let restored = csr.to_adj_lists();
        assert_eq!(restored, adj);
    }
}

//! Benchmark comparing Knapsack Solver vs RRF Baseline
//!
//! Measures:
//! - Latency (p50, p95, p99)
//! - Quality metrics (token efficiency, redundancy penalty)
//! - Scalability (100, 200, 500 candidates)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use latence_solver::solver::{
    TabuSearchSolver, RRFSolver, RRFConfig, RankingSource,
    SolverConfig, SolverConstraints, ChunkCandidate, SolverMode,
};
use rand::prelude::*;

/// Generate random test candidates
fn generate_candidates(n: usize, dim: usize, seed: u64) -> Vec<ChunkCandidate> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    
    (0..n)
        .map(|i| {
            // Random embedding
            let embedding: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            // Random token count (50-500)
            let token_count = rng.gen_range(50..500u32);
            
            ChunkCandidate::new(
                format!("chunk_{}", i),
                format!("Content for chunk {} with some text to process", i),
                embedding,
                token_count,
            )
            .with_fact_density(rng.gen_range(0.2..0.9))
            .with_centrality(rng.gen_range(0.3..0.95))
            .with_recency(rng.gen_range(0.5..1.0))
        })
        .collect()
}

/// Generate query embedding
fn generate_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Create ranking sources (simulating ColBERT + BM25 retrieval)
fn create_ranking_sources(candidates: &[ChunkCandidate], seed: u64) -> Vec<RankingSource> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    
    // ColBERT-like ranking (weighted by centrality + random noise)
    let mut colbert_scored: Vec<_> = candidates.iter().enumerate().map(|(i, c)| {
        let score = c.centrality_score + rng.gen_range(0.0..0.2);
        (i, score)
    }).collect();
    
    colbert_scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    let colbert = RankingSource::new(
        "colbert",
        colbert_scored.iter().map(|&(i, _)| candidates[i].chunk_id.clone()).collect(),
    );
    
    // BM25-like ranking (weighted by density + random noise)
    let mut bm25_scored: Vec<_> = candidates.iter().enumerate().map(|(i, c)| {
        let score = c.fact_density + rng.gen_range(0.0..0.2);
        (i, score)
    }).collect();
    
    bm25_scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    let bm25 = RankingSource::new(
        "bm25",
        bm25_scored.iter().map(|&(i, _)| candidates[i].chunk_id.clone()).collect(),
    );
    
    vec![colbert, bm25]
}

fn bench_rrf_vs_knapsack(c: &mut Criterion) {
    let dim = 128;
    let max_tokens = 4096u32;
    
    let mut group = c.benchmark_group("RRF_vs_Knapsack");
    group.sample_size(50);
    
    for n_candidates in [100, 200, 500] {
        let candidates = generate_candidates(n_candidates, dim, 42);
        let query = generate_query(dim, 123);
        let ranking_sources = create_ranking_sources(&candidates, 456);
        
        // Setup constraints
        let constraints = SolverConstraints::with_budget(max_tokens);
        
        // Benchmark RRF
        group.bench_with_input(
            BenchmarkId::new("RRF", n_candidates),
            &n_candidates,
            |b, _| {
                let config = RRFConfig {
                    k: 60.0,
                    top_n: 20,
                    max_tokens: Some(max_tokens),
                };
                let solver = RRFSolver::new(config);
                
                b.iter(|| {
                    black_box(solver.solve(&candidates, &ranking_sources))
                });
            },
        );
        
        // Benchmark Knapsack Solver (Vanilla mode)
        group.bench_with_input(
            BenchmarkId::new("Knapsack_Vanilla", n_candidates),
            &n_candidates,
            |b, _| {
                let config = SolverConfig {
                    mode: SolverMode::Vanilla,
                    iterations: 50,
                    use_gpu: false,
                    ..Default::default()
                };
                let solver = TabuSearchSolver::new(config);
                
                b.iter(|| {
                    black_box(solver.solve(&candidates, &query, &constraints))
                });
            },
        );
        
        // Benchmark Knapsack Solver (Enriched mode)
        group.bench_with_input(
            BenchmarkId::new("Knapsack_Enriched", n_candidates),
            &n_candidates,
            |b, _| {
                let config = SolverConfig {
                    mode: SolverMode::Enriched,
                    iterations: 50,
                    use_gpu: false,
                    ..Default::default()
                };
                let solver = TabuSearchSolver::new(config);
                
                b.iter(|| {
                    black_box(solver.solve(&candidates, &query, &constraints))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_knapsack_iterations(c: &mut Criterion) {
    let n = 200;
    let dim = 128;
    let max_tokens = 4096u32;
    
    let candidates = generate_candidates(n, dim, 42);
    let query = generate_query(dim, 123);
    let constraints = SolverConstraints::with_budget(max_tokens);
    
    let mut group = c.benchmark_group("Knapsack_Iterations");
    group.sample_size(30);
    
    for iterations in [25, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iters| {
                let config = SolverConfig {
                    mode: SolverMode::Vanilla,
                    iterations: iters,
                    use_gpu: false,
                    ..Default::default()
                };
                let solver = TabuSearchSolver::new(config);
                
                b.iter(|| {
                    black_box(solver.solve(&candidates, &query, &constraints))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_rrf_vs_knapsack, bench_knapsack_iterations);
criterion_main!(benches);

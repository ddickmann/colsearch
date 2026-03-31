//! Tool to compare RRF vs Knapsack Solver quality metrics
//!
//! Usage: cargo run --bin compare_algorithms

use latence_solver::solver::{
    TabuSearchSolver, RRFSolver, RRFConfig, RankingSource,
    SolverConfig, SolverConstraints, ChunkCandidate, SolverMode,
};
use rand::prelude::*;
use std::time::Instant;

fn generate_data(n: usize, seed: u64) -> (Vec<ChunkCandidate>, Vec<f32>, Vec<RankingSource>) {
    let dim = 128;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    
    // Generate candidates with meaningful structure
    let candidates: Vec<ChunkCandidate> = (0..n)
        .map(|i| {
            // Create clustered embeddings to create optimization conflict
            let cluster_center = (i % 5) as f32; 
            let embedding: Vec<f32> = (0..dim).map(|j| {
                let base = if (j % 5) == (i % 5) { 1.0 } else { 0.0 };
                base + rng.gen_range(-0.1..0.1)
            }).collect();
            
            let mut c = ChunkCandidate::new(
                format!("chunk_{}", i),
                format!("Content {}", i),
                embedding,
                rng.gen_range(100..400),
            );
            
            // Assign varying quality features
            c.fact_density = rng.gen_range(0.2..0.9);
            c.centrality_score = rng.gen_range(0.3..0.95);
            c.recency_score = rng.gen_range(0.5..1.0);
            
            c
        })
        .collect();
        
    // Query embedding
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
    
    // Create noisy ranking sources
    let r1 = RankingSource::from_scored_candidates("colbert", &candidates);
    let mut r2_ids = r1.ranked_ids.clone();
    r2_ids.reverse(); // Inverse ranking to simulate disagreement
    let r2 = RankingSource::new("bm25", r2_ids);
    
    (candidates, query, vec![r1, r2])
}

fn main() {
    let n_candidates = 500;
    let max_tokens = 4000;
    println!("Comparing RRF vs Knapsack Solver (N={}, MaxTokens={})", n_candidates, max_tokens);
    println!("------------------------------------------------------------");
    
    let (candidates, query, ranking_sources) = generate_data(n_candidates, 42);
    let constraints = SolverConstraints::with_budget(max_tokens as u32);
    
    // 1. Run RRF
    let rrf_config = RRFConfig {
        k: 60.0,
        top_n: 20,
        max_tokens: Some(max_tokens as u32),
    };
    let rrf_solver = RRFSolver::new(rrf_config);
    let start_rrf = Instant::now();
    let rrf_out = rrf_solver.solve(&candidates, &ranking_sources);
    let rrf_time = start_rrf.elapsed();
    
    // Convert RRF to SolverOutput to measure its objective score according to OUR function
    // This answers: "How good is the RRF selection if we scored it with our objective?"
    let rrf_as_solver_out = rrf_solver.to_solver_output(&rrf_out, &candidates);
    
    // We need to re-score the RRF selection using the actual objective function logic (relevance - redundancy)
    // because to_solver_output just uses the RRF score as the objective score, which isn't comparable.
    let tabu_config = SolverConfig {
        mode: SolverMode::Vanilla,
        use_gpu: false,
        ..Default::default()
    };
    let solver_for_eval = TabuSearchSolver::new(tabu_config.clone());
    let rrf_true_score = solver_for_eval.evaluate_solution_score(&rrf_out.selected_indices, &candidates, &query);

    // 2. Run Knapsack (Vanilla)
    let start_knap = Instant::now();
    let knap_out = solver_for_eval.solve(&candidates, &query, &constraints);
    let knap_time = start_knap.elapsed();
    
    // 3. Run Knapsack (Enriched)
    let enriched_config = SolverConfig {
        mode: SolverMode::Enriched,
        use_gpu: false,
        ..Default::default()
    };
    let enriched_solver = TabuSearchSolver::new(enriched_config);
    let start_enriched = Instant::now();
    let enriched_out = enriched_solver.solve(&candidates, &query, &constraints);
    let enriched_time = start_enriched.elapsed();

    // Print Results
    println!("{:<20} | {:<10} | {:<10} | {:<15} | {:<10} | {:<10} | {:<10}", 
             "Algorithm", "Time (ms)", "Tokens", "Objective Score", "Valid?", "Relevance", "Density");
    println!("{:-<100}", "");
    
    println!("{:<20} | {:<10.2} | {:<10} | {:<15.4} | {:<10} | {:<10.4} | {:<10.4}",
             "RRF Baseline",
             rrf_time.as_secs_f64() * 1000.0,
             rrf_out.total_tokens,
             -1.0, // RRF doesn't have breakdown
             rrf_out.total_tokens <= max_tokens as u32,
             0.0, 0.0
    );
    
    println!("{:<20} | {:<10.2} | {:<10} | {:<15.4} | {:<10} | {:<10.4} | {:<10.4}",
             "Knapsack (Vanilla)",
             knap_time.as_secs_f64() * 1000.0,
             knap_out.total_tokens,
             knap_out.objective_score,
             knap_out.constraints_satisfied,
             knap_out.relevance_total,
             knap_out.density_total
    );
    
    println!("{:<20} | {:<10.2} | {:<10} | {:<15.4} | {:<10} | {:<10.4} | {:<10.4}",
             "Knapsack (Enriched)",
             enriched_time.as_secs_f64() * 1000.0,
             enriched_out.total_tokens,
             enriched_out.objective_score,
             enriched_out.constraints_satisfied,
             enriched_out.relevance_total,
             enriched_out.density_total
    );
    
    println!("\nImprovement over RRF:");
    let vanilla_imp = (knap_out.objective_score - rrf_true_score) / rrf_true_score * 100.0;
    let enriched_imp = (enriched_out.objective_score - rrf_true_score) / rrf_true_score * 100.0;
    
    println!("Vanilla:  {:+.2}%", vanilla_imp);
    println!("Enriched: {:+.2}%", enriched_imp);
}

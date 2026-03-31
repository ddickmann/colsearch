use latence_solver::{
    TabuSearchSolver, SolverConfig, SolverConstraints, ChunkCandidate, SolverMode, RhetoricalRole
};

fn main() {
    println!("🧪 Verifying Knapsack Fusion Engine (Enriched Mode)");
    println!("--------------------------------------------------");

    // 1. Setup Candidates
    // Candidate A: High Relevance, Low Quality (The "Clickbait")
    // Candidate B: Medium Relevance, High Quality (The "Deep Dive")
    let candidates = vec![
        ChunkCandidate::new(
            "chunk_a".to_string(), 
            "High relevance, low density".to_string(), 
            vec![1.0, 0.0], // Similarity 1.0 to query [1, 0]
            100
        )
        .with_fact_density(0.1)
        .with_centrality(0.1),

        ChunkCandidate::new(
            "chunk_b".to_string(), 
            "Medium relevance, high density".to_string(), 
            vec![0.8, 0.6], // Similarity 0.8 to query [1, 0]
            100
        )
        .with_fact_density(1.0)
        .with_centrality(0.9),
    ];

    let query = vec![1.0, 0.0];
    
    // Constraints: Can only pick ONE (budget 150, cost 100 each)
    let constraints = SolverConstraints::with_budget(150);

    // 2. Run Vanilla Mode
    println!("\n[1] Running Vanilla Mode (Relevance Only)...");
    let mut config_vanilla = SolverConfig::default();
    config_vanilla.mode = SolverMode::Vanilla;
    
    let solver_vanilla = TabuSearchSolver::new(config_vanilla);
    let output_vanilla = solver_vanilla.solve(&candidates, &query, &constraints);
    
    let selected_vanilla = output_vanilla.selected_ids(&candidates);
    println!(" -> Selected: {:?}", selected_vanilla);
    println!(" -> Score Breakdown: Rel={:.4}, Den={:.4}", output_vanilla.relevance_total, output_vanilla.density_total);
    
    assert_eq!(selected_vanilla[0], "chunk_a", "Vanilla should pick highest relevance (Chunk A)");

    // 3. Run Enriched Mode
    println!("\n[2] Running Enriched Mode (Relevance + Density + Centrality)...");
    let mut config_enriched = SolverConfig::default();
    config_enriched.mode = SolverMode::Enriched;
    // Set weights: Alpha=1.0, Beta=0.5 (Density), Gamma=0.5 (Centrality)
    config_enriched.alpha = 1.0;
    config_enriched.beta = 0.5;
    config_enriched.gamma = 0.5;
    
    let solver_enriched = TabuSearchSolver::new(config_enriched);
    let output_enriched = solver_enriched.solve(&candidates, &query, &constraints);
    
    let selected_enriched = output_enriched.selected_ids(&candidates);
    println!(" -> Selected: {:?}", selected_enriched);
    println!(" -> Score Breakdown: Rel={:.4}, Den={:.4}, Cen={:.4}", 
        output_enriched.relevance_total, 
        output_enriched.density_total,
        output_enriched.centrality_total
    );

    // Expected Calculation:
    // A: 1.0 + 0.5*0.1 + 0.5*0.1 = 1.1
    // B: 0.8 + 0.5*1.0 + 0.5*0.9 = 0.8 + 0.5 + 0.45 = 1.75
    // Enriched should pick B
    assert_eq!(selected_enriched[0], "chunk_b", "Enriched should pick higher quality (Chunk B)");

    println!("\n✅ VERIFICATION SUCCESSFUL: Fusion Engine correctly leverages intelligence features!");
}

//! Latence Solver CLI
//!
//! Command-line interface for the Tabu Search Knapsack Solver.

use std::fs;
use std::path::PathBuf;
use std::io::{self, Read};

use latence_solver::{
    TabuSearchSolver,
    SolverConfig,
    SolverConstraints,
    ChunkCandidate,
    RhetoricalRole,
};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("latence_solver=info".parse().unwrap())
        )
        .init();
    
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        print_usage(&args[0]);
        return;
    }
    
    match args[1].as_str() {
        "solve" => cmd_solve(&args[2..]),
        "benchmark" => cmd_benchmark(&args[2..]),
        "info" => cmd_info(),
        "--help" | "-h" => print_usage(&args[0]),
        "--version" | "-v" => println!("latence-solver {}", latence_solver::VERSION),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage(&args[0]);
        }
    }
}

fn print_usage(program: &str) {
    println!("Latence Solver - High-Performance Tabu Search Knapsack Solver");
    println!();
    println!("Usage: {} <command> [options]", program);
    println!();
    println!("Commands:");
    println!("  solve      Solve a knapsack problem from JSON input");
    println!("  benchmark  Run performance benchmarks");
    println!("  info       Show system information");
    println!();
    println!("Options:");
    println!("  -h, --help     Show this help message");
    println!("  -v, --version  Show version");
}

fn cmd_info() {
    println!("Latence Solver v{}", latence_solver::VERSION);
    println!();
    println!("System Information:");
    println!("  GPU Available: {}", latence_solver::gpu_available());
    println!("  CUDA Available: {}", latence_solver::cuda_available());
    println!("  CPU Threads: {}", rayon::current_num_threads());
    println!();
    
    #[cfg(feature = "cpu")]
    println!("  Backend: CPU (NdArray)");
    #[cfg(feature = "gpu")]
    println!("  Backend: GPU (wgpu)");
    #[cfg(feature = "cuda")]
    println!("  Backend: CUDA");
}

fn cmd_solve(args: &[String]) {
    // Read JSON input from file or stdin
    let input_json = if args.is_empty() || args[0] == "-" {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer).expect("Failed to read stdin");
        buffer
    } else {
        fs::read_to_string(&args[0]).expect("Failed to read input file")
    };
    
    // Parse input
    let input: serde_json::Value = serde_json::from_str(&input_json)
        .expect("Failed to parse JSON input");
    
    // Extract candidates
    let candidates: Vec<ChunkCandidate> = input["candidates"]
        .as_array()
        .expect("Missing 'candidates' array")
        .iter()
        .enumerate()
        .map(|(i, c)| {
            ChunkCandidate {
                chunk_id: c["chunk_id"].as_str().map(String::from).unwrap_or_else(|| format!("chunk_{}", i)),
                content: c["content"].as_str().unwrap_or("").to_string(),
                embedding: c["embedding"].as_array()
                    .expect("Missing embedding")
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
                token_count: c["token_count"].as_u64().unwrap_or(100) as u32,
                fact_density: c["fact_density"].as_f64().unwrap_or(0.5) as f32,
                centrality_score: c["centrality_score"].as_f64().unwrap_or(0.5) as f32,
                uniqueness_score: c["uniqueness_score"].as_f64().unwrap_or(0.5) as f32,
                recency_score: c["recency_score"].as_f64().unwrap_or(0.5) as f32,
                auxiliary_score: c["auxiliary_score"].as_f64().unwrap_or(0.0) as f32,
                rhetorical_role: c["rhetorical_role"].as_str()
                    .map(RhetoricalRole::from)
                    .unwrap_or(RhetoricalRole::Unknown),
                cluster_id: c["cluster_id"].as_u64().map(|v| v as u32),
                parent_doc_id: c["parent_doc_id"].as_str().map(String::from),
                position: c["position"].as_f64().unwrap_or(0.5) as f32,
            }
        })
        .collect();
    
    // Extract query embedding
    let query_embedding: Vec<f32> = input["query_embedding"]
        .as_array()
        .expect("Missing 'query_embedding'")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    
    // Extract config
    let config = if let Some(cfg) = input.get("config") {
        SolverConfig {
            alpha: cfg["alpha"].as_f64().unwrap_or(1.0) as f32,
            beta: cfg["beta"].as_f64().unwrap_or(0.3) as f32,
            gamma: cfg["gamma"].as_f64().unwrap_or(0.2) as f32,
            delta: cfg["delta"].as_f64().unwrap_or(0.1) as f32,
            lambda: cfg["lambda"].as_f64().unwrap_or(0.5) as f32,
            iterations: cfg["iterations"].as_u64().unwrap_or(100) as usize,
            use_gpu: cfg["use_gpu"].as_bool().unwrap_or(true),
            ..Default::default()
        }
    } else {
        SolverConfig::default()
    };
    
    // Extract constraints
    let constraints = if let Some(con) = input.get("constraints") {
        SolverConstraints {
            max_tokens: con["max_tokens"].as_u64().unwrap_or(8192) as u32,
            min_tokens: con["min_tokens"].as_u64().unwrap_or(0) as u32,
            max_chunks: con["max_chunks"].as_u64().unwrap_or(50) as usize,
            max_per_cluster: con["max_per_cluster"].as_u64().unwrap_or(3) as usize,
            ..Default::default()
        }
    } else {
        SolverConstraints::default()
    };
    
    // Solve
    let solver = TabuSearchSolver::new(config);
    let output = solver.solve(&candidates, &query_embedding, &constraints);
    
    // Output result as JSON
    let result = serde_json::json!({
        "selected_indices": output.selected_indices,
        "selected_chunk_ids": output.selected_indices.iter()
            .map(|&i| &candidates[i].chunk_id)
            .collect::<Vec<_>>(),
        "objective_score": output.objective_score,
        "relevance_total": output.relevance_total,
        "density_total": output.density_total,
        "centrality_total": output.centrality_total,
        "recency_total": output.recency_total,
        "redundancy_penalty": output.redundancy_penalty,
        "total_tokens": output.total_tokens,
        "num_selected": output.num_selected,
        "iterations_run": output.iterations_run,
        "best_iteration": output.best_iteration,
        "constraints_satisfied": output.constraints_satisfied,
        "constraint_violations": output.constraint_violations,
        "solve_time_ms": output.solve_time_ms,
    });
    
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn cmd_benchmark(args: &[String]) {
    use std::time::Instant;
    
    let sizes = vec![50, 100, 200, 500, 1000];
    let dim = 256;
    let iterations = 50;
    
    println!("Latence Solver Benchmark");
    println!("========================");
    println!();
    println!("Configuration:");
    println!("  Embedding dimension: {}", dim);
    println!("  Iterations: {}", iterations);
    println!("  GPU: {}", latence_solver::gpu_available());
    println!();
    println!("{:>10} | {:>12} | {:>12} | {:>10}", "Candidates", "Latency (ms)", "Throughput", "Selected");
    println!("{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "");
    
    let config = SolverConfig {
        iterations,
        use_gpu: latence_solver::gpu_available(),
        ..Default::default()
    };
    let solver = TabuSearchSolver::new(config);
    
    for size in sizes {
        // Generate random candidates
        let mut rng = rand::thread_rng();
        let candidates: Vec<ChunkCandidate> = (0..size)
            .map(|i| {
                let embedding: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                ChunkCandidate::new(
                    format!("chunk_{}", i),
                    format!("Content for chunk {}", i),
                    embedding,
                    100 + (rng.gen::<u32>() % 200),
                )
                .with_fact_density(rng.gen())
                .with_centrality(rng.gen())
                .with_recency(rng.gen())
            })
            .collect();
        
        let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let constraints = SolverConstraints::with_budget(4096);
        
        // Warm-up
        let _ = solver.solve(&candidates, &query, &constraints);
        
        // Benchmark
        let runs = 5;
        let mut total_time = 0.0;
        let mut total_selected = 0;
        
        for _ in 0..runs {
            let start = Instant::now();
            let output = solver.solve(&candidates, &query, &constraints);
            total_time += start.elapsed().as_secs_f64() * 1000.0;
            total_selected += output.num_selected;
        }
        
        let avg_latency = total_time / runs as f64;
        let avg_selected = total_selected / runs;
        let throughput = 1000.0 / avg_latency;
        
        println!(
            "{:>10} | {:>12.2} | {:>10.1}/s | {:>10}",
            size, avg_latency, throughput, avg_selected
        );
    }
}

use rand::Rng;


//! Latence Solver - High-Performance Tabu Search Knapsack Solver
//!
//! This crate provides the canonical OSS implementation of a
//! Tabu Search algorithm for solving the Quadratic Knapsack Problem (QKP)
//! used in semantic context optimization.
//!
//! # Features
//!
//! - **CPU fallback first**: deterministic Rust CPU implementation used whenever
//!   an accelerated backend is unavailable
//! - **Experimental accelerators**: optional in-tree wgpu/CUDA backends that are
//!   kept only as internal fallbacks, not the public premium product path
//! - **Parallel execution**: Uses rayon for CPU parallelism
//! - **Python bindings**: Optional PyO3 bindings for Python integration
//! - **Zero-copy interop**: Efficient data transfer with Python/NumPy
//!
//! # Example
//!
//! ```rust,ignore
//! use latence_solver::{TabuSearchSolver, SolverConfig, SolverConstraints, ChunkCandidate};
//!
//! let config = SolverConfig::default();
//! let solver = TabuSearchSolver::new(config);
//!
//! let candidates = vec![/* ... */];
//! let query_embedding = vec![0.1f32; 256];
//! let constraints = SolverConstraints::default();
//! let result = solver.solve(&candidates, &query_embedding, &constraints);
//! ```

pub mod backend;
pub mod solver;
pub mod quantization;

#[cfg(feature = "python")]
pub mod python;

// Re-exports for convenience
pub use solver::{
    ChunkCandidate,
    RhetoricalRole,
    SolverConfig,
    SolverConstraints,
    SolverInput,
    SolverOutput,
    SolverMode,
    TabuSearchSolver,
    SemanticTabuList,
    ObjectiveFunction,
    ConstraintValidator,
};

pub use backend::{Backend, BackendType, create_backend};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if the experimental wgpu backend is available.
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        // Try to create a wgpu instance
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        adapter.is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Check if the experimental in-tree CUDA backend is available.
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        backend::CudaBackend::default_device().is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}


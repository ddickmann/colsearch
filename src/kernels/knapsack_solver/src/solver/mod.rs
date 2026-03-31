//! Tabu Search Knapsack Solver
//!
//! This module implements a GPU/CPU accelerated Tabu Search algorithm
//! for solving the Quadratic Knapsack Problem (QKP) used in semantic
//! context optimization.

mod config;
mod objective;
mod tabu;
mod constraints;
mod knapsack;
mod rrf_baseline;

pub use config::{
    SolverConfig,
    SolverConstraints,
    SolverInput,
    SolverOutput,
    ChunkCandidate,
    RhetoricalRole,
    SolverMode,
};

pub use objective::{FulfilmentState, ObjectiveFunction};
pub use tabu::SemanticTabuList;
pub use constraints::ConstraintValidator;
pub use knapsack::TabuSearchSolver;
pub use rrf_baseline::{RRFSolver, RRFConfig, RRFOutput, RankingSource};


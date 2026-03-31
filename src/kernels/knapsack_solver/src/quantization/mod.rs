// Rotational Quantization Module
//
// High-performance 8-bit rotational quantization using Fast Walsh-Hadamard Transform.
// Matches Weaviate's production approach for near-lossless 4x compression.

pub mod fwht;
pub mod rotational;

pub use fwht::FastWalshHadamard;
pub use rotational::{RotationalQuantizer, RoQConfig, QuantizedVector};

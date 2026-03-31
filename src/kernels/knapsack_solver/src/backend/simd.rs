use wide::f32x8;

/// Compute dot product of two vectors using SIMD
#[inline]
pub fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut i = 0;
    
    // Process 8 elements at a time
    while i + 8 <= a.len() {
        let a_chunk = f32x8::from(&a[i..i+8]);
        let b_chunk = f32x8::from(&b[i..i+8]);
        sum += a_chunk * b_chunk;
        i += 8;
    }
    
    let mut result = sum.reduce_add();
    
    // Handle remainder
    while i < a.len() {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// Compute cosine similarity between two vectors using SIMD
#[inline]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_sum = f32x8::splat(0.0);
    let mut norm_a_sum = f32x8::splat(0.0);
    let mut norm_b_sum = f32x8::splat(0.0);
    
    let mut i = 0;
    
    // Process 8 elements at a time (AVX2 width)
    while i + 8 <= a.len() {
        let a_chunk = f32x8::from(&a[i..i+8]);
        let b_chunk = f32x8::from(&b[i..i+8]);
        
        dot_sum += a_chunk * b_chunk;
        norm_a_sum += a_chunk * a_chunk;
        norm_b_sum += b_chunk * b_chunk;
        
        i += 8;
    }
    
    let dot = dot_sum.reduce_add();
    let norm_a = norm_a_sum.reduce_add();
    let norm_b = norm_b_sum.reduce_add();
    
    // Handle remainder
    let mut tail_dot = 0.0;
    let mut tail_norm_a = 0.0;
    let mut tail_norm_b = 0.0;
    
    while i < a.len() {
        tail_dot += a[i] * b[i];
        tail_norm_a += a[i] * a[i];
        tail_norm_b += b[i] * b[i];
        i += 1;
    }
    
    let final_dot = dot + tail_dot;
    let final_norm_a = (norm_a + tail_norm_a).sqrt();
    let final_norm_b = (norm_b + tail_norm_b).sqrt();
    
    if final_norm_a > 1e-6 && final_norm_b > 1e-6 {
        final_dot / (final_norm_a * final_norm_b)
    } else {
        0.0
    }
}

/// Compute cosine similarity matrix using SIMD kernels
/// This avoids the overhead of creating extensive ndarray views
pub fn cosine_similarity_matrix_simd(embeddings: &[f32], n: usize, dim: usize) -> Vec<f32> {
    use rayon::prelude::*;
    
    // Pre-normalize embeddings to simplify pairwise calcs
    // Normalized vectors: dot(u, v) == cosine_similarity(u, v)
    let pre_normalized: Vec<f32> = embeddings
        .par_chunks(dim)
        .flat_map(|chunk| {
            let mut sum_sq = f32x8::splat(0.0);
            let mut i = 0;
            while i + 8 <= chunk.len() {
                let v = f32x8::from(&chunk[i..i+8]);
                sum_sq += v * v;
                i += 8;
            }
            let mut norm = sum_sq.reduce_add();
            while i < chunk.len() {
                norm += chunk[i] * chunk[i];
                i += 1;
            }
            
            let inv_norm = if norm > 1e-6 { 1.0 / norm.sqrt() } else { 0.0 };
            
            // Return normalized vector
            chunk.iter().map(|&x| x * inv_norm).collect::<Vec<f32>>()
        })
        .collect();

    let mut similarity = vec![0.0f32; n * n];
    
    // Parallelize outer loop (rows)
    // We only need to compute upper triangle + diagonal, then mirror
    // BUT mirroring causes cache contention writes.
    // Better to compute full matrix or block it. 
    // For simplicity with Rayon: Just compute full matrix row by row.
    // Vectorized dot product of normalized vectors is fast.
    
    // Parallelize over rows with coarser granularity to reduce Raon overhead
    // processing 500 rows individually is too much overhead.
    // Chunk size of 64 lines implies ~8 tasks for N=500.
    const CHUNK_SIZE: usize = 64;
    
    similarity.par_chunks_mut(n * CHUNK_SIZE).enumerate().for_each(|(chunk_idx, chunk)| {
        let start_row = chunk_idx * CHUNK_SIZE;
        
        // Iterate over rows in this chunk
        for i in 0..CHUNK_SIZE {
            let row_idx = start_row + i;
            if row_idx >= n { break; }
            
            // Slice for the current row in the result
            let row_slice_start = i * n;
            if row_slice_start >= chunk.len() { break; }
            
            let row_vec = &mut chunk[row_slice_start..row_slice_start + n];
            let vec_i = &pre_normalized[row_idx * dim..(row_idx + 1) * dim];
            
            for j in 0..n {
                let vec_j = &pre_normalized[j * dim..(j + 1) * dim];
                row_vec[j] = dot_simd(vec_i, vec_j);
            }
        }
    });
    
    similarity
}

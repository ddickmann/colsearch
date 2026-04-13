/// Per-dimension scalar quantization codec for residual compression.
///
/// Each token embedding is stored as:
///   - centroid_code (u16): index into the centroid table
///   - packed_residuals (u8 row): nbits-bit bucket indices packed into bytes
///
/// Decompression: output[d] = centroid[code][d] + bucket_weights[bucket_idx_d]
/// followed by L2 normalization and conversion to f16 bytes.
use std::sync::Arc;

/// Precomputed lookup table: for each possible byte value (0..255),
/// stores the `keys_per_byte` bucket indices it encodes.
type ByteLookup = Vec<Vec<usize>>;

pub struct ResidualCodec {
    centroids: Arc<Vec<f32>>,
    n_centroids: usize,
    dim: usize,
    bucket_weights: Vec<f32>,
    #[allow(dead_code)]
    nbits: usize,
    #[allow(dead_code)]
    keys_per_byte: usize,
    packed_dim: usize,
    byte_lookup: ByteLookup,
}

impl ResidualCodec {
    pub fn new(
        centroids: Arc<Vec<f32>>,
        n_centroids: usize,
        dim: usize,
        bucket_weights: Vec<f32>,
        nbits: usize,
    ) -> Self {
        assert!(nbits > 0 && 8 % nbits == 0, "nbits must divide 8");
        let keys_per_byte = 8 / nbits;
        let packed_dim = dim * nbits / 8;
        let mask = (1usize << nbits) - 1;

        let mut byte_lookup: ByteLookup = Vec::with_capacity(256);
        for byte_val in 0..256usize {
            let mut indices = Vec::with_capacity(keys_per_byte);
            for k in 0..keys_per_byte {
                let shift = (keys_per_byte - 1 - k) * nbits;
                let idx = (byte_val >> shift) & mask;
                indices.push(idx);
            }
            byte_lookup.push(indices);
        }

        Self {
            centroids,
            n_centroids,
            dim,
            bucket_weights,
            nbits,
            keys_per_byte,
            packed_dim,
            byte_lookup,
        }
    }

    /// Decompress packed residuals + centroid codes into raw f16 bytes.
    ///
    /// `codes`: u16 centroid codes, length = n_tokens
    /// `packed`: packed residual bytes, length = n_tokens * packed_dim
    ///
    /// Returns: Vec<u8> of f16 bytes, length = n_tokens * dim * 2
    pub fn decompress_to_f16(&self, codes: &[u16], packed: &[u8], n_tokens: usize) -> Vec<u8> {
        let dim = self.dim;
        let packed_dim = self.packed_dim;
        let out_bytes_per_token = dim * 2; // f16
        let mut output = vec![0u8; n_tokens * out_bytes_per_token];

        for t in 0..n_tokens {
            let code = codes[t] as usize;
            let centroid_offset = if code < self.n_centroids { code * dim } else { 0 };
            let packed_row = &packed[t * packed_dim..(t + 1) * packed_dim];

            let out_row = &mut output[t * out_bytes_per_token..(t + 1) * out_bytes_per_token];

            // Reconstruct f32 values: centroid[d] + bucket_weights[bucket_idx]
            let mut f32_buf = [0.0f32; 256]; // dim <= 256 for typical use
            let mut d_idx = 0usize;
            for &byte_val in packed_row {
                let indices = &self.byte_lookup[byte_val as usize];
                for &bucket_idx in indices {
                    if d_idx < dim {
                        let centroid_val = unsafe {
                            *self.centroids.get_unchecked(centroid_offset + d_idx)
                        };
                        let weight = unsafe {
                            *self.bucket_weights.get_unchecked(bucket_idx)
                        };
                        f32_buf[d_idx] = centroid_val + weight;
                        d_idx += 1;
                    }
                }
            }

            // L2 normalize
            let mut norm_sq = 0.0f32;
            for i in 0..dim {
                norm_sq += f32_buf[i] * f32_buf[i];
            }
            let inv_norm = if norm_sq > 1e-24 { 1.0 / norm_sq.sqrt() } else { 0.0 };

            // Convert to f16 (IEEE 754 half-precision) and write bytes
            for i in 0..dim {
                let val = f32_buf[i] * inv_norm;
                let f16_bits = f32_to_f16_bits(val);
                let byte_offset = i * 2;
                out_row[byte_offset] = (f16_bits & 0xFF) as u8;
                out_row[byte_offset + 1] = (f16_bits >> 8) as u8;
            }
        }

        output
    }

    /// Decompress packed residuals for multiple documents, returning a single
    /// flat f16 byte buffer plus per-document offsets (in token units).
    ///
    /// Each entry in `docs` is (centroid_codes_slice, packed_residuals_slice, n_tokens).
    pub fn decompress_docs(
        &self,
        docs: &[(&[u16], &[u8], usize)],
    ) -> (Vec<u8>, Vec<[i64; 2]>) {
        let dim = self.dim;
        let f16_bytes_per_token = dim * 2;
        let total_tokens: usize = docs.iter().map(|(_, _, n)| *n).sum();
        let mut flat = Vec::with_capacity(total_tokens * f16_bytes_per_token);
        let mut offsets = Vec::with_capacity(docs.len());
        let mut pos: i64 = 0;

        for &(codes, packed, n_tokens) in docs {
            let decompressed = self.decompress_to_f16(codes, packed, n_tokens);
            flat.extend_from_slice(&decompressed);
            offsets.push([pos, pos + n_tokens as i64]);
            pos += n_tokens as i64;
        }

        (flat, offsets)
    }

    pub fn packed_dim(&self) -> usize {
        self.packed_dim
    }
}

/// Convert f32 to f16 bits (IEEE 754 half-precision).
/// Handles normals, denormals, infinities, and NaN.
#[inline]
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exponent == 255 {
        // Inf or NaN
        let m = if mantissa != 0 { 0x0200 } else { 0 };
        return (sign | 0x7C00 | m) as u16;
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        // Overflow -> Inf
        return (sign | 0x7C00) as u16;
    }
    if unbiased < -24 {
        // Too small -> zero
        return sign as u16;
    }
    if unbiased < -14 {
        // Denormal
        let shift = -14 - unbiased;
        let m = (mantissa | 0x00800000) >> (shift + 13);
        return (sign | m) as u16;
    }

    let exp16 = ((unbiased + 15) as u32) << 10;
    let m16 = mantissa >> 13;
    (sign | exp16 | m16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.00006103515625];
        for &v in &vals {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            assert!(
                (v - back).abs() < v.abs() * 0.002 + 1e-7,
                "roundtrip failed: {v} -> {bits:#06x} -> {back}"
            );
        }
    }

    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x03FF) as u32;

        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign << 31);
            }
            let mut m = mant;
            let mut e = 1i32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let f32_exp = (127 + e - 15) as u32;
            return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
        }
        if exp == 31 {
            let f32_mant = if mant != 0 { 0x00400000 } else { 0 };
            return f32::from_bits((sign << 31) | 0x7F800000 | f32_mant);
        }
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }

    #[test]
    fn test_decompress_basic() {
        let dim = 4;
        let n_centroids = 2;
        let nbits = 2;
        // 2 centroids, dim=4
        let centroids = Arc::new(vec![
            1.0, 0.0, 0.0, 0.0,  // centroid 0
            0.0, 1.0, 0.0, 0.0,  // centroid 1
        ]);
        // 4 buckets for nbits=2: bucket_weights for bucket indices 0,1,2,3
        let bucket_weights = vec![-0.1, 0.0, 0.0, 0.1];

        let codec = ResidualCodec::new(centroids, n_centroids, dim, bucket_weights, nbits);
        assert_eq!(codec.packed_dim(), 1); // 4 * 2 / 8 = 1

        let codes: Vec<u16> = vec![0]; // centroid 0
        // 4 dims, nbits=2, packed into 1 byte
        // bucket indices: [3, 1, 0, 2] -> bits: 11_01_00_10 = 0b11010010 = 0xD2
        let packed: Vec<u8> = vec![0xD2];

        let result = codec.decompress_to_f16(&codes, &packed, 1);
        assert_eq!(result.len(), 4 * 2); // 4 dims * 2 bytes per f16

        // Verify: centroid[0] = [1,0,0,0], bucket_weights[3,1,0,2] = [0.1, 0.0, -0.1, 0.0]
        // raw = [1.1, 0.0, -0.1, 0.0], then L2-normalized
        let raw = [1.1f32, 0.0, -0.1, 0.0];
        let norm = (raw.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let expected: Vec<f32> = raw.iter().map(|x| x / norm).collect();

        for (i, &exp) in expected.iter().enumerate() {
            let lo = result[i * 2] as u16;
            let hi = (result[i * 2 + 1] as u16) << 8;
            let got = f16_bits_to_f32(lo | hi);
            assert!(
                (exp - got).abs() < 0.01,
                "dim {i}: expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_byte_lookup_nbits2() {
        let centroids = Arc::new(vec![0.0; 4]);
        let codec = ResidualCodec::new(centroids, 1, 4, vec![0.0; 4], 2);
        // byte 0b_11_10_01_00 = 0xE4 -> indices [3, 2, 1, 0]
        assert_eq!(codec.byte_lookup[0xE4], vec![3, 2, 1, 0]);
        // byte 0b_00_00_00_00 = 0x00 -> indices [0, 0, 0, 0]
        assert_eq!(codec.byte_lookup[0x00], vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_byte_lookup_nbits4() {
        let centroids = Arc::new(vec![0.0; 4]);
        let codec = ResidualCodec::new(centroids, 1, 4, vec![0.0; 16], 4);
        // byte 0xAB -> high nibble A=10, low nibble B=11
        assert_eq!(codec.byte_lookup[0xAB], vec![10, 11]);
    }
}

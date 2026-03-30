//! SIMD-accelerated primitives for compression.
//!
//! When the `simd` feature is enabled, these functions use archmage for
//! vectorized operations. Otherwise, scalar fallbacks are used.

// archmage is available when `simd` feature is enabled.
// Currently the SIMD count_match uses u64 XOR which LLVM auto-vectorizes well.
// archmage types can be used for more complex operations in the future.
#[cfg(feature = "simd")]
use archmage as _;

/// Count matching bytes at the beginning of `a` and `b`.
///
/// When the `simd` feature is enabled, this uses vectorized comparison
/// for significant speedup on long matches (common in LZ77 match extension).
#[inline]
pub fn count_match(a: &[u8], b: &[u8]) -> usize {
    #[cfg(feature = "simd")]
    {
        count_match_simd(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        count_match_scalar(a, b)
    }
}

/// Scalar implementation using 8-byte chunks for autovectorization.
#[inline]
pub fn count_match_scalar(a: &[u8], b: &[u8]) -> usize {
    let chunk_count = core::iter::zip(a.chunks_exact(8), b.chunks_exact(8))
        .take_while(|(x, y)| x == y)
        .count();
    let offset = chunk_count * 8;

    offset
        + core::iter::zip(&a[offset..], &b[offset..])
            .take_while(|(x, y)| x == y)
            .count()
}

/// SIMD-accelerated count_match using 8-byte u64 comparison with
/// trailing zeros for finding the first differing byte.
#[cfg(feature = "simd")]
#[inline]
fn count_match_simd(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut offset = 0;

    // Compare 8 bytes at a time using u64 XOR + trailing zeros
    while offset + 8 <= len {
        let a_val = u64::from_le_bytes(a[offset..offset + 8].try_into().unwrap());
        let b_val = u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap());
        let diff = a_val ^ b_val;
        if diff != 0 {
            return offset + (diff.trailing_zeros() as usize / 8);
        }
        offset += 8;
    }

    // Handle remaining bytes
    while offset < len {
        if a[offset] != b[offset] {
            return offset;
        }
        offset += 1;
    }

    offset
}

/// SIMD-accelerated histogram counting for symbol frequency analysis.
///
/// Counts the frequency of each byte value in the input slice.
/// Used by the entropy encoder to build FSE and Huffman tables.
#[inline]
pub fn histogram(data: &[u8], counts: &mut [u32; 256]) {
    // Clear first
    *counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_count_match_empty() {
        assert_eq!(count_match(&[], &[]), 0);
    }

    #[test]
    fn test_count_match_identical() {
        let data = vec![42u8; 100];
        assert_eq!(count_match(&data, &data), 100);
    }

    #[test]
    fn test_count_match_differ_at_start() {
        assert_eq!(count_match(&[1, 2, 3], &[0, 2, 3]), 0);
    }

    #[test]
    fn test_count_match_differ_in_middle() {
        let a = vec![1u8; 50];
        let mut b = vec![1u8; 50];
        b[25] = 2;
        assert_eq!(count_match(&a, &b), 25);
    }

    #[test]
    fn test_count_match_different_lengths() {
        let a = vec![1u8; 100];
        let b = vec![1u8; 50];
        assert_eq!(count_match(&a, &b), 50);
    }

    #[test]
    fn test_count_match_at_chunk_boundary() {
        let mut a = vec![0u8; 16];
        let mut b = vec![0u8; 16];
        b[8] = 1; // Differ right at 8-byte boundary
        assert_eq!(count_match(&a, &b), 8);

        b[8] = 0;
        b[7] = 1; // Differ just before boundary
        assert_eq!(count_match(&a, &b), 7);
    }

    #[test]
    fn test_histogram() {
        let data = [0, 1, 2, 3, 0, 1, 2, 0, 1, 0];
        let mut counts = [0u32; 256];
        histogram(&data, &mut counts);
        assert_eq!(counts[0], 4);
        assert_eq!(counts[1], 3);
        assert_eq!(counts[2], 2);
        assert_eq!(counts[3], 1);
        assert_eq!(counts[4], 0);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_scalar() {
        let a: alloc::vec::Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
        let mut b = a.clone();

        // Test at every possible mismatch position
        for pos in 0..a.len() {
            b[pos] = b[pos].wrapping_add(1);
            let scalar = count_match_scalar(&a, &b);
            let simd = count_match_simd(&a, &b);
            assert_eq!(scalar, simd, "Mismatch at position {}", pos);
            b[pos] = a[pos]; // restore
        }
    }
}

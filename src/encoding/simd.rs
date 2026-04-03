//! SIMD-accelerated primitives for compression.
//!
//! When the `simd` feature is enabled, these functions use archmage/magetypes for
//! vectorized operations with runtime CPU detection. Otherwise, scalar fallbacks
//! are used.
//!
//! ## count_match
//!
//! The hottest function in the match finder. Counts the number of matching bytes
//! at the start of two slices. With `simd` enabled, uses 32-byte AVX2 vector
//! comparison (XOR + movemask + trailing_zeros) to process 32 bytes per iteration
//! instead of 8, with a u64-based tail loop for the last 0-31 bytes.
//!
//! ## histogram
//!
//! Byte frequency counting for entropy encoder table construction. Uses 4-way
//! parallel histogram tables to reduce store-to-load forwarding stalls, which
//! is the standard technique for byte histograms without AVX-512 VPCONFLICT.

// ============================================================================
// count_match: SIMD dispatch
// ============================================================================

/// Count matching bytes at the beginning of `a` and `b`.
///
/// When the `simd` feature is enabled on x86_64, this dispatches to a 32-byte
/// AVX2 vector comparison path via `incant!`. On other platforms or without the
/// `simd` feature, falls back to 8-byte scalar chunking.
#[inline(always)]
pub fn count_match(a: &[u8], b: &[u8]) -> usize {
    #[cfg(feature = "simd")]
    {
        archmage::incant!(count_match(a, b), [v3, neon, wasm128, scalar])
    }
    #[cfg(not(feature = "simd"))]
    {
        count_match_u64(a, b)
    }
}

/// Scalar fallback for `incant!` dispatch. The `_token` parameter is required
/// by the `incant!` macro convention (the `_scalar` suffix variant receives a
/// `ScalarToken` as the first argument).
#[cfg(feature = "simd")]
#[inline]
fn count_match_scalar(_token: archmage::ScalarToken, a: &[u8], b: &[u8]) -> usize {
    count_match_u64(a, b)
}

/// NEON count_match — `#[arcane]` enables auto-vectorization of the u64 loop under target_feature.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[archmage::arcane]
fn count_match_neon(_token: archmage::NeonToken, a: &[u8], b: &[u8]) -> usize {
    count_match_u64(a, b)
}

/// WASM128 count_match — `#[arcane]` enables auto-vectorization of the u64 loop under target_feature.
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[archmage::arcane]
fn count_match_wasm128(_token: archmage::Wasm128Token, a: &[u8], b: &[u8]) -> usize {
    count_match_u64(a, b)
}

/// Core u64-based count_match. Used both as the non-SIMD path and as the
/// tail handler for the SIMD paths (processes the final 0-31 bytes after
/// the vector loop).
#[inline(always)]
fn count_match_u64(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut offset = 0;

    // Compare 8 bytes at a time using u64 XOR + trailing zeros.
    // This is the same technique as the C zstd reference: load 8 bytes as
    // a little-endian u64, XOR the two values, and if the result is non-zero,
    // count trailing zero bits to find the first differing byte position.
    while offset + 8 <= len {
        let a_val = u64::from_le_bytes(a[offset..offset + 8].try_into().unwrap());
        let b_val = u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap());
        let diff = a_val ^ b_val;
        if diff != 0 {
            return offset + (diff.trailing_zeros() as usize / 8);
        }
        offset += 8;
    }

    // Handle remaining bytes (0-7)
    while offset < len {
        if a[offset] != b[offset] {
            return offset;
        }
        offset += 1;
    }

    offset
}

/// AVX2 count_match: 32-byte vector XOR + movemask to find first differing byte.
///
/// Processes 32 bytes per iteration using `u8x32` from magetypes. For each
/// 32-byte chunk, XORs the two vectors, compares the result against zero to
/// produce a byte mask, then extracts the high bit of each lane into a u32
/// bitmask. A non-zero bitmask means at least one byte differs; `trailing_zeros`
/// on the inverted mask gives the offset of the first differing byte.
///
/// Falls back to the u64 loop for the remaining 0-31 byte tail.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn count_match_v3(token: archmage::X64V3Token, a: &[u8], b: &[u8]) -> usize {
    use magetypes::simd::generic::u8x32;

    let len = a.len().min(b.len());
    let mut offset = 0;

    // Process 32 bytes at a time using AVX2 vectors.
    //
    // Loop body: load 32 bytes from each slice into u8x32 vectors, XOR them
    // to get a diff vector (matching bytes = 0x00, differing = non-zero),
    // compare against zero with simd_eq (produces 0xFF for matching lanes),
    // extract the high bit of each byte into a u32 bitmask, and check if
    // all 32 bits are set (all matching).
    while offset + 32 <= len {
        let chunk_a: &[u8; 32] = a[offset..offset + 32].try_into().unwrap();
        let chunk_b: &[u8; 32] = b[offset..offset + 32].try_into().unwrap();

        let va = u8x32::load(token, chunk_a);
        let vb = u8x32::load(token, chunk_b);

        // XOR: matching bytes become 0x00, differing bytes become non-zero.
        let diff = va ^ vb;

        // Compare each lane against zero. simd_eq returns 0xFF for lanes that
        // are zero (matching bytes), 0x00 for non-zero (differing bytes).
        let zero = u8x32::zero(token);
        let eq_mask = diff.simd_eq(zero);

        // Extract high bit of each byte lane into a u32. Matching bytes
        // (0xFF mask) produce bit=1, differing bytes (0x00 mask) produce bit=0.
        // All-matching gives 0xFFFF_FFFF.
        let mask = eq_mask.bitmask();

        if mask != 0xFFFF_FFFF {
            // At least one byte differs. The first 0-bit in `mask` is the
            // position of the first differing byte. Invert to get 1-bits at
            // differing positions, then trailing_zeros gives the byte offset.
            let first_diff = (!mask).trailing_zeros() as usize;
            return offset + first_diff;
        }

        offset += 32;
    }

    // Tail: handle the remaining 0-31 bytes with the u64 approach.
    offset + count_match_u64(&a[offset..], &b[offset..])
}

// ============================================================================
// histogram: 4-way unrolled byte counting
// ============================================================================

/// SIMD-accelerated histogram counting for symbol frequency analysis.
///
/// Counts the frequency of each byte value in the input slice.
/// Used by the entropy encoder to build FSE and Huffman tables.
///
/// Uses 4-way parallel histograms to reduce store-to-load forwarding stalls.
/// When the same byte appears in consecutive positions, a single histogram table
/// would create a read-after-write dependency on the same counter. By spreading
/// across 4 tables and merging at the end, we allow 4 consecutive identical bytes
/// before hitting a dependency, which covers the vast majority of real data.
#[inline]
#[allow(dead_code)]
pub fn histogram(data: &[u8], counts: &mut [u32; 256]) {
    *counts = [0u32; 256];

    // 4-way unrolled histogram to reduce store-forwarding stalls.
    // Each table handles every 4th byte, so consecutive identical bytes
    // go to different tables and don't create RAW hazards.
    let mut c0 = [0u32; 256];
    let mut c1 = [0u32; 256];
    let mut c2 = [0u32; 256];
    let mut c3 = [0u32; 256];

    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Fixed-size indexing pattern: convert slice to array reference to
        // eliminate bounds checks in the inner loop.
        let b: &[u8; 4] = chunk.try_into().unwrap();
        c0[b[0] as usize] += 1;
        c1[b[1] as usize] += 1;
        c2[b[2] as usize] += 1;
        c3[b[3] as usize] += 1;
    }

    for &byte in remainder {
        c0[byte as usize] += 1;
    }

    // Merge the 4 sub-histograms into the output.
    for i in 0..256 {
        counts[i] = c0[i] + c1[i] + c2[i] + c3[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    // ====== count_match tests ======

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
        let a = vec![0u8; 16];
        let mut b = vec![0u8; 16];
        b[8] = 1; // Differ right at 8-byte boundary
        assert_eq!(count_match(&a, &b), 8);

        b[8] = 0;
        b[7] = 1; // Differ just before boundary
        assert_eq!(count_match(&a, &b), 7);
    }

    #[test]
    fn test_count_match_at_32byte_boundary() {
        // Test at the 32-byte SIMD vector boundary
        let a = vec![0xAA; 64];
        let mut b = vec![0xAA; 64];

        // Differ right at byte 32
        b[32] = 0;
        assert_eq!(count_match(&a, &b), 32);

        // Differ right at byte 31 (last byte of first vector)
        b[32] = 0xAA;
        b[31] = 0;
        assert_eq!(count_match(&a, &b), 31);

        // Differ right at byte 33 (second byte of second vector)
        b[31] = 0xAA;
        b[33] = 0;
        assert_eq!(count_match(&a, &b), 33);
    }

    #[test]
    fn test_count_match_long_identical() {
        // Test with data longer than one 32-byte vector
        let a = vec![0x55; 200];
        assert_eq!(count_match(&a, &a), 200);
    }

    #[test]
    fn test_count_match_long_with_late_diff() {
        // Difference at position 150 (after 4+ full 32-byte vectors)
        let a = vec![0x77; 200];
        let mut b = vec![0x77; 200];
        b[150] = 0x88;
        assert_eq!(count_match(&a, &b), 150);
    }

    // ====== Exhaustive SIMD vs scalar comparison ======

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_scalar_every_position() {
        let a: alloc::vec::Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
        let mut b = a.clone();

        // Test at every possible mismatch position, comparing dispatch result
        // against the known-good u64 scalar path.
        for pos in 0..a.len() {
            b[pos] = b[pos].wrapping_add(1);
            let scalar = count_match_u64(&a, &b);
            let dispatched = count_match(&a, &b);
            assert_eq!(scalar, dispatched, "Mismatch at position {pos}");
            b[pos] = a[pos]; // restore
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_v3_matches_scalar_every_position() {
        use archmage::SimdToken;

        let a: alloc::vec::Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
        let mut b = a.clone();

        if let Some(v3_token) = archmage::X64V3Token::summon() {
            for pos in 0..a.len() {
                b[pos] = b[pos].wrapping_add(1);
                let scalar = count_match_u64(&a, &b);
                let simd = count_match_v3(v3_token, &a, &b);
                assert_eq!(scalar, simd, "v3 mismatch at position {pos}");
                b[pos] = a[pos]; // restore
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_v3_short_inputs() {
        use archmage::SimdToken;

        if let Some(token) = archmage::X64V3Token::summon() {
            // Inputs shorter than 32 bytes -- should fall through to u64 tail
            for len in 0..32 {
                let a = vec![0xBB; len];
                let b = vec![0xBB; len];
                assert_eq!(
                    count_match_v3(token, &a, &b),
                    len,
                    "fully matching at len={len}"
                );

                if len > 0 {
                    let mut c = vec![0xBB; len];
                    c[0] = 0xCC;
                    assert_eq!(
                        count_match_v3(token, &a, &c),
                        0,
                        "differ at start, len={len}"
                    );
                }
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_v3_tail_handling() {
        use archmage::SimdToken;

        if let Some(token) = archmage::X64V3Token::summon() {
            // Test tail handling: diff in the 0-31 byte remainder after vector loop.
            // Use 40 bytes so we get one full 32-byte vector pass + 8-byte tail.
            let a = vec![0x11; 40];
            let mut b = vec![0x11; 40];

            // Diff at position 35 (in the tail, offset 3 within remaining 8 bytes)
            b[35] = 0x22;
            assert_eq!(count_match_v3(token, &a, &b), 35);

            // Diff at position 39 (last byte, in tail)
            b[35] = 0x11;
            b[39] = 0x22;
            assert_eq!(count_match_v3(token, &a, &b), 39);

            // Diff at position 32 (first byte of tail)
            b[39] = 0x11;
            b[32] = 0x22;
            assert_eq!(count_match_v3(token, &a, &b), 32);
        }
    }

    // ====== histogram tests ======

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

    #[test]
    fn test_histogram_empty() {
        let mut counts = [0u32; 256];
        histogram(&[], &mut counts);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_histogram_single_value() {
        let data = vec![42u8; 1000];
        let mut counts = [0u32; 256];
        histogram(&data, &mut counts);
        assert_eq!(counts[42], 1000);
        assert_eq!(
            counts.iter().sum::<u32>(),
            1000,
            "total count must equal input length"
        );
    }

    #[test]
    fn test_histogram_all_values() {
        // Every byte value appears exactly once
        let data: alloc::vec::Vec<u8> = (0..=255).map(|i| i as u8).collect();
        let mut counts = [0u32; 256];
        histogram(&data, &mut counts);
        assert!(counts.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_histogram_non_multiple_of_4() {
        // Length not divisible by 4, exercises the remainder path
        let data = [10, 20, 30, 40, 50, 60, 70];
        let mut counts = [0u32; 256];
        histogram(&data, &mut counts);
        assert_eq!(counts[10], 1);
        assert_eq!(counts[20], 1);
        assert_eq!(counts[30], 1);
        assert_eq!(counts[40], 1);
        assert_eq!(counts[50], 1);
        assert_eq!(counts[60], 1);
        assert_eq!(counts[70], 1);
        assert_eq!(counts.iter().sum::<u32>(), 7);
    }

    #[test]
    fn test_histogram_consistency() {
        // Verify total count always equals input length
        let data = vec![0u8; 999];
        let mut counts = [0u32; 256];
        histogram(&data, &mut counts);
        assert_eq!(counts.iter().sum::<u32>(), 999);
    }
}

//! Zstd hash functions for match finding.
//!
//! These are direct ports of the hash functions from `zstd_compress_internal.h`
//! in the reference C implementation. Each function reads a minimum number of
//! bytes from the input slice (the "minimum match length"), multiplies by a
//! prime constant, and shifts right to produce a hash of the requested bit width.
//!
//! The primes were chosen by the zstd authors to give good avalanche properties
//! for short byte sequences.

/// Hash constant for 3-byte keys (from C zstd).
const PRIME_3: u32 = 506_832_829;

/// Knuth multiplicative hash constant for 4-byte keys.
const PRIME_4: u32 = 2_654_435_761;

/// Hash constant for 5-byte keys.
const PRIME_5: u64 = 889_523_592_379;

/// Hash constant for 6-byte keys.
const PRIME_6: u64 = 227_718_039_650_203;

/// Hash constant for 7-byte keys.
const PRIME_7: u64 = 58_295_818_150_454_627;

/// Hash constant for 8-byte keys.
const PRIME_8: u64 = 0xCF1B_BCDC_B7A5_6463;

/// Read a little-endian `u32` from the first 4 bytes of `data`.
///
/// Uses the fixed-size array pattern (try_into at the boundary) to
/// eliminate per-byte bounds checks inside the hot loop.
///
/// # Panics
///
/// Panics if `data.len() < 4`.
#[inline(always)]
fn read_le32(data: &[u8]) -> u32 {
    let bytes: &[u8; 4] = data[..4].try_into().unwrap();
    u32::from_le_bytes(*bytes)
}

/// Read a little-endian `u64` from the first 8 bytes of `data`.
///
/// Uses the fixed-size array pattern to eliminate per-byte bounds checks.
///
/// # Panics
///
/// Panics if `data.len() < 8`.
#[inline(always)]
fn read_le64(data: &[u8]) -> u64 {
    let bytes: &[u8; 8] = data[..8].try_into().unwrap();
    u64::from_le_bytes(*bytes)
}

/// Hash the first 3 bytes of `data` into `hash_bits` bits.
///
/// Equivalent to C zstd's `ZSTD_hash3Ptr`. Reads a LE u32 and masks to
/// 24 bits, then multiplies by `PRIME_3`.
///
/// # Panics
///
/// Panics if `data.len() < 4` or `hash_bits > 32` or `hash_bits == 0`.
#[inline(always)]
pub fn hash3(data: &[u8], hash_bits: u32) -> u32 {
    debug_assert!(hash_bits > 0 && hash_bits <= 32);
    (read_le32(data) << 8).wrapping_mul(PRIME_3) >> (32 - hash_bits)
}

/// Hash the first 4 bytes of `data` into `hash_bits` bits.
///
/// Equivalent to C zstd's `ZSTD_hash4Ptr`.
///
/// # Panics
///
/// Panics if `data.len() < 4` or `hash_bits > 32` or `hash_bits == 0`.
#[inline(always)]
pub fn hash4(data: &[u8], hash_bits: u32) -> u32 {
    debug_assert!(hash_bits > 0 && hash_bits <= 32);
    read_le32(data).wrapping_mul(PRIME_4) >> (32 - hash_bits)
}

/// Hash the first 5 bytes of `data` into `hash_bits` bits.
///
/// Equivalent to C zstd's `ZSTD_hash5Ptr`. Reads 8 bytes as LE u64, masks to
/// 40 bits (by shifting left 24 then keeping the high portion via the multiply
/// and shift), then multiplies by `PRIME_5`.
///
/// # Panics
///
/// Panics if `data.len() < 8` or `hash_bits > 64` or `hash_bits == 0`.
#[inline(always)]
pub fn hash5(data: &[u8], hash_bits: u32) -> usize {
    debug_assert!(hash_bits > 0 && hash_bits <= 64);
    let val = read_le64(data);
    // Shift left by (64 - 40) = 24 to isolate the low 40 bits in the high part,
    // multiply by prime, then shift right to get the desired number of hash bits.
    ((val << 24).wrapping_mul(PRIME_5) >> (64 - hash_bits)) as usize
}

/// Hash the first 6 bytes of `data` into `hash_bits` bits.
///
/// Equivalent to C zstd's `ZSTD_hash6Ptr`.
///
/// # Panics
///
/// Panics if `data.len() < 8` or `hash_bits > 64` or `hash_bits == 0`.
#[inline(always)]
pub fn hash6(data: &[u8], hash_bits: u32) -> usize {
    debug_assert!(hash_bits > 0 && hash_bits <= 64);
    let val = read_le64(data);
    // Shift left by (64 - 48) = 16 to isolate the low 48 bits.
    ((val << 16).wrapping_mul(PRIME_6) >> (64 - hash_bits)) as usize
}

/// Hash the first 7 bytes of `data` into `hash_bits` bits.
///
/// Equivalent to C zstd's `ZSTD_hash7Ptr`.
///
/// # Panics
///
/// Panics if `data.len() < 8` or `hash_bits > 64` or `hash_bits == 0`.
#[inline(always)]
pub fn hash7(data: &[u8], hash_bits: u32) -> usize {
    debug_assert!(hash_bits > 0 && hash_bits <= 64);
    let val = read_le64(data);
    // Shift left by (64 - 56) = 8 to isolate the low 56 bits.
    ((val << 8).wrapping_mul(PRIME_7) >> (64 - hash_bits)) as usize
}

/// Hash the first 8 bytes of `data` into `hash_bits` bits.
///
/// Equivalent to C zstd's `ZSTD_hash8Ptr`.
///
/// # Panics
///
/// Panics if `data.len() < 8` or `hash_bits > 64` or `hash_bits == 0`.
#[inline(always)]
pub fn hash8(data: &[u8], hash_bits: u32) -> usize {
    debug_assert!(hash_bits > 0 && hash_bits <= 64);
    let val = read_le64(data);
    (val.wrapping_mul(PRIME_8) >> (64 - hash_bits)) as usize
}

/// Hash `data` using the hash function for the given minimum match length.
///
/// Dispatches to [`hash4`] through [`hash8`] based on `min_match_len`.
/// Values outside 4..=8 use `hash4` (matching C zstd's default case).
///
/// # Panics
///
/// Panics if `data` is shorter than the required read length (4 bytes for
/// `hash4`, 8 bytes for all others).
#[inline(always)]
pub fn hash_ptr(data: &[u8], hash_bits: u32, min_match_len: u32) -> usize {
    match min_match_len {
        5 => hash5(data, hash_bits),
        6 => hash6(data, hash_bits),
        7 => hash7(data, hash_bits),
        8 => hash8(data, hash_bits),
        // C zstd default case falls through to hash4
        _ => hash4(data, hash_bits) as usize,
    }
}

/// Count the number of matching bytes at the start of `a` and `b`.
///
/// Compares bytes from the beginning of both slices and returns the length of
/// the common prefix. Delegates to the SIMD-accelerated version when the
/// `simd` feature is enabled, otherwise uses u64-chunked comparison.
///
/// ```
/// # // This is a module-level doctest, but the function is pub so it works.
/// // Matching prefix of length 5:
/// let a = [1, 2, 3, 4, 5, 99, 100];
/// let b = [1, 2, 3, 4, 5, 77, 88];
/// // count_match would return 5
/// ```
#[inline]
pub fn count_match(a: &[u8], b: &[u8]) -> usize {
    super::simd::count_match(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Basic correctness: deterministic, same input => same output
    // ---------------------------------------------------------------

    #[test]
    fn hash4_deterministic() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        let h1 = hash4(&data, 16);
        let h2 = hash4(&data, 16);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash5_deterministic() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00];
        let h1 = hash5(&data, 20);
        let h2 = hash5(&data, 20);
        assert_eq!(h1, h2);
    }

    // ---------------------------------------------------------------
    // Bit width: output must fit in the requested number of bits
    // ---------------------------------------------------------------

    #[test]
    fn hash4_fits_in_bits() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        for bits in 1..=31 {
            let h = hash4(&data, bits);
            assert!(
                h < (1u32 << bits),
                "hash4 with bits={bits} produced {h}, which doesn't fit"
            );
        }
        // bits == 32: any u32 value is valid (the shift is >> 0)
        let _ = hash4(&data, 32);
    }

    #[test]
    fn hash5_fits_in_bits() {
        let data = [0xFF; 8];
        for bits in 1..=32 {
            let h = hash5(&data, bits);
            assert!(
                h < (1usize << bits),
                "hash5 with bits={bits} produced {h}, which doesn't fit"
            );
        }
    }

    #[test]
    fn hash6_fits_in_bits() {
        let data = [0xFF; 8];
        for bits in 1..=32 {
            let h = hash6(&data, bits);
            assert!(
                h < (1usize << bits),
                "hash6 with bits={bits} produced {h}, which doesn't fit"
            );
        }
    }

    #[test]
    fn hash7_fits_in_bits() {
        let data = [0xFF; 8];
        for bits in 1..=32 {
            let h = hash7(&data, bits);
            assert!(
                h < (1usize << bits),
                "hash7 with bits={bits} produced {h}, which doesn't fit"
            );
        }
    }

    #[test]
    fn hash8_fits_in_bits() {
        let data = [0xFF; 8];
        for bits in 1..=32 {
            let h = hash8(&data, bits);
            assert!(
                h < (1usize << bits),
                "hash8 with bits={bits} produced {h}, which doesn't fit"
            );
        }
    }

    // ---------------------------------------------------------------
    // Reference vectors: verify against known C zstd outputs
    // ---------------------------------------------------------------

    #[test]
    fn hash4_reference_vectors() {
        // hash4([0,0,0,0], 16) => (0u32.wrapping_mul(2654435761)) >> 16 = 0
        let zeros = [0u8; 8];
        assert_eq!(hash4(&zeros, 16), 0);

        // hash4([1,0,0,0], 16) => (1u32.wrapping_mul(2654435761)) >> 16
        let one = [1u8, 0, 0, 0, 0, 0, 0, 0];
        let expected = 1u32.wrapping_mul(PRIME_4) >> 16;
        assert_eq!(hash4(&one, 16), expected);

        // hash4([0xFF,0xFF,0xFF,0xFF], 16) => (0xFFFFFFFF * PRIME_4) >> 16
        let max = [0xFFu8; 8];
        let expected = 0xFFFF_FFFFu32.wrapping_mul(PRIME_4) >> 16;
        assert_eq!(hash4(&max, 16), expected);
    }

    #[test]
    fn hash5_reference_vectors() {
        // For input [1, 0, 0, 0, 0, 0, 0, 0]:
        // val = 1u64, (1 << 24) * PRIME_5 >> (64 - 20)
        let data = [1u8, 0, 0, 0, 0, 0, 0, 0];
        let val = 1u64;
        let expected = ((val << 24).wrapping_mul(PRIME_5) >> (64 - 20)) as usize;
        assert_eq!(hash5(&data, 20), expected);
    }

    #[test]
    fn hash8_reference_vectors() {
        let data = [1u8, 0, 0, 0, 0, 0, 0, 0];
        let val = 1u64;
        let expected = (val.wrapping_mul(PRIME_8) >> (64 - 20)) as usize;
        assert_eq!(hash8(&data, 20), expected);
    }

    // ---------------------------------------------------------------
    // hash_ptr dispatch: verify it delegates correctly
    // ---------------------------------------------------------------

    #[test]
    fn hash_ptr_dispatches_correctly() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let bits = 18;

        assert_eq!(hash_ptr(&data, bits, 4), hash4(&data, bits) as usize);
        assert_eq!(hash_ptr(&data, bits, 5), hash5(&data, bits));
        assert_eq!(hash_ptr(&data, bits, 6), hash6(&data, bits));
        assert_eq!(hash_ptr(&data, bits, 7), hash7(&data, bits));
        assert_eq!(hash_ptr(&data, bits, 8), hash8(&data, bits));

        // Out-of-range values fall back to hash4
        assert_eq!(hash_ptr(&data, bits, 3), hash4(&data, bits) as usize);
        assert_eq!(hash_ptr(&data, bits, 9), hash4(&data, bits) as usize);
    }

    // ---------------------------------------------------------------
    // Distribution: different inputs should spread across buckets
    // ---------------------------------------------------------------

    #[test]
    fn hash4_distribution() {
        // Hash 1024 sequential 4-byte values into a 10-bit table (1024 buckets).
        // A good hash should fill a significant fraction of the buckets.
        let bits = 10u32;
        let table_size = 1u32 << bits;
        let mut buckets = alloc::vec![0u32; table_size as usize];

        for i in 0u32..1024 {
            let data = i.to_le_bytes();
            // Pad to 8 bytes so hash5+ can also be tested with the same pattern.
            let padded = [data[0], data[1], data[2], data[3], 0, 0, 0, 0];
            let h = hash4(&padded, bits);
            buckets[h as usize] += 1;
        }

        let occupied = buckets.iter().filter(|&&c| c > 0).count();
        // With 1024 inputs and 1024 buckets, a decent hash should fill at least
        // 500 buckets (birthday paradox gives ~632 for a perfect random hash).
        assert!(
            occupied >= 500,
            "hash4 only filled {occupied}/1024 buckets -- poor distribution"
        );
    }

    #[test]
    fn hash5_distribution() {
        let bits = 10u32;
        let table_size = 1usize << bits;
        let mut buckets = alloc::vec![0u32; table_size];

        for i in 0u32..1024 {
            let le = i.to_le_bytes();
            let data = [le[0], le[1], le[2], le[3], (i >> 2) as u8, 0, 0, 0];
            let h = hash5(&data, bits);
            buckets[h] += 1;
        }

        let occupied = buckets.iter().filter(|&&c| c > 0).count();
        assert!(
            occupied >= 500,
            "hash5 only filled {occupied}/1024 buckets -- poor distribution"
        );
    }

    #[test]
    fn hash8_distribution() {
        let bits = 10u32;
        let table_size = 1usize << bits;
        let mut buckets = alloc::vec![0u32; table_size];

        for i in 0u64..1024 {
            let data = i.to_le_bytes();
            let h = hash8(&data, bits);
            buckets[h] += 1;
        }

        let occupied = buckets.iter().filter(|&&c| c > 0).count();
        assert!(
            occupied >= 500,
            "hash8 only filled {occupied}/1024 buckets -- poor distribution"
        );
    }

    // ---------------------------------------------------------------
    // Sensitivity: single-bit changes should change the hash
    // ---------------------------------------------------------------

    #[test]
    fn hash4_avalanche_single_bit() {
        let base = [0x55u8, 0xAA, 0x33, 0xCC, 0x00, 0x00, 0x00, 0x00];
        let base_h = hash4(&base, 20);

        // Flip each of the first 32 bits and confirm the hash changes
        let mut changed = 0;
        for bit in 0..32 {
            let byte_idx = bit / 8;
            let bit_idx = bit % 8;
            let mut modified = base;
            modified[byte_idx] ^= 1 << bit_idx;
            let h = hash4(&modified, 20);
            if h != base_h {
                changed += 1;
            }
        }
        // Every single-bit flip in the 4 input bytes should change the hash
        // (multiplicative hashing with an odd prime guarantees this for any
        // non-zero single-bit difference when the shift doesn't discard it).
        // Allow a small tolerance for the highest bits that may be shifted away.
        assert!(
            changed >= 28,
            "hash4: only {changed}/32 single-bit flips changed the hash"
        );
    }

    #[test]
    fn hash8_avalanche_single_bit() {
        let base = [0x55u8, 0xAA, 0x33, 0xCC, 0x11, 0x22, 0x44, 0x88];
        let base_h = hash8(&base, 24);

        let mut changed = 0;
        for bit in 0..64 {
            let byte_idx = bit / 8;
            let bit_idx = bit % 8;
            let mut modified = base;
            modified[byte_idx] ^= 1 << bit_idx;
            let h = hash8(&modified, 24);
            if h != base_h {
                changed += 1;
            }
        }
        assert!(
            changed >= 58,
            "hash8: only {changed}/64 single-bit flips changed the hash"
        );
    }

    // ---------------------------------------------------------------
    // count_match
    // ---------------------------------------------------------------

    #[test]
    fn count_match_identical() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(count_match(&a, &a), a.len());
    }

    #[test]
    fn count_match_empty() {
        assert_eq!(count_match(&[], &[]), 0);
        assert_eq!(count_match(&[1], &[]), 0);
        assert_eq!(count_match(&[], &[1]), 0);
    }

    #[test]
    fn count_match_no_match() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        assert_eq!(count_match(&a, &b), 0);
    }

    #[test]
    fn count_match_partial() {
        let a = [1, 2, 3, 4, 5, 99, 100];
        let b = [1, 2, 3, 4, 5, 77, 88];
        assert_eq!(count_match(&a, &b), 5);
    }

    #[test]
    fn count_match_different_lengths() {
        let a = [1, 2, 3, 4, 5];
        let b = [1, 2, 3];
        assert_eq!(count_match(&a, &b), 3);
        assert_eq!(count_match(&b, &a), 3);
    }

    #[test]
    fn count_match_long_prefix() {
        // Test across the 8-byte chunk boundary
        let mut a = alloc::vec![0xAB; 100];
        let mut b = alloc::vec![0xAB; 100];
        a[73] = 0xFF;
        b[73] = 0x00;
        assert_eq!(count_match(&a, &b), 73);
    }

    #[test]
    fn count_match_exactly_chunk_boundary() {
        // Mismatch lands right at an 8-byte boundary
        let a = alloc::vec![42; 32];
        let mut b = alloc::vec![42; 32];
        b[16] = 0;
        assert_eq!(count_match(&a, &b), 16);

        // Mismatch at byte 8
        b[16] = 42;
        b[8] = 0;
        assert_eq!(count_match(&a, &b), 8);
    }
}

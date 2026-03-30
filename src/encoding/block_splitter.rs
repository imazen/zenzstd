//! Pre-compression block splitter based on byte distribution fingerprints.
//!
//! Port of C zstd's `zstd_preSplit.c`. Splits a 128KB block into smaller
//! sub-blocks at points where the byte distribution changes significantly,
//! allowing per-block entropy tables tuned to local statistics.
//!
//! The algorithm works by sliding an 8KB window across the data, building
//! byte histograms (fingerprints) for each chunk, and comparing them to the
//! accumulated history. When the current chunk's fingerprint diverges too much
//! from the past, a split point is emitted.

use alloc::vec;
use alloc::vec::Vec;

/// Minimum sub-block size in bytes. Blocks smaller than this are not worth
/// splitting because the overhead of a new block header + entropy tables
/// would outweigh any benefit.
const BLOCKSIZE_MIN: usize = 3500;

/// Chunk size for fingerprint computation (8KB).
const CHUNKSIZE: usize = 8 << 10;

/// Threshold parameters matching C zstd.
const THRESHOLD_PENALTY_RATE: u64 = 16;
const THRESHOLD_BASE: u64 = THRESHOLD_PENALTY_RATE - 2;
const THRESHOLD_PENALTY: i32 = 3;

/// Maximum hash log for fingerprint tables.
const HASHLOG_MAX: u32 = 10;

/// Size of the hash table used for fingerprinting.
const HASHTABLESIZE: usize = 1 << HASHLOG_MAX;

/// Knuth multiplicative hash constant.
const KNUTH: u32 = 0x9e37_79b9;

/// A byte distribution fingerprint: a histogram of hash buckets.
#[derive(Clone)]
struct Fingerprint {
    events: [u32; HASHTABLESIZE],
    nb_events: u64,
}

impl Fingerprint {
    fn new() -> Self {
        Self {
            events: [0; HASHTABLESIZE],
            nb_events: 0,
        }
    }

    /// Add events from a data slice using the given sampling rate and hash log.
    fn add_events(&mut self, data: &[u8], sampling_rate: usize, hash_log: u32) {
        if data.len() < 2 {
            return;
        }
        let limit = data.len() - 1; // need at least 2 bytes for hash
        let mask = (1u32 << hash_log) - 1;
        let mut n = 0;
        while n < limit {
            let h = hash2(&data[n..], hash_log, mask);
            self.events[h as usize] += 1;
            n += sampling_rate;
        }
        self.nb_events += (limit / sampling_rate) as u64;
    }

    /// Record a fingerprint (clear then add) from a data slice.
    fn record(&mut self, data: &[u8], sampling_rate: usize, hash_log: u32) {
        // Only clear the portion of the table that matters for this hash_log
        let table_size = 1usize << hash_log;
        self.events[..table_size].fill(0);
        self.nb_events = 0;
        self.add_events(data, sampling_rate, hash_log);
    }

    /// Merge another fingerprint into this one (accumulate).
    fn merge(&mut self, other: &Fingerprint) {
        for i in 0..HASHTABLESIZE {
            self.events[i] += other.events[i];
        }
        self.nb_events += other.nb_events;
    }
}

/// Hash 2 bytes from the given position.
/// For hash_log == 8, just returns the first byte (no hashing needed).
/// For hash_log > 8, hashes 2 bytes with Knuth multiplication.
#[inline]
fn hash2(data: &[u8], hash_log: u32, _mask: u32) -> u32 {
    debug_assert!(data.len() >= 2);
    debug_assert!(hash_log >= 8);
    if hash_log == 8 {
        return data[0] as u32;
    }
    let val = u16::from_le_bytes([data[0], data[1]]) as u32;
    val.wrapping_mul(KNUTH) >> (32 - hash_log)
}

/// Compute the distance (divergence) between two fingerprints.
///
/// Uses cross-multiplication to avoid floating point: for each bucket,
/// computes |fp1.events[i] * fp2.nb_events - fp2.events[i] * fp1.nb_events|.
fn fp_distance(fp1: &Fingerprint, fp2: &Fingerprint, hash_log: u32) -> u64 {
    let table_size = 1usize << hash_log;
    let mut distance: u64 = 0;
    for i in 0..table_size {
        let a = fp1.events[i] as i64 * fp2.nb_events as i64;
        let b = fp2.events[i] as i64 * fp1.nb_events as i64;
        distance += (a - b).unsigned_abs();
    }
    distance
}

/// Compare two fingerprints. Returns true when they are "too different"
/// (divergence exceeds the threshold).
fn fingerprints_diverge(
    reference: &Fingerprint,
    new: &Fingerprint,
    penalty: i32,
    hash_log: u32,
) -> bool {
    debug_assert!(reference.nb_events > 0);
    debug_assert!(new.nb_events > 0);
    let p50 = reference.nb_events * new.nb_events;
    let deviation = fp_distance(reference, new, hash_log);
    let threshold = p50 * (THRESHOLD_BASE + penalty as u64) / THRESHOLD_PENALTY_RATE;
    deviation >= threshold
}

/// Split level determines the sampling rate and hash table size.
/// Higher levels are slower but more accurate.
///
/// - Level 0: fast border comparison (not used in the chunk-based path)
/// - Level 1-4: chunk-based with increasing accuracy
///
/// Maps to C zstd compression strategies:
/// - fast/dfast -> level 0-1
/// - greedy/lazy -> level 2-3
/// - btlazy2+ -> level 3-4
fn split_params(level: u32) -> (usize, u32) {
    match level {
        0 => (43, 8),
        1 => (11, 9),
        2 => (5, 10),
        3 => (1, 10),
        _ => (1, 10),
    }
}

/// Determine the split level based on the zstd compression level (1-22).
///
/// C zstd maps compression strategies to split levels:
/// - fast (L1-2) -> split level 0
/// - dfast (L3-4) -> split level 1
/// - greedy (L5) -> split level 2
/// - lazy (L6-7) -> split level 2
/// - lazy2 (L8-12) -> split level 3
/// - btlazy2 (L13-15) -> split level 3
/// - btopt+ (L16-22) -> split level 4
fn split_level_for_compression_level(level: i32) -> u32 {
    match level {
        1..=2 => 0,
        3..=4 => 1,
        5..=7 => 2,
        8..=15 => 3,
        16..=22 => 4,
        _ => 2,
    }
}

/// Find a single split point in the given data using the chunk-based method.
///
/// Returns the offset of the split point, or `data.len()` if no split is found.
fn find_split_point(data: &[u8], split_level: u32) -> usize {
    if data.len() < CHUNKSIZE * 2 {
        return data.len(); // Too small to split
    }

    let (sampling_rate, hash_log) = split_params(split_level.min(3));

    let mut past = Fingerprint::new();
    let mut current = Fingerprint::new();

    // Initialize with the first chunk
    past.record(&data[..CHUNKSIZE], sampling_rate, hash_log);

    let mut penalty = THRESHOLD_PENALTY;
    let mut pos = CHUNKSIZE;

    while pos + CHUNKSIZE <= data.len() {
        current.record(&data[pos..pos + CHUNKSIZE], sampling_rate, hash_log);

        if fingerprints_diverge(&past, &current, penalty, hash_log) {
            return pos;
        }

        // No split: merge current into past and reduce penalty
        past.merge(&current);
        if penalty > 0 {
            penalty -= 1;
        }

        pos += CHUNKSIZE;
    }

    data.len()
}

/// Find all split points in a block of data, returning the boundaries.
///
/// Returns a vector of split offsets. Each pair of adjacent offsets defines
/// a sub-block. The first sub-block starts at 0 and the last ends at `data.len()`.
///
/// For example, if the data is 128KB and splits are found at 32KB and 80KB,
/// this returns `[0, 32768, 81920, 131072]`.
pub fn find_split_points(data: &[u8], compression_level: i32) -> Vec<usize> {
    // Only split blocks that are at least 128KB (matching C zstd's heuristic)
    if data.len() < 128 * 1024 {
        return vec![0, data.len()];
    }

    let split_level = split_level_for_compression_level(compression_level);

    let mut boundaries = Vec::with_capacity(8);
    boundaries.push(0);

    // Recursively find split points
    find_splits_recursive(data, 0, data.len(), split_level, &mut boundaries);

    boundaries.push(data.len());
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

/// Recursively split a region of data.
fn find_splits_recursive(
    data: &[u8],
    start: usize,
    end: usize,
    split_level: u32,
    boundaries: &mut Vec<usize>,
) {
    let region_size = end - start;

    // Don't split if the region is too small
    if region_size < BLOCKSIZE_MIN * 2 || region_size < CHUNKSIZE * 2 {
        return;
    }

    // Cap recursion depth (max ~10 splits from a 128KB block)
    if boundaries.len() >= 16 {
        return;
    }

    let split = find_split_point(&data[start..end], split_level);

    if split < end - start {
        // Found a split point — record it and recurse on both halves
        let abs_split = start + split;

        // Only add if both halves are large enough
        if split >= BLOCKSIZE_MIN && (end - start - split) >= BLOCKSIZE_MIN {
            // Recurse on the left half first
            find_splits_recursive(data, start, abs_split, split_level, boundaries);
            boundaries.push(abs_split);
            // Recurse on the right half
            find_splits_recursive(data, abs_split, end, split_level, boundaries);
        }
    }
}

// ---------------------------------------------------------------------------
// Post-match-finding sequence splitting (entropy cost estimation)
// ---------------------------------------------------------------------------

use crate::encoding::zstd_match::SequenceOut;

/// Minimum number of sequences in a partition for splitting to be worthwhile.
/// Below this threshold, the entropy table overhead dominates any savings.
/// Matches C zstd's MIN_SEQUENCES_BLOCK_SPLITTING.
const MIN_SEQUENCES_BLOCK_SPLITTING: usize = 300;

/// Maximum number of split points we'll produce.
const MAX_SPLITS: usize = 196;

/// Measure the actual compressed size (in bytes) of a sub-block by trial-encoding
/// it with fresh entropy tables. This gives an exact size rather than an estimate.
///
/// Returns the number of bytes the compressed block data would take (not including
/// the 3-byte block header).
fn measure_sub_block_size(literals: &[u8], sequences: &[SequenceOut]) -> usize {
    if sequences.is_empty() {
        // No sequences: would be a raw or literals-only block
        return literals.len();
    }

    let mut output = Vec::with_capacity(literals.len() + sequences.len() * 4);
    crate::encoding::blocks::encode_compressed_block_standalone(literals, sequences, &mut output);
    output.len()
}

/// Extract the literal bytes corresponding to a sub-range of sequences.
///
/// Each sequence has a `lit_len` field indicating how many literal bytes
/// precede it. The literals for sequences `start_seq..end_seq` are the
/// bytes from the cumulative lit_len sum up to `start_seq` through `end_seq`.
///
/// If `end_seq == total_sequences` (last partition), also includes trailing
/// literals that aren't covered by any sequence.
fn extract_partition_literals<'a>(
    all_literals: &'a [u8],
    all_sequences: &[SequenceOut],
    start_seq: usize,
    end_seq: usize,
) -> &'a [u8] {
    // Find the start offset in literals
    let lit_start: usize = all_sequences[..start_seq]
        .iter()
        .map(|s| s.lit_len as usize)
        .sum();

    // Find the end offset: sum of lit_lens up to end_seq
    let lit_end_seqs: usize = all_sequences[..end_seq]
        .iter()
        .map(|s| s.lit_len as usize)
        .sum();

    // If this is the last partition, include trailing literals
    let lit_end = if end_seq == all_sequences.len() {
        all_literals.len()
    } else {
        lit_end_seqs
    };

    &all_literals[lit_start..lit_end]
}

/// Recursively find sequence-based split points by trial-encoding.
///
/// The algorithm tries splitting at the midpoint and actually encodes both
/// halves and the whole to compare real compressed sizes. A split is accepted
/// when two separate blocks are smaller than one combined block (accounting
/// for the extra 3-byte block header of the additional block).
fn derive_splits_recursive(
    all_literals: &[u8],
    all_sequences: &[SequenceOut],
    start: usize,
    end: usize,
    splits: &mut Vec<usize>,
) {
    let n_seq = end - start;
    if n_seq < MIN_SEQUENCES_BLOCK_SPLITTING || splits.len() >= MAX_SPLITS {
        return;
    }

    let mid = (start + end) / 2;

    // Measure actual compressed size of the whole range
    let whole_lits = extract_partition_literals(all_literals, all_sequences, start, end);
    let whole_size = measure_sub_block_size(whole_lits, &all_sequences[start..end]);

    // Measure actual compressed size of first half
    let first_lits = extract_partition_literals(all_literals, all_sequences, start, mid);
    let first_size = measure_sub_block_size(first_lits, &all_sequences[start..mid]);

    // Measure actual compressed size of second half
    let second_lits = extract_partition_literals(all_literals, all_sequences, mid, end);
    let second_size = measure_sub_block_size(second_lits, &all_sequences[mid..end]);

    // Split if two blocks (each with 3-byte header) are smaller than one block.
    // The extra cost is one additional 3-byte block header.
    let split_cost = first_size + second_size + 3; // extra block header
    if split_cost < whole_size {
        derive_splits_recursive(all_literals, all_sequences, start, mid, splits);
        splits.push(mid);
        derive_splits_recursive(all_literals, all_sequences, mid, end, splits);
    }
}

/// A partition of a compressed block: a sub-range of sequences with their literals.
pub struct BlockPartition {
    /// Literal bytes for this partition.
    pub literals: Vec<u8>,
    /// Sequences for this partition.
    pub sequences: Vec<SequenceOut>,
}

/// Split a compressed block's sequences into multiple partitions based on
/// entropy cost estimation. Each partition gets its own entropy tables,
/// which improves compression ratio when the data has varying statistical
/// properties.
///
/// Returns a list of partitions. If splitting is not beneficial (or the block
/// is too small), returns a single partition containing all data.
pub fn split_sequences(literals: &[u8], sequences: &[SequenceOut]) -> Vec<BlockPartition> {
    let n_seq = sequences.len();

    // Don't attempt splitting with very few sequences
    if n_seq <= 4 {
        return vec![BlockPartition {
            literals: literals.to_vec(),
            sequences: sequences.to_vec(),
        }];
    }

    let mut split_points = Vec::with_capacity(8);
    derive_splits_recursive(literals, sequences, 0, n_seq, &mut split_points);

    if split_points.is_empty() {
        // No beneficial splits found
        return vec![BlockPartition {
            literals: literals.to_vec(),
            sequences: sequences.to_vec(),
        }];
    }

    // Build partitions from split points
    // split_points contains sequence indices where we split.
    // Partitions: [0..split_points[0]], [split_points[0]..split_points[1]], ...
    let mut boundaries: Vec<usize> = Vec::with_capacity(split_points.len() + 2);
    boundaries.push(0);
    boundaries.extend_from_slice(&split_points);
    boundaries.push(n_seq);

    let mut partitions = Vec::with_capacity(boundaries.len() - 1);

    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        let part_lits = extract_partition_literals(literals, sequences, start, end);
        partitions.push(BlockPartition {
            literals: part_lits.to_vec(),
            sequences: sequences[start..end].to_vec(),
        });
    }

    partitions
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn no_split_small_data() {
        let data = vec![0u8; 1000];
        let boundaries = find_split_points(&data, 19);
        assert_eq!(boundaries, vec![0, 1000]);
    }

    #[test]
    fn no_split_uniform_data() {
        // Uniform data should not trigger a split (same distribution everywhere)
        let data = vec![42u8; 128 * 1024];
        let boundaries = find_split_points(&data, 19);
        assert_eq!(boundaries, vec![0, 128 * 1024]);
    }

    #[test]
    fn split_mixed_data() {
        // Create data with distinctly different regions
        let mut data = Vec::with_capacity(128 * 1024);

        // First half: repeating ASCII pattern
        while data.len() < 64 * 1024 {
            data.push(b'A' + (data.len() % 26) as u8);
        }

        // Second half: pseudo-random bytes
        let mut s = 0x12345678u64;
        while data.len() < 128 * 1024 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.push((s >> 33) as u8);
        }

        let boundaries = find_split_points(&data, 19);
        // Should find at least one split point
        assert!(
            boundaries.len() > 2,
            "expected split in mixed data, got boundaries: {:?}",
            boundaries
        );
        // The split should be somewhere around the 64KB boundary
        let split = boundaries[1];
        assert!(
            split >= 32 * 1024 && split <= 96 * 1024,
            "split at {} is not near the 64KB transition",
            split
        );
    }

    #[test]
    fn boundaries_are_sorted_and_cover_full_range() {
        let mut data = Vec::with_capacity(128 * 1024);
        // Create 4 distinct regions
        for region in 0..4u8 {
            let base = region * 64;
            while data.len() < (region as usize + 1) * 32 * 1024 {
                data.push(base + (data.len() % 8) as u8);
            }
        }

        let boundaries = find_split_points(&data, 11);
        // Check sorted
        for w in boundaries.windows(2) {
            assert!(w[0] < w[1], "boundaries not sorted: {:?}", boundaries);
        }
        // Check range
        assert_eq!(*boundaries.first().unwrap(), 0);
        assert_eq!(*boundaries.last().unwrap(), data.len());
    }

    #[test]
    fn fingerprint_divergence_basic() {
        let mut fp1 = Fingerprint::new();
        let mut fp2 = Fingerprint::new();

        // Create two very different fingerprints
        let data1: Vec<u8> = (0u16..1000).map(|i| (i % 10) as u8).collect();
        let data2: Vec<u8> = (0u16..1000).map(|i| (200 + i % 5) as u8).collect();

        fp1.record(&data1, 1, 8);
        fp2.record(&data2, 1, 8);

        assert!(
            fingerprints_diverge(&fp1, &fp2, 0, 8),
            "very different distributions should diverge"
        );
    }

    #[test]
    fn fingerprint_same_data_no_divergence() {
        let mut fp1 = Fingerprint::new();
        let mut fp2 = Fingerprint::new();

        let data: Vec<u8> = (0u16..1000).map(|i| (i % 26) as u8).collect();

        fp1.record(&data, 1, 10);
        fp2.record(&data, 1, 10);

        assert!(
            !fingerprints_diverge(&fp1, &fp2, THRESHOLD_PENALTY, 10),
            "identical distributions should not diverge"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn sequence_split_preserves_data() {
        use crate::encoding::compress_params::params_for_level;
        use crate::encoding::zstd_match::compress_block_zstd;

        // Create data with different statistical regions
        let mut data = Vec::with_capacity(100_000);
        let mut i = 0u32;
        while data.len() < 100_000 {
            if i % 100 < 50 {
                data.push(b'A' + (i % 26) as u8);
            } else {
                data.push(((i.wrapping_mul(2654435761) >> 16) & 0xFF) as u8);
            }
            i += 1;
        }

        let params = params_for_level(19, None);
        let block = compress_block_zstd(&data, &params);
        let partitions = split_sequences(&block.literals, &block.sequences);

        // Verify that all literals and sequences are preserved after splitting
        let mut all_lits = Vec::new();
        let mut all_seqs = Vec::new();
        for p in &partitions {
            all_lits.extend_from_slice(&p.literals);
            all_seqs.extend_from_slice(&p.sequences);
        }
        assert_eq!(all_lits, block.literals, "literal mismatch after split");
        assert_eq!(all_seqs, block.sequences, "sequence mismatch after split");
    }

    #[cfg(feature = "std")]
    #[test]
    fn sequence_split_improves_diverse_data() {
        use crate::encoding::compress_params::params_for_level;
        use crate::encoding::zstd_match::compress_block_zstd;

        // Create data with very different regions:
        // Region 1: highly repetitive text (low entropy)
        // Region 2: pseudo-random bytes (high entropy)
        let mut data = Vec::with_capacity(100_000);

        // 50KB of repetitive text pattern
        let phrase = b"Hello World! ";
        while data.len() < 50_000 {
            let remaining = 50_000 - data.len();
            let take = remaining.min(phrase.len());
            data.extend_from_slice(&phrase[..take]);
        }

        // 50KB of pseudo-random data
        let mut s = 0x12345678u64;
        while data.len() < 100_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.push((s >> 33) as u8);
        }

        let params = params_for_level(19, None);
        let block = compress_block_zstd(&data, &params);
        let partitions = split_sequences(&block.literals, &block.sequences);

        // Verify integrity
        let mut all_lits = Vec::new();
        let mut all_seqs = Vec::new();
        for p in &partitions {
            all_lits.extend_from_slice(&p.literals);
            all_seqs.extend_from_slice(&p.sequences);
        }
        assert_eq!(all_lits, block.literals);
        assert_eq!(all_seqs, block.sequences);

        // The whole block vs split blocks: measure actual sizes
        let whole_size = measure_sub_block_size(&block.literals, &block.sequences);
        let split_size: usize = partitions
            .iter()
            .map(|p| {
                measure_sub_block_size(&p.literals, &p.sequences) + 3 // block header
            })
            .sum::<usize>()
            - 3; // first block doesn't add extra header

        // If splitting occurred and saved bytes, the split_size should be <= whole_size
        // (This test validates the split decision logic is sound)
        if partitions.len() > 1 {
            assert!(
                split_size <= whole_size,
                "split should not increase size: split={} whole={}",
                split_size,
                whole_size
            );
        }
    }
}

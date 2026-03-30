//! Zstd match-finding engine supporting all strategies: Fast through BtUltra2.
//!
//! This module is a thin dispatch layer that selects the appropriate strategy
//! based on compression parameters. The actual algorithms live in:
//!
//! - [`super::match_state`] — Shared types and persistent cross-block state
//! - [`super::zstd_fast`] — Fast and DFast strategies
//! - [`super::zstd_lazy`] — Greedy, Lazy, Lazy2, and BtLazy2 strategies
//! - [`super::zstd_opt`] — BtOpt, BtUltra, and BtUltra2 optimal parsing
//!
//! # Strategy overview
//!
//! - **Fast** (levels 1-2): Single hash table lookup, step forward on miss.
//! - **DFast** (levels 3-4): Dual hash tables (short + long), take the longer match.
//! - **Greedy** (levels 5): Hash chains with best-match search.
//! - **Lazy** (levels 6-7): Greedy + check pos+1, use the better match.
//! - **Lazy2** (levels 8-12): Lazy + also check pos+2, three-way comparison.
//! - **BtLazy2** (levels 13-15): Binary tree match finder + lazy2 evaluation.
//! - **BtOpt/BtUltra/BtUltra2** (levels 16-22): Binary tree match finder with
//!   price-based optimal parsing (forward price table + backward trace).
//!
//! All functions are `#![forbid(unsafe_code)]` and operate on `&[u8]` slices.

use alloc::vec::Vec;

use super::compress_params::{CompressionParams, Strategy};

// Re-export public types from match_state
pub use super::match_state::{CompressedBlock, MatchState, RepCodes, SequenceOut};

// ---------------------------------------------------------------------------
// Core: compress_block_zstd
// ---------------------------------------------------------------------------

/// Compress a single block of source data using zstd match-finding algorithms.
///
/// Selects the appropriate strategy based on `params.strategy`:
/// Fast, DFast, Greedy, Lazy, Lazy2, BtLazy2, BtOpt, BtUltra, or BtUltra2.
///
/// Returns a [`CompressedBlock`] containing the literal bytes and match sequences,
/// ready for entropy encoding.
///
/// The sequences use the standard zstd offset encoding:
/// - `off_base` 1-3: repeat offsets
/// - `off_base` >= 4: real_offset = off_base - 3
#[inline]
pub fn compress_block_zstd(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    compress_block_zstd_with_dict(src, params, &[], &[1, 4, 8])
}

/// Compress a single block with optional dictionary content prepended as match history.
#[inline]
pub fn compress_block_zstd_with_dict(
    src: &[u8],
    params: &CompressionParams,
    dict_content: &[u8],
    initial_rep_offsets: &[u32; 3],
) -> CompressedBlock {
    if src.is_empty() {
        return CompressedBlock {
            literals: Vec::new(),
            sequences: Vec::new(),
        };
    }
    if dict_content.is_empty() {
        return match params.strategy {
            Strategy::Fast => super::zstd_fast::compress_fast(src, params),
            Strategy::DFast => super::zstd_fast::compress_dfast(src, params),
            Strategy::Greedy => super::zstd_lazy::compress_greedy(src, params),
            Strategy::Lazy => super::zstd_lazy::compress_lazy(src, params),
            Strategy::Lazy2 => super::zstd_lazy::compress_lazy2(src, params),
            Strategy::BtLazy2 => super::zstd_lazy::compress_btlazy2(src, params),
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                super::zstd_opt::compress_btopt(src, params)
            }
        };
    }
    let dict_len = dict_content.len();
    let mut combined = Vec::with_capacity(dict_len + src.len());
    combined.extend_from_slice(dict_content);
    combined.extend_from_slice(src);
    match params.strategy {
        Strategy::Fast => {
            super::zstd_fast::compress_fast_dict(&combined, dict_len, params, initial_rep_offsets)
        }
        Strategy::DFast => {
            super::zstd_fast::compress_dfast_dict(&combined, dict_len, params, initial_rep_offsets)
        }
        Strategy::Greedy => {
            super::zstd_lazy::compress_greedy_dict(&combined, dict_len, params, initial_rep_offsets)
        }
        Strategy::Lazy => {
            super::zstd_lazy::compress_lazy_dict(&combined, dict_len, params, initial_rep_offsets)
        }
        Strategy::Lazy2 => {
            super::zstd_lazy::compress_lazy2_dict(&combined, dict_len, params, initial_rep_offsets)
        }
        Strategy::BtLazy2 => super::zstd_lazy::compress_btlazy2_dict(
            &combined,
            dict_len,
            params,
            initial_rep_offsets,
        ),
        Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
            super::zstd_opt::compress_btopt_dict(&combined, dict_len, params, initial_rep_offsets)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::encoding::compress_params::params_for_level;

    /// Helper: verify that the sequences + literals can reconstruct the original data.
    fn verify_reconstruction(src: &[u8], block: &CompressedBlock) {
        let mut output: Vec<u8> = Vec::new();
        let mut lit_cursor = 0usize;
        let mut rep = [1u32, 4, 8];

        for seq in &block.sequences {
            let lit_end = lit_cursor + seq.lit_len as usize;
            assert!(
                lit_end <= block.literals.len(),
                "lit_cursor={lit_cursor}, lit_len={}, literals.len()={}",
                seq.lit_len,
                block.literals.len(),
            );
            output.extend_from_slice(&block.literals[lit_cursor..lit_end]);
            lit_cursor = lit_end;

            let real_offset = if seq.off_base >= 4 {
                let off = seq.off_base - 3;
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = off;
                off as usize
            } else if seq.off_base == 1 {
                if seq.lit_len == 0 {
                    let off = rep[1];
                    rep.swap(0, 1);
                    off as usize
                } else {
                    rep[0] as usize
                }
            } else if seq.off_base == 2 {
                if seq.lit_len == 0 {
                    let off = rep[2];
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = off;
                    off as usize
                } else {
                    let off = rep[1];
                    rep.swap(0, 1);
                    off as usize
                }
            } else {
                // off_base == 3
                if seq.lit_len == 0 {
                    let off = rep[0].wrapping_sub(1);
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = off;
                    off as usize
                } else {
                    let off = rep[2];
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = off;
                    off as usize
                }
            };

            assert!(
                real_offset > 0 && real_offset <= output.len(),
                "invalid offset: real_offset={real_offset}, output.len()={}, off_base={}",
                output.len(),
                seq.off_base,
            );
            let match_start = output.len() - real_offset;
            for i in 0..seq.match_len as usize {
                let byte = output[match_start + i];
                output.push(byte);
            }
        }

        if lit_cursor < block.literals.len() {
            output.extend_from_slice(&block.literals[lit_cursor..]);
        }

        assert_eq!(
            output.as_slice(),
            src,
            "reconstruction mismatch: output.len()={}, src.len()={}",
            output.len(),
            src.len(),
        );
    }

    // ---------------------------------------------------------------
    // All-zeros: should use repcodes heavily
    // ---------------------------------------------------------------

    #[test]
    fn all_zeros_fast() {
        let src = vec![0u8; 1024];
        let params = params_for_level(1, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(
            !block.sequences.is_empty(),
            "expected sequences for all-zero input"
        );
        let rep_count = block.sequences.iter().filter(|s| s.off_base <= 3).count();
        assert!(
            rep_count > block.sequences.len() / 2,
            "expected mostly repcodes for all-zero input, got {rep_count}/{} rep matches",
            block.sequences.len(),
        );
    }

    #[test]
    fn all_zeros_dfast() {
        let src = vec![0u8; 1024];
        let params = params_for_level(3, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    #[test]
    fn all_zeros_greedy() {
        let src = vec![0u8; 1024];
        let params = params_for_level(5, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    #[test]
    fn all_zeros_lazy() {
        let src = vec![0u8; 1024];
        let params = params_for_level(6, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    #[test]
    fn all_zeros_lazy2() {
        let src = vec![0u8; 1024];
        let params = params_for_level(9, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    // ---------------------------------------------------------------
    // Repetitive pattern
    // ---------------------------------------------------------------

    #[test]
    fn repetitive_pattern_fast() {
        let pattern = b"ABCDEFGH";
        let mut src = Vec::new();
        for _ in 0..128 {
            src.extend_from_slice(pattern);
        }
        let params = params_for_level(1, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(
            !block.sequences.is_empty(),
            "expected matches in repetitive data"
        );
        assert!(
            block.literals.len() < src.len() / 2,
            "expected compression from repetitive data, lits={} src={}",
            block.literals.len(),
            src.len(),
        );
    }

    #[test]
    fn repetitive_pattern_dfast() {
        let pattern = b"ABCDEFGH";
        let mut src = Vec::new();
        for _ in 0..128 {
            src.extend_from_slice(pattern);
        }
        let params = params_for_level(3, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    #[test]
    fn repetitive_pattern_greedy() {
        let pattern = b"ABCDEFGH";
        let mut src = Vec::new();
        for _ in 0..128 {
            src.extend_from_slice(pattern);
        }
        let params = params_for_level(5, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    #[test]
    fn repetitive_pattern_lazy() {
        let pattern = b"ABCDEFGH";
        let mut src = Vec::new();
        for _ in 0..128 {
            src.extend_from_slice(pattern);
        }
        let params = params_for_level(6, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    #[test]
    fn repetitive_pattern_lazy2() {
        let pattern = b"ABCDEFGH";
        let mut src = Vec::new();
        for _ in 0..128 {
            src.extend_from_slice(pattern);
        }
        let params = params_for_level(9, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(!block.sequences.is_empty());
    }

    // ---------------------------------------------------------------
    // Random data
    // ---------------------------------------------------------------

    #[test]
    fn random_data_fast() {
        let mut rng = 0x12345678u64;
        let mut src = vec![0u8; 4096];
        for b in src.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        let params = params_for_level(1, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
    }

    #[test]
    fn random_data_greedy() {
        let mut rng = 0x12345678u64;
        let mut src = vec![0u8; 4096];
        for b in src.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        let params = params_for_level(5, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
    }

    #[test]
    fn random_data_lazy2() {
        let mut rng = 0x12345678u64;
        let mut src = vec![0u8; 4096];
        for b in src.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        let params = params_for_level(9, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
    }

    // ---------------------------------------------------------------
    // Empty / tiny input
    // ---------------------------------------------------------------

    #[test]
    fn empty_input() {
        let src = vec![];
        let params = params_for_level(1, Some(0));
        let block = compress_block_zstd(&src, &params);
        assert!(block.sequences.is_empty());
        assert!(block.literals.is_empty());
    }

    #[test]
    fn tiny_input() {
        let src = vec![1, 2, 3];
        let params = params_for_level(1, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        assert!(block.sequences.is_empty());
        assert_eq!(block.literals.as_slice(), src.as_slice());
    }

    // ---------------------------------------------------------------
    // Mixed data: all strategies
    // ---------------------------------------------------------------

    #[test]
    fn mixed_data_all_strategies() {
        let mut src = Vec::new();
        src.extend_from_slice(b"HEADER__unique_data_here___");
        for _ in 0..20 {
            src.extend_from_slice(b"repeated_block_");
        }
        src.extend_from_slice(b"__separator__different_stuff__");
        for _ in 0..15 {
            src.extend_from_slice(b"another_repeat!");
        }

        for level in [1, 3, 5, 6, 9, 12, 13, 16, 22] {
            let params = params_for_level(level, Some(src.len() as u64));
            let block = compress_block_zstd(&src, &params);
            verify_reconstruction(&src, &block);
            assert!(
                !block.sequences.is_empty(),
                "level {level}: expected sequences in mixed data"
            );
        }
    }

    // ---------------------------------------------------------------
    // BT strategies
    // ---------------------------------------------------------------

    #[test]
    fn bt_strategies_reconstruction() {
        let mut src = Vec::new();
        for _ in 0..64 {
            src.extend_from_slice(b"bt_tree_test_data_pattern_");
        }

        for level in [13, 14, 15, 16, 17, 18, 19, 20, 22] {
            let params = params_for_level(level, Some(src.len() as u64));
            let block = compress_block_zstd(&src, &params);
            verify_reconstruction(&src, &block);
            assert!(
                !block.sequences.is_empty(),
                "level {level}: expected sequences in repetitive data"
            );
        }
    }

    #[test]
    fn bt_all_zeros() {
        let src = vec![0u8; 2048];
        for level in [13, 16, 19, 22] {
            let params = params_for_level(level, Some(src.len() as u64));
            let block = compress_block_zstd(&src, &params);
            verify_reconstruction(&src, &block);
            assert!(
                !block.sequences.is_empty(),
                "level {level}: expected sequences for all-zero input"
            );
        }
    }

    #[test]
    fn bt_random_data() {
        let mut rng = 0xDEADBEEFu64;
        let mut src = vec![0u8; 4096];
        for b in src.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        for level in [13, 16, 22] {
            let params = params_for_level(level, Some(src.len() as u64));
            let block = compress_block_zstd(&src, &params);
            verify_reconstruction(&src, &block);
        }
    }

    #[test]
    fn bt_mixed_data() {
        let mut src = Vec::new();
        src.extend_from_slice(b"UNIQUE_HEADER_DATA_123456789___");
        for _ in 0..30 {
            src.extend_from_slice(b"repeated_pattern_block__");
        }
        src.extend_from_slice(b"___separator_unique_content___");
        for _ in 0..20 {
            src.extend_from_slice(b"another_pattern!");
        }

        for level in [13, 15, 16, 18, 19, 22] {
            let params = params_for_level(level, Some(src.len() as u64));
            let block = compress_block_zstd(&src, &params);
            verify_reconstruction(&src, &block);
            assert!(
                !block.sequences.is_empty(),
                "level {level}: expected sequences in mixed data"
            );
        }
    }

    #[test]
    fn bt_better_than_lazy2_on_repetitive() {
        let mut src = Vec::new();
        for i in 0u8..20 {
            for _ in 0..10 {
                src.extend_from_slice(&[i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7]);
            }
        }

        let lazy2_params = params_for_level(12, Some(src.len() as u64));
        let bt_params = params_for_level(13, Some(src.len() as u64));

        let lazy2_block = compress_block_zstd(&src, &lazy2_params);
        let bt_block = compress_block_zstd(&src, &bt_params);

        verify_reconstruction(&src, &lazy2_block);
        verify_reconstruction(&src, &bt_block);

        assert!(
            bt_block.literals.len() <= lazy2_block.literals.len() + lazy2_block.literals.len() / 10,
            "BT (level 13) produced more literals than lazy2 (level 12): bt={}, lazy2={}",
            bt_block.literals.len(),
            lazy2_block.literals.len(),
        );
    }

    #[test]
    fn bt_large_block() {
        let mut data = Vec::new();
        for _ in 0..2000 {
            data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
        }
        assert!(data.len() > 80_000);

        for level in [13, 16, 19, 22] {
            let params = params_for_level(level, Some(data.len() as u64));
            let block = compress_block_zstd(&data, &params);
            verify_reconstruction(&data, &block);
        }
    }

    // ---------------------------------------------------------------
    // Round-trip tests
    // ---------------------------------------------------------------

    #[test]
    fn round_trip_via_full_encoder() {
        use crate::decoding::FrameDecoder;
        use crate::encoding::{CompressionLevel, compress_to_vec};

        let mut src = Vec::new();
        src.extend_from_slice(b"Hello, world! ");
        for _ in 0..100 {
            src.extend_from_slice(b"Hello, world! ");
        }

        let compressed = compress_to_vec(src.as_slice(), CompressionLevel::Fastest);

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(src.len());
        decoder
            .decode_all_to_vec(&compressed, &mut decoded)
            .unwrap();
        assert_eq!(decoded, src);

        let mut decoded2 = Vec::new();
        zstd::stream::copy_decode(compressed.as_slice(), &mut decoded2).unwrap();
        assert_eq!(decoded2, src);
    }

    // ---------------------------------------------------------------
    // Repcode tests
    // ---------------------------------------------------------------

    #[test]
    fn repcode_update_real_offset() {
        let mut rep = RepCodes::new();
        assert_eq!(rep.rep, [1, 4, 8]);
        rep.update(13, 5);
        assert_eq!(rep.rep, [10, 1, 4]);
        rep.update(23, 0);
        assert_eq!(rep.rep, [20, 10, 1]);
    }

    #[test]
    fn repcode_update_rep0() {
        let mut rep = RepCodes { rep: [10, 20, 30] };
        rep.update(1, 5);
        assert_eq!(rep.rep, [10, 20, 30]);
        rep.update(1, 0);
        assert_eq!(rep.rep, [20, 10, 30]);
    }

    #[test]
    fn repcode_update_rep1() {
        let mut rep = RepCodes { rep: [10, 20, 30] };
        rep.update(2, 5);
        assert_eq!(rep.rep, [20, 10, 30]);
    }

    #[test]
    fn repcode_update_rep2() {
        let mut rep = RepCodes { rep: [10, 20, 30] };
        rep.update(3, 5);
        assert_eq!(rep.rep, [30, 10, 20]);
    }

    // ---------------------------------------------------------------
    // Stress tests
    // ---------------------------------------------------------------

    #[test]
    fn all_same_byte_various_sizes() {
        for size in [16, 64, 256, 1000, 4096] {
            let src = vec![0xAA; size];
            for level in [1, 3, 5, 6, 9, 13, 16] {
                let params = params_for_level(level, Some(src.len() as u64));
                let block = compress_block_zstd(&src, &params);
                verify_reconstruction(&src, &block);
            }
        }
    }

    #[test]
    fn offset_encoding() {
        assert_eq!(4u32 - 3, 1);
        assert_eq!(100u32 - 3, 97);
    }

    #[test]
    fn large_block_reconstruction() {
        let mut data = Vec::new();
        for _ in 0..1500 {
            data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
        }
        assert!(data.len() > 60_000);

        for level in [1, 3, 5, 7, 9, 11, 13, 16, 22] {
            let params = params_for_level(level, Some(data.len() as u64));
            let block = compress_block_zstd(&data, &params);
            verify_reconstruction(&data, &block);
        }
    }

    #[test]
    fn bt_round_trip_c_zstd_decode() {
        use crate::decoding::FrameDecoder;
        use crate::encoding::{CompressionLevel, compress_to_vec};

        let mut src = Vec::new();
        src.extend_from_slice(b"Round-trip test for binary tree strategies. ");
        for _ in 0..200 {
            src.extend_from_slice(b"Repeated pattern for BT match finder testing. ");
        }

        for level in [13, 14, 15, 16, 17, 18, 19, 22] {
            let compressed = compress_to_vec(src.as_slice(), CompressionLevel::Level(level));

            let mut decoder = FrameDecoder::new();
            let mut decoded = Vec::with_capacity(src.len());
            decoder
                .decode_all_to_vec(&compressed, &mut decoded)
                .expect(&alloc::format!("our decoder failed at level {level}"));
            assert_eq!(decoded, src, "our decoder: mismatch at level {level}");

            let mut decoded_c = Vec::new();
            zstd::stream::copy_decode(compressed.as_slice(), &mut decoded_c)
                .expect(&alloc::format!("C zstd failed at level {level}"));
            assert_eq!(decoded_c, src, "C zstd: mismatch at level {level}");
        }
    }

    #[test]
    fn bt_round_trip_small_data() {
        use crate::encoding::{CompressionLevel, compress_to_vec};

        let mut src = Vec::new();
        for _ in 0..50 {
            src.extend_from_slice(b"small BT test ");
        }
        assert!(src.len() < 16 * 1024);

        for level in [13, 16, 19, 22] {
            let compressed = compress_to_vec(src.as_slice(), CompressionLevel::Level(level));

            let mut decoded = Vec::new();
            zstd::stream::copy_decode(compressed.as_slice(), &mut decoded)
                .expect(&alloc::format!("C zstd failed at level {level}"));
            assert_eq!(decoded, src, "C zstd: small data mismatch at level {level}");
        }
    }

    // ---------------------------------------------------------------
    // MatchState tests
    // ---------------------------------------------------------------

    #[test]
    fn match_state_window_update_small_blocks() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);
        assert!(ms.window().is_empty());

        let block1 = b"hello world data";
        ms.update_window_only(block1);
        assert_eq!(ms.window(), block1);

        let block2 = b" more data here";
        ms.update_window_only(block2);
        let expected: Vec<u8> = [block1.as_slice(), block2.as_slice()].concat();
        assert_eq!(ms.window(), expected.as_slice());
    }

    #[test]
    fn match_state_window_caps_at_window_size() {
        use crate::encoding::compress_params::CompressionParams;
        use crate::encoding::compress_params::Strategy;
        let params = CompressionParams {
            window_log: 4,
            chain_log: 4,
            hash_log: 4,
            search_log: 1,
            min_match: 4,
            target_length: 0,
            strategy: Strategy::Fast,
        };
        let mut ms = MatchState::new(&params);

        ms.update_window_only(&[0u8; 10]);
        ms.update_window_only(&[1u8; 10]);
        assert_eq!(ms.window().len(), 16);
        assert_eq!(&ms.window()[..6], &[0u8; 6]);
        assert_eq!(&ms.window()[6..], &[1u8; 10]);
    }

    #[test]
    fn match_state_compress_block_basic() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);

        let mut block1 = Vec::new();
        for _ in 0..50 {
            block1.extend_from_slice(b"block1_pattern ");
        }
        let result1 = ms.compress_block(&block1, &params);
        verify_reconstruction(&block1, &result1);
        assert!(!result1.sequences.is_empty());
        assert!(!ms.window().is_empty());
    }

    #[test]
    fn match_state_cross_block_finds_matches() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);

        let pattern = b"cross_block_match_test_data_";
        let mut block1 = Vec::new();
        for _ in 0..30 {
            block1.extend_from_slice(pattern);
        }
        let result1 = ms.compress_block(&block1, &params);
        verify_reconstruction(&block1, &result1);

        let mut block2 = Vec::new();
        for _ in 0..30 {
            block2.extend_from_slice(pattern);
        }
        let result2_with_history = ms.compress_block(&block2, &params);

        let result2_standalone = compress_block_zstd(&block2, &params);

        assert!(!result2_with_history.sequences.is_empty());
        assert!(!result2_standalone.sequences.is_empty());

        assert!(
            result2_with_history.literals.len() <= result2_standalone.literals.len(),
            "cross-block history should produce same or fewer literals: \
             with_history={}, standalone={}",
            result2_with_history.literals.len(),
            result2_standalone.literals.len(),
        );
    }

    #[test]
    fn match_state_rep_offsets_carry_over() {
        let params = params_for_level(5, None);
        let mut ms = MatchState::new(&params);
        assert_eq!(ms.rep_offsets(), &[1, 4, 8]);

        let mut block = Vec::new();
        for _ in 0..100 {
            block.extend_from_slice(b"ABCDEFGH");
        }
        let _result = ms.compress_block(&block, &params);

        let new_reps = ms.rep_offsets();
        assert!(
            new_reps != &[1, 4, 8] || block.len() < 16,
            "rep offsets should have changed after compressing repetitive data"
        );
    }

    #[test]
    fn match_state_seed_from_dict() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);

        let dict = b"dictionary content for seeding the window buffer";
        ms.seed_from_dict(dict, &[10, 20, 30]);

        assert_eq!(ms.window(), dict.as_slice());
        assert_eq!(ms.rep_offsets(), &[10, 20, 30]);
    }

    #[test]
    fn match_state_all_strategies() {
        let mut block1 = Vec::new();
        let mut block2 = Vec::new();
        for _ in 0..50 {
            block1.extend_from_slice(b"shared_pattern_data_here_");
            block2.extend_from_slice(b"shared_pattern_data_here_");
        }
        block2.extend_from_slice(b"unique_tail");

        for level in [1, 3, 5, 6, 9, 13, 16, 22] {
            let params = params_for_level(level, Some((block1.len() + block2.len()) as u64));
            let mut ms = MatchState::new(&params);

            let r1 = ms.compress_block(&block1, &params);
            verify_reconstruction(&block1, &r1);

            let r2 = ms.compress_block(&block2, &params);
            assert!(
                !r2.sequences.is_empty(),
                "level {level}: expected sequences in block 2"
            );
        }
    }
}

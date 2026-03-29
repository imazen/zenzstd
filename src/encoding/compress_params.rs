//! Zstd compression parameters matching the C reference implementation's `clevels.h` tables.
//!
//! Provides [`Strategy`], [`CompressionParams`], and [`params_for_level`] which return
//! the exact parameter sets from the upstream C zstd source for a given compression
//! level and source size.

/// Match-finding strategy, ordered from fastest/lowest-ratio to slowest/highest-ratio.
///
/// Values match the C `ZSTD_strategy` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Strategy {
    Fast = 1,
    DFast = 2,
    Greedy = 3,
    Lazy = 4,
    Lazy2 = 5,
    BtLazy2 = 6,
    BtOpt = 7,
    BtUltra = 8,
    BtUltra2 = 9,
}

/// All parameters that control zstd compression behavior for a single block/frame.
///
/// These correspond to the fields in `ZSTD_compressionParameters` from the C library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionParams {
    /// Log2 of the maximum back-reference distance (window size).
    pub window_log: u32,
    /// Log2 of the size of the chained hash table used for match finding.
    pub chain_log: u32,
    /// Log2 of the size of the hash table used for match finding.
    pub hash_log: u32,
    /// Log2 of the number of searches performed per position.
    pub search_log: u32,
    /// Minimum match length.
    pub min_match: u32,
    /// Optimal parser target length (strategy-dependent meaning).
    pub target_length: u32,
    /// Match-finding strategy.
    pub strategy: Strategy,
}

impl CompressionParams {
    /// Returns the window size in bytes: `1 << window_log`.
    #[inline]
    pub fn window_size(&self) -> usize {
        1 << self.window_log
    }

    /// Returns the hash table size in entries: `1 << hash_log`.
    #[inline]
    pub fn hash_table_size(&self) -> usize {
        1 << self.hash_log
    }

    /// Returns the chain table size in entries: `1 << chain_log`.
    #[inline]
    pub fn chain_table_size(&self) -> usize {
        1 << self.chain_log
    }

    /// Returns the search depth: `1 << search_log`.
    #[inline]
    pub fn search_depth(&self) -> usize {
        1 << self.search_log
    }
}

/// Maximum valid compression level.
const MAX_CLEVEL: i32 = 22;

/// Helper to build a `CompressionParams` from the table row format.
const fn p(
    window_log: u32,
    chain_log: u32,
    hash_log: u32,
    search_log: u32,
    min_match: u32,
    target_length: u32,
    strategy: Strategy,
) -> CompressionParams {
    CompressionParams {
        window_log,
        chain_log,
        hash_log,
        search_log,
        min_match,
        target_length,
        strategy,
    }
}

use Strategy::*;

/// Default table: srcSize > 256 KB (or unknown). Index 0 = level 0 (base/fallback).
static DEFAULT_TABLE: [CompressionParams; 23] = [
    p(19, 12, 13, 1, 6, 1, Fast),      // level 0 (base)
    p(19, 13, 14, 1, 7, 0, Fast),      // level 1
    p(20, 15, 16, 1, 6, 0, Fast),      // level 2
    p(21, 16, 17, 1, 5, 0, DFast),     // level 3
    p(21, 18, 18, 1, 5, 0, DFast),     // level 4
    p(21, 18, 19, 3, 5, 2, Greedy),    // level 5
    p(21, 18, 19, 3, 5, 4, Lazy),      // level 6
    p(21, 19, 20, 4, 5, 8, Lazy),      // level 7
    p(21, 19, 20, 4, 5, 16, Lazy2),    // level 8
    p(22, 20, 21, 4, 5, 16, Lazy2),    // level 9
    p(22, 21, 22, 5, 5, 16, Lazy2),    // level 10
    p(22, 21, 22, 6, 5, 16, Lazy2),    // level 11
    p(22, 22, 23, 6, 5, 32, Lazy2),    // level 12
    p(22, 22, 22, 4, 5, 32, BtLazy2),  // level 13
    p(22, 22, 23, 5, 5, 32, BtLazy2),  // level 14
    p(22, 23, 23, 6, 5, 32, BtLazy2),  // level 15
    p(22, 22, 22, 5, 5, 48, BtOpt),    // level 16
    p(23, 23, 22, 5, 4, 64, BtOpt),    // level 17
    p(23, 23, 22, 6, 3, 64, BtUltra),  // level 18
    p(23, 24, 22, 7, 3, 256, BtUltra2), // level 19
    p(25, 25, 23, 7, 3, 256, BtUltra2), // level 20
    p(26, 26, 24, 7, 3, 512, BtUltra2), // level 21
    p(27, 27, 25, 9, 3, 999, BtUltra2), // level 22
];

/// srcSize <= 256 KB table.
static TABLE_256KB: [CompressionParams; 23] = [
    p(18, 12, 13, 1, 5, 1, Fast),      // level 0
    p(18, 13, 14, 1, 6, 0, Fast),      // level 1
    p(18, 14, 14, 1, 5, 0, DFast),     // level 2
    p(18, 16, 16, 1, 4, 0, DFast),     // level 3
    p(18, 16, 17, 3, 5, 2, Greedy),    // level 4
    p(18, 17, 18, 5, 5, 2, Greedy),    // level 5
    p(18, 18, 19, 3, 5, 4, Lazy),      // level 6
    p(18, 18, 19, 4, 4, 4, Lazy),      // level 7
    p(18, 18, 19, 4, 4, 8, Lazy2),     // level 8
    p(18, 18, 19, 5, 4, 8, Lazy2),     // level 9
    p(18, 18, 19, 6, 4, 8, Lazy2),     // level 10
    p(18, 18, 19, 5, 4, 12, BtLazy2),  // level 11
    p(18, 19, 19, 7, 4, 12, BtLazy2),  // level 12
    p(18, 18, 19, 4, 4, 16, BtOpt),    // level 13
    p(18, 18, 19, 4, 3, 32, BtOpt),    // level 14
    p(18, 18, 19, 6, 3, 128, BtOpt),   // level 15
    p(18, 19, 19, 6, 3, 128, BtUltra), // level 16
    p(18, 19, 19, 8, 3, 256, BtUltra), // level 17
    p(18, 19, 19, 6, 3, 128, BtUltra2), // level 18
    p(18, 19, 19, 8, 3, 256, BtUltra2), // level 19
    p(18, 19, 19, 10, 3, 512, BtUltra2), // level 20
    p(18, 19, 19, 12, 3, 512, BtUltra2), // level 21
    p(18, 19, 19, 13, 3, 999, BtUltra2), // level 22
];

/// srcSize <= 128 KB table.
static TABLE_128KB: [CompressionParams; 23] = [
    p(17, 12, 12, 1, 5, 1, Fast),      // level 0
    p(17, 12, 13, 1, 6, 0, Fast),      // level 1
    p(17, 13, 15, 1, 5, 0, Fast),      // level 2
    p(17, 15, 16, 2, 5, 0, DFast),     // level 3
    p(17, 17, 17, 2, 4, 0, DFast),     // level 4
    p(17, 16, 17, 3, 4, 2, Greedy),    // level 5
    p(17, 16, 17, 3, 4, 4, Lazy),      // level 6
    p(17, 16, 17, 3, 4, 8, Lazy2),     // level 7
    p(17, 16, 17, 4, 4, 8, Lazy2),     // level 8
    p(17, 16, 17, 5, 4, 8, Lazy2),     // level 9
    p(17, 16, 17, 6, 4, 8, Lazy2),     // level 10
    p(17, 17, 17, 5, 4, 8, BtLazy2),   // level 11
    p(17, 18, 17, 7, 4, 12, BtLazy2),  // level 12
    p(17, 18, 17, 3, 4, 12, BtOpt),    // level 13
    p(17, 18, 17, 4, 3, 32, BtOpt),    // level 14
    p(17, 18, 17, 6, 3, 256, BtOpt),   // level 15
    p(17, 18, 17, 6, 3, 128, BtUltra), // level 16
    p(17, 18, 17, 8, 3, 256, BtUltra), // level 17
    p(17, 18, 17, 10, 3, 512, BtUltra), // level 18
    p(17, 18, 17, 5, 3, 256, BtUltra2), // level 19
    p(17, 18, 17, 7, 3, 512, BtUltra2), // level 20
    p(17, 18, 17, 9, 3, 512, BtUltra2), // level 21
    p(17, 18, 17, 11, 3, 999, BtUltra2), // level 22
];

/// srcSize <= 16 KB table.
static TABLE_16KB: [CompressionParams; 23] = [
    p(14, 12, 13, 1, 5, 1, Fast),      // level 0
    p(14, 14, 15, 1, 5, 0, Fast),      // level 1
    p(14, 14, 15, 1, 4, 0, Fast),      // level 2
    p(14, 14, 15, 2, 4, 0, DFast),     // level 3
    p(14, 14, 14, 4, 4, 2, Greedy),    // level 4
    p(14, 14, 14, 3, 4, 4, Lazy),      // level 5
    p(14, 14, 14, 4, 4, 8, Lazy2),     // level 6
    p(14, 14, 14, 6, 4, 8, Lazy2),     // level 7
    p(14, 14, 14, 8, 4, 8, Lazy2),     // level 8
    p(14, 15, 14, 5, 4, 8, BtLazy2),   // level 9
    p(14, 15, 14, 9, 4, 8, BtLazy2),   // level 10
    p(14, 15, 14, 3, 4, 12, BtOpt),    // level 11
    p(14, 15, 14, 4, 3, 24, BtOpt),    // level 12
    p(14, 15, 14, 5, 3, 32, BtUltra),  // level 13
    p(14, 15, 15, 6, 3, 64, BtUltra),  // level 14
    p(14, 15, 15, 7, 3, 256, BtUltra), // level 15
    p(14, 15, 15, 5, 3, 48, BtUltra2), // level 16
    p(14, 15, 15, 6, 3, 128, BtUltra2), // level 17
    p(14, 15, 15, 7, 3, 256, BtUltra2), // level 18
    p(14, 15, 15, 8, 3, 256, BtUltra2), // level 19
    p(14, 15, 15, 8, 3, 512, BtUltra2), // level 20
    p(14, 15, 15, 9, 3, 512, BtUltra2), // level 21
    p(14, 15, 15, 10, 3, 999, BtUltra2), // level 22
];

/// Returns the compression parameters for a given level and optional source size.
///
/// Levels are clamped to `0..=22`. Negative levels are treated as 0.
///
/// The source size determines which parameter table is used:
/// - `None` or `> 256 KB`: default table
/// - `<= 256 KB`: 256 KB table
/// - `<= 128 KB`: 128 KB table
/// - `<= 16 KB`: 16 KB table
pub fn params_for_level(level: i32, src_size: Option<u64>) -> CompressionParams {
    let level = level.clamp(0, MAX_CLEVEL) as usize;

    let table = match src_size {
        Some(s) if s <= 16 * 1024 => &TABLE_16KB,
        Some(s) if s <= 128 * 1024 => &TABLE_128KB,
        Some(s) if s <= 256 * 1024 => &TABLE_256KB,
        _ => &DEFAULT_TABLE,
    };

    table[level]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level3_default_params() {
        let p = params_for_level(3, None);
        assert_eq!(p.window_log, 21);
        assert_eq!(p.chain_log, 16);
        assert_eq!(p.hash_log, 17);
        assert_eq!(p.search_log, 1);
        assert_eq!(p.min_match, 5);
        assert_eq!(p.target_length, 0);
        assert_eq!(p.strategy, Strategy::DFast);
    }

    #[test]
    fn level3_default_helpers() {
        let p = params_for_level(3, None);
        assert_eq!(p.window_size(), 1 << 21);
        assert_eq!(p.hash_table_size(), 1 << 17);
        assert_eq!(p.chain_table_size(), 1 << 16);
        assert_eq!(p.search_depth(), 1 << 1);
    }

    #[test]
    fn negative_level_clamps_to_zero() {
        let p = params_for_level(-5, None);
        let p0 = params_for_level(0, None);
        assert_eq!(p, p0);
    }

    #[test]
    fn excessive_level_clamps_to_22() {
        let p = params_for_level(99, None);
        let p22 = params_for_level(22, None);
        assert_eq!(p, p22);
    }

    #[test]
    fn small_source_selects_16kb_table() {
        let p = params_for_level(3, Some(1024));
        // 16KB table level 3
        assert_eq!(p.window_log, 14);
        assert_eq!(p.chain_log, 14);
        assert_eq!(p.hash_log, 15);
        assert_eq!(p.search_log, 2);
        assert_eq!(p.min_match, 4);
        assert_eq!(p.target_length, 0);
        assert_eq!(p.strategy, Strategy::DFast);
    }

    #[test]
    fn source_128kb_selects_128kb_table() {
        let p = params_for_level(1, Some(128 * 1024));
        // 128KB table level 1
        assert_eq!(p.window_log, 17);
        assert_eq!(p.chain_log, 12);
        assert_eq!(p.hash_log, 13);
        assert_eq!(p.search_log, 1);
        assert_eq!(p.min_match, 6);
        assert_eq!(p.target_length, 0);
        assert_eq!(p.strategy, Strategy::Fast);
    }

    #[test]
    fn source_256kb_selects_256kb_table() {
        let p = params_for_level(22, Some(256 * 1024));
        // 256KB table level 22
        assert_eq!(p.window_log, 18);
        assert_eq!(p.chain_log, 19);
        assert_eq!(p.hash_log, 19);
        assert_eq!(p.search_log, 13);
        assert_eq!(p.min_match, 3);
        assert_eq!(p.target_length, 999);
        assert_eq!(p.strategy, Strategy::BtUltra2);
    }

    #[test]
    fn large_source_uses_default_table() {
        let p = params_for_level(22, Some(1_000_000));
        // Default table level 22
        assert_eq!(p.window_log, 27);
        assert_eq!(p.chain_log, 27);
        assert_eq!(p.hash_log, 25);
        assert_eq!(p.search_log, 9);
        assert_eq!(p.min_match, 3);
        assert_eq!(p.target_length, 999);
        assert_eq!(p.strategy, Strategy::BtUltra2);
    }

    #[test]
    fn none_source_uses_default_table() {
        let p_none = params_for_level(10, None);
        let p_large = params_for_level(10, Some(1_000_000));
        assert_eq!(p_none, p_large);
    }

    #[test]
    fn boundary_16kb_exact() {
        // Exactly 16KB should use the 16KB table
        let p = params_for_level(0, Some(16 * 1024));
        assert_eq!(p.window_log, 14);
    }

    #[test]
    fn boundary_16kb_plus_one() {
        // 16KB + 1 should use the 128KB table
        let p = params_for_level(0, Some(16 * 1024 + 1));
        assert_eq!(p.window_log, 17);
    }

    #[test]
    fn boundary_128kb_plus_one() {
        // 128KB + 1 should use the 256KB table
        let p = params_for_level(0, Some(128 * 1024 + 1));
        assert_eq!(p.window_log, 18);
    }

    #[test]
    fn boundary_256kb_plus_one() {
        // 256KB + 1 should use the default table
        let p = params_for_level(0, Some(256 * 1024 + 1));
        assert_eq!(p.window_log, 19);
    }

    #[test]
    fn all_levels_valid_default() {
        for level in 0..=22 {
            let p = params_for_level(level, None);
            // Strategy should be in valid range
            assert!((p.strategy as u8) >= 1 && (p.strategy as u8) <= 9);
            // min_match should be 3..=7 for all levels
            assert!(p.min_match >= 3 || level <= 2, "level {level} min_match={}", p.min_match);
        }
    }

    #[test]
    fn strategy_ordering() {
        assert!(Strategy::Fast < Strategy::DFast);
        assert!(Strategy::DFast < Strategy::Greedy);
        assert!(Strategy::Greedy < Strategy::Lazy);
        assert!(Strategy::Lazy < Strategy::Lazy2);
        assert!(Strategy::Lazy2 < Strategy::BtLazy2);
        assert!(Strategy::BtLazy2 < Strategy::BtOpt);
        assert!(Strategy::BtOpt < Strategy::BtUltra);
        assert!(Strategy::BtUltra < Strategy::BtUltra2);
    }
}

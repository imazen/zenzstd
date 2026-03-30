//! Shared state and types for all zstd match-finding strategies.
//!
//! This module corresponds to the shared types from `zstd_compress_internal.h`
//! in the C reference implementation. It contains:
//!
//! - [`MatchState`]: persistent cross-block state (window, rep offsets, tables)
//! - [`RepCodes`]: offset repeat-code state machine
//! - [`CompressedBlock`]: output of a compression pass (literals + sequences)
//! - [`SequenceOut`]: a single match sequence
//! - [`MatchCandidate`]: internal match candidate during search

use alloc::vec::Vec;

use super::compress_params::{CompressionParams, Strategy};
use super::hash::{count_match, hash_ptr};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single output sequence from the match finder.
///
/// Uses the same offset encoding as `crate::blocks::sequence_section::Sequence`:
/// - `off_base` 1-3: repcode match (repeat offset 0, 1, 2)
/// - `off_base` >= 4: real offset, where the actual byte offset = `off_base - 3`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceOut {
    /// Encoded offset (1-3 = repcodes, >=4 = real offset + 3).
    pub off_base: u32,
    /// Number of literal bytes preceding this match.
    pub lit_len: u32,
    /// Length of the match in bytes (minimum 3).
    pub match_len: u32,
}

/// Result of `compress_block_zstd`: the sequences and literal bytes for one block.
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// The literal bytes, in order. The sum of all `lit_len` fields plus any
    /// trailing literals equals `literals.len()`.
    pub literals: Vec<u8>,
    /// The match sequences in forward order.
    pub sequences: Vec<SequenceOut>,
}

// ---------------------------------------------------------------------------
// RepCodes
// ---------------------------------------------------------------------------

/// Offset repeat-code state, tracking the three most recent offsets.
///
/// Initialized to `[1, 4, 8]` per the zstd spec (these are the actual byte
/// offsets, not the off_base encoding).
#[derive(Debug, Clone)]
pub struct RepCodes {
    pub rep: [u32; 3],
}

impl RepCodes {
    /// Create a fresh repcode state with the zstd default initial offsets.
    pub fn new() -> Self {
        Self { rep: [1, 4, 8] }
    }

    /// Update repcodes after encoding a sequence with the given `off_base`.
    ///
    /// - off_base 1: rep[0] used, no rotation needed.
    /// - off_base 2: swap rep[0] and rep[1].
    /// - off_base 3: rotate rep[2] to front.
    /// - off_base >= 4: push real offset, shift others down.
    #[inline]
    pub fn update(&mut self, off_base: u32, lit_len: u32) {
        if off_base >= 4 {
            // Real offset: push onto stack
            let real_offset = off_base - 3;
            self.rep[2] = self.rep[1];
            self.rep[1] = self.rep[0];
            self.rep[0] = real_offset;
        } else if off_base == 3 {
            if lit_len == 0 {
                // When ll==0, off_base 3 means rep[0]-1
                let new_off = self.rep[0].wrapping_sub(1);
                self.rep[2] = self.rep[1];
                self.rep[1] = self.rep[0];
                self.rep[0] = new_off;
            } else {
                let tmp = self.rep[2];
                self.rep[2] = self.rep[1];
                self.rep[1] = self.rep[0];
                self.rep[0] = tmp;
            }
        } else if off_base == 2 {
            if lit_len == 0 {
                // When ll==0, off_base 2 means rep[2]
                let tmp = self.rep[2];
                self.rep[2] = self.rep[1];
                self.rep[1] = self.rep[0];
                self.rep[0] = tmp;
            } else {
                let tmp = self.rep[1];
                self.rep[1] = self.rep[0];
                self.rep[0] = tmp;
            }
        }
        // off_base == 1: rep[0] stays as is (or rep[1] when ll==0)
        else if lit_len == 0 {
            // off_base 1 with ll==0 means rep[1]
            let tmp = self.rep[1];
            self.rep[1] = self.rep[0];
            self.rep[0] = tmp;
        }
    }

    /// Get the actual byte offset for a given repcode index (0, 1, 2)
    /// considering whether literal length is zero (which shifts semantics).
    #[inline]
    pub fn get_offset(&self, rep_idx: usize, lit_len: u32) -> u32 {
        if lit_len == 0 {
            match rep_idx {
                0 => self.rep[1],
                1 => self.rep[2],
                2 => self.rep[0].wrapping_sub(1),
                _ => 0,
            }
        } else {
            self.rep[rep_idx]
        }
    }

    /// Get the off_base value for a given repcode index.
    #[inline]
    pub fn off_base_for_rep(rep_idx: usize) -> u32 {
        (rep_idx as u32) + 1
    }
}

impl Default for RepCodes {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MatchCandidate
// ---------------------------------------------------------------------------

/// A match found by the binary tree, used by the optimal parser.
/// Matches are returned sorted by increasing length (each has a distinct offset).
#[derive(Debug, Clone, Copy)]
pub struct MatchFound {
    /// Encoded off_base (>=4 for real offsets).
    pub off_base: u32,
    /// Match length in bytes.
    pub len: u32,
}

/// Internal representation of a match candidate during search.
#[derive(Debug, Clone, Copy)]
pub struct MatchCandidate {
    /// Encoded off_base (1-3 for repcodes, >=4 for real offset + 3).
    pub off_base: u32,
    /// Match length in bytes.
    pub match_len: u32,
}

impl MatchCandidate {
    #[inline]
    pub fn gain(&self) -> i64 {
        // Heuristic from zstd: longer matches are better, closer offsets are better.
        // gain = match_len * 4 - log2(offset)
        let offset_cost = if self.off_base < 4 {
            // Repcode: very cheap
            1i64
        } else {
            let real_offset = self.off_base - 3;
            if real_offset == 0 {
                1
            } else {
                real_offset.ilog2() as i64 + 1
            }
        };
        (self.match_len as i64) * 4 - offset_cost
    }

    /// Returns true if `other` is a better match than `self` when found one
    /// position later (lazy evaluation). The +1 position costs one literal.
    #[inline]
    pub fn is_better_lazy(&self, other: &MatchCandidate) -> bool {
        // The new match at pos+1 needs to compensate for the literal we emit.
        // Standard heuristic: other is better if its gain exceeds ours + threshold.
        other.gain() > self.gain() + 4
    }
}

// ---------------------------------------------------------------------------
// Repcode checking helper
// ---------------------------------------------------------------------------

/// Try to find a repcode match at position `pos` in `src`.
/// Returns the best repcode match (longest), or None.
#[inline]
pub fn try_repcodes(
    src: &[u8],
    pos: usize,
    rep: &RepCodes,
    min_match: u32,
    lit_len: u32,
) -> Option<MatchCandidate> {
    let remaining = src.len() - pos;
    if remaining < min_match as usize {
        return None;
    }

    let mut best: Option<MatchCandidate> = None;

    for rep_idx in 0..3 {
        let offset = rep.get_offset(rep_idx, lit_len);
        if offset == 0 || (offset as usize) > pos {
            continue;
        }
        let ref_pos = pos - offset as usize;

        // Check if at least min_match bytes match
        let ml = count_match(&src[pos..], &src[ref_pos..]);
        if ml >= min_match as usize {
            let candidate = MatchCandidate {
                off_base: RepCodes::off_base_for_rep(rep_idx),
                match_len: ml as u32,
            };
            if best.map_or(true, |b| candidate.match_len > b.match_len) {
                best = Some(candidate);
            }
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Emit helpers
// ---------------------------------------------------------------------------

/// Finalize the block: collect trailing literals and build CompressedBlock.
pub fn build_block(
    src: &[u8],
    sequences: Vec<SequenceOut>,
    literals: Vec<u8>,
    anchor: usize,
) -> CompressedBlock {
    let mut lits = literals;
    // Append any trailing literals after the last match
    if anchor < src.len() {
        lits.extend_from_slice(&src[anchor..]);
    }
    CompressedBlock {
        literals: lits,
        sequences,
    }
}

/// Build block for dict-mode: trailing literals from anchor to end of combined buffer.
pub fn build_block_dict(
    combined: &[u8],
    _dict_len: usize,
    sequences: Vec<SequenceOut>,
    literals: Vec<u8>,
    anchor: usize,
) -> CompressedBlock {
    let mut lits = literals;
    if anchor < combined.len() {
        lits.extend_from_slice(&combined[anchor..]);
    }
    CompressedBlock {
        literals: lits,
        sequences,
    }
}

/// Emit a match sequence: append literals, push the sequence, update repcodes.
///
/// This is the "cold" path — called only when a match is found. Keeping it
/// out-of-line lets LLVM optimize the hot miss-step loop tightly.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[cold]
pub fn emit_match_fast(
    src: &[u8],
    literals: &mut Vec<u8>,
    sequences: &mut Vec<SequenceOut>,
    rep: &mut RepCodes,
    anchor: usize,
    pos: usize,
    off_base: u32,
    match_len: u32,
) {
    literals.extend_from_slice(&src[anchor..pos]);
    let lit_len = (pos - anchor) as u32;
    let seq = SequenceOut {
        off_base,
        lit_len,
        match_len,
    };
    rep.update(seq.off_base, lit_len);
    sequences.push(seq);
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

/// Compute hash of `src[pos..]` using a function selected by `min_match`.
#[inline(always)]
pub fn hash_at(src: &[u8], pos: usize, hash_log: u32, min_match: u32) -> usize {
    hash_ptr(&src[pos..], hash_log, min_match)
}

/// Insert hash table entries for positions in `start..end` with step 1.
#[inline(never)]
pub fn insert_hashes_dense(
    htable: &mut [u32],
    src: &[u8],
    start: usize,
    end: usize,
    hash_log: u32,
    min_match: u32,
) {
    let mask = htable.len() - 1;
    let mut p = start;
    while p < end {
        let h = hash_at(src, p, hash_log, min_match);
        htable[h & mask] = p as u32;
        p += 1;
    }
}

/// Insert hash table entries for positions in `start..end` with step 4.
#[inline(never)]
pub fn insert_hashes_sparse(
    htable: &mut [u32],
    src: &[u8],
    start: usize,
    end: usize,
    hash_log: u32,
    min_match: u32,
) {
    let mask = htable.len() - 1;
    let mut p = start;
    while p < end {
        let h = hash_at(src, p, hash_log, min_match);
        htable[h & mask] = p as u32;
        p += 4;
    }
}

/// Prefill a raw hash table slice with dict positions.
pub fn prefill_hash_table_ext(
    table: &mut [u32],
    hash_log: u32,
    combined: &[u8],
    dict_len: usize,
    min_match: u32,
) {
    if dict_len < 8 {
        return;
    }
    let end = dict_len.saturating_sub(7);
    for pos in 0..end {
        let h = hash_ptr(&combined[pos..], hash_log, min_match);
        table[h] = pos as u32;
    }
}

// ---------------------------------------------------------------------------
// Hash + Chain table helpers
// ---------------------------------------------------------------------------

/// Prefill raw hash + chain table slices with dict positions.
pub fn prefill_hash_chain_ext(
    htable: &mut [u32],
    chain: &mut [u32],
    hash_log: u32,
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
) {
    if dict_len < 8 {
        return;
    }
    let chain_mask = (chain.len() as u32).wrapping_sub(1);
    let end = dict_len.saturating_sub(7);
    for pos in 0..end {
        let h = hash_ptr(&combined[pos..], hash_log, params.min_match);
        let prev = htable[h];
        htable[h] = pos as u32;
        chain[(pos as u32 & chain_mask) as usize] = prev;
    }
}

/// Search hash chain using raw table slices. Returns best match at `pos`.
///
/// CRITICAL: This now implements the C zstd `nextToUpdate` concept.
/// The `next_to_update` parameter tracks where we stopped inserting.
/// Before searching, we catch up by inserting all deferred positions
/// from `next_to_update` to `pos`. When `lazy_skipping` is true, we
/// only insert one position (the current one) instead of catching up fully.
pub fn search_hash_chain_ext(
    src: &[u8],
    pos: usize,
    htable: &mut [u32],
    chain: &mut [u32],
    rep: &RepCodes,
    params: &CompressionParams,
    lit_len: u32,
    next_to_update: &mut usize,
    lazy_skipping: bool,
) -> Option<MatchCandidate> {
    let min_match = params.min_match.max(4) as usize;
    let search_depth = params.search_depth();
    let window_size = params.window_size();
    let chain_mask = (chain.len() as u32).wrapping_sub(1);

    if pos + 8 > src.len() {
        return None;
    }

    // First check repcodes
    let mut best = try_repcodes(src, pos, rep, min_match as u32, lit_len);

    // Catch up: insert positions from next_to_update to pos
    // This is the C zstd `ZSTD_insertAndFindFirstIndex_internal` pattern.
    let mut idx = *next_to_update;
    while idx < pos {
        if idx + 8 <= src.len() {
            let h = hash_ptr(&src[idx..], params.hash_log, params.min_match);
            let prev = htable[h];
            htable[h] = idx as u32;
            chain[(idx as u32 & chain_mask) as usize] = prev;
        }
        idx += 1;
        // When lazy skipping, only insert one position then jump to target
        if lazy_skipping {
            break;
        }
    }
    *next_to_update = pos;

    // Hash lookup at current position
    let h = hash_ptr(&src[pos..], params.hash_log, params.min_match);
    let mut candidate_pos = htable[h] as usize;

    // Insert current position into hash table + chain
    let prev = htable[h];
    htable[h] = pos as u32;
    chain[(pos as u32 & chain_mask) as usize] = prev;

    // Update next_to_update past current position
    *next_to_update = pos + 1;

    let min_chain = if pos > chain.len() {
        pos - chain.len()
    } else {
        0
    };

    let mut depth = search_depth;
    while depth > 0 && candidate_pos > 0 && candidate_pos < pos {
        let dist = pos - candidate_pos;
        if dist > window_size {
            break;
        }

        // Pre-check: compare bytes at match[ml-3] before full count_match
        // This is a key optimization from C zstd that avoids expensive
        // count_match calls when the match is clearly shorter than best.
        if let Some(ref b) = best {
            let ml = b.match_len as usize;
            if ml >= 4 {
                let check_pos = ml.saturating_sub(3);
                if check_pos < src.len() - pos && check_pos < src.len() - candidate_pos {
                    if src[pos + check_pos] != src[candidate_pos + check_pos] {
                        // Walk the chain
                        let next = chain[(candidate_pos as u32 & chain_mask) as usize] as usize;
                        if next >= candidate_pos || next <= min_chain {
                            break;
                        }
                        candidate_pos = next;
                        depth -= 1;
                        continue;
                    }
                }
            }
        }

        let ml = count_match(&src[pos..], &src[candidate_pos..]);
        if ml >= min_match {
            let cand = MatchCandidate {
                off_base: dist as u32 + 3,
                match_len: ml as u32,
            };
            if best.map_or(true, |b| {
                cand.match_len > b.match_len
                    || (cand.match_len == b.match_len && cand.gain() > b.gain())
            }) {
                best = Some(cand);
            }
        }

        // Walk the chain
        let next = chain[(candidate_pos as u32 & chain_mask) as usize] as usize;
        if next >= candidate_pos || next <= min_chain {
            break;
        }
        candidate_pos = next;
        depth -= 1;
    }

    best
}

/// Insert position into hash + chain tables (raw slices) without searching.
/// Updates next_to_update.
#[inline]
pub fn insert_hash_chain_ext(
    src: &[u8],
    pos: usize,
    htable: &mut [u32],
    chain: &mut [u32],
    params: &CompressionParams,
) {
    if pos + 8 > src.len() {
        return;
    }
    let chain_mask = (chain.len() as u32).wrapping_sub(1);
    let h = hash_ptr(&src[pos..], params.hash_log, params.min_match);
    let prev = htable[h];
    htable[h] = pos as u32;
    chain[(pos as u32 & chain_mask) as usize] = prev;
}

// ---------------------------------------------------------------------------
// MatchState — persistent state across blocks for cross-block matching
// ---------------------------------------------------------------------------

/// Persistent match state that carries across blocks within a frame.
///
/// When compressing multi-block data, each block can reference matches in
/// previous blocks (up to `window_size` bytes back). Without `MatchState`,
/// each block starts fresh and cannot find cross-block matches, hurting
/// compression ratio.
///
/// # Usage
///
/// ```ignore
/// let params = params_for_level(3, None);
/// let mut state = MatchState::new(&params);
///
/// let block1 = state.compress_block(&data[..block_size], &params);
/// let block2 = state.compress_block(&data[block_size..], &params);
/// // block2 can find matches that reference data from block1
/// ```
pub struct MatchState {
    /// Window buffer — holds the last `window_size` bytes from previous blocks
    /// so the current block can reference them for matches.
    window: Vec<u8>,
    /// Maximum window size in bytes.
    window_size: usize,
    /// Repeat offsets carried over from the previous block.
    rep_offsets: [u32; 3],
    /// Reusable hash table storage (avoids per-block allocation).
    hash_table: Vec<u32>,
    /// Reusable secondary hash table (for DFast long table).
    hash_table2: Vec<u32>,
    /// Reusable chain table storage (for Greedy/Lazy/Lazy2).
    chain_table: Vec<u32>,
    /// Current hash_log the tables are sized for.
    table_hash_log: u32,
    /// Current hash_log for the secondary table.
    table_hash_log2: u32,
    /// Current chain_log the chain table is sized for.
    table_chain_log: u32,
    /// Length of the combined buffer (window + src) used in the previous
    /// compression call. Used to compute the shift when adjusting table
    /// entries for cross-block persistence.
    prev_combined_len: usize,
    /// Whether the hash/chain tables contain valid entries from a previous block.
    /// When true, we shift entries instead of clearing and repopulating.
    tables_populated: bool,
}

impl MatchState {
    /// Create a new `MatchState` for the given compression parameters.
    pub fn new(params: &CompressionParams) -> Self {
        Self {
            window: Vec::new(),
            window_size: params.window_size(),
            rep_offsets: [1, 4, 8],
            hash_table: Vec::new(),
            hash_table2: Vec::new(),
            chain_table: Vec::new(),
            table_hash_log: 0,
            table_hash_log2: 0,
            table_chain_log: 0,
            prev_combined_len: 0,
            tables_populated: false,
        }
    }

    /// Reset the match state for a new frame (clears window, resets rep offsets).
    pub fn reset(&mut self, params: &CompressionParams) {
        self.window.clear();
        self.window_size = params.window_size();
        self.rep_offsets = [1, 4, 8];
        self.prev_combined_len = 0;
        self.tables_populated = false;
    }

    /// Ensure the hash table is sized for `hash_log`, clearing it.
    pub fn ensure_hash_table(&mut self, hash_log: u32) {
        let size = 1usize << hash_log;
        if self.table_hash_log != hash_log || self.hash_table.len() != size {
            self.hash_table.resize(size, 0);
            self.table_hash_log = hash_log;
        }
        self.hash_table.fill(0);
    }

    /// Ensure the secondary hash table is sized for `hash_log`, clearing it.
    pub fn ensure_hash_table2(&mut self, hash_log: u32) {
        let size = 1usize << hash_log;
        if self.table_hash_log2 != hash_log || self.hash_table2.len() != size {
            self.hash_table2.resize(size, 0);
            self.table_hash_log2 = hash_log;
        }
        self.hash_table2.fill(0);
    }

    /// Ensure the chain table is sized for `chain_log`, clearing it.
    pub fn ensure_chain_table(&mut self, chain_log: u32) {
        let size = 1usize << chain_log;
        if self.table_chain_log != chain_log || self.chain_table.len() != size {
            self.chain_table.resize(size, 0);
            self.table_chain_log = chain_log;
        }
        self.chain_table.fill(0);
    }

    /// Get the current repeat offsets.
    pub fn rep_offsets(&self) -> &[u32; 3] {
        &self.rep_offsets
    }

    /// Get the current window content.
    pub fn window(&self) -> &[u8] {
        &self.window
    }

    /// Get mutable reference to hash table.
    pub fn hash_table_mut(&mut self) -> &mut Vec<u32> {
        &mut self.hash_table
    }

    /// Get mutable reference to secondary hash table.
    pub fn hash_table2_mut(&mut self) -> &mut Vec<u32> {
        &mut self.hash_table2
    }

    /// Get mutable reference to chain table.
    pub fn chain_table_mut(&mut self) -> &mut Vec<u32> {
        &mut self.chain_table
    }

    /// Compress a block using persistent cross-block state.
    ///
    /// Cross-block matching works by keeping hash/chain table entries valid
    /// across blocks. Instead of clearing the tables each block and
    /// repopulating from the window prefix (which loses fine-grained chain
    /// entries), we shift existing entries to account for the buffer layout
    /// change between blocks.
    ///
    /// Between blocks, the combined buffer changes from
    /// `[old_window | old_src]` to `[new_window | new_src]`, where
    /// `new_window` is the tail of the old combined buffer. The shift amount
    /// is `prev_combined_len - new_window.len()`: old positions are adjusted
    /// by this delta, and entries that fall below zero (out of the window)
    /// are clamped to zero (treated as empty by the match finder).
    pub fn compress_block(&mut self, src: &[u8], params: &CompressionParams) -> CompressedBlock {
        let block = if self.window.is_empty() {
            // First block (no window): clear tables and compress from scratch
            let rep = self.rep_offsets;
            let result = self.compress_block_with_tables(src, params, &[], &rep);
            self.prev_combined_len = src.len();
            self.tables_populated = true;
            result
        } else if self.tables_populated {
            // Subsequent block with persistent tables: shift entries instead
            // of clearing and repopulating.
            let rep = self.rep_offsets;
            let dict_len = self.window.len();
            let shift = self.prev_combined_len.saturating_sub(dict_len);

            // Shift hash/chain table entries to account for the buffer trim.
            // Entries pointing to data that has been trimmed (< shift) become 0.
            if shift > 0 {
                let shift32 = shift as u32;
                for entry in self.hash_table.iter_mut() {
                    *entry = entry.saturating_sub(shift32);
                }
                for entry in self.hash_table2.iter_mut() {
                    *entry = entry.saturating_sub(shift32);
                }
                for entry in self.chain_table.iter_mut() {
                    *entry = entry.saturating_sub(shift32);
                }
            }

            // Build the combined buffer
            let mut combined = Vec::with_capacity(dict_len + src.len());
            combined.extend_from_slice(&self.window);
            combined.extend_from_slice(src);

            // Call match finders directly WITHOUT clearing/repopulating tables.
            let result = self.compress_block_dict_persistent(&combined, dict_len, params, &rep);
            self.prev_combined_len = combined.len();
            result
        } else {
            // Window is non-empty but tables aren't populated yet (e.g. after
            // seed_from_dict). Use the standard dict path with table clearing.
            let rep = self.rep_offsets;
            let dict_len = self.window.len();
            let mut combined = Vec::with_capacity(dict_len + src.len());
            combined.extend_from_slice(&self.window);
            combined.extend_from_slice(src);
            let result = self.compress_block_dict_with_tables(&combined, dict_len, params, &rep);
            self.prev_combined_len = combined.len();
            self.tables_populated = true;
            result
        };

        self.update_rep_offsets(&block.sequences);
        self.update_window(src);

        block
    }

    /// Compress a block without dict prefix, using owned tables.
    fn compress_block_with_tables(
        &mut self,
        src: &[u8],
        params: &CompressionParams,
        _dict: &[u8],
        initial_rep: &[u32; 3],
    ) -> CompressedBlock {
        if src.is_empty() {
            return CompressedBlock {
                literals: Vec::new(),
                sequences: Vec::new(),
            };
        }
        match params.strategy {
            Strategy::Fast => {
                self.ensure_hash_table(params.hash_log);
                super::zstd_fast::compress_fast_ext(src, params, &mut self.hash_table, initial_rep)
            }
            Strategy::DFast => {
                self.ensure_hash_table(params.hash_log);
                let long_hash_log = params.hash_log.min(27);
                self.ensure_hash_table2(long_hash_log);
                super::zstd_fast::compress_dfast_ext(
                    src,
                    params,
                    &mut self.hash_table,
                    &mut self.hash_table2,
                    initial_rep,
                )
            }
            Strategy::Greedy => {
                self.ensure_hash_table(params.hash_log);
                self.ensure_chain_table(params.chain_log);
                super::zstd_lazy::compress_greedy_ext(
                    src,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::Lazy => {
                self.ensure_hash_table(params.hash_log);
                self.ensure_chain_table(params.chain_log);
                super::zstd_lazy::compress_lazy_ext(
                    src,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::Lazy2 => {
                self.ensure_hash_table(params.hash_log);
                self.ensure_chain_table(params.chain_log);
                super::zstd_lazy::compress_lazy2_ext(
                    src,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::BtLazy2 => super::zstd_lazy::compress_btlazy2(src, params),
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                super::zstd_opt::compress_btopt(src, params)
            }
        }
    }

    /// Compress a block with dict prefix, using owned tables.
    fn compress_block_dict_with_tables(
        &mut self,
        combined: &[u8],
        dict_len: usize,
        params: &CompressionParams,
        initial_rep: &[u32; 3],
    ) -> CompressedBlock {
        match params.strategy {
            Strategy::Fast => {
                self.ensure_hash_table(params.hash_log);
                prefill_hash_table_ext(
                    &mut self.hash_table,
                    params.hash_log,
                    combined,
                    dict_len,
                    params.min_match.max(4),
                );
                super::zstd_fast::compress_fast_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    initial_rep,
                )
            }
            Strategy::DFast => {
                self.ensure_hash_table(params.hash_log);
                let long_hash_log = params.hash_log.min(27);
                self.ensure_hash_table2(long_hash_log);
                prefill_hash_table_ext(
                    &mut self.hash_table,
                    params.hash_log,
                    combined,
                    dict_len,
                    params.min_match.max(4),
                );
                if dict_len >= 8 {
                    let end = dict_len.saturating_sub(7);
                    for pos in 0..end {
                        let lh = super::hash::hash8(&combined[pos..], long_hash_log);
                        self.hash_table2[lh] = pos as u32;
                    }
                }
                super::zstd_fast::compress_dfast_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.hash_table2,
                    initial_rep,
                )
            }
            Strategy::Greedy => {
                self.ensure_hash_table(params.hash_log);
                self.ensure_chain_table(params.chain_log);
                prefill_hash_chain_ext(
                    &mut self.hash_table,
                    &mut self.chain_table,
                    params.hash_log,
                    combined,
                    dict_len,
                    params,
                );
                super::zstd_lazy::compress_greedy_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::Lazy => {
                self.ensure_hash_table(params.hash_log);
                self.ensure_chain_table(params.chain_log);
                prefill_hash_chain_ext(
                    &mut self.hash_table,
                    &mut self.chain_table,
                    params.hash_log,
                    combined,
                    dict_len,
                    params,
                );
                super::zstd_lazy::compress_lazy_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::Lazy2 => {
                self.ensure_hash_table(params.hash_log);
                self.ensure_chain_table(params.chain_log);
                prefill_hash_chain_ext(
                    &mut self.hash_table,
                    &mut self.chain_table,
                    params.hash_log,
                    combined,
                    dict_len,
                    params,
                );
                super::zstd_lazy::compress_lazy2_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::BtLazy2 => {
                super::zstd_lazy::compress_btlazy2_dict(combined, dict_len, params, initial_rep)
            }
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                super::zstd_opt::compress_btopt_dict(combined, dict_len, params, initial_rep)
            }
        }
    }

    /// Compress a block with dict prefix, using persistent (shifted) tables.
    ///
    /// Unlike `compress_block_dict_with_tables`, this does NOT clear or
    /// prefill the hash/chain tables. The caller must have already shifted
    /// existing entries so they are valid for the current combined buffer.
    /// Tables are only resized if the log parameters changed.
    fn compress_block_dict_persistent(
        &mut self,
        combined: &[u8],
        dict_len: usize,
        params: &CompressionParams,
        initial_rep: &[u32; 3],
    ) -> CompressedBlock {
        match params.strategy {
            Strategy::Fast => {
                self.ensure_hash_table_no_clear(params.hash_log);
                super::zstd_fast::compress_fast_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    initial_rep,
                )
            }
            Strategy::DFast => {
                self.ensure_hash_table_no_clear(params.hash_log);
                let long_hash_log = params.hash_log.min(27);
                self.ensure_hash_table2_no_clear(long_hash_log);
                super::zstd_fast::compress_dfast_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.hash_table2,
                    initial_rep,
                )
            }
            Strategy::Greedy => {
                self.ensure_hash_table_no_clear(params.hash_log);
                self.ensure_chain_table_no_clear(params.chain_log);
                super::zstd_lazy::compress_greedy_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::Lazy => {
                self.ensure_hash_table_no_clear(params.hash_log);
                self.ensure_chain_table_no_clear(params.chain_log);
                super::zstd_lazy::compress_lazy_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::Lazy2 => {
                self.ensure_hash_table_no_clear(params.hash_log);
                self.ensure_chain_table_no_clear(params.chain_log);
                super::zstd_lazy::compress_lazy2_dict_ext(
                    combined,
                    dict_len,
                    params,
                    &mut self.hash_table,
                    &mut self.chain_table,
                    initial_rep,
                )
            }
            Strategy::BtLazy2 => {
                super::zstd_lazy::compress_btlazy2_dict(combined, dict_len, params, initial_rep)
            }
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                super::zstd_opt::compress_btopt_dict(combined, dict_len, params, initial_rep)
            }
        }
    }

    /// Ensure hash table is sized correctly without clearing existing entries.
    /// Only clears if the size changed (which means old entries are meaningless anyway).
    fn ensure_hash_table_no_clear(&mut self, hash_log: u32) {
        let size = 1usize << hash_log;
        if self.table_hash_log != hash_log || self.hash_table.len() != size {
            self.hash_table.resize(size, 0);
            self.hash_table.fill(0);
            self.table_hash_log = hash_log;
        }
        // Do NOT clear — entries are valid from the previous block (shifted by caller).
    }

    /// Ensure secondary hash table is sized correctly without clearing.
    fn ensure_hash_table2_no_clear(&mut self, hash_log: u32) {
        let size = 1usize << hash_log;
        if self.table_hash_log2 != hash_log || self.hash_table2.len() != size {
            self.hash_table2.resize(size, 0);
            self.hash_table2.fill(0);
            self.table_hash_log2 = hash_log;
        }
    }

    /// Ensure chain table is sized correctly without clearing.
    fn ensure_chain_table_no_clear(&mut self, chain_log: u32) {
        let size = 1usize << chain_log;
        if self.table_chain_log != chain_log || self.chain_table.len() != size {
            self.chain_table.resize(size, 0);
            self.chain_table.fill(0);
            self.table_chain_log = chain_log;
        }
    }

    /// Replay sequences to compute the final rep_offsets after a block.
    fn update_rep_offsets(&mut self, sequences: &[SequenceOut]) {
        let mut rep = RepCodes {
            rep: self.rep_offsets,
        };
        for seq in sequences {
            rep.update(seq.off_base, seq.lit_len);
        }
        self.rep_offsets = rep.rep;
    }

    /// Seed the match state from a dictionary.
    pub fn seed_from_dict(&mut self, dict_content: &[u8], rep_offsets: &[u32; 3]) {
        self.rep_offsets = *rep_offsets;
        self.window.clear();
        if dict_content.len() > self.window_size {
            self.window
                .extend_from_slice(&dict_content[dict_content.len() - self.window_size..]);
        } else {
            self.window.extend_from_slice(dict_content);
        }
    }

    /// Update window with block data stored as-is (RLE or raw).
    pub fn update_window_only(&mut self, src: &[u8]) {
        self.update_window(src);
    }

    /// Update the window buffer after compressing a block.
    fn update_window(&mut self, src: &[u8]) {
        if src.len() >= self.window_size {
            self.window.clear();
            self.window
                .extend_from_slice(&src[src.len() - self.window_size..]);
        } else {
            let total = self.window.len() + src.len();
            if total > self.window_size {
                let to_remove = total - self.window_size;
                self.window.drain(..to_remove);
            }
            self.window.extend_from_slice(src);
        }
    }
}

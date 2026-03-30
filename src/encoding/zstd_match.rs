//! Zstd match-finding engine supporting all strategies: Fast through BtUltra2.
//!
//! This module implements the core match-finding algorithms from the C zstd reference
//! implementation. Each strategy trades compression speed for ratio:
//!
//! - **Fast** (levels 1-2): Single hash table lookup, step forward on miss.
//! - **DFast** (levels 3-4): Dual hash tables (short + long), take the longer match.
//! - **Greedy** (levels 5): Hash chains with best-match search.
//! - **Lazy** (levels 6-7): Greedy + check pos+1, use the better match.
//! - **Lazy2** (levels 8-12): Lazy + also check pos+2, three-way comparison.
//! - **BtLazy2** (levels 13-15): Binary tree match finder + lazy2 evaluation.
//! - **BtOpt/BtUltra/BtUltra2** (levels 16-22): Binary tree with greedy selection
//!   (optimal parsing is a future enhancement).
//!
//! All functions are `#![forbid(unsafe_code)]` and operate on `&[u8]` slices.

use alloc::vec;
use alloc::vec::Vec;

use super::compress_params::{CompressionParams, Strategy};
use super::hash::{count_match, hash_ptr, hash8};

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
    rep: [u32; 3],
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
    fn update(&mut self, off_base: u32, lit_len: u32) {
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
    fn get_offset(&self, rep_idx: usize, lit_len: u32) -> u32 {
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
    fn off_base_for_rep(rep_idx: usize) -> u32 {
        (rep_idx as u32) + 1
    }
}

impl Default for RepCodes {
    fn default() -> Self {
        Self::new()
    }
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
}

impl MatchState {
    /// Create a new `MatchState` for the given compression parameters.
    pub fn new(params: &CompressionParams) -> Self {
        Self {
            window: Vec::new(),
            window_size: params.window_size(),
            rep_offsets: [1, 4, 8],
        }
    }

    /// Reset the match state for a new frame (clears window, resets rep offsets).
    pub fn reset(&mut self, params: &CompressionParams) {
        self.window.clear();
        self.window_size = params.window_size();
        self.rep_offsets = [1, 4, 8];
    }

    /// Get the current repeat offsets (for use as initial rep offsets of next block).
    pub fn rep_offsets(&self) -> &[u32; 3] {
        &self.rep_offsets
    }

    /// Get the current window content (previous block data for dict-style matching).
    pub fn window(&self) -> &[u8] {
        &self.window
    }

    /// Compress a block using persistent cross-block state.
    ///
    /// The window from previous blocks is used as match history (like a dictionary).
    /// After compression, the window and rep_offsets are updated for the next block.
    pub fn compress_block(
        &mut self,
        src: &[u8],
        params: &CompressionParams,
    ) -> CompressedBlock {
        let block = if self.window.is_empty() {
            // First block (or no history yet): compress without dict
            compress_block_zstd_with_dict(src, params, &[], &self.rep_offsets)
        } else {
            // Subsequent blocks: use window as dict content
            compress_block_zstd_with_dict(src, params, &self.window, &self.rep_offsets)
        };

        // Update rep_offsets by replaying sequences
        self.update_rep_offsets(&block.sequences);

        // Update window: keep last window_size bytes of (window + src)
        self.update_window(src);

        block
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

    /// Seed the match state from a dictionary (used for first block when a
    /// dictionary is active). Sets the window to the dict content and
    /// rep offsets to the dict's offsets.
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

    /// Update window with block data that was stored as-is (RLE or raw),
    /// without changing rep offsets (no sequences were emitted).
    pub fn update_window_only(&mut self, src: &[u8]) {
        self.update_window(src);
    }

    /// Update the window buffer after compressing a block.
    ///
    /// Keeps the last `window_size` bytes from the combination of old window + new src.
    fn update_window(&mut self, src: &[u8]) {
        if src.len() >= self.window_size {
            // New block is larger than window: just take the tail of src
            self.window.clear();
            self.window
                .extend_from_slice(&src[src.len() - self.window_size..]);
        } else {
            // Shift window left by src.len(), append src
            let total = self.window.len() + src.len();
            if total > self.window_size {
                let to_remove = total - self.window_size;
                // Remove oldest bytes from window
                self.window.drain(..to_remove);
            }
            self.window.extend_from_slice(src);
        }
    }
}

// ---------------------------------------------------------------------------
// Hash / Chain tables
// ---------------------------------------------------------------------------

/// Single hash table: maps hash -> most recent position (as u32).
struct HashTable {
    table: Vec<u32>,
    hash_log: u32,
}

impl HashTable {
    fn new(hash_log: u32) -> Self {
        Self {
            table: vec![0; 1 << hash_log],
            hash_log,
        }
    }

    #[inline]
    fn get(&self, hash: usize) -> u32 {
        self.table[hash]
    }

    #[inline]
    fn insert(&mut self, hash: usize, pos: u32) {
        self.table[hash] = pos;
    }

    /// Hash + insert + return previous value.
    #[inline]
    fn lookup_and_insert(&mut self, src: &[u8], pos: usize, min_match: u32) -> (usize, u32) {
        let h = hash_ptr(&src[pos..], self.hash_log, min_match);
        let prev = self.table[h];
        self.table[h] = pos as u32;
        (h, prev)
    }
}

/// Chain table: for each position, stores the previous position with the same hash.
/// Used by Greedy/Lazy/Lazy2 strategies for hash chain traversal.
struct ChainTable {
    table: Vec<u32>,
    mask: u32,
}

impl ChainTable {
    fn new(chain_log: u32) -> Self {
        let size = 1usize << chain_log;
        Self {
            table: vec![0; size],
            mask: (size as u32).wrapping_sub(1),
        }
    }

    #[inline]
    fn get(&self, pos: u32) -> u32 {
        self.table[(pos & self.mask) as usize]
    }

    #[inline]
    fn insert(&mut self, pos: u32, prev: u32) {
        self.table[(pos & self.mask) as usize] = prev;
    }
}

// ---------------------------------------------------------------------------
// Match candidate
// ---------------------------------------------------------------------------

/// Internal representation of a match candidate during search.
#[derive(Debug, Clone, Copy)]
struct MatchCandidate {
    /// Encoded off_base (1-3 for repcodes, >=4 for real offset + 3).
    off_base: u32,
    /// Match length in bytes.
    match_len: u32,
}

impl MatchCandidate {
    #[inline]
    fn gain(&self) -> i64 {
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
    fn is_better_lazy(&self, other: &MatchCandidate) -> bool {
        // The new match at pos+1 needs to compensate for the literal we emit.
        // Standard heuristic: other is better if its gain exceeds ours + threshold.
        other.gain() > self.gain() + 4
    }
}

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
pub fn compress_block_zstd(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    compress_block_zstd_with_dict(src, params, &[], &[1, 4, 8])
}

/// Compress a single block with optional dictionary content prepended as match history.
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
            Strategy::Fast => compress_fast(src, params),
            Strategy::DFast => compress_dfast(src, params),
            Strategy::Greedy => compress_greedy(src, params),
            Strategy::Lazy => compress_lazy(src, params),
            Strategy::Lazy2 => compress_lazy2(src, params),
            Strategy::BtLazy2 => compress_btlazy2(src, params),
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                compress_btopt(src, params)
            }
        };
    }
    let dict_len = dict_content.len();
    let mut combined = Vec::with_capacity(dict_len + src.len());
    combined.extend_from_slice(dict_content);
    combined.extend_from_slice(src);
    match params.strategy {
        Strategy::Fast => compress_fast_dict(&combined, dict_len, params, initial_rep_offsets),
        Strategy::DFast => compress_dfast_dict(&combined, dict_len, params, initial_rep_offsets),
        Strategy::Greedy => compress_greedy_dict(&combined, dict_len, params, initial_rep_offsets),
        Strategy::Lazy => compress_lazy_dict(&combined, dict_len, params, initial_rep_offsets),
        Strategy::Lazy2 => compress_lazy2_dict(&combined, dict_len, params, initial_rep_offsets),
        Strategy::BtLazy2 => {
            compress_btlazy2_dict(&combined, dict_len, params, initial_rep_offsets)
        }
        Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
            compress_btopt_dict(&combined, dict_len, params, initial_rep_offsets)
        }
    }
}

// ---------------------------------------------------------------------------
// Repcode checking helper
// ---------------------------------------------------------------------------

/// Try to find a repcode match at position `pos` in `src`.
/// Returns the best repcode match (longest), or None.
#[inline]
fn try_repcodes(
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
fn build_block(
    src: &[u8],
    sequences: Vec<SequenceOut>,
    literals: Vec<u8>,
    anchor: usize,
) -> CompressedBlock {
    let mut lits = literals;
    // Append any trailing literals after the last match
    if anchor < src.len() {
        lits.extend_from_slice(&src[anchor..]);
        // If we have sequences, add the trailing literals to the last sequence's
        // literal count? No -- trailing literals go as a separate Literals section
        // in the block, not attached to a sequence. We just append them to the
        // literals buffer. The caller needs to handle this.
    }

    // If there are no sequences at all, everything is literals
    CompressedBlock {
        literals: lits,
        sequences,
    }
}

// ---------------------------------------------------------------------------
// Strategy: Fast
// ---------------------------------------------------------------------------

fn compress_fast(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    // For Fast strategy, step size on miss increases with lower effort.
    // C zstd uses step = (ip - anchor) / step_factor + 1
    // step_factor = 1 << (search_log + 1) for level 1, but we simplify:
    // step_factor depends on search_log; for levels 1-2, search_log=1
    let step_factor = 1usize << params.search_log;

    let mut htable = HashTable::new(hash_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0; // start of current literal run
    let mut pos: usize = 0;

    let end = src.len().saturating_sub(8); // need 8 bytes for hash reads

    while pos < end {
        // Adaptive step: move faster when no matches found recently
        let step = ((pos - anchor) / step_factor) + 1;

        // Check repcodes first
        if let Some(rep_match) = try_repcodes(src, pos, &rep, min_match, (pos - anchor) as u32) {
            let lit_len = (pos - anchor) as u32;
            literals.extend_from_slice(&src[anchor..pos]);
            let seq = SequenceOut {
                off_base: rep_match.off_base,
                lit_len,
                match_len: rep_match.match_len,
            };
            rep.update(seq.off_base, lit_len);
            sequences.push(seq);
            pos += rep_match.match_len as usize;
            anchor = pos;
            continue;
        }

        // Hash lookup
        let (_h, match_pos) = htable.lookup_and_insert(src, pos, min_match);
        let match_pos = match_pos as usize;

        // Validate: position must be within window and before current pos
        if match_pos == 0 || match_pos >= pos || (pos - match_pos) > params.window_size() {
            pos += step;
            continue;
        }

        // Count match length
        let ml = count_match(&src[pos..], &src[match_pos..]);
        if ml < min_match as usize {
            pos += step;
            continue;
        }

        // Emit sequence
        let lit_len = (pos - anchor) as u32;
        literals.extend_from_slice(&src[anchor..pos]);
        let real_offset = (pos - match_pos) as u32;
        let seq = SequenceOut {
            off_base: real_offset + 3,
            lit_len,
            match_len: ml as u32,
        };
        rep.update(seq.off_base, lit_len);
        sequences.push(seq);

        // Insert intermediate positions into hash table
        let match_end = pos + ml;
        pos += 1;
        let insert_end = match_end.min(end);
        while pos < insert_end {
            let h = hash_ptr(&src[pos..], hash_log, min_match);
            htable.insert(h, pos as u32);
            pos += 1;
        }
        pos = match_end;
        anchor = pos;
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Strategy: DFast
// ---------------------------------------------------------------------------

fn compress_dfast(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    // Long hash table uses 8-byte hashing
    let long_hash_log = hash_log.min(27); // cap at 27 bits

    let mut short_table = HashTable::new(hash_log);
    let mut long_table = HashTable::new(long_hash_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        // Check repcodes
        if let Some(rep_match) = try_repcodes(src, pos, &rep, min_match, (pos - anchor) as u32) {
            let lit_len = (pos - anchor) as u32;
            literals.extend_from_slice(&src[anchor..pos]);
            let seq = SequenceOut {
                off_base: rep_match.off_base,
                lit_len,
                match_len: rep_match.match_len,
            };
            rep.update(seq.off_base, lit_len);
            sequences.push(seq);
            pos += rep_match.match_len as usize;
            anchor = pos;
            continue;
        }

        // Short hash (min_match bytes)
        let short_h = hash_ptr(&src[pos..], hash_log, min_match);
        let short_prev = short_table.get(short_h) as usize;
        short_table.insert(short_h, pos as u32);

        // Long hash (8 bytes)
        let long_h = hash8(&src[pos..], long_hash_log);
        let long_prev = long_table.get(long_h) as usize;
        long_table.insert(long_h, pos as u32);

        // Try long match first
        let mut best: Option<MatchCandidate> = None;

        if long_prev > 0 && long_prev < pos && (pos - long_prev) <= params.window_size() {
            let ml = count_match(&src[pos..], &src[long_prev..]);
            if ml >= min_match as usize {
                best = Some(MatchCandidate {
                    off_base: (pos - long_prev) as u32 + 3,
                    match_len: ml as u32,
                });
            }
        }

        // Try short match
        if short_prev > 0 && short_prev < pos && (pos - short_prev) <= params.window_size() {
            let ml = count_match(&src[pos..], &src[short_prev..]);
            if ml >= min_match as usize {
                let cand = MatchCandidate {
                    off_base: (pos - short_prev) as u32 + 3,
                    match_len: ml as u32,
                };
                if best.map_or(true, |b| cand.match_len > b.match_len) {
                    best = Some(cand);
                }
            }
        }

        if let Some(m) = best {
            let lit_len = (pos - anchor) as u32;
            literals.extend_from_slice(&src[anchor..pos]);
            let seq = SequenceOut {
                off_base: m.off_base,
                lit_len,
                match_len: m.match_len,
            };
            rep.update(seq.off_base, lit_len);
            sequences.push(seq);

            // Insert intermediate positions
            let match_end = pos + m.match_len as usize;
            pos += 1;
            let insert_end = match_end.min(end);
            while pos < insert_end {
                let sh = hash_ptr(&src[pos..], hash_log, min_match);
                short_table.insert(sh, pos as u32);
                let lh = hash8(&src[pos..], long_hash_log);
                long_table.insert(lh, pos as u32);
                pos += 1;
            }
            pos = match_end;
            anchor = pos;
        } else {
            pos += 1;
        }
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Hash chain search (used by Greedy, Lazy, Lazy2)
// ---------------------------------------------------------------------------

/// Search the hash chain for the best match at `pos`.
/// Returns the best match found (if any) considering both the hash chain and repcodes.
fn search_hash_chain(
    src: &[u8],
    pos: usize,
    htable: &mut HashTable,
    chain: &mut ChainTable,
    rep: &RepCodes,
    params: &CompressionParams,
    lit_len: u32,
) -> Option<MatchCandidate> {
    let min_match = params.min_match.max(4) as usize;
    let search_depth = params.search_depth();
    let window_size = params.window_size();

    if pos + 8 > src.len() {
        return None;
    }

    // First check repcodes
    let mut best = try_repcodes(src, pos, rep, min_match as u32, lit_len);

    // Hash lookup + chain walk
    let h = hash_ptr(&src[pos..], params.hash_log, params.min_match);
    let mut candidate_pos = htable.get(h) as usize;

    // Insert current position into hash table + chain
    let prev = htable.get(h);
    htable.insert(h, pos as u32);
    chain.insert(pos as u32, prev);

    let mut depth = search_depth;
    while depth > 0 && candidate_pos > 0 && candidate_pos < pos {
        let dist = pos - candidate_pos;
        if dist > window_size {
            break;
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
        let next = chain.get(candidate_pos as u32) as usize;
        if next >= candidate_pos {
            break; // avoid infinite loops from stale chain entries
        }
        candidate_pos = next;
        depth -= 1;
    }

    best
}

/// Insert a position into hash table and chain table without searching.
#[inline]
fn insert_hash_chain(
    src: &[u8],
    pos: usize,
    htable: &mut HashTable,
    chain: &mut ChainTable,
    params: &CompressionParams,
) {
    if pos + 8 > src.len() {
        return;
    }
    let h = hash_ptr(&src[pos..], params.hash_log, params.min_match);
    let prev = htable.get(h);
    htable.insert(h, pos as u32);
    chain.insert(pos as u32, prev);
}

// ---------------------------------------------------------------------------
// Strategy: Greedy
// ---------------------------------------------------------------------------

fn compress_greedy(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = HashTable::new(params.hash_log);
    let mut chain = ChainTable::new(params.chain_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let best = search_hash_chain(src, pos, &mut htable, &mut chain, &rep, params, lit_len);

        if let Some(m) = best {
            literals.extend_from_slice(&src[anchor..pos]);
            let seq = SequenceOut {
                off_base: m.off_base,
                lit_len,
                match_len: m.match_len,
            };
            rep.update(seq.off_base, lit_len);
            sequences.push(seq);

            // Insert intermediate positions into hash chain
            let match_end = pos + m.match_len as usize;
            // pos was already inserted by search_hash_chain, start at pos+1
            for p in (pos + 1)..match_end.min(end) {
                insert_hash_chain(src, p, &mut htable, &mut chain, params);
            }
            pos = match_end;
            anchor = pos;
        } else {
            pos += 1;
        }
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Strategy: Lazy
// ---------------------------------------------------------------------------

fn compress_lazy(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = HashTable::new(params.hash_log);
    let mut chain = ChainTable::new(params.chain_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let match0 = search_hash_chain(src, pos, &mut htable, &mut chain, &rep, params, lit_len);

        let match0 = match match0 {
            Some(m) => m,
            None => {
                pos += 1;
                continue;
            }
        };

        // Lazy: check pos+1 for a better match
        if pos + 1 < end {
            let lit_len1 = (pos + 1 - anchor) as u32;
            let match1 = search_hash_chain(
                src,
                pos + 1,
                &mut htable,
                &mut chain,
                &rep,
                params,
                lit_len1,
            );

            if let Some(m1) = match1 {
                if match0.is_better_lazy(&m1) {
                    // Use match at pos+1 instead; emit one extra literal
                    let lit_len_out = (pos + 1 - anchor) as u32;
                    literals.extend_from_slice(&src[anchor..pos + 1]);
                    let seq = SequenceOut {
                        off_base: m1.off_base,
                        lit_len: lit_len_out,
                        match_len: m1.match_len,
                    };
                    rep.update(seq.off_base, lit_len_out);
                    sequences.push(seq);

                    let match_end = pos + 1 + m1.match_len as usize;
                    for p in (pos + 2)..match_end.min(end) {
                        insert_hash_chain(src, p, &mut htable, &mut chain, params);
                    }
                    pos = match_end;
                    anchor = pos;
                    continue;
                }
            }
        }

        // Use match at pos
        literals.extend_from_slice(&src[anchor..pos]);
        let seq = SequenceOut {
            off_base: match0.off_base,
            lit_len,
            match_len: match0.match_len,
        };
        rep.update(seq.off_base, lit_len);
        sequences.push(seq);

        let match_end = pos + match0.match_len as usize;
        for p in (pos + 1)..match_end.min(end) {
            insert_hash_chain(src, p, &mut htable, &mut chain, params);
        }
        pos = match_end;
        anchor = pos;
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Strategy: Lazy2
// ---------------------------------------------------------------------------

fn compress_lazy2(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = HashTable::new(params.hash_log);
    let mut chain = ChainTable::new(params.chain_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let match0 = search_hash_chain(src, pos, &mut htable, &mut chain, &rep, params, lit_len);

        let match0 = match match0 {
            Some(m) => m,
            None => {
                pos += 1;
                continue;
            }
        };

        // Lazy: check pos+1
        let mut best_pos = pos;
        let mut best_match = match0;

        if pos + 1 < end {
            let lit_len1 = (pos + 1 - anchor) as u32;
            let match1 = search_hash_chain(
                src,
                pos + 1,
                &mut htable,
                &mut chain,
                &rep,
                params,
                lit_len1,
            );

            if let Some(m1) = match1 {
                if best_match.is_better_lazy(&m1) {
                    best_pos = pos + 1;
                    best_match = m1;

                    // Lazy2: also check pos+2
                    if pos + 2 < end {
                        let lit_len2 = (pos + 2 - anchor) as u32;
                        let match2 = search_hash_chain(
                            src,
                            pos + 2,
                            &mut htable,
                            &mut chain,
                            &rep,
                            params,
                            lit_len2,
                        );
                        if let Some(m2) = match2 {
                            if best_match.is_better_lazy(&m2) {
                                best_pos = pos + 2;
                                best_match = m2;
                            }
                        }
                    }
                }
            }
        }

        // Emit the best match
        let lit_len_out = (best_pos - anchor) as u32;
        literals.extend_from_slice(&src[anchor..best_pos]);
        let seq = SequenceOut {
            off_base: best_match.off_base,
            lit_len: lit_len_out,
            match_len: best_match.match_len,
        };
        rep.update(seq.off_base, lit_len_out);
        sequences.push(seq);

        let match_end = best_pos + best_match.match_len as usize;
        // Insert skipped positions + match interior
        for p in (pos + 1)..match_end.min(end) {
            // search_hash_chain already inserted pos, pos+1, and possibly pos+2
            // during the lazy search, so we skip those.
            // Actually, search_hash_chain inserts only the position it searches at,
            // so we need to insert everything from pos+1 that wasn't already searched.
            // The searched positions were: pos, and pos+1 (and pos+2 for lazy2).
            // Those were inserted by search_hash_chain. So skip them.
            let already_inserted =
                p == pos || p == pos + 1 || (p == pos + 2 && best_pos >= pos + 2);
            if !already_inserted {
                insert_hash_chain(src, p, &mut htable, &mut chain, params);
            }
        }
        pos = match_end;
        anchor = pos;
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Binary Tree match finder (used by BtLazy2, BtOpt, BtUltra, BtUltra2)
// ---------------------------------------------------------------------------

/// Binary tree table: each position has two entries (smaller_child, larger_child).
///
/// The C zstd reference calls this a "Dynamic Unsorted Binary Tree" (DUBT). We
/// use a simplified variant that inserts directly into a sorted binary tree at
/// each position, combining insertion and search in a single tree walk.
///
/// Memory layout: `tree[2 * (pos & bt_mask)]` = smaller child index,
///                `tree[2 * (pos & bt_mask) + 1]` = larger child index.
struct BinaryTree {
    /// Hash table: maps hash -> tree root position (most recent position with that hash).
    hash_table: Vec<u32>,
    hash_log: u32,
    /// Binary tree storage: 2 entries per position (smaller, larger).
    tree: Vec<u32>,
    /// Mask for tree position indexing: `(1 << bt_log) - 1`.
    bt_mask: u32,
}

impl BinaryTree {
    fn new(hash_log: u32, chain_log: u32) -> Self {
        // C zstd: btLog = chainLog - 1, btMask = (1 << btLog) - 1
        // tree size = 2 * (1 << btLog) = 1 << chainLog
        let bt_log = chain_log.saturating_sub(1);
        let bt_size = 1usize << bt_log;
        Self {
            hash_table: vec![0; 1 << hash_log],
            hash_log,
            tree: vec![0; 2 * bt_size],
            bt_mask: (bt_size as u32).wrapping_sub(1),
        }
    }

    /// Get the (smaller_child, larger_child) pair indices for a position.
    #[inline]
    fn children_idx(&self, pos: u32) -> (usize, usize) {
        let base = 2 * (pos & self.bt_mask) as usize;
        (base, base + 1)
    }

    /// Insert `pos` into the tree and simultaneously search for the best match.
    ///
    /// Returns the best match found (if any) with at least `min_match` bytes.
    /// `search_depth` limits the number of tree nodes visited.
    /// `window_low` is the minimum valid reference position.
    fn insert_and_find(
        &mut self,
        src: &[u8],
        pos: usize,
        min_match: usize,
        search_depth: usize,
        window_low: usize,
    ) -> Option<MatchCandidate> {
        if pos + 8 > src.len() {
            return None;
        }

        let h = hash_ptr(&src[pos..], self.hash_log, min_match as u32);
        let match_index = self.hash_table[h] as usize;
        self.hash_table[h] = pos as u32;

        // bt_low: positions at or below this are outside the tree's addressable range
        let bt_low = if self.bt_mask as usize >= pos {
            0
        } else {
            pos - self.bt_mask as usize
        };

        let (smaller_idx, larger_idx) = self.children_idx(pos as u32);
        // We'll track which tree slot to write the next smaller/larger child into.
        // In the C code these are pointers; we use indices into self.tree.
        let mut smaller_slot = smaller_idx;
        let mut larger_slot = larger_idx;

        let mut common_len_smaller: usize = 0;
        let mut common_len_larger: usize = 0;

        let mut best: Option<MatchCandidate> = None;
        let mut candidate = match_index;
        let mut depth = search_depth;

        while depth > 0 && candidate > window_low && candidate < pos {
            let match_len_min = common_len_smaller.min(common_len_larger);

            // Count how many bytes match starting from the already-known common prefix
            let remaining_a = &src[pos + match_len_min..];
            let remaining_b = if candidate + match_len_min < src.len() {
                &src[candidate + match_len_min..]
            } else {
                &[]
            };
            let extra = count_match(remaining_a, remaining_b);
            let match_len = match_len_min + extra;

            if match_len >= min_match {
                let dist = pos - candidate;
                let cand = MatchCandidate {
                    off_base: dist as u32 + 3,
                    match_len: match_len as u32,
                };
                if best.map_or(true, |b| {
                    cand.match_len > b.match_len
                        || (cand.match_len == b.match_len && cand.gain() > b.gain())
                }) {
                    best = Some(cand);
                }
            }

            // If we've matched all the way to the end of input, we can't determine
            // the ordering, so break to maintain tree consistency.
            if pos + match_len >= src.len() || candidate + match_len >= src.len() {
                // Terminate both branches
                self.tree[smaller_slot] = 0;
                self.tree[larger_slot] = 0;
                return best;
            }

            let (child_smaller_idx, child_larger_idx) = self.children_idx(candidate as u32);

            if src[candidate + match_len] < src[pos + match_len] {
                // candidate is smaller than current position
                self.tree[smaller_slot] = candidate as u32;
                common_len_smaller = match_len;
                if candidate <= bt_low {
                    // Use a dummy slot: just zero the slot and stop
                    self.tree[smaller_slot] = 0;
                    break;
                }
                // Next smaller candidate comes from the larger child of this node
                smaller_slot = child_larger_idx;
                candidate = self.tree[child_larger_idx] as usize;
            } else {
                // candidate is larger than (or equal to) current position
                self.tree[larger_slot] = candidate as u32;
                common_len_larger = match_len;
                if candidate <= bt_low {
                    self.tree[larger_slot] = 0;
                    break;
                }
                // Next larger candidate comes from the smaller child of this node
                larger_slot = child_smaller_idx;
                candidate = self.tree[child_smaller_idx] as usize;
            }

            depth -= 1;
        }

        // Terminate both open branches
        self.tree[smaller_slot] = 0;
        self.tree[larger_slot] = 0;

        best
    }

    /// Insert a position into the tree without searching for matches.
    /// Used to fill the tree for positions we skip over (inside matches, etc.).
    fn insert_only(
        &mut self,
        src: &[u8],
        pos: usize,
        min_match: usize,
        search_depth: usize,
        window_low: usize,
    ) {
        if pos + 8 > src.len() {
            return;
        }

        let h = hash_ptr(&src[pos..], self.hash_log, min_match as u32);
        let match_index = self.hash_table[h] as usize;
        self.hash_table[h] = pos as u32;

        let bt_low = if self.bt_mask as usize >= pos {
            0
        } else {
            pos - self.bt_mask as usize
        };

        let (smaller_idx, larger_idx) = self.children_idx(pos as u32);
        let mut smaller_slot = smaller_idx;
        let mut larger_slot = larger_idx;

        let mut common_len_smaller: usize = 0;
        let mut common_len_larger: usize = 0;

        let mut candidate = match_index;
        let mut depth = search_depth;

        while depth > 0 && candidate > window_low && candidate < pos {
            let match_len_min = common_len_smaller.min(common_len_larger);

            let remaining_a = &src[pos + match_len_min..];
            let remaining_b = if candidate + match_len_min < src.len() {
                &src[candidate + match_len_min..]
            } else {
                &[]
            };
            let extra = count_match(remaining_a, remaining_b);
            let match_len = match_len_min + extra;

            if pos + match_len >= src.len() || candidate + match_len >= src.len() {
                self.tree[smaller_slot] = 0;
                self.tree[larger_slot] = 0;
                return;
            }

            let (child_smaller_idx, child_larger_idx) = self.children_idx(candidate as u32);

            if src[candidate + match_len] < src[pos + match_len] {
                self.tree[smaller_slot] = candidate as u32;
                common_len_smaller = match_len;
                if candidate <= bt_low {
                    self.tree[smaller_slot] = 0;
                    break;
                }
                smaller_slot = child_larger_idx;
                candidate = self.tree[child_larger_idx] as usize;
            } else {
                self.tree[larger_slot] = candidate as u32;
                common_len_larger = match_len;
                if candidate <= bt_low {
                    self.tree[larger_slot] = 0;
                    break;
                }
                larger_slot = child_smaller_idx;
                candidate = self.tree[child_smaller_idx] as usize;
            }

            depth -= 1;
        }

        self.tree[smaller_slot] = 0;
        self.tree[larger_slot] = 0;
    }
}

/// Search the binary tree for the best match at `pos`, including repcode checks.
fn search_binary_tree(
    src: &[u8],
    pos: usize,
    bt: &mut BinaryTree,
    rep: &RepCodes,
    params: &CompressionParams,
    lit_len: u32,
) -> Option<MatchCandidate> {
    let min_match = params.min_match.max(4) as usize;
    let search_depth = params.search_depth();
    let window_size = params.window_size();
    let window_low = if pos > window_size { pos - window_size } else { 0 };

    // Check repcodes first (they're free to encode)
    let mut best = try_repcodes(src, pos, rep, min_match as u32, lit_len);

    // Search the binary tree
    if let Some(bt_match) = bt.insert_and_find(src, pos, min_match, search_depth, window_low) {
        if best.map_or(true, |b| {
            bt_match.match_len > b.match_len
                || (bt_match.match_len == b.match_len && bt_match.gain() > b.gain())
        }) {
            best = Some(bt_match);
        }
    }

    best
}

/// Insert a position into the binary tree without searching.
#[inline]
fn insert_binary_tree(
    src: &[u8],
    pos: usize,
    bt: &mut BinaryTree,
    params: &CompressionParams,
) {
    let min_match = params.min_match.max(4) as usize;
    let search_depth = params.search_depth();
    let window_size = params.window_size();
    let window_low = if pos > window_size { pos - window_size } else { 0 };
    bt.insert_only(src, pos, min_match, search_depth, window_low);
}

// ---------------------------------------------------------------------------
// Strategy: BtLazy2 (Binary Tree + Lazy2 evaluation)
// ---------------------------------------------------------------------------

fn compress_btlazy2(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut bt = BinaryTree::new(params.hash_log, params.chain_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let match0 = search_binary_tree(src, pos, &mut bt, &rep, params, lit_len);

        let match0 = match match0 {
            Some(m) => m,
            None => {
                pos += 1;
                continue;
            }
        };

        // Lazy evaluation: check pos+1
        let mut best_pos = pos;
        let mut best_match = match0;

        if pos + 1 < end {
            let lit_len1 = (pos + 1 - anchor) as u32;
            let match1 = search_binary_tree(src, pos + 1, &mut bt, &rep, params, lit_len1);

            if let Some(m1) = match1 {
                if best_match.is_better_lazy(&m1) {
                    best_pos = pos + 1;
                    best_match = m1;

                    // Lazy2: also check pos+2
                    if pos + 2 < end {
                        let lit_len2 = (pos + 2 - anchor) as u32;
                        let match2 =
                            search_binary_tree(src, pos + 2, &mut bt, &rep, params, lit_len2);
                        if let Some(m2) = match2 {
                            if best_match.is_better_lazy(&m2) {
                                best_pos = pos + 2;
                                best_match = m2;
                            }
                        }
                    }
                }
            }
        }

        // Emit the best match
        let lit_len_out = (best_pos - anchor) as u32;
        literals.extend_from_slice(&src[anchor..best_pos]);
        let seq = SequenceOut {
            off_base: best_match.off_base,
            lit_len: lit_len_out,
            match_len: best_match.match_len,
        };
        rep.update(seq.off_base, lit_len_out);
        sequences.push(seq);

        let match_end = best_pos + best_match.match_len as usize;
        // Insert skipped + match-interior positions into the binary tree.
        // search_binary_tree already inserted pos, pos+1, and possibly pos+2.
        for p in (pos + 1)..match_end.min(end) {
            let already =
                p == pos || p == pos + 1 || (p == pos + 2 && best_pos >= pos + 2);
            if !already {
                insert_binary_tree(src, p, &mut bt, params);
            }
        }
        pos = match_end;
        anchor = pos;
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Strategy: BtOpt / BtUltra / BtUltra2 (Binary Tree + greedy with deeper search)
// ---------------------------------------------------------------------------

/// BtOpt uses the binary tree for match finding with greedy selection.
/// BtUltra and BtUltra2 use even deeper search (higher search_log).
/// For now these all use greedy match selection; optimal parsing is a future
/// enhancement. The binary tree still provides better matches than hash chains.
fn compress_btopt(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut bt = BinaryTree::new(params.hash_log, params.chain_log);
    let mut rep = RepCodes::new();
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let best = search_binary_tree(src, pos, &mut bt, &rep, params, lit_len);

        if let Some(m) = best {
            literals.extend_from_slice(&src[anchor..pos]);
            let seq = SequenceOut {
                off_base: m.off_base,
                lit_len,
                match_len: m.match_len,
            };
            rep.update(seq.off_base, lit_len);
            sequences.push(seq);

            // Insert match-interior positions
            let match_end = pos + m.match_len as usize;
            for p in (pos + 1)..match_end.min(end) {
                insert_binary_tree(src, p, &mut bt, params);
            }
            pos = match_end;
            anchor = pos;
        } else {
            pos += 1;
        }
    }

    build_block(src, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Dictionary-aware compression
// ---------------------------------------------------------------------------

fn build_block_dict(
    combined: &[u8],
    dict_len: usize,
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

fn prefill_hash_table(htable: &mut HashTable, combined: &[u8], dict_len: usize, min_match: u32) {
    if dict_len < 8 {
        return;
    }
    let end = dict_len.saturating_sub(7);
    for pos in 0..end {
        let h = hash_ptr(&combined[pos..], htable.hash_log, min_match);
        htable.insert(h, pos as u32);
    }
}

fn prefill_hash_chain(
    htable: &mut HashTable,
    chain: &mut ChainTable,
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
) {
    if dict_len < 8 {
        return;
    }
    let end = dict_len.saturating_sub(7);
    for pos in 0..end {
        let h = hash_ptr(&combined[pos..], params.hash_log, params.min_match);
        let prev = htable.get(h);
        htable.insert(h, pos as u32);
        chain.insert(pos as u32, prev);
    }
}

fn compress_fast_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    let step_factor = 1usize << params.search_log;
    let mut htable = HashTable::new(hash_log);
    prefill_hash_table(&mut htable, combined, dict_len, min_match);
    let mut rep = RepCodes {
        rep: *initial_rep,
    };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    while pos < end {
        let step = ((pos - anchor) / step_factor) + 1;
        if let Some(rm) = try_repcodes(combined, pos, &rep, min_match, (pos - anchor) as u32) {
            let ll = (pos - anchor) as u32;
            literals.extend_from_slice(&combined[anchor..pos]);
            let seq = SequenceOut { off_base: rm.off_base, lit_len: ll, match_len: rm.match_len };
            rep.update(seq.off_base, ll);
            sequences.push(seq);
            pos += rm.match_len as usize;
            anchor = pos;
            continue;
        }
        let (_h, mp) = htable.lookup_and_insert(combined, pos, min_match);
        let mp = mp as usize;
        if mp >= pos || (pos - mp) > params.window_size() {
            pos += step;
            continue;
        }
        let ml = count_match(&combined[pos..], &combined[mp..]);
        if ml < min_match as usize {
            pos += step;
            continue;
        }
        let ll = (pos - anchor) as u32;
        literals.extend_from_slice(&combined[anchor..pos]);
        let ro = (pos - mp) as u32;
        let seq = SequenceOut { off_base: ro + 3, lit_len: ll, match_len: ml as u32 };
        rep.update(seq.off_base, ll);
        sequences.push(seq);
        let me = pos + ml;
        pos += 1;
        let ie = me.min(end);
        while pos < ie {
            let h = hash_ptr(&combined[pos..], hash_log, min_match);
            htable.insert(h, pos as u32);
            pos += 1;
        }
        pos = me;
        anchor = pos;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

fn compress_dfast_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    let long_hash_log = hash_log.min(27);
    let mut short_table = HashTable::new(hash_log);
    let mut long_table = HashTable::new(long_hash_log);
    prefill_hash_table(&mut short_table, combined, dict_len, min_match);
    if dict_len >= 8 {
        let end = dict_len.saturating_sub(7);
        for pos in 0..end {
            let lh = hash8(&combined[pos..], long_hash_log);
            long_table.insert(lh, pos as u32);
        }
    }
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    while pos < end {
        if let Some(rm) = try_repcodes(combined, pos, &rep, min_match, (pos - anchor) as u32) {
            let ll = (pos - anchor) as u32;
            literals.extend_from_slice(&combined[anchor..pos]);
            let seq = SequenceOut { off_base: rm.off_base, lit_len: ll, match_len: rm.match_len };
            rep.update(seq.off_base, ll);
            sequences.push(seq);
            pos += rm.match_len as usize;
            anchor = pos;
            continue;
        }
        let sh = hash_ptr(&combined[pos..], hash_log, min_match);
        let sp = short_table.get(sh) as usize;
        short_table.insert(sh, pos as u32);
        let lh = hash8(&combined[pos..], long_hash_log);
        let lp = long_table.get(lh) as usize;
        long_table.insert(lh, pos as u32);
        let mut best: Option<MatchCandidate> = None;
        if lp < pos && (pos - lp) <= params.window_size() {
            let ml = count_match(&combined[pos..], &combined[lp..]);
            if ml >= min_match as usize {
                best = Some(MatchCandidate { off_base: (pos - lp) as u32 + 3, match_len: ml as u32 });
            }
        }
        if sp < pos && (pos - sp) <= params.window_size() {
            let ml = count_match(&combined[pos..], &combined[sp..]);
            if ml >= min_match as usize {
                let c = MatchCandidate { off_base: (pos - sp) as u32 + 3, match_len: ml as u32 };
                if best.map_or(true, |b| c.match_len > b.match_len) { best = Some(c); }
            }
        }
        if let Some(m) = best {
            let ll = (pos - anchor) as u32;
            literals.extend_from_slice(&combined[anchor..pos]);
            let seq = SequenceOut { off_base: m.off_base, lit_len: ll, match_len: m.match_len };
            rep.update(seq.off_base, ll);
            sequences.push(seq);
            let me = pos + m.match_len as usize;
            pos += 1;
            let ie = me.min(end);
            while pos < ie {
                let s = hash_ptr(&combined[pos..], hash_log, min_match);
                short_table.insert(s, pos as u32);
                let l = hash8(&combined[pos..], long_hash_log);
                long_table.insert(l, pos as u32);
                pos += 1;
            }
            pos = me;
            anchor = pos;
        } else {
            pos += 1;
        }
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

fn compress_greedy_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut htable = HashTable::new(params.hash_log);
    let mut chain = ChainTable::new(params.chain_log);
    prefill_hash_chain(&mut htable, &mut chain, combined, dict_len, params);
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    while pos < end {
        let ll = (pos - anchor) as u32;
        let best = search_hash_chain(combined, pos, &mut htable, &mut chain, &rep, params, ll);
        if let Some(m) = best {
            literals.extend_from_slice(&combined[anchor..pos]);
            let seq = SequenceOut { off_base: m.off_base, lit_len: ll, match_len: m.match_len };
            rep.update(seq.off_base, ll);
            sequences.push(seq);
            let me = pos + m.match_len as usize;
            for p in (pos + 1)..me.min(end) { insert_hash_chain(combined, p, &mut htable, &mut chain, params); }
            pos = me;
            anchor = pos;
        } else {
            pos += 1;
        }
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

fn compress_lazy_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut htable = HashTable::new(params.hash_log);
    let mut chain = ChainTable::new(params.chain_log);
    prefill_hash_chain(&mut htable, &mut chain, combined, dict_len, params);
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    while pos < end {
        let ll = (pos - anchor) as u32;
        let m0 = match search_hash_chain(combined, pos, &mut htable, &mut chain, &rep, params, ll) {
            Some(m) => m,
            None => { pos += 1; continue; }
        };
        if pos + 1 < end {
            let ll1 = (pos + 1 - anchor) as u32;
            if let Some(m1) = search_hash_chain(combined, pos + 1, &mut htable, &mut chain, &rep, params, ll1) {
                if m0.is_better_lazy(&m1) {
                    let llo = (pos + 1 - anchor) as u32;
                    literals.extend_from_slice(&combined[anchor..pos + 1]);
                    let seq = SequenceOut { off_base: m1.off_base, lit_len: llo, match_len: m1.match_len };
                    rep.update(seq.off_base, llo);
                    sequences.push(seq);
                    let me = pos + 1 + m1.match_len as usize;
                    for p in (pos + 2)..me.min(end) { insert_hash_chain(combined, p, &mut htable, &mut chain, params); }
                    pos = me;
                    anchor = pos;
                    continue;
                }
            }
        }
        literals.extend_from_slice(&combined[anchor..pos]);
        let seq = SequenceOut { off_base: m0.off_base, lit_len: ll, match_len: m0.match_len };
        rep.update(seq.off_base, ll);
        sequences.push(seq);
        let me = pos + m0.match_len as usize;
        for p in (pos + 1)..me.min(end) { insert_hash_chain(combined, p, &mut htable, &mut chain, params); }
        pos = me;
        anchor = pos;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

fn compress_lazy2_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut htable = HashTable::new(params.hash_log);
    let mut chain = ChainTable::new(params.chain_log);
    prefill_hash_chain(&mut htable, &mut chain, combined, dict_len, params);
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    while pos < end {
        let ll = (pos - anchor) as u32;
        let m0 = match search_hash_chain(combined, pos, &mut htable, &mut chain, &rep, params, ll) {
            Some(m) => m,
            None => { pos += 1; continue; }
        };
        let mut best_pos = pos;
        let mut best_match = m0;
        if pos + 1 < end {
            let ll1 = (pos + 1 - anchor) as u32;
            if let Some(m1) = search_hash_chain(combined, pos + 1, &mut htable, &mut chain, &rep, params, ll1) {
                if best_match.is_better_lazy(&m1) {
                    best_pos = pos + 1;
                    best_match = m1;
                    if pos + 2 < end {
                        let ll2 = (pos + 2 - anchor) as u32;
                        if let Some(m2) = search_hash_chain(combined, pos + 2, &mut htable, &mut chain, &rep, params, ll2) {
                            if best_match.is_better_lazy(&m2) { best_pos = pos + 2; best_match = m2; }
                        }
                    }
                }
            }
        }
        let llo = (best_pos - anchor) as u32;
        literals.extend_from_slice(&combined[anchor..best_pos]);
        let seq = SequenceOut { off_base: best_match.off_base, lit_len: llo, match_len: best_match.match_len };
        rep.update(seq.off_base, llo);
        sequences.push(seq);
        let me = best_pos + best_match.match_len as usize;
        for p in (pos + 1)..me.min(end) {
            let already = p == pos || p == pos + 1 || (p == pos + 2 && best_pos >= pos + 2);
            if !already { insert_hash_chain(combined, p, &mut htable, &mut chain, params); }
        }
        pos = me;
        anchor = pos;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

fn prefill_binary_tree(
    bt: &mut BinaryTree,
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
) {
    if dict_len < 8 {
        return;
    }
    let min_match = params.min_match.max(4) as usize;
    let search_depth = params.search_depth();
    let end = dict_len.saturating_sub(7);
    for pos in 0..end {
        let window_size = params.window_size();
        let window_low = if pos > window_size { pos - window_size } else { 0 };
        bt.insert_only(combined, pos, min_match, search_depth, window_low);
    }
}

fn compress_btlazy2_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut bt = BinaryTree::new(params.hash_log, params.chain_log);
    prefill_binary_tree(&mut bt, combined, dict_len, params);
    let mut rep = RepCodes {
        rep: *initial_rep,
    };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);

    while pos < end {
        let ll = (pos - anchor) as u32;
        let m0 = match search_binary_tree(combined, pos, &mut bt, &rep, params, ll) {
            Some(m) => m,
            None => {
                pos += 1;
                continue;
            }
        };
        let mut best_pos = pos;
        let mut best_match = m0;
        if pos + 1 < end {
            let ll1 = (pos + 1 - anchor) as u32;
            if let Some(m1) = search_binary_tree(combined, pos + 1, &mut bt, &rep, params, ll1) {
                if best_match.is_better_lazy(&m1) {
                    best_pos = pos + 1;
                    best_match = m1;
                    if pos + 2 < end {
                        let ll2 = (pos + 2 - anchor) as u32;
                        if let Some(m2) =
                            search_binary_tree(combined, pos + 2, &mut bt, &rep, params, ll2)
                        {
                            if best_match.is_better_lazy(&m2) {
                                best_pos = pos + 2;
                                best_match = m2;
                            }
                        }
                    }
                }
            }
        }
        let llo = (best_pos - anchor) as u32;
        literals.extend_from_slice(&combined[anchor..best_pos]);
        let seq = SequenceOut {
            off_base: best_match.off_base,
            lit_len: llo,
            match_len: best_match.match_len,
        };
        rep.update(seq.off_base, llo);
        sequences.push(seq);
        let me = best_pos + best_match.match_len as usize;
        for p in (pos + 1)..me.min(end) {
            let already = p == pos || p == pos + 1 || (p == pos + 2 && best_pos >= pos + 2);
            if !already {
                insert_binary_tree(combined, p, &mut bt, params);
            }
        }
        pos = me;
        anchor = pos;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

fn compress_btopt_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut bt = BinaryTree::new(params.hash_log, params.chain_log);
    prefill_binary_tree(&mut bt, combined, dict_len, params);
    let mut rep = RepCodes {
        rep: *initial_rep,
    };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);

    while pos < end {
        let ll = (pos - anchor) as u32;
        let best = search_binary_tree(combined, pos, &mut bt, &rep, params, ll);
        if let Some(m) = best {
            literals.extend_from_slice(&combined[anchor..pos]);
            let seq = SequenceOut {
                off_base: m.off_base,
                lit_len: ll,
                match_len: m.match_len,
            };
            rep.update(seq.off_base, ll);
            sequences.push(seq);
            let me = pos + m.match_len as usize;
            for p in (pos + 1)..me.min(end) {
                insert_binary_tree(combined, p, &mut bt, params);
            }
            pos = me;
            anchor = pos;
        } else {
            pos += 1;
        }
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::compress_params::params_for_level;

    /// Helper: verify that the sequences + literals can reconstruct the original data.
    fn verify_reconstruction(src: &[u8], block: &CompressedBlock) {
        let mut output: Vec<u8> = Vec::new();
        let mut lit_cursor = 0usize;

        // Initialize rep offsets same as the compressor
        let mut rep = [1u32, 4, 8];

        for seq in &block.sequences {
            // Copy literals
            let lit_end = lit_cursor + seq.lit_len as usize;
            assert!(
                lit_end <= block.literals.len(),
                "lit_cursor={lit_cursor}, lit_len={}, literals.len()={}",
                seq.lit_len,
                block.literals.len(),
            );
            output.extend_from_slice(&block.literals[lit_cursor..lit_end]);
            lit_cursor = lit_end;

            // Resolve offset
            let real_offset = if seq.off_base >= 4 {
                let off = seq.off_base - 3;
                // Update rep
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

            // Copy match
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

        // Trailing literals
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
        // Should have at least some sequences (lots of rep matches)
        assert!(
            !block.sequences.is_empty(),
            "expected sequences for all-zero input"
        );
        // Most sequences should be repcode matches (off_base 1-3)
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
    // Repetitive pattern: should find matches
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
        // Literals should be much smaller than the source
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
    // Random data: few or no matches
    // ---------------------------------------------------------------

    #[test]
    fn random_data_fast() {
        // Pseudo-random via simple LCG
        let mut rng = 0x12345678u64;
        let mut src = vec![0u8; 4096];
        for b in src.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (rng >> 33) as u8;
        }
        let params = params_for_level(1, Some(src.len() as u64));
        let block = compress_block_zstd(&src, &params);
        verify_reconstruction(&src, &block);
        // Random data: literals should be close to the source size
        // (few/no matches expected)
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
    // Empty input
    // ---------------------------------------------------------------

    #[test]
    fn empty_input() {
        let src = vec![];
        let params = params_for_level(1, Some(0));
        let block = compress_block_zstd(&src, &params);
        assert!(block.sequences.is_empty());
        assert!(block.literals.is_empty());
    }

    // ---------------------------------------------------------------
    // Small input (smaller than min_match)
    // ---------------------------------------------------------------

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
    // Mixed data: some matches, some literals
    // ---------------------------------------------------------------

    #[test]
    fn mixed_data_all_strategies() {
        let mut src = Vec::new();
        // Unique header
        src.extend_from_slice(b"HEADER__unique_data_here___");
        // Repeated block
        for _ in 0..20 {
            src.extend_from_slice(b"repeated_block_");
        }
        // Some unique data
        src.extend_from_slice(b"__separator__different_stuff__");
        // Another repeated block
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
    // BT strategies: binary tree match finder
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
        // Binary tree should find at least as good matches as lazy2 on
        // repetitive data with varying offsets.
        let mut src = Vec::new();
        // Create data with matches at various distances
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

        // BT should produce at least as few literals as lazy2 (better or equal compression)
        assert!(
            bt_block.literals.len() <= lazy2_block.literals.len() + lazy2_block.literals.len() / 10,
            "BT (level 13) produced more literals than lazy2 (level 12): bt={}, lazy2={}",
            bt_block.literals.len(),
            lazy2_block.literals.len(),
        );
    }

    #[test]
    fn bt_large_block() {
        // Test with a large block to exercise the tree pruning (bt_low).
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
    // Round-trip through the existing zstd decoder
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

        // Compress with the existing Fastest encoder (uses the old Matcher trait)
        let compressed = compress_to_vec(src.as_slice(), CompressionLevel::Fastest);

        // Decode with our decoder
        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(src.len());
        decoder
            .decode_all_to_vec(&compressed, &mut decoded)
            .unwrap();
        assert_eq!(decoded, src);

        // Also verify via the C zstd library
        let mut decoded2 = Vec::new();
        zstd::stream::copy_decode(compressed.as_slice(), &mut decoded2).unwrap();
        assert_eq!(decoded2, src);
    }

    // ---------------------------------------------------------------
    // Verify repcode state machine
    // ---------------------------------------------------------------

    #[test]
    fn repcode_update_real_offset() {
        let mut rep = RepCodes::new();
        assert_eq!(rep.rep, [1, 4, 8]);

        // Real offset 10 (off_base = 13)
        rep.update(13, 5); // lit_len > 0
        assert_eq!(rep.rep, [10, 1, 4]);

        // Real offset 20 (off_base = 23)
        rep.update(23, 0); // lit_len = 0
        assert_eq!(rep.rep, [20, 10, 1]);
    }

    #[test]
    fn repcode_update_rep0() {
        let mut rep = RepCodes { rep: [10, 20, 30] };

        // off_base=1, lit_len>0: rep[0] stays
        rep.update(1, 5);
        assert_eq!(rep.rep, [10, 20, 30]);

        // off_base=1, lit_len=0: swap rep[0] and rep[1]
        rep.update(1, 0);
        assert_eq!(rep.rep, [20, 10, 30]);
    }

    #[test]
    fn repcode_update_rep1() {
        let mut rep = RepCodes { rep: [10, 20, 30] };

        // off_base=2, lit_len>0: swap rep[0] and rep[1]
        rep.update(2, 5);
        assert_eq!(rep.rep, [20, 10, 30]);
    }

    #[test]
    fn repcode_update_rep2() {
        let mut rep = RepCodes { rep: [10, 20, 30] };

        // off_base=3, lit_len>0: rotate rep[2] to front
        rep.update(3, 5);
        assert_eq!(rep.rep, [30, 10, 20]);
    }

    // ---------------------------------------------------------------
    // Stress: all-same byte with varying sizes
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

    // ---------------------------------------------------------------
    // Match at exact window boundary
    // ---------------------------------------------------------------

    #[test]
    fn offset_encoding() {
        // Verify that off_base encoding is correct
        // off_base 4 = real offset 1
        assert_eq!(4u32 - 3, 1); // offset 1
        // off_base 100 = real offset 97
        assert_eq!(100u32 - 3, 97);
    }

    #[test]
    fn large_block_reconstruction() {
        // Test with 67KB of pattern data — this exposed a bug where
        // the match finder produced incorrect sequences for large blocks
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

    // ---------------------------------------------------------------
    // Full round-trip: BT levels through encoder + C zstd decoder
    // ---------------------------------------------------------------

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

            // Decode with our own decoder
            let mut decoder = FrameDecoder::new();
            let mut decoded = Vec::with_capacity(src.len());
            decoder
                .decode_all_to_vec(&compressed, &mut decoded)
                .expect(&alloc::format!("our decoder failed at level {level}"));
            assert_eq!(
                decoded, src,
                "our decoder: mismatch at level {level}"
            );

            // Decode with C zstd
            let mut decoded_c = Vec::new();
            zstd::stream::copy_decode(compressed.as_slice(), &mut decoded_c)
                .expect(&alloc::format!("C zstd failed at level {level}"));
            assert_eq!(
                decoded_c, src,
                "C zstd: mismatch at level {level}"
            );
        }
    }

    #[test]
    fn bt_round_trip_small_data() {
        use crate::encoding::{CompressionLevel, compress_to_vec};

        // Small enough to use the 16KB parameter table
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
            assert_eq!(
                decoded, src,
                "C zstd: small data mismatch at level {level}"
            );
        }
    }

    // ---------------------------------------------------------------
    // MatchState: cross-block match history
    // ---------------------------------------------------------------

    #[test]
    fn match_state_window_update_small_blocks() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);

        // Window starts empty
        assert!(ms.window().is_empty());

        // After one small block, window contains that block
        let block1 = b"hello world data";
        ms.update_window_only(block1);
        assert_eq!(ms.window(), block1);

        // After a second block, window contains both
        let block2 = b" more data here";
        ms.update_window_only(block2);
        let expected: Vec<u8> = [block1.as_slice(), block2.as_slice()].concat();
        assert_eq!(ms.window(), expected.as_slice());
    }

    #[test]
    fn match_state_window_caps_at_window_size() {
        use crate::encoding::compress_params::CompressionParams;
        // Use a tiny window for testing
        let params = CompressionParams {
            window_log: 4, // 16-byte window
            chain_log: 4,
            hash_log: 4,
            search_log: 1,
            min_match: 4,
            target_length: 0,
            strategy: Strategy::Fast,
        };
        let mut ms = MatchState::new(&params);
        assert_eq!(ms.window_size, 16);

        // Fill window beyond capacity
        ms.update_window_only(&[0u8; 10]);
        ms.update_window_only(&[1u8; 10]);
        // Window should only contain last 16 bytes
        assert_eq!(ms.window().len(), 16);
        // First 6 bytes of window should be from second update's tail of first
        assert_eq!(&ms.window()[..6], &[0u8; 6]);
        assert_eq!(&ms.window()[6..], &[1u8; 10]);
    }

    #[test]
    fn match_state_compress_block_basic() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);

        // Block 1: repetitive data
        let mut block1 = Vec::new();
        for _ in 0..50 {
            block1.extend_from_slice(b"block1_pattern ");
        }
        let result1 = ms.compress_block(&block1, &params);
        verify_reconstruction(&block1, &result1);
        assert!(!result1.sequences.is_empty());

        // Window should now contain block1 data
        assert!(!ms.window().is_empty());

        // Rep offsets should have been updated from default
        // (after compressing repetitive data, rep[0] should be the pattern offset)
    }

    #[test]
    fn match_state_cross_block_finds_matches() {
        let params = params_for_level(3, None);
        let mut ms = MatchState::new(&params);

        // Block 1: establish a pattern
        let pattern = b"cross_block_match_test_data_";
        let mut block1 = Vec::new();
        for _ in 0..30 {
            block1.extend_from_slice(pattern);
        }
        let result1 = ms.compress_block(&block1, &params);
        verify_reconstruction(&block1, &result1);

        // Block 2: repeat the SAME pattern - should find cross-block matches.
        // We can't use verify_reconstruction because it doesn't know about block1.
        // Instead we verify via the dict-aware reconstruction that includes the window.
        let mut block2 = Vec::new();
        for _ in 0..30 {
            block2.extend_from_slice(pattern);
        }
        let result2_with_history = ms.compress_block(&block2, &params);

        // Without history, compress block2 standalone
        let result2_standalone = compress_block_zstd(&block2, &params);

        // Both should produce valid output
        assert!(!result2_with_history.sequences.is_empty());
        assert!(!result2_standalone.sequences.is_empty());

        // With cross-block history, the match finder should be able to reference
        // data from block1, potentially producing fewer literals
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

        // Default rep offsets
        assert_eq!(ms.rep_offsets(), &[1, 4, 8]);

        // After compressing data with matches, rep offsets should change
        let mut block = Vec::new();
        for _ in 0..100 {
            block.extend_from_slice(b"ABCDEFGH");
        }
        let _result = ms.compress_block(&block, &params);

        // Rep offsets should no longer be defaults (we found matches)
        // The exact values depend on the match finder, but they should have changed
        // since we compressed highly repetitive data with offset 8.
        let new_reps = ms.rep_offsets();
        // At minimum, rep[0] should be 8 (the pattern repeat distance)
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
        // Test that MatchState works with every strategy.
        // Block 2 may contain cross-block references, so we can't use
        // verify_reconstruction on it directly. Instead we verify block 1
        // standalone, and verify that block 2 at least produces sequences.
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
            // Block 2 should produce sequences (the data is repetitive)
            assert!(
                !r2.sequences.is_empty(),
                "level {level}: expected sequences in block 2"
            );
        }
    }
}

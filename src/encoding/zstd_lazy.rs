//! Greedy, Lazy, Lazy2, and BtLazy2 match-finding strategies.
//!
//! Port of C zstd's `zstd_lazy.c`. Key algorithms:
//!
//! - **Greedy** (level 5): Hash chain search, take best match immediately.
//! - **Lazy** (levels 6-7): Search at pos, then pos+1; take better match.
//! - **Lazy2** (levels 8-12): Lazy + also check pos+2.
//! - **BtLazy2** (levels 13-15): Binary tree match finder + lazy2 evaluation.
//!
//! CRITICAL fix in this file: The binary tree match finder now uses the
//! `nextToUpdate` concept from C zstd's `ZSTD_DUBT_findBestMatch`. After
//! finding a match, `next_to_update` is set to `match_end_idx - 8`, which
//! skips over repetitive positions that would otherwise cause O(n^2) behavior.
//! This fixes the hang on 1MB repetitive text at L15/L19.

use alloc::vec;
use alloc::vec::Vec;

use super::compress_params::CompressionParams;
use super::hash::{count_match, hash_ptr};
use super::match_state::{
    CompressedBlock, MatchCandidate, MatchFound, RepCodes, SequenceOut, build_block,
    build_block_dict, insert_hash_chain_ext, prefill_hash_chain_ext, search_hash_chain_ext,
};

// ---------------------------------------------------------------------------
// Lazy skipping constant (from C zstd)
// ---------------------------------------------------------------------------

/// When step exceeds this, enable lazy skipping mode (fewer hash insertions).
const K_LAZY_SKIPPING_STEP: usize = 8;

// ---------------------------------------------------------------------------
// Greedy strategy
// ---------------------------------------------------------------------------

/// Greedy strategy using externally-owned hash + chain tables.
///
/// Now uses `next_to_update` for lazy hash insertion, matching C zstd's
/// `ZSTD_insertAndFindFirstIndex` catch-up behavior.
pub fn compress_greedy_ext(
    src: &[u8],
    params: &CompressionParams,
    htable: &mut [u32],
    chain: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);
    let mut next_to_update: usize = 0;
    let lazy_skipping = false;

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let best = search_hash_chain_ext(
            src,
            pos,
            htable,
            chain,
            &rep,
            params,
            lit_len,
            &mut next_to_update,
            lazy_skipping,
        );

        if let Some(m) = best {
            literals.extend_from_slice(&src[anchor..pos]);
            let seq = SequenceOut {
                off_base: m.off_base,
                lit_len,
                match_len: m.match_len,
            };
            rep.update(seq.off_base, lit_len);
            sequences.push(seq);

            let match_end = pos + m.match_len as usize;
            // Insert positions within the match
            for p in (pos + 1)..match_end.min(end) {
                insert_hash_chain_ext(src, p, htable, chain, params);
            }
            next_to_update = match_end;
            pos = match_end;
            anchor = pos;
        } else {
            pos += 1;
        }
    }

    build_block(src, sequences, literals, anchor)
}

/// Greedy strategy with dict prefix, using externally-owned tables.
pub fn compress_greedy_dict_ext(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    htable: &mut [u32],
    chain: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    let mut next_to_update: usize = dict_len;

    while pos < end {
        let ll = (pos - anchor) as u32;
        let best = search_hash_chain_ext(
            combined,
            pos,
            htable,
            chain,
            &rep,
            params,
            ll,
            &mut next_to_update,
            false,
        );
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
                insert_hash_chain_ext(combined, p, htable, chain, params);
            }
            next_to_update = me;
            pos = me;
            anchor = pos;
        } else {
            pos += 1;
        }
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Lazy strategy
// ---------------------------------------------------------------------------

/// Lazy strategy using externally-owned hash + chain tables.
pub fn compress_lazy_ext(
    src: &[u8],
    params: &CompressionParams,
    htable: &mut [u32],
    chain: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);
    let mut next_to_update: usize = 0;
    let mut lazy_skipping = false;

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let match0 = search_hash_chain_ext(
            src,
            pos,
            htable,
            chain,
            &rep,
            params,
            lit_len,
            &mut next_to_update,
            lazy_skipping,
        );

        let match0 = match match0 {
            Some(m) => m,
            None => {
                let step = ((pos - anchor) >> 8) + 1;
                pos += step;
                lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                continue;
            }
        };

        // Lazy: check pos+1 for a better match
        if pos + 1 < end {
            let lit_len1 = (pos + 1 - anchor) as u32;
            let match1 = search_hash_chain_ext(
                src,
                pos + 1,
                htable,
                chain,
                &rep,
                params,
                lit_len1,
                &mut next_to_update,
                false,
            );

            if let Some(m1) = match1 {
                if match0.is_better_lazy(&m1) {
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
                        insert_hash_chain_ext(src, p, htable, chain, params);
                    }
                    next_to_update = match_end;
                    pos = match_end;
                    anchor = pos;
                    lazy_skipping = false;
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
            insert_hash_chain_ext(src, p, htable, chain, params);
        }
        next_to_update = match_end;
        pos = match_end;
        anchor = pos;
        lazy_skipping = false;
    }

    build_block(src, sequences, literals, anchor)
}

/// Lazy strategy with dict prefix, using externally-owned tables.
pub fn compress_lazy_dict_ext(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    htable: &mut [u32],
    chain: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    let mut next_to_update: usize = dict_len;
    let mut lazy_skipping = false;

    while pos < end {
        let ll = (pos - anchor) as u32;
        let m0 = match search_hash_chain_ext(
            combined,
            pos,
            htable,
            chain,
            &rep,
            params,
            ll,
            &mut next_to_update,
            lazy_skipping,
        ) {
            Some(m) => m,
            None => {
                let step = ((pos - anchor) >> 8) + 1;
                pos += step;
                lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                continue;
            }
        };
        if pos + 1 < end {
            let ll1 = (pos + 1 - anchor) as u32;
            if let Some(m1) = search_hash_chain_ext(
                combined,
                pos + 1,
                htable,
                chain,
                &rep,
                params,
                ll1,
                &mut next_to_update,
                false,
            ) {
                if m0.is_better_lazy(&m1) {
                    let llo = (pos + 1 - anchor) as u32;
                    literals.extend_from_slice(&combined[anchor..pos + 1]);
                    let seq = SequenceOut {
                        off_base: m1.off_base,
                        lit_len: llo,
                        match_len: m1.match_len,
                    };
                    rep.update(seq.off_base, llo);
                    sequences.push(seq);
                    let me = pos + 1 + m1.match_len as usize;
                    for p in (pos + 2)..me.min(end) {
                        insert_hash_chain_ext(combined, p, htable, chain, params);
                    }
                    next_to_update = me;
                    pos = me;
                    anchor = pos;
                    lazy_skipping = false;
                    continue;
                }
            }
        }
        literals.extend_from_slice(&combined[anchor..pos]);
        let seq = SequenceOut {
            off_base: m0.off_base,
            lit_len: ll,
            match_len: m0.match_len,
        };
        rep.update(seq.off_base, ll);
        sequences.push(seq);
        let me = pos + m0.match_len as usize;
        for p in (pos + 1)..me.min(end) {
            insert_hash_chain_ext(combined, p, htable, chain, params);
        }
        next_to_update = me;
        pos = me;
        anchor = pos;
        lazy_skipping = false;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Lazy2 strategy
// ---------------------------------------------------------------------------

/// Lazy2 strategy using externally-owned hash + chain tables.
pub fn compress_lazy2_ext(
    src: &[u8],
    params: &CompressionParams,
    htable: &mut [u32],
    chain: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);
    let mut next_to_update: usize = 0;
    let mut lazy_skipping = false;

    while pos < end {
        let lit_len = (pos - anchor) as u32;
        let match0 = search_hash_chain_ext(
            src,
            pos,
            htable,
            chain,
            &rep,
            params,
            lit_len,
            &mut next_to_update,
            lazy_skipping,
        );

        let match0 = match match0 {
            Some(m) => m,
            None => {
                let step = ((pos - anchor) >> 8) + 1;
                pos += step;
                lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                continue;
            }
        };

        let mut best_pos = pos;
        let mut best_match = match0;

        if pos + 1 < end {
            let lit_len1 = (pos + 1 - anchor) as u32;
            let match1 = search_hash_chain_ext(
                src,
                pos + 1,
                htable,
                chain,
                &rep,
                params,
                lit_len1,
                &mut next_to_update,
                false,
            );

            if let Some(m1) = match1 {
                if best_match.is_better_lazy(&m1) {
                    best_pos = pos + 1;
                    best_match = m1;

                    if pos + 2 < end {
                        let lit_len2 = (pos + 2 - anchor) as u32;
                        let match2 = search_hash_chain_ext(
                            src,
                            pos + 2,
                            htable,
                            chain,
                            &rep,
                            params,
                            lit_len2,
                            &mut next_to_update,
                            false,
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
        for p in (pos + 1)..match_end.min(end) {
            let already_inserted =
                p == pos || p == pos + 1 || (p == pos + 2 && best_pos >= pos + 2);
            if !already_inserted {
                insert_hash_chain_ext(src, p, htable, chain, params);
            }
        }
        next_to_update = match_end;
        pos = match_end;
        anchor = pos;
        lazy_skipping = false;
    }

    build_block(src, sequences, literals, anchor)
}

/// Lazy2 strategy with dict prefix, using externally-owned tables.
pub fn compress_lazy2_dict_ext(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    htable: &mut [u32],
    chain: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);
    let mut next_to_update: usize = dict_len;
    let mut lazy_skipping = false;

    while pos < end {
        let ll = (pos - anchor) as u32;
        let m0 = match search_hash_chain_ext(
            combined,
            pos,
            htable,
            chain,
            &rep,
            params,
            ll,
            &mut next_to_update,
            lazy_skipping,
        ) {
            Some(m) => m,
            None => {
                let step = ((pos - anchor) >> 8) + 1;
                pos += step;
                lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                continue;
            }
        };
        let mut best_pos = pos;
        let mut best_match = m0;
        if pos + 1 < end {
            let ll1 = (pos + 1 - anchor) as u32;
            if let Some(m1) = search_hash_chain_ext(
                combined,
                pos + 1,
                htable,
                chain,
                &rep,
                params,
                ll1,
                &mut next_to_update,
                false,
            ) {
                if best_match.is_better_lazy(&m1) {
                    best_pos = pos + 1;
                    best_match = m1;
                    if pos + 2 < end {
                        let ll2 = (pos + 2 - anchor) as u32;
                        if let Some(m2) = search_hash_chain_ext(
                            combined,
                            pos + 2,
                            htable,
                            chain,
                            &rep,
                            params,
                            ll2,
                            &mut next_to_update,
                            false,
                        ) {
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
                insert_hash_chain_ext(combined, p, htable, chain, params);
            }
        }
        next_to_update = me;
        pos = me;
        anchor = pos;
        lazy_skipping = false;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Binary Tree match finder (used by BtLazy2)
// ---------------------------------------------------------------------------

/// Binary tree table: each position has two entries (smaller_child, larger_child).
///
/// The C zstd reference calls this a "Dynamic Unsorted Binary Tree" (DUBT).
///
/// Memory layout: `tree[2 * (pos & bt_mask)]` = smaller child index,
///                `tree[2 * (pos & bt_mask) + 1]` = larger child index.
pub struct BinaryTree {
    /// Hash table: maps hash -> tree root position.
    pub hash_table: Vec<u32>,
    pub hash_log: u32,
    /// Binary tree storage: 2 entries per position (smaller, larger).
    pub tree: Vec<u32>,
    /// Mask for tree position indexing.
    pub bt_mask: u32,
    /// CRITICAL: Lazy insertion point. Positions from here to current pos
    /// have not been inserted into the tree yet. C zstd's `nextToUpdate`.
    pub next_to_update: usize,
}

impl BinaryTree {
    pub fn new(hash_log: u32, chain_log: u32) -> Self {
        let bt_log = chain_log.saturating_sub(1);
        let bt_size = 1usize << bt_log;
        Self {
            hash_table: vec![0; 1 << hash_log],
            hash_log,
            tree: vec![0; 2 * bt_size],
            bt_mask: (bt_size as u32).wrapping_sub(1),
            next_to_update: 0,
        }
    }

    /// Get the (smaller_child, larger_child) pair indices for a position.
    #[inline]
    fn children_idx(&self, pos: u32) -> (usize, usize) {
        let base = 2 * (pos & self.bt_mask) as usize;
        (base, base + 1)
    }

    /// Update the tree: insert positions from `next_to_update` up to (not including) `target`.
    /// This is C zstd's `ZSTD_updateDUBT`.
    pub fn update_tree(
        &mut self,
        src: &[u8],
        target: usize,
        min_match: usize,
        search_depth: usize,
        window_low: usize,
    ) {
        while self.next_to_update < target {
            if self.next_to_update + 8 > src.len() {
                self.next_to_update = target;
                break;
            }
            self.insert_only(
                src,
                self.next_to_update,
                min_match,
                search_depth,
                window_low,
            );
            self.next_to_update += 1;
        }
    }

    /// Insert `pos` into the tree and simultaneously search for the best match.
    ///
    /// CRITICAL FIX: After the search, sets `next_to_update = match_end_idx - 8`
    /// to skip repetitive patterns, matching C zstd's `ZSTD_DUBT_findBestMatch`.
    pub fn insert_and_find(
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

        let bt_low = pos.saturating_sub(self.bt_mask as usize);

        let (smaller_idx, larger_idx) = self.children_idx(pos as u32);
        let mut smaller_slot = smaller_idx;
        let mut larger_slot = larger_idx;

        let mut common_len_smaller: usize = 0;
        let mut common_len_larger: usize = 0;

        let mut best: Option<MatchCandidate> = None;
        let mut candidate = match_index;
        let mut depth = search_depth;

        // Track the furthest match end position (for nextToUpdate skip)
        let mut match_end_idx: usize = pos + 8 + 1;

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

            if match_len >= min_match {
                // Track match_end_idx for the nextToUpdate skip
                if match_len > match_end_idx.wrapping_sub(candidate) {
                    match_end_idx = candidate + match_len;
                }

                let dist = pos - candidate;
                let cand = MatchCandidate {
                    off_base: dist as u32 + 3,
                    match_len: match_len as u32,
                };
                if best.is_none_or(|b| {
                    cand.match_len > b.match_len
                        || (cand.match_len == b.match_len && cand.gain() > b.gain())
                }) {
                    best = Some(cand);
                }
            }

            if pos + match_len >= src.len() || candidate + match_len >= src.len() {
                self.tree[smaller_slot] = 0;
                self.tree[larger_slot] = 0;
                // CRITICAL: Skip repetitive patterns
                self.next_to_update = match_end_idx.saturating_sub(8);
                return best;
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

        // CRITICAL: Skip repetitive patterns (C zstd line 381)
        // ms->nextToUpdate = matchEndIdx - 8;
        self.next_to_update = match_end_idx.saturating_sub(8);

        best
    }

    /// Insert `pos` into the tree and return ALL matches found, sorted by
    /// increasing length. Used by the optimal parser.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_and_find_all(
        &mut self,
        src: &[u8],
        pos: usize,
        min_match: usize,
        best_length_in: usize,
        search_depth: usize,
        window_low: usize,
        matches_out: &mut Vec<MatchFound>,
    ) {
        matches_out.clear();

        if pos + 8 > src.len() {
            return;
        }

        let h = hash_ptr(&src[pos..], self.hash_log, min_match as u32);
        let match_index = self.hash_table[h] as usize;
        self.hash_table[h] = pos as u32;

        let bt_low = pos.saturating_sub(self.bt_mask as usize);

        let (smaller_idx, larger_idx) = self.children_idx(pos as u32);
        let mut smaller_slot = smaller_idx;
        let mut larger_slot = larger_idx;

        let mut common_len_smaller: usize = 0;
        let mut common_len_larger: usize = 0;

        let mut best_length: usize = best_length_in;
        let mut candidate = match_index;
        let mut depth = search_depth;

        /// Maximum positions the optimal parser considers in one forward pass.
        const ZSTD_OPT_NUM: usize = 1 << 12;

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

            if match_len > best_length {
                best_length = match_len;
                let dist = pos - candidate;
                matches_out.push(MatchFound {
                    off_base: dist as u32 + 3,
                    len: match_len as u32,
                });

                if match_len > ZSTD_OPT_NUM
                    || pos + match_len >= src.len()
                    || candidate + match_len >= src.len()
                {
                    self.tree[smaller_slot] = 0;
                    self.tree[larger_slot] = 0;
                    return;
                }
            }

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

    /// Insert a position into the tree without searching for matches.
    pub fn insert_only(
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

        let bt_low = pos.saturating_sub(self.bt_mask as usize);

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
///
/// CRITICAL: Uses `bt.update_tree()` before searching to catch up any deferred
/// positions, then calls `insert_and_find` which sets `next_to_update` to skip
/// repetitive patterns.
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
    let window_low = pos.saturating_sub(window_size);

    // CRITICAL: If we're in a skipped area (pos < next_to_update), return nothing.
    // This matches C zstd line 402: "if (ip < ms->window.base + ms->nextToUpdate) return 0;"
    if pos < bt.next_to_update {
        return None;
    }

    // Update tree: insert deferred positions from next_to_update to pos.
    bt.update_tree(src, pos, min_match, search_depth, window_low);

    // Check repcodes first (they're free to encode)
    let mut best = super::match_state::try_repcodes(src, pos, rep, min_match as u32, lit_len);

    // Search the binary tree
    if let Some(bt_match) = bt.insert_and_find(src, pos, min_match, search_depth, window_low) {
        if best.is_none_or(|b| {
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
fn insert_binary_tree(src: &[u8], pos: usize, bt: &mut BinaryTree, params: &CompressionParams) {
    let min_match = params.min_match.max(4) as usize;
    let search_depth = params.search_depth();
    let window_size = params.window_size();
    let window_low = pos.saturating_sub(window_size);
    bt.insert_only(src, pos, min_match, search_depth, window_low);
}

// ---------------------------------------------------------------------------
// BtLazy2 strategy
// ---------------------------------------------------------------------------

/// BtLazy2: Binary tree match finder + lazy2 evaluation.
pub fn compress_btlazy2(src: &[u8], params: &CompressionParams) -> CompressedBlock {
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
        // Insert skipped + match-interior positions.
        // search_binary_tree already used update_tree + insert_and_find for
        // the positions it searched, and set next_to_update to skip ahead.
        // We only need to insert positions not yet in the tree.
        for p in (pos + 1)..match_end.min(end) {
            // Only insert if not already handled by search_binary_tree
            if p >= bt.next_to_update {
                insert_binary_tree(src, p, &mut bt, params);
            }
        }
        bt.next_to_update = bt.next_to_update.max(match_end);
        pos = match_end;
        anchor = pos;
    }

    build_block(src, sequences, literals, anchor)
}

pub fn compress_btlazy2_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut bt = BinaryTree::new(params.hash_log, params.chain_log);
    prefill_binary_tree(&mut bt, combined, dict_len, params);
    let mut rep = RepCodes { rep: *initial_rep };
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
            if p >= bt.next_to_update {
                insert_binary_tree(combined, p, &mut bt, params);
            }
        }
        bt.next_to_update = bt.next_to_update.max(me);
        pos = me;
        anchor = pos;
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// Standalone wrappers
// ---------------------------------------------------------------------------

pub fn compress_greedy(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    let mut chain = vec![0u32; 1 << params.chain_log];
    compress_greedy_ext(src, params, &mut htable, &mut chain, &[1, 4, 8])
}

pub fn compress_lazy(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    let mut chain = vec![0u32; 1 << params.chain_log];
    compress_lazy_ext(src, params, &mut htable, &mut chain, &[1, 4, 8])
}

pub fn compress_lazy2(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    let mut chain = vec![0u32; 1 << params.chain_log];
    compress_lazy2_ext(src, params, &mut htable, &mut chain, &[1, 4, 8])
}

pub fn compress_greedy_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    let mut chain = vec![0u32; 1 << params.chain_log];
    prefill_hash_chain_ext(
        &mut htable,
        &mut chain,
        params.hash_log,
        combined,
        dict_len,
        params,
    );
    compress_greedy_dict_ext(
        combined,
        dict_len,
        params,
        &mut htable,
        &mut chain,
        initial_rep,
    )
}

pub fn compress_lazy_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    let mut chain = vec![0u32; 1 << params.chain_log];
    prefill_hash_chain_ext(
        &mut htable,
        &mut chain,
        params.hash_log,
        combined,
        dict_len,
        params,
    );
    compress_lazy_dict_ext(
        combined,
        dict_len,
        params,
        &mut htable,
        &mut chain,
        initial_rep,
    )
}

pub fn compress_lazy2_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    let mut chain = vec![0u32; 1 << params.chain_log];
    prefill_hash_chain_ext(
        &mut htable,
        &mut chain,
        params.hash_log,
        combined,
        dict_len,
        params,
    );
    compress_lazy2_dict_ext(
        combined,
        dict_len,
        params,
        &mut htable,
        &mut chain,
        initial_rep,
    )
}

/// Prefill the binary tree's HASH TABLE with dict positions.
///
/// Unlike the old approach that did full tree insertions for every dict position
/// (O(n * search_depth)), this only updates the hash table, matching C zstd's
/// `ZSTD_updateDUBT` which chains unsorted entries. The actual tree insertion
/// happens lazily during `update_tree` / `insert_and_find`.
///
/// This is critical for performance: a 2MB window with full tree prefill was
/// O(n^2), causing hangs on 1MB repetitive data at L15/L19.
pub fn prefill_binary_tree(
    bt: &mut BinaryTree,
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
) {
    if dict_len < 8 {
        bt.next_to_update = dict_len;
        return;
    }
    let min_match = params.min_match.max(4);
    let end = dict_len.saturating_sub(7);

    // Only prefill the hash table (like C zstd's ZSTD_updateDUBT).
    // Each position's hash points to the previous position with the same hash,
    // stored as an unsorted chain via the tree's smaller_child slot.
    for pos in 0..end {
        if pos + 8 <= combined.len() {
            let h = hash_ptr(&combined[pos..], bt.hash_log, min_match);
            let match_index = bt.hash_table[h];
            let (smaller_idx, larger_idx) = bt.children_idx(pos as u32);
            bt.hash_table[h] = pos as u32;
            // Store previous entry as unsorted chain
            bt.tree[smaller_idx] = match_index;
            // Mark as unsorted (use 1 as sentinel since 0 would be a valid "no child")
            bt.tree[larger_idx] = 1; // ZSTD_DUBT_UNSORTED_MARK equivalent
        }
    }
    // Set next_to_update to dict_len so the search knows dict positions
    // are in the hash table but may need sorting during the actual search.
    // The update_tree call in search_binary_tree will handle positions
    // from dict_len onward.
    bt.next_to_update = dict_len;
}

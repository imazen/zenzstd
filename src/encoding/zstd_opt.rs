//! Optimal parsing match finder for BtOpt/BtUltra/BtUltra2 strategies.
//!
//! Port of C zstd's `zstd_opt.c`. Uses price-based optimal parsing:
//! 1. Find all matches at the current position
//! 2. Forward-fill a price table evaluating literal vs match at each position
//! 3. Backtrack to find the minimum-cost sequence
//! 4. Emit those sequences

use alloc::vec;
use alloc::vec::Vec;

use super::compress_params::{CompressionParams, Strategy};
use super::hash::{count_match, hash3};
use super::match_state::{CompressedBlock, MatchFound, RepCodes, SequenceOut, build_block};
use super::zstd_lazy::{BinaryTree, prefill_binary_tree};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of positions the optimal parser considers in one forward pass.
const ZSTD_OPT_NUM: usize = 1 << 12;

/// Size of the opt table (OPT_NUM + 3 for backtrack headroom).
const ZSTD_OPT_SIZE: usize = ZSTD_OPT_NUM + 3;

/// Cost multiplier for fractional-bit precision (8 fractional bits = 256x).
const BITCOST_MULTIPLIER: i32 = 1 << 8;

/// Sentinel cost: unreachable position.
const ZSTD_MAX_PRICE: i32 = 1 << 30;

/// LL_bits table from the zstd specification.
const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

/// ML_bits table from the zstd specification.
const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// Maximum symbol values from the zstd specification.
const MAX_LIT: usize = 255;
const MAX_LL: usize = 35;
const MAX_ML: usize = 52;
const MAX_OFF: usize = 31;

// ---------------------------------------------------------------------------
// Cost model helpers
// ---------------------------------------------------------------------------

/// Compute the LL FSE code for a literal length.
#[inline]
fn ll_code(lit_length: u32) -> u32 {
    #[rustfmt::skip]
    const LL_CODE: [u8; 64] = [
         0,  1,  2,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13, 14, 15,
        16, 16, 17, 17, 18, 18, 19, 19,
        20, 20, 20, 20, 21, 21, 21, 21,
        22, 22, 22, 22, 22, 22, 22, 22,
        23, 23, 23, 23, 23, 23, 23, 23,
        24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24,
    ];
    const LL_DELTA_CODE: u32 = 19;
    if lit_length > 63 {
        highbit32(lit_length) + LL_DELTA_CODE
    } else {
        LL_CODE[lit_length as usize] as u32
    }
}

/// Compute the ML FSE code for a match length base.
#[inline]
fn ml_code(ml_base: u32) -> u32 {
    #[rustfmt::skip]
    const ML_CODE: [u8; 128] = [
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
        38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    ];
    const ML_DELTA_CODE: u32 = 36;
    if ml_base > 127 {
        highbit32(ml_base) + ML_DELTA_CODE
    } else {
        ML_CODE[ml_base as usize] as u32
    }
}

#[inline]
fn highbit32(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

#[inline]
fn frac_weight(raw_stat: u32) -> i32 {
    let stat = raw_stat + 1;
    let hb = highbit32(stat);
    let b_weight = hb as i32 * BITCOST_MULTIPLIER;
    let f_weight = ((stat << 8) >> hb) as i32;
    b_weight + f_weight
}

#[inline]
fn bit_weight(raw_stat: u32) -> i32 {
    (highbit32(raw_stat + 1) as i32) * BITCOST_MULTIPLIER
}

#[inline]
fn weight(stat: u32, opt_level: i32) -> i32 {
    if opt_level > 0 {
        frac_weight(stat)
    } else {
        bit_weight(stat)
    }
}

// ---------------------------------------------------------------------------
// OptState: frequency tracking and cost model
// ---------------------------------------------------------------------------

struct OptState {
    lit_freq: Vec<u32>,
    lit_length_freq: Vec<u32>,
    match_length_freq: Vec<u32>,
    off_code_freq: Vec<u32>,
    lit_sum: u32,
    lit_length_sum: u32,
    match_length_sum: u32,
    off_code_sum: u32,
    lit_sum_base_price: i32,
    lit_length_sum_base_price: i32,
    match_length_sum_base_price: i32,
    off_code_sum_base_price: i32,
    opt_level: i32,
}

impl OptState {
    fn new(opt_level: i32) -> Self {
        Self {
            lit_freq: vec![0; MAX_LIT + 1],
            lit_length_freq: vec![0; MAX_LL + 1],
            match_length_freq: vec![0; MAX_ML + 1],
            off_code_freq: vec![0; MAX_OFF + 1],
            lit_sum: 0,
            lit_length_sum: 0,
            match_length_sum: 0,
            off_code_sum: 0,
            lit_sum_base_price: 0,
            lit_length_sum_base_price: 0,
            match_length_sum_base_price: 0,
            off_code_sum_base_price: 0,
            opt_level,
        }
    }

    fn set_base_prices(&mut self) {
        self.lit_sum_base_price = weight(self.lit_sum, self.opt_level);
        self.lit_length_sum_base_price = weight(self.lit_length_sum, self.opt_level);
        self.match_length_sum_base_price = weight(self.match_length_sum, self.opt_level);
        self.off_code_sum_base_price = weight(self.off_code_sum, self.opt_level);
    }

    fn init_from_source(&mut self, src: &[u8]) {
        for &b in src {
            self.lit_freq[b as usize] += 1;
        }
        self.lit_sum = downscale_stats(&mut self.lit_freq, MAX_LIT, 8, false);

        #[rustfmt::skip]
        let base_ll_freqs: [u32; MAX_LL + 1] = [
            4, 2, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1,
        ];
        self.lit_length_freq.copy_from_slice(&base_ll_freqs);
        self.lit_length_sum = base_ll_freqs.iter().sum();

        for f in self.match_length_freq.iter_mut() {
            *f = 1;
        }
        self.match_length_sum = (MAX_ML + 1) as u32;

        #[rustfmt::skip]
        let base_of_freqs: [u32; MAX_OFF + 1] = [
            6, 2, 1, 1, 2, 3, 4, 4,
            4, 3, 2, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        ];
        self.off_code_freq.copy_from_slice(&base_of_freqs);
        self.off_code_sum = base_of_freqs.iter().sum();

        self.set_base_prices();
    }

    fn rescale(&mut self) {
        self.lit_sum = scale_stats(&mut self.lit_freq, MAX_LIT, 12);
        self.lit_length_sum = scale_stats(&mut self.lit_length_freq, MAX_LL, 11);
        self.match_length_sum = scale_stats(&mut self.match_length_freq, MAX_ML, 11);
        self.off_code_sum = scale_stats(&mut self.off_code_freq, MAX_OFF, 11);
        self.set_base_prices();
    }

    #[inline]
    fn single_literal_cost(&self, byte: u8) -> i32 {
        let mut lit_price = weight(self.lit_freq[byte as usize], self.opt_level);
        let lit_price_max = self.lit_sum_base_price - BITCOST_MULTIPLIER;
        if lit_price > lit_price_max {
            lit_price = lit_price_max;
        }
        self.lit_sum_base_price - lit_price
    }

    fn lit_length_price(&self, lit_length: u32) -> i32 {
        let ll_c = ll_code(lit_length);
        (LL_BITS[ll_c as usize] as i32) * BITCOST_MULTIPLIER + self.lit_length_sum_base_price
            - weight(self.lit_length_freq[ll_c as usize], self.opt_level)
    }

    fn match_price(&self, off_base: u32, match_length: u32) -> i32 {
        let off_code = highbit32(off_base);
        let ml_base = match_length - 3;

        let mut price = (off_code as i32) * BITCOST_MULTIPLIER
            + (self.off_code_sum_base_price
                - weight(self.off_code_freq[off_code as usize], self.opt_level));

        if self.opt_level < 2 && off_code >= 20 {
            price += ((off_code - 19) * 2) as i32 * BITCOST_MULTIPLIER;
        }

        let ml_c = ml_code(ml_base);
        price += (ML_BITS[ml_c as usize] as i32) * BITCOST_MULTIPLIER
            + (self.match_length_sum_base_price
                - weight(self.match_length_freq[ml_c as usize], self.opt_level));

        price += BITCOST_MULTIPLIER / 5;

        price
    }

    fn update_stats(&mut self, lit_length: u32, literals: &[u8], off_base: u32, match_length: u32) {
        for &b in literals {
            self.lit_freq[b as usize] += 2;
            self.lit_sum += 2;
        }

        let ll_c = ll_code(lit_length) as usize;
        self.lit_length_freq[ll_c] += 1;
        self.lit_length_sum += 1;

        let off_c = highbit32(off_base) as usize;
        if off_c <= MAX_OFF {
            self.off_code_freq[off_c] += 1;
            self.off_code_sum += 1;
        }

        let ml_base = match_length - 3;
        let ml_c = ml_code(ml_base) as usize;
        self.match_length_freq[ml_c] += 1;
        self.match_length_sum += 1;
    }
}

fn downscale_stats(table: &mut [u32], last_elt_index: usize, shift: u32, base1: bool) -> u32 {
    let mut sum = 0u32;
    for entry in table.iter_mut().take(last_elt_index + 1) {
        let base = if base1 { 1 } else { u32::from(*entry > 0) };
        let new_stat = base + (*entry >> shift);
        sum += new_stat;
        *entry = new_stat;
    }
    sum
}

#[allow(dead_code)]
fn scale_stats(table: &mut [u32], last_elt_index: usize, log_target: u32) -> u32 {
    let prev_sum: u32 = table[..=last_elt_index].iter().sum();
    let factor = prev_sum >> log_target;
    if factor <= 1 {
        return prev_sum;
    }
    downscale_stats(table, last_elt_index, highbit32(factor), true)
}

// ---------------------------------------------------------------------------
// OptNode
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct OptNode {
    price: i32,
    off: u32,
    mlen: u32,
    litlen: u32,
    rep: [u32; 3],
}

impl Default for OptNode {
    fn default() -> Self {
        Self {
            price: ZSTD_MAX_PRICE,
            off: 0,
            mlen: 0,
            litlen: 0,
            rep: [1, 4, 8],
        }
    }
}

/// Compute new rep offsets after encoding a sequence.
fn new_rep(rep: &[u32; 3], off_base: u32, ll0: bool) -> [u32; 3] {
    let mut r = *rep;
    if off_base > 3 {
        r[2] = r[1];
        r[1] = r[0];
        r[0] = off_base - 3;
    } else {
        let rep_code = (off_base - 1 + if ll0 { 1 } else { 0 }) as usize;
        if rep_code > 0 {
            let current_offset = if rep_code == 3 {
                r[0].wrapping_sub(1)
            } else {
                r[rep_code]
            };
            if rep_code >= 2 {
                r[2] = r[1];
            }
            r[1] = r[0];
            r[0] = current_offset;
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Hash3 table for 3-byte matches
// ---------------------------------------------------------------------------

struct Hash3Table {
    table: Vec<u32>,
    hash_log: u32,
    next_to_update: usize,
}

impl Hash3Table {
    fn new(window_log: u32) -> Self {
        let hash_log = window_log.min(17);
        Self {
            table: vec![0; 1 << hash_log],
            hash_log,
            next_to_update: 0,
        }
    }

    fn insert_and_find(&mut self, src: &[u8], pos: usize) -> u32 {
        let mut idx = self.next_to_update;
        while idx < pos {
            if idx + 4 <= src.len() {
                let h = hash3(&src[idx..], self.hash_log) as usize;
                self.table[h] = idx as u32;
            }
            idx += 1;
        }
        self.next_to_update = pos;
        if pos + 4 <= src.len() {
            let h = hash3(&src[pos..], self.hash_log) as usize;
            self.table[h]
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// get_all_matches
// ---------------------------------------------------------------------------

/// Get all matches at `pos` using the binary tree, including repcodes and hash3.
#[allow(clippy::too_many_arguments)]
fn get_all_matches(
    bt: &mut BinaryTree,
    src: &[u8],
    pos: usize,
    rep: &[u32; 3],
    ll0: bool,
    min_match: usize,
    search_depth: usize,
    window_size: usize,
    sufficient_len: usize,
    hash3_table: Option<&mut Hash3Table>,
    matches_out: &mut Vec<MatchFound>,
) {
    matches_out.clear();

    if pos + 8 > src.len() {
        return;
    }

    let window_low = pos.saturating_sub(window_size);

    // Update tree for skipped positions
    let min_match_u = min_match;
    bt.update_tree(src, pos, min_match_u, search_depth, window_low);

    let remaining = src.len() - pos;
    if remaining < min_match {
        return;
    }

    // Check repcodes first
    let last_r = 3 + if ll0 { 1 } else { 0 };
    let mut best_length = min_match.saturating_sub(1);
    let start_rep = if ll0 { 1usize } else { 0 };

    for rep_idx in start_rep..last_r {
        let rep_offset = if rep_idx == 3 {
            rep[0].wrapping_sub(1)
        } else {
            rep[rep_idx]
        };
        if rep_offset == 0 || rep_offset as usize > pos {
            continue;
        }
        let ref_pos = pos - rep_offset as usize;

        if src[pos..pos + min_match.min(remaining)]
            != src[ref_pos..ref_pos + min_match.min(remaining)]
        {
            continue;
        }

        let ml = count_match(&src[pos..], &src[ref_pos..]);
        if ml >= min_match && ml > best_length {
            best_length = ml;
            let off_base = if ll0 {
                rep_idx as u32
            } else {
                (rep_idx as u32) + 1
            };
            matches_out.push(MatchFound {
                off_base,
                len: ml as u32,
            });
            if ml >= sufficient_len || pos + ml >= src.len() {
                bt.next_to_update = bt.next_to_update.max(pos + 1);
                return;
            }
        }
    }

    // HC3 match finder for mls==3
    if let Some(h3) = hash3_table {
        if min_match == 3 && best_length < 3 {
            let match_index = h3.insert_and_find(src, pos) as usize;
            let match_low = if window_low > 0 { window_low } else { 1 };
            if match_index >= match_low && match_index < pos && pos - match_index < (1 << 18) {
                let ml = count_match(&src[pos..], &src[match_index..]);
                if ml >= 3 {
                    best_length = ml;
                    matches_out.push(MatchFound {
                        off_base: (pos - match_index) as u32 + 3,
                        len: ml as u32,
                    });
                    if ml > sufficient_len || pos + ml >= src.len() {
                        bt.next_to_update = bt.next_to_update.max(pos + 1);
                        return;
                    }
                }
            }
        }
    }

    // Search binary tree for all matches.
    let mut bt_matches = Vec::new();
    bt.insert_and_find_all(
        src,
        pos,
        min_match,
        best_length,
        search_depth,
        window_low,
        &mut bt_matches,
    );

    for m in &bt_matches {
        if m.len as usize > best_length {
            best_length = m.len as usize;
            matches_out.push(*m);
            if best_length >= sufficient_len || pos + best_length >= src.len() {
                break;
            }
        }
    }

    bt.next_to_update = bt.next_to_update.max(pos + 1);
}

// ---------------------------------------------------------------------------
// Optimal parser entry points
// ---------------------------------------------------------------------------

/// Optimal parsing compressor for BtOpt/BtUltra/BtUltra2.
pub fn compress_btopt(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    if params.strategy == Strategy::BtUltra2 && src.len() > 8 {
        let mut stats_state = OptState::new(2);
        stats_state.init_from_source(src);
        compress_optimal_generic_with_stats(src, params, &[1, 4, 8], 0, &mut stats_state, true);
        compress_optimal_generic_with_stats(src, params, &[1, 4, 8], 0, &mut stats_state, false)
    } else {
        compress_optimal_generic(src, params, &[1, 4, 8], 0)
    }
}

pub fn compress_btopt_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    compress_optimal_generic(combined, params, initial_rep, dict_len)
}

fn compress_optimal_generic(
    src: &[u8],
    params: &CompressionParams,
    initial_rep: &[u32; 3],
    start_pos: usize,
) -> CompressedBlock {
    let opt_level: i32 = match params.strategy {
        Strategy::BtUltra | Strategy::BtUltra2 => 2,
        _ => 0,
    };

    let mut opt_state = OptState::new(opt_level);
    opt_state.init_from_source(&src[start_pos..]);
    compress_optimal_generic_with_stats(src, params, initial_rep, start_pos, &mut opt_state, false)
}

fn compress_optimal_generic_with_stats(
    src: &[u8],
    params: &CompressionParams,
    initial_rep: &[u32; 3],
    start_pos: usize,
    opt_state: &mut OptState,
    stats_only: bool,
) -> CompressedBlock {
    let opt_level = opt_state.opt_level;

    let min_match = if params.min_match == 3 { 3 } else { 4 };
    let sufficient_len = (params.target_length as usize).min(ZSTD_OPT_NUM - 1);
    let search_depth = params.search_depth();
    let window_size = params.window_size();

    let mut bt = BinaryTree::new(params.hash_log, params.chain_log);
    let mut rep = *initial_rep;

    let mut hash3_table = if min_match == 3 {
        Some(Hash3Table::new(params.window_log))
    } else {
        None
    };

    if start_pos > 0 {
        prefill_binary_tree(&mut bt, src, start_pos, params);
    }

    if stats_only {
        // First pass: stats are freshly initialized
    } else if opt_state.lit_length_sum > 0 {
        opt_state.rescale();
    }

    let mut sequences: Vec<SequenceOut> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();

    let end = src.len().saturating_sub(8);
    let mut anchor = start_pos;
    let mut pos = start_pos;

    let mut opt: Vec<OptNode> = (0..ZSTD_OPT_SIZE).map(|_| OptNode::default()).collect();
    let mut matches_buf: Vec<MatchFound> = Vec::with_capacity(64);

    while pos < end {
        let litlen = (pos - anchor) as u32;
        let ll0 = litlen == 0;

        get_all_matches(
            &mut bt,
            src,
            pos,
            &rep,
            ll0,
            min_match,
            search_depth,
            window_size,
            sufficient_len,
            hash3_table.as_mut(),
            &mut matches_buf,
        );

        if matches_buf.is_empty() {
            pos += 1;
            continue;
        }

        opt[0].mlen = 0;
        opt[0].litlen = litlen;
        opt[0].price = opt_state.lit_length_price(litlen);
        opt[0].rep = rep;

        let max_ml = matches_buf.last().unwrap().len as usize;
        let max_off = matches_buf.last().unwrap().off_base;

        if max_ml > sufficient_len {
            if !stats_only {
                literals.extend_from_slice(&src[anchor..pos]);
                let seq = SequenceOut {
                    off_base: max_off,
                    lit_len: litlen,
                    match_len: max_ml as u32,
                };
                sequences.push(seq);
            }
            opt_state.update_stats(litlen, &src[anchor..pos], max_off, max_ml as u32);
            let mut tmp_rep = RepCodes { rep };
            tmp_rep.update(max_off, litlen);
            rep = tmp_rep.rep;
            pos += max_ml;
            anchor = pos;
            opt_state.set_base_prices();
            continue;
        }

        for p in 1..min_match {
            if p < opt.len() {
                opt[p].price = ZSTD_MAX_PRICE;
                opt[p].mlen = 0;
                opt[p].litlen = litlen + p as u32;
            }
        }

        let mut fill_pos = min_match;
        for m in &matches_buf {
            let end_pos = m.len as usize;
            while fill_pos <= end_pos && fill_pos < ZSTD_OPT_SIZE {
                let match_price = opt_state.match_price(m.off_base, fill_pos as u32);
                let seq_price = opt[0].price + match_price;
                opt[fill_pos].mlen = fill_pos as u32;
                opt[fill_pos].off = m.off_base;
                opt[fill_pos].litlen = 0;
                opt[fill_pos].price = seq_price + opt_state.lit_length_price(0);
                fill_pos += 1;
            }
        }

        let mut last_pos = fill_pos - 1;
        if last_pos + 1 < opt.len() {
            opt[last_pos + 1].price = ZSTD_MAX_PRICE;
        }

        let mut last_stretch_mlen = 0u32;
        let mut last_stretch_off = 0u32;
        let mut last_stretch_litlen = 0u32;
        let mut went_to_shortest = false;

        {
            let mut cur = 1usize;
            'forward: loop {
                if cur > last_pos || cur >= ZSTD_OPT_SIZE {
                    break;
                }

                let inr = pos + cur;
                if inr >= src.len() {
                    break;
                }

                {
                    let prev_litlen = opt[cur - 1].litlen + 1;
                    let ll_inc_price = opt_state.lit_length_price(prev_litlen)
                        - opt_state.lit_length_price(prev_litlen - 1);
                    let price = opt[cur - 1].price
                        + opt_state.single_literal_cost(src[inr - 1])
                        + ll_inc_price;

                    if price <= opt[cur].price {
                        let prev_match =
                            if opt_level >= 1 && opt[cur].litlen == 0 && opt[cur].mlen > 0 {
                                Some(opt[cur].clone())
                            } else {
                                None
                            };

                        opt[cur] = opt[cur - 1].clone();
                        opt[cur].litlen = prev_litlen;
                        opt[cur].price = price;

                        if let Some(prev_match_node) = prev_match {
                            let ll1_inc =
                                opt_state.lit_length_price(1) - opt_state.lit_length_price(0);
                            if opt_level >= 1
                                && ll1_inc < 0
                                && inr < src.len()
                                && cur + 1 < opt.len()
                            {
                                let with_1lit = prev_match_node.price
                                    + opt_state.single_literal_cost(src[inr])
                                    + ll1_inc;
                                let with_more_lits = price
                                    + opt_state.single_literal_cost(src[inr])
                                    + (opt_state.lit_length_price(prev_litlen + 1)
                                        - opt_state.lit_length_price(prev_litlen));
                                if with_1lit < with_more_lits && with_1lit < opt[cur + 1].price {
                                    let prev = cur - prev_match_node.mlen as usize;
                                    let new_reps = new_rep(
                                        &opt[prev].rep,
                                        prev_match_node.off,
                                        opt[prev].litlen == 0,
                                    );
                                    opt[cur + 1] = prev_match_node;
                                    opt[cur + 1].rep = new_reps;
                                    opt[cur + 1].litlen = 1;
                                    opt[cur + 1].price = with_1lit;
                                    if last_pos < cur + 1 {
                                        last_pos = cur + 1;
                                    }
                                }
                            }
                        }
                    }
                }

                if opt[cur].litlen == 0 && opt[cur].mlen > 0 {
                    let prev = cur - opt[cur].mlen as usize;
                    opt[cur].rep = new_rep(&opt[prev].rep, opt[cur].off, opt[prev].litlen == 0);
                }

                if inr > end {
                    cur += 1;
                    continue;
                }

                if cur == last_pos {
                    break;
                }

                if opt_level == 0
                    && cur + 1 < opt.len()
                    && opt[cur + 1].price <= opt[cur].price + (BITCOST_MULTIPLIER / 2)
                {
                    cur += 1;
                    continue;
                }

                let cur_ll0 = opt[cur].litlen == 0;
                get_all_matches(
                    &mut bt,
                    src,
                    inr,
                    &opt[cur].rep,
                    cur_ll0,
                    min_match,
                    search_depth,
                    window_size,
                    sufficient_len,
                    hash3_table.as_mut(),
                    &mut matches_buf,
                );

                if matches_buf.is_empty() {
                    cur += 1;
                    continue;
                }

                let longest_ml = matches_buf.last().unwrap().len as usize;

                if longest_ml > sufficient_len
                    || cur + longest_ml >= ZSTD_OPT_NUM
                    || inr + longest_ml >= src.len()
                {
                    last_stretch_mlen = longest_ml as u32;
                    last_stretch_off = matches_buf.last().unwrap().off_base;
                    last_stretch_litlen = 0;
                    last_pos = cur + longest_ml;
                    went_to_shortest = true;
                    break 'forward;
                }

                let base_price = opt[cur].price + opt_state.lit_length_price(0);
                for (match_nb, m) in matches_buf.iter().enumerate() {
                    let start_ml = if match_nb > 0 {
                        matches_buf[match_nb - 1].len as usize + 1
                    } else {
                        min_match
                    };
                    let end_ml = m.len as usize;

                    let mut mlen = end_ml;
                    while mlen >= start_ml {
                        let p = cur + mlen;
                        if p >= ZSTD_OPT_SIZE {
                            mlen -= 1;
                            continue;
                        }
                        let price = base_price + opt_state.match_price(m.off_base, mlen as u32);

                        if p > last_pos || price < opt[p].price {
                            while last_pos < p {
                                last_pos += 1;
                                if last_pos < opt.len() {
                                    opt[last_pos].price = ZSTD_MAX_PRICE;
                                    opt[last_pos].litlen = 1;
                                }
                            }
                            if p < opt.len() {
                                opt[p].mlen = mlen as u32;
                                opt[p].off = m.off_base;
                                opt[p].litlen = 0;
                                opt[p].price = price;
                            }
                        } else if opt_level == 0 {
                            break;
                        }
                        mlen -= 1;
                    }
                }

                if last_pos + 1 < opt.len() {
                    opt[last_pos + 1].price = ZSTD_MAX_PRICE;
                }

                cur += 1;
            }
        }

        if !went_to_shortest && last_pos < opt.len() {
            last_stretch_mlen = opt[last_pos].mlen;
            last_stretch_off = opt[last_pos].off;
            last_stretch_litlen = opt[last_pos].litlen;
        }

        if last_stretch_mlen == 0 {
            pos += last_pos;
            continue;
        }

        let mut cur = last_pos.saturating_sub(last_stretch_mlen as usize);

        if last_stretch_litlen == 0 {
            rep = new_rep(&opt[cur].rep, last_stretch_off, opt[cur].litlen == 0);
        } else {
            rep = opt[last_pos.min(opt.len() - 1)].rep;
            if cur >= last_stretch_litlen as usize {
                cur -= last_stretch_litlen as usize;
            }
        }

        let mut seq_list: Vec<(u32, u32, u32)> = Vec::new();

        seq_list.push((last_stretch_litlen, last_stretch_mlen, last_stretch_off));

        let mut stretch_pos = cur;
        loop {
            if stretch_pos >= opt.len() {
                break;
            }
            let node = &opt[stretch_pos];
            let litlen_n = node.litlen;
            let mlen = node.mlen;
            let off = node.off;

            if mlen == 0 {
                if let Some(last) = seq_list.last_mut() {
                    last.0 = litlen_n;
                }
                break;
            }

            seq_list.push((litlen_n, mlen, off));
            let advance = (litlen_n + mlen) as usize;
            if stretch_pos < advance {
                break;
            }
            stretch_pos -= advance;
        }

        seq_list.reverse();

        for &(llen, mlen, off) in &seq_list {
            if mlen == 0 {
                pos = anchor + llen as usize;
                continue;
            }

            let lit_start = anchor;
            let lit_end = anchor + llen as usize;
            if !stats_only {
                if lit_end > lit_start {
                    literals.extend_from_slice(&src[lit_start..lit_end]);
                }
                sequences.push(SequenceOut {
                    off_base: off,
                    lit_len: llen,
                    match_len: mlen,
                });
            }
            opt_state.update_stats(llen, &src[lit_start..lit_end], off, mlen);

            anchor = lit_end + mlen as usize;
            pos = anchor;
        }

        opt_state.set_base_prices();
    }

    if stats_only {
        return CompressedBlock {
            literals: Vec::new(),
            sequences: Vec::new(),
        };
    }

    if anchor > start_pos {
        build_block(&src[start_pos..], sequences, literals, anchor - start_pos)
    } else {
        build_block(&src[start_pos..], sequences, literals, 0)
    }
}

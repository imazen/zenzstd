//! Fast and DFast match-finding strategies.
//!
//! Port of C zstd's `zstd_fast.c` and `zstd_double_fast.c`.
//!
//! - **Fast** (levels 1-2): Single hash table lookup, step forward on miss.
//! - **DFast** (levels 3-4): Dual hash tables (short + long), take the longer match.

use alloc::vec;
use alloc::vec::Vec;

use super::compress_params::CompressionParams;
use super::hash::{count_match, hash8};
use super::match_state::{
    CompressedBlock, RepCodes, SequenceOut, build_block, build_block_dict, emit_match_fast,
    hash_at, insert_hashes_dense, insert_hashes_sparse, prefill_hash_table_ext,
};

// ---------------------------------------------------------------------------
// Fast strategy
// ---------------------------------------------------------------------------

/// Fast strategy using an externally-owned hash table.
///
/// Optimized with pipelining, cold-path splitting, and pre-allocated buffers
/// to match C zstd's `ZSTD_compressBlock_fast_generic` structure.
#[inline]
pub fn compress_fast_ext(
    src: &[u8],
    params: &CompressionParams,
    htable: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    let step_factor = 1usize << params.search_log;
    let window_size = params.window_size();
    let ht_mask = htable.len() - 1;

    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::with_capacity(src.len() / 32 + 4);
    let mut literals: Vec<u8> = Vec::with_capacity(src.len());

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    if end < 2 {
        return build_block(src, sequences, literals, anchor);
    }

    let mut h0 = hash_at(src, 0, hash_log, min_match);

    while pos < end {
        // Rep0 check
        {
            let lit_len = (pos - anchor) as u32;
            let rep_offset = if lit_len == 0 { rep.rep[1] } else { rep.rep[0] };
            let rep_offset_usize = rep_offset as usize;
            if rep_offset > 0 && rep_offset_usize <= pos && pos + (min_match as usize) <= src.len()
            {
                let ref_pos = pos - rep_offset_usize;
                let rml = count_match(&src[pos..], &src[ref_pos..]);
                if rml >= min_match as usize {
                    emit_match_fast(
                        src,
                        &mut literals,
                        &mut sequences,
                        &mut rep,
                        anchor,
                        pos,
                        1,
                        rml as u32,
                    );
                    pos += rml;
                    anchor = pos;
                    if pos < end {
                        h0 = hash_at(src, pos, hash_log, min_match);
                    }
                    continue;
                }
            }
        }

        let step = ((pos - anchor) / step_factor) + 1;

        let match_pos = htable[h0 & ht_mask] as usize;
        htable[h0 & ht_mask] = pos as u32;

        let next_pos = pos + step;
        let h_next = if next_pos < end {
            hash_at(src, next_pos, hash_log, min_match)
        } else {
            0
        };

        if match_pos == 0 || match_pos >= pos || (pos - match_pos) > window_size {
            pos = next_pos;
            h0 = h_next;
            continue;
        }

        let ml = count_match(&src[pos..], &src[match_pos..]);
        if ml < min_match as usize {
            pos = next_pos;
            h0 = h_next;
            continue;
        }

        let real_offset = (pos - match_pos) as u32;
        emit_match_fast(
            src,
            &mut literals,
            &mut sequences,
            &mut rep,
            anchor,
            pos,
            real_offset + 3,
            ml as u32,
        );

        let match_end = pos + ml;
        let insert_start = pos + 1;
        let insert_end = match_end.min(end);
        if ml > 32 {
            insert_hashes_sparse(htable, src, insert_start, insert_end, hash_log, min_match);
        } else {
            insert_hashes_dense(htable, src, insert_start, insert_end, hash_log, min_match);
        }

        pos = match_end;
        anchor = pos;
        if pos < end {
            h0 = hash_at(src, pos, hash_log, min_match);
        }
    }

    build_block(src, sequences, literals, anchor)
}

/// Fast strategy with dict prefix, using externally-owned hash table.
#[inline]
pub fn compress_fast_dict_ext(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    htable: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    let step_factor = 1usize << params.search_log;
    let window_size = params.window_size();
    let ht_mask = htable.len() - 1;
    let mut rep = RepCodes { rep: *initial_rep };
    let src_len = combined.len() - dict_len;
    let mut sequences: Vec<SequenceOut> = Vec::with_capacity(src_len / 32 + 4);
    let mut literals: Vec<u8> = Vec::with_capacity(src_len);
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);

    if pos >= end {
        return build_block_dict(combined, dict_len, sequences, literals, anchor);
    }

    let mut h0 = hash_at(combined, pos, hash_log, min_match);

    while pos < end {
        {
            let lit_len = (pos - anchor) as u32;
            let rep_offset = if lit_len == 0 { rep.rep[1] } else { rep.rep[0] };
            let rep_offset_usize = rep_offset as usize;
            if rep_offset > 0
                && rep_offset_usize <= pos
                && pos + (min_match as usize) <= combined.len()
            {
                let ref_pos = pos - rep_offset_usize;
                let rml = count_match(&combined[pos..], &combined[ref_pos..]);
                if rml >= min_match as usize {
                    emit_match_fast(
                        combined,
                        &mut literals,
                        &mut sequences,
                        &mut rep,
                        anchor,
                        pos,
                        1,
                        rml as u32,
                    );
                    pos += rml;
                    anchor = pos;
                    if pos < end {
                        h0 = hash_at(combined, pos, hash_log, min_match);
                    }
                    continue;
                }
            }
        }

        let step = ((pos - anchor) / step_factor) + 1;

        let mp = htable[h0 & ht_mask] as usize;
        htable[h0 & ht_mask] = pos as u32;

        let next_pos = pos + step;
        let h_next = if next_pos < end {
            hash_at(combined, next_pos, hash_log, min_match)
        } else {
            0
        };

        if mp >= pos || (pos - mp) > window_size {
            pos = next_pos;
            h0 = h_next;
            continue;
        }
        let ml = count_match(&combined[pos..], &combined[mp..]);
        if ml < min_match as usize {
            pos = next_pos;
            h0 = h_next;
            continue;
        }

        let ro = (pos - mp) as u32;
        emit_match_fast(
            combined,
            &mut literals,
            &mut sequences,
            &mut rep,
            anchor,
            pos,
            ro + 3,
            ml as u32,
        );

        let me = pos + ml;
        let insert_start = pos + 1;
        let ie = me.min(end);
        if ml > 32 {
            insert_hashes_sparse(htable, combined, insert_start, ie, hash_log, min_match);
        } else {
            insert_hashes_dense(htable, combined, insert_start, ie, hash_log, min_match);
        }
        pos = me;
        anchor = pos;
        if pos < end {
            h0 = hash_at(combined, pos, hash_log, min_match);
        }
    }
    build_block_dict(combined, dict_len, sequences, literals, anchor)
}

// ---------------------------------------------------------------------------
// DFast strategy
// ---------------------------------------------------------------------------

/// Insert dual hash table entries (short + long) for positions in `start..end`
/// with step 1.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn insert_dfast_hashes_dense(
    short_table: &mut [u32],
    long_table: &mut [u32],
    src: &[u8],
    start: usize,
    end: usize,
    hash_log: u32,
    long_hash_log: u32,
    min_match: u32,
) {
    let short_mask = short_table.len() - 1;
    let long_mask = long_table.len() - 1;
    let mut p = start;
    while p < end {
        let sh = hash_at(src, p, hash_log, min_match);
        short_table[sh & short_mask] = p as u32;
        let lh = hash8(&src[p..], long_hash_log);
        long_table[lh & long_mask] = p as u32;
        p += 1;
    }
}

/// Insert dual hash table entries (short + long) with step 4.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn insert_dfast_hashes_sparse(
    short_table: &mut [u32],
    long_table: &mut [u32],
    src: &[u8],
    start: usize,
    end: usize,
    hash_log: u32,
    long_hash_log: u32,
    min_match: u32,
) {
    let short_mask = short_table.len() - 1;
    let long_mask = long_table.len() - 1;
    let mut p = start;
    while p < end {
        let sh = hash_at(src, p, hash_log, min_match);
        short_table[sh & short_mask] = p as u32;
        let lh = hash8(&src[p..], long_hash_log);
        long_table[lh & long_mask] = p as u32;
        p += 4;
    }
}

/// DFast strategy using externally-owned hash tables.
#[inline]
pub fn compress_dfast_ext(
    src: &[u8],
    params: &CompressionParams,
    short_table: &mut [u32],
    long_table: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    let long_hash_log = params.hash_log.min(27);
    let window_size = params.window_size();
    let short_mask = short_table.len() - 1;
    let long_mask = long_table.len() - 1;

    let mut rep = RepCodes { rep: *initial_rep };
    let mut sequences: Vec<SequenceOut> = Vec::with_capacity(src.len() / 32 + 4);
    let mut literals: Vec<u8> = Vec::with_capacity(src.len());

    let mut anchor: usize = 0;
    let mut pos: usize = 0;
    let end = src.len().saturating_sub(8);

    while pos < end {
        {
            let lit_len = (pos - anchor) as u32;
            let rep_offset = if lit_len == 0 { rep.rep[1] } else { rep.rep[0] };
            let rep_offset_usize = rep_offset as usize;
            if rep_offset > 0 && rep_offset_usize <= pos && pos + (min_match as usize) <= src.len()
            {
                let ref_pos = pos - rep_offset_usize;
                let rml = count_match(&src[pos..], &src[ref_pos..]);
                if rml >= min_match as usize {
                    emit_match_fast(
                        src,
                        &mut literals,
                        &mut sequences,
                        &mut rep,
                        anchor,
                        pos,
                        1,
                        rml as u32,
                    );
                    pos += rml;
                    anchor = pos;
                    continue;
                }
            }
        }

        let short_h = hash_at(src, pos, hash_log, min_match);
        let short_prev = short_table[short_h & short_mask] as usize;
        short_table[short_h & short_mask] = pos as u32;

        let long_h = hash8(&src[pos..], long_hash_log);
        let long_prev = long_table[long_h & long_mask] as usize;
        long_table[long_h & long_mask] = pos as u32;

        let mut best: Option<super::match_state::MatchCandidate> = None;

        if long_prev > 0 && long_prev < pos && (pos - long_prev) <= window_size {
            let ml = count_match(&src[pos..], &src[long_prev..]);
            if ml >= min_match as usize {
                best = Some(super::match_state::MatchCandidate {
                    off_base: (pos - long_prev) as u32 + 3,
                    match_len: ml as u32,
                });
            }
        }

        if short_prev > 0 && short_prev < pos && (pos - short_prev) <= window_size {
            let ml = count_match(&src[pos..], &src[short_prev..]);
            if ml >= min_match as usize {
                let cand = super::match_state::MatchCandidate {
                    off_base: (pos - short_prev) as u32 + 3,
                    match_len: ml as u32,
                };
                if best.map_or(true, |b| cand.match_len > b.match_len) {
                    best = Some(cand);
                }
            }
        }

        if let Some(m) = best {
            emit_match_fast(
                src,
                &mut literals,
                &mut sequences,
                &mut rep,
                anchor,
                pos,
                m.off_base,
                m.match_len,
            );

            let ml = m.match_len as usize;
            let match_end = pos + ml;
            let insert_start = pos + 1;
            let insert_end = match_end.min(end);
            if ml > 32 {
                insert_dfast_hashes_sparse(
                    short_table,
                    long_table,
                    src,
                    insert_start,
                    insert_end,
                    hash_log,
                    long_hash_log,
                    min_match,
                );
            } else {
                insert_dfast_hashes_dense(
                    short_table,
                    long_table,
                    src,
                    insert_start,
                    insert_end,
                    hash_log,
                    long_hash_log,
                    min_match,
                );
            }
            pos = match_end;
            anchor = pos;
        } else {
            pos += 1;
        }
    }

    build_block(src, sequences, literals, anchor)
}

/// DFast strategy with dict prefix, using externally-owned hash tables.
#[inline]
pub fn compress_dfast_dict_ext(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    short_table: &mut [u32],
    long_table: &mut [u32],
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let hash_log = params.hash_log;
    let long_hash_log = hash_log.min(27);
    let window_size = params.window_size();
    let short_mask = short_table.len() - 1;
    let long_mask = long_table.len() - 1;
    let mut rep = RepCodes { rep: *initial_rep };
    let src_len = combined.len() - dict_len;
    let mut sequences: Vec<SequenceOut> = Vec::with_capacity(src_len / 32 + 4);
    let mut literals: Vec<u8> = Vec::with_capacity(src_len);
    let mut anchor = dict_len;
    let mut pos = dict_len;
    let end = combined.len().saturating_sub(8);

    while pos < end {
        {
            let lit_len = (pos - anchor) as u32;
            let rep_offset = if lit_len == 0 { rep.rep[1] } else { rep.rep[0] };
            let rep_offset_usize = rep_offset as usize;
            if rep_offset > 0
                && rep_offset_usize <= pos
                && pos + (min_match as usize) <= combined.len()
            {
                let ref_pos = pos - rep_offset_usize;
                let rml = count_match(&combined[pos..], &combined[ref_pos..]);
                if rml >= min_match as usize {
                    emit_match_fast(
                        combined,
                        &mut literals,
                        &mut sequences,
                        &mut rep,
                        anchor,
                        pos,
                        1,
                        rml as u32,
                    );
                    pos += rml;
                    anchor = pos;
                    continue;
                }
            }
        }
        let sh = hash_at(combined, pos, hash_log, min_match);
        let sp = short_table[sh & short_mask] as usize;
        short_table[sh & short_mask] = pos as u32;
        let lh = hash8(&combined[pos..], long_hash_log);
        let lp = long_table[lh & long_mask] as usize;
        long_table[lh & long_mask] = pos as u32;
        let mut best: Option<super::match_state::MatchCandidate> = None;
        if lp < pos && (pos - lp) <= window_size {
            let ml = count_match(&combined[pos..], &combined[lp..]);
            if ml >= min_match as usize {
                best = Some(super::match_state::MatchCandidate {
                    off_base: (pos - lp) as u32 + 3,
                    match_len: ml as u32,
                });
            }
        }
        if sp < pos && (pos - sp) <= window_size {
            let ml = count_match(&combined[pos..], &combined[sp..]);
            if ml >= min_match as usize {
                let c = super::match_state::MatchCandidate {
                    off_base: (pos - sp) as u32 + 3,
                    match_len: ml as u32,
                };
                if best.map_or(true, |b| c.match_len > b.match_len) {
                    best = Some(c);
                }
            }
        }
        if let Some(m) = best {
            emit_match_fast(
                combined,
                &mut literals,
                &mut sequences,
                &mut rep,
                anchor,
                pos,
                m.off_base,
                m.match_len,
            );
            let ml = m.match_len as usize;
            let me = pos + ml;
            let insert_start = pos + 1;
            let ie = me.min(end);
            if ml > 32 {
                insert_dfast_hashes_sparse(
                    short_table,
                    long_table,
                    combined,
                    insert_start,
                    ie,
                    hash_log,
                    long_hash_log,
                    min_match,
                );
            } else {
                insert_dfast_hashes_dense(
                    short_table,
                    long_table,
                    combined,
                    insert_start,
                    ie,
                    hash_log,
                    long_hash_log,
                    min_match,
                );
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
// Standalone wrappers (allocate own tables)
// ---------------------------------------------------------------------------

pub fn compress_fast(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let mut htable = vec![0u32; 1 << params.hash_log];
    compress_fast_ext(src, params, &mut htable, &[1, 4, 8])
}

pub fn compress_dfast(src: &[u8], params: &CompressionParams) -> CompressedBlock {
    let long_hash_log = params.hash_log.min(27);
    let mut short_table = vec![0u32; 1 << params.hash_log];
    let mut long_table = vec![0u32; 1 << long_hash_log];
    compress_dfast_ext(src, params, &mut short_table, &mut long_table, &[1, 4, 8])
}

pub fn compress_fast_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let mut htable = vec![0u32; 1 << params.hash_log];
    prefill_hash_table_ext(&mut htable, params.hash_log, combined, dict_len, min_match);
    compress_fast_dict_ext(combined, dict_len, params, &mut htable, initial_rep)
}

pub fn compress_dfast_dict(
    combined: &[u8],
    dict_len: usize,
    params: &CompressionParams,
    initial_rep: &[u32; 3],
) -> CompressedBlock {
    let min_match = params.min_match.max(4);
    let long_hash_log = params.hash_log.min(27);
    let mut short_table = vec![0u32; 1 << params.hash_log];
    let mut long_table = vec![0u32; 1 << long_hash_log];
    prefill_hash_table_ext(
        &mut short_table,
        params.hash_log,
        combined,
        dict_len,
        min_match,
    );
    if dict_len >= 8 {
        let end = dict_len.saturating_sub(7);
        for pos in 0..end {
            let lh = hash8(&combined[pos..], long_hash_log);
            long_table[lh] = pos as u32;
        }
    }
    compress_dfast_dict_ext(
        combined,
        dict_len,
        params,
        &mut short_table,
        &mut long_table,
        initial_rep,
    )
}

use super::super::blocks::sequence_section::ModeType;
use super::super::blocks::sequence_section::Sequence;
use super::super::blocks::sequence_section::SequencesHeader;
use super::scratch::{DecoderScratch, FSEScratch};
use crate::bit_io::BitReaderReversed;
use crate::blocks::sequence_section::{
    MAX_LITERAL_LENGTH_CODE, MAX_MATCH_LENGTH_CODE, MAX_OFFSET_CODE,
};
use crate::decoding::errors::{DecodeSequenceError, DecompressBlockError, ExecuteSequencesError};
use crate::fse::FSEDecoder;
use alloc::vec::Vec;

/// Decode the provided source as a series of sequences into the supplied `target`.
/// This is the old two-pass decode path, kept as a fallback for testing/validation.
/// The fused decode_and_execute_sequences is used by default.
#[allow(dead_code)]
pub fn decode_sequences(
    section: &SequencesHeader,
    source: &[u8],
    scratch: &mut FSEScratch,
    target: &mut Vec<Sequence>,
) -> Result<(), DecodeSequenceError> {
    let bytes_read = maybe_update_fse_tables(section, source, scratch)?;

    vprintln!("Updating tables used {} bytes", bytes_read);

    let bit_stream = &source[bytes_read..];

    let mut br = BitReaderReversed::new(bit_stream);

    //skip the 0 padding at the end of the last byte of the bit stream and throw away the first 1 found
    let mut skipped_bits = 0;
    loop {
        let val = br.get_bits(1);
        skipped_bits += 1;
        if val == 1 || skipped_bits > 8 {
            break;
        }
    }
    if skipped_bits > 8 {
        //if more than 7 bits are 0, this is not the correct end of the bitstream. Either a bug or corrupted data
        return Err(DecodeSequenceError::ExtraPadding { skipped_bits });
    }

    if scratch.ll_rle.is_some() || scratch.ml_rle.is_some() || scratch.of_rle.is_some() {
        decode_sequences_with_rle(section, &mut br, scratch, target)
    } else {
        decode_sequences_without_rle(section, &mut br, scratch, target)
    }
}

#[allow(clippy::needless_range_loop)] // indexed loop intentional: seq_idx drives last-sequence branching
fn decode_sequences_with_rle(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    scratch: &FSEScratch,
    target: &mut Vec<Sequence>,
) -> Result<(), DecodeSequenceError> {
    let mut ll_dec = FSEDecoder::new(&scratch.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&scratch.match_lengths);
    let mut of_dec = FSEDecoder::new(&scratch.offsets);

    if scratch.ll_rle.is_none() {
        ll_dec.init_state(br)?;
    }
    if scratch.of_rle.is_none() {
        of_dec.init_state(br)?;
    }
    if scratch.ml_rle.is_none() {
        ml_dec.init_state(br)?;
    }

    let num_sequences = section.num_sequences as usize;
    target.clear();
    target.resize(
        num_sequences,
        Sequence {
            ll: 0,
            ml: 0,
            of: 0,
        },
    );

    let ll_rle = scratch.ll_rle;
    let ml_rle = scratch.ml_rle;
    let of_rle = scratch.of_rle;

    for seq_idx in 0..num_sequences {
        // Get the codes from either the RLE byte or from the fused decode+params
        let (ll_code, ll_nbits) = match ll_rle {
            Some(rle) => (rle, 0u8),
            None => {
                let (sym, nbits, _baseline) = ll_dec.decode_and_params();
                (sym, nbits)
            }
        };
        let (ml_code, ml_nbits) = match ml_rle {
            Some(rle) => (rle, 0u8),
            None => {
                let (sym, nbits, _baseline) = ml_dec.decode_and_params();
                (sym, nbits)
            }
        };
        let (of_code, of_nbits) = match of_rle {
            Some(rle) => (rle, 0u8),
            None => {
                let (sym, nbits, _baseline) = of_dec.decode_and_params();
                (sym, nbits)
            }
        };

        let (ll_value, ll_extra_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_extra_bits) = lookup_ml_code(ml_code);

        if of_code > MAX_OFFSET_CODE {
            return Err(DecodeSequenceError::UnsupportedOffset {
                offset_code: of_code,
            });
        }

        let (obits, ml_add, ll_add);

        if seq_idx + 1 < num_sequences {
            let extra_sum = of_code + ml_extra_bits + ll_extra_bits;
            let state_sum = ll_nbits + ml_nbits + of_nbits;
            let total = extra_sum + state_sum;

            let (ll_state_add, ml_state_add, of_state_add);

            if total <= 56 {
                if br.bits_consumed() + total > 64 {
                    br.refill_unconditional();
                }
                obits = br.peek_and_advance(of_code);
                ml_add = br.peek_and_advance(ml_extra_bits);
                ll_add = br.peek_and_advance(ll_extra_bits);
                ll_state_add = br.peek_and_advance(ll_nbits);
                ml_state_add = br.peek_and_advance(ml_nbits);
                of_state_add = br.peek_and_advance(of_nbits);
            } else {
                (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
                (ll_state_add, ml_state_add, of_state_add) =
                    br.get_bits_triple(ll_nbits, ml_nbits, of_nbits);
            }

            if ll_rle.is_none() {
                ll_dec.apply_state_update(ll_state_add);
            }
            if ml_rle.is_none() {
                ml_dec.apply_state_update(ml_state_add);
            }
            if of_rle.is_none() {
                of_dec.apply_state_update(of_state_add);
            }
        } else {
            (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
        }

        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err(DecodeSequenceError::ZeroOffset);
        }

        target[seq_idx] = Sequence {
            ll: ll_value + ll_add as u32,
            ml: ml_value + ml_add as u32,
            of: offset,
        };

        if br.bits_remaining() < 0 {
            return Err(DecodeSequenceError::NotEnoughBytesForNumSequences);
        }
    }

    if br.bits_remaining() > 0 {
        Err(DecodeSequenceError::ExtraBits {
            bits_remaining: br.bits_remaining(),
        })
    } else {
        Ok(())
    }
}

fn decode_sequences_without_rle(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    scratch: &FSEScratch,
    target: &mut Vec<Sequence>,
) -> Result<(), DecodeSequenceError> {
    let mut ll_dec = FSEDecoder::new(&scratch.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&scratch.match_lengths);
    let mut of_dec = FSEDecoder::new(&scratch.offsets);

    ll_dec.init_state(br)?;
    of_dec.init_state(br)?;
    ml_dec.init_state(br)?;

    let num_sequences = section.num_sequences as usize;
    target.clear();
    // Pre-size the vec so we can write directly by index instead of pushing.
    // This eliminates the capacity check on every iteration.
    target.resize(
        num_sequences,
        Sequence {
            ll: 0,
            ml: 0,
            of: 0,
        },
    );

    decode_sequences_fast(br, &mut ll_dec, &mut ml_dec, &mut of_dec, target)
}

/// Core decode loop with single-refill-per-sequence optimization.
///
/// Each sequence needs at most ~45 bits (3 extra-bit reads + 3 state updates).
/// With a 64-bit container and max 8 bits consumed before a refill, we can do
/// one refill at the top of the loop and then extract all 6 values without any
/// intermediate refill checks.
///
/// The loop also fuses symbol decoding with state-update parameter extraction
/// using `decode_and_params()` to avoid redundant Entry reads.
///
/// The target slice is pre-sized by the caller; we write by index to avoid
/// per-iteration capacity checks from Vec::push.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // indexed loop intentional: seq_idx drives last-sequence branching
fn decode_sequences_fast(
    br: &mut BitReaderReversed<'_>,
    ll_dec: &mut FSEDecoder<'_>,
    ml_dec: &mut FSEDecoder<'_>,
    of_dec: &mut FSEDecoder<'_>,
    target: &mut [Sequence],
) -> Result<(), DecodeSequenceError> {
    let num_sequences = target.len();
    for seq_idx in 0..num_sequences {
        // Fuse decode_symbol + state_update_params into one Entry read each.
        let (of_code, of_nbits, _of_baseline) = of_dec.decode_and_params();
        let (ml_code, ml_nbits, _ml_baseline) = ml_dec.decode_and_params();
        let (ll_code, ll_nbits, _ll_baseline) = ll_dec.decode_and_params();

        let (ll_value, ll_extra_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_extra_bits) = lookup_ml_code(ml_code);

        if of_code > MAX_OFFSET_CODE {
            return Err(DecodeSequenceError::UnsupportedOffset {
                offset_code: of_code,
            });
        }

        // Total bits needed: of_code + ml_extra_bits + ll_extra_bits for values,
        // plus ll_nbits + ml_nbits + of_nbits for state updates.
        // Max per zstd spec: of_code<=31, ml_extra<=16, ll_extra<=16, states<=9+9+8=26
        // Worst case total: 31+16+16+26 = 89 bits > 56.
        // So we need up to two refills: one for extra bits, one for state updates.
        //
        // Strategy: use get_bits_triple for the extra bits (which handles its own
        // refill), then use get_bits_triple for the state updates.
        // But we can often do it in a single refill if the total is <= 56.
        let extra_sum = of_code + ml_extra_bits + ll_extra_bits;
        let state_sum = ll_nbits + ml_nbits + of_nbits;

        let (obits, ml_add, ll_add, ll_state_add, ml_state_add, of_state_add);

        if seq_idx + 1 < num_sequences {
            // Common case: both extra bits and state update bits fit in 56 total
            let total = extra_sum + state_sum;
            if total <= 56 {
                // Single refill covers everything
                if br.bits_consumed() + total > 64 {
                    br.refill_unconditional();
                }
                // Extract all values without any further refill checks
                obits = br.peek_and_advance(of_code);
                ml_add = br.peek_and_advance(ml_extra_bits);
                ll_add = br.peek_and_advance(ll_extra_bits);
                ll_state_add = br.peek_and_advance(ll_nbits);
                ml_state_add = br.peek_and_advance(ml_nbits);
                of_state_add = br.peek_and_advance(of_nbits);
            } else {
                // Need two refills: one for extra bits, one for state updates
                (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
                (ll_state_add, ml_state_add, of_state_add) =
                    br.get_bits_triple(ll_nbits, ml_nbits, of_nbits);
            }

            ll_dec.apply_state_update(ll_state_add);
            ml_dec.apply_state_update(ml_state_add);
            of_dec.apply_state_update(of_state_add);
        } else {
            // Last sequence: no state update needed
            (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
        }

        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err(DecodeSequenceError::ZeroOffset);
        }

        target[seq_idx] = Sequence {
            ll: ll_value + ll_add as u32,
            ml: ml_value + ml_add as u32,
            of: offset,
        };

        if br.bits_remaining() < 0 {
            return Err(DecodeSequenceError::NotEnoughBytesForNumSequences);
        }
    }

    if br.bits_remaining() > 0 {
        Err(DecodeSequenceError::ExtraBits {
            bits_remaining: br.bits_remaining(),
        })
    } else {
        Ok(())
    }
}

/// Look up the provided state value from a literal length table predefined
/// by the Zstandard reference document. Returns a tuple of (value, number of bits).
///
/// <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#appendix-a---decoding-tables-for-predefined-codes>
/// Table-driven literal length lookup. Each entry is (baseline_value, extra_bits).
/// Indexed by literal length code (0..=35).
static LL_CODE_TABLE: [(u32, u8); 36] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (15, 0),
    (16, 1),
    (18, 1),
    (20, 1),
    (22, 1),
    (24, 2),
    (28, 2),
    (32, 3),
    (40, 3),
    (48, 4),
    (64, 6),
    (128, 7),
    (256, 8),
    (512, 9),
    (1024, 10),
    (2048, 11),
    (4096, 12),
    (8192, 13),
    (16384, 14),
    (32768, 15),
    (65536, 16),
];

#[inline(always)]
fn lookup_ll_code(code: u8) -> (u32, u8) {
    LL_CODE_TABLE[code as usize]
}

/// Look up the provided state value from a match length table predefined
/// by the Zstandard reference document. Returns a tuple of (value, number of bits).
///
/// <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#appendix-a---decoding-tables-for-predefined-codes>
/// Table-driven match length lookup. Each entry is (baseline_value, extra_bits).
/// Indexed by match length code (0..=52).
static ML_CODE_TABLE: [(u32, u8); 53] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (15, 0),
    (16, 0),
    (17, 0),
    (18, 0),
    (19, 0),
    (20, 0),
    (21, 0),
    (22, 0),
    (23, 0),
    (24, 0),
    (25, 0),
    (26, 0),
    (27, 0),
    (28, 0),
    (29, 0),
    (30, 0),
    (31, 0),
    (32, 0),
    (33, 0),
    (34, 0),
    (35, 1),
    (37, 1),
    (39, 1),
    (41, 1),
    (43, 2),
    (47, 2),
    (51, 3),
    (59, 3),
    (67, 4),
    (83, 4),
    (99, 5),
    (131, 7),
    (259, 8),
    (515, 9),
    (1027, 10),
    (2051, 11),
    (4099, 12),
    (8195, 13),
    (16387, 14),
    (32771, 15),
    (65539, 16),
];

#[inline(always)]
fn lookup_ml_code(code: u8) -> (u32, u8) {
    ML_CODE_TABLE[code as usize]
}

// This info is buried in the symbol compression mode table
/// "The maximum allowed accuracy log for literals length and match length tables is 9"
pub const LL_MAX_LOG: u8 = 9;
/// "The maximum allowed accuracy log for literals length and match length tables is 9"
pub const ML_MAX_LOG: u8 = 9;
/// "The maximum accuracy log for the offset table is 8."
pub const OF_MAX_LOG: u8 = 8;

fn maybe_update_fse_tables(
    section: &SequencesHeader,
    source: &[u8],
    scratch: &mut FSEScratch,
) -> Result<usize, DecodeSequenceError> {
    let modes = section
        .modes
        .ok_or(DecodeSequenceError::MissingCompressionMode)?;

    let mut bytes_read = 0;

    match modes.ll_mode() {
        ModeType::FSECompressed => {
            let bytes = scratch.literal_lengths.build_decoder(source, LL_MAX_LOG)?;
            bytes_read += bytes;

            vprintln!("Updating ll table");
            vprintln!("Used bytes: {}", bytes);
            scratch.ll_rle = None;
        }
        ModeType::RLE => {
            vprintln!("Use RLE ll table");
            if source.is_empty() {
                return Err(DecodeSequenceError::MissingByteForRleLlTable);
            }
            bytes_read += 1;
            if source[0] > MAX_LITERAL_LENGTH_CODE {
                return Err(DecodeSequenceError::MissingByteForRleMlTable);
            }
            scratch.ll_rle = Some(source[0]);
        }
        ModeType::Predefined => {
            vprintln!("Use predefined ll table");
            scratch
                .build_predefined_ll(LL_DEFAULT_ACC_LOG, &LITERALS_LENGTH_DEFAULT_DISTRIBUTION)?;
            scratch.ll_rle = None;
        }
        ModeType::Repeat => {
            vprintln!("Repeat ll table");
            /* Nothing to do */
        }
    };

    let of_source = &source[bytes_read..];

    match modes.of_mode() {
        ModeType::FSECompressed => {
            let bytes = scratch.offsets.build_decoder(of_source, OF_MAX_LOG)?;
            vprintln!("Updating of table");
            vprintln!("Used bytes: {}", bytes);
            bytes_read += bytes;
            scratch.of_rle = None;
        }
        ModeType::RLE => {
            vprintln!("Use RLE of table");
            if of_source.is_empty() {
                return Err(DecodeSequenceError::MissingByteForRleOfTable);
            }
            bytes_read += 1;
            if of_source[0] > MAX_OFFSET_CODE {
                return Err(DecodeSequenceError::MissingByteForRleMlTable);
            }
            scratch.of_rle = Some(of_source[0]);
        }
        ModeType::Predefined => {
            vprintln!("Use predefined of table");
            scratch.build_predefined_of(OF_DEFAULT_ACC_LOG, &OFFSET_DEFAULT_DISTRIBUTION)?;
            scratch.of_rle = None;
        }
        ModeType::Repeat => {
            vprintln!("Repeat of table");
            /* Nothing to do */
        }
    };

    let ml_source = &source[bytes_read..];

    match modes.ml_mode() {
        ModeType::FSECompressed => {
            let bytes = scratch.match_lengths.build_decoder(ml_source, ML_MAX_LOG)?;
            bytes_read += bytes;
            vprintln!("Updating ml table");
            vprintln!("Used bytes: {}", bytes);
            scratch.ml_rle = None;
        }
        ModeType::RLE => {
            vprintln!("Use RLE ml table");
            if ml_source.is_empty() {
                return Err(DecodeSequenceError::MissingByteForRleMlTable);
            }
            bytes_read += 1;
            if ml_source[0] > MAX_MATCH_LENGTH_CODE {
                return Err(DecodeSequenceError::MissingByteForRleMlTable);
            }
            scratch.ml_rle = Some(ml_source[0]);
        }
        ModeType::Predefined => {
            vprintln!("Use predefined ml table");
            scratch.build_predefined_ml(ML_DEFAULT_ACC_LOG, &MATCH_LENGTH_DEFAULT_DISTRIBUTION)?;
            scratch.ml_rle = None;
        }
        ModeType::Repeat => {
            vprintln!("Repeat ml table");
            /* Nothing to do */
        }
    };

    Ok(bytes_read)
}

// The default Literal Length decoding table uses an accuracy logarithm of 6 bits.
const LL_DEFAULT_ACC_LOG: u8 = 6;
/// If [ModeType::Predefined] is selected for a symbol type, its FSE decoding
/// table is generated using a predefined distribution table.
///
/// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#literals-length
const LITERALS_LENGTH_DEFAULT_DISTRIBUTION: [i32; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];

// The default Match Length decoding table uses an accuracy logarithm of 6 bits.
const ML_DEFAULT_ACC_LOG: u8 = 6;
/// If [ModeType::Predefined] is selected for a symbol type, its FSE decoding
/// table is generated using a predefined distribution table.
///
/// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#match-length
const MATCH_LENGTH_DEFAULT_DISTRIBUTION: [i32; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

// The default Match Length decoding table uses an accuracy logarithm of 5 bits.
const OF_DEFAULT_ACC_LOG: u8 = 5;
/// If [ModeType::Predefined] is selected for a symbol type, its FSE decoding
/// table is generated using a predefined distribution table.
///
/// https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#match-length
const OFFSET_DEFAULT_DISTRIBUTION: [i32; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

/// Fused decode+execute: decode each sequence from the bitstream and immediately
/// execute it (copy literals + match) into the decode buffer. Eliminates the
/// intermediate Vec<Sequence> allocation and improves cache locality since each
/// sequence's data is hot when executed.
///
/// This replaces the two-pass approach of decode_sequences() + execute_sequences().
///
/// `seq_data_offset` and `seq_data_len` specify the byte range within
/// `scratch.block_content_buffer` that contains the sequence section data
/// (FSE table descriptions + bitstream).
pub fn decode_and_execute_sequences(
    section: &SequencesHeader,
    scratch: &mut DecoderScratch,
    seq_data_offset: usize,
    seq_data_len: usize,
) -> Result<(), DecompressBlockError> {
    // Phase 1: Update FSE tables. Only needs &mut scratch.fse and the source slice.
    let bytes_read = {
        let source = &scratch.block_content_buffer[seq_data_offset..seq_data_offset + seq_data_len];
        maybe_update_fse_tables(section, source, &mut scratch.fse)?
    };

    // Phase 2: Destructure scratch to get non-overlapping borrows.
    // BitReaderReversed borrows block_content_buffer (immutable),
    // while we need mutable access to buffer and offset_hist.
    let DecoderScratch {
        fse,
        buffer,
        offset_hist,
        literals_buffer,
        block_content_buffer,
        ..
    } = scratch;

    let bit_stream_offset = seq_data_offset + bytes_read;
    let bit_stream_end = seq_data_offset + seq_data_len;
    let bit_stream = &block_content_buffer[bit_stream_offset..bit_stream_end];
    let mut br = BitReaderReversed::new(bit_stream);

    // Skip the 0 padding at the end of the last byte of the bit stream
    let mut skipped_bits = 0;
    loop {
        let val = br.get_bits(1);
        skipped_bits += 1;
        if val == 1 || skipped_bits > 8 {
            break;
        }
    }
    if skipped_bits > 8 {
        return Err(DecodeSequenceError::ExtraPadding { skipped_bits }.into());
    }

    if fse.ll_rle.is_some() || fse.ml_rle.is_some() || fse.of_rle.is_some() {
        fused_decode_execute_rle_inner(section, &mut br, fse, buffer, offset_hist, literals_buffer)
    } else {
        fused_decode_execute_fast_inner(section, &mut br, fse, buffer, offset_hist, literals_buffer)
    }
}

/// Fused decode+execute for the common case (no RLE modes).
///
/// Hot state (pos, total_output_counter, offset history) is hoisted to stack locals
/// to eliminate indirect stores through DecodeBuffer -> FlatBuffer on every sequence.
/// The profile shows 44% of this function's time is a single store instruction writing
/// `FlatBuffer::pos` through the struct indirection — hoisting it to a register
/// eliminates that L1 cache miss.
///
/// The struct fields are written back once at the end (or on error/dict-path exit).
#[inline(always)]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn fused_decode_execute_fast_inner(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    fse: &FSEScratch,
    buffer: &mut super::decode_buffer::DecodeBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    let mut ll_dec = FSEDecoder::new(&fse.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&fse.match_lengths);
    let mut of_dec = FSEDecoder::new(&fse.offsets);

    ll_dec.init_state(br)?;
    of_dec.init_state(br)?;
    ml_dec.init_state(br)?;

    let num_sequences = section.num_sequences as usize;
    let literals_len = literals_buffer.len();
    let mut literals_copy_counter = 0;

    // Reserve the max possible block output (128KB). This single reserve call
    // lets us use the no-reserve push/repeat variants in the hot loop,
    // eliminating per-call capacity checks.
    buffer.reserve(crate::common::MAX_BLOCK_SIZE as usize);

    // --- Hoist hot state to stack locals ---
    // These live in registers instead of being stored back to FlatBuffer.pos
    // (through DecodeBuffer -> FlatBuffer indirection) on every push/repeat.
    let drain_pos = buffer.buffer.drain_pos;
    let mut pos = buffer.buffer.pos;
    let mut total_out = buffer.total_output_counter;

    // Hoist offset history to scalar locals — avoids array indexing overhead
    let mut off1 = offset_hist[0];
    let mut off2 = offset_hist[1];
    let mut off3 = offset_hist[2];

    for seq_idx in 0..num_sequences {
        let (of_code, of_nbits, _of_baseline) = of_dec.decode_and_params();
        let (ml_code, ml_nbits, _ml_baseline) = ml_dec.decode_and_params();
        let (ll_code, ll_nbits, _ll_baseline) = ll_dec.decode_and_params();

        let (ll_value, ll_extra_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_extra_bits) = lookup_ml_code(ml_code);

        if of_code > MAX_OFFSET_CODE {
            buffer.buffer.pos = pos;
            buffer.total_output_counter = total_out;
            offset_hist[0] = off1;
            offset_hist[1] = off2;
            offset_hist[2] = off3;
            return Err(DecodeSequenceError::UnsupportedOffset {
                offset_code: of_code,
            }
            .into());
        }

        let extra_sum = of_code + ml_extra_bits + ll_extra_bits;
        let state_sum = ll_nbits + ml_nbits + of_nbits;

        let (obits, ml_add, ll_add, ll_state_add, ml_state_add, of_state_add);

        if seq_idx + 1 < num_sequences {
            let total = extra_sum + state_sum;
            if total <= 56 {
                if br.bits_consumed() + total > 64 {
                    br.refill_unconditional();
                }
                obits = br.peek_and_advance(of_code);
                ml_add = br.peek_and_advance(ml_extra_bits);
                ll_add = br.peek_and_advance(ll_extra_bits);
                ll_state_add = br.peek_and_advance(ll_nbits);
                ml_state_add = br.peek_and_advance(ml_nbits);
                of_state_add = br.peek_and_advance(of_nbits);
            } else {
                (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
                (ll_state_add, ml_state_add, of_state_add) =
                    br.get_bits_triple(ll_nbits, ml_nbits, of_nbits);
            }

            ll_dec.apply_state_update(ll_state_add);
            ml_dec.apply_state_update(ml_state_add);
            of_dec.apply_state_update(of_state_add);
        } else {
            (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
        }

        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            buffer.buffer.pos = pos;
            buffer.total_output_counter = total_out;
            offset_hist[0] = off1;
            offset_hist[1] = off2;
            offset_hist[2] = off3;
            return Err(DecodeSequenceError::ZeroOffset.into());
        }

        if br.bits_remaining() < 0 {
            buffer.buffer.pos = pos;
            buffer.total_output_counter = total_out;
            offset_hist[0] = off1;
            offset_hist[1] = off2;
            offset_hist[2] = off3;
            return Err(DecodeSequenceError::NotEnoughBytesForNumSequences.into());
        }

        // --- Execute this sequence inline ---
        let ll = (ll_value + ll_add as u32) as usize;
        let ml = (ml_value + ml_add as u32) as usize;

        // Copy literals — direct buf write, no method call overhead.
        // Access buffer.buffer.buf directly to get &mut Vec<u8>.
        if ll > 0 {
            let high = literals_copy_counter + ll;
            if high > literals_len {
                buffer.buffer.pos = pos;
                buffer.total_output_counter = total_out;
                offset_hist[0] = off1;
                offset_hist[1] = off2;
                offset_hist[2] = off3;
                return Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                    wanted: high,
                    have: literals_len,
                }
                .into());
            }
            buffer.buffer.buf[pos..pos + ll]
                .copy_from_slice(&literals_buffer[literals_copy_counter..high]);
            pos += ll;
            total_out += ll as u64;
            literals_copy_counter += ll;
        }

        // Resolve offset and update history — fully inlined with stack locals
        let actual_offset =
            do_offset_history_hoisted(offset, ll as u32, &mut off1, &mut off2, &mut off3);
        if actual_offset == 0 {
            buffer.buffer.pos = pos;
            buffer.total_output_counter = total_out;
            offset_hist[0] = off1;
            offset_hist[1] = off2;
            offset_hist[2] = off3;
            return Err(ExecuteSequencesError::ZeroOffset.into());
        }

        // Copy match — direct buf operations, no method call overhead
        if ml > 0 {
            let buf_len = pos - drain_pos; // logical length of buffer content
            let actual_off = actual_offset as usize;

            if actual_off > buf_len {
                // Cold path: match references dictionary content.
                // Write back state, delegate to DecodeBuffer method, re-hoist.
                buffer.buffer.pos = pos;
                buffer.total_output_counter = total_out;
                offset_hist[0] = off1;
                offset_hist[1] = off2;
                offset_hist[2] = off3;
                buffer.repeat_no_reserve(actual_off, ml)?;
                pos = buffer.buffer.pos;
                total_out = buffer.total_output_counter;
            } else {
                // Hot path: match is within the buffer — direct copy
                let src_abs = drain_pos + (buf_len - actual_off);
                let distance = pos - src_abs;

                if distance >= ml {
                    // Non-overlapping: single copy_within
                    buffer.buffer.buf.copy_within(src_abs..src_abs + ml, pos);
                } else if distance == 1 {
                    // RLE: fill with repeated byte
                    let byte = buffer.buffer.buf[src_abs];
                    buffer.buffer.buf[pos..pos + ml].fill(byte);
                } else {
                    // Overlapping: doubling copy
                    buffer
                        .buffer
                        .buf
                        .copy_within(src_abs..src_abs + distance, pos);
                    let mut written = distance;
                    while written < ml {
                        let copy_len = written.min(ml - written);
                        buffer
                            .buffer
                            .buf
                            .copy_within(pos..pos + copy_len, pos + written);
                        written += copy_len;
                    }
                }
                pos += ml;
                total_out += ml as u64;
            }
        }
    }

    // Trailing literals — direct buf write
    if literals_copy_counter < literals_len {
        let remaining = literals_len - literals_copy_counter;
        buffer.buffer.buf[pos..pos + remaining]
            .copy_from_slice(&literals_buffer[literals_copy_counter..]);
        pos += remaining;
        total_out += remaining as u64;
    }

    // --- Write back all hoisted state ---
    buffer.buffer.pos = pos;
    buffer.total_output_counter = total_out;
    offset_hist[0] = off1;
    offset_hist[1] = off2;
    offset_hist[2] = off3;

    if br.bits_remaining() > 0 {
        Err(DecodeSequenceError::ExtraBits {
            bits_remaining: br.bits_remaining(),
        }
        .into())
    } else {
        Ok(())
    }
}

/// Fused decode+execute with RLE mode support.
/// Takes destructured scratch fields to avoid borrow conflicts.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn fused_decode_execute_rle_inner(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    fse: &FSEScratch,
    buffer: &mut super::decode_buffer::DecodeBuffer,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
) -> Result<(), DecompressBlockError> {
    let mut ll_dec = FSEDecoder::new(&fse.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&fse.match_lengths);
    let mut of_dec = FSEDecoder::new(&fse.offsets);

    if fse.ll_rle.is_none() {
        ll_dec.init_state(br)?;
    }
    if fse.of_rle.is_none() {
        of_dec.init_state(br)?;
    }
    if fse.ml_rle.is_none() {
        ml_dec.init_state(br)?;
    }

    let num_sequences = section.num_sequences as usize;
    let literals_len = literals_buffer.len();
    let mut literals_copy_counter = 0;

    let ll_rle = fse.ll_rle;
    let ml_rle = fse.ml_rle;
    let of_rle = fse.of_rle;

    buffer.reserve(crate::common::MAX_BLOCK_SIZE as usize);

    for seq_idx in 0..num_sequences {
        let (ll_code, ll_nbits) = match ll_rle {
            Some(rle) => (rle, 0u8),
            None => {
                let (sym, nbits, _baseline) = ll_dec.decode_and_params();
                (sym, nbits)
            }
        };
        let (ml_code, ml_nbits) = match ml_rle {
            Some(rle) => (rle, 0u8),
            None => {
                let (sym, nbits, _baseline) = ml_dec.decode_and_params();
                (sym, nbits)
            }
        };
        let (of_code, of_nbits) = match of_rle {
            Some(rle) => (rle, 0u8),
            None => {
                let (sym, nbits, _baseline) = of_dec.decode_and_params();
                (sym, nbits)
            }
        };

        let (ll_value, ll_extra_bits) = lookup_ll_code(ll_code);
        let (ml_value, ml_extra_bits) = lookup_ml_code(ml_code);

        if of_code > MAX_OFFSET_CODE {
            return Err(DecodeSequenceError::UnsupportedOffset {
                offset_code: of_code,
            }
            .into());
        }

        let (obits, ml_add, ll_add);

        if seq_idx + 1 < num_sequences {
            let extra_sum = of_code + ml_extra_bits + ll_extra_bits;
            let state_sum = ll_nbits + ml_nbits + of_nbits;
            let total = extra_sum + state_sum;

            let (ll_state_add, ml_state_add, of_state_add);

            if total <= 56 {
                if br.bits_consumed() + total > 64 {
                    br.refill_unconditional();
                }
                obits = br.peek_and_advance(of_code);
                ml_add = br.peek_and_advance(ml_extra_bits);
                ll_add = br.peek_and_advance(ll_extra_bits);
                ll_state_add = br.peek_and_advance(ll_nbits);
                ml_state_add = br.peek_and_advance(ml_nbits);
                of_state_add = br.peek_and_advance(of_nbits);
            } else {
                (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
                (ll_state_add, ml_state_add, of_state_add) =
                    br.get_bits_triple(ll_nbits, ml_nbits, of_nbits);
            }

            if ll_rle.is_none() {
                ll_dec.apply_state_update(ll_state_add);
            }
            if ml_rle.is_none() {
                ml_dec.apply_state_update(ml_state_add);
            }
            if of_rle.is_none() {
                of_dec.apply_state_update(of_state_add);
            }
        } else {
            (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_extra_bits, ll_extra_bits);
        }

        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err(DecodeSequenceError::ZeroOffset.into());
        }

        if br.bits_remaining() < 0 {
            return Err(DecodeSequenceError::NotEnoughBytesForNumSequences.into());
        }

        // --- Execute this sequence inline ---
        let ll = (ll_value + ll_add as u32) as usize;
        let ml = (ml_value + ml_add as u32) as usize;

        if ll > 0 {
            let high = literals_copy_counter + ll;
            if high > literals_len {
                return Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                    wanted: high,
                    have: literals_len,
                }
                .into());
            }
            let literals = &literals_buffer[literals_copy_counter..high];
            buffer.push_no_reserve(literals);
            literals_copy_counter += ll;
        }

        let actual_offset = do_offset_history_inline(offset, ll as u32, offset_hist);
        if actual_offset == 0 {
            return Err(ExecuteSequencesError::ZeroOffset.into());
        }

        if ml > 0 {
            buffer.repeat_no_reserve(actual_offset as usize, ml)?;
        }
    }

    if literals_copy_counter < literals_len {
        let rest_literals = &literals_buffer[literals_copy_counter..];
        buffer.push_no_reserve(rest_literals);
    }

    if br.bits_remaining() > 0 {
        Err(DecodeSequenceError::ExtraBits {
            bits_remaining: br.bits_remaining(),
        }
        .into())
    } else {
        Ok(())
    }
}

/// Offset history resolution, inlined for the fused path.
/// Most common case (offset > 3) is first for branch predictor.
#[inline(always)]
fn do_offset_history_inline(offset_value: u32, lit_len: u32, hist: &mut [u32; 3]) -> u32 {
    if offset_value > 3 {
        let actual_offset = offset_value - 3;
        hist[2] = hist[1];
        hist[1] = hist[0];
        hist[0] = actual_offset;
        return actual_offset;
    }

    let actual_offset;
    if lit_len > 0 {
        actual_offset = hist[offset_value as usize - 1];
        match offset_value {
            1 => {}
            2 => {
                hist[1] = hist[0];
                hist[0] = actual_offset;
            }
            _ => {
                hist[2] = hist[1];
                hist[1] = hist[0];
                hist[0] = actual_offset;
            }
        }
    } else {
        actual_offset = match offset_value {
            1 => hist[1],
            2 => hist[2],
            _ => hist[0].wrapping_sub(1),
        };
        match offset_value {
            1 => {
                hist[1] = hist[0];
                hist[0] = actual_offset;
            }
            _ => {
                hist[2] = hist[1];
                hist[1] = hist[0];
                hist[0] = actual_offset;
            }
        }
    }

    actual_offset
}

/// Offset history resolution using hoisted stack locals instead of array indexing.
/// Avoids array bounds checks and keeps all three offsets in registers.
#[inline(always)]
/// Resolve the actual byte offset from the zstd offset code and update history.
///
/// Uses an array-based approach to minimize branching. The offset history
/// is stored as [off1, off2, off3] and the lookup/rotation is done via
/// indexed access rather than if/match chains.
#[inline(always)]
fn do_offset_history_hoisted(
    offset_value: u32,
    lit_len: u32,
    off1: &mut u32,
    off2: &mut u32,
    off3: &mut u32,
) -> u32 {
    // Common case: real offset (not a repcode). ~70% of sequences.
    if offset_value > 3 {
        let actual_offset = offset_value - 3;
        *off3 = *off2;
        *off2 = *off1;
        *off1 = actual_offset;
        return actual_offset;
    }

    // Repcode case: build a temp array so we can index instead of branch.
    // This helps the branch predictor by reducing the number of
    // unpredictable branches from 4-6 to 1 (the lit_len check).
    let offsets = [*off1, *off2, *off3];
    let actual_offset;

    if lit_len > 0 {
        // With literals: offset_value 1/2/3 maps to off1/off2/off3
        actual_offset = offsets[(offset_value - 1) as usize];
    } else {
        // Without literals: shifted — 1→off2, 2→off3, 3→off1-1
        actual_offset = if offset_value == 3 {
            off1.wrapping_sub(1)
        } else {
            offsets[offset_value as usize] // 1→offsets[1]=off2, 2→offsets[2]=off3
        };
    }

    // Rotate: push actual_offset to front
    if offset_value == 1 && lit_len > 0 {
        // No rotation needed — off1 stays the same
    } else {
        // Shift down and insert at front
        if offset_value >= 3 || (offset_value >= 2 && lit_len == 0) {
            *off3 = *off2;
        }
        *off2 = *off1;
        *off1 = actual_offset;
    }

    actual_offset
}

#[test]
fn test_ll_default() {
    let mut table = crate::fse::FSETable::new(MAX_LITERAL_LENGTH_CODE);
    table
        .build_from_probabilities(
            LL_DEFAULT_ACC_LOG,
            &Vec::from(&LITERALS_LENGTH_DEFAULT_DISTRIBUTION[..]),
        )
        .unwrap();

    #[cfg(feature = "std")]
    for idx in 0..table.decode.len() {
        std::println!(
            "{:3}: {:3} {:3} {:3}",
            idx,
            table.decode[idx].symbol,
            table.decode[idx].num_bits,
            table.decode[idx].base_line
        );
    }

    assert!(table.decode.len() == 64);

    //just test a few values. TODO test all values
    assert!(table.decode[0].symbol == 0);
    assert!(table.decode[0].num_bits == 4);
    assert!(table.decode[0].base_line == 0);

    assert!(table.decode[19].symbol == 27);
    assert!(table.decode[19].num_bits == 6);
    assert!(table.decode[19].base_line == 0);

    assert!(table.decode[39].symbol == 25);
    assert!(table.decode[39].num_bits == 4);
    assert!(table.decode[39].base_line == 16);

    assert!(table.decode[60].symbol == 35);
    assert!(table.decode[60].num_bits == 6);
    assert!(table.decode[60].base_line == 0);

    assert!(table.decode[59].symbol == 24);
    assert!(table.decode[59].num_bits == 5);
    assert!(table.decode[59].base_line == 32);
}

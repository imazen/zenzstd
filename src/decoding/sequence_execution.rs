use super::scratch::DecoderScratch;
use crate::decoding::errors::ExecuteSequencesError;

/// Take the provided decoder and execute the sequences stored within.
/// This is the old two-pass execution path, kept as a fallback for testing/validation.
/// The fused decode+execute path in sequence_section_decoder.rs is used by default.
#[allow(dead_code)]
#[inline(always)]
pub fn execute_sequences(scratch: &mut DecoderScratch) -> Result<(), ExecuteSequencesError> {
    let mut literals_copy_counter = 0;
    let sequences = &scratch.sequences;
    let literals_buf = &scratch.literals_buffer;
    let literals_len = literals_buf.len();

    // Pre-reserve the total output size so per-operation reserve() calls are no-ops.
    // Total output = sum of all (ll + ml) + any trailing literals.
    // Single pass to compute total.
    {
        let mut total_ll: u64 = 0;
        let mut total_ml: u64 = 0;
        for seq in sequences.iter() {
            total_ll += u64::from(seq.ll);
            total_ml += u64::from(seq.ml);
        }
        let trailing = (literals_len as u64).saturating_sub(total_ll);
        scratch
            .buffer
            .reserve((total_ll + total_ml + trailing) as usize);
    }

    #[cfg(debug_assertions)]
    let old_buffer_size = scratch.buffer.len();
    #[cfg(debug_assertions)]
    let mut seq_sum: u32 = 0;

    for seq in sequences {
        let seq = *seq;

        if seq.ll > 0 {
            let high = literals_copy_counter + seq.ll as usize;
            if high > literals_len {
                return Err(ExecuteSequencesError::NotEnoughBytesForSequence {
                    wanted: high,
                    have: literals_len,
                });
            }
            let literals = &literals_buf[literals_copy_counter..high];
            literals_copy_counter += seq.ll as usize;

            scratch.buffer.push_no_reserve(literals);
        }

        let actual_offset = do_offset_history(seq.of, seq.ll, &mut scratch.offset_hist);
        if actual_offset == 0 {
            return Err(ExecuteSequencesError::ZeroOffset);
        }
        if seq.ml > 0 {
            scratch
                .buffer
                .repeat_no_reserve(actual_offset as usize, seq.ml as usize)?;
        }

        #[cfg(debug_assertions)]
        {
            seq_sum += seq.ml;
            seq_sum += seq.ll;
        }
    }
    if literals_copy_counter < literals_len {
        let rest_literals = &literals_buf[literals_copy_counter..];
        scratch.buffer.push_no_reserve(rest_literals);
        #[cfg(debug_assertions)]
        {
            seq_sum += rest_literals.len() as u32;
        }
    }

    #[cfg(debug_assertions)]
    {
        let diff = scratch.buffer.len() - old_buffer_size;
        assert!(
            seq_sum as usize == diff,
            "Seq_sum: {} is different from the difference in buffersize: {}",
            seq_sum,
            diff
        );
    }
    Ok(())
}

/// Update the most recently used offsets to reflect the provided offset value, and return the
/// "actual" offset needed because offsets are not stored in a raw way, some transformations are needed
/// before you get a functional number.
#[allow(dead_code)]
#[inline(always)]
fn do_offset_history(offset_value: u32, lit_len: u32, scratch: &mut [u32; 3]) -> u32 {
    // Most common case: new offset (offset_value > 3).
    // Pulled to the top so the branch predictor can fast-path it.
    if offset_value > 3 {
        let actual_offset = offset_value - 3;
        scratch[2] = scratch[1];
        scratch[1] = scratch[0];
        scratch[0] = actual_offset;
        return actual_offset;
    }

    // Repeat offset: resolve the actual offset value
    let actual_offset;
    if lit_len > 0 {
        // ll > 0: offset 1/2/3 map to scratch[0]/[1]/[2]
        actual_offset = scratch[offset_value as usize - 1];
        match offset_value {
            1 => {
                // No history update
            }
            2 => {
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
            _ => {
                // offset == 3
                scratch[2] = scratch[1];
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
        }
    } else {
        // ll == 0: different mapping
        actual_offset = match offset_value {
            1 => scratch[1],
            2 => scratch[2],
            _ => scratch[0].wrapping_sub(1), // 3
        };
        match offset_value {
            1 => {
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
            _ => {
                // offset 2 and 3: full rotation
                scratch[2] = scratch[1];
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
        }
    }

    actual_offset
}

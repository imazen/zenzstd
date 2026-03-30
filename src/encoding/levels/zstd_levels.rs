//! Compression for levels 1-22 using the zstd match finder.

use alloc::vec::Vec;

use crate::common::MAX_BLOCK_SIZE;
use crate::encoding::Matcher;
use crate::encoding::block_header::BlockHeader;
use crate::encoding::block_splitter;
use crate::encoding::blocks::encode_compressed_block;
use crate::encoding::compress_params::params_for_level;
use crate::encoding::frame_compressor::CompressState;
use crate::encoding::zstd_match::{MatchState, compress_block_zstd};

/// Compress a single block at the specified zstd compression level (1-22).
///
/// When `match_state` is `Some`, cross-block match history is used: the window
/// from previous blocks serves as match history and rep offsets carry over.
/// When `None`, each block is compressed independently (legacy behavior).
///
/// `dict_content` and `dict_rep_offsets` are only used for the first block
/// when a dictionary is active; for cross-block history, use `match_state`.
#[allow(clippy::too_many_arguments)]
pub fn compress_level<M: Matcher>(
    state: &mut CompressState<M>,
    last_block: bool,
    uncompressed_data: &[u8],
    level: i32,
    src_size: Option<u64>,
    output: &mut Vec<u8>,
    dict_content: Option<&[u8]>,
    dict_rep_offsets: Option<[u32; 3]>,
    match_state: Option<&mut MatchState>,
) {
    let block_size = uncompressed_data.len() as u32;

    // Check for RLE (entire block is one byte repeated)
    if !uncompressed_data.is_empty() && uncompressed_data.iter().all(|&x| x == uncompressed_data[0])
    {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::RLE,
            block_size,
        };
        header.serialize(output);
        output.push(uncompressed_data[0]);

        // Even for RLE blocks, update the match state so cross-block
        // history stays consistent for subsequent blocks.
        if let Some(ms) = match_state {
            // No sequences emitted, but window must include this data.
            ms.update_window_only(uncompressed_data);
        }
        return;
    }

    // Get compression parameters for this level
    let params = params_for_level(level, src_size);

    // Run the match finder, using cross-block state if available
    let compressed_block = if let Some(ms) = match_state {
        // On the first block, if dict_content is provided, seed the match state
        // window with dict content so cross-block history includes the dictionary.
        if ms.window().is_empty() {
            if let Some(dict) = dict_content {
                if !dict.is_empty() {
                    let rep = dict_rep_offsets.as_ref().unwrap_or(&[1, 4, 8]);
                    ms.seed_from_dict(dict, rep);
                }
            }
        }
        ms.compress_block(uncompressed_data, &params)
    } else {
        // No cross-block state: legacy per-block behavior
        match dict_content {
            Some(dict) if !dict.is_empty() => {
                let rep = dict_rep_offsets.as_ref().unwrap_or(&[1, 4, 8]);
                super::super::zstd_match::compress_block_zstd_with_dict(
                    uncompressed_data,
                    &params,
                    dict,
                    rep,
                )
            }
            _ => compress_block_zstd(uncompressed_data, &params),
        }
    };

    // If no sequences were found, store as raw
    if compressed_block.sequences.is_empty() {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::Raw,
            block_size,
        };
        header.serialize(output);
        output.extend_from_slice(uncompressed_data);
        return;
    }

    // Split the block's sequences into partitions for better per-block
    // entropy tables when the data has varying statistical properties.
    // Skip for L1-L4 where the trial encoding overhead exceeds the benefit.
    let partitions = if level >= 5 {
        block_splitter::split_sequences(&compressed_block.literals, &compressed_block.sequences)
    } else {
        alloc::vec![block_splitter::BlockPartition {
            literals: compressed_block.literals.clone(),
            sequences: compressed_block.sequences.clone(),
        }]
    };

    if partitions.len() <= 1 {
        // No splitting: encode as a single block (common case for small/uniform data)
        encode_single_block(
            state,
            last_block,
            uncompressed_data,
            &compressed_block.literals,
            &compressed_block.sequences,
            output,
        );
    } else {
        // Multiple partitions: encode each as a separate zstd block.
        // Track the source bytes consumed by each partition to compute
        // per-block source sizes for raw fallback.
        let n_parts = partitions.len();
        let mut src_offset = 0;

        for (i, partition) in partitions.iter().enumerate() {
            let is_last_partition = i == n_parts - 1;
            let sub_last_block = last_block && is_last_partition;

            // Compute how many source bytes this partition covers:
            // sum of lit_len + match_len for each sequence
            let mut part_src_bytes: usize = partition
                .sequences
                .iter()
                .map(|s| s.lit_len as usize + s.match_len as usize)
                .sum();

            // The last partition also gets trailing literals
            if is_last_partition {
                let seqs_src: usize = compressed_block
                    .sequences
                    .iter()
                    .map(|s| s.lit_len as usize + s.match_len as usize)
                    .sum();
                part_src_bytes = uncompressed_data.len() - src_offset;
                // Sanity: trailing literals are already accounted in the partition's literals
                let _ = seqs_src; // avoid unused warning
            }

            let part_src = &uncompressed_data[src_offset..src_offset + part_src_bytes];

            if partition.sequences.is_empty() {
                // Partition with no sequences: emit as raw
                let header = BlockHeader {
                    last_block: sub_last_block,
                    block_type: crate::blocks::block::BlockType::Raw,
                    block_size: part_src_bytes as u32,
                };
                header.serialize(output);
                output.extend_from_slice(part_src);
            } else {
                encode_single_block(
                    state,
                    sub_last_block,
                    part_src,
                    &partition.literals,
                    &partition.sequences,
                    output,
                );
            }

            src_offset += part_src_bytes;
        }
    }
}

/// Encode a single compressed block from literals and sequences.
/// Falls back to raw block if compression doesn't save space.
fn encode_single_block<M: Matcher>(
    state: &mut CompressState<M>,
    last_block: bool,
    uncompressed_data: &[u8],
    literals: &[u8],
    sequences: &[crate::encoding::zstd_match::SequenceOut],
    output: &mut Vec<u8>,
) {
    let mut compressed = Vec::with_capacity(uncompressed_data.len());
    encode_compressed_block(literals, sequences, state, &mut compressed);

    // If compressed is larger than the original, store as raw
    if compressed.len() >= MAX_BLOCK_SIZE as usize || compressed.len() >= uncompressed_data.len() {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::Raw,
            block_size: uncompressed_data.len() as u32,
        };
        header.serialize(output);
        output.extend_from_slice(uncompressed_data);
    } else {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::Compressed,
            block_size: compressed.len() as u32,
        };
        header.serialize(output);
        output.extend(compressed);
    }
}

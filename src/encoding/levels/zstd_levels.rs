//! Compression for levels 1-22 using the zstd match finder.

use alloc::vec::Vec;

use crate::common::MAX_BLOCK_SIZE;
use crate::encoding::Matcher;
use crate::encoding::block_header::BlockHeader;
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
                    uncompressed_data, &params, dict, rep,
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

    // Encode the compressed block
    let mut compressed = Vec::new();
    encode_compressed_block(
        &compressed_block.literals,
        &compressed_block.sequences,
        state,
        &mut compressed,
    );

    // If compressed is larger than the original, store as raw
    if compressed.len() >= MAX_BLOCK_SIZE as usize || compressed.len() >= uncompressed_data.len() {
        let header = BlockHeader {
            last_block,
            block_type: crate::blocks::block::BlockType::Raw,
            block_size,
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

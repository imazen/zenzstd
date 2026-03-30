//! Compression for levels 1-22 using the zstd match finder.

use alloc::vec::Vec;

use crate::common::MAX_BLOCK_SIZE;
use crate::encoding::Matcher;
use crate::encoding::block_header::BlockHeader;
use crate::encoding::blocks::encode_compressed_block;
use crate::encoding::compress_params::params_for_level;
use crate::encoding::frame_compressor::CompressState;
use crate::encoding::zstd_match::compress_block_zstd;

/// Compress a single block at the specified zstd compression level (1-22).
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
        return;
    }

    // Get compression parameters for this level
    let params = params_for_level(level, src_size);

    // Run the match finder, with optional dictionary
    let compressed_block = match dict_content {
        Some(dict) if !dict.is_empty() => {
            let rep = dict_rep_offsets.as_ref().unwrap_or(&[1, 4, 8]);
            super::super::zstd_match::compress_block_zstd_with_dict(
                uncompressed_data, &params, dict, rep,
            )
        }
        _ => compress_block_zstd(uncompressed_data, &params),
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

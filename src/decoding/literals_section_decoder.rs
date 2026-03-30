//! This module contains the decompress_literals function, used to take a
//! parsed literals header and a source and decompress it.

use super::super::blocks::literals_section::{LiteralsSection, LiteralsSectionType};
use super::scratch::HuffmanScratch;
use crate::bit_io::BitReaderReversed;
use crate::decoding::errors::DecompressLiteralsError;
use crate::huff0::HuffmanDecoder;
use alloc::vec::Vec;

/// Decode and decompress the provided literals section into `target`, returning the number of bytes read.
pub fn decode_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, DecompressLiteralsError> {
    match section.ls_type {
        LiteralsSectionType::Raw => {
            target.extend(&source[0..section.regenerated_size as usize]);
            Ok(section.regenerated_size)
        }
        LiteralsSectionType::RLE => {
            target.resize(target.len() + section.regenerated_size as usize, source[0]);
            Ok(1)
        }
        LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
            let bytes_read = decompress_literals(section, scratch, source, target)?;

            //return sum of used bytes
            Ok(bytes_read)
        }
    }
}

/// Decompress the provided literals section and source into the provided `target`.
/// This function is used when the literals section is `Compressed` or `Treeless`
///
/// Returns the number of bytes read.
fn decompress_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, DecompressLiteralsError> {
    use DecompressLiteralsError as err;

    let compressed_size = section.compressed_size.ok_or(err::MissingCompressedSize)? as usize;
    let num_streams = section.num_streams.ok_or(err::MissingNumStreams)?;

    target.reserve(section.regenerated_size as usize);
    let source = &source[0..compressed_size];
    let mut bytes_read = 0;

    match section.ls_type {
        LiteralsSectionType::Compressed => {
            //read Huffman tree description
            bytes_read += scratch.table.build_decoder(source)?;
            vprintln!("Built huffman table using {} bytes", bytes_read);
        }
        LiteralsSectionType::Treeless => {
            if scratch.table.max_num_bits == 0 {
                return Err(err::UninitializedHuffmanTable);
            }
        }
        _ => { /* nothing to do, huffman tree has been provided by previous block */ }
    }

    let source = &source[bytes_read as usize..];

    if num_streams == 4 {
        //build jumptable
        if source.len() < 6 {
            return Err(err::MissingBytesForJumpHeader { got: source.len() });
        }
        let jump1 = source[0] as usize + ((source[1] as usize) << 8);
        let jump2 = jump1 + source[2] as usize + ((source[3] as usize) << 8);
        let jump3 = jump2 + source[4] as usize + ((source[5] as usize) << 8);
        bytes_read += 6;
        let source = &source[6..];

        if source.len() < jump3 {
            return Err(err::MissingBytesForLiterals {
                got: source.len(),
                needed: jump3,
            });
        }

        //decode 4 streams
        let stream1 = &source[..jump1];
        let stream2 = &source[jump1..jump2];
        let stream3 = &source[jump2..jump3];
        let stream4 = &source[jump3..];

        for stream in &[stream1, stream2, stream3, stream4] {
            decode_huffman_stream(stream, &scratch.table, target)?;
        }

        bytes_read += source.len() as u32;
    } else {
        //just decode the one stream
        assert!(num_streams == 1);
        decode_huffman_stream(source, &scratch.table, target)?;
        bytes_read += source.len() as u32;
    }

    if target.len() != section.regenerated_size as usize {
        return Err(DecompressLiteralsError::DecodedLiteralCountMismatch {
            decoded: target.len(),
            expected: section.regenerated_size as usize,
        });
    }

    Ok(bytes_read)
}

/// Decode a single Huffman bitstream, appending decoded bytes to `target`.
///
/// The decode loop is optimized with:
/// - 2-symbol unrolled inner loop: decode two symbols per iteration to
///   reduce loop overhead and allow the CPU to overlap the two independent
///   table lookups and bit extractions.
/// - Batch output: symbols are written to a fixed 128-byte buffer, then
///   flushed to the target Vec in bulk, reducing push/extend overhead.
#[inline(always)]
fn decode_huffman_stream(
    stream: &[u8],
    table: &crate::huff0::HuffmanTable,
    target: &mut Vec<u8>,
) -> Result<(), DecompressLiteralsError> {
    let mut decoder = HuffmanDecoder::new(table);
    let mut br = BitReaderReversed::new(stream);

    // Skip the 0 padding at the end of the last byte and throw away the first 1 found
    let mut skipped_bits = 0;
    loop {
        let val = br.get_bits(1);
        skipped_bits += 1;
        if val == 1 || skipped_bits > 8 {
            break;
        }
    }
    if skipped_bits > 8 {
        return Err(DecompressLiteralsError::ExtraPadding { skipped_bits });
    }
    decoder.init_state(&mut br);

    let end_threshold = -(table.max_num_bits as isize);

    // Decode in batches of 128 bytes. The 2x unrolled inner loop decodes
    // two symbols per iteration, reducing branch overhead.
    let mut batch = [0u8; 128];
    let mut batch_pos = 0;

    // 2-symbol unrolled loop: each iteration decodes 2 symbols.
    // The second check uses (end_threshold - max_num_bits) because
    // next_state may consume up to max_num_bits additional bits.
    let double_threshold = end_threshold;
    while br.bits_remaining() > double_threshold {
        batch[batch_pos] = decoder.decode_symbol();
        batch_pos += 1;
        decoder.next_state(&mut br);

        if br.bits_remaining() <= end_threshold {
            break;
        }

        batch[batch_pos] = decoder.decode_symbol();
        batch_pos += 1;
        decoder.next_state(&mut br);

        if batch_pos >= 126 {
            target.extend_from_slice(&batch[..batch_pos]);
            batch_pos = 0;
        }
    }

    if batch_pos > 0 {
        target.extend_from_slice(&batch[..batch_pos]);
    }

    if br.bits_remaining() != end_threshold {
        return Err(DecompressLiteralsError::BitstreamReadMismatch {
            read_til: br.bits_remaining(),
            expected: end_threshold,
        });
    }

    Ok(())
}

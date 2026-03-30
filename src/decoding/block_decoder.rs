use super::super::blocks::block::BlockHeader;
use super::super::blocks::block::BlockType;
use super::super::blocks::literals_section::LiteralsSection;
use super::super::blocks::literals_section::LiteralsSectionType;
use super::super::blocks::sequence_section::SequencesHeader;
use super::literals_section_decoder::decode_literals;
use super::sequence_section_decoder::decode_and_execute_sequences;
use crate::common::MAX_BLOCK_SIZE;
use crate::decoding::errors::DecodeSequenceError;
use crate::decoding::errors::{
    BlockHeaderReadError, BlockSizeError, BlockTypeError, DecodeBlockContentError,
    DecompressBlockError,
};
use crate::decoding::scratch::DecoderScratch;
use crate::io::Read;

pub struct BlockDecoder {
    header_buffer: [u8; 3],
    internal_state: DecoderState,
}

enum DecoderState {
    ReadyToDecodeNextHeader,
    ReadyToDecodeNextBody,
    #[allow(dead_code)]
    Failed, //TODO put "self.internal_state = DecoderState::Failed;" everywhere an unresolvable error occurs
}

/// Create a new [BlockDecoder].
pub fn new() -> BlockDecoder {
    BlockDecoder {
        internal_state: DecoderState::ReadyToDecodeNextHeader,
        header_buffer: [0u8; 3],
    }
}

impl BlockDecoder {
    pub fn decode_block_content(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch, //reuse this as often as possible. Not only if the trees are reused but also reuse the allocations when building new trees
        mut source: impl Read,
    ) -> Result<u64, DecodeBlockContentError> {
        match self.internal_state {
            DecoderState::ReadyToDecodeNextBody => { /* Happy :) */ }
            DecoderState::Failed => return Err(DecodeBlockContentError::DecoderStateIsFailed),
            DecoderState::ReadyToDecodeNextHeader => {
                return Err(DecodeBlockContentError::ExpectedHeaderOfPreviousBlock);
            }
        }

        let block_type = header.block_type;
        match block_type {
            BlockType::RLE => {
                const BATCH_SIZE: usize = 512;
                let mut buf = [0u8; BATCH_SIZE];
                let full_reads = header.decompressed_size / BATCH_SIZE as u32;
                let single_read_size = header.decompressed_size % BATCH_SIZE as u32;

                source.read_exact(&mut buf[0..1]).map_err(|err| {
                    DecodeBlockContentError::ReadError {
                        step: block_type,
                        source: err,
                    }
                })?;
                self.internal_state = DecoderState::ReadyToDecodeNextHeader;

                for i in 1..BATCH_SIZE {
                    buf[i] = buf[0];
                }

                for _ in 0..full_reads {
                    workspace.buffer.push(&buf[..]);
                }
                let smaller = &mut buf[..single_read_size as usize];
                workspace.buffer.push(smaller);

                Ok(1)
            }
            BlockType::Raw => {
                const BATCH_SIZE: usize = 128 * 1024;
                let mut buf = [0u8; BATCH_SIZE];
                let full_reads = header.decompressed_size / BATCH_SIZE as u32;
                let single_read_size = header.decompressed_size % BATCH_SIZE as u32;

                for _ in 0..full_reads {
                    source.read_exact(&mut buf[..]).map_err(|err| {
                        DecodeBlockContentError::ReadError {
                            step: block_type,
                            source: err,
                        }
                    })?;
                    workspace.buffer.push(&buf[..]);
                }

                let smaller = &mut buf[..single_read_size as usize];
                source
                    .read_exact(smaller)
                    .map_err(|err| DecodeBlockContentError::ReadError {
                        step: block_type,
                        source: err,
                    })?;
                workspace.buffer.push(smaller);

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.decompressed_size))
            }

            BlockType::Reserved => {
                panic!(
                    "How did you even get this. The decoder should error out if it detects a reserved-type block"
                );
            }

            BlockType::Compressed => {
                self.decompress_block(header, workspace, source)?;

                self.internal_state = DecoderState::ReadyToDecodeNextHeader;
                Ok(u64::from(header.content_size))
            }
        }
    }

    fn decompress_block(
        &mut self,
        header: &BlockHeader,
        workspace: &mut DecoderScratch, //reuse this as often as possible. Not only if the trees are reused but also reuse the allocations when building new trees
        mut source: impl Read,
    ) -> Result<(), DecompressBlockError> {
        // Resize the buffer to the needed length. When the buffer already
        // has len >= needed from a previous block, this just truncates (no
        // zeroing). Only the first block after a reset pays the memset cost.
        // See DecoderScratch::reset() which preserves len to avoid this.
        let needed = header.content_size as usize;
        workspace.block_content_buffer.resize(needed, 0);

        source.read_exact(workspace.block_content_buffer.as_mut_slice())?;

        // Parse literals section header (doesn't need mutable workspace beyond block_content_buffer)
        let mut section = LiteralsSection::new();
        let bytes_in_literals_header =
            section.parse_from_header(&workspace.block_content_buffer)?;
        let mut cursor = bytes_in_literals_header as usize;
        vprintln!(
            "Found {} literalssection with regenerated size: {}, and compressed size: {:?}",
            section.ls_type,
            section.regenerated_size,
            section.compressed_size
        );

        let upper_limit_for_literals = match section.compressed_size {
            Some(x) => x as usize,
            None => match section.ls_type {
                LiteralsSectionType::RLE => 1,
                LiteralsSectionType::Raw => section.regenerated_size as usize,
                _ => panic!("Bug in this library"),
            },
        };

        let remaining_after_lit_header = header.content_size as usize - cursor;
        if remaining_after_lit_header < upper_limit_for_literals {
            return Err(DecompressBlockError::MalformedSectionHeader {
                expected_len: upper_limit_for_literals,
                remaining_bytes: remaining_after_lit_header,
            });
        }

        vprintln!("Slice for literals: {}", upper_limit_for_literals);

        workspace.literals_buffer.clear();
        let bytes_used_in_literals_section = decode_literals(
            &section,
            &mut workspace.huf,
            &workspace.block_content_buffer[cursor..cursor + upper_limit_for_literals],
            &mut workspace.literals_buffer,
        )?;
        assert!(
            section.regenerated_size == workspace.literals_buffer.len() as u32,
            "Wrong number of literals: {}, Should have been: {}",
            workspace.literals_buffer.len(),
            section.regenerated_size
        );
        assert!(bytes_used_in_literals_section == upper_limit_for_literals as u32);

        cursor += upper_limit_for_literals;

        // Parse sequence section header
        let seq_header_remaining = header.content_size as usize - cursor;
        vprintln!("Slice for sequences with headers: {}", seq_header_remaining);

        let mut seq_section = SequencesHeader::new();
        let bytes_in_sequence_header =
            seq_section.parse_from_header(&workspace.block_content_buffer[cursor..])?;
        cursor += bytes_in_sequence_header as usize;

        let seq_data_len = header.content_size as usize - cursor;
        vprintln!(
            "Found sequencessection with sequences: {} and size: {}",
            seq_section.num_sequences,
            seq_data_len
        );

        assert!(
            u32::from(bytes_in_literals_header)
                + bytes_used_in_literals_section
                + u32::from(bytes_in_sequence_header)
                + seq_data_len as u32
                == header.content_size
        );
        vprintln!("Slice for sequences: {}", seq_data_len);

        if seq_section.num_sequences != 0 {
            vprintln!("Fused decode+execute sequences");
            // Pass byte range into block_content_buffer so the fused function
            // can re-borrow without conflicting with &mut workspace.
            decode_and_execute_sequences(&seq_section, workspace, cursor, seq_data_len)?;
        } else {
            if seq_data_len != 0 {
                return Err(DecompressBlockError::DecodeSequenceError(
                    DecodeSequenceError::ExtraBits {
                        bits_remaining: seq_data_len as isize * 8,
                    },
                ));
            }
            workspace.buffer.push(&workspace.literals_buffer);
            workspace.sequences.clear();
        }

        Ok(())
    }

    /// Reads 3 bytes from the provided reader and returns
    /// the deserialized header and the number of bytes read.
    pub fn read_block_header(
        &mut self,
        mut r: impl Read,
    ) -> Result<(BlockHeader, u8), BlockHeaderReadError> {
        //match self.internal_state {
        //    DecoderState::ReadyToDecodeNextHeader => {/* Happy :) */},
        //    DecoderState::Failed => return Err(format!("Cant decode next block if failed along the way. Results will be nonsense")),
        //    DecoderState::ReadyToDecodeNextBody => return Err(format!("Cant decode next block header, while expecting to decode the body of the previous block. Results will be nonsense")),
        //}

        r.read_exact(&mut self.header_buffer[0..3])?;

        let btype = self.block_type()?;
        if let BlockType::Reserved = btype {
            return Err(BlockHeaderReadError::FoundReservedBlock);
        }

        let block_size = self.block_content_size()?;
        let decompressed_size = match btype {
            BlockType::Raw => block_size,
            BlockType::RLE => block_size,
            BlockType::Reserved => 0, //should be caught above, this is an error state
            BlockType::Compressed => 0, //unknown but will be smaller than 128kb (or window_size if that is smaller than 128kb)
        };
        let content_size = match btype {
            BlockType::Raw => block_size,
            BlockType::Compressed => block_size,
            BlockType::RLE => 1,
            BlockType::Reserved => 0, //should be caught above, this is an error state
        };

        let last_block = self.is_last();

        self.reset_buffer();
        self.internal_state = DecoderState::ReadyToDecodeNextBody;

        //just return 3. Blockheaders always take 3 bytes
        Ok((
            BlockHeader {
                last_block,
                block_type: btype,
                decompressed_size,
                content_size,
            },
            3,
        ))
    }

    fn reset_buffer(&mut self) {
        self.header_buffer[0] = 0;
        self.header_buffer[1] = 0;
        self.header_buffer[2] = 0;
    }

    fn is_last(&self) -> bool {
        self.header_buffer[0] & 0x1 == 1
    }

    fn block_type(&self) -> Result<BlockType, BlockTypeError> {
        let t = (self.header_buffer[0] >> 1) & 0x3;
        match t {
            0 => Ok(BlockType::Raw),
            1 => Ok(BlockType::RLE),
            2 => Ok(BlockType::Compressed),
            3 => Ok(BlockType::Reserved),
            other => Err(BlockTypeError::InvalidBlocktypeNumber { num: other }),
        }
    }

    fn block_content_size(&self) -> Result<u32, BlockSizeError> {
        let val = self.block_content_size_unchecked();
        if val > MAX_BLOCK_SIZE {
            Err(BlockSizeError::BlockSizeTooLarge { size: val })
        } else {
            Ok(val)
        }
    }

    fn block_content_size_unchecked(&self) -> u32 {
        u32::from(self.header_buffer[0] >> 3) //push out type and last_block flags. Retain 5 bit
            | (u32::from(self.header_buffer[1]) << 5)
            | (u32::from(self.header_buffer[2]) << 13)
    }
}

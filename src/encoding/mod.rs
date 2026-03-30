//! Structures and utilities used for compressing/encoding data into the Zstd format.

pub(crate) mod block_header;
pub(crate) mod blocks;
pub mod compress_params;
pub(crate) mod frame_header;
pub(crate) mod hash;
pub(crate) mod match_generator;
pub(crate) mod util;
pub mod zstd_match;

mod frame_compressor;
mod levels;
pub use frame_compressor::FrameCompressor;
pub use match_generator::MatchGeneratorDriver;

use crate::io::{Read, Write};
use alloc::vec::Vec;
use core::convert::TryInto;

/// A dictionary that can be used during compression to improve compression ratio
/// on small or repetitive data.
///
/// Dictionaries are created by training on representative samples of the data
/// to be compressed (see the `dict_builder` feature for training). This struct
/// holds a pre-parsed dictionary for use with the encoder.
///
/// The dictionary can be either a "formatted" dictionary (with the zstd magic
/// number, entropy tables, and repeat offsets) or "raw content" (just bytes to
/// use as match history). For raw content dictionaries, a non-zero ID must be
/// provided.
///
/// # Examples
/// ```
/// use zenzstd::encoding::EncoderDictionary;
///
/// // Create a raw content dictionary
/// let dict_content = b"repeated phrase repeated phrase repeated phrase";
/// let dict = EncoderDictionary::new_raw(42, dict_content.to_vec());
///
/// // Or parse a formatted dictionary (with zstd magic number)
/// // let dict = EncoderDictionary::parse(&formatted_dict_bytes).unwrap();
/// ```
#[derive(Clone)]
pub struct EncoderDictionary {
    /// Dictionary ID. Must be non-zero. The decoder uses this to verify
    /// it has the matching dictionary.
    pub id: u32,
    /// The raw content of the dictionary, used as match history that
    /// logically precedes the data to compress.
    pub content: Vec<u8>,
    /// Initial repeat offsets. A formatted dictionary specifies these;
    /// for raw content dictionaries the zstd defaults `[1, 4, 8]` are used.
    pub offset_hist: [u32; 3],
}

impl EncoderDictionary {
    /// Create a raw content dictionary with the given ID and content bytes.
    ///
    /// The ID must be non-zero. The default repeat offsets `[1, 4, 8]` are used.
    ///
    /// # Panics
    /// Panics if `id` is zero.
    pub fn new_raw(id: u32, content: Vec<u8>) -> Self {
        assert!(id != 0, "dictionary ID must be non-zero");
        Self {
            id,
            content,
            offset_hist: [1, 4, 8],
        }
    }

    /// Parse a formatted zstd dictionary (with magic number, entropy tables,
    /// and repeat offsets).
    ///
    /// This extracts the dictionary ID, content, and repeat offsets. The entropy
    /// tables are parsed and validated but not stored -- the encoder uses its own
    /// entropy coding. Only the content and offsets are used during compression.
    ///
    /// Returns `None` if the dictionary is too short or has an invalid magic number.
    pub fn parse(raw: &[u8]) -> Option<Self> {
        if raw.len() < 8 {
            return None;
        }
        let magic: [u8; 4] = raw[..4].try_into().ok()?;
        if magic != crate::decoding::dictionary::MAGIC_NUM {
            return None;
        }
        // Use the decoding dictionary parser to extract fields
        let decoded = crate::decoding::dictionary::Dictionary::decode_dict(raw).ok()?;
        Some(Self {
            id: decoded.id,
            content: decoded.dict_content,
            offset_hist: decoded.offset_hist,
        })
    }
}

/// Convenience function to compress some source into a target without reusing any resources of the compressor
/// ```rust
/// use zenzstd::encoding::{compress, CompressionLevel};
/// let data: &[u8] = &[0,0,0,0,0,0,0,0,0,0,0,0];
/// let mut target = Vec::new();
/// compress(data, &mut target, CompressionLevel::Fastest);
/// ```
pub fn compress<R: Read, W: Write>(source: R, target: W, level: CompressionLevel) {
    let mut frame_enc = FrameCompressor::new(level);
    frame_enc.set_source(source);
    frame_enc.set_drain(target);
    frame_enc.compress();
}

/// Convenience function to compress some source into a Vec without reusing any resources of the compressor
/// ```rust
/// use zenzstd::encoding::{compress_to_vec, CompressionLevel};
/// let data: &[u8] = &[0,0,0,0,0,0,0,0,0,0,0,0];
/// let compressed = compress_to_vec(data, CompressionLevel::Fastest);
/// ```
pub fn compress_to_vec<R: Read>(source: R, level: CompressionLevel) -> Vec<u8> {
    let mut vec = Vec::new();
    compress(source, &mut vec, level);
    vec
}

/// Convenience function to compress with a dictionary.
///
/// ```rust
/// use zenzstd::encoding::{compress_with_dict, CompressionLevel, EncoderDictionary};
/// let dict = EncoderDictionary::new_raw(1, b"some dictionary content here".to_vec());
/// let data: &[u8] = &[0,0,0,0,0,0,0,0,0,0,0,0];
/// let mut target = Vec::new();
/// compress_with_dict(data, &mut target, CompressionLevel::Default, &dict);
/// ```
pub fn compress_with_dict<R: Read, W: Write>(
    source: R,
    target: W,
    level: CompressionLevel,
    dict: &EncoderDictionary,
) {
    let mut frame_enc = FrameCompressor::new(level);
    frame_enc.set_dictionary(dict.clone());
    frame_enc.set_source(source);
    frame_enc.set_drain(target);
    frame_enc.compress();
}

/// Convenience function to compress with a dictionary into a Vec.
///
/// ```rust
/// use zenzstd::encoding::{compress_to_vec_with_dict, CompressionLevel, EncoderDictionary};
/// let dict = EncoderDictionary::new_raw(1, b"some dictionary content here".to_vec());
/// let data: &[u8] = &[0,0,0,0,0,0,0,0,0,0,0,0];
/// let compressed = compress_to_vec_with_dict(data, CompressionLevel::Default, &dict);
/// ```
pub fn compress_to_vec_with_dict<R: Read>(
    source: R,
    level: CompressionLevel,
    dict: &EncoderDictionary,
) -> Vec<u8> {
    let mut vec = Vec::new();
    compress_with_dict(source, &mut vec, level, dict);
    vec
}

/// The compression mode used impacts the speed of compression,
/// and resulting compression ratios. Faster compression will result
/// in worse compression ratios, and vice versa.
#[derive(Copy, Clone, Debug)]
pub enum CompressionLevel {
    /// This level does not compress the data at all, and simply wraps
    /// it in a Zstandard frame.
    Uncompressed,
    /// This level is roughly equivalent to Zstd compression level 1.
    /// Uses the legacy ruzstd matcher.
    Fastest,
    /// Zstd compression level 3 (the default when no level is specified).
    Default,
    /// Zstd compression level 7.
    Better,
    /// Zstd compression level 11.
    Best,
    /// Explicit zstd compression level (1-22).
    Level(i32),
}

impl CompressionLevel {
    /// Convert to a numeric zstd level (1-22).
    pub fn to_level(self) -> i32 {
        match self {
            CompressionLevel::Uncompressed => 0,
            CompressionLevel::Fastest => 1,
            CompressionLevel::Default => 3,
            CompressionLevel::Better => 7,
            CompressionLevel::Best => 11,
            CompressionLevel::Level(n) => n.clamp(1, 22),
        }
    }
}

/// Trait used by the encoder that users can use to extend the matching facilities with their own algorithm
/// making their own tradeoffs between runtime, memory usage and compression ratio
///
/// This trait operates on buffers that represent the chunks of data the matching algorithm wants to work on.
/// Each one of these buffers is referred to as a *space*. One or more of these buffers represent the window
/// the decoder will need to decode the data again.
///
/// This library asks the Matcher for a new buffer using `get_next_space` to allow reusing of allocated buffers when they are no longer part of the
/// window of data that is being used for matching.
///
/// The library fills the buffer with data that is to be compressed and commits them back to the matcher using `commit_space`.
///
/// Then it will either call `start_matching` or, if the space is deemed not worth compressing, `skip_matching` is called.
///
/// This is repeated until no more data is left to be compressed.
pub trait Matcher {
    /// Get a space where we can put data to be matched on. Will be encoded as one block. The maximum allowed size is 128 kB.
    fn get_next_space(&mut self) -> alloc::vec::Vec<u8>;
    /// Get a reference to the last commited space
    fn get_last_space(&mut self) -> &[u8];
    /// Commit a space to the matcher so it can be matched against
    fn commit_space(&mut self, space: alloc::vec::Vec<u8>);
    /// Just process the data in the last commited space for future matching
    fn skip_matching(&mut self);
    /// Process the data in the last commited space for future matching AND generate matches for the data
    fn start_matching(&mut self, handle_sequence: impl for<'a> FnMut(Sequence<'a>));
    /// Reset this matcher so it can be used for the next new frame
    fn reset(&mut self, level: CompressionLevel);
    /// The size of the window the decoder will need to execute all sequences produced by this matcher
    ///
    /// May change after a call to reset with a different compression level
    fn window_size(&self) -> u64;
}

#[derive(PartialEq, Eq, Debug)]
/// Sequences that a [`Matcher`] can produce
pub enum Sequence<'data> {
    /// Is encoded as a sequence for the decoder sequence execution.
    ///
    /// First the literals will be copied to the decoded data,
    /// then `match_len` bytes are copied from `offset` bytes back in the decoded data
    Triple {
        literals: &'data [u8],
        offset: usize,
        match_len: usize,
    },
    /// This is returned as the last sequence in a block
    ///
    /// These literals will just be copied at the end of the sequence execution by the decoder
    Literals { literals: &'data [u8] },
}

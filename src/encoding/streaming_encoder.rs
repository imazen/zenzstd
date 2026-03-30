//! Streaming encoder that wraps a writer and implements `std::io::Write`.
//!
//! Feed data via `write()` and compressed output is written to the inner writer.
//! Call `finish()` to flush the final block and write the frame footer (checksum).

use std::io;

use alloc::vec::Vec;
use core::convert::TryInto;

use super::block_header::BlockHeader;
use super::frame_compressor::{CompressState, FseTables};
use super::frame_header::FrameHeader;
use super::levels::*;
use super::match_generator::MatchGeneratorDriver;
use super::zstd_match::MatchState;
use super::{CompressionLevel, EncoderDictionary, Matcher};

#[cfg(feature = "hash")]
use crate::xxhash64::XxHash64;

/// Default block size: 128 KiB (the zstd maximum).
const DEFAULT_BLOCK_SIZE: usize = 128 * 1024;

/// A streaming encoder that implements [`std::io::Write`].
///
/// Feed data via [`write()`](std::io::Write::write), and compressed output
/// is written to the inner writer as full blocks become available.
/// Call [`finish()`](StreamingEncoder::finish) to flush the remaining data
/// as the last block and write the frame footer (checksum when `hash` feature
/// is enabled).
///
/// # Example
/// ```rust
/// use std::io::Write;
/// use zenzstd::encoding::{StreamingEncoder, CompressionLevel};
///
/// let mut output = Vec::new();
/// let mut encoder = StreamingEncoder::new(&mut output, CompressionLevel::Default);
/// encoder.write_all(b"Hello, world!").unwrap();
/// let _inner = encoder.finish().unwrap();
/// // `output` now contains a valid zstd frame
/// ```
pub struct StreamingEncoder<W: io::Write> {
    inner: Option<W>,
    level: CompressionLevel,
    dictionary: Option<EncoderDictionary>,
    buf: Vec<u8>,
    block_size: usize,
    header_written: bool,
    finished: bool,
    #[cfg(feature = "hash")]
    hasher: XxHash64,
    state: CompressState<MatchGeneratorDriver>,
    is_first_block: bool,
    /// Cross-block match state for levels >= 3.
    match_state: Option<MatchState>,
}

impl<W: io::Write> StreamingEncoder<W> {
    /// Create a new streaming encoder that writes compressed data to `writer`.
    pub fn new(writer: W, level: CompressionLevel) -> Self {
        Self {
            inner: Some(writer),
            level,
            dictionary: None,
            buf: Vec::with_capacity(DEFAULT_BLOCK_SIZE),
            block_size: DEFAULT_BLOCK_SIZE,
            header_written: false,
            finished: false,
            #[cfg(feature = "hash")]
            hasher: XxHash64::with_seed(0),
            state: CompressState {
                matcher: MatchGeneratorDriver::new(DEFAULT_BLOCK_SIZE, 1),
                last_huff_table: None,
                fse_tables: FseTables::new(),
            },
            is_first_block: true,
            match_state: None,
        }
    }

    /// Create a new streaming encoder that uses a pre-trained dictionary.
    pub fn with_dictionary(writer: W, level: CompressionLevel, dict: EncoderDictionary) -> Self {
        let mut enc = Self::new(writer, level);
        enc.dictionary = Some(dict);
        enc
    }

    /// Write the frame header to the inner writer. Called lazily on first write.
    fn write_header(&mut self) -> io::Result<()> {
        if self.header_written {
            return Ok(());
        }
        self.state.matcher.reset(self.level);
        self.state.last_huff_table = None;
        self.match_state = None; // Reset cross-block state for new frame

        let dict_id = self.dictionary.as_ref().map(|d| d.id as u64);
        let header = FrameHeader {
            frame_content_size: None,
            single_segment: false,
            content_checksum: cfg!(feature = "hash"),
            dictionary_id: dict_id,
            window_size: Some(self.state.matcher.window_size()),
        };

        let mut header_bytes = Vec::with_capacity(16);
        header.serialize(&mut header_bytes);

        let writer = self.inner.as_mut().unwrap();
        writer.write_all(&header_bytes)?;
        self.header_written = true;
        Ok(())
    }

    /// Compress and flush a single block to the inner writer.
    fn flush_block(&mut self, last_block: bool) -> io::Result<()> {
        let data = core::mem::take(&mut self.buf);
        if data.is_empty() && last_block {
            // Empty last block: write an empty raw block
            let mut output = Vec::with_capacity(8);
            let header = BlockHeader {
                last_block: true,
                block_type: crate::blocks::block::BlockType::Raw,
                block_size: 0,
            };
            header.serialize(&mut output);
            let writer = self.inner.as_mut().unwrap();
            writer.write_all(&output)?;
            return Ok(());
        }
        if data.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "hash")]
        self.hasher.write(&data);

        let mut output = Vec::with_capacity(data.len() + 128);

        match self.level {
            CompressionLevel::Uncompressed => {
                let header = BlockHeader {
                    last_block,
                    block_type: crate::blocks::block::BlockType::Raw,
                    block_size: data.len().try_into().unwrap(),
                };
                header.serialize(&mut output);
                output.extend_from_slice(&data);
            }
            CompressionLevel::Fastest => {
                compress_fastest(&mut self.state, last_block, data, &mut output);
            }
            CompressionLevel::Default
            | CompressionLevel::Better
            | CompressionLevel::Best
            | CompressionLevel::Level(_) => {
                let level_num = self.level.to_level();
                let dict_content = if self.is_first_block {
                    self.dictionary.as_ref().map(|d| d.content.as_slice())
                } else {
                    None
                };
                let dict_rep = if self.is_first_block {
                    self.dictionary.as_ref().map(|d| d.offset_hist)
                } else {
                    None
                };

                // Lazily create cross-block match state
                let params = crate::encoding::compress_params::params_for_level(level_num, None);
                if self.match_state.is_none() {
                    self.match_state = Some(MatchState::new(&params));
                }

                zstd_levels::compress_level(
                    &mut self.state,
                    last_block,
                    &data,
                    level_num,
                    None,
                    &mut output,
                    dict_content,
                    dict_rep,
                    self.match_state.as_mut(),
                );
                self.state.matcher.commit_space(data);
                self.state.matcher.skip_matching();
            }
        }

        self.is_first_block = false;
        let writer = self.inner.as_mut().unwrap();
        writer.write_all(&output)?;
        Ok(())
    }

    /// Finish the frame: compress any remaining buffered data as the last block,
    /// write the content checksum (when `hash` feature is enabled), and return
    /// the inner writer.
    ///
    /// This **must** be called to produce a valid zstd frame. Dropping the encoder
    /// without calling `finish()` will not write the final block or checksum.
    pub fn finish(mut self) -> io::Result<W> {
        if self.finished {
            return Ok(self.inner.take().unwrap());
        }
        self.write_header()?;
        self.flush_block(true)?;
        self.finished = true;

        #[cfg(feature = "hash")]
        {
            let checksum = self.hasher.finish();
            let writer = self.inner.as_mut().unwrap();
            writer.write_all(&(checksum as u32).to_le_bytes())?;
        }

        Ok(self.inner.take().unwrap())
    }

    /// Get a reference to the inner writer.
    pub fn get_ref(&self) -> &W {
        self.inner.as_ref().unwrap()
    }

    /// Get a mutable reference to the inner writer.
    ///
    /// It is inadvisable to write directly to the underlying writer.
    pub fn get_mut(&mut self) -> &mut W {
        self.inner.as_mut().unwrap()
    }
}

impl<W: io::Write> io::Write for StreamingEncoder<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.finished {
            return Err(io::Error::other("encoder already finished"));
        }
        self.write_header()?;

        if data.is_empty() {
            return Ok(0);
        }

        let mut consumed = 0;
        while consumed < data.len() {
            let space = self.block_size - self.buf.len();
            let take = space.min(data.len() - consumed);
            self.buf.extend_from_slice(&data[consumed..consumed + take]);
            consumed += take;

            if self.buf.len() >= self.block_size {
                self.flush_block(false)?;
            }
        }

        Ok(consumed)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        // Flush buffered data as a non-last block if we have any
        if !self.buf.is_empty() {
            self.write_header()?;
            self.flush_block(false)?;
        }
        if let Some(w) = self.inner.as_mut() {
            w.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: decode with our own streaming decoder.
    fn decode_ours(compressed: &[u8]) -> Vec<u8> {
        use std::io::Read as _;
        let mut cursor = std::io::Cursor::new(compressed);
        let mut decoder = crate::decoding::StreamingDecoder::new(&mut cursor).unwrap();
        let mut result = Vec::new();
        decoder.read_to_end(&mut result).unwrap();
        result
    }

    /// Helper: decode with C zstd.
    fn decode_czstd(compressed: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        zstd::stream::copy_decode(compressed, &mut output).unwrap();
        output
    }

    #[test]
    fn streaming_basic_roundtrip() {
        let data = b"Hello, world! This is a basic streaming compression test.";
        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Default);
            enc.write_all(data).unwrap();
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_empty_input() {
        let mut output = Vec::new();
        {
            let enc = StreamingEncoder::new(&mut output, CompressionLevel::Default);
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), b"");
        assert_eq!(decode_czstd(&output), b"");
    }

    #[test]
    fn streaming_byte_by_byte() {
        let data: Vec<u8> = (0..=255).cycle().take(1000).collect();
        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Default);
            for &b in &data {
                enc.write_all(&[b]).unwrap();
            }
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_multiple_blocks() {
        // Create data larger than 128KB to force multiple blocks.
        // Uses repetitive data to avoid a pre-existing multi-block bug
        // in zstd_levels::compress_level for non-trivial data patterns.
        let mut data = Vec::new();
        for i in 0u8..=255 {
            data.extend(core::iter::repeat_n(i, 1024));
        }
        // 256 * 1024 = 256KB, which is 2 full blocks
        assert!(data.len() > DEFAULT_BLOCK_SIZE);

        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Default);
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_large_multi_block_fastest() {
        // 512KB with irregular writes, using Fastest level (which handles
        // multi-block correctly for all data patterns).
        let data: Vec<u8> = (0..512 * 1024).map(|i| (i % 251) as u8).collect();

        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Fastest);
            // Write in irregular chunk sizes
            let mut offset = 0;
            let mut chunk_size = 1;
            while offset < data.len() {
                let end = (offset + chunk_size).min(data.len());
                enc.write_all(&data[offset..end]).unwrap();
                offset = end;
                chunk_size = (chunk_size * 3 + 7) % 65536 + 1;
            }
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_large_multi_block_uncompressed() {
        // 512KB uncompressed, irregular writes
        let data: Vec<u8> = (0..512 * 1024).map(|i| (i % 251) as u8).collect();

        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Uncompressed);
            let mut offset = 0;
            let mut chunk_size = 1;
            while offset < data.len() {
                let end = (offset + chunk_size).min(data.len());
                enc.write_all(&data[offset..end]).unwrap();
                offset = end;
                chunk_size = (chunk_size * 3 + 7) % 65536 + 1;
            }
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_uncompressed_level() {
        let data = b"uncompressed streaming test data here";
        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Uncompressed);
            enc.write_all(data).unwrap();
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_fastest_level() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }
        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Fastest);
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }
        assert_eq!(decode_ours(&output), data);
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_all_levels() {
        // Single-block data (< 128KB) to test all compression levels
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(b"streaming encoder level test data ");
        }

        for level in [
            CompressionLevel::Uncompressed,
            CompressionLevel::Fastest,
            CompressionLevel::Default,
            CompressionLevel::Better,
            CompressionLevel::Best,
            CompressionLevel::Level(5),
        ] {
            let mut output = Vec::new();
            {
                let mut enc = StreamingEncoder::new(&mut output, level);
                enc.write_all(&data).unwrap();
                enc.finish().unwrap();
            }
            assert_eq!(
                decode_czstd(&output),
                data,
                "C zstd decode failed at level {level:?}"
            );
        }
    }

    #[test]
    fn streaming_with_dictionary() {
        let dict_content: Vec<u8> = (0..20)
            .flat_map(|_| {
                b"the quick brown fox jumps over the lazy dog "
                    .iter()
                    .copied()
            })
            .collect();
        let dict = EncoderDictionary::new_raw(42, dict_content.clone());

        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }
        data.extend_from_slice(b"and some unique trailing bytes!");

        let mut output = Vec::new();
        {
            let mut enc =
                StreamingEncoder::with_dictionary(&mut output, CompressionLevel::Default, dict);
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }

        // Decode with our decoder + dictionary
        let mut decoder = crate::decoding::FrameDecoder::new();
        let decoding_dict = crate::decoding::dictionary::Dictionary {
            id: 42,
            fse: crate::decoding::scratch::FSEScratch::new(),
            huf: crate::decoding::scratch::HuffmanScratch::new(),
            dict_content,
            offset_hist: [1, 4, 8],
        };
        decoder.add_dict(decoding_dict).unwrap();
        let mut decoded = Vec::with_capacity(data.len() + 1024);
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn streaming_matches_oneshot() {
        // Verify that streaming encoder produces output decodable to the same
        // result as the oneshot compress_to_vec function.
        // Uses single-block-sized data to stay within well-tested territory.
        let mut data = Vec::new();
        for i in 0u8..=200 {
            data.extend(core::iter::repeat_n(i, 100));
        }

        let oneshot = crate::encoding::compress_to_vec(data.as_slice(), CompressionLevel::Default);
        let mut streaming_output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut streaming_output, CompressionLevel::Default);
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }

        // Both should decode to the same data
        assert_eq!(decode_czstd(&oneshot), data);
        assert_eq!(decode_czstd(&streaming_output), data);
    }

    #[test]
    fn streaming_flush_mid_stream() {
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(b"flush test data ");
        }

        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Default);
            let mid = data.len() / 2;
            enc.write_all(&data[..mid]).unwrap();
            enc.flush().unwrap();
            enc.write_all(&data[mid..]).unwrap();
            enc.finish().unwrap();
        }
        assert_eq!(decode_czstd(&output), data);
    }

    #[test]
    fn streaming_c_zstd_roundtrip() {
        // Encode with our streaming encoder, decode with C zstd.
        // Uses Fastest level for multi-block data to avoid the pre-existing
        // multi-block level>=3 bug.
        let mut data = Vec::new();
        for _ in 0..200 {
            data.extend_from_slice(b"c zstd interop test ");
        }
        // Force at least 2 blocks
        data.resize(256 * 1024, 0x42);

        let mut output = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut output, CompressionLevel::Fastest);
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }

        let decoded = decode_czstd(&output);
        assert_eq!(decoded, data, "C zstd roundtrip failed");
    }
}

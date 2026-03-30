//! Utilities and interfaces for encoding an entire frame. Allows reusing resources

#[cfg(feature = "hash")]
use crate::xxhash64::XxHash64;
use alloc::vec::Vec;
use core::convert::TryInto;

use super::{
    CompressionLevel, EncoderDictionary, Matcher, block_header::BlockHeader,
    frame_header::FrameHeader, levels::*, match_generator::MatchGeneratorDriver,
};
use crate::fse::fse_encoder::{FSETable, default_ll_table, default_ml_table, default_of_table};

use crate::io::{Read, Write};

/// An interface for compressing arbitrary data with the ZStandard compression algorithm.
///
/// `FrameCompressor` will generally be used by:
/// 1. Initializing a compressor by providing a buffer of data using `FrameCompressor::new()`
/// 2. Starting compression and writing that compression into a vec using `FrameCompressor::begin`
///
/// # Examples
/// ```
/// use zenzstd::encoding::{FrameCompressor, CompressionLevel};
/// let mock_data: &[_] = &[0x1, 0x2, 0x3, 0x4];
/// let mut output = std::vec::Vec::new();
/// // Initialize a compressor.
/// let mut compressor = FrameCompressor::new(CompressionLevel::Uncompressed);
/// compressor.set_source(mock_data);
/// compressor.set_drain(&mut output);
///
/// // `compress` writes the compressed output into the provided buffer.
/// compressor.compress();
/// ```
pub struct FrameCompressor<R: Read, W: Write, M: Matcher> {
    uncompressed_data: Option<R>,
    compressed_data: Option<W>,
    compression_level: CompressionLevel,
    dictionary: Option<EncoderDictionary>,
    state: CompressState<M>,
    #[cfg(feature = "hash")]
    hasher: XxHash64,
}

pub(crate) struct FseTables {
    pub(crate) ll_default: FSETable,
    pub(crate) ll_previous: Option<FSETable>,
    pub(crate) ml_default: FSETable,
    pub(crate) ml_previous: Option<FSETable>,
    pub(crate) of_default: FSETable,
    pub(crate) of_previous: Option<FSETable>,
}

impl FseTables {
    pub fn new() -> Self {
        Self {
            ll_default: default_ll_table(),
            ll_previous: None,
            ml_default: default_ml_table(),
            ml_previous: None,
            of_default: default_of_table(),
            of_previous: None,
        }
    }
}

pub(crate) struct CompressState<M: Matcher> {
    pub(crate) matcher: M,
    pub(crate) last_huff_table: Option<crate::huff0::huff0_encoder::HuffmanTable>,
    pub(crate) fse_tables: FseTables,
}

impl<R: Read, W: Write> FrameCompressor<R, W, MatchGeneratorDriver> {
    /// Create a new `FrameCompressor`
    pub fn new(compression_level: CompressionLevel) -> Self {
        Self {
            uncompressed_data: None,
            compressed_data: None,
            compression_level,
            dictionary: None,
            state: CompressState {
                matcher: MatchGeneratorDriver::new(1024 * 128, 1),
                last_huff_table: None,
                fse_tables: FseTables::new(),
            },
            #[cfg(feature = "hash")]
            hasher: XxHash64::with_seed(0),
        }
    }
}

impl<R: Read, W: Write, M: Matcher> FrameCompressor<R, W, M> {
    /// Create a new `FrameCompressor` with a custom matching algorithm implementation
    pub fn new_with_matcher(matcher: M, compression_level: CompressionLevel) -> Self {
        Self {
            uncompressed_data: None,
            compressed_data: None,
            dictionary: None,
            state: CompressState {
                matcher,
                last_huff_table: None,
                fse_tables: FseTables::new(),
            },
            compression_level,
            #[cfg(feature = "hash")]
            hasher: XxHash64::with_seed(0),
        }
    }

    /// Before calling [FrameCompressor::compress] you need to set the source.
    ///
    /// This is the data that is compressed and written into the drain.
    pub fn set_source(&mut self, uncompressed_data: R) -> Option<R> {
        self.uncompressed_data.replace(uncompressed_data)
    }

    /// Before calling [FrameCompressor::compress] you need to set the drain.
    ///
    /// As the compressor compresses data, the drain serves as a place for the output to be writte.
    pub fn set_drain(&mut self, compressed_data: W) -> Option<W> {
        self.compressed_data.replace(compressed_data)
    }

    /// Set a dictionary to use during compression.
    ///
    /// When a dictionary is set, the dictionary ID is written into the frame header
    /// and the dictionary content is used as match history for finding back-references.
    /// The decoder must use the same dictionary (matched by ID) to decompress.
    pub fn set_dictionary(&mut self, dictionary: EncoderDictionary) -> Option<EncoderDictionary> {
        self.dictionary.replace(dictionary)
    }

    /// Clear the dictionary so subsequent frames are compressed without one.
    pub fn clear_dictionary(&mut self) -> Option<EncoderDictionary> {
        self.dictionary.take()
    }

    /// Compress the uncompressed data from the provided source as one Zstd frame and write it to the provided drain
    ///
    /// This will repeatedly call [Read::read] on the source to fill up blocks until the source returns 0 on the read call.
    /// Also [Write::write_all] will be called on the drain after each block has been encoded.
    ///
    /// To avoid endlessly encoding from a potentially endless source (like a network socket) you can use the
    /// [Read::take] function
    pub fn compress(&mut self) {
        // Clearing buffers to allow re-using of the compressor
        self.state.matcher.reset(self.compression_level);
        self.state.last_huff_table = None;
        #[cfg(feature = "hash")]
        {
            self.hasher = XxHash64::with_seed(0);
        }
        let source = self.uncompressed_data.as_mut().unwrap();
        let drain = self.compressed_data.as_mut().unwrap();
        let output: &mut Vec<u8> = &mut Vec::with_capacity(1024 * 130);

        let dict_id = self.dictionary.as_ref().map(|d| d.id as u64);

        let header = FrameHeader {
            frame_content_size: None,
            single_segment: false,
            content_checksum: cfg!(feature = "hash"),
            dictionary_id: dict_id,
            window_size: Some(self.state.matcher.window_size()),
        };
        header.serialize(output);

        let mut is_first_block = true;

        loop {
            let mut uncompressed_data = self.state.matcher.get_next_space();
            let mut read_bytes = 0;
            let last_block;
            'read_loop: loop {
                let new_bytes = source.read(&mut uncompressed_data[read_bytes..]).unwrap();
                if new_bytes == 0 {
                    last_block = true;
                    break 'read_loop;
                }
                read_bytes += new_bytes;
                if read_bytes == uncompressed_data.len() {
                    last_block = false;
                    break 'read_loop;
                }
            }
            uncompressed_data.resize(read_bytes, 0);
            #[cfg(feature = "hash")]
            self.hasher.write(&uncompressed_data);
            if uncompressed_data.is_empty() {
                let header = BlockHeader {
                    last_block: true,
                    block_type: crate::blocks::block::BlockType::Raw,
                    block_size: 0,
                };
                header.serialize(output);
                drain.write_all(output).unwrap();
                output.clear();
                break;
            }

            let dict_content = self.dictionary.as_ref().map(|d| d.content.as_slice());
            let dict_rep = self.dictionary.as_ref().map(|d| d.offset_hist);

            match self.compression_level {
                CompressionLevel::Uncompressed => {
                    let header = BlockHeader {
                        last_block,
                        block_type: crate::blocks::block::BlockType::Raw,
                        block_size: read_bytes.try_into().unwrap(),
                    };
                    header.serialize(output);
                    output.extend_from_slice(&uncompressed_data);
                }
                CompressionLevel::Fastest => {
                    compress_fastest(&mut self.state, last_block, uncompressed_data, output)
                }
                CompressionLevel::Default
                | CompressionLevel::Better
                | CompressionLevel::Best
                | CompressionLevel::Level(_) => {
                    let level = self.compression_level.to_level();
                    let dict_for_block = if is_first_block { dict_content } else { None };
                    let rep_for_block = if is_first_block { dict_rep } else { None };
                    super::levels::zstd_levels::compress_level(
                        &mut self.state,
                        last_block,
                        &uncompressed_data,
                        level,
                        None,
                        output,
                        dict_for_block,
                        rep_for_block,
                    );
                    self.state.matcher.commit_space(uncompressed_data);
                    self.state.matcher.skip_matching();
                }
            }
            is_first_block = false;
            drain.write_all(output).unwrap();
            output.clear();
            if last_block {
                break;
            }
        }

        // If the `hash` feature is enabled, then `content_checksum` is set to true in the header
        // and a 32 bit hash is written at the end of the data.
        #[cfg(feature = "hash")]
        {
            // Because we only have the data as a reader, we need to read all of it to calculate the checksum
            // Possible TODO: create a wrapper around self.uncompressed data that hashes the data as it's read?
            let content_checksum = self.hasher.finish();
            drain
                .write_all(&(content_checksum as u32).to_le_bytes())
                .unwrap();
        }
    }

    /// Get a mutable reference to the source
    pub fn source_mut(&mut self) -> Option<&mut R> {
        self.uncompressed_data.as_mut()
    }

    /// Get a mutable reference to the drain
    pub fn drain_mut(&mut self) -> Option<&mut W> {
        self.compressed_data.as_mut()
    }

    /// Get a reference to the source
    pub fn source(&self) -> Option<&R> {
        self.uncompressed_data.as_ref()
    }

    /// Get a reference to the drain
    pub fn drain(&self) -> Option<&W> {
        self.compressed_data.as_ref()
    }

    /// Retrieve the source
    pub fn take_source(&mut self) -> Option<R> {
        self.uncompressed_data.take()
    }

    /// Retrieve the drain
    pub fn take_drain(&mut self) -> Option<W> {
        self.compressed_data.take()
    }

    /// Before calling [FrameCompressor::compress] you can replace the matcher
    pub fn replace_matcher(&mut self, mut match_generator: M) -> M {
        core::mem::swap(&mut match_generator, &mut self.state.matcher);
        match_generator
    }

    /// Before calling [FrameCompressor::compress] you can replace the compression level
    pub fn set_compression_level(
        &mut self,
        compression_level: CompressionLevel,
    ) -> CompressionLevel {
        let old = self.compression_level;
        self.compression_level = compression_level;
        old
    }

    /// Get the current compression level
    pub fn compression_level(&self) -> CompressionLevel {
        self.compression_level
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::FrameCompressor;
    use crate::common::MAGIC_NUM;
    use crate::decoding::FrameDecoder;
    use alloc::vec::Vec;

    #[test]
    fn frame_starts_with_magic_num() {
        let mock_data = [1_u8, 2, 3].as_slice();
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data);
        compressor.set_drain(&mut output);

        compressor.compress();
        assert!(output.starts_with(&MAGIC_NUM.to_le_bytes()));
    }

    #[test]
    fn very_simple_raw_compress() {
        let mock_data = [1_u8, 2, 3].as_slice();
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data);
        compressor.set_drain(&mut output);

        compressor.compress();
    }

    #[test]
    fn very_simple_compress() {
        let mut mock_data = vec![0; 1 << 17];
        mock_data.extend(vec![1; (1 << 17) - 1]);
        mock_data.extend(vec![2; (1 << 18) - 1]);
        mock_data.extend(vec![2; 1 << 17]);
        mock_data.extend(vec![3; (1 << 17) - 1]);
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data.as_slice());
        compressor.set_drain(&mut output);

        compressor.compress();

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(mock_data.len());
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);

        let mut decoded = Vec::new();
        zstd::stream::copy_decode(output.as_slice(), &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);
    }

    #[test]
    fn rle_compress() {
        let mock_data = vec![0; 1 << 19];
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data.as_slice());
        compressor.set_drain(&mut output);

        compressor.compress();

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(mock_data.len());
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);
    }

    #[test]
    fn aaa_compress() {
        let mock_data = vec![0, 1, 3, 4, 5];
        let mut output: Vec<u8> = Vec::new();
        let mut compressor = FrameCompressor::new(super::CompressionLevel::Uncompressed);
        compressor.set_source(mock_data.as_slice());
        compressor.set_drain(&mut output);

        compressor.compress();

        let mut decoder = FrameDecoder::new();
        let mut decoded = Vec::with_capacity(mock_data.len());
        decoder.decode_all_to_vec(&output, &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);

        let mut decoded = Vec::new();
        zstd::stream::copy_decode(output.as_slice(), &mut decoded).unwrap();
        assert_eq!(mock_data, decoded);
    }

    #[cfg(feature = "std")]
    #[test]
    fn fuzz_targets() {
        use std::io::Read;
        fn decode_ruzstd(data: &mut dyn std::io::Read) -> Vec<u8> {
            let mut decoder = crate::decoding::StreamingDecoder::new(data).unwrap();
            let mut result: Vec<u8> = Vec::new();
            decoder.read_to_end(&mut result).expect("Decoding failed");
            result
        }

        fn decode_ruzstd_writer(mut data: impl Read) -> Vec<u8> {
            let mut decoder = crate::decoding::FrameDecoder::new();
            decoder.reset(&mut data).unwrap();
            let mut result = vec![];
            while !decoder.is_finished() || decoder.can_collect() > 0 {
                decoder
                    .decode_blocks(
                        &mut data,
                        crate::decoding::BlockDecodingStrategy::UptoBytes(1024 * 1024),
                    )
                    .unwrap();
                decoder.collect_to_writer(&mut result).unwrap();
            }
            result
        }

        fn encode_zstd(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
            zstd::stream::encode_all(std::io::Cursor::new(data), 3)
        }

        fn encode_ruzstd_uncompressed(data: &mut dyn std::io::Read) -> Vec<u8> {
            let mut input = Vec::new();
            data.read_to_end(&mut input).unwrap();

            crate::encoding::compress_to_vec(
                input.as_slice(),
                crate::encoding::CompressionLevel::Uncompressed,
            )
        }

        fn encode_ruzstd_compressed(data: &mut dyn std::io::Read) -> Vec<u8> {
            let mut input = Vec::new();
            data.read_to_end(&mut input).unwrap();

            crate::encoding::compress_to_vec(
                input.as_slice(),
                crate::encoding::CompressionLevel::Fastest,
            )
        }

        fn decode_zstd(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
            let mut output = Vec::new();
            zstd::stream::copy_decode(data, &mut output)?;
            Ok(output)
        }
        if std::fs::exists("fuzz/artifacts/interop").unwrap_or(false) {
            for file in std::fs::read_dir("fuzz/artifacts/interop").unwrap() {
                if file.as_ref().unwrap().file_type().unwrap().is_file() {
                    let data = std::fs::read(file.unwrap().path()).unwrap();
                    let data = data.as_slice();
                    // Decoding
                    let compressed = encode_zstd(data).unwrap();
                    let decoded = decode_ruzstd(&mut compressed.as_slice());
                    let decoded2 = decode_ruzstd_writer(&mut compressed.as_slice());
                    assert!(
                        decoded == data,
                        "Decoded data did not match the original input during decompression"
                    );
                    assert_eq!(
                        decoded2, data,
                        "Decoded data did not match the original input during decompression"
                    );

                    // Encoding
                    // Uncompressed encoding
                    let mut input = data;
                    let compressed = encode_ruzstd_uncompressed(&mut input);
                    let decoded = decode_zstd(&compressed).unwrap();
                    assert_eq!(
                        decoded, data,
                        "Decoded data did not match the original input during compression"
                    );
                    // Compressed encoding
                    let mut input = data;
                    let compressed = encode_ruzstd_compressed(&mut input);
                    let decoded = decode_zstd(&compressed).unwrap();
                    assert_eq!(
                        decoded, data,
                        "Decoded data did not match the original input during compression"
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Dictionary compression tests
    // -------------------------------------------------------------------

    fn make_test_dict_content() -> Vec<u8> {
        let mut content = Vec::new();
        for _ in 0..20 {
            content.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }
        content
    }

    fn make_raw_dict(id: u32, content: &[u8]) -> super::EncoderDictionary {
        super::EncoderDictionary::new_raw(id, content.to_vec())
    }

    #[test]
    fn dict_compress_default_roundtrip_our_decoder() {
        let dict_content = make_test_dict_content();
        let dict = make_raw_dict(42, &dict_content);

        let mut data = Vec::new();
        data.extend_from_slice(b"the quick brown fox jumps over ");
        data.extend_from_slice(b"something unique here ");
        data.extend_from_slice(b"the lazy dog the quick brown fox ");
        for _ in 0..10 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }

        let compressed = crate::encoding::compress_to_vec_with_dict(
            data.as_slice(),
            super::CompressionLevel::Default,
            &dict,
        );

        let mut decoder = FrameDecoder::new();
        let decoding_dict = crate::decoding::dictionary::Dictionary {
            id: 42,
            fse: crate::decoding::scratch::FSEScratch::new(),
            huf: crate::decoding::scratch::HuffmanScratch::new(),
            dict_content: dict_content.clone(),
            offset_hist: [1, 4, 8],
        };
        decoder.add_dict(decoding_dict).unwrap();
        let mut decoded = Vec::with_capacity(data.len() + 1024);
        decoder.decode_all_to_vec(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded, data, "our decoder: dictionary roundtrip failed");
    }

    #[test]
    fn dict_compress_c_zstd_can_decompress() {
        let phrase = b"the quick brown fox jumps over the lazy dog ";
        let mut samples: Vec<Vec<u8>> = Vec::new();
        for i in 0..100 {
            let mut sample = Vec::new();
            for _ in 0..(3 + i % 5) {
                sample.extend_from_slice(phrase);
            }
            sample.extend_from_slice(alloc::format!(" sample {i} ").as_bytes());
            samples.push(sample);
        }
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let trained_dict = zstd::dict::from_samples(&sample_refs, 4096)
            .expect("failed to train dictionary with C zstd");

        let encoder_dict = super::EncoderDictionary::parse(&trained_dict)
            .expect("failed to parse trained dictionary");

        let mut data = Vec::new();
        for _ in 0..15 {
            data.extend_from_slice(phrase);
        }
        data.extend_from_slice(b"and some unique trailing data!");

        let compressed = crate::encoding::compress_to_vec_with_dict(
            data.as_slice(),
            super::CompressionLevel::Default,
            &encoder_dict,
        );

        let mut decompressor = zstd::bulk::Decompressor::with_dictionary(&trained_dict)
            .expect("C zstd failed to create decompressor with dictionary");
        let decompressed = decompressor
            .decompress(&compressed, 1 << 20)
            .expect("C zstd failed to decompress with dictionary");

        assert_eq!(decompressed, data, "C zstd: dictionary decompression mismatch");
    }

    #[test]
    fn dict_compress_improves_ratio() {
        let dict_content = make_test_dict_content();
        let dict = make_raw_dict(99, &dict_content);

        let mut data = Vec::new();
        for _ in 0..5 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }

        let compressed_with = crate::encoding::compress_to_vec_with_dict(
            data.as_slice(), super::CompressionLevel::Default, &dict);
        let compressed_without = crate::encoding::compress_to_vec(
            data.as_slice(), super::CompressionLevel::Default);

        assert!(
            compressed_with.len() <= compressed_without.len(),
            "dict should not be worse: with={}, without={}",
            compressed_with.len(), compressed_without.len(),
        );
    }

    #[test]
    fn dict_compress_multiple_levels() {
        let dict_content = make_test_dict_content();
        let dict = make_raw_dict(1, &dict_content);

        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }
        data.extend_from_slice(b"unique tail bytes");

        for level in [
            super::CompressionLevel::Default,
            super::CompressionLevel::Better,
            super::CompressionLevel::Best,
            super::CompressionLevel::Level(5),
            super::CompressionLevel::Level(9),
        ] {
            let compressed = crate::encoding::compress_to_vec_with_dict(
                data.as_slice(), level, &dict);
            let mut decoder = FrameDecoder::new();
            let decoding_dict = crate::decoding::dictionary::Dictionary {
                id: 1,
                fse: crate::decoding::scratch::FSEScratch::new(),
                huf: crate::decoding::scratch::HuffmanScratch::new(),
                dict_content: dict_content.clone(),
                offset_hist: [1, 4, 8],
            };
            decoder.add_dict(decoding_dict).unwrap();
            let mut decoded = Vec::with_capacity(data.len() + 1024);
            decoder.decode_all_to_vec(&compressed, &mut decoded).unwrap();
            assert_eq!(decoded, data, "dictionary roundtrip failed at level {level:?}");
        }
    }

    #[test]
    fn dict_frame_header_contains_dict_id() {
        let dict = make_raw_dict(123, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let data = vec![0u8; 100];
        let compressed = crate::encoding::compress_to_vec_with_dict(
            data.as_slice(), super::CompressionLevel::Default, &dict);
        let (header, _) = crate::decoding::frame::read_frame_header(compressed.as_slice()).unwrap();
        assert_eq!(header.dictionary_id(), Some(123), "frame header should contain dictionary ID 123");
    }

    #[test]
    fn dict_compress_empty_input() {
        let dict = make_raw_dict(1, &make_test_dict_content());
        let data: &[u8] = &[];
        let compressed = crate::encoding::compress_to_vec_with_dict(
            data, super::CompressionLevel::Default, &dict);
        let mut decoder = FrameDecoder::new();
        let decoding_dict = crate::decoding::dictionary::Dictionary {
            id: 1,
            fse: crate::decoding::scratch::FSEScratch::new(),
            huf: crate::decoding::scratch::HuffmanScratch::new(),
            dict_content: make_test_dict_content(),
            offset_hist: [1, 4, 8],
        };
        decoder.add_dict(decoding_dict).unwrap();
        let mut decoded = Vec::with_capacity(1024);
        decoder.decode_all_to_vec(&compressed, &mut decoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn encoder_dictionary_new_raw() {
        let dict = super::EncoderDictionary::new_raw(42, vec![1, 2, 3]);
        assert_eq!(dict.id, 42);
        assert_eq!(dict.content, vec![1, 2, 3]);
        assert_eq!(dict.offset_hist, [1, 4, 8]);
    }

    #[test]
    #[should_panic(expected = "dictionary ID must be non-zero")]
    fn encoder_dictionary_rejects_zero_id() {
        super::EncoderDictionary::new_raw(0, vec![1, 2, 3]);
    }
}

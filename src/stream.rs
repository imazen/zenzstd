//! Streaming compression and decompression with an API matching the `zstd` crate.
//!
//! This module provides compatibility functions that mirror `zstd::stream`:
//!
//! ```rust,ignore
//! // Compress
//! let compressed = zenzstd::stream::encode_all(source, 3)?;
//! zenzstd::stream::copy_encode(source, dest, 3)?;
//!
//! // Decompress
//! let data = zenzstd::stream::decode_all(source)?;
//! zenzstd::stream::copy_decode(source, dest)?;
//! ```
//!
//! For the streaming `Encoder<W>` (impl Write) and `Decoder<R>` (impl Read),
//! use [`crate::encoding::StreamingEncoder`] and [`crate::decoding::StreamingDecoder`].

#[cfg(feature = "std")]
use std::io;

/// Compress all data from `source` at the given compression level.
///
/// Equivalent to `zstd::stream::encode_all`.
#[cfg(feature = "std")]
pub fn encode_all<R: io::Read>(mut source: R, level: i32) -> io::Result<alloc::vec::Vec<u8>> {
    let mut input = alloc::vec::Vec::new();
    source.read_to_end(&mut input)?;
    let level = crate::encoding::CompressionLevel::Level(level);
    Ok(crate::encoding::compress_to_vec(input.as_slice(), level))
}

/// Decompress all data from `source`.
///
/// Equivalent to `zstd::stream::decode_all`, but applies a 1 GiB output-size
/// cap to defend against decompression bombs (chained RLE blocks can expand
/// a few KB of compressed input into terabytes of output). Returns
/// [`io::ErrorKind::InvalidData`] if the cap would be exceeded.
///
/// For a custom cap, use [`decode_all_with_max`]. For trusted sources where
/// the input is known not to be hostile, use [`decode_all_unbounded`].
#[cfg(feature = "std")]
pub fn decode_all<R: io::Read>(source: R) -> io::Result<alloc::vec::Vec<u8>> {
    decode_all_with_max(source, Some(crate::decoding::DEFAULT_DECODE_OUTPUT_CAP))
}

/// Decompress all data from `source` with an explicit output-size cap.
///
/// `max_output_size` of `None` disables the cap (equivalent to
/// [`decode_all_unbounded`]). Otherwise reads past the cap return
/// [`io::ErrorKind::InvalidData`].
#[cfg(feature = "std")]
pub fn decode_all_with_max<R: io::Read>(
    source: R,
    max_output_size: Option<usize>,
) -> io::Result<alloc::vec::Vec<u8>> {
    let mut decoder = crate::decoding::StreamingDecoder::new(source)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, alloc::format!("{:?}", e)))?;
    decoder.set_max_output_size(max_output_size);
    let mut result = alloc::vec::Vec::new();
    io::Read::read_to_end(&mut decoder, &mut result)?;
    Ok(result)
}

/// Decompress all data from `source` with no output-size cap. Use only when
/// the source is fully trusted — a malicious zstd stream can expand to
/// arbitrarily large output and exhaust memory.
#[cfg(feature = "std")]
pub fn decode_all_unbounded<R: io::Read>(source: R) -> io::Result<alloc::vec::Vec<u8>> {
    decode_all_with_max(source, None)
}

/// Copy data from `source` to `destination`, compressing at the given level.
///
/// Equivalent to `zstd::stream::copy_encode`.
#[cfg(feature = "std")]
pub fn copy_encode<R: io::Read, W: io::Write>(
    mut source: R,
    destination: W,
    level: i32,
) -> io::Result<()> {
    let mut input = alloc::vec::Vec::new();
    source.read_to_end(&mut input)?;
    let mut encoder = crate::encoding::StreamingEncoder::new(
        destination,
        crate::encoding::CompressionLevel::Level(level),
    );
    io::Write::write_all(&mut encoder, &input)?;
    encoder.finish()?;
    Ok(())
}

/// Copy data from `source` to `destination`, decompressing.
///
/// Equivalent to `zstd::stream::copy_decode`, but applies a 1 GiB output-size
/// cap to defend against decompression bombs. Returns
/// [`io::ErrorKind::InvalidData`] if the cap would be exceeded.
///
/// For a custom cap, use [`copy_decode_with_max`]. For trusted sources, use
/// [`copy_decode_unbounded`].
#[cfg(feature = "std")]
pub fn copy_decode<R: io::Read, W: io::Write>(source: R, destination: W) -> io::Result<()> {
    copy_decode_with_max(
        source,
        destination,
        Some(crate::decoding::DEFAULT_DECODE_OUTPUT_CAP),
    )
}

/// Copy data from `source` to `destination`, decompressing with an explicit
/// output-size cap. `None` disables the cap (use only on trusted sources).
#[cfg(feature = "std")]
pub fn copy_decode_with_max<R: io::Read, W: io::Write>(
    source: R,
    mut destination: W,
    max_output_size: Option<usize>,
) -> io::Result<()> {
    let mut decoder = crate::decoding::StreamingDecoder::new(source)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, alloc::format!("{:?}", e)))?;
    decoder.set_max_output_size(max_output_size);
    io::copy(&mut decoder, &mut destination)?;
    Ok(())
}

/// Copy data from `source` to `destination`, decompressing without any
/// output-size cap. Use only when the source is fully trusted.
#[cfg(feature = "std")]
pub fn copy_decode_unbounded<R: io::Read, W: io::Write>(
    source: R,
    destination: W,
) -> io::Result<()> {
    copy_decode_with_max(source, destination, None)
}

/// Re-export the streaming encoder as `Encoder` for zstd crate compatibility.
#[cfg(feature = "std")]
pub use crate::encoding::StreamingEncoder as Encoder;

/// Re-export the streaming decoder as `Decoder` for zstd crate compatibility.
#[cfg(feature = "std")]
pub use crate::decoding::StreamingDecoder as Decoder;

#[cfg(all(test, feature = "std"))]
mod tests {
    extern crate std;
    use std::io::Cursor;

    #[test]
    fn encode_decode_roundtrip() {
        let data = b"Hello, zenzstd stream API! ".repeat(100);
        let compressed = super::encode_all(Cursor::new(&data), 3).unwrap();
        let decoded = super::decode_all(Cursor::new(&compressed)).unwrap();
        assert_eq!(data.as_slice(), decoded.as_slice());
    }

    #[test]
    fn copy_encode_decode_roundtrip() {
        let data = b"Copy encode/decode test data ".repeat(100);
        let mut compressed = std::vec::Vec::new();
        super::copy_encode(Cursor::new(&data), &mut compressed, 3).unwrap();
        let mut decoded = std::vec::Vec::new();
        super::copy_decode(Cursor::new(&compressed), &mut decoded).unwrap();
        assert_eq!(data.as_slice(), decoded.as_slice());
    }

    #[test]
    fn empty_roundtrip() {
        let data: &[u8] = &[];
        let compressed = super::encode_all(Cursor::new(data), 3).unwrap();
        let decoded = super::decode_all(Cursor::new(&compressed)).unwrap();
        assert_eq!(data, decoded.as_slice());
    }

    #[test]
    fn decode_all_caps_decompression_bomb() {
        // Build a real "bomb": 16 MiB of zeros compresses to a tiny RLE-heavy
        // frame. With a 1 MiB cap, decode_all_with_max must error rather than
        // allocate the full 16 MiB.
        let bomb_raw = std::vec![0u8; 16 * 1024 * 1024];
        let compressed = crate::encoding::compress_to_vec(
            bomb_raw.as_slice(),
            crate::encoding::CompressionLevel::Fastest,
        );
        // Sanity: compressed is small (< 1% of raw).
        assert!(
            compressed.len() < bomb_raw.len() / 100,
            "expected high compression ratio, got {} -> {}",
            compressed.len(),
            bomb_raw.len()
        );

        // With a 1 MiB cap, decoding 16 MiB of zeros must fail.
        let err = super::decode_all_with_max(Cursor::new(&compressed), Some(1024 * 1024))
            .expect_err("decode_all_with_max should reject output past the cap");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        // With no cap, the same payload decodes successfully.
        let decoded = super::decode_all_unbounded(Cursor::new(&compressed)).unwrap();
        assert_eq!(decoded.len(), bomb_raw.len());

        // The default decode_all (1 GiB cap) handles this 16 MiB payload fine.
        let decoded = super::decode_all(Cursor::new(&compressed)).unwrap();
        assert_eq!(decoded.len(), bomb_raw.len());
    }

    #[test]
    fn copy_decode_caps_decompression_bomb() {
        // Same scenario for copy_decode_with_max.
        let bomb_raw = std::vec![0u8; 16 * 1024 * 1024];
        let compressed = crate::encoding::compress_to_vec(
            bomb_raw.as_slice(),
            crate::encoding::CompressionLevel::Fastest,
        );

        let mut sink = std::vec::Vec::new();
        let err =
            super::copy_decode_with_max(Cursor::new(&compressed), &mut sink, Some(1024 * 1024))
                .expect_err("copy_decode_with_max should reject output past the cap");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        // Unbounded variant accepts the same payload.
        let mut sink2 = std::vec::Vec::new();
        super::copy_decode_unbounded(Cursor::new(&compressed), &mut sink2).unwrap();
        assert_eq!(sink2.len(), bomb_raw.len());
    }

    #[test]
    fn streaming_decoder_max_output_cap_at_exact_boundary() {
        // Compress exactly 4 KiB of repeating bytes, cap at 4 KiB — must succeed.
        let raw = std::vec![0xABu8; 4096];
        let compressed = crate::encoding::compress_to_vec(
            raw.as_slice(),
            crate::encoding::CompressionLevel::Fastest,
        );
        let mut decoder = crate::decoding::StreamingDecoder::new(Cursor::new(&compressed)).unwrap();
        decoder.set_max_output_size(Some(raw.len()));
        let mut decoded = std::vec::Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut decoded).unwrap();
        assert_eq!(decoded, raw);
        assert_eq!(decoder.max_output_size(), Some(raw.len()));

        // Cap one byte short — must error.
        let mut decoder = crate::decoding::StreamingDecoder::new(Cursor::new(&compressed)).unwrap();
        decoder.set_max_output_size(Some(raw.len() - 1));
        let mut decoded = std::vec::Vec::new();
        let err = std::io::Read::read_to_end(&mut decoder, &mut decoded).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn interop_with_c_zstd() {
        // Compress with us, decompress with C zstd
        let data = b"Interop test ".repeat(200);
        let compressed = super::encode_all(Cursor::new(&data), 3).unwrap();
        let mut decoded = std::vec::Vec::new();
        zstd::stream::copy_decode(compressed.as_slice(), &mut decoded).unwrap();
        assert_eq!(data.as_slice(), decoded.as_slice());

        // Compress with C zstd, decompress with us
        let c_compressed = zstd::stream::encode_all(Cursor::new(&data), 3).unwrap();
        let decoded = super::decode_all(Cursor::new(&c_compressed)).unwrap();
        assert_eq!(data.as_slice(), decoded.as_slice());
    }
}

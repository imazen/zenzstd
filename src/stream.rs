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
/// Equivalent to `zstd::stream::decode_all`.
#[cfg(feature = "std")]
pub fn decode_all<R: io::Read>(source: R) -> io::Result<alloc::vec::Vec<u8>> {
    let mut decoder = crate::decoding::StreamingDecoder::new(source)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, alloc::format!("{:?}", e)))?;
    let mut result = alloc::vec::Vec::new();
    io::Read::read_to_end(&mut decoder, &mut result)?;
    Ok(result)
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
/// Equivalent to `zstd::stream::copy_decode`.
#[cfg(feature = "std")]
pub fn copy_decode<R: io::Read, W: io::Write>(source: R, mut destination: W) -> io::Result<()> {
    let mut decoder = crate::decoding::StreamingDecoder::new(source)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, alloc::format!("{:?}", e)))?;
    io::copy(&mut decoder, &mut destination)?;
    Ok(())
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

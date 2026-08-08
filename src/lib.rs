//! A pure Rust implementation of the [Zstandard compression format](https://www.rfc-editor.org/rfc/rfc8878.pdf).
//!
//! `zenzstd` is a `#![forbid(unsafe_code)]` fork of [`ruzstd`](https://crates.io/crates/ruzstd),
//! extended with full compression support (levels 1-22), SIMD acceleration, and conformance testing.
//!
//! ## Quick start
//! One-shot helpers cover the common case — compress a buffer, or decompress a
//! buffer with a mandatory output ceiling that guards against decompression
//! bombs. See [`compress`] and [`decompress`].
//!
//! ```
//! use zenzstd::CompressionLevel;
//!
//! let data: &[u8] = b"the quick brown fox jumps over the lazy dog, again and again";
//! let compressed = zenzstd::compress(data, CompressionLevel::Default);
//! // Cap output at 64 KiB: ample for this payload, fatal for a bomb.
//! let restored = zenzstd::decompress(&compressed, 64 * 1024).unwrap();
//! assert_eq!(restored, data);
//! ```
//!
//! ## Decompression
//! The [decoding] module contains the code for decompression.
//! Decompression can be achieved by using the [`decoding::StreamingDecoder`]
//! or the more low-level [`decoding::FrameDecoder`]
//!
//! ## Compression
//! The [encoding] module contains the code for compression.
//! Compression can be achieved by using the [`encoding::compress`]/[`encoding::compress_to_vec`]
//! functions or [`encoding::FrameCompressor`]
//!
#![no_std]
// Safe by default. The `unsafe-decompress` and `unsafe-compress` features
// enable targeted unsafe optimizations in hot paths (unchecked indexing,
// raw pointer copies). All unsafe code is isolated in `unsafe_ops` modules
// and only active when the feature is explicitly opted into.
#![cfg_attr(
    not(any(feature = "unsafe-decompress", feature = "unsafe-compress")),
    forbid(unsafe_code)
)]
#![cfg_attr(
    any(feature = "unsafe-decompress", feature = "unsafe-compress"),
    deny(unsafe_code)
)]
#![deny(trivial_casts, trivial_numeric_casts)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[cfg(feature = "std")]
pub(crate) const VERBOSE: bool = false;

macro_rules! vprintln {
    ($($x:expr),*) => {
        #[cfg(feature = "std")]
        if crate::VERBOSE {
            std::println!($($x),*);
        }
    }
}

mod bit_io;
mod common;
pub mod decoding;
#[cfg(feature = "dict_builder")]
#[cfg_attr(docsrs, doc(cfg(feature = "dict_builder")))]
pub mod dictionary;
pub mod encoding;

/// Dev-only access to the match-length kernel, for `benches/kernel_tiers.rs`.
///
/// NOT public API. `count_match` is the LZ inner loop — it runs at every
/// candidate position during encode — and lives in a `pub(crate)` module. The
/// existing benches measure whole compression, which cannot reveal this kernel
/// being SLOWER than its own scalar fallback; that failure mode was real in
/// the 2026-07-28 aarch64 sweep (three zenfilters NEON kernels lost to their
/// scalar tier).
#[doc(hidden)]
pub mod __bench_match {
    pub fn count_match(a: &[u8], b: &[u8]) -> usize {
        crate::encoding::simd::count_match(a, b)
    }
}

pub(crate) mod blocks;

#[cfg(feature = "fuzz_exports")]
pub mod fse;
#[cfg(feature = "fuzz_exports")]
pub mod huff0;

#[cfg(not(feature = "fuzz_exports"))]
pub(crate) mod fse;
#[cfg(not(feature = "fuzz_exports"))]
pub(crate) mod huff0;

pub(crate) mod xxhash64;

/// Streaming compression/decompression API matching the `zstd` crate.
///
/// Provides `encode_all`, `decode_all`, `copy_encode`, `copy_decode`,
/// plus `Encoder` and `Decoder` type aliases.
#[cfg(feature = "std")]
pub mod stream;

#[cfg(feature = "std")]
pub mod io_std;

#[cfg(feature = "std")]
pub use io_std as io;

#[cfg(not(feature = "std"))]
pub mod io_nostd;

#[cfg(not(feature = "std"))]
pub use io_nostd as io;

/// The compression level / preset used by the one-shot [`compress`] helper.
///
/// Re-exported at the crate root from [`encoding::CompressionLevel`] so the
/// one-shot API is self-contained. [`CompressionLevel::Default`] (zstd level 3)
/// is a sensible default.
pub use encoding::CompressionLevel;

/// Compress a whole buffer into a standalone Zstandard frame in one call.
///
/// This is the one-shot convenience wrapper over [`encoding::compress_to_vec`]:
/// it takes the input as a byte slice and returns a freshly allocated `Vec`
/// holding the complete frame. Compression into a `Vec` cannot fail, so there is
/// no `Result` — but be aware allocation follows the crate's infallible policy
/// (it aborts on OOM rather than returning an error).
///
/// `level` selects the speed/ratio tradeoff — see [`CompressionLevel`].
/// [`CompressionLevel::Default`] (zstd level 3) is a sensible default. Prefer the
/// named presets or `Level(1..=15)`; levels 16-22 are experimental (see the
/// crate README).
///
/// For streaming input, compressing into a caller-owned buffer, dictionaries, or
/// cancellation, drive [`encoding::FrameCompressor`] /
/// [`encoding::StreamingEncoder`] (or the `std`-only [`crate::stream`] helpers)
/// directly.
///
/// ```
/// use zenzstd::CompressionLevel;
///
/// let data: &[u8] = b"the quick brown fox jumps over the lazy dog, again and again";
/// let compressed = zenzstd::compress(data, CompressionLevel::Default);
/// let restored = zenzstd::decompress(&compressed, 64 * 1024).unwrap();
/// assert_eq!(restored, data);
/// ```
pub fn compress(data: &[u8], level: CompressionLevel) -> alloc::vec::Vec<u8> {
    encoding::compress_to_vec(data, level)
}

/// Decompress a single Zstandard frame in one call, capping output at
/// `max_output_size` bytes.
///
/// A zstd frame carries an attacker-controlled content size, and a few KB of
/// crafted RLE blocks can expand to terabytes of output. `max_output_size` is a
/// hard ceiling on the decompressed length: if decoding would exceed it, this
/// returns [`DecompressError::OutputSizeExceeded`] — the same cap
/// [`decoding::StreamingDecoder`] enforces via
/// [`decoding::StreamingDecoder::set_max_output_size`] — instead of allocating
/// unbounded memory. Pass `usize::MAX` only for fully trusted input.
///
/// Expects the input to contain exactly one frame (see
/// [`decoding::StreamingDecoder`] for the multi-frame caveat). For incremental
/// decoding into a caller-owned buffer, drive [`decoding::FrameDecoder`] /
/// [`decoding::StreamingDecoder`], or use the `std`-only [`crate::stream`]
/// helpers.
///
/// ```
/// use zenzstd::CompressionLevel;
///
/// let data: &[u8] = b"the quick brown fox jumps over the lazy dog, again and again";
/// let compressed = zenzstd::compress(data, CompressionLevel::Default);
/// // Cap output at 64 KiB: ample for this payload, fatal for a bomb.
/// let restored = zenzstd::decompress(&compressed, 64 * 1024).unwrap();
/// assert_eq!(restored, data);
/// ```
pub fn decompress(
    data: &[u8],
    max_output_size: usize,
) -> Result<alloc::vec::Vec<u8>, DecompressError> {
    let mut decoder =
        decoding::StreamingDecoder::new(data).map_err(DecompressError::InvalidInput)?;
    decoder.set_max_output_size(Some(max_output_size));
    let mut out = alloc::vec::Vec::new();
    io::Read::read_to_end(&mut decoder, &mut out).map_err(|e| {
        // `StreamingDecoder::read` never writes past the configured cap (see
        // its `Read` impl), so the running output length can only reach
        // `max_output_size` exactly by hitting the cap — a genuine decode
        // failure (corrupt block, truncated stream, checksum mismatch, ...)
        // always surfaces strictly before that point. This lets us recover
        // which of the two happened from the one `io::Error` the streaming
        // reader returns, without needing it to carry a typed cause itself.
        if out.len() >= max_output_size {
            DecompressError::OutputSizeExceeded { max_output_size }
        } else {
            DecompressError::Decompressor(e)
        }
    })?;
    Ok(out)
}

/// Error returned by the one-shot [`decompress`] helper.
///
/// Distinguishes three distinct failure modes so callers can react
/// appropriately instead of pattern-matching an opaque I/O error message:
/// malformed input should simply be rejected, an exceeded output cap usually
/// means either raising the cap or treating the input as a probable
/// decompression bomb, and an underlying decompressor error means the frame
/// parsed fine but its block data was corrupt or truncated.
#[derive(Debug)]
#[non_exhaustive]
pub enum DecompressError {
    /// The input could not be parsed as a valid Zstandard frame — bad magic
    /// number, invalid frame descriptor, unsupported/oversized window, and so
    /// on. Surfaced by [`decoding::StreamingDecoder::new`].
    InvalidInput(decoding::errors::FrameDecoderError),
    /// Decoding would have produced more than `max_output_size` bytes of
    /// output. Raise the cap if the input is fully trusted, or treat this as
    /// a likely decompression bomb.
    OutputSizeExceeded {
        /// The cap that was passed to [`decompress`].
        max_output_size: usize,
    },
    /// The frame header parsed successfully but decoding its block data
    /// failed partway through — corrupt sequences, a truncated stream, a
    /// checksum mismatch, and so on.
    Decompressor(io::Error),
}

impl core::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecompressError::InvalidInput(e) => write!(f, "invalid zstd frame: {e}"),
            DecompressError::OutputSizeExceeded { max_output_size } => write!(
                f,
                "decompressed output exceeded max_output_size ({max_output_size} bytes)"
            ),
            DecompressError::Decompressor(e) => write!(f, "zstd decompression failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecompressError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecompressError::InvalidInput(e) => Some(e),
            DecompressError::OutputSizeExceeded { .. } => None,
            DecompressError::Decompressor(e) => Some(e),
        }
    }
}

mod tests;

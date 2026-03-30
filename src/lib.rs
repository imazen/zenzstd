//! A pure Rust implementation of the [Zstandard compression format](https://www.rfc-editor.org/rfc/rfc8878.pdf).
//!
//! `zenzstd` is a `#![forbid(unsafe_code)]` fork of [`ruzstd`](https://crates.io/crates/ruzstd),
//! extended with full compression support (levels 1-22), SIMD acceleration, and conformance testing.
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

#[cfg(feature = "std")]
pub mod io_std;

#[cfg(feature = "std")]
pub use io_std as io;

#[cfg(not(feature = "std"))]
pub mod io_nostd;

#[cfg(not(feature = "std"))]
pub use io_nostd as io;

mod tests;

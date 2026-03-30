//! Structures and utilities used for decoding zstd formatted data

pub mod errors;
mod frame_decoder;
mod streaming_decoder;

pub use frame_decoder::{BlockDecodingStrategy, FrameDecoder};
pub use streaming_decoder::StreamingDecoder;

pub(crate) mod block_decoder;
pub(crate) mod decode_buffer;
#[cfg(feature = "fuzz_exports")]
pub mod dictionary;
#[cfg(not(feature = "fuzz_exports"))]
pub(crate) mod dictionary;
mod flat_buffer;
pub(crate) mod frame;
pub(crate) mod literals_section_decoder;
mod ringbuffer;
#[allow(dead_code)]
#[cfg(feature = "fuzz_exports")]
pub mod scratch;
#[allow(dead_code)]
#[cfg(not(feature = "fuzz_exports"))]
pub(crate) mod scratch;
pub(crate) mod sequence_execution;
pub(crate) mod sequence_section_decoder;
#[cfg_attr(feature = "unsafe-decompress", allow(unsafe_code))]
pub(crate) mod unsafe_ops;

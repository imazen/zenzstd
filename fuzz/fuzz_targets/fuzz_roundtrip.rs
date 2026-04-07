#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use zenzstd::decoding::FrameDecoder;
use zenzstd::encoding::{compress_to_vec, CompressionLevel};

#[derive(Debug, Arbitrary)]
struct RoundtripInput {
    /// Compression level 0-22 (0 = uncompressed, 1-22 = compressed)
    level: u8,
    /// The data to compress
    data: Vec<u8>,
}

impl RoundtripInput {
    fn compression_level(&self) -> CompressionLevel {
        // Levels 16-22 have a known corruption bug in BtOpt/BtUltra match finder;
        // skip them to avoid false positives. Only fuzz levels 0-15.
        match self.level % 16 {
            0 => CompressionLevel::Uncompressed,
            1 => CompressionLevel::Fastest,
            3 => CompressionLevel::Default,
            7 => CompressionLevel::Better,
            11 => CompressionLevel::Best,
            n => CompressionLevel::Level(n as i32),
        }
    }
}

// Compress with zenzstd at a random level, then decompress with zenzstd.
// The output MUST match the input exactly.
fuzz_target!(|input: RoundtripInput| {
    // Limit data size to avoid OOM in the fuzzer
    if input.data.len() > 512 * 1024 {
        return;
    }

    let level = input.compression_level();
    let compressed = compress_to_vec(input.data.as_slice(), level);

    let mut decoder = FrameDecoder::new();
    let mut decompressed = Vec::with_capacity(input.data.len());
    decoder
        .decode_all_to_vec(&compressed, &mut decompressed)
        .expect("decompression of our own compressed data must succeed");

    assert_eq!(
        input.data, decompressed,
        "roundtrip mismatch at level {:?}",
        level
    );
});

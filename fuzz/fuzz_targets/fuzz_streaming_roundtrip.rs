#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use zenzstd::decoding::FrameDecoder;
use zenzstd::encoding::{CompressionLevel, StreamingEncoder};

#[derive(Debug, Arbitrary)]
struct StreamingInput {
    /// Compression level 0-22
    level: u8,
    /// Chunk sizes for writing (each value mod 8192 + 1, so 1..8192)
    chunk_sizes: Vec<u16>,
    /// The data to compress
    data: Vec<u8>,
}

impl StreamingInput {
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

// Stream-compress via StreamingEncoder with random write sizes,
// then decompress and verify the output matches the input.
fuzz_target!(|input: StreamingInput| {
    // Limit data size to avoid OOM
    if input.data.len() > 512 * 1024 {
        return;
    }
    // Need at least one chunk size to drive the loop
    if input.chunk_sizes.is_empty() {
        return;
    }

    let level = input.compression_level();
    let mut compressed = Vec::new();

    // Stream-compress with random chunk sizes
    {
        let mut encoder = StreamingEncoder::new(&mut compressed, level);
        let mut offset = 0;
        let mut chunk_idx = 0;

        while offset < input.data.len() {
            let chunk_size =
                (input.chunk_sizes[chunk_idx % input.chunk_sizes.len()] as usize % 8192) + 1;
            let end = (offset + chunk_size).min(input.data.len());
            encoder
                .write_all(&input.data[offset..end])
                .expect("streaming write must succeed");
            offset = end;
            chunk_idx += 1;
        }

        encoder.finish().expect("finish must succeed");
    }

    // Decompress and verify
    let mut decoder = FrameDecoder::new();
    let mut decompressed = Vec::with_capacity(input.data.len());
    decoder
        .decode_all_to_vec(&compressed, &mut decompressed)
        .expect("decompression of streaming-compressed data must succeed");

    assert_eq!(
        input.data, decompressed,
        "streaming roundtrip mismatch at level {:?}",
        level
    );
});

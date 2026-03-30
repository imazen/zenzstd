#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Read;
use zenzstd::decoding::{BlockDecodingStrategy, FrameDecoder};

// Feed arbitrary bytes to the decoder. It MUST NOT panic on any input.
// Errors are fine, panics are bugs.
fuzz_target!(|data: &[u8]| {
    // Path 1: FrameDecoder with decode_all
    {
        let mut decoder = FrameDecoder::new();
        let mut output = vec![0u8; 1024 * 1024]; // 1 MiB cap
        let _ = decoder.decode_all(data, &mut output);
    }

    // Path 2: FrameDecoder with decode_all_to_vec
    {
        let mut decoder = FrameDecoder::new();
        let mut output = Vec::with_capacity(64 * 1024);
        let _ = decoder.decode_all_to_vec(data, &mut output);
    }

    // Path 3: FrameDecoder with block-at-a-time decoding
    {
        let mut decoder = FrameDecoder::new();
        let mut cursor = data;
        if decoder.reset(&mut cursor).is_ok() {
            let _ = decoder.decode_blocks(&mut cursor, BlockDecodingStrategy::All);
        }
    }

    // Path 4: StreamingDecoder
    {
        let mut cursor = data;
        if let Ok(mut stream) = zenzstd::decoding::StreamingDecoder::new(&mut cursor) {
            let mut output = Vec::new();
            let _ = stream.read_to_end(&mut output);
        }
    }

    // Path 5: decode_from_to (incremental)
    {
        let mut decoder = FrameDecoder::new();
        let mut cursor = data;
        if decoder.reset(&mut cursor).is_ok() {
            let mut output = vec![0u8; 256 * 1024];
            let remaining = &data[..]; // feed all remaining bytes
            let _ = decoder.decode_from_to(remaining, &mut output);
        }
    }
});

//! Focused mixed-data decode loop for profiling with valgrind/callgrind.
//!
//! Usage: cargo build --release --example decode_mixed && valgrind --tool=callgrind target/release/examples/decode_mixed

use std::io::Cursor;

fn make_mixed(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut i = 0u32;
    while data.len() < size {
        if i % 100 < 50 {
            data.push(b'A' + (i % 26) as u8);
        } else {
            data.push(((i.wrapping_mul(2654435761) >> 16) & 0xFF) as u8);
        }
        i += 1;
    }
    data
}

fn main() {
    let mixed_100k = make_mixed(100_000);
    // Compress with C zstd for a fair decode comparison
    let compressed = zstd::stream::encode_all(Cursor::new(&mixed_100k), 3).unwrap();

    eprintln!(
        "Compressed {} -> {} bytes ({:.1}x)",
        mixed_100k.len(),
        compressed.len(),
        mixed_100k.len() as f64 / compressed.len() as f64
    );

    // Decode 1000 times, reusing decoder and target buffer
    let iterations = 1000;
    let mut total_bytes = 0usize;
    let mut dec = zenzstd::decoding::FrameDecoder::new();
    let mut target = vec![0u8; 100_000 + 4096];
    for _ in 0..iterations {
        let n = dec.decode_all(&compressed, &mut target).unwrap();
        total_bytes += n;
    }
    eprintln!(
        "Decoded {} iterations, {} total bytes",
        iterations, total_bytes
    );
}

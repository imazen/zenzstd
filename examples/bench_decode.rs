use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;

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
    let data = make_mixed(100_000);
    // Compress with C zstd at level 3
    let compressed = zstd::stream::encode_all(Cursor::new(&data), 3).unwrap();
    eprintln!("compressed {} -> {} bytes", data.len(), compressed.len());

    // Warmup
    for _ in 0..50 {
        let mut dec = zenzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; data.len() + 4096];
        dec.decode_all(&compressed, &mut target).unwrap();
        black_box(&target);
    }

    let iters = 500;
    let start = Instant::now();
    for _ in 0..iters {
        let mut dec = zenzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; data.len() + 4096];
        dec.decode_all(&compressed, &mut target).unwrap();
        black_box(&target);
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    let mbps = data.len() as f64 / elapsed / 1_000_000.0;
    eprintln!(
        "mixed_100k decode: {:.0} MB/s ({:.1} µs/call)",
        mbps,
        elapsed * 1e6
    );
}

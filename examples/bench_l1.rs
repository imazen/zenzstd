use std::io::Cursor;
use std::time::Instant;
use std::hint::black_box;

fn make_mixed(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut i = 0u32;
    while data.len() < size {
        if i % 100 < 50 { data.push(b'A' + (i % 26) as u8); }
        else { data.push(((i.wrapping_mul(2654435761) >> 16) & 0xFF) as u8); }
        i += 1;
    }
    data
}

fn main() {
    let data = make_mixed(100_000);
    // Warmup
    for _ in 0..10 {
        black_box(zenzstd::encoding::compress_to_vec(
            Cursor::new(&data), zenzstd::encoding::CompressionLevel::Level(1)));
    }
    // Measure
    let start = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        black_box(zenzstd::encoding::compress_to_vec(
            Cursor::new(&data), zenzstd::encoding::CompressionLevel::Level(1)));
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    let mbps = data.len() as f64 / elapsed / 1_000_000.0;
    eprintln!("L1 mixed_100k: {:.1} MB/s ({:.1} µs/call)", mbps, elapsed * 1e6);
}

//! Decode benchmark focused on L3 (the default level) across data types.

use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;

fn make_text(size: usize) -> Vec<u8> {
    let phrase = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let n = (size - data.len()).min(phrase.len());
        data.extend_from_slice(&phrase[..n]);
    }
    data
}

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

fn make_random(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut s = 0x12345678u64;
    for _ in 0..size {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push((s >> 33) as u8);
    }
    data
}

fn bench_decode(name: &str, data: &[u8], level: i32) {
    // Compress with C zstd
    let c_compressed = zstd::stream::encode_all(Cursor::new(data), level).unwrap();
    // Compress with zenzstd
    let zen_compressed = zenzstd::encoding::compress_to_vec(
        Cursor::new(data),
        zenzstd::encoding::CompressionLevel::Level(level),
    );

    let warmup = 50;
    let iters = 500;

    // Benchmark: zenzstd decoding C-compressed data
    for _ in 0..warmup {
        let mut d = zenzstd::decoding::FrameDecoder::new();
        let mut t = vec![0u8; data.len() + 4096];
        d.decode_all(&c_compressed, &mut t).unwrap();
        black_box(&t);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut d = zenzstd::decoding::FrameDecoder::new();
        let mut t = vec![0u8; data.len() + 4096];
        d.decode_all(&c_compressed, &mut t).unwrap();
        black_box(&t);
    }
    let zen_time = start.elapsed().as_secs_f64() / iters as f64;
    let zen_mbps = data.len() as f64 / zen_time / 1_000_000.0;

    // Benchmark: C zstd decoding C-compressed data
    for _ in 0..warmup {
        let mut out = Vec::with_capacity(data.len());
        zstd::stream::copy_decode(c_compressed.as_slice(), &mut out).unwrap();
        black_box(&out);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut out = Vec::with_capacity(data.len());
        zstd::stream::copy_decode(c_compressed.as_slice(), &mut out).unwrap();
        black_box(&out);
    }
    let c_time = start.elapsed().as_secs_f64() / iters as f64;
    let c_mbps = data.len() as f64 / c_time / 1_000_000.0;

    let gap = c_mbps / zen_mbps;
    eprintln!(
        "{:<20} L{:<2}  {:>6} -> {:>6} bytes  zen:{:>7.0} MB/s  c:{:>7.0} MB/s  gap:{:.2}x  zen_sz:{:>6}",
        name, level,
        data.len(), c_compressed.len(),
        zen_mbps, c_mbps, gap,
        zen_compressed.len(),
    );
}

fn main() {
    let text_100k = make_text(100_000);
    let mixed_100k = make_mixed(100_000);
    let random_100k = make_random(100_000);
    let mixed_1m = make_mixed(1_000_000);

    eprintln!("=== L3 (default) decode benchmark ===");
    eprintln!("{:<20} {:>3}  {:>15}  {:>14}  {:>14}  {:>5}  {:>10}",
        "DATASET", "LVL", "SIZE", "DEC_ZEN", "DEC_C", "GAP", "ZEN_SZ");
    eprintln!("{}", "-".repeat(95));

    bench_decode("text_100k", &text_100k, 3);
    bench_decode("mixed_100k", &mixed_100k, 3);
    bench_decode("random_100k", &random_100k, 3);
    bench_decode("mixed_1m", &mixed_1m, 3);

    eprintln!();
    eprintln!("=== Mixed 100k across levels ===");
    for level in [1, 3, 5, 7, 9, 11] {
        bench_decode("mixed_100k", &mixed_100k, level);
    }
}

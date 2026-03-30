//! Compare zenzstd vs C zstd: compression ratio and speed at all levels.

use std::io::Cursor;
use std::time::Instant;

fn make_text_data(size: usize) -> Vec<u8> {
    let phrase = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(phrase.len());
        data.extend_from_slice(&phrase[..chunk]);
    }
    data
}

fn make_mixed_data(size: usize) -> Vec<u8> {
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

fn make_random_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut state = 0x12345678u64;
    for _ in 0..size {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push((state >> 33) as u8);
    }
    data
}

fn bench_compress(data: &[u8], level: i32, iterations: u32) -> (usize, f64, usize, f64) {
    // zenzstd
    let start = Instant::now();
    let mut zen_size = 0;
    for _ in 0..iterations {
        let compressed = zenzstd::encoding::compress_to_vec(
            Cursor::new(data),
            zenzstd::encoding::CompressionLevel::Level(level),
        );
        zen_size = compressed.len();
    }
    let zen_time = start.elapsed().as_secs_f64() / iterations as f64;

    // C zstd
    let start = Instant::now();
    let mut c_size = 0;
    for _ in 0..iterations {
        let compressed = zstd::stream::encode_all(Cursor::new(data), level).unwrap();
        c_size = compressed.len();
    }
    let c_time = start.elapsed().as_secs_f64() / iterations as f64;

    (zen_size, zen_time, c_size, c_time)
}

fn bench_decompress(data: &[u8], iterations: u32) -> (f64, f64) {
    // Compress with C zstd at level 3
    let compressed = zstd::stream::encode_all(Cursor::new(data), 3).unwrap();

    // zenzstd decode
    let start = Instant::now();
    for _ in 0..iterations {
        let mut decoder = zenzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; data.len() + 4096];
        decoder.decode_all(&compressed, &mut target).unwrap();
    }
    let zen_time = start.elapsed().as_secs_f64() / iterations as f64;

    // C zstd decode
    let start = Instant::now();
    for _ in 0..iterations {
        let mut output = Vec::with_capacity(data.len());
        zstd::stream::copy_decode(compressed.as_slice(), &mut output).unwrap();
    }
    let c_time = start.elapsed().as_secs_f64() / iterations as f64;

    (zen_time, c_time)
}

fn main() {
    let sizes = [(100_000, "100KB"), (1_000_000, "1MB")];
    let datasets: Vec<(&str, Vec<u8>)> = sizes
        .iter()
        .flat_map(|&(size, label)| {
            vec![
                (format!("text_{}", label).leak() as &str, make_text_data(size)),
                (format!("mixed_{}", label).leak() as &str, make_mixed_data(size)),
                (format!("random_{}", label).leak() as &str, make_random_data(size)),
            ]
        })
        .collect();

    println!("=== COMPRESSION RATIO & SPEED ===");
    println!("{:<15} {:>5} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "Dataset", "Level", "Zen Size", "C Size", "Ratio", "Zen MB/s", "C MB/s", "Speedup", "ZenRatio");

    for (name, data) in &datasets {
        let iters = if data.len() >= 1_000_000 { 3 } else { 10 };
        for level in [1, 3, 5, 7, 9, 11, 15, 19, 22] {
            let (zen_size, zen_time, c_size, c_time) = bench_compress(data, level, iters);
            let zen_ratio = data.len() as f64 / zen_size as f64;
            let c_ratio = data.len() as f64 / c_size as f64;
            let size_ratio = zen_size as f64 / c_size as f64;
            let zen_mbps = data.len() as f64 / zen_time / 1_000_000.0;
            let c_mbps = data.len() as f64 / c_time / 1_000_000.0;
            let speedup = c_mbps / zen_mbps;

            println!("{:<15} {:>5} {:>10} {:>10} {:>7.2}x {:>9.1} {:>9.1} {:>7.1}x {:>7.1}x",
                name, level, zen_size, c_size, size_ratio, zen_mbps, c_mbps, speedup, zen_ratio);
        }
        println!();
    }

    println!("\n=== DECOMPRESSION SPEED ===");
    println!("{:<15} {:>10} {:>10} {:>8}",
        "Dataset", "Zen MB/s", "C MB/s", "Speedup");

    for (name, data) in &datasets {
        let iters = if data.len() >= 1_000_000 { 10 } else { 50 };
        let (zen_time, c_time) = bench_decompress(data, iters);
        let zen_mbps = data.len() as f64 / zen_time / 1_000_000.0;
        let c_mbps = data.len() as f64 / c_time / 1_000_000.0;
        let speedup = c_mbps / zen_mbps;

        println!("{:<15} {:>9.1} {:>9.1} {:>7.1}x",
            name, zen_mbps, c_mbps, speedup);
    }
}

//! Compare zenzstd vs C zstd: compression ratio, encode speed, and decode speed at all levels.
//!
//! Every row shows the full picture: size, ratio, encode throughput, decode throughput.

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
        if i % 100 < 50 {
            data.push(b'A' + (i % 26) as u8);
        } else {
            data.push(((i.wrapping_mul(2654435761) >> 16) & 0xFF) as u8);
        }
        i += 1;
    }
    data
}

fn make_random(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut s = 0x12345678u64;
    for _ in 0..size {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push((s >> 33) as u8);
    }
    data
}

fn mb_per_sec(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1_000_000.0
}

/// Measure encode speed, returning (compressed_bytes, seconds_per_call).
fn bench_encode_zen(data: &[u8], level: i32, iters: u32) -> (Vec<u8>, f64) {
    let mut compressed = Vec::new();
    let start = Instant::now();
    for _ in 0..iters {
        compressed = zenzstd::encoding::compress_to_vec(
            Cursor::new(data),
            zenzstd::encoding::CompressionLevel::Level(level),
        );
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    (compressed, elapsed)
}

fn bench_encode_c(data: &[u8], level: i32, iters: u32) -> (Vec<u8>, f64) {
    let mut compressed = Vec::new();
    let start = Instant::now();
    for _ in 0..iters {
        compressed = zstd::stream::encode_all(Cursor::new(data), level).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    (compressed, elapsed)
}

/// Measure decode speed of a pre-compressed buffer, returning seconds_per_call.
fn bench_decode_zen(compressed: &[u8], original_len: usize, iters: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let mut dec = zenzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; original_len + 4096];
        dec.decode_all(compressed, &mut target).unwrap();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_decode_c(compressed: &[u8], original_len: usize, iters: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let mut out = Vec::with_capacity(original_len);
        zstd::stream::copy_decode(compressed, &mut out).unwrap();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_decode_ruzstd(compressed: &[u8], original_len: usize, iters: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let mut dec = ruzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; original_len + 4096];
        dec.decode_all(compressed, &mut target).unwrap();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("text_100k", make_text(100_000)),
        ("mixed_100k", make_mixed(100_000)),
        ("random_100k", make_random(100_000)),
        ("text_1m", make_text(1_000_000)),
        ("mixed_1m", make_mixed(1_000_000)),
    ];

    let levels = [1, 3, 5, 7, 9, 11, 15, 19, 22];

    println!(
        "{:<12} {:>3}  {:>7} {:>7} {:>6}  {:>8} {:>8} {:>6}  {:>8} {:>8} {:>8} {:>6} {:>6}",
        "DATASET", "LVL",
        "ZEN_SZ", "C_SZ", "ZN/C",
        "ENC_ZEN", "ENC_C", "E_GAP",
        "DEC_ZEN", "DEC_RUZ", "DEC_C",
        "vs_RUZ", "vs_C",
    );
    println!("{}", "-".repeat(140));

    for (name, data) in &datasets {
        let enc_iters = if data.len() >= 1_000_000 { 3 } else { 10 };
        let dec_iters = if data.len() >= 1_000_000 { 5 } else { 30 };

        for &level in &levels {
            // Encode both
            let (zen_compressed, zen_enc_time) = bench_encode_zen(data, level, enc_iters);
            let (c_compressed, c_enc_time) = bench_encode_c(data, level, enc_iters);

            // Decode all three: zenzstd, ruzstd (upstream), C zstd
            // All decode C-compressed data for fair comparison
            let zen_dec_time = bench_decode_zen(&c_compressed, data.len(), dec_iters);
            let ruz_dec_time = bench_decode_ruzstd(&c_compressed, data.len(), dec_iters);
            let c_dec_time = bench_decode_c(&c_compressed, data.len(), dec_iters);

            let zen_sz = zen_compressed.len();
            let c_sz = c_compressed.len();
            let size_ratio = zen_sz as f64 / c_sz as f64;

            let enc_zen = mb_per_sec(data.len(), zen_enc_time);
            let enc_c = mb_per_sec(data.len(), c_enc_time);
            let enc_gap = enc_c / enc_zen;

            let dec_zen = mb_per_sec(data.len(), zen_dec_time);
            let dec_ruz = mb_per_sec(data.len(), ruz_dec_time);
            let dec_c = mb_per_sec(data.len(), c_dec_time);
            let vs_ruz = dec_zen / dec_ruz; // >1 = we're faster than ruzstd
            let vs_c = dec_c / dec_zen; // >1 = C is faster

            println!(
                "{:<12} {:>3}  {:>7} {:>7} {:>5.2}x  {:>7.0}M {:>7.0}M {:>5.1}x  {:>7.0}M {:>7.0}M {:>7.0}M {:>5.1}x {:>5.1}x",
                name, level,
                zen_sz, c_sz, size_ratio,
                enc_zen, enc_c, enc_gap,
                dec_zen, dec_ruz, dec_c,
                vs_ruz, vs_c,
            );
        }
        println!();
    }
}

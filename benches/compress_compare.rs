//! Benchmark zenzstd vs C zstd: compression speed, decompression speed, and ratio.
//!
//! Run:   cargo bench --bench compress_compare
//! Save:  cargo bench --bench compress_compare -- --save-baseline main
//! Check: cargo bench --bench compress_compare -- --baseline main --max-regression 5

use std::io::Cursor;
use zenbench::prelude::*;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

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
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push((s >> 33) as u8);
    }
    data
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn compress_benchmarks(suite: &mut Suite) {
    let text_100k = make_text(100_000);
    let mixed_100k = make_mixed(100_000);

    // --- Text 100KB compression at key levels ---
    suite.group("compress_text_100k", |g| {
        g.throughput(Throughput::Bytes(100_000));

        for level in [1, 3, 7, 11, 19] {
            let data = text_100k.clone();
            g.bench(format!("zenzstd_L{level}"), move |b| {
                let data = data.clone();
                b.iter(|| {
                    black_box(zenzstd::encoding::compress_to_vec(
                        Cursor::new(&data),
                        zenzstd::encoding::CompressionLevel::Level(level),
                    ))
                })
            });

            let data = text_100k.clone();
            g.bench(format!("c_zstd_L{level}"), move |b| {
                let data = data.clone();
                b.iter(|| {
                    black_box(zstd::stream::encode_all(Cursor::new(&data), level).unwrap())
                })
            });
        }
    });

    // --- Mixed 100KB compression ---
    suite.group("compress_mixed_100k", |g| {
        g.throughput(Throughput::Bytes(100_000));

        for level in [1, 3, 7, 11] {
            let data = mixed_100k.clone();
            g.bench(format!("zenzstd_L{level}"), move |b| {
                let data = data.clone();
                b.iter(|| {
                    black_box(zenzstd::encoding::compress_to_vec(
                        Cursor::new(&data),
                        zenzstd::encoding::CompressionLevel::Level(level),
                    ))
                })
            });

            let data = mixed_100k.clone();
            g.bench(format!("c_zstd_L{level}"), move |b| {
                let data = data.clone();
                b.iter(|| {
                    black_box(zstd::stream::encode_all(Cursor::new(&data), level).unwrap())
                })
            });
        }
    });
}

fn decompress_benchmarks(suite: &mut Suite) {
    let text_100k = make_text(100_000);

    // Pre-compress
    let c_compressed = zstd::stream::encode_all(Cursor::new(&text_100k), 3).unwrap();
    let zen_compressed = zenzstd::encoding::compress_to_vec(
        Cursor::new(&text_100k),
        zenzstd::encoding::CompressionLevel::Level(3),
    );

    suite.group("decompress_text_100k", |g| {
        g.throughput(Throughput::Bytes(100_000));

        let cc = c_compressed.clone();
        g.bench("zenzstd_decode", move |b| {
            let cc = cc.clone();
            b.iter(move || {
                let mut dec = zenzstd::decoding::FrameDecoder::new();
                let mut target = vec![0u8; 100_000 + 4096];
                dec.decode_all(&cc, &mut target).unwrap();
                black_box(target)
            })
        });

        let cc = c_compressed.clone();
        g.bench("c_zstd_decode", move |b| {
            let cc = cc.clone();
            b.iter(move || {
                let mut out = Vec::with_capacity(100_000);
                zstd::stream::copy_decode(cc.as_slice(), &mut out).unwrap();
                black_box(out)
            })
        });
    });
}

fn ratio_report(suite: &mut Suite) {
    // Print ratio comparison — not timed, just informational
    suite.group("ratio_report", |g| {
        g.config().max_rounds(1);

        let datasets: Vec<(&str, Vec<u8>)> = vec![
            ("text_100k", make_text(100_000)),
            ("mixed_100k", make_mixed(100_000)),
            ("random_100k", make_random(100_000)),
            ("text_1m", make_text(1_000_000)),
        ];

        for (name, data) in &datasets {
            for level in [1, 3, 7, 11, 19, 22] {
                let zen = zenzstd::encoding::compress_to_vec(
                    Cursor::new(data),
                    zenzstd::encoding::CompressionLevel::Level(level),
                );
                let c = zstd::stream::encode_all(Cursor::new(data), level).unwrap();

                eprintln!(
                    "RATIO {:<12} L{:>2}: zen={:>7} ({:>5.1}x)  c={:>7} ({:>5.1}x)  zen/c={:.2}",
                    name,
                    level,
                    zen.len(),
                    data.len() as f64 / zen.len() as f64,
                    c.len(),
                    data.len() as f64 / c.len() as f64,
                    zen.len() as f64 / c.len() as f64,
                );
            }
        }

        // Dummy bench so the group isn't empty
        g.bench("noop", |b| b.iter(|| black_box(42)));
    });
}

zenbench::main!(compress_benchmarks, decompress_benchmarks, ratio_report);

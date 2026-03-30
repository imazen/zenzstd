use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::io::Cursor;

/// Generate test data: repetitive text (compresses well)
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

/// Generate test data: mixed (moderate compressibility)
fn make_mixed_data(size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut i = 0u32;
    while data.len() < size {
        // Mix of patterns: some repetitive, some pseudo-random
        if i % 100 < 50 {
            data.push(b'A' + (i % 26) as u8);
        } else {
            data.push(((i.wrapping_mul(2654435761) >> 16) & 0xFF) as u8);
        }
        i += 1;
    }
    data
}

fn bench_compress(c: &mut Criterion) {
    let text_100k = make_text_data(100_000);
    let mixed_100k = make_mixed_data(100_000);

    let mut group = c.benchmark_group("compress_text_100k");
    group.throughput(criterion::Throughput::Bytes(text_100k.len() as u64));

    for level in [1, 3, 5, 7, 9, 11, 15, 19] {
        group.bench_with_input(BenchmarkId::new("zenzstd", level), &level, |b, &level| {
            b.iter(|| {
                zenzstd::encoding::compress_to_vec(
                    Cursor::new(&text_100k),
                    zenzstd::encoding::CompressionLevel::Level(level),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("c_zstd", level), &level, |b, &level| {
            b.iter(|| zstd::stream::encode_all(Cursor::new(&text_100k), level).unwrap());
        });
    }
    group.finish();

    let mut group = c.benchmark_group("compress_mixed_100k");
    group.throughput(criterion::Throughput::Bytes(mixed_100k.len() as u64));

    for level in [1, 3, 7, 11, 19] {
        group.bench_with_input(BenchmarkId::new("zenzstd", level), &level, |b, &level| {
            b.iter(|| {
                zenzstd::encoding::compress_to_vec(
                    Cursor::new(&mixed_100k),
                    zenzstd::encoding::CompressionLevel::Level(level),
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("c_zstd", level), &level, |b, &level| {
            b.iter(|| zstd::stream::encode_all(Cursor::new(&mixed_100k), level).unwrap());
        });
    }
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let text_100k = make_text_data(100_000);

    // Pre-compress with C zstd at level 3
    let compressed_c = zstd::stream::encode_all(Cursor::new(&text_100k), 3).unwrap();
    // Pre-compress with zenzstd at level 3
    let compressed_zen = zenzstd::encoding::compress_to_vec(
        Cursor::new(&text_100k),
        zenzstd::encoding::CompressionLevel::Level(3),
    );

    let mut group = c.benchmark_group("decompress_text_100k");
    group.throughput(criterion::Throughput::Bytes(text_100k.len() as u64));

    group.bench_function("zenzstd_decode_c_compressed", |b| {
        let mut decoder = zenzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; text_100k.len() + 4096];
        b.iter(|| {
            decoder.decode_all(&compressed_c, &mut target).unwrap();
        });
    });

    group.bench_function("zenzstd_decode_zen_compressed", |b| {
        let mut decoder = zenzstd::decoding::FrameDecoder::new();
        let mut target = vec![0u8; text_100k.len() + 4096];
        b.iter(|| {
            decoder.decode_all(&compressed_zen, &mut target).unwrap();
        });
    });

    group.bench_function("c_zstd_decode", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            zstd::stream::copy_decode(compressed_c.as_slice(), &mut output).unwrap();
        });
    });

    group.finish();
}

fn bench_ratio(c: &mut Criterion) {
    let text_100k = make_text_data(100_000);
    let mixed_100k = make_mixed_data(100_000);

    let mut group = c.benchmark_group("ratio_comparison");
    // One iteration per measurement — we just want ratio, not speed
    group.sample_size(10);

    for (name, data) in [("text", &text_100k), ("mixed", &mixed_100k)] {
        for level in [1, 3, 7, 11, 19, 22] {
            let zen_compressed = zenzstd::encoding::compress_to_vec(
                Cursor::new(data),
                zenzstd::encoding::CompressionLevel::Level(level),
            );
            let c_compressed = zstd::stream::encode_all(Cursor::new(data), level).unwrap();

            println!(
                "{}@L{}: zenzstd={} bytes ({:.1}x), c_zstd={} bytes ({:.1}x), ratio={:.2}",
                name,
                level,
                zen_compressed.len(),
                data.len() as f64 / zen_compressed.len() as f64,
                c_compressed.len(),
                data.len() as f64 / c_compressed.len() as f64,
                zen_compressed.len() as f64 / c_compressed.len() as f64,
            );

            group.bench_function(format!("{}_L{}_{}", name, level, "noop"), |b| b.iter(|| {}));
        }
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress, bench_ratio);
criterion_main!(benches);

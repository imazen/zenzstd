//! End-to-end compression, to validate that the `count_match` dispatch change
//! is a real win at the pipeline level and not just a microbenchmark artifact.
//!
//! A kernel-level win can evaporate end-to-end (it did for a zensim blur
//! rewrite earlier in this sweep: 1.46x on the stage, -1.4% on the pipeline).
//! This is the gate.

use zenbench::prelude::*;

/// Compressible but not degenerate: repeated structure with noise, so the
/// match finder actually runs. Pure random would find no matches and pure
/// repetition would find only long ones — either would misrepresent the
/// match-length distribution this change is about.
fn corpus(n: usize) -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    let mut v = Vec::with_capacity(n);
    let words: Vec<&[u8]> = vec![
        b"the quick brown fox ",
        b"jumps over ",
        b"lazy dog ",
        b"pack my box ",
        b"with five dozen ",
        b"liquor jugs ",
        b"sphinx of black quartz ",
    ];
    while v.len() < n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let w = words[(s >> 16) as usize % words.len()];
        v.extend_from_slice(w);
        if (s >> 28) & 1 == 0 {
            v.push((s >> 24) as u8);
        }
    }
    v.truncate(n);
    v
}

fn bench_compress(suite: &mut Suite) {
    for &(label, n) in &[("256KiB", 256 * 1024usize), ("4MiB", 4 * 1024 * 1024)] {
        let data: &'static [u8] = Box::leak(corpus(n).into_boxed_slice());
        suite.compare(format!("compress/{label}"), |g| {
            g.throughput(Throughput::Bytes(n as u64));
            g.bench("level3", move |b| {
                b.iter(move || {
                    zenzstd::encoding::compress_to_vec(
                        data,
                        zenzstd::encoding::CompressionLevel::Level(3),
                    )
                })
            });
        });
    }
}

zenbench::main!(bench_compress);

//! Per-kernel NEON-vs-forced-scalar for `count_match`, the LZ inner loop.
//!
//! It runs at every candidate match position during encode, so it is the
//! hottest byte-level loop in the encoder. Whole-compression benches cannot
//! reveal it being slower than its own scalar fallback.
//!
//! NOTE: on aarch64 NEON is BASELINE, so the "scalar" arm is autovectorized
//! too. ~1.00x means both compiled to equivalent work; BELOW 1.00 is the bug.
//!
//! Match length is swept because this kernel exits as soon as bytes differ:
//! short matches measure entry/dispatch cost, long ones measure steady-state
//! scan rate, and a single length would hide one of them. Real zstd match
//! distributions are dominated by short matches, so the short cases matter
//! more than the headline throughput.
//!
//! Run: `cargo bench --bench kernel_tiers --features simd`

use zenbench::prelude::*;
use zenzstd::__bench_match::count_match;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_e: bool) -> bool {
    false
}

/// Two buffers sharing a prefix of exactly `shared` bytes, then differing.
fn pair(len: usize, shared: usize) -> (Vec<u8>, Vec<u8>) {
    let mut s = 0x9e37_79b9u32;
    let a: Vec<u8> = (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect();
    let mut b = a.clone();
    if shared < len {
        b[shared] = b[shared].wrapping_add(1);
    }
    (a, b)
}

fn bench_match(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    const LEN: usize = 1 << 16;
    for &shared in &[4usize, 16, 64, 512, 16384] {
        let (a, b) = pair(LEN, shared);
        let a: &'static [u8] = Box::leak(a.into_boxed_slice());
        let b: &'static [u8] = Box::leak(b.into_boxed_slice());
        suite.compare(format!("count_match/shared{shared}"), |g| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |bch| {
                    bch.iter(move || {
                        set_simd(simd);
                        count_match(a, b)
                    })
                });
            }
        });
    }
    set_simd(true);
}

zenbench::main!(bench_match);

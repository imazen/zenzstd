//! `count_match` must return the same length on every dispatch path.
//!
//! aarch64 routes to the u64 path rather than the vector one (see the comment
//! in `encoding/simd.rs`), so this pins that the two agree — a match-length
//! disagreement would silently change the compressed output.
#![cfg(feature = "simd")]

use zenzstd::__bench_match::count_match;

/// Reference: the plainest possible definition.
fn reference(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[test]
fn count_match_matches_the_plain_definition() {
    let mut s = 0x9e37_79b9u32;
    let mut next = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 24) as u8
    };
    let base: Vec<u8> = (0..600).map(|_| next()).collect();

    // Every shared-prefix length across and past the 8/16/32-byte block
    // boundaries, plus unequal slice lengths and empty inputs.
    for len in [
        0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200, 599,
    ] {
        for shared in 0..=len.min(70) {
            let a = &base[..len];
            let mut bv = base[..len].to_vec();
            if shared < len {
                bv[shared] = bv[shared].wrapping_add(1);
            }
            let got = count_match(a, &bv);
            let want = reference(a, &bv);
            assert_eq!(got, want, "len={len} shared={shared}");
        }
    }

    // Mismatched lengths: must stop at the shorter slice.
    assert_eq!(count_match(&base[..10], &base[..4]), 4);
    assert_eq!(count_match(&base[..0], &base[..10]), 0);
}

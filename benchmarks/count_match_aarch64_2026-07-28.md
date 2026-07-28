# `count_match` on aarch64: vector path removed — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Benches: `benches/kernel_tiers.rs` (per-kernel), `benches/compress_e2e.rs` (end-to-end)

`count_match` is the LZ inner loop — it runs at every candidate match position during encode.
The module doc describes the SIMD path as *"32-byte AVX2 vector comparison (XOR + movemask +
trailing_zeros)"*: an algorithm designed for x86. On aarch64 the same generic body ran via the
`neon` tier, and it lost.

## Per-kernel, shared-prefix length swept

| shared prefix | NEON | u64 scalar | winner |
|---|---|---|---|
| 4 B | 6.96 ns | 4.65 ns | **u64 33% faster** |
| 16 B | 7.97 ns | 5.10 ns | **u64 36% faster** |
| 64 B | 10.6 ns | 9.2 ns | u64 13% faster |
| 512 B | 37.8 ns | 44.4 ns | neon 18% faster |
| 16 KiB | 1.2 µs | 1.2 µs | tied |

The vector path only wins in a narrow band around 512 bytes. LZ match lengths are dominated by
short matches (zstd's minimum is 3–4 bytes and the distribution decays fast), so the common
case was paying 33–36% for a win that almost never applies.

## Why NEON loses here specifically

The algorithm is XOR + **movemask** + `trailing_zeros`. On x86 the movemask is a single
instruction. NEON has no equivalent and must narrow/extract across several instructions, so
the per-iteration overhead is much higher — and with a short match you pay that overhead once
and then exit, never amortizing it over enough bytes to beat a plain 8-byte u64 XOR.

This is a general lesson for this workspace, not a one-off: **a movemask-shaped algorithm
ported to NEON should be re-measured, not assumed to carry over.**

## End-to-end validation

A kernel win can evaporate at the pipeline level (a zensim blur rewrite in this same sweep was
1.46× on the stage and −1.4% on the pipeline), so this was gated end-to-end:

| input | vector path (was) | u64 path (now) | |
|---|---|---|---|
| 256 KiB | 2.0 ms (125 MB/s) | **1.7 ms (146 MB/s)** | **1.18×** |
| 4 MiB | 20.9 ms (191 MB/s) | **19.9 ms (201 MB/s)** | 1.05× |

## Output is unchanged

`count_match` returns a match length, and both paths compute the same quantity — this is not
an approximation swap. `tests/count_match_parity.rs` pins that against the plainest possible
definition across every shared-prefix length spanning the 8/16/32-byte block boundaries, plus
mismatched and empty slices. Identical match lengths mean identical compressor decisions and
therefore byte-identical output.

## x86 is untouched

The vector path remains on x86, where movemask makes the tradeoff genuinely different. That is
unmeasurable from this machine, so it was left alone rather than changed on inference.

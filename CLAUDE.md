# zenzstd

Pure Rust zstd compression/decompression. Fork of ruzstd 0.8.2, extended with full compression.

## Architecture

- `src/decoding/` — RFC 8878 compliant decompressor (from ruzstd, battle-tested 39M downloads)
- `src/encoding/` — Full compressor with levels 1-22
  - `match_state.rs` — Shared types (MatchState, RepCodes, CompressedBlock) and cross-block state (maps to zstd_compress_internal.h)
  - `zstd_fast.rs` — Fast + DFast strategies with hash pipelining (maps to zstd_fast.c + zstd_double_fast.c)
  - `zstd_lazy.rs` — Greedy/Lazy/Lazy2/BtLazy2 with nextToUpdate + BT match finder (maps to zstd_lazy.c)
  - `zstd_opt.rs` — BtOpt/BtUltra/BtUltra2 optimal parsing (maps to zstd_opt.c)
  - `zstd_match.rs` — Thin dispatch layer + tests + re-exports
  - `compress_params.rs` — All 4 zstd compression parameter tables (default/256K/128K/16K)
  - `hash.rs` — zstd hash functions (hash3-hash8 matching C primes)
  - `simd.rs` — AVX2 count_match (32B/iter via archmage incant!), 4-way histogram
  - `blocks/compressed.rs` — Sequence/literal encoding (FSE + Huffman)
  - `streaming_encoder.rs` — `impl std::io::Write` streaming encoder
  - `levels/zstd_levels.rs` — Level dispatch bridging match finder to block encoder
- `src/xxhash64.rs` — Pure Rust XXHash64 (replaces twox-hash dependency)
- `src/fse/` — FSE (Finite State Entropy) encoder/decoder
- `src/huff0/` — Huffman encoder/decoder
- `block_splitter.rs` — Pre-split (fingerprint-based, port of C zstd_preSplit.c) + post-split (trial-encode sequence splitting)
- `vendor/zstd/` — C zstd submodule for reference. **Run
  `git submodule update --init --depth 1 vendor/zstd` on a fresh checkout**:
  without it the 7 `tests::conformance::golden_*` tests fail with
  `NotFound: No such file or directory` (they read
  `vendor/zstd/tests/golden-decompression/*.zst` directly). That failure looks
  like a real conformance break and is not one — it cost time on 2026-07-28.

## Key Design Decisions

- `#![forbid(unsafe_code)]` — all code is safe Rust
- `#![no_std]` with alloc — std is optional
- Safe ringbuffer using Vec<u8> with power-of-2 capacity bitmask
- Cross-block match history (MatchState persists window + rep offsets + tables)
- BT* strategies use DUBT-style binary tree match finder
- BtOpt/BtUltra/BtUltra2 use price-based optimal parsing (forward price table + backward trace)
- Raw slice-based _ext match functions avoid HashTable struct overhead
- Step-based hash insertion for long matches (step=4 when ml>32)

## Performance (100KB, vs C zstd)

### Compression speed
| Level | zenzstd | C zstd | Gap |
|-------|---------|--------|-----|
| L1 | 377 MiB/s | 3.93 GiB/s | 10x |
| L3 | 295 MiB/s | 2.18 GiB/s | 7.4x |
| L7 | 118 MiB/s | 446 MiB/s | 3.8x |
| L11 | 43.3 MiB/s | 132 MiB/s | 3x |
| L19 | 110 MiB/s | 1.6 MiB/s | **69x faster** |

### Compression ratio (mixed data, zen/c where <1.0 = better than C)
| Level | zen/c ratio |
|-------|-------------|
| L1 | 1.03 (3% worse) |
| L3 | 0.96 (4% better) |
| L7 | 0.77 (23% better) |
| L11 | 0.65 (35% better) |

### Decompression speed
zenzstd 5.54 GiB/s vs C 5.66 GiB/s (2% gap)

## Known Issues

### ~~Frame header under-declares the window size — C zstd mis-decodes our output~~ FIXED (#9)
The frame header wrote `self.state.matcher.window_size()`, the
`MatchGeneratorDriver`'s fixed **128 KiB**. That is only correct for
`CompressionLevel::Fastest` and `Uncompressed`. Every other level matches
through `MatchState`, whose window is
`params_for_level(level, None).window_size() = 1 << window_log` — **512 KiB at
L1, 1 MiB at L2, 2 MiB at L3-L8, 4 MiB at L9-L16, 8 MiB at L17-L19**, up to
`window_log 27` (128 MiB) at L22. So the encoder emitted back-references further
than the header said the decoder needed to retain.

Our own decoder keeps more than the declared window and resolved those offsets
anyway, which is exactly why every in-repo round-trip test passed. C zstd sizes
its window from the header, mis-resolved the far offsets, and reported
`Restored data doesn't match checksum` (~500-800 KB) or `Data corruption
detected` (~900 KB+). Measured 2026-08-29: 112 of the 12,842 `byte_identity`
cases, all at 1,000,000-byte inputs, at every level **except** `Fastest`.

**Fix:** `compress_params::encoder_params_for_level` is the single source of
truth for how far back a match may reach. `frame_compressor::header_window_size`
(used by both the one-shot and the streaming encoder), `MatchState`'s retained
history, and the match finders' `dist <= params.window_size()` bound all derive
from it; `levels::zstd_levels::compress_level` debug-asserts the `MatchState`'s
window equals the search params'. Post-fix: C-zstd-only failures **112 -> 0**,
our own decoder's 458-case failure set unchanged, and 11,760 of 12,842 outputs
moved by exactly one byte each — offset 5, the `Window_Descriptor`, with no
length change anywhere. Regression: `tests/window_declaration.rs`.

`MAX_ENCODER_WINDOW_LOG = 23` caps the window at 8 MiB. Two reasons, both worth
keeping in mind before raising it:
- The format spec asks encoders not to require more than 8 MiB, since that is
  the floor decoders are asked to support.
- Declaring the uncapped `window_log 27` (128 MiB) at L22 makes **our own**
  `FrameDecoder` reject the frame with `WindowSizeTooBig`. Note that limit is
  asymmetric: `MAXIMUM_ALLOWED_WINDOW_SIZE` (100 MiB) is checked in
  `FrameDecoderState::reset` but **not** in `FrameDecoderState::new`, so a fresh
  decoder accepts a window a reused one refuses. Measured, not inferred.

The cap only changes levels 20-22 (table window_logs 25/26/27) and only for
inputs larger than 8 MiB; nothing in the test corpus reaches that, so no
compressed payload byte moved.

### ~~Raw-dict roundtrip corruption at L13-15 (BtLazy2)~~ FIXED (#5)
The BtLazy2 binary-tree match finder (levels 13-15 in the default param table)
produced a single corrupted byte when compressing with a raw dictionary: a
back-reference resolved into the dict prefix at a wrong offset, so the decoder
copied a dict byte where the original differed. Root cause: `prefill_binary_tree`
seeds dict-prefix positions as an *unsorted* DUBT chain (`tree[larger] =
ZSTD_DUBT_UNSORTED_MARK`), but `insert_and_find` / `insert_only` traversed the
tree as if every node were sorted. That violated the `commonLengthSmaller/Larger`
monotonicity invariant and let `count_match` skip past a real mismatch,
overstating the match length.
Fix: port C zstd's two leading loops of `ZSTD_DUBT_findBestMatch` —
`sort_unsorted_chain` walks the unsorted hash chain and sorts each entry into the
tree (`insert_dubt1`, a port of `ZSTD_insertDUBT1`) before the sorted traversal.
Regression: `src/tests/dict_test.rs::dict_roundtrip_{l15,all_levels}_issue5` +
the `fuzz/regression/dict_roundtrip_l15_issue5` seed exercised by
`tests/fuzz_regression.rs` (run with `--features fuzz_exports`).

### ~~Cross-block L5/L7 regression on 1MB repetitive text~~ FIXED
Fixed in two parts: (1) persistent hash/chain tables across blocks via position shifting
instead of clearing/repopulating each block; (2) block splitter sampling rate fix to avoid
false splits on uniform repetitive data due to phase-offset aliasing at rate=5.
Result: 1MB text at L5/L7 now produces 247 bytes (down from 2662), vs C's 148.

### L16-22 compression ratio gap (zen/c = 1.17 on mixed_100KB)
FSE table mode selection and Huffman literal threshold have been fixed:
- `choose_table` now estimates cost for predefined, repeat-last, and new (encoded) modes using
  cross-entropy calculation (matching C zstd's ZSTD_selectEncodingType approach), picks cheapest
- Huffman compression threshold lowered from 1024 to 32 bytes (with fallback to raw if Huffman
  doesn't reduce size)
- Previous-table tracking updated so repeat-last mode persists across blocks

The remaining gap is likely in the optimal parser (BtOpt/BtUltra) match quality, not entropy coding.

### L19+ decoding corruption with mixed-entropy data
The BtOpt/BtUltra match finder produces corrupt output for certain data patterns at levels 16-22.
Specifically, the benchmark's `make_mixed` pattern (alternating ASCII letters and pseudo-random
bytes) fails round-trip at any size >= ~1000 bytes. Repetitive text data and small inputs work
correctly. This is a match finder bug, not an entropy coding issue.
- Repro: compress `make_mixed(10000)` at L19, decode with our decoder — data mismatch
- All levels 1-15 work correctly with the same data

## Features

- `default` = ["hash", "std", "simd"]
- `std` — enables std::io traits, StreamingEncoder
- `hash` — enables xxhash64 checksums in frames
- `dict_builder` — dictionary training (from ruzstd)
- `fuzz_exports` — exposes FSE/Huffman internals
- `simd` — archmage/magetypes acceleration in two places, both x86_64-centric:
  - **encoder** `count_match` (`encoding/simd.rs`): AVX2 `u8x32` on x86_64,
    `u8x16` on wasm32. **aarch64 deliberately uses the scalar u64 path** — the
    NEON dispatch measured slower at the short match lengths that dominate, so
    the feature is a no-op for `count_match` there (measurements in the comment
    on `count_match`).
  - **decoder** hot loops: `#[archmage::autoversion]` on
    `fused_decode_execute_fast_inner` (`sequence_section_decoder.rs`) and the
    Huffman bit extraction (`literals_section_decoder.rs`), for TZCNT / BMI2
    PDEP-PEXT / wider copies. The two are **not gated alike**, and the
    difference is load-bearing:
    - `literals_section_decoder` gates the attribute *and* the call site to
      x86_64, so aarch64 has no `decode_huffman_stream_neon` at all.
    - `sequence_section_decoder` gates only the `incant!` call site. Its
      `#[autoversion]` attribute is not arch-gated, so on aarch64 the
      non-x86 arm calls the autoversion dispatcher, which **does** run
      `___arcane_fused_decode_execute_fast_inner_neon`. Verified in the emitted
      LLVM IR, 2026-08-29.
    That call site's comment claiming "NEON autoversion has a known decode
    correctness issue" (from `4602a83`) is **wrong on both halves** and should
    not be repeated. The gate does not keep NEON out (above), and no NEON
    divergence exists: arch-gating the attribute as well, so aarch64 truly
    decodes scalar, produces byte-for-byte identical results across the full
    suite, the 101-file decode corpus and all 12,842 `byte_identity` frames.
    The original windows-arm symptom ("output truncated by small amounts") was
    root-caused two commits later, in `7d16d87`/`be8af63`, to git inflating the
    binary corpus files with CRLF — `b2fdf39`, 13 minutes after the gate, had
    already recorded that ARM still failed with the gate in place and that the
    cause was "not SIMD related". The gate is kept only because removing it buys
    nothing measurable and the verification covered aarch64-apple-darwin, not
    windows-11-arm; the NEON variant that already runs is worth 1.9%
    (3.7762 ms vs 3.8501 ms on `benches/decode_all`, p < 0.05, M-series).
- `unsafe-decompress` / `unsafe-compress` — opt-in unchecked indexing in hot
  paths. Off by default; the crate is `#![forbid(unsafe_code)]` without them.

## Commands

```
cargo test                      # all tests (271 lib + 6 integration as of 2026-08-29)
cargo test --features simd      # with SIMD (simd is on by default)
cargo bench --bench compress_compare              # benchmark
cargo bench --bench compress_compare -- --save-baseline main
cargo bench --bench compress_compare -- --baseline main --max-regression 5
```

### Before and after ANY change to the encoder

Round-trip tests cannot catch a change in emitted bytes: the encoder and decoder
just agree with each other on the new output. Hash it instead.

```
cargo run --release --example byte_identity -- ~/tmp/before   # on the base commit
cargo run --release --example byte_identity -- ~/tmp/after    # with your change
shasum -a 256 ~/tmp/{before,after}/all.bin                    # must match
diff ~/tmp/{before,after}/cases.tsv                           # names the case that moved
```

12,842 cases, ~35 s per run. If the sha256 moves and you did not intend it to,
you changed the bitstream — find out why before committing. The current
reference is sha256 `45f69496…` over 256,991,165 blob bytes, with 458
self-round-trip failures (all L16-22, the optimal-parser bug) and **0** cases
that only C zstd rejects. That last number going nonzero means we are emitting
streams a conformant decoder cannot read while ours accepts them, which is
exactly how #9 hid. If you added a chunked loop, add a size or content kind that
reaches its remainder path, then break that loop on purpose and confirm the case
count moves; a harness that covers nothing reports success just as loudly as one
that covers everything.

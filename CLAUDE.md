# zenzstd

Pure Rust zstd compression/decompression. Fork of ruzstd 0.8.2, extended with full compression.

## Architecture

- `src/decoding/` — RFC 8878 compliant decompressor (from ruzstd, battle-tested 39M downloads)
- `src/encoding/` — Full compressor with levels 1-22
  - `zstd_match.rs` — Core match finder: Fast, DFast, Greedy, Lazy, Lazy2, BtLazy2, BtOpt/BtUltra/BtUltra2 strategies
  - `compress_params.rs` — All 4 zstd compression parameter tables (default/256K/128K/16K)
  - `hash.rs` — zstd hash functions (hash4-hash8 matching C primes)
  - `simd.rs` — AVX2 count_match (32B/iter via archmage incant!), 4-way histogram
  - `blocks/compressed.rs` — Sequence/literal encoding (FSE + Huffman)
  - `streaming_encoder.rs` — `impl std::io::Write` streaming encoder
  - `levels/zstd_levels.rs` — Level dispatch bridging match finder to block encoder
- `src/xxhash64.rs` — Pure Rust XXHash64 (replaces twox-hash dependency)
- `src/fse/` — FSE (Finite State Entropy) encoder/decoder
- `src/huff0/` — Huffman encoder/decoder
- `vendor/zstd/` — C zstd submodule for reference

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
zenzstd 1.91 GiB/s vs C 5.64 GiB/s (3x gap)

## Known Bugs

None currently.

## Features

- `default` = ["hash", "std"]
- `std` — enables std::io traits, StreamingEncoder
- `hash` — enables xxhash64 checksums in frames
- `dict_builder` — dictionary training (from ruzstd)
- `fuzz_exports` — exposes FSE/Huffman internals
- `simd` — AVX2 acceleration via archmage/magetypes

## Commands

```
cargo test                      # all tests (209)
cargo test --features simd      # with SIMD
cargo bench --bench compress_compare              # benchmark
cargo bench --bench compress_compare -- --save-baseline main
cargo bench --bench compress_compare -- --baseline main --max-regression 5
```

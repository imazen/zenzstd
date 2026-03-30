# zenzstd

Pure Rust zstd compression/decompression. Fork of ruzstd 0.8.2, extended with full compression.

## Architecture

- `src/decoding/` — RFC 8878 compliant decompressor (from ruzstd, battle-tested 39M downloads)
- `src/encoding/` — Full compressor with levels 1-22
  - `zstd_match.rs` — Core match finder: Fast, DFast, Greedy, Lazy, Lazy2, BtLazy2, BtOpt/BtUltra/BtUltra2 strategies
  - `compress_params.rs` — All 4 zstd compression parameter tables (default/256K/128K/16K)
  - `hash.rs` — zstd hash functions (hash4-hash8 matching C primes)
  - `simd.rs` — SIMD-accelerated count_match (archmage optional)
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
- BT* strategies (levels 13-22) use a binary tree match finder (DUBT-style)
- BtLazy2 uses binary tree + lazy2 evaluation
- BtOpt/BtUltra/BtUltra2 use price-based optimal parsing (forward price table + backward trace, following C zstd's `ZSTD_compressBlock_opt_generic`)
- Multi-block encoder has a known bug at levels >= 3 with non-trivial patterns

## Known Bugs

None currently. The match length code 52 bug (wrong baseline) has been fixed.

## Features

- `default` = ["hash", "std"]
- `std` — enables std::io traits, StreamingEncoder
- `hash` — enables xxhash64 checksums in frames
- `dict_builder` — dictionary training (from ruzstd)
- `fuzz_exports` — exposes FSE/Huffman internals
- `simd` — optional archmage SIMD acceleration

## Test Commands

```
cargo test                      # all tests (199)
cargo test --features simd      # with SIMD
cargo check --features simd     # verify SIMD compiles
```

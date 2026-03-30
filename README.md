# zenzstd ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenzstd/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenzstd?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/zenzstd?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/zenzstd?style=flat-square) ![license](https://img.shields.io/crates/l/zenzstd?style=flat-square)

[lib.rs](https://lib.rs/crates/zenzstd) | [docs.rs](https://docs.rs/zenzstd)

A pure Rust Zstandard ([RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.pdf)) compressor and decompressor. `#![forbid(unsafe_code)]`, `no_std + alloc`.

Fork of [ruzstd](https://crates.io/crates/ruzstd) (39M+ downloads), extended with full compression (levels 1-22), streaming encoding, dictionary support, and optional SIMD acceleration.

## Quick start

Compress data:

```rust
use zenzstd::encoding::{compress_to_vec, CompressionLevel};

let data = b"some data to compress";
let compressed = compress_to_vec(&data[..], CompressionLevel::Default);
```

Decompress data:

```rust
use zenzstd::decoding::FrameDecoder;

let compressed: &[u8] = // ... zstd-compressed bytes
let mut decoder = FrameDecoder::new();
let mut output = vec![0u8; 1024 * 1024]; // must be large enough
let bytes_written = decoder.decode_all(compressed, &mut output).unwrap();
output.truncate(bytes_written);
```

Streaming compression (requires `std` feature, enabled by default):

```rust
use std::io::Write;
use zenzstd::encoding::{StreamingEncoder, CompressionLevel};

let mut output = Vec::new();
let mut encoder = StreamingEncoder::new(&mut output, CompressionLevel::Default);
encoder.write_all(b"Hello, world!").unwrap();
let _inner = encoder.finish().unwrap();
// `output` now contains a valid zstd frame
```

## Performance

Measured on 100KB inputs, default target (no `-C target-cpu=native`). See `benches/compress_compare.rs` for methodology.

### Compression speed

| Level | zenzstd | C zstd | Ratio |
|-------|---------|--------|-------|
| L1 | 377 MiB/s | 3.93 GiB/s | 0.09x |
| L3 | 295 MiB/s | 2.18 GiB/s | 0.13x |
| L7 | 118 MiB/s | 446 MiB/s | 0.26x |
| L11 | 43 MiB/s | 132 MiB/s | 0.33x |
| L19 | 110 MiB/s | 1.6 MiB/s | 69x |

At low levels, C zstd is 3-10x faster -- it has hand-tuned assembly for hash tables and match finders. At L19+, zenzstd's optimal parser is faster because C zstd's BtUltra strategy is more expensive for the same search depth.

### Compression ratio (mixed data, zen/c where <1.0 = smaller output)

| Level | zen/c |
|-------|-------|
| L1 | 1.03 |
| L3 | 0.96 |
| L7 | 0.77 |
| L11 | 0.65 |
| L19 | 1.17 |

At L3-L11, zenzstd produces 4-35% smaller output than C zstd on mixed data. At L19, C zstd's entropy encoder is more sophisticated (repeat-mode FSE tables, better predefined table selection) and wins on ratio.

### Decompression speed

5.54 GiB/s vs C zstd's 5.66 GiB/s (98% of C speed).

## Features

- **`std`** (default) -- enables `std::io` traits, `StreamingEncoder`, `StreamingDecoder`
- **`hash`** (default) -- xxhash64 content checksums in frames
- **`dict_builder`** -- dictionary training from sample data (implies `std`)
- **`simd`** -- AVX2 acceleration for match counting via [archmage](https://crates.io/crates/archmage)

## Limitations

Compression speed at levels 1-11 is 3-10x slower than C zstd. If compression throughput at low levels is your bottleneck, use the [zstd](https://crates.io/crates/zstd) crate (C bindings) instead. zenzstd is a better fit when you need pure Rust, `no_std`, `forbid(unsafe_code)`, or high compression levels where the gap narrows.

The L16-22 compression ratio is ~17% worse than C zstd due to entropy encoder differences (FSE table mode selection, Huffman on small literal sections). This is a known gap being worked on.

## License

MIT. Fork of [ruzstd](https://github.com/KillingSpark/zstd-rs) by Moritz Borcherding.

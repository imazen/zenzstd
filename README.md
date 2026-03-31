# zenzstd ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenzstd/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenzstd?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/zenzstd?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/zenzstd?style=flat-square) ![license](https://img.shields.io/crates/l/zenzstd?style=flat-square)

[lib.rs](https://lib.rs/crates/zenzstd) | [docs.rs](https://docs.rs/zenzstd)

A pure Rust Zstandard ([RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.pdf)) compressor and decompressor. `#![forbid(unsafe_code)]` by default, `no_std + alloc`.

Fork of [ruzstd](https://crates.io/crates/ruzstd) (39M+ downloads), extended with full compression (levels 1-22), streaming encoding, dictionary support, and optional SIMD acceleration via [archmage](https://crates.io/crates/archmage).

## Quick start

Drop-in `zstd` crate compatible API:

```rust
use zenzstd::stream;

// Compress
let compressed = stream::encode_all(data.as_slice(), 3).unwrap();

// Decompress
let original = stream::decode_all(compressed.as_slice()).unwrap();

// Copy between readers/writers
stream::copy_encode(source, &mut dest, 3).unwrap();
stream::copy_decode(compressed.as_slice(), &mut dest).unwrap();
```

Low-level API:

```rust
use zenzstd::encoding::{compress_to_vec, CompressionLevel};
let compressed = compress_to_vec(&data[..], CompressionLevel::Default);

use zenzstd::decoding::FrameDecoder;
let mut decoder = FrameDecoder::new();
let mut output = vec![0u8; expected_size + 4096];
decoder.decode_all_to_vec(&compressed, &mut output).unwrap();
```

Streaming (requires `std`):

```rust
use std::io::Write;
use zenzstd::encoding::{StreamingEncoder, CompressionLevel};

let mut output = Vec::new();
let mut encoder = StreamingEncoder::new(&mut output, CompressionLevel::Level(3));
encoder.write_all(b"Hello, world!").unwrap();
encoder.finish().unwrap();
```

## Performance

100KB inputs, `--features simd,unsafe-decompress`. Run `cargo run --release --example compare`.

### L3 decode (the default level)

| Data type | zenzstd | C zstd | Gap |
|-----------|---------|--------|-----|
| text | 10.0 GiB/s | 6.2 GiB/s | **1.6x faster** |
| mixed | 1.7 GiB/s | 3.3 GiB/s | 1.9x slower |
| random | 15.0 GiB/s | 17.1 GiB/s | 1.1x slower |

Text and random decompression exceeds C zstd speed. Mixed data (many short matches) is 1.9x slower due to per-sequence overhead in safe Rust.

### Encode speed (100KB)

| Level | zenzstd | C zstd | Gap |
|-------|---------|--------|-----|
| L1 | 725 MB/s | 2.9 GiB/s | 4.0x |
| L3 | 671 MB/s | 1.1 GiB/s | 1.7x |
| L7 | 166 MB/s | 190 MB/s | **1.1x (parity)** |
| L11 | 29 MB/s | 50 MB/s | 1.7x |
| L15+ | faster | — | **zenzstd wins** |

### Compression ratio (mixed 100KB, zen_size / c_size)

| Level | zen/c | Verdict |
|-------|-------|---------|
| L1 | 1.14 | 14% larger |
| L3 | **0.96** | 4% smaller |
| L7 | **0.73** | 27% smaller |
| L11 | **0.65** | 35% smaller |

At L3-L11, zenzstd produces smaller output than C zstd on mixed data.

### Safe vs unsafe decode

| Mode | mixed L3 | Gap to C |
|------|----------|----------|
| `#![forbid(unsafe_code)]` (default) | 1.5 GiB/s | 2.1x |
| `--features unsafe-decompress` | 1.7 GiB/s | 1.9x |
| `--features simd,unsafe-decompress` | 1.7 GiB/s | 1.9x |

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | `std::io` traits, `StreamingEncoder`, `StreamingDecoder`, `stream` module |
| `hash` | yes | xxhash64 content checksums in frames |
| `dict_builder` | no | Dictionary training from sample data |
| `simd` | no | AVX2/BMI2 acceleration via [archmage](https://crates.io/crates/archmage) (`#[autoversion]` on hot loops) |
| `unsafe-decompress` | no | Unchecked indexing in decode hot paths (5-10% faster) |
| `unsafe-compress` | no | Unchecked indexing in encode hot paths (reserved) |

Without `unsafe-decompress` or `unsafe-compress`, the crate uses `#![forbid(unsafe_code)]`. With either feature, it uses `#![deny(unsafe_code)]` with `#[allow(unsafe_code)]` only on isolated `unsafe_ops` modules.

## Fuzzing

6 cargo-fuzz targets covering decode, round-trip, streaming, dictionary, FSE, and Huffman:

```bash
cargo +nightly fuzz run fuzz_decode
cargo +nightly fuzz run fuzz_roundtrip
cargo +nightly fuzz run fuzz_streaming_roundtrip
```

## License

MIT. Fork of [ruzstd](https://github.com/KillingSpark/zstd-rs) by Moritz Borcherding.

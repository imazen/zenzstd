<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenzstd [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenzstd/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenzstd/actions/workflows/ci.yml)

A pure-Rust [Zstandard](https://www.rfc-editor.org/rfc/rfc8878) (RFC 8878) compressor and decompressor. `#![forbid(unsafe_code)]` by default and `no_std + alloc`, so it runs anywhere from servers to embedded targets to WebAssembly.

The decoder builds on the well-maintained [`ruzstd`](https://crates.io/crates/ruzstd) decoder (39M+ downloads). `zenzstd` adds a full compressor (levels 1-22), `std::io` streaming encode/decode, dictionary support, and optional SIMD acceleration via [archmage](https://crates.io/crates/archmage).

## Quick start

```toml
[dependencies]
zenzstd = "0.1.0"
```

The `stream` module mirrors the `zstd` crate's one-shot helpers (requires the default `std` feature):

```rust
use zenzstd::stream;

// Compress a byte slice at level 3 (Zstandard's default level).
let compressed = stream::encode_all(&b"hello hello hello world"[..], 3).unwrap();

// Decompress. `decode_all` applies a 1 GiB output-size cap to guard against
// decompression bombs; use `decode_all_unbounded` for fully trusted input.
let original = stream::decode_all(&compressed[..]).unwrap();
assert_eq!(original.as_slice(), &b"hello hello hello world"[..]);

// Reader-to-writer variants, also `zstd`-crate compatible:
// stream::copy_encode(source, &mut dest, 3)?;
// stream::copy_decode(compressed_reader, &mut dest)?;
```

### Low-level / `no_std`

Without `std`, use the buffer-oriented API directly:

```rust
use zenzstd::encoding::{compress_to_vec, CompressionLevel};
use zenzstd::decoding::FrameDecoder;

let data = b"the quick brown fox the quick brown fox";
let compressed = compress_to_vec(&data[..], CompressionLevel::Default);

// `decode_all_to_vec` writes into the vector's spare capacity and never
// reallocates, so reserve enough room up front.
let mut decoder = FrameDecoder::new();
let mut out = Vec::with_capacity(data.len() + 4096);
decoder.decode_all_to_vec(&compressed, &mut out).unwrap();
assert_eq!(out.as_slice(), &data[..]);
```

### Streaming encode (`std`)

```rust
use std::io::Write;
use zenzstd::encoding::{StreamingEncoder, CompressionLevel};

let mut output = Vec::new();
let mut encoder = StreamingEncoder::new(&mut output, CompressionLevel::Level(3));
encoder.write_all(b"streamed payload").unwrap();
encoder.finish().unwrap();
```

For incremental decoding, [`decoding::StreamingDecoder`](https://docs.rs/zenzstd/latest/zenzstd/decoding/struct.StreamingDecoder.html) implements `std::io::Read`.

## Compression levels

`CompressionLevel` maps named presets onto numeric zstd levels, or you can pass an explicit `Level(1..=22)`:

| Preset | zstd level | Status |
|--------|-----------:|--------|
| `Uncompressed` | 0 | raw block, wrapped in a Zstandard frame |
| `Fastest` | 1 | stable |
| `Default` | 3 | stable (used when no level is given) |
| `Better` | 7 | stable |
| `Best` | 11 | stable |
| `Level(1..=15)` | 1-15 | stable — round-trip fuzzed; output verified decodable by the reference C `zstd` library |
| `Level(16..=22)` | 16-22 | experimental — see note below |

**Levels 16-22 are experimental.** The optimal-parse match finder (BtOpt/BtUltra) has a known, data-dependent corruption bug at these levels, so they are excluded from round-trip fuzzing and not yet recommended for production. Use levels 1-15 today; level 3 is the default. The **decoder** handles valid Zstandard streams at all levels (verified against C `zstd` output for levels 1-22).

## Decompression safety

`stream::decode_all` and `stream::copy_decode` apply a 1 GiB output-size cap by default — a few KB of crafted RLE blocks can otherwise expand to terabytes. For other policies:

- `decode_all_with_max(reader, Some(bytes))` / `copy_decode_with_max(...)` — custom cap.
- `decode_all_unbounded(reader)` / `copy_decode_unbounded(...)` — no cap (trusted input only).
- On `StreamingDecoder`, `set_max_output_size(Some(bytes))`.

## Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `std` | yes | `std::io` traits, `StreamingEncoder`/`StreamingDecoder`, and the `stream` module |
| `hash` | yes | XXH64 content checksums in frames |
| `simd` | yes | AVX2/BMI2 acceleration via [archmage](https://crates.io/crates/archmage) / [magetypes](https://crates.io/crates/magetypes) (`#[autoversion]` on hot loops) |
| `dict_builder` | no | Dictionary training from sample data (pulls in `fastrand`; implies `std`) |
| `unsafe-decompress` | no | Unchecked indexing in decode hot paths |
| `unsafe-compress` | no | Unchecked indexing in encode hot paths (reserved) |
| `fuzz_exports` | no | Exposes FSE/Huffman internals for fuzz targets |

For `no_std`, disable default features and re-enable what you need:

```toml
[dependencies]
zenzstd = { version = "0.1.0", default-features = false, features = ["hash"] }
```

## Safety

By default the crate is `#![forbid(unsafe_code)]`. Enabling `unsafe-decompress` or `unsafe-compress` switches it to `#![deny(unsafe_code)]`, with `unsafe` permitted only inside small, documented `unsafe_ops` modules in the hot paths. The safe-by-default build is the one fuzzed and tested in CI.


## Fuzzing

Six `cargo-fuzz` targets cover decode, round-trip, streaming, dictionary, FSE, and Huffman paths (round-trip fuzzing is restricted to levels 0-15 while 16-22 are experimental):

```bash
cargo +nightly fuzz run fuzz_decode
cargo +nightly fuzz run fuzz_roundtrip
cargo +nightly fuzz run fuzz_streaming_roundtrip
```

## Minimum supported Rust version

zenzstd builds on Rust **1.89** and newer. Bumping the MSRV is treated as a minor-version change while the crate is pre-1.0.

## License

MIT. A fork of [`ruzstd`](https://github.com/KillingSpark/zstd-rs) by Moritz Borcherding, extended with full compression. See [LICENSE](https://github.com/imazen/zenzstd/blob/main/LICENSE).

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · **zenzstd** |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go

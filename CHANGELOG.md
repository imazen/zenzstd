# Changelog

All notable changes to zenzstd are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this crate is pre-1.0, so
breaking changes bump the minor version.

## [Unreleased]

### Added
- `examples/byte_identity.rs` — a byte-identity harness for encoder refactors.
  Compresses a 12,842-case grid (12 content kinds x 38 sizes x 27 levels, plus
  the streaming encoder at five write chunkings and the raw-dictionary encoder),
  writes a per-case `label/length/fnv1a64` TSV plus one concatenated blob to
  sha256, and round-trips every output through both this decoder and the C zstd
  reference decoder. Run it before and after a change to the encoder; identical
  sha256 means not one emitted byte moved. This is the tool that produced the
  evidence for the `as_chunks` rewrite below, and running it surfaced the
  window-size bug noted under Known Issues.
- `tests/window_declaration.rs` — pins the frame header's `Window_Descriptor`
  against a longhand copy of C zstd's `clevels.h` window logs (so it cannot
  agree with the encoder by calling into it), checks the one-shot, streaming and
  dictionary encoders declare the same window, and round-trips input that
  forces a match past 128 KiB through the C reference decoder. Mutation-verified
  both ways: reverting the header to the old fixed 128 KiB fails the table check
  at level 1 and the reference round-trip at `text/1000000` level 1; dropping
  the 8 MiB cap fails the table check at level 20.

### Changed
- README overhaul: corrected feature defaults (`simd` ships on by default), documented the experimental level 16-22 status and the 1 GiB decode-bomb output cap, added the standard badge row + MSRV badge, and split the crates.io README into `README.crates.md`.

### Fixed
- **Frame header under-declared the window size, so C zstd mis-decoded our
  output above ~500 KB (#9).** The header advertised `MatchGeneratorDriver`'s
  fixed 128 KiB at every level, but only `Fastest` and `Uncompressed` match
  through that driver — every other level matches through `MatchState`, bounded
  by `params_for_level(level, None).window_size()` (512 KiB at L1 up to 128 MiB
  at L22). The encoder therefore emitted back-references past the window it told
  the decoder to keep. This crate's own decoder retains more history than the
  header declares and resolved them anyway, which is why every in-repo
  round-trip passed; the reference decoder keeps exactly what it was told to and
  reported `Restored data doesn't match checksum` or `Data corruption detected`.
  `compress_params::encoder_params_for_level` is now the single source of truth
  for how far back a match may reach — the frame header
  (`frame_compressor::header_window_size`, shared by the one-shot and streaming
  encoders), `MatchState`'s retained history and the match finders' `dist`
  bound all read it, and `compress_level` debug-asserts the two agree. It caps
  `window_log` at 23 (8 MiB): the format spec asks encoders to stay within that,
  and the uncapped 128 MiB at L22 was measured to make this crate's own
  `FrameDecoder` reject the frame with `WindowSizeTooBig` when reused across
  frames (`MAXIMUM_ALLOWED_WINDOW_SIZE` is 100 MiB). The cap only reaches levels
  20-22, and only for inputs over 8 MiB.
  Measured with `examples/byte_identity` before and after: C-zstd-only
  round-trip failures **112 -> 0**, this decoder's own failure set byte-for-byte
  unchanged at 458 (all of them the separate L16-22 optimal-parser bug, which
  both decoders reject). The change is confined to the frame header: of 12,842
  cases, 11,760 moved and every one of them differs at exactly one byte offset —
  index 5, the `Window_Descriptor` — with not one case changing length. The
  1,082 byte-identical cases are precisely `Uncompressed` and `Fastest`, the two
  levels whose declaration was already honest. Output blob sha256
  `4a6e9694…` -> `45f69496…` over an unchanged 256,991,165 bytes.
- CI's `Clippy` and `Format` jobs were red on Rust 1.98 (runs 32900982140 and
  33249577364). Eight clippy errors and four rustfmt diffs, all pre-existing
  lint debt rather than a regression:
  - `chunks_exact(N)` with a constant `N` replaced by `as_chunks::<N>()` at six
    call sites — the huff0 weight-table writer, both xxhash64 frame-checksum
    loops, the histogram kernel, and the LZ match generator (`4c1dcda`). These
    decide emitted bytes, so the rewrite was verified byte-identical rather
    than assumed: 12,842 compressions (12 content kinds x 38 sizes across every
    2/4/8/32-byte and block boundary x 27 levels, plus the streaming encoder at
    five write chunkings and the raw-dictionary encoder) hash to the same
    sha256 (`4a6e9694…`) over 256,991,165 output bytes before and after, with
    an unchanged decode-round-trip pass/fail set against both this decoder and
    C zstd. Harness sensitivity was checked with deliberate tail-handling
    canaries at each site (609 / 1,216 / 9,428 / 381 cases moved).
  - `match`-on-`Option` replaced by `?` in `FrameDecoder::get_checksum_from_data`
    and `get_calculated_checksum` (`06f7b4b`).
  - `cargo fmt` over `benches/compress_e2e.rs`, `benches/kernel_tiers.rs` and
    `tests/count_match_parity.rs` — reflow only (`1d21bed`).
  No `#[allow]` was added for any of them.
- `cargo clippy --all-targets -- -D warnings` failed on aarch64 (invisible to
  CI, which lints on x86_64) with three `dead_code` errors: `count_match_scalar`,
  `count_match_generic` and `count_match_neon` were still compiled on aarch64
  after f700f17 cfg'd the `incant!` dispatcher that calls them out of that
  target. The cfgs now match what is reachable, and the unreachable-everywhere
  `count_match_neon` adapter is removed; the u8x16 algorithm itself remains as
  the wasm128 path (`afc5480`).
- The `Fuzz regression` CI job silently skipped itself when the corpus was
  missing. The step was wrapped in
  `if [ -d fuzz/regression ] && [ "$(ls fuzz/regression/ | wc -l)" -gt 0 ]`, so
  a missing or partial checkout skipped the suite and reported green — a runtime
  self-skip invisible to the caller chain, which the repo rules forbid. (Unlike
  the sibling zen codecs this step never had a `|| echo` swallow, so a failing
  seed did already turn the job red; the hole was the skip.) The guard is gone,
  and the harness's seed count now ignores `README`/dotfiles and is pinned to
  `>= MIN_SEEDS`, so a corpus emptied down to documentation fails instead of
  replaying nothing. Mutation-verified: a README-only corpus and a panic
  injected into the replay loop each exit 101.
- Raw-dictionary compress/decompress roundtrip corruption at compression levels
  13-15 (the BtLazy2 binary-tree match finder). Dict-prefix positions were seeded
  into the tree as an unsorted DUBT chain but traversed as a sorted tree, which
  broke the match-length invariant and let the encoder emit a back-reference whose
  bytes differed from the forward data, decoding to a single wrong byte. Fixed by
  porting C zstd's unsorted-chain sort phase (`ZSTD_insertDUBT1` +
  the leading loops of `ZSTD_DUBT_findBestMatch`) into the binary-tree finder.
  Regression-gated by `dict_roundtrip_{l15,all_levels}_issue5` and the
  `fuzz/regression/dict_roundtrip_l15_issue5` seed. (#5)

## Known issues

- The frame header under-declares the window size for every compression level
  except `Fastest`, so a spec-conformant decoder (C zstd) mis-decodes our output
  once an input exceeds roughly 500 KB. Our own decoder is lenient enough to hide
  it. Details and measurements in CLAUDE.md; tracked as issue #9.
- Levels 16-22 (BtOpt/BtUltra optimal parsing) produce output that fails to
  round-trip on some content. Details in CLAUDE.md.

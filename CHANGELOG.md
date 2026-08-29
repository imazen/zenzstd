# Changelog

All notable changes to zenzstd are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this crate is pre-1.0, so
breaking changes bump the minor version.

## [Unreleased]

### Changed
- README overhaul: corrected feature defaults (`simd` ships on by default), documented the experimental level 16-22 status and the 1 GiB decode-bomb output cap, added the standard badge row + MSRV badge, and split the crates.io README into `README.crates.md`.

### Fixed
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

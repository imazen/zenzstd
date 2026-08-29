# Changelog

All notable changes to zenzstd are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this crate is pre-1.0, so
breaking changes bump the minor version.

## [Unreleased]

### Changed
- README overhaul: corrected feature defaults (`simd` ships on by default), documented the experimental level 16-22 status and the 1 GiB decode-bomb output cap, added the standard badge row + MSRV badge, and split the crates.io README into `README.crates.md`.

### Fixed
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

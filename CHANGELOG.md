# Changelog

All notable changes to zenzstd are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this crate is pre-1.0, so
breaking changes bump the minor version.

## [Unreleased]

### Fixed
- Raw-dictionary compress/decompress roundtrip corruption at compression levels
  13-15 (the BtLazy2 binary-tree match finder). Dict-prefix positions were seeded
  into the tree as an unsorted DUBT chain but traversed as a sorted tree, which
  broke the match-length invariant and let the encoder emit a back-reference whose
  bytes differed from the forward data, decoding to a single wrong byte. Fixed by
  porting C zstd's unsorted-chain sort phase (`ZSTD_insertDUBT1` +
  the leading loops of `ZSTD_DUBT_findBestMatch`) into the binary-tree finder.
  Regression-gated by `dict_roundtrip_{l15,all_levels}_issue5` and the
  `fuzz/regression/dict_roundtrip_l15_issue5` seed. (#5)

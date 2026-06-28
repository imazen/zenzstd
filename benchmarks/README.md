# zenzstd benchmarks

This directory documents how to reproduce zenzstd's performance comparisons
against the reference C library and upstream `ruzstd`. **No fixed throughput
numbers are committed in this README** — they are CPU-, build-, and
load-specific, so the only honest figure is the one you measure on your own
machine. Run the harness below and record the results alongside the commit SHA.

## What is compared

- **zenzstd** — this crate (pure Rust).
- **C `zstd`** — the reference implementation, via the [`zstd`](https://crates.io/crates/zstd)
  crate **0.13.3** (which bundles libzstd). This is the C baseline.
- **`ruzstd`** — the upstream pure-Rust decoder, [`ruzstd`](https://crates.io/crates/ruzstd)
  **0.8.2** (decode-only comparison; zenzstd's decoder derives from it).

Competitor versions are pinned in `[dev-dependencies]` of the crate's
`Cargo.toml`, so `cargo` resolves the exact versions above automatically — there
is nothing to install separately.

## Environment to record

Capture these every time you publish numbers (they are not filled in here
because this checkout has not been measured):

- CPU model and core count
- RAM
- OS / kernel
- `rustc -V`
- Build profile (`--release`)
- Feature flags used (e.g. `--features simd`)

**Build without `-C target-cpu=native`.** zenzstd uses runtime SIMD dispatch
(via [archmage](https://crates.io/crates/archmage)); compiling for the host's
exact ISA bakes in extensions end users won't have and produces misleading
numbers. Leave `RUSTFLAGS` unset (or free of `target-cpu`).

## Reproduce

```sh
git clone https://github.com/imazen/zenzstd && cd zenzstd
git checkout <commit-sha>      # the SHA your numbers came from — record it

# Canonical comparison (zenbench): ratio + encode + decode vs C zstd.
cargo bench --bench compress_compare --features simd

# Quick all-levels / all-datasets table vs C zstd and ruzstd:
cargo run --release --example compare --features simd
```

Useful zenbench flags on `compress_compare`:

```sh
cargo bench --bench compress_compare -- --save-baseline main
cargo bench --bench compress_compare -- --baseline main --max-regression 5
```

## Methodology and fairness notes

- **Single-threaded.** Both zenzstd and the C `zstd` calls used here run
  single-threaded; the comparison is like-for-like. State the thread count if you
  change it.
- **Inputs are pre-loaded into RAM.** The datasets (`make_text`, `make_mixed`,
  `make_random`) are generated into `Vec<u8>` before the timed region; no file
  I/O happens inside the measured loop. Compression reads from an in-memory
  `Cursor`, output goes to a `Vec<u8>`, and outputs are consumed via `black_box`
  so they are not optimized away.
- **Apples-to-apples.** Every contender sees the same bytes, the same length, and
  the same numeric compression level.
- **Decode-buffer caveat.** In the current decode benchmarks (and the `compare`
  example) the output buffer is allocated *inside* the timed loop — zenzstd
  allocates `vec![0u8; len + 4096]` plus a fresh `FrameDecoder`, while the C path
  allocates a `Vec::with_capacity(len)`. Both sides pay one allocation per
  iteration, so the comparison stays roughly symmetric, but treat decode
  throughput as approximate rather than allocation-free steady state.
- **`ratio_report`** prints compression ratio (`zen/c`) per dataset and level to
  stderr and is not part of the timed region.

## Datasets

Defined in `examples/compare.rs` and `benches/compress_compare.rs`:

| Name | Shape |
|------|-------|
| `text` | repeating English phrase (long matches, few sequences) |
| `mixed` | alternating ASCII runs and pseudo-random bytes (many short matches) |
| `random` | LCG pseudo-random bytes (incompressible; isolates raw throughput) |

These are synthetic and chosen to span the easy/hard ends of the match-finder's
behavior. For decisions that ship as source constants, benchmark against a real,
named corpus as well — synthetic inputs are a smoke test, not a calibration set.

## Charts

zenbench can emit a sorted throughput bar chart in the terminal, a self-contained
HTML report (`--format=html`), or standalone SVGs (`charts` feature). Use a
sorted horizontal bar chart (MB/s) for "which is fastest", and a ratio table or
RD scatter for size-vs-level questions.

## Historical raw logs

The `*_2026-03-29.log` files in this directory are raw output from an earlier
`criterion`-based harness on 2026-03-29. They predate the current zenbench
`compress_compare` bench and have no recorded environment, so they are kept only
as historical artifacts — do not cite them as current figures. Regenerate with
the commands above.

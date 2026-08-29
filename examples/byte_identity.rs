//! Byte-identity harness: proves an encoder refactor did not move a single
//! emitted byte.
//!
//! Compression is a place where a "mechanical" rewrite can silently change
//! output — a dropped tail in a chunked loop, a reordered entropy-table write,
//! an off-by-one in a match length. None of that shows up as a compile error,
//! and round-trip tests still pass because the encoder and decoder agree with
//! each other on the *new* bytes. The only thing that catches it is hashing the
//! output before and after.
//!
//! Usage:
//!
//! ```text
//! git stash                                   # or check out the base commit
//! cargo run --release --example byte_identity -- /tmp/before
//! git stash pop
//! cargo run --release --example byte_identity -- /tmp/after
//! shasum -a 256 /tmp/{before,after}/all.bin   # must match
//! diff /tmp/{before,after}/cases.tsv          # localizes any case that moved
//! ```
//!
//! `all.bin` is every compressed output concatenated with a length prefix, so a
//! matching sha256 means every byte of every case is unchanged. `cases.tsv` is
//! one `label / length / fnv1a64` row per case, which is what tells you *which*
//! input and level moved when the sha256 does not match.
//!
//! The grid is built around the boundaries a chunked encoder loop can get wrong:
//! sizes at and around every 2/4/8/32-byte multiple, at the 128 KiB block
//! boundary, and past it; content kinds that reach both huff0 table-writing
//! branches (the FSE-coded one and the `weights.len() <= 16` direct-weight one,
//! which needs *low byte values*, not merely few distinct symbols); every
//! compression level; the streaming encoder at several write chunkings; and the
//! raw-dictionary encoder.
//!
//! Every output is also round-tripped through this crate's decoder and through
//! the C zstd reference decoder, and both counts are printed. Two known bugs
//! make some of those fail on a healthy tree, so read the counts, don't just
//! read the exit code:
//!
//!   * self round-trip at levels 16-22 — the BtOpt/BtUltra match finder is
//!     known-broken (see "L19+ decoding corruption" in CLAUDE.md).
//!   * C zstd round-trip on inputs over ~500 KB at every level except
//!     `Fastest` — the frame header under-declares the window size, so a
//!     spec-conformant decoder mis-resolves far offsets (issue #9). At the time
//!     of writing that is 112 of the 12,842 cases.
//!
//! The run exits non-zero only when *this crate's own* encoder and decoder
//! disagree at a level with no known bug, which is a self-consistency failure
//! with no innocent explanation. Gating on the two documented bugs would just
//! pin them in place.
//!
//! When you add a chunked loop to the encoder, add a content kind or size here
//! that reaches its remainder path — and check the harness actually covers it by
//! breaking the loop on purpose and confirming the case count moves.

use std::io::Write;
use zenzstd::encoding::{CompressionLevel, EncoderDictionary};

/// FNV-1a. Not cryptographic — it only has to localize which case moved. The
/// authoritative check is sha256 over `all.bin`, taken outside this program.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Lcg(u32);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}

/// Deterministic content generators. Every kind must be a pure function of
/// `(kind, n)` — the whole point is reproducing identical bytes across runs.
fn make(kind: &str, n: usize) -> Vec<u8> {
    match kind {
        // Maximally repetitive: long matches and rep-offset reuse.
        "zeros" => vec![0u8; n],
        // Incompressible: raw literals and raw blocks.
        "random" => {
            let mut r = Lcg(0x1234_5678);
            (0..n).map(|_| r.next_u8()).collect()
        }
        // huff0 writes its weight table one of two ways, and which one it picks
        // depends on the *highest literal byte value*, not the number of distinct
        // symbols: `weights` has one entry per symbol index up to the max byte,
        // minus the last. So reaching the `weights.len() <= 16` direct-weight
        // branch needs max byte <= 16, and that length's parity decides whether
        // the branch's odd-byte tail runs at all.
        //   low15 -> 15 weights (odd,  tail runs)
        //   low16 -> 16 weights (even, tail empty)
        //   low11 -> 11 weights (odd), low08 -> 8 weights (even)
        "low15" => {
            let mut r = Lcg(0x0515_0515);
            (0..n).map(|_| r.next_u8() % 16).collect()
        }
        "low16" => {
            let mut r = Lcg(0x0616_0616);
            (0..n).map(|_| r.next_u8() % 17).collect()
        }
        "low11" => {
            let mut r = Lcg(0x0b11_0b11);
            (0..n).map(|_| r.next_u8() % 12).collect()
        }
        "low08" => {
            let mut r = Lcg(0x0808_0808);
            (0..n).map(|_| r.next_u8() % 9).collect()
        }
        // Few distinct symbols but high byte values — the FSE-coded table branch.
        "binary2" => {
            let mut r = Lcg(0xdead_beef);
            (0..n)
                .map(|_| if r.next_u8() & 1 == 0 { b'a' } else { b'b' })
                .collect()
        }
        "alpha8" => {
            let mut r = Lcg(0x0bad_f00d);
            (0..n).map(|_| b'a' + (r.next_u8() % 8)).collect()
        }
        // Realistic match-length distribution: repeated words plus noise.
        "text" => {
            let words: [&[u8]; 7] = [
                b"the quick brown fox ",
                b"jumps over ",
                b"lazy dog ",
                b"pack my box ",
                b"with five dozen ",
                b"liquor jugs ",
                b"sphinx of black quartz ",
            ];
            let mut r = Lcg(0x9e37_79b9);
            let mut v = Vec::with_capacity(n);
            while v.len() < n {
                let s = r.next_u8() as usize;
                v.extend_from_slice(words[s % words.len()]);
                if s & 1 == 0 {
                    v.push(r.next_u8());
                }
            }
            v.truncate(n);
            v
        }
        // Alternating structure and noise — the pattern that first exposed the
        // L19+ optimal-parser corruption.
        "mixed" => {
            let mut r = Lcg(0xfeed_face);
            (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        b'A' + (i % 26) as u8
                    } else {
                        r.next_u8()
                    }
                })
                .collect()
        }
        // Period 37 is coprime with 2/4/8/16/32, so repeats never line up with a
        // chunk boundary.
        "period37" => (0..n).map(|i| (i % 37) as u8).collect(),
        // Dense 256-symbol alphabet, perfectly predictable.
        "counter" => (0..n).map(|i| (i % 256) as u8).collect(),
        _ => panic!("unknown content kind {kind}"),
    }
}

/// Levels 16-22 are known-broken (BtOpt/BtUltra, see CLAUDE.md), so a
/// self-round-trip failure there is expected. Below this, encoder and decoder
/// disagreeing is a hard error.
const KNOWN_GOOD_MAX_LEVEL: i32 = 15;

struct Harness {
    tsv: String,
    blob: Vec<u8>,
    dec: zenzstd::decoding::FrameDecoder,
    cases: usize,
    rt_fail: usize,
    c_fail: usize,
    c_only_fail: usize,
    hard_failures: Vec<String>,
}

impl Harness {
    fn record(&mut self, label: String, level: i32, original: &[u8], out: &[u8]) {
        self.cases += 1;
        self.tsv
            .push_str(&format!("{label}\t{}\t{:016x}\n", out.len(), fnv1a64(out)));
        self.blob
            .extend_from_slice(&(out.len() as u64).to_le_bytes());
        self.blob.extend_from_slice(out);

        let mut back = Vec::with_capacity(original.len() + 64);
        let ours = match self.dec.decode_all_to_vec(out, &mut back) {
            Ok(()) if back == original => None,
            Ok(()) => Some("output differs from input".to_string()),
            Err(e) => Some(format!("{e:?}")),
        };
        let ours_failed = ours.clone();
        if let Some(what) = ours {
            self.rt_fail += 1;
            eprintln!("ROUNDTRIP zenzstd: {label}: {what}");
            // Our own encoder and decoder disagreeing is a self-consistency
            // failure, and outside 16-22 there is no known cause for it.
            if level <= KNOWN_GOOD_MAX_LEVEL {
                self.hard_failures.push(format!("zenzstd: {label}: {what}"));
            }
        }

        let theirs = match zstd::stream::decode_all(out) {
            Ok(v) if v == original => None,
            Ok(_) => Some("output differs from input".to_string()),
            Err(e) => Some(format!("{e}")),
        };
        if let Some(what) = theirs {
            self.c_fail += 1;
            // A case only *we* decode is the interesting one: it means we
            // emitted something the reference decoder reads differently, which
            // our own decoder is lenient enough to hide.
            if ours_failed.is_none() {
                self.c_only_fail += 1;
            }
            eprintln!("ROUNDTRIP C zstd: {label}: {what}");
        }
    }

    /// Dictionary frames need the dictionary to decode, which neither decoder
    /// here is handed, so these are hashed but not round-tripped.
    fn record_no_roundtrip(&mut self, label: String, out: &[u8]) {
        self.cases += 1;
        self.tsv
            .push_str(&format!("{label}\t{}\t{:016x}\n", out.len(), fnv1a64(out)));
        self.blob
            .extend_from_slice(&(out.len() as u64).to_le_bytes());
        self.blob.extend_from_slice(out);
    }
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: byte_identity <out-dir>");
        std::process::exit(2);
    });
    std::fs::create_dir_all(&out_dir).unwrap();

    // Every 2/4/8/32-byte boundary and its neighbours, the 128 KiB block
    // boundary and its neighbours, and sizes on either side of a multi-block
    // frame.
    let sizes: &[usize] = &[
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 511, 512, 1000, 1024, 4095, 4096, 4097, 65535, 65536, 131_071, 131_072, 131_073,
        200_000, 1_000_000,
    ];
    let kinds = [
        "zeros", "random", "binary2", "alpha8", "text", "mixed", "period37", "counter", "low15",
        "low16", "low11", "low08",
    ];

    let mut levels: Vec<(String, CompressionLevel)> = vec![
        ("Uncompressed".into(), CompressionLevel::Uncompressed),
        ("Fastest".into(), CompressionLevel::Fastest),
        ("Default".into(), CompressionLevel::Default),
        ("Better".into(), CompressionLevel::Better),
        ("Best".into(), CompressionLevel::Best),
    ];
    for n in 1..=22i32 {
        levels.push((format!("L{n}"), CompressionLevel::Level(n)));
    }

    let mut h = Harness {
        tsv: String::new(),
        blob: Vec::new(),
        dec: zenzstd::decoding::FrameDecoder::new(),
        cases: 0,
        rt_fail: 0,
        c_fail: 0,
        c_only_fail: 0,
        hard_failures: Vec::new(),
    };

    for kind in kinds {
        for &n in sizes {
            let data = make(kind, n);
            for (lname, lvl) in &levels {
                let out = zenzstd::encoding::compress_to_vec(data.as_slice(), *lvl);
                h.record(
                    format!("oneshot/{kind}/{n}/{lname}"),
                    lvl.to_level(),
                    &data,
                    &out,
                );
            }
        }
    }

    // The streaming encoder drives the same MatchGeneratorDriver but buffers
    // differently, so vary the write chunking to move its internal boundaries.
    for kind in ["text", "zeros", "random", "binary2", "low15", "low16"] {
        for &n in &[0usize, 33, 1000, 131_073, 200_000] {
            let data = make(kind, n);
            for &wchunk in &[1usize, 7, 32, 4096, usize::MAX] {
                for (lname, lvl) in [
                    ("Fastest", CompressionLevel::Fastest),
                    ("Default", CompressionLevel::Default),
                    ("L11", CompressionLevel::Level(11)),
                ] {
                    let mut sink = Vec::new();
                    {
                        let mut enc = zenzstd::encoding::StreamingEncoder::new(&mut sink, lvl);
                        let mut off = 0;
                        while off < data.len() {
                            let take = wchunk.min(data.len() - off);
                            enc.write_all(&data[off..off + take]).unwrap();
                            off += take;
                        }
                        enc.finish().unwrap();
                    }
                    let wl = if wchunk == usize::MAX {
                        "all".to_string()
                    } else {
                        wchunk.to_string()
                    };
                    h.record(
                        format!("stream/{kind}/{n}/w{wl}/{lname}"),
                        lvl.to_level(),
                        &data,
                        &sink,
                    );
                }
            }
        }
    }

    // Raw dictionary: exercises dict-prefix match history.
    let dict = EncoderDictionary::new_raw(42, make("text", 8192));
    for kind in ["text", "zeros", "binary2", "low15"] {
        for &n in &[0usize, 33, 1000, 65536, 200_000] {
            let data = make(kind, n);
            for (lname, lvl) in [
                ("Fastest", CompressionLevel::Fastest),
                ("Default", CompressionLevel::Default),
                ("L11", CompressionLevel::Level(11)),
                ("L15", CompressionLevel::Level(15)),
            ] {
                let out = zenzstd::encoding::compress_to_vec_with_dict(data.as_slice(), lvl, &dict);
                h.record_no_roundtrip(format!("dict/{kind}/{n}/{lname}"), &out);
            }
        }
    }

    std::fs::write(format!("{out_dir}/cases.tsv"), &h.tsv).unwrap();
    std::fs::write(format!("{out_dir}/all.bin"), &h.blob).unwrap();

    println!(
        "cases={} blob_bytes={} blob_fnv={:016x}",
        h.cases,
        h.blob.len(),
        fnv1a64(&h.blob)
    );
    println!(
        "roundtrip failures: zenzstd={} (expected: levels 16-22 only, BtOpt/BtUltra bug, CLAUDE.md)",
        h.rt_fail
    );
    println!(
        "roundtrip failures: C-zstd={} of which {} are cases zenzstd itself decodes fine \
         (expected: 112, under-declared window on inputs >~500 KB, issue #9)",
        h.c_fail, h.c_only_fail
    );
    println!("wrote {out_dir}/cases.tsv and {out_dir}/all.bin");

    if !h.hard_failures.is_empty() {
        eprintln!(
            "\n{} case(s) where this crate's own decoder disagrees with its own encoder at \
             level <= {KNOWN_GOOD_MAX_LEVEL}, which has no known cause:",
            h.hard_failures.len()
        );
        for f in &h.hard_failures {
            eprintln!("  {f}");
        }
        std::process::exit(1);
    }
}

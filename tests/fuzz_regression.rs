//! Fuzz crash regression suite.
//!
//! Runs every file in `fuzz/regression/` through the fuzz-target entry points,
//! on the stable toolchain (no nightly / `cargo fuzz` needed). Each seed is a
//! previously-found crash that has been fixed; this test ensures none of them
//! re-introduce the failure.
//!
//! The roundtrip targets (`fuzz_dict_roundtrip`, `fuzz_roundtrip`) decode their
//! input with `arbitrary`, exactly as the fuzz harnesses do, and then assert the
//! compress -> decompress roundtrip is lossless — so this gate catches *silent
//! corruption*, not just panics. The raw decode path additionally feeds the seed
//! bytes straight to the decoder, which must never panic.
//!
//! To add a new seed: drop the (preferably minimized) crash file into
//! `fuzz/regression/` — no other action required.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use arbitrary::{Arbitrary, Unstructured};
use zenzstd::decoding::{BlockDecodingStrategy, FrameDecoder};
use zenzstd::encoding::{CompressionLevel, compress_to_vec};

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

// --- Mirror of fuzz_targets/fuzz_dict_roundtrip.rs ---------------------------
//
// The decoder-side `Dictionary` and entropy scratch types are only public under
// the `fuzz_exports` feature (the same feature the fuzz crate enables), so the
// dict roundtrip — the gate for issue #5 — is compiled only when that feature is
// on. Run with `cargo test --features fuzz_exports --test fuzz_regression` to
// exercise it; CI does this. Without the feature, the seed still flows through
// the raw decode path below (which must never panic).

#[cfg(feature = "fuzz_exports")]
#[derive(Debug, Arbitrary)]
struct DictInput {
    dict_content: Vec<u8>,
    dict_id: u32,
    level: u8,
    data: Vec<u8>,
}

#[cfg(feature = "fuzz_exports")]
impl DictInput {
    fn compression_level(&self) -> CompressionLevel {
        match self.level % 16 {
            0 => CompressionLevel::Uncompressed,
            1 => CompressionLevel::Fastest,
            3 => CompressionLevel::Default,
            7 => CompressionLevel::Better,
            11 => CompressionLevel::Best,
            n => CompressionLevel::Level(n as i32),
        }
    }
}

#[cfg(feature = "fuzz_exports")]
fn run_dict_roundtrip(seed: &[u8]) {
    use zenzstd::decoding::dictionary::Dictionary;
    use zenzstd::decoding::scratch::{FSEScratch, HuffmanScratch};
    use zenzstd::encoding::{EncoderDictionary, compress_to_vec_with_dict};

    let Ok(input) = DictInput::arbitrary(&mut Unstructured::new(seed)) else {
        return;
    };
    if input.data.len() > 256 * 1024 || input.dict_content.len() > 64 * 1024 {
        return;
    }
    if input.dict_content.is_empty() {
        return;
    }
    let dict_id = if input.dict_id == 0 { 1 } else { input.dict_id };

    let enc_dict = EncoderDictionary::new_raw(dict_id, input.dict_content.clone());
    let level = input.compression_level();
    let compressed = compress_to_vec_with_dict(input.data.as_slice(), level, &enc_dict);

    let dec_dict = Dictionary {
        id: dict_id,
        fse: FSEScratch::new(),
        huf: HuffmanScratch::new(),
        dict_content: input.dict_content.clone(),
        offset_hist: [1, 4, 8],
    };
    let mut decoder = FrameDecoder::new();
    decoder.add_dict(dec_dict).expect("add_dict must succeed");

    let mut decompressed = Vec::with_capacity(input.data.len());
    decoder
        .decode_all_to_vec(&compressed, &mut decompressed)
        .expect("dict decompression must succeed");

    assert_eq!(
        input.data, decompressed,
        "dict roundtrip mismatch at level {level:?} with dict_id {dict_id}"
    );
}

#[cfg(not(feature = "fuzz_exports"))]
fn run_dict_roundtrip(_seed: &[u8]) {}

// --- Mirror of fuzz_targets/fuzz_roundtrip.rs --------------------------------

#[derive(Debug, Arbitrary)]
struct RoundtripInput {
    level: u8,
    data: Vec<u8>,
}

impl RoundtripInput {
    fn compression_level(&self) -> CompressionLevel {
        match self.level % 16 {
            0 => CompressionLevel::Uncompressed,
            1 => CompressionLevel::Fastest,
            3 => CompressionLevel::Default,
            7 => CompressionLevel::Better,
            11 => CompressionLevel::Best,
            n => CompressionLevel::Level(n as i32),
        }
    }
}

fn run_roundtrip(seed: &[u8]) {
    let Ok(input) = RoundtripInput::arbitrary(&mut Unstructured::new(seed)) else {
        return;
    };
    if input.data.len() > 512 * 1024 {
        return;
    }
    let level = input.compression_level();
    let compressed = compress_to_vec(input.data.as_slice(), level);

    let mut decoder = FrameDecoder::new();
    let mut decompressed = Vec::with_capacity(input.data.len());
    decoder
        .decode_all_to_vec(&compressed, &mut decompressed)
        .expect("roundtrip decompression must succeed");

    assert_eq!(
        input.data, decompressed,
        "roundtrip mismatch at level {level:?}"
    );
}

// --- Mirror of fuzz_targets/fuzz_decode.rs (must never panic) ----------------

fn run_decode(seed: &[u8]) {
    {
        let mut decoder = FrameDecoder::new();
        let mut output = vec![0u8; 1024 * 1024];
        let _ = decoder.decode_all(seed, &mut output);
    }
    {
        let mut decoder = FrameDecoder::new();
        let mut cursor = seed;
        if decoder.reset(&mut cursor).is_ok() {
            let _ = decoder.decode_blocks(&mut cursor, BlockDecodingStrategy::All);
        }
    }
    {
        let mut cursor = seed;
        if let Ok(mut stream) = zenzstd::decoding::StreamingDecoder::new(&mut cursor) {
            let mut output = Vec::new();
            let _ = stream.read_to_end(&mut output);
        }
    }
}

#[test]
fn fuzz_regression_seeds() {
    let dir = regression_dir();
    let entries: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect();

    assert!(
        !entries.is_empty(),
        "fuzz/regression/ is empty — at least one fixed-crash seed must be present"
    );

    for entry in entries {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>");
        let seed = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));

        // Run the seed through every fuzz entry point. A panic (or a roundtrip
        // assertion failure) here means a previously-fixed bug has regressed;
        // the seed name is in the unwind message.
        run_dict_roundtrip(&seed);
        run_roundtrip(&seed);
        run_decode(&seed);

        eprintln!("ok: {name} ({} bytes)", seed.len());
    }
}

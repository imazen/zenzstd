//! Tests for the crate-root one-shot [`crate::compress`] / [`crate::decompress`]
//! helpers, in particular [`crate::DecompressError`]'s three variants.
//!
//! Imports are declared inside each `#[test]` fn (rather than at module scope)
//! to match this crate's convention: `#[test]`-attributed bodies are only
//! type-checked under `--cfg test`, so module-level `use`s that are only
//! referenced from test bodies show up as spurious `unused_imports` warnings
//! in a plain (non-test) `cargo build`.

#[test]
fn roundtrip_ok() {
    use crate::CompressionLevel;

    let data: &[u8] = b"the quick brown fox jumps over the lazy dog, again and again";
    let compressed = crate::compress(data, CompressionLevel::Default);
    let restored = crate::decompress(&compressed, 64 * 1024).unwrap();
    assert_eq!(restored, data);
}

#[test]
fn invalid_input_is_invalid_input_variant() {
    use crate::DecompressError;

    // Not a zstd frame at all: bad magic number.
    let garbage = [0u8, 1, 2, 3, 4, 5, 6, 7];
    let err = crate::decompress(&garbage, 1024).expect_err("garbage input must not decode");
    assert!(
        matches!(err, DecompressError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn output_size_exceeded_is_output_size_exceeded_variant() {
    use crate::{CompressionLevel, DecompressError};
    use alloc::vec;

    // 16 MiB of zeros compresses to a tiny RLE-heavy frame -- a real "bomb".
    let bomb_raw = vec![0u8; 16 * 1024 * 1024];
    let compressed = crate::compress(&bomb_raw, CompressionLevel::Fastest);
    assert!(
        compressed.len() < bomb_raw.len() / 100,
        "expected a high compression ratio, got {} -> {}",
        compressed.len(),
        bomb_raw.len()
    );

    let cap = 1024 * 1024;
    let err = crate::decompress(&compressed, cap)
        .expect_err("decompress must reject output past the cap");
    match err {
        DecompressError::OutputSizeExceeded { max_output_size } => {
            assert_eq!(max_output_size, cap);
        }
        other => panic!("expected OutputSizeExceeded, got {other:?}"),
    }
}

#[test]
fn output_size_cap_at_exact_boundary() {
    use crate::{CompressionLevel, DecompressError};
    use alloc::vec;

    // Cap exactly matching the decompressed length must succeed; one byte
    // short must fail with OutputSizeExceeded. Exercises the boundary the fix
    // relies on to distinguish a cap-hit from a genuine decompressor error.
    let raw = vec![0xABu8; 4096];
    let compressed = crate::compress(&raw, CompressionLevel::Fastest);

    let restored = crate::decompress(&compressed, raw.len()).unwrap();
    assert_eq!(restored, raw);

    let err = crate::decompress(&compressed, raw.len() - 1)
        .expect_err("one byte under the true output size must fail");
    assert!(
        matches!(
            err,
            DecompressError::OutputSizeExceeded { max_output_size } if max_output_size == raw.len() - 1
        ),
        "expected OutputSizeExceeded, got {err:?}"
    );
}

#[test]
fn corrupt_block_content_is_decompressor_variant() {
    use crate::{CompressionLevel, DecompressError};
    use alloc::vec::Vec;

    // A valid frame header followed by truncated block content: the frame
    // parses fine (so this is not InvalidInput), but decoding the body fails
    // partway through -- and it fails well before the real output size,
    // so it must not be misclassified as OutputSizeExceeded either.
    let raw: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    let compressed = crate::compress(&raw, CompressionLevel::Default);
    assert!(
        compressed.len() > 64,
        "need a compressed frame long enough to truncate meaningfully"
    );

    let truncated = &compressed[..compressed.len() / 2];
    let err = crate::decompress(truncated, raw.len() * 2)
        .expect_err("truncated block content must not decode");
    assert!(
        matches!(err, DecompressError::Decompressor(_)),
        "expected Decompressor, got {err:?}"
    );
}

#[test]
fn decompress_error_implements_display_and_std_error() {
    let garbage = [0u8, 1, 2, 3, 4, 5, 6, 7];
    let err = crate::decompress(&garbage, 1024).unwrap_err();
    // Must produce a non-empty, human-readable message (not a raw `{:?}` dump
    // of an unrelated type).
    let message = alloc::format!("{err}");
    assert!(!message.is_empty());

    #[cfg(feature = "std")]
    {
        let _: &dyn std::error::Error = &err;
    }
}

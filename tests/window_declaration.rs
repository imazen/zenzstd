//! The frame header's `Window_Size` must cover every offset the encoder emits.
//!
//! A decoder sizes its history buffer from the `Window_Descriptor` byte and
//! throws away everything older. Under-declare it and a conformant decoder
//! resolves far back-references against bytes it no longer has — silent
//! corruption, not a decode error. That was issue #9: the header said 128 KiB
//! at every level while the match finder reached up to 4 MiB back.
//!
//! This crate's own decoder cannot catch it. It keeps more than it was told to
//! keep, so encoder and decoder agree with each other on a stream neither is
//! entitled to produce. Both tests here are built to be immune to that:
//!
//! * [`declared_window_matches_the_c_level_table`] reads the descriptor byte
//!   straight out of the emitted frame and compares it against a table written
//!   from C zstd's `clevels.h`, independently of whatever the encoder computed.
//! * [`far_offsets_survive_the_c_reference_decoder`] round-trips input that
//!   forces a match further back than 128 KiB through the reference decoder,
//!   which honours the declared window exactly.

use zenzstd::encoding::{CompressionLevel, EncoderDictionary, StreamingEncoder};
use zenzstd::encoding::{compress_to_vec, compress_to_vec_with_dict};

/// `Window_Size` as a decoder reads it: out of the frame's own bytes.
///
/// Layout: magic (4) + `Frame_Header_Descriptor` (1) + `Window_Descriptor` (1,
/// present because our encoder never sets `Single_Segment_flag`).
/// <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#window_descriptor>
fn declared_window(frame: &[u8]) -> u64 {
    assert!(
        frame.len() >= 6,
        "frame too short to hold a window descriptor"
    );
    assert_eq!(
        &frame[..4],
        &0xFD2FB528u32.to_le_bytes(),
        "not a zstd frame"
    );
    let descriptor = frame[4];
    assert_eq!(
        (descriptor >> 5) & 1,
        0,
        "Single_Segment_flag set: there is no Window_Descriptor to read"
    );
    let window_descriptor = frame[5];
    let exponent = u64::from(window_descriptor >> 3);
    let mantissa = u64::from(window_descriptor & 0x7);
    let base = 1u64 << (10 + exponent);
    base + (base / 8) * mantissa
}

/// `window_log` per level from C zstd's `clevels.h` default table (the table
/// used when the source size is unknown, which is always the case here because
/// the encoder takes a `Read`), capped at the 8 MiB the format spec asks
/// encoders to stay within.
///
/// Written out longhand on purpose: if this agreed with the encoder by calling
/// into it, it would pass no matter what the encoder did.
fn expected_window_log(level: i32) -> u32 {
    let uncapped = match level {
        1 => 19,
        2 => 20,
        3..=8 => 21,
        9..=16 => 22,
        17..=19 => 23,
        20 => 25,
        21 => 26,
        22 => 27,
        other => panic!("level {other} out of range"),
    };
    uncapped.min(23)
}

/// The same LCG `examples/byte_identity.rs` uses, so the two harnesses probe
/// the same byte streams.
struct Lcg(u32);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}

/// `byte_identity`'s "text" kind: repeated words plus noise, i.e. a realistic
/// match-length distribution.
fn text(n: usize) -> Vec<u8> {
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

/// `byte_identity`'s "low15" kind: a 16-symbol alphabet of low byte values.
fn low15(n: usize) -> Vec<u8> {
    let mut r = Lcg(0x0515_0515);
    (0..n).map(|_| r.next_u8() % 16).collect()
}

/// Inputs that actually make the encoder reach back past 128 KiB, which is
/// harder to arrange than it looks and is the whole reason this test is worth
/// anything.
///
/// Two things have to hold at once:
///
/// 1. The match finder has to *choose* an offset over 128 KiB. Long stretches
///    of highly repetitive filler do not do it — they collapse into near
///    matches and leave the far one unused. A first attempt here (a
///    distinctive 48 KiB marker repeated across 144 KiB of period-3 filler)
///    moved zero cases under the mutation below, at any level.
/// 2. The frame has to be long enough that a decoder honouring a 128 KiB
///    declaration has genuinely dropped the bytes. C zstd's streaming decoder
///    sizes its buffer around `Window_Size + Block_Size`, so a short frame
///    keeps everything it ever decoded regardless of what the header said.
///    That is why the original report's onset was ~500 KB, not 128 KiB.
///
/// Verified by mutation (see the module docs on `header_window_size`): with the
/// header reverted to the buggy fixed 128 KiB, `text` at 1,000,000 bytes fails
/// the reference decoder at levels 1-5 and 7-15, and `low15` at 700,000 bytes
/// fails at 3-15 — level 6 being the one only `low15` catches. Between them
/// every level in range is covered.
fn far_offset_inputs() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("text/1000000", text(1_000_000)),
        ("low15/700000", low15(700_000)),
    ]
}

/// Levels 16-22 use the BtOpt/BtUltra optimal parser, which is separately
/// known to emit sequences that do not reconstruct the input (see "L19+
/// decoding corruption" under Known Issues in CLAUDE.md). Round-tripping them
/// here would assert that unrelated bug rather than the window declaration, so
/// the round-trip test stops below it — the header check above still covers
/// every level up to 22.
const HIGHEST_ROUNDTRIPPABLE_LEVEL: i32 = 15;

#[test]
fn declared_window_matches_the_c_level_table() {
    // Small input: the declared window is a property of the level, not of how
    // much data happens to show up.
    let data = b"the quick brown fox jumps over the lazy dog".repeat(64);

    for level in 1..=22i32 {
        let frame = compress_to_vec(data.as_slice(), CompressionLevel::Level(level));
        let expected = 1u64 << expected_window_log(level);
        assert_eq!(
            declared_window(&frame),
            expected,
            "level {level} declares the wrong window"
        );
    }

    // The named variants map onto numbered levels; `Fastest` is the odd one
    // out because it matches through `MatchGeneratorDriver`, whose window is a
    // fixed 128 KiB regardless of level.
    for (name, level, expected) in [
        ("Fastest", CompressionLevel::Fastest, 128 * 1024),
        ("Default", CompressionLevel::Default, 1 << 21),
        ("Better", CompressionLevel::Better, 1 << 21),
        ("Best", CompressionLevel::Best, 1 << 22),
        ("Uncompressed", CompressionLevel::Uncompressed, 128 * 1024),
    ] {
        let frame = compress_to_vec(data.as_slice(), level);
        assert_eq!(
            declared_window(&frame),
            expected,
            "{name} declares the wrong window"
        );
    }
}

#[test]
fn streaming_encoder_declares_the_same_window() {
    use std::io::Write;

    let data = b"the quick brown fox jumps over the lazy dog".repeat(64);
    let levels = (1..=22i32).map(CompressionLevel::Level).chain([
        CompressionLevel::Fastest,
        CompressionLevel::Default,
        CompressionLevel::Better,
        CompressionLevel::Best,
        CompressionLevel::Uncompressed,
    ]);

    for level in levels {
        let mut streamed = Vec::new();
        {
            let mut enc = StreamingEncoder::new(&mut streamed, level);
            enc.write_all(&data).unwrap();
            enc.finish().unwrap();
        }
        let oneshot = compress_to_vec(data.as_slice(), level);
        assert_eq!(
            declared_window(&streamed),
            declared_window(&oneshot),
            "{level:?}: streaming and one-shot encoders disagree on the window"
        );
    }
}

#[test]
fn dictionary_frames_declare_the_same_window() {
    // The dictionary ID is serialized *after* the window descriptor, so the
    // descriptor stays at byte 5 — but the dictionary also seeds the match
    // window, so the declaration has to hold for these frames too.
    let dict = EncoderDictionary::new_raw(
        42,
        b"the quick brown fox jumps over the lazy dog ".repeat(128),
    );
    let data = b"pack my box with five dozen liquor jugs ".repeat(256);

    for level in 1..=22i32 {
        let level = CompressionLevel::Level(level);
        let frame = compress_to_vec_with_dict(data.as_slice(), level, &dict);
        let plain = compress_to_vec(data.as_slice(), level);
        assert_eq!(
            declared_window(&frame),
            declared_window(&plain),
            "{level:?}: dictionary frame declares a different window"
        );
    }
}

#[test]
fn far_offsets_survive_the_c_reference_decoder() {
    let mut levels: Vec<CompressionLevel> = (1..=HIGHEST_ROUNDTRIPPABLE_LEVEL)
        .map(CompressionLevel::Level)
        .collect();
    levels.extend([
        CompressionLevel::Fastest,
        CompressionLevel::Default,
        CompressionLevel::Better,
        CompressionLevel::Best,
        CompressionLevel::Uncompressed,
    ]);

    for (kind, data) in far_offset_inputs() {
        for &level in &levels {
            let frame = compress_to_vec(data.as_slice(), level);
            let window = declared_window(&frame);

            // Our own decoder keeps more history than it is told to, so it
            // agreeing proves nothing about the declaration. The reference
            // decoder keeps exactly `window` bytes, so it is the one that
            // catches an offset reaching past it.
            let back = zstd::stream::decode_all(frame.as_slice()).unwrap_or_else(|e| {
                panic!("{kind} {level:?} (declared window {window}): C zstd: {e}")
            });
            assert!(
                back == data,
                "{kind} {level:?} (declared window {window}): C zstd decoded different bytes"
            );
        }
    }
}

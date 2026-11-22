#[cfg(test)]
use alloc::vec;

#[cfg(test)]
use alloc::vec::Vec;

#[cfg(test)]
extern crate std;

#[cfg(all(test, not(feature = "std")))]
impl crate::io_nostd::Read for std::fs::File {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, crate::io_nostd::Error> {
        std::io::Read::read(self, buf).map_err(|e| {
            if e.get_ref().is_none() {
                crate::io_nostd::Error::from(crate::io_nostd::ErrorKind::Other)
            } else {
                crate::io_nostd::Error::new(
                    crate::io_nostd::ErrorKind::Other,
                    alloc::boxed::Box::new(e.into_inner().unwrap()),
                )
            }
        })
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(dead_code)]
fn assure_error_impl() {
    // not a real test just there to throw an compiler error if Error is not derived correctly

    use crate::decoding::errors::FrameDecoderError;
    let _err: &dyn std::error::Error = &FrameDecoderError::NotYetInitialized;
}

#[cfg(all(test, feature = "std"))]
#[allow(dead_code)]
fn assure_decoder_send_sync() {
    // not a real test just there to throw an compiler error if FrameDecoder is Send + Sync

    use crate::decoding::FrameDecoder;
    let decoder = FrameDecoder::new();
    std::thread::spawn(move || {
        drop(decoder);
    });
}

#[test]
fn skippable_frame() {
    use crate::decoding::errors;
    use crate::decoding::frame;

    let mut content = vec![];
    content.extend_from_slice(&0x184D2A50u32.to_le_bytes());
    content.extend_from_slice(&300u32.to_le_bytes());
    assert_eq!(8, content.len());
    let err = frame::read_frame_header(content.as_slice());
    assert!(matches!(
        err,
        Err(errors::ReadFrameHeaderError::SkipFrame {
            magic_number: 0x184D2A50u32,
            length: 300
        })
    ));

    content.clear();
    content.extend_from_slice(&0x184D2A5Fu32.to_le_bytes());
    content.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes());
    assert_eq!(8, content.len());
    let err = frame::read_frame_header(content.as_slice());
    assert!(matches!(
        err,
        Err(errors::ReadFrameHeaderError::SkipFrame {
            magic_number: 0x184D2A5Fu32,
            length: 0xFFFFFFFF
        })
    ));
}

#[cfg(test)]
#[test]
fn test_frame_header_reading() {
    use crate::decoding::frame;
    use std::fs;

    let mut content = fs::File::open("./decodecorpus_files/z000088.zst").unwrap();
    let (_frame, _) = frame::read_frame_header(&mut content).unwrap();
}

#[test]
fn test_block_header_reading() {
    use crate::decoding;
    use crate::decoding::frame;
    use std::fs;

    let mut content = fs::File::open("./decodecorpus_files/z000088.zst").unwrap();
    let (_frame, _) = frame::read_frame_header(&mut content).unwrap();

    let mut block_dec = decoding::block_decoder::new();
    let block_header = block_dec.read_block_header(&mut content).unwrap();
    let _ = block_header; //TODO validate blockheader in a smart way
}

#[test]
fn test_frame_decoder() {
    use crate::decoding::BlockDecodingStrategy;
    use crate::decoding::FrameDecoder;
    use std::fs;

    let mut content = fs::File::open("./decodecorpus_files/z000088.zst").unwrap();

    struct NullWriter(());
    impl std::io::Write for NullWriter {
        fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> Result<(), std::io::Error> {
            Ok(())
        }
    }
    let mut _null_target = NullWriter(());

    let mut frame_dec = FrameDecoder::new();
    frame_dec.reset(&mut content).unwrap();
    frame_dec
        .decode_blocks(&mut content, BlockDecodingStrategy::All)
        .unwrap();
}

#[test]
fn test_decode_from_to() {
    use crate::decoding::FrameDecoder;
    use std::fs::File;
    use std::io::BufReader;
    use std::io::Read;
    let f = BufReader::new(File::open("./decodecorpus_files/z000088.zst").unwrap());
    let mut frame_dec = FrameDecoder::new();

    let content: Vec<u8> = f.bytes().map(|x| x.unwrap()).collect();

    let mut target = vec![0u8; 1024 * 1024];

    // first part
    let source1 = &content[..50 * 1024];
    let (read1, written1) = frame_dec
        .decode_from_to(source1, target.as_mut_slice())
        .unwrap();

    //second part explicitely without checksum
    let source2 = &content[read1..content.len() - 4];
    let (read2, written2) = frame_dec
        .decode_from_to(source2, &mut target[written1..])
        .unwrap();

    //must have decoded until checksum
    assert!(read1 + read2 == content.len() - 4);

    //insert checksum separatly to test that this is handled correctly
    let chksum_source = &content[read1 + read2..];
    let (read3, written3) = frame_dec
        .decode_from_to(chksum_source, &mut target[written1 + written2..])
        .unwrap();

    //this must result in these values because just the checksum was processed
    assert!(read3 == 4);
    assert!(written3 == 0);

    let read = read1 + read2 + read3;
    let written = written1 + written2;

    let result = &target.as_slice()[..written];

    if read != content.len() {
        panic!(
            "Byte counter: {} was wrong. Should be: {}",
            read,
            content.len()
        );
    }

    match frame_dec.get_checksum_from_data() {
        Some(chksum) => {
            #[cfg(feature = "hash")]
            if frame_dec.get_calculated_checksum().unwrap() != chksum {
                std::println!(
                    "Checksum did not match! From data: {}, calculated while decoding: {}\n",
                    chksum,
                    frame_dec.get_calculated_checksum().unwrap()
                );
            } else {
                std::println!("Checksums are ok!\n");
            }
            #[cfg(not(feature = "hash"))]
            std::println!(
                "Checksum feature not enabled, skipping. From data: {}\n",
                chksum
            );
        }
        None => std::println!("No checksums to test\n"),
    }

    let original_f = BufReader::new(File::open("./decodecorpus_files/z000088").unwrap());
    let original: Vec<u8> = original_f.bytes().map(|x| x.unwrap()).collect();

    if original.len() != result.len() {
        panic!(
            "Result has wrong length: {}, should be: {}",
            result.len(),
            original.len()
        );
    }

    let mut counter = 0;
    let min = if original.len() < result.len() {
        original.len()
    } else {
        result.len()
    };
    for idx in 0..min {
        if original[idx] != result[idx] {
            counter += 1;
            //std::println!(
            //    "Original {:3} not equal to result {:3} at byte: {}",
            //    original[idx], result[idx], idx,
            //);
        }
    }
    if counter > 0 {
        panic!("Result differs in at least {} bytes from original", counter);
    }
}

#[test]
fn test_specific_file() {
    use crate::decoding::BlockDecodingStrategy;
    use crate::decoding::FrameDecoder;
    use std::fs;
    use std::io::BufReader;
    use std::io::Read;

    let path = "./decodecorpus_files/z000068.zst";
    let mut content = fs::File::open(path).unwrap();

    struct NullWriter(());
    impl std::io::Write for NullWriter {
        fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> Result<(), std::io::Error> {
            Ok(())
        }
    }
    let mut _null_target = NullWriter(());

    let mut frame_dec = FrameDecoder::new();
    frame_dec.reset(&mut content).unwrap();
    frame_dec
        .decode_blocks(&mut content, BlockDecodingStrategy::All)
        .unwrap();
    let result = frame_dec.collect().unwrap();

    let original_f = BufReader::new(fs::File::open("./decodecorpus_files/z000088").unwrap());
    let original: Vec<u8> = original_f.bytes().map(|x| x.unwrap()).collect();

    std::println!("Results for file: {path}");

    if original.len() != result.len() {
        std::println!(
            "Result has wrong length: {}, should be: {}",
            result.len(),
            original.len()
        );
    }

    let mut counter = 0;
    let min = if original.len() < result.len() {
        original.len()
    } else {
        result.len()
    };
    for idx in 0..min {
        if original[idx] != result[idx] {
            counter += 1;
            //std::println!(
            //    "Original {:3} not equal to result {:3} at byte: {}",
            //    original[idx], result[idx], idx,
            //);
        }
    }
    if counter > 0 {
        std::println!("Result differs in at least {counter} bytes from original");
    }
}

#[test]
#[cfg(feature = "std")]
fn test_streaming() {
    use std::fs;
    use std::io::BufReader;
    use std::io::Read;

    let mut content = fs::File::open("./decodecorpus_files/z000088.zst").unwrap();
    let mut stream = crate::decoding::StreamingDecoder::new(&mut content).unwrap();

    let mut result = Vec::new();
    Read::read_to_end(&mut stream, &mut result).unwrap();

    let original_f = BufReader::new(fs::File::open("./decodecorpus_files/z000088").unwrap());
    let original: Vec<u8> = original_f.bytes().map(|x| x.unwrap()).collect();

    if original.len() != result.len() {
        panic!(
            "Result has wrong length: {}, should be: {}",
            result.len(),
            original.len()
        );
    }

    let mut counter = 0;
    let min = if original.len() < result.len() {
        original.len()
    } else {
        result.len()
    };
    for idx in 0..min {
        if original[idx] != result[idx] {
            counter += 1;
            //std::println!(
            //    "Original {:3} not equal to result {:3} at byte: {}",
            //    original[idx], result[idx], idx,
            //);
        }
    }
    if counter > 0 {
        panic!("Result differs in at least {} bytes from original", counter);
    }

    // Test resetting to a new file while keeping the old decoder

    let mut content = fs::File::open("./decodecorpus_files/z000068.zst").unwrap();
    let mut stream = crate::decoding::StreamingDecoder::new_with_decoder(
        &mut content,
        stream.into_frame_decoder(),
    )
    .unwrap();

    let mut result = Vec::new();
    Read::read_to_end(&mut stream, &mut result).unwrap();

    let original_f = BufReader::new(fs::File::open("./decodecorpus_files/z000068").unwrap());
    let original: Vec<u8> = original_f.bytes().map(|x| x.unwrap()).collect();

    std::println!("Results for file:");

    if original.len() != result.len() {
        panic!(
            "Result has wrong length: {}, should be: {}",
            result.len(),
            original.len()
        );
    }

    let mut counter = 0;
    let min = if original.len() < result.len() {
        original.len()
    } else {
        result.len()
    };
    for idx in 0..min {
        if original[idx] != result[idx] {
            counter += 1;
            //std::println!(
            //    "Original {:3} not equal to result {:3} at byte: {}",
            //    original[idx], result[idx], idx,
            //);
        }
    }
    if counter > 0 {
        panic!("Result differs in at least {} bytes from original", counter);
    }
}

#[test]
fn test_incremental_read() {
    use crate::decoding::FrameDecoder;

    let mut unread_compressed_content =
        include_bytes!("../../decodecorpus_files/abc.txt.zst").as_slice();

    let mut frame_dec = FrameDecoder::new();
    frame_dec.reset(&mut unread_compressed_content).unwrap();

    let mut output = [0u8; 3];
    let (_, written) = frame_dec
        .decode_from_to(unread_compressed_content, &mut output)
        .unwrap();

    assert_eq!(written, 3);
    assert_eq!(output.map(char::from), ['a', 'b', 'c']);

    assert!(frame_dec.is_finished());
    let written = frame_dec.collect_to_writer(&mut &mut output[..]).unwrap();
    assert_eq!(written, 3);
    assert_eq!(output.map(char::from), ['d', 'e', 'f']);
}

#[test]
#[cfg(not(feature = "std"))]
fn test_streaming_no_std() {
    use crate::io::Read;

    let content = include_bytes!("../../decodecorpus_files/z000088.zst");
    let mut content = content.as_slice();
    let mut stream = crate::decoding::StreamingDecoder::new(&mut content).unwrap();

    let original = include_bytes!("../../decodecorpus_files/z000088");
    let mut result = vec![0; original.len()];
    Read::read_exact(&mut stream, &mut result).unwrap();

    if original.len() != result.len() {
        panic!(
            "Result has wrong length: {}, should be: {}",
            result.len(),
            original.len()
        );
    }

    let mut counter = 0;
    let min = if original.len() < result.len() {
        original.len()
    } else {
        result.len()
    };
    for idx in 0..min {
        if original[idx] != result[idx] {
            counter += 1;
            //std::println!(
            //    "Original {:3} not equal to result {:3} at byte: {}",
            //    original[idx], result[idx], idx,
            //);
        }
    }
    if counter > 0 {
        panic!("Result differs in at least {} bytes from original", counter);
    }

    // Test resetting to a new file while keeping the old decoder

    let content = include_bytes!("../../decodecorpus_files/z000068.zst");
    let mut content = content.as_slice();
    let mut stream = crate::decoding::StreamingDecoder::new_with_decoder(
        &mut content,
        stream.into_frame_decoder(),
    )
    .unwrap();

    let original = include_bytes!("../../decodecorpus_files/z000068");
    let mut result = vec![0; original.len()];
    Read::read_exact(&mut stream, &mut result).unwrap();

    std::println!("Results for file:");

    if original.len() != result.len() {
        panic!(
            "Result has wrong length: {}, should be: {}",
            result.len(),
            original.len()
        );
    }

    let mut counter = 0;
    let min = if original.len() < result.len() {
        original.len()
    } else {
        result.len()
    };
    for idx in 0..min {
        if original[idx] != result[idx] {
            counter += 1;
            //std::println!(
            //    "Original {:3} not equal to result {:3} at byte: {}",
            //    original[idx], result[idx], idx,
            //);
        }
    }
    if counter > 0 {
        panic!("Result differs in at least {} bytes from original", counter);
    }
}

#[test]
fn test_decode_all() {
    use crate::decoding::FrameDecoder;
    use crate::decoding::errors::FrameDecoderError;

    let skip_frame = |input: &mut Vec<u8>, length: usize| {
        input.extend_from_slice(&0x184D2A50u32.to_le_bytes());
        input.extend_from_slice(&(length as u32).to_le_bytes());
        input.resize(input.len() + length, 0);
    };

    let mut original = Vec::new();
    let mut input = Vec::new();

    skip_frame(&mut input, 300);
    input.extend_from_slice(include_bytes!("../../decodecorpus_files/z000089.zst"));
    original.extend_from_slice(include_bytes!("../../decodecorpus_files/z000089"));
    skip_frame(&mut input, 400);
    input.extend_from_slice(include_bytes!("../../decodecorpus_files/z000090.zst"));
    original.extend_from_slice(include_bytes!("../../decodecorpus_files/z000090"));
    skip_frame(&mut input, 500);

    let mut decoder = FrameDecoder::new();

    // decode_all with correct buffers.
    let mut output = vec![0; original.len()];
    let result = decoder.decode_all(&input, &mut output).unwrap();
    assert_eq!(result, original.len());
    assert_eq!(output, original);

    // decode_all with smaller output length.
    let mut output = vec![0; original.len() - 1];
    let result = decoder.decode_all(&input, &mut output);
    assert!(
        matches!(result, Err(FrameDecoderError::TargetTooSmall)),
        "{:?}",
        result
    );

    // decode_all with larger output length.
    let mut output = vec![0; original.len() + 1];
    let result = decoder.decode_all(&input, &mut output).unwrap();
    assert_eq!(result, original.len());
    assert_eq!(&output[..result], original);

    // decode_all with truncated regular frame.
    let mut output = vec![0; original.len()];
    let result = decoder.decode_all(&input[..input.len() - 600], &mut output);
    assert!(
        matches!(result, Err(FrameDecoderError::FailedToReadBlockBody(_))),
        "{:?}",
        result
    );

    // decode_all with truncated skip frame.
    let mut output = vec![0; original.len()];
    let result = decoder.decode_all(&input[..input.len() - 1], &mut output);
    assert!(
        matches!(result, Err(FrameDecoderError::FailedToSkipFrame)),
        "{:?}",
        result
    );

    // decode_all_to_vec with correct output capacity.
    let mut output = Vec::new();
    output.reserve_exact(original.len());
    decoder.decode_all_to_vec(&input, &mut output).unwrap();
    assert_eq!(output, original);

    // decode_all_to_vec with smaller output capacity.
    let mut output = Vec::new();
    output.reserve_exact(original.len() - 1);
    let result = decoder.decode_all_to_vec(&input, &mut output);
    assert!(
        matches!(result, Err(FrameDecoderError::TargetTooSmall)),
        "{:?}",
        result
    );

    // decode_all_to_vec with larger output capacity.
    let mut output = Vec::new();
    output.reserve_exact(original.len() + 1);
    decoder.decode_all_to_vec(&input, &mut output).unwrap();
    assert_eq!(output, original);
}

pub mod bit_reader;
#[cfg(feature = "std")]
pub mod conformance;
pub mod decode_corpus;
pub mod dict_test;
#[cfg(feature = "std")]
pub mod encode_corpus;
pub mod fuzz_regressions;
#[cfg(feature = "std")]
pub mod level_roundtrip;

#[cfg(feature = "std")]
#[test]
fn verbose_disabled() {
    use crate::VERBOSE;
    const { assert!(!VERBOSE) };
}

#[test]
fn window_limit_enforced_on_init_and_reset() {
    use crate::decoding::errors::FrameDecoderError;
    use crate::decoding::{BlockDecodingStrategy, FrameDecoder, StreamingDecoder};

    // Valid empty zstd frame (1 KiB window, last empty raw block).
    let valid_frame: [u8; 9] = [0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x00, 0x01, 0x00, 0x00];
    // Exact boundary: 128 MiB (window_log = 27, exp = 17, mant = 0 => 134_217_728 bytes)
    // and single-segment 134_217_728 bytes (4-byte FCS, no Window_Descriptor).
    let exact_128m: [u8; 9] = [0x28, 0xB5, 0x2F, 0xFD, 0x00, 17 << 3, 0x01, 0x00, 0x00];
    let exact_128m_single_segment: [u8; 12] = [
        0x28, 0xB5, 0x2F, 0xFD, 0xA0, 0x00, 0x00, 0x00, 0x08, 0x01, 0x00, 0x00,
    ];
    // Over-limit frames (> 128 MiB):
    // - 144 MiB: exp = 17, mant = 1 => 150_994_944 bytes
    // - 256 MiB: window_log = 28, exp = 18, mant = 0 => 268_435_456 bytes
    // - Single-segment 134_217_729 bytes (128 MiB + 1)
    let over_limit_144m: [u8; 9] = [
        0x28,
        0xB5,
        0x2F,
        0xFD,
        0x00,
        (17 << 3) | 1,
        0x01,
        0x00,
        0x00,
    ];
    let over_limit_256m: [u8; 9] = [0x28, 0xB5, 0x2F, 0xFD, 0x00, 18 << 3, 0x01, 0x00, 0x00];
    let over_limit_single_segment: [u8; 12] = [
        0x28, 0xB5, 0x2F, 0xFD, 0xA0, 0x01, 0x00, 0x00, 0x08, 0x01, 0x00, 0x00,
    ];

    let over_limit_cases: [(&str, &[u8], u64); 3] = [
        ("144 MiB", &over_limit_144m, 150_994_944),
        ("256 MiB", &over_limit_256m, 268_435_456),
        (
            "single-segment 128 MiB + 1",
            &over_limit_single_segment,
            134_217_729,
        ),
    ];

    let assert_too_big = |res: Result<(), FrameDecoderError>, expected: u64, ctx: &str| {
        assert!(
            matches!(res, Err(FrameDecoderError::WindowSizeTooBig { requested }) if requested == expected),
            "{ctx}: expected WindowSizeTooBig({expected}), got {res:?}"
        );
    };

    // 0. Exact 128 MiB boundary succeeds on fresh init and reused reset/init + empty block decode.
    for (label, frame) in [
        ("128 MiB descriptor", &exact_128m[..]),
        ("128 MiB single-segment", &exact_128m_single_segment[..]),
    ] {
        let mut dec = FrameDecoder::new();
        let mut src = frame;
        dec.init(&mut src)
            .unwrap_or_else(|e| panic!("fresh init {label}: {e:?}"));
        assert!(
            dec.decode_blocks(&mut src, BlockDecodingStrategy::All)
                .unwrap()
        );
        assert_eq!(dec.collect().unwrap(), &[]);
    }
    {
        let mut dec = FrameDecoder::new();
        let mut out = [0u8; 16];
        assert_eq!(dec.decode_all(&valid_frame, &mut out).unwrap(), 0);
        for (label, frame) in [
            ("128 MiB descriptor", &exact_128m[..]),
            ("128 MiB single-segment", &exact_128m_single_segment[..]),
        ] {
            let mut src = frame;
            dec.reset(&mut src)
                .unwrap_or_else(|e| panic!("reused reset {label}: {e:?}"));
            assert!(
                dec.decode_blocks(&mut src, BlockDecodingStrategy::All)
                    .unwrap()
            );
            assert_eq!(dec.collect().unwrap(), &[]);

            let mut src = frame;
            dec.init(&mut src)
                .unwrap_or_else(|e| panic!("reused init {label}: {e:?}"));
            assert!(
                dec.decode_blocks(&mut src, BlockDecodingStrategy::All)
                    .unwrap()
            );
            assert_eq!(dec.collect().unwrap(), &[]);
        }
    }

    let mut out = [0u8; 16];
    let mut out_vec = Vec::new();

    // 1. Fresh FrameDecoder paths (hit FrameDecoderState::new).
    for (label, frame, expected) in over_limit_cases {
        assert_too_big(
            FrameDecoder::new().init(&mut { frame }),
            expected,
            &alloc::format!("fresh FrameDecoder::init ({label})"),
        );
        assert_too_big(
            FrameDecoder::new().reset(&mut { frame }),
            expected,
            &alloc::format!("fresh FrameDecoder::reset ({label})"),
        );
        assert_too_big(
            FrameDecoder::new()
                .decode_from_to(frame, &mut out)
                .map(|_| ()),
            expected,
            &alloc::format!("fresh FrameDecoder::decode_from_to ({label})"),
        );
        assert_too_big(
            FrameDecoder::new().decode_all(frame, &mut out).map(|_| ()),
            expected,
            &alloc::format!("fresh FrameDecoder::decode_all ({label})"),
        );
        assert_too_big(
            FrameDecoder::new().decode_all_to_vec(frame, &mut out_vec),
            expected,
            &alloc::format!("fresh FrameDecoder::decode_all_to_vec ({label})"),
        );
    }

    // 2. Reused FrameDecoder paths (hit FrameDecoderState::reset).
    let mut reused = FrameDecoder::new();
    assert_eq!(reused.decode_all(&valid_frame, &mut out).unwrap(), 0);
    for (label, frame, expected) in over_limit_cases {
        assert_too_big(
            reused.reset(&mut { frame }),
            expected,
            &alloc::format!("reused FrameDecoder::reset ({label})"),
        );
        assert_too_big(
            reused.init(&mut { frame }),
            expected,
            &alloc::format!("reused FrameDecoder::init ({label})"),
        );
        assert_too_big(
            reused.decode_all(frame, &mut out).map(|_| ()),
            expected,
            &alloc::format!("reused FrameDecoder::decode_all ({label})"),
        );
        assert_too_big(
            reused.decode_all_to_vec(frame, &mut out_vec),
            expected,
            &alloc::format!("reused FrameDecoder::decode_all_to_vec ({label})"),
        );

        // Multi-frame input: valid frame (new) followed by oversized frame (reset).
        let mut multi = Vec::with_capacity(valid_frame.len() + frame.len());
        multi.extend_from_slice(&valid_frame);
        multi.extend_from_slice(frame);
        assert_too_big(
            FrameDecoder::new().decode_all(&multi, &mut out).map(|_| ()),
            expected,
            &alloc::format!("multi-frame decode_all ({label})"),
        );
    }

    // 3. StreamingDecoder paths.
    for (label, frame, expected) in over_limit_cases {
        assert_too_big(
            StreamingDecoder::new(frame).map(|_| ()),
            expected,
            &alloc::format!("StreamingDecoder::new ({label})"),
        );
        assert_too_big(
            StreamingDecoder::new_with_decoder(frame, FrameDecoder::new()).map(|_| ()),
            expected,
            &alloc::format!("StreamingDecoder::new_with_decoder fresh ({label})"),
        );
        let mut reused_for_stream = FrameDecoder::new();
        assert_eq!(
            reused_for_stream
                .decode_all(&valid_frame, &mut out)
                .unwrap(),
            0
        );
        assert_too_big(
            StreamingDecoder::new_with_decoder(frame, reused_for_stream).map(|_| ()),
            expected,
            &alloc::format!("StreamingDecoder::new_with_decoder reused ({label})"),
        );
    }

    // 4. std oneshot helpers.
    #[cfg(feature = "std")]
    {
        let assert_io_window_err = |res: std::io::Result<()>, expected_str: &str, ctx: &str| {
            let err = res.expect_err(ctx);
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{ctx}: {err}");
            let msg = std::format!("{err}");
            assert!(
                msg.contains("WindowSizeTooBig") && msg.contains(expected_str),
                "{ctx}: unexpected message {msg}"
            );
        };

        for (label, frame, expected) in over_limit_cases {
            let expected_str = alloc::format!("{expected}");
            assert_io_window_err(
                crate::stream::decode_all(frame).map(|_| ()),
                &expected_str,
                &alloc::format!("stream::decode_all ({label})"),
            );
            assert_io_window_err(
                crate::stream::decode_all_with_max(frame, Some(1024)).map(|_| ()),
                &expected_str,
                &alloc::format!("stream::decode_all_with_max ({label})"),
            );
            assert_io_window_err(
                crate::stream::decode_all_unbounded(frame).map(|_| ()),
                &expected_str,
                &alloc::format!("stream::decode_all_unbounded ({label})"),
            );
            let mut sink = Vec::new();
            assert_io_window_err(
                crate::stream::copy_decode(frame, &mut sink),
                &expected_str,
                &alloc::format!("stream::copy_decode ({label})"),
            );
            assert_io_window_err(
                crate::stream::copy_decode_with_max(frame, &mut sink, Some(1024)),
                &expected_str,
                &alloc::format!("stream::copy_decode_with_max ({label})"),
            );
            assert_io_window_err(
                crate::stream::copy_decode_unbounded(frame, &mut sink),
                &expected_str,
                &alloc::format!("stream::copy_decode_unbounded ({label})"),
            );
        }
    }
}

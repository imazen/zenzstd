//! Cooperative cancellation (`enough::Stop`) in the frame decoder.

use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use enough::{Stop, StopReason};
use zenzstd::decoding::errors::FrameDecoderError;
use zenzstd::decoding::{BlockDecodingStrategy, FrameDecoder, StreamingDecoder};
use zenzstd::encoding::{compress_to_vec, CompressionLevel};

/// A stop source that lets `remaining` polls pass, then reports `Cancelled`.
/// Used to prove the check fires *inside* the block loop: with `remaining = 1`
/// the first block is decoded and the second poll aborts the frame.
struct FailAfter(AtomicUsize);

impl FailAfter {
    fn new(remaining: usize) -> Self {
        Self(AtomicUsize::new(remaining))
    }
    fn polls(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

impl Stop for FailAfter {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.0.load(Ordering::Relaxed);
        if n == 0 {
            Err(StopReason::Cancelled)
        } else {
            self.0.store(n - 1, Ordering::Relaxed);
            Ok(())
        }
    }
    fn should_stop(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }
}

fn multi_block_frame() -> (Vec<u8>, Vec<u8>) {
    // 512 KiB of compressible data => several zstd blocks (128 KiB each).
    let raw = vec![b'a'; 512 * 1024];
    let compressed = compress_to_vec(raw.as_slice(), CompressionLevel::Level(1));
    (raw, compressed)
}

#[test]
fn decode_blocks_polls_stop_per_block() {
    let (raw, compressed) = multi_block_frame();

    // Sanity: without a stop the frame decodes fully.
    let mut src = std::io::Cursor::new(&compressed);
    let mut dec = FrameDecoder::new();
    dec.init(&mut src).unwrap();
    let finished = dec
        .decode_blocks(&mut src, BlockDecodingStrategy::All)
        .unwrap();
    assert!(finished);
    assert_eq!(dec.collect().unwrap().len(), raw.len());

    // One allowed poll decodes the first block; the second poll must abort.
    let stop = Arc::new(FailAfter::new(1));
    let mut src = std::io::Cursor::new(&compressed);
    let mut dec = FrameDecoder::new().with_stop(stop.clone());
    dec.init(&mut src).unwrap();
    let err = dec
        .decode_blocks(&mut src, BlockDecodingStrategy::All)
        .unwrap_err();
    assert!(matches!(
        err,
        FrameDecoderError::Cancelled(StopReason::Cancelled)
    ));
    // The frame is not finished: cancellation happened mid-decode.
    assert!(!dec.is_finished());
}

#[test]
fn decode_blocks_honours_pre_stopped_token() {
    let (_raw, compressed) = multi_block_frame();
    let stop = Arc::new(FailAfter::new(0));
    let mut src = std::io::Cursor::new(&compressed);
    let mut dec = FrameDecoder::new().with_stop(stop);
    dec.init(&mut src).unwrap();
    let err = dec
        .decode_blocks(&mut src, BlockDecodingStrategy::All)
        .unwrap_err();
    assert!(matches!(
        err,
        FrameDecoderError::Cancelled(StopReason::Cancelled)
    ));
}

#[test]
fn streaming_decoder_forwards_stop() {
    let (raw, compressed) = multi_block_frame();

    let mut stream = StreamingDecoder::new_with_decoder(&compressed[..], FrameDecoder::new())
        .unwrap();
    stream.set_stop(Arc::new(FailAfter::new(0)));

    let mut out = Vec::new();
    let err = stream.read_to_end(&mut out).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Cancelled") || out.len() < raw.len(),
        "expected cancellation error, got {msg}"
    );
}

//! The [StreamingDecoder] wraps a [FrameDecoder] and provides a Read impl that decodes data when necessary

use core::borrow::BorrowMut;

use crate::decoding::errors::FrameDecoderError;
use crate::decoding::{BlockDecodingStrategy, FrameDecoder};
#[cfg(not(feature = "std"))]
use crate::io::ErrorKind;
use crate::io::{Error, Read};

/// High level Zstandard frame decoder that can be used to decompress a given Zstandard frame.
///
/// This decoder implements `io::Read`, so you can interact with it by calling
/// `io::Read::read_to_end` / `io::Read::read_exact` or passing this to another library / module as a source for the decoded content
///
/// If you need more control over how decompression takes place, you can use
/// the lower level [FrameDecoder], which allows for greater control over how
/// decompression takes place but the implementor must call
/// [FrameDecoder::decode_blocks] repeatedly to decode the entire frame.
///
/// ## Caveat
/// [StreamingDecoder] expects the underlying stream to only contain a single frame,
/// yet the specification states that a single archive may contain multiple frames.
///
/// To decode all the frames in a finite stream, the calling code needs to recreate
/// the instance of the decoder and handle
/// [crate::decoding::errors::ReadFrameHeaderError::SkipFrame]
/// errors by skipping forward the `length` amount of bytes, see <https://github.com/KillingSpark/zstd-rs/issues/57>
///
/// ```no_run
/// // `read_to_end` is not implemented by the no_std implementation.
/// #[cfg(feature = "std")]
/// {
///     use std::fs::File;
///     use std::io::Read;
///     use zenzstd::decoding::StreamingDecoder;
///
///     // Read a Zstandard archive from the filesystem then decompress it into a vec.
///     let mut f: File = todo!("Read a .zstd archive from somewhere");
///     let mut decoder = StreamingDecoder::new(f).unwrap();
///     let mut result = Vec::new();
///     Read::read_to_end(&mut decoder, &mut result).unwrap();
/// }
/// ```
pub struct StreamingDecoder<READ: Read, DEC: BorrowMut<FrameDecoder>> {
    pub decoder: DEC,
    source: READ,
    /// Maximum number of decompressed bytes the [`Read`] impl is allowed to
    /// emit. Reads past this cap fail with an error rather than allocating
    /// unbounded memory.
    ///
    /// `None` means no cap (legacy behavior). Set via [`Self::set_max_output_size`].
    /// The 100 MiB internal window cap (see [`crate::common::MAX_WINDOW_SIZE`])
    /// protects ring-buffer state but **not** the caller's output `Vec`, so
    /// untrusted input — chained RLE blocks reaching petabyte-scale logical
    /// output from a few KB of compressed bytes — must always set a cap.
    max_output_size: Option<usize>,
    /// Running total of bytes successfully emitted from [`Read::read`].
    bytes_emitted: u64,
}

/// Default decompression-output cap used by [`crate::stream::decode_all`] and
/// [`crate::stream::copy_decode`]. 1 GiB is generous for legitimate use while
/// still bounding the worst-case decompression-bomb output that fits in a Vec
/// on a typical host. Use the `_unbounded` helpers if you genuinely trust the
/// source.
pub const DEFAULT_DECODE_OUTPUT_CAP: usize = 1024 * 1024 * 1024;

impl<READ: Read, DEC: BorrowMut<FrameDecoder>> StreamingDecoder<READ, DEC> {
    pub fn new_with_decoder(
        mut source: READ,
        mut decoder: DEC,
    ) -> Result<StreamingDecoder<READ, DEC>, FrameDecoderError> {
        decoder.borrow_mut().init(&mut source)?;
        Ok(StreamingDecoder {
            decoder,
            source,
            max_output_size: None,
            bytes_emitted: 0,
        })
    }
}

impl<READ: Read> StreamingDecoder<READ, FrameDecoder> {
    pub fn new(
        mut source: READ,
    ) -> Result<StreamingDecoder<READ, FrameDecoder>, FrameDecoderError> {
        let mut decoder = FrameDecoder::new();
        decoder.init(&mut source)?;
        Ok(StreamingDecoder {
            decoder,
            source,
            max_output_size: None,
            bytes_emitted: 0,
        })
    }
}

impl<READ: Read, DEC: BorrowMut<FrameDecoder>> StreamingDecoder<READ, DEC> {
    /// Gets a reference to the underlying reader.
    pub fn get_ref(&self) -> &READ {
        &self.source
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    pub fn get_mut(&mut self) -> &mut READ {
        &mut self.source
    }

    /// Destructures this object into the inner reader.
    pub fn into_inner(self) -> READ
    where
        READ: Sized,
    {
        self.source
    }

    /// Destructures this object into both the inner reader and [FrameDecoder].
    pub fn into_parts(self) -> (READ, DEC)
    where
        READ: Sized,
    {
        (self.source, self.decoder)
    }

    /// Destructures this object into the inner [FrameDecoder].
    pub fn into_frame_decoder(self) -> DEC {
        self.decoder
    }

    /// Cap the total number of decompressed bytes this decoder will emit
    /// across all [`Read::read`] calls. Reads that would push the running
    /// total past `max` fail with [`crate::io::ErrorKind::InvalidData`] rather
    /// than continuing to allocate.
    ///
    /// This protects against zstd "decompression bombs" where a small
    /// compressed payload (e.g. chained RLE blocks) expands to gigabytes or
    /// terabytes of output. The internal 100 MiB window cap protects ring-buffer
    /// state but **not** the caller's destination buffer.
    ///
    /// Pass `None` to remove the cap (use only when the source is fully trusted).
    pub fn set_max_output_size(&mut self, max: Option<usize>) {
        self.max_output_size = max;
    }

    /// Returns the current output-size cap, if any.
    pub fn max_output_size(&self) -> Option<usize> {
        self.max_output_size
    }
}

impl<READ: Read, DEC: BorrowMut<FrameDecoder>> Read for StreamingDecoder<READ, DEC> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Error> {
        // Enforce the configured output cap before consuming any further
        // compressed bytes. We slice `buf` down to `remaining_cap` so callers
        // that pass an oversized buffer (e.g. read_to_end) cannot push us past
        // the cap by accident; once `remaining_cap` reaches zero, the next read
        // returns InvalidData if the frame is not yet finished, and EOF (Ok(0))
        // if it is — same shape as a truncated underlying reader.
        let remaining_cap = match self.max_output_size {
            None => buf.len(),
            Some(max) => {
                let emitted = self.bytes_emitted;
                let cap = max as u64;
                if emitted >= cap {
                    // Already at the cap. If decoder still has work to do, this
                    // is a bomb — surface the failure instead of silently
                    // truncating. If the decoder is done, it's a clean EOF.
                    let decoder = self.decoder.borrow_mut();
                    return if decoder.is_finished() && decoder.can_collect() == 0 {
                        Ok(0)
                    } else {
                        let msg = "zenzstd: decompressed output exceeded max_output_size";
                        #[cfg(feature = "std")]
                        {
                            Err(Error::new(std::io::ErrorKind::InvalidData, msg))
                        }
                        #[cfg(not(feature = "std"))]
                        {
                            // no_std ErrorKind has no InvalidData variant; use Other.
                            Err(Error::new(ErrorKind::Other, alloc::boxed::Box::new(msg)))
                        }
                    };
                }
                (cap - emitted).min(buf.len() as u64) as usize
            }
        };

        let decoder = self.decoder.borrow_mut();
        if decoder.is_finished() && decoder.can_collect() == 0 {
            //No more bytes can ever be decoded
            return Ok(0);
        }

        // need to loop. The UpToBytes strategy doesn't take any effort to actually reach that limit.
        // The first few calls can result in just filling the decode buffer but these bytes can not be collected.
        // So we need to call this until we can actually collect enough bytes

        // TODO add BlockDecodingStrategy::UntilCollectable(usize) that pushes this logic into the decode_blocks function
        while decoder.can_collect() < remaining_cap && !decoder.is_finished() {
            //More bytes can be decoded
            let additional_bytes_needed = remaining_cap - decoder.can_collect();
            match decoder.decode_blocks(
                &mut self.source,
                BlockDecodingStrategy::UptoBytes(additional_bytes_needed),
            ) {
                Ok(_) => { /*Nothing to do*/ }
                Err(e) => {
                    let err;
                    #[cfg(feature = "std")]
                    {
                        err = Error::other(e);
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        err = Error::new(ErrorKind::Other, alloc::boxed::Box::new(e));
                    }
                    return Err(err);
                }
            }
        }

        let n = decoder.read(&mut buf[..remaining_cap])?;
        self.bytes_emitted = self.bytes_emitted.saturating_add(n as u64);
        Ok(n)
    }
}

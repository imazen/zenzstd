//! Unsafe fast-path operations for decompression.
//!
//! When the `unsafe-decompress` feature is enabled, these functions use
//! unchecked indexing and raw pointer copies to eliminate bounds checks
//! in the hot decode loop. The safety invariants are maintained by the
//! callers (reserve + bounds assertions).
//!
//! When the feature is NOT enabled, these are just wrappers around the
//! safe equivalents.

/// Copy `src` bytes into `dst[pos..pos+src.len()]` without bounds checking.
///
/// # Safety (when unsafe-decompress enabled)
/// Caller must guarantee `pos + src.len() <= dst.len()`.
#[inline(always)]
pub fn copy_to_buf(dst: &mut [u8], pos: usize, src: &[u8]) {
    #[cfg(feature = "unsafe-decompress")]
    {
        // SAFETY: caller guarantees pos + src.len() <= dst.len()
        // via the reserve(MAX_BLOCK_SIZE) at the top of the decode loop.
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_mut_ptr().add(pos),
                src.len(),
            );
        }
    }
    #[cfg(not(feature = "unsafe-decompress"))]
    {
        dst[pos..pos + src.len()].copy_from_slice(src);
    }
}

/// Copy `len` bytes from `src_pos` to `dst_pos` within the same buffer.
/// Handles overlapping regions correctly (like memmove).
///
/// # Safety (when unsafe-decompress enabled)
/// Caller must guarantee both ranges are within `buf.len()`.
#[inline(always)]
pub fn copy_within_buf(buf: &mut [u8], src_pos: usize, dst_pos: usize, len: usize) {
    #[cfg(feature = "unsafe-decompress")]
    {
        // SAFETY: caller guarantees src_pos+len <= buf.len() and dst_pos+len <= buf.len()
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::copy(
                buf.as_ptr().add(src_pos),
                buf.as_mut_ptr().add(dst_pos),
                len,
            );
        }
    }
    #[cfg(not(feature = "unsafe-decompress"))]
    {
        buf.copy_within(src_pos..src_pos + len, dst_pos);
    }
}

/// Read a single byte from `buf[pos]` without bounds checking.
///
/// # Safety (when unsafe-decompress enabled)
/// Caller must guarantee `pos < buf.len()`.
#[inline(always)]
pub fn read_byte(buf: &[u8], pos: usize) -> u8 {
    #[cfg(feature = "unsafe-decompress")]
    {
        // SAFETY: caller guarantees pos < buf.len()
        #[allow(unsafe_code)]
        unsafe { *buf.get_unchecked(pos) }
    }
    #[cfg(not(feature = "unsafe-decompress"))]
    {
        buf[pos]
    }
}

/// Fill `buf[pos..pos+len]` with `byte` without bounds checking.
///
/// # Safety (when unsafe-decompress enabled)
/// Caller must guarantee `pos + len <= buf.len()`.
#[inline(always)]
pub fn fill_buf(buf: &mut [u8], pos: usize, len: usize, byte: u8) {
    #[cfg(feature = "unsafe-decompress")]
    {
        // SAFETY: caller guarantees pos + len <= buf.len()
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::write_bytes(buf.as_mut_ptr().add(pos), byte, len);
        }
    }
    #[cfg(not(feature = "unsafe-decompress"))]
    {
        buf[pos..pos + len].fill(byte);
    }
}

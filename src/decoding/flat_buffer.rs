//! Flat (non-ring) decode buffer that eliminates bitmask overhead on every byte access.
//!
//! Instead of a ring buffer with power-of-2 capacity and `& mask` on every read/write,
//! this uses a flat `Vec<u8>` with a write cursor. When space runs low, the last
//! `window_size` bytes are memmoved to the front (one `copy_within` per block,
//! amortized over ~128KB of output).
//!
//! Benefits over ring buffer:
//! - No `& mask` on every byte access
//! - `as_slices()` always returns one contiguous slice (no two-part split)
//! - `copy_within` for match copies is always a single call (no wrap handling)
//! - Simpler drain logic (one contiguous region)

use alloc::vec::Vec;

pub struct FlatBuffer {
    pub(crate) buf: Vec<u8>,
    /// Write cursor: next byte goes at `buf[pos]`.
    pub(crate) pos: usize,
    /// Start of data not yet drained. `buf[drain_pos..pos]` is valid data.
    pub(crate) drain_pos: usize,
    /// Window size: how many bytes to keep when compacting.
    window_size: usize,
}

impl FlatBuffer {
    pub fn new() -> Self {
        FlatBuffer {
            buf: Vec::new(),
            pos: 0,
            drain_pos: 0,
            window_size: 0,
        }
    }

    /// Set the window size used for compaction decisions.
    pub fn set_window_size(&mut self, window_size: usize) {
        self.window_size = window_size;
    }

    /// Number of valid bytes in the buffer (from drain_pos to pos).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.pos - self.drain_pos
    }

    /// Available space after `pos` without growing.
    #[inline(always)]
    fn free(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Empty the buffer and reset cursors.
    pub fn clear(&mut self) {
        self.pos = 0;
        self.drain_pos = 0;
    }

    /// Ensure at least `amount` bytes of space after `pos`.
    /// Tries compaction first, then grows if needed.
    #[inline(always)]
    pub fn reserve(&mut self, amount: usize) {
        if self.free() >= amount {
            return;
        }
        self.reserve_slow(amount);
    }

    #[inline(never)]
    #[cold]
    fn reserve_slow(&mut self, amount: usize) {
        // First try compacting: move the tail window_size bytes to the front.
        // This only helps if drain_pos > 0 (we have drained data we can reclaim).
        if self.drain_pos > 0 {
            self.compact();
            if self.free() >= amount {
                return;
            }
        }

        // Still not enough: grow the buffer.
        let needed = self.pos + amount;
        // Grow to at least 2x current or needed, whichever is larger.
        let new_cap = needed.max(self.buf.len() * 2).max(64);
        self.buf.resize(new_cap, 0);
    }

    /// Move the last `window_size` bytes (or all data if less) to the front.
    /// Reclaims space from already-drained data.
    fn compact(&mut self) {
        if self.drain_pos == 0 {
            return;
        }

        // Keep everything from drain_pos to pos.
        let data_len = self.pos - self.drain_pos;
        if self.drain_pos > 0 && data_len > 0 {
            self.buf.copy_within(self.drain_pos..self.pos, 0);
        }
        self.pos = data_len;
        self.drain_pos = 0;
    }

    /// Append data to the buffer, reserving space if needed.
    #[inline(always)]
    pub fn extend(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.reserve(data.len());
        self.buf[self.pos..self.pos + data.len()].copy_from_slice(data);
        self.pos += data.len();
    }

    /// Append data without checking capacity. Caller must ensure sufficient space.
    #[inline(always)]
    pub fn extend_no_reserve(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        debug_assert!(
            self.free() >= data.len(),
            "extend_no_reserve: not enough space ({} free, {} needed)",
            self.free(),
            data.len()
        );
        self.buf[self.pos..self.pos + data.len()].copy_from_slice(data);
        self.pos += data.len();
    }

    /// Copy from within the buffer. `start` is a logical index from drain_pos.
    /// Handles overlapping copies (offset < len) for LZ77 repeat patterns.
    /// Reserves space if needed.
    #[inline(always)]
    pub fn extend_from_within_unchecked(&mut self, start: usize, len: usize) {
        debug_assert!(
            start <= self.len(),
            "start ({start}) > len ({})",
            self.len()
        );
        self.reserve(len);

        let src_abs = self.drain_pos + start;
        let distance = self.pos - src_abs;

        // Fast path: non-overlapping copy (most common case).
        if distance >= len {
            self.buf.copy_within(src_abs..src_abs + len, self.pos);
            self.pos += len;
            return;
        }

        // RLE fast path: distance == 1.
        if distance == 1 {
            let byte = self.buf[src_abs];
            self.buf[self.pos..self.pos + len].fill(byte);
            self.pos += len;
            return;
        }

        self.extend_from_within_slow(src_abs, distance, len);
    }

    /// Copy from within without capacity check. Caller must ensure space.
    #[inline(always)]
    pub fn extend_from_within_no_reserve(&mut self, start: usize, len: usize) {
        debug_assert!(
            start <= self.len(),
            "start ({start}) > len ({})",
            self.len()
        );
        debug_assert!(
            self.free() >= len,
            "extend_from_within_no_reserve: not enough space ({} free, {} needed)",
            self.free(),
            len
        );

        let src_abs = self.drain_pos + start;
        let distance = self.pos - src_abs;

        // Fast path: non-overlapping copy.
        if distance >= len {
            self.buf.copy_within(src_abs..src_abs + len, self.pos);
            self.pos += len;
            return;
        }

        // RLE fast path: distance == 1.
        if distance == 1 {
            let byte = self.buf[src_abs];
            self.buf[self.pos..self.pos + len].fill(byte);
            self.pos += len;
            return;
        }

        self.extend_from_within_slow(src_abs, distance, len);
    }

    /// Handle overlapping copies where 1 < distance < len.
    /// Uses doubling copy for efficiency: write the pattern once, then
    /// double what's written until we've produced `len` bytes.
    #[inline(never)]
    fn extend_from_within_slow(&mut self, src_abs: usize, distance: usize, len: usize) {
        // Copy the initial pattern.
        self.buf.copy_within(src_abs..src_abs + distance, self.pos);

        // Doubling copy: double what we've written so far.
        let mut written = distance;
        while written < len {
            let copy_len = written.min(len - written);
            self.buf
                .copy_within(self.pos..self.pos + copy_len, self.pos + written);
            written += copy_len;
        }
        self.pos += len;
    }

    /// Copy from within using an `extend_from_within`-style API (with bounds check + reserve).
    #[allow(dead_code)]
    pub fn extend_from_within(&mut self, start: usize, len: usize) {
        assert!(
            start + len <= self.len(),
            "extend_from_within: start ({}) + len ({}) > self.len() ({})",
            start,
            len,
            self.len()
        );
        self.extend_from_within_unchecked(start, len);
    }

    /// Advance the drain cursor past `amount` bytes.
    pub fn drop_first_n(&mut self, amount: usize) {
        debug_assert!(amount <= self.len());
        let amount = amount.min(self.len());
        self.drain_pos += amount;

        // If we've drained everything, reset to reclaim all space.
        if self.drain_pos == self.pos {
            self.pos = 0;
            self.drain_pos = 0;
        }
    }

    /// Return a view of the valid data. Always contiguous (single slice).
    /// Returns (&[u8], &[]) for API compatibility with RingBuffer::as_slices.
    pub fn as_slices(&self) -> (&[u8], &[u8]) {
        (&self.buf[self.drain_pos..self.pos], &[])
    }

    /// Push a single byte.
    #[allow(dead_code)]
    pub fn push_back(&mut self, byte: u8) {
        self.reserve(1);
        self.buf[self.pos] = byte;
        self.pos += 1;
    }

    /// Fetch the byte at the given logical index.
    #[allow(dead_code)]
    pub fn get(&self, idx: usize) -> Option<u8> {
        if idx < self.len() {
            Some(self.buf[self.drain_pos + idx])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FlatBuffer;

    fn collect(fb: &FlatBuffer) -> alloc::vec::Vec<u8> {
        let (s1, s2) = fb.as_slices();
        let mut v = s1.to_vec();
        v.extend_from_slice(s2);
        v
    }

    #[test]
    fn smoke() {
        let mut fb = FlatBuffer::new();
        fb.reserve(15);

        fb.extend(b"0123456789");
        assert_eq!(fb.len(), 10);
        assert_eq!(collect(&fb), b"0123456789");

        fb.drop_first_n(5);
        assert_eq!(fb.len(), 5);
        assert_eq!(collect(&fb), b"56789");

        fb.extend_from_within(2, 3);
        assert_eq!(fb.len(), 8);
        assert_eq!(collect(&fb), b"56789789");

        fb.extend_from_within(0, 3);
        assert_eq!(fb.len(), 11);
        assert_eq!(collect(&fb), b"56789789567");

        fb.extend_from_within(0, 2);
        assert_eq!(fb.len(), 13);
        assert_eq!(collect(&fb), b"5678978956756");

        fb.drop_first_n(11);
        assert_eq!(fb.len(), 2);

        fb.extend(b"0123456789");
        assert_eq!(fb.len(), 12);

        fb.drop_first_n(11);
        assert_eq!(fb.len(), 1);

        fb.extend(b"0123456789");
        assert_eq!(fb.len(), 11);
    }

    #[test]
    fn edge_cases() {
        let mut fb = FlatBuffer::new();
        fb.reserve(16);
        fb.extend(b"0123456789012345");
        assert_eq!(16, fb.len());
        fb.drop_first_n(16);
        assert_eq!(0, fb.len());
        fb.extend(b"0123456789012345");
        assert_eq!(16, fb.len());

        fb.clear();

        fb.extend(b"0123456789012345");
        fb.drop_first_n(8);
        fb.extend(b"67890123");
        assert_eq!(16, fb.len());

        fb.clear();

        fb.extend(b"0123456789012345");
        fb.extend_from_within(0, 16);
        assert_eq!(32, fb.len());
    }

    #[test]
    fn test_repeat_via_unchecked() {
        let mut fb = FlatBuffer::new();
        fb.extend(b"ABCD");
        fb.extend_from_within_unchecked(3, 1);
        assert_eq!(collect(&fb), b"ABCDD");

        let mut fb = FlatBuffer::new();
        fb.extend(b"ABCDE");
        fb.extend_from_within_unchecked(3, 4);
        assert_eq!(collect(&fb), b"ABCDEDEDE");

        let mut fb = FlatBuffer::new();
        fb.extend(b"HELLO");
        fb.extend_from_within_unchecked(0, 5);
        assert_eq!(collect(&fb), b"HELLOHELLO");
    }

    #[test]
    fn test_get() {
        let mut fb = FlatBuffer::new();
        fb.extend(b"hello");
        assert_eq!(fb.get(0), Some(b'h'));
        assert_eq!(fb.get(4), Some(b'o'));
        assert_eq!(fb.get(5), None);
    }

    #[test]
    fn test_push_back() {
        let mut fb = FlatBuffer::new();
        fb.push_back(b'a');
        fb.push_back(b'b');
        fb.push_back(b'c');
        assert_eq!(fb.len(), 3);
        assert_eq!(fb.get(0), Some(b'a'));
        assert_eq!(fb.get(1), Some(b'b'));
        assert_eq!(fb.get(2), Some(b'c'));
    }

    #[test]
    fn test_rle_repeat() {
        let mut fb = FlatBuffer::new();
        fb.extend(b"X");
        fb.extend_from_within_unchecked(0, 10);
        assert_eq!(collect(&fb), b"XXXXXXXXXXX");
    }

    #[test]
    fn test_short_repeat_pattern() {
        let mut fb = FlatBuffer::new();
        fb.extend(b"AB");
        fb.extend_from_within_unchecked(0, 8);
        assert_eq!(collect(&fb), b"ABABABABAB");
    }

    #[test]
    fn test_large_non_overlapping_copy() {
        let mut fb = FlatBuffer::new();
        let data: alloc::vec::Vec<u8> = (0..=255).cycle().take(512).collect();
        fb.extend(&data);
        fb.extend_from_within(0, 512);
        let result = collect(&fb);
        assert_eq!(result.len(), 1024);
        assert_eq!(&result[..512], &data[..]);
        assert_eq!(&result[512..], &data[..]);
    }

    #[test]
    fn test_large_repeating_pattern() {
        let mut fb = FlatBuffer::new();
        let pattern: alloc::vec::Vec<u8> = (0..=255).cycle().take(300).collect();
        fb.extend(&pattern);
        fb.extend_from_within_unchecked(0, 600);
        let result = collect(&fb);
        assert_eq!(result.len(), 900);
        for i in 0..900 {
            assert_eq!(result[i], pattern[i % 300], "mismatch at {i}");
        }
    }

    #[test]
    fn test_medium_non_overlapping_copy() {
        let mut fb = FlatBuffer::new();
        let data: alloc::vec::Vec<u8> = (0..200).collect();
        fb.extend(&data);
        fb.extend_from_within(0, 200);
        let result = collect(&fb);
        assert_eq!(result.len(), 400);
        assert_eq!(&result[..200], &data[..]);
        assert_eq!(&result[200..], &data[..]);
    }

    #[test]
    fn test_drain_and_extend() {
        // Simulate a decode buffer pattern: extend, drain partially, extend more.
        let mut fb = FlatBuffer::new();
        fb.set_window_size(100);

        // Fill with 200 bytes
        let data: alloc::vec::Vec<u8> = (0..200).collect();
        fb.extend(&data);
        assert_eq!(fb.len(), 200);

        // Drain first 100 (keeping window_size)
        fb.drop_first_n(100);
        assert_eq!(fb.len(), 100);

        // Extend more
        let more: alloc::vec::Vec<u8> = (200..250).map(|x| x as u8).collect();
        fb.extend(&more);
        assert_eq!(fb.len(), 150);

        // Verify data integrity
        let result = collect(&fb);
        for i in 0..100 {
            assert_eq!(result[i], (i + 100) as u8, "mismatch at {i}");
        }
        for i in 0..50 {
            assert_eq!(result[100 + i], (200 + i) as u8, "mismatch at {}", 100 + i);
        }
    }

    #[test]
    fn test_compact_on_reserve() {
        let mut fb = FlatBuffer::new();
        fb.set_window_size(50);

        // Write 100 bytes
        let data: alloc::vec::Vec<u8> = (0..100).collect();
        fb.extend(&data);

        // Drain first 80
        fb.drop_first_n(80);
        assert_eq!(fb.len(), 20);

        // The buffer should compact when we reserve a large amount
        // and drain_pos > 0 frees enough space.
        let old_cap = fb.buf.len();
        fb.reserve(old_cap); // This needs more space than free() provides
        // After compact, drain_pos should be 0
        assert_eq!(fb.drain_pos, 0);
        assert_eq!(fb.pos, 20);

        // Verify data integrity
        let result = collect(&fb);
        for i in 0..20 {
            assert_eq!(result[i], (i + 80) as u8);
        }
    }

    #[test]
    fn test_extend_from_within_after_drain() {
        // This tests the critical path: extend_from_within after some data has been drained.
        // The logical index `start` is relative to drain_pos.
        let mut fb = FlatBuffer::new();
        fb.extend(b"ABCDEFGHIJ");
        fb.drop_first_n(5); // Now contains "FGHIJ"

        // Copy from logical index 2 (= 'H'), length 3
        fb.extend_from_within(2, 3);
        assert_eq!(collect(&fb), b"FGHIJHIJ");

        // Copy from logical index 0 (= 'F'), length 3
        fb.extend_from_within(0, 3);
        assert_eq!(collect(&fb), b"FGHIJHIJFGH");
    }
}

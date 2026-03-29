//! Safe ring buffer implementation using `Vec<u8>`.
//!
//! This replaces the original unsafe ringbuffer from ruzstd with a fully safe
//! implementation that uses `Vec<u8>` as backing storage. The ring buffer
//! maintains head/tail indices and uses power-of-two capacity for fast
//! modular arithmetic via bitmask.

use alloc::vec;
use alloc::vec::Vec;

pub struct RingBuffer {
    buf: Vec<u8>,
    /// Capacity is always a power of two, so (index & mask) == (index % cap).
    /// We store cap - 1 here for fast modular arithmetic.
    mask: usize,
    head: usize,
    tail: usize,
}

impl RingBuffer {
    pub fn new() -> Self {
        RingBuffer {
            buf: Vec::new(),
            mask: 0,
            head: 0,
            tail: 0,
        }
    }

    /// Return the number of bytes in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        if self.buf.is_empty() {
            return 0;
        }
        let cap = self.mask + 1;
        self.tail.wrapping_sub(self.head) & (cap - 1)
    }

    /// Return the amount of available space (in bytes) of the buffer.
    pub fn free(&self) -> usize {
        if self.buf.is_empty() {
            return 0;
        }
        let cap = self.mask + 1;
        // We keep one sentinel slot unused to distinguish full from empty
        cap - 1 - self.len()
    }

    /// Empty the buffer and reset the head and tail.
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
    }

    /// Ensure that there's space for `amount` elements in the buffer.
    pub fn reserve(&mut self, amount: usize) {
        if self.free() >= amount {
            return;
        }
        self.grow(amount);
    }

    #[inline(never)]
    #[cold]
    fn grow(&mut self, amount: usize) {
        let old_len = self.len();
        // Need space for old data + new data + 1 sentinel
        let new_cap = (old_len + amount + 1).next_power_of_two();

        let mut new_buf = vec![0u8; new_cap];

        // Copy existing data contiguously into the new buffer
        if !self.buf.is_empty() && old_len > 0 {
            let cap = self.mask + 1;
            if self.tail >= self.head {
                // Data is contiguous: [head..tail)
                new_buf[..old_len].copy_from_slice(&self.buf[self.head..self.tail]);
            } else {
                // Data wraps: [head..cap) + [0..tail)
                let first = cap - self.head;
                new_buf[..first].copy_from_slice(&self.buf[self.head..cap]);
                new_buf[first..first + self.tail].copy_from_slice(&self.buf[..self.tail]);
            }
        }

        self.buf = new_buf;
        self.mask = new_cap - 1;
        self.head = 0;
        self.tail = old_len;
    }

    #[allow(dead_code)]
    pub fn push_back(&mut self, byte: u8) {
        self.reserve(1);
        self.buf[self.tail] = byte;
        self.tail = (self.tail + 1) & self.mask;
    }

    /// Fetch the byte stored at the selected index from the buffer.
    #[allow(dead_code)]
    pub fn get(&self, idx: usize) -> Option<u8> {
        if idx < self.len() {
            let actual = (self.head + idx) & self.mask;
            Some(self.buf[actual])
        } else {
            None
        }
    }

    /// Append the provided data to the end of `self`.
    pub fn extend(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.reserve(data.len());

        let cap = self.mask + 1;
        let first_len = cap - self.tail;

        if first_len >= data.len() {
            // Fits without wrapping
            self.buf[self.tail..self.tail + data.len()].copy_from_slice(data);
        } else {
            // Wraps around
            self.buf[self.tail..self.tail + first_len].copy_from_slice(&data[..first_len]);
            let remaining = data.len() - first_len;
            self.buf[..remaining].copy_from_slice(&data[first_len..]);
        }

        self.tail = (self.tail + data.len()) & self.mask;
    }

    /// Advance head past `amount` elements, effectively removing them.
    pub fn drop_first_n(&mut self, amount: usize) {
        debug_assert!(amount <= self.len());
        let amount = amount.min(self.len());
        self.head = (self.head + amount) & self.mask;
    }

    /// Return references to each contiguous part of the ring buffer data.
    pub fn as_slices(&self) -> (&[u8], &[u8]) {
        if self.buf.is_empty() {
            return (&[], &[]);
        }
        let cap = self.mask + 1;
        if self.tail >= self.head {
            (&self.buf[self.head..self.tail], &[])
        } else {
            (&self.buf[self.head..cap], &self.buf[..self.tail])
        }
    }

    /// Copies elements from the provided range to the end of the buffer.
    #[allow(dead_code)]
    pub fn extend_from_within(&mut self, start: usize, len: usize) {
        assert!(
            start + len <= self.len(),
            "extend_from_within: start ({}) + len ({}) > self.len() ({})",
            start,
            len,
            self.len()
        );

        self.reserve(len);
        self.do_extend_from_within(start, len);
    }

    fn do_extend_from_within(&mut self, start: usize, len: usize) {
        let current_len = self.len();
        let distance = current_len - start;

        if distance >= len {
            // No overlap: copy via temp buffer
            let mut tmp = vec![0u8; len];
            self.read_at(start, &mut tmp);
            self.extend_raw(&tmp);
        } else {
            // Overlapping: the pattern repeats with given period
            let mut pattern = vec![0u8; distance];
            self.read_at(start, &mut pattern);

            let mut remaining = len;
            while remaining > 0 {
                let chunk = remaining.min(distance);
                self.extend_raw(&pattern[..chunk]);
                remaining -= chunk;
            }
        }
    }

    /// Low-level extend that assumes space is already reserved.
    fn extend_raw(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let cap = self.mask + 1;
        let first_len = cap - self.tail;

        if first_len >= data.len() {
            self.buf[self.tail..self.tail + data.len()].copy_from_slice(data);
        } else {
            self.buf[self.tail..self.tail + first_len].copy_from_slice(&data[..first_len]);
            let remaining = data.len() - first_len;
            self.buf[..remaining].copy_from_slice(&data[first_len..]);
        }
        self.tail = (self.tail + data.len()) & self.mask;
    }

    /// Read `dst.len()` bytes starting at logical index `start` into `dst`.
    fn read_at(&self, start: usize, dst: &mut [u8]) {
        let cap = self.mask + 1;
        let begin = (self.head + start) & self.mask;
        let end_unwrapped = begin + dst.len();

        if end_unwrapped <= cap {
            dst.copy_from_slice(&self.buf[begin..end_unwrapped]);
        } else {
            let first = cap - begin;
            let second = dst.len() - first;
            dst[..first].copy_from_slice(&self.buf[begin..cap]);
            dst[first..].copy_from_slice(&self.buf[..second]);
        }
    }

    /// Safe version retained for API compatibility. Despite the name "unchecked",
    /// this is fully safe. Allows start + len > self.len() for repeat/overlap patterns.
    pub fn extend_from_within_unchecked(&mut self, start: usize, len: usize) {
        debug_assert!(start <= self.len(), "start ({start}) > len ({})", self.len());
        self.reserve(len);
        self.do_extend_from_within(start, len);
    }

    /// Also fully safe. Retained for API compatibility.
    #[allow(dead_code)]
    pub fn extend_from_within_unchecked_branchless(&mut self, start: usize, len: usize) {
        self.extend_from_within_unchecked(start, len);
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    fn collect(rb: &RingBuffer) -> alloc::vec::Vec<u8> {
        let (s1, s2) = rb.as_slices();
        let mut v = s1.to_vec();
        v.extend_from_slice(s2);
        v
    }

    #[test]
    fn smoke() {
        let mut rb = RingBuffer::new();
        rb.reserve(15);

        rb.extend(b"0123456789");
        assert_eq!(rb.len(), 10);
        assert_eq!(collect(&rb), b"0123456789");

        rb.drop_first_n(5);
        assert_eq!(rb.len(), 5);
        assert_eq!(collect(&rb), b"56789");

        rb.extend_from_within(2, 3);
        assert_eq!(rb.len(), 8);
        assert_eq!(collect(&rb), b"56789789");

        rb.extend_from_within(0, 3);
        assert_eq!(rb.len(), 11);
        assert_eq!(collect(&rb), b"56789789567");

        rb.extend_from_within(0, 2);
        assert_eq!(rb.len(), 13);
        assert_eq!(collect(&rb), b"5678978956756");

        rb.drop_first_n(11);
        assert_eq!(rb.len(), 2);

        rb.extend(b"0123456789");
        assert_eq!(rb.len(), 12);

        rb.drop_first_n(11);
        assert_eq!(rb.len(), 1);

        rb.extend(b"0123456789");
        assert_eq!(rb.len(), 11);
    }

    #[test]
    fn edge_cases() {
        let mut rb = RingBuffer::new();
        rb.reserve(16);
        rb.extend(b"0123456789012345");
        assert_eq!(16, rb.len());
        rb.drop_first_n(16);
        assert_eq!(0, rb.len());
        rb.extend(b"0123456789012345");
        assert_eq!(16, rb.len());

        rb.clear();

        rb.extend(b"0123456789012345");
        rb.drop_first_n(8);
        rb.extend(b"67890123");
        assert_eq!(16, rb.len());
        rb.reserve(1);
        assert_eq!(16, rb.len());

        rb.clear();

        rb.extend(b"0123456789012345");
        rb.extend_from_within(0, 16);
        assert_eq!(32, rb.len());

        // extend from within with wrapping
        let mut rb = RingBuffer::new();
        rb.reserve(8);
        rb.extend(b"01234567");
        rb.drop_first_n(5);
        rb.extend_from_within(0, 3);
        assert_eq!(collect(&rb), b"567567");
    }

    #[test]
    fn test_repeat_via_unchecked() {
        // Test extend_from_within_unchecked with distance < len
        // (the overlap/repeat pattern used by LZ77)
        // Buffer: "ABCD" (4 bytes). Copy from start=3, len=1.
        // distance = 4-3 = 1, len=1. No overlap. Result: "ABCDD"
        let mut rb = RingBuffer::new();
        rb.extend(b"ABCD");
        rb.extend_from_within_unchecked(3, 1);
        assert_eq!(collect(&rb), b"ABCDD");

        // Now copy from start=2, len=4 in the 5-byte buffer.
        // distance=5-2=3, len=4. Overlap: copies "DDD" then repeats.
        let mut rb = RingBuffer::new();
        rb.extend(b"ABCDE");
        rb.extend_from_within_unchecked(3, 4);
        assert_eq!(collect(&rb), b"ABCDEDEDE");

        // Copy the entire buffer
        let mut rb = RingBuffer::new();
        rb.extend(b"HELLO");
        rb.extend_from_within_unchecked(0, 5);
        assert_eq!(collect(&rb), b"HELLOHELLO");
    }

    #[test]
    fn test_get() {
        let mut rb = RingBuffer::new();
        rb.extend(b"hello");
        assert_eq!(rb.get(0), Some(b'h'));
        assert_eq!(rb.get(4), Some(b'o'));
        assert_eq!(rb.get(5), None);
    }

    #[test]
    fn test_push_back() {
        let mut rb = RingBuffer::new();
        rb.push_back(b'a');
        rb.push_back(b'b');
        rb.push_back(b'c');
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.get(0), Some(b'a'));
        assert_eq!(rb.get(1), Some(b'b'));
        assert_eq!(rb.get(2), Some(b'c'));
    }
}

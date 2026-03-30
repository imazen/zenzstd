use core::convert::TryInto;

/// Zstandard encodes some types of data in a way that the data must be read
/// back to front to decode it properly. `BitReaderReversed` provides a
/// convenient interface to do that.
pub struct BitReaderReversed<'s> {
    /// The index of the last read byte in the source.
    index: usize,

    /// How many bits have been consumed from `bit_container`.
    bits_consumed: u8,

    /// How many bits have been consumed past the end of the input. Will be zero until all the input
    /// has been read.
    extra_bits: usize,

    /// The source data to read from.
    source: &'s [u8],

    /// The reader doesn't read directly from the source, it reads bits from here, and the container
    /// is "refilled" as it's emptied.
    bit_container: u64,
}

impl<'s> BitReaderReversed<'s> {
    /// How many bits are left to read by the reader.
    pub fn bits_remaining(&self) -> isize {
        self.index as isize * 8 + (64 - self.bits_consumed as isize) - self.extra_bits as isize
    }

    /// How many bits have been consumed from the current 64-bit container.
    /// Used to check whether a refill is needed before a batch of peek_and_advance calls.
    #[inline(always)]
    pub fn bits_consumed(&self) -> u8 {
        self.bits_consumed
    }

    pub fn new(source: &'s [u8]) -> BitReaderReversed<'s> {
        BitReaderReversed {
            index: source.len(),
            bits_consumed: 64,
            source,
            bit_container: 0,
            extra_bits: 0,
        }
    }

    /// Refill the bit container from source bytes.
    /// Fast path: when at least `bytes_consumed` bytes remain in the source,
    /// this is just an index subtract + an 8-byte load.
    #[inline(always)]
    fn refill(&mut self) {
        let bytes_consumed = self.bits_consumed as usize / 8;
        if bytes_consumed == 0 {
            return;
        }

        if self.index >= bytes_consumed {
            // Fast path: plenty of source remaining. This is the common case.
            self.index -= bytes_consumed;
            self.bits_consumed &= 7;
            // Fixed-array cast: convert the slice to &[u8; 8] to eliminate
            // interior bounds checks. The bounds are validated once by the
            // slice operation, then the try_into gives LLVM proof that all
            // 8 bytes are in-bounds.
            let chunk: &[u8; 8] = self.source[self.index..][..8].try_into().unwrap();
            self.bit_container = u64::from_le_bytes(*chunk);
        } else {
            self.refill_slow();
        }
    }

    /// Refill unconditionally, pulling up to 7 fresh bytes into the container.
    /// This is the "always refill" variant used by the fused sequence decoder
    /// to guarantee enough bits are available for a full triple of FSE state
    /// updates + extra bit reads without intermediate refill checks.
    ///
    /// After this call, at least 56 bits are available (assuming source has
    /// enough data).
    #[inline(always)]
    pub fn refill_unconditional(&mut self) {
        // Compute how many bytes we've consumed since last refill
        let bytes_consumed = self.bits_consumed as usize / 8;
        // Even if bytes_consumed == 0, the rest is harmless

        if self.index >= bytes_consumed {
            self.index -= bytes_consumed;
            self.bits_consumed &= 7;
            let chunk: &[u8; 8] = self.source[self.index..][..8].try_into().unwrap();
            self.bit_container = u64::from_le_bytes(*chunk);
        } else {
            self.refill_slow();
        }
    }

    /// Cold slow path for refill when we're near the end of the source.
    #[cold]
    #[inline(never)]
    fn refill_slow(&mut self) {
        if self.index > 0 {
            // Read the last portion of source into the `bit_container`
            if self.source.len() >= 8 {
                let chunk: &[u8; 8] = self.source[..8].try_into().unwrap();
                self.bit_container = u64::from_le_bytes(*chunk);
            } else {
                let mut value = [0; 8];
                value[..self.source.len()].copy_from_slice(self.source);
                self.bit_container = u64::from_le_bytes(value);
            }

            self.bits_consumed -= 8 * self.index as u8;
            self.index = 0;

            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else if self.bits_consumed < 64 {
            // Shift out already used bits and fill up with zeroes
            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else {
            // All useful bits have already been read, return zeroes
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            self.bit_container = 0;
        }

        // Assert that at least `56 = 64 - 8` bits are available to read.
        debug_assert!(self.bits_consumed < 8);
    }

    /// Read `n` number of bits from the source. Will read at most 56 bits.
    /// If there are no more bits to be read from the source zero bits will be returned instead.
    #[inline(always)]
    pub fn get_bits(&mut self, n: u8) -> u64 {
        if self.bits_consumed + n > 64 {
            self.refill();
        }

        let value = self.peek_bits(n);
        self.consume(n);
        value
    }

    /// Get the next `n` bits from the source without consuming them.
    /// Caller is responsible for making sure that `n` many bits have been refilled.
    ///
    /// Branchless: handles n=0 correctly (returns 0) without a branch.
    #[inline(always)]
    pub fn peek_bits(&mut self, n: u8) -> u64 {
        let shift_by = (64u8.wrapping_sub(self.bits_consumed).wrapping_sub(n)) & 63;
        let mask = (1u64 << n) - 1u64;
        (self.bit_container >> shift_by) & mask
    }

    /// Get the next `n1`, `n2`, and `n3` bits from the source without consuming them.
    /// Caller is responsible for making sure that `sum` many bits have been refilled.
    ///
    /// Branchless: handles sum=0 and individual n=0 values correctly without branches.
    #[inline(always)]
    pub fn peek_bits_triple(&mut self, sum: u8, n1: u8, n2: u8, n3: u8) -> (u64, u64, u64) {
        // all_three contains bits like this: |XXXX..XXX111122223333|
        // Where XXX are already consumed bytes, 1/2/3 are bits of the respective value
        // Lower bits are to the right
        let shift = (64u8.wrapping_sub(self.bits_consumed).wrapping_sub(sum)) & 63;
        let all_three = self.bit_container >> shift;

        let mask1 = (1u64 << n1) - 1u64;
        let shift_by1 = (n3 + n2) & 63;
        let val1 = (all_three >> shift_by1) & mask1;

        let mask2 = (1u64 << n2) - 1u64;
        let val2 = (all_three >> (n3 & 63)) & mask2;

        let mask3 = (1u64 << n3) - 1u64;
        let val3 = all_three & mask3;

        (val1, val2, val3)
    }

    /// Consume `n` bits from the source.
    #[inline(always)]
    pub fn consume(&mut self, n: u8) {
        self.bits_consumed += n;
        debug_assert!(self.bits_consumed <= 64);
    }

    /// Same as calling get_bits three times but slightly more performant
    #[inline(always)]
    pub fn get_bits_triple(&mut self, n1: u8, n2: u8, n3: u8) -> (u64, u64, u64) {
        let sum = n1 + n2 + n3;
        if sum <= 56 {
            // Only refill if we actually need more bits
            if self.bits_consumed + sum > 64 {
                self.refill();
            }

            let triple = self.peek_bits_triple(sum, n1, n2, n3);
            self.consume(sum);
            return triple;
        }

        (self.get_bits(n1), self.get_bits(n2), self.get_bits(n3))
    }

    /// Returns true if `refill_unconditional` can use its fast path (8-byte load).
    /// This is false near the end of the source where the slow path would fire.
    #[inline(always)]
    pub fn can_refill_fast(&self) -> bool {
        let bytes_consumed = self.bits_consumed as usize / 8;
        self.index >= bytes_consumed && (self.index - bytes_consumed) + 8 <= self.source.len()
    }

    /// Peek at the next `n` bits and advance, WITHOUT checking whether enough
    /// bits are available. The caller must guarantee that at least `n` bits
    /// remain in the container (i.e., `bits_consumed + n <= 64`).
    ///
    /// This is the building block for the "single refill per sequence" pattern:
    /// refill once, then extract multiple values with `peek_and_advance`.
    ///
    /// Branchless: handles n=0 correctly (returns 0) without a branch.
    /// When n=0, mask=0 so the result is zero regardless of the shift.
    #[inline(always)]
    pub fn peek_and_advance(&mut self, n: u8) -> u64 {
        // Mask shift_by to [0,63] to avoid shift overflow when n=0 and bits_consumed=0.
        // When n=0, mask=(1<<0)-1=0, so value=0 regardless of the shifted container.
        let shift_by = (64u8.wrapping_sub(self.bits_consumed).wrapping_sub(n)) & 63;
        let mask = (1u64 << n) - 1u64;
        let value = (self.bit_container >> shift_by) & mask;
        self.bits_consumed += n;
        value
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn it_works() {
        let data = [0b10101010, 0b01010101];
        let mut br = super::BitReaderReversed::new(&data);
        assert_eq!(br.get_bits(1), 0);
        assert_eq!(br.get_bits(1), 1);
        assert_eq!(br.get_bits(1), 0);
        assert_eq!(br.get_bits(4), 0b1010);
        assert_eq!(br.get_bits(4), 0b1101);
        assert_eq!(br.get_bits(4), 0b0101);
        // Last 0 from source, three zeroes filled in
        assert_eq!(br.get_bits(4), 0b0000);
        // All zeroes filled in
        assert_eq!(br.get_bits(4), 0b0000);
        assert_eq!(br.bits_remaining(), -7);
    }

    #[test]
    fn peek_and_advance_matches_get_bits() {
        // Verify that peek_and_advance produces the same results as get_bits
        // when the container is adequately filled.
        let data: [u8; 16] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF,
        ];

        // Use get_bits (the reference)
        let mut br1 = super::BitReaderReversed::new(&data);
        let a1 = br1.get_bits(5);
        let a2 = br1.get_bits(8);
        let a3 = br1.get_bits(13);

        // Use refill_unconditional + peek_and_advance
        let mut br2 = super::BitReaderReversed::new(&data);
        br2.refill_unconditional();
        let b1 = br2.peek_and_advance(5);
        let b2 = br2.peek_and_advance(8);
        let b3 = br2.peek_and_advance(13);

        assert_eq!(a1, b1);
        assert_eq!(a2, b2);
        assert_eq!(a3, b3);
        assert_eq!(br1.bits_remaining(), br2.bits_remaining());
    }

    #[test]
    fn refill_unconditional_basic() {
        let data = [0xFF; 16];
        let mut br = super::BitReaderReversed::new(&data);
        // Initial state: all 64 bits consumed, need refill
        br.refill_unconditional();
        // Should now have 64 bits available
        assert!(br.bits_remaining() >= 56);
        let val = br.peek_and_advance(8);
        assert_eq!(val, 0xFF);
    }
}

//! Pure Rust XXHash64 implementation (no unsafe).
//!
//! Implements the XXH64 algorithm used by Zstandard for frame checksums.
//! Reference: <https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md>

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

/// Compute XXH64 hash of the given data with the specified seed.
pub fn xxhash64(data: &[u8], seed: u64) -> u64 {
    let len = data.len() as u64;
    let mut h64: u64;

    if data.len() >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);

        // Process 32 bytes at a time. Use chunks_exact to get &[u8] slices
        // that LLVM knows are exactly 32 bytes, then cast to &[u8; 32] once.
        // All sub-array indexing on a [u8; 32] is compile-time provable in-bounds.
        let chunks = data.chunks_exact(32);
        let remainder = chunks.remainder();
        for chunk in chunks {
            let c: &[u8; 32] = chunk.try_into().unwrap();
            // Sub-slicing a fixed array: LLVM proves [0..8] etc. are in-bounds at compile time
            v1 = round(v1, u64::from_le_bytes(read8(c, 0)));
            v2 = round(v2, u64::from_le_bytes(read8(c, 8)));
            v3 = round(v3, u64::from_le_bytes(read8(c, 16)));
            v4 = round(v4, u64::from_le_bytes(read8(c, 24)));
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));

        h64 = merge_round(h64, v1);
        h64 = merge_round(h64, v2);
        h64 = merge_round(h64, v3);
        h64 = merge_round(h64, v4);

        h64 = finalize_remaining(h64, remainder, len);
    } else {
        h64 = seed.wrapping_add(PRIME64_5);
        h64 = finalize_remaining(h64, data, len);
    }

    h64
}

fn finalize_remaining(mut h64: u64, data: &[u8], total_len: u64) -> u64 {
    h64 = h64.wrapping_add(total_len);

    let mut offset = 0;

    while offset + 8 <= data.len() {
        let k1 = round(0, read_u64_le(&data[offset..]));
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(PRIME64_1)
            .wrapping_add(PRIME64_4);
        offset += 8;
    }

    if offset + 4 <= data.len() {
        let k1 = u64::from(read_u32_le(&data[offset..]));
        h64 ^= k1.wrapping_mul(PRIME64_1);
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(PRIME64_2)
            .wrapping_add(PRIME64_3);
        offset += 4;
    }

    while offset < data.len() {
        h64 ^= u64::from(data[offset]).wrapping_mul(PRIME64_5);
        h64 = h64.rotate_left(11).wrapping_mul(PRIME64_1);
        offset += 1;
    }

    avalanche(h64)
}

/// Extract 8 bytes from a 32-byte array at a known offset.
/// Because `src` is `&[u8; 32]` and `off` is a constant 0/8/16/24,
/// LLVM proves the sub-slice is in-bounds at compile time — zero runtime checks.
#[inline(always)]
fn read8(src: &[u8; 32], off: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&src[off..off + 8]);
    out
}

#[inline(always)]
fn round(mut acc: u64, input: u64) -> u64 {
    acc = acc.wrapping_add(input.wrapping_mul(PRIME64_2));
    acc = acc.rotate_left(31);
    acc.wrapping_mul(PRIME64_1)
}

#[inline(always)]
fn merge_round(mut acc: u64, val: u64) -> u64 {
    let val = round(0, val);
    acc ^= val;
    acc.wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4)
}

#[inline(always)]
fn avalanche(mut h64: u64) -> u64 {
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(PRIME64_2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(PRIME64_3);
    h64 ^= h64 >> 32;
    h64
}

#[inline(always)]
fn read_u64_le(data: &[u8]) -> u64 {
    u64::from_le_bytes(data[..8].try_into().unwrap())
}

#[inline(always)]
fn read_u32_le(data: &[u8]) -> u32 {
    u32::from_le_bytes(data[..4].try_into().unwrap())
}

/// Streaming XXH64 hasher, compatible with `std::hash::Hasher` semantics.
pub struct XxHash64 {
    seed: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    total_len: u64,
    buf: [u8; 32],
    buf_len: usize,
}

impl XxHash64 {
    /// Create a new hasher with the given seed.
    pub fn with_seed(seed: u64) -> Self {
        XxHash64 {
            seed,
            v1: seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2),
            v2: seed.wrapping_add(PRIME64_2),
            v3: seed,
            v4: seed.wrapping_sub(PRIME64_1),
            total_len: 0,
            buf: [0u8; 32],
            buf_len: 0,
        }
    }

    /// Feed data into the hasher.
    #[inline]
    pub fn write(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        let mut offset = 0;

        // If we have buffered data, try to fill the buffer
        if self.buf_len > 0 {
            let need = 32 - self.buf_len;
            let take = data.len().min(need);
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            offset += take;

            if self.buf_len == 32 {
                // self.buf is [u8; 32] — sub-indexing proven in-bounds at compile time
                self.v1 = round(self.v1, u64::from_le_bytes(read8(&self.buf, 0)));
                self.v2 = round(self.v2, u64::from_le_bytes(read8(&self.buf, 8)));
                self.v3 = round(self.v3, u64::from_le_bytes(read8(&self.buf, 16)));
                self.v4 = round(self.v4, u64::from_le_bytes(read8(&self.buf, 24)));
                self.buf_len = 0;
            }
        }

        // Process 32-byte chunks via chunks_exact — one try_into per 32 bytes
        let tail = &data[offset..];
        let chunks = tail.chunks_exact(32);
        let leftover = chunks.remainder();
        for chunk in chunks {
            let c: &[u8; 32] = chunk.try_into().unwrap();
            self.v1 = round(self.v1, u64::from_le_bytes(read8(c, 0)));
            self.v2 = round(self.v2, u64::from_le_bytes(read8(c, 8)));
            self.v3 = round(self.v3, u64::from_le_bytes(read8(c, 16)));
            self.v4 = round(self.v4, u64::from_le_bytes(read8(c, 24)));
        }

        // Buffer remaining bytes (0-31)
        if !leftover.is_empty() {
            self.buf[..leftover.len()].copy_from_slice(leftover);
            self.buf_len = leftover.len();
        }
    }

    /// Finalize and return the hash value.
    pub fn finish(&self) -> u64 {
        let mut h64: u64;

        if self.total_len >= 32 {
            h64 = self
                .v1
                .rotate_left(1)
                .wrapping_add(self.v2.rotate_left(7))
                .wrapping_add(self.v3.rotate_left(12))
                .wrapping_add(self.v4.rotate_left(18));

            h64 = merge_round(h64, self.v1);
            h64 = merge_round(h64, self.v2);
            h64 = merge_round(h64, self.v3);
            h64 = merge_round(h64, self.v4);
        } else {
            h64 = self.seed.wrapping_add(PRIME64_5);
        }

        finalize_remaining(h64, &self.buf[..self.buf_len], self.total_len)
    }

    /// Reset the hasher state.
    pub fn reset(&mut self) {
        *self = Self::with_seed(self.seed);
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_empty() {
        assert_eq!(xxhash64(b"", 0), 0xEF46DB3751D8E999);
    }

    #[test]
    fn test_one_byte() {
        assert_eq!(xxhash64(&[0x00], 0), 0xE934A84ADB052768);
    }

    #[test]
    fn test_known_vectors() {
        let h = xxhash64(b"abc", 0);
        assert_eq!(h, 0x44BC2CF5AD770999);
    }

    #[test]
    fn test_32_bytes() {
        let data: Vec<u8> = (0..32u8).collect();
        let h = xxhash64(&data, 0);
        assert_ne!(h, 0);
    }

    #[test]
    fn test_streaming_matches_oneshot() {
        let data: Vec<u8> = (0..200u8).collect();

        let oneshot = xxhash64(&data, 0);

        let mut hasher = XxHash64::with_seed(0);
        hasher.write(&data);
        let streaming = hasher.finish();

        assert_eq!(oneshot, streaming);
    }

    #[test]
    fn test_streaming_incremental() {
        let data: Vec<u8> = (0..200u8).collect();
        let expected = xxhash64(&data, 0);

        let mut hasher = XxHash64::with_seed(0);
        for byte in &data {
            hasher.write(core::slice::from_ref(byte));
        }
        assert_eq!(hasher.finish(), expected);

        let mut hasher = XxHash64::with_seed(0);
        for chunk in data.chunks(7) {
            hasher.write(chunk);
        }
        assert_eq!(hasher.finish(), expected);
    }

    #[test]
    fn test_with_seed() {
        let h0 = xxhash64(b"hello", 0);
        let h1 = xxhash64(b"hello", 1);
        assert_ne!(h0, h1);
    }
}

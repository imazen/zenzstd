//! Round-trip tests for all compression levels.

#[test]
fn test_level_1_roundtrip() {
    extern crate std;
    use alloc::vec;
    use alloc::vec::Vec;
    let data = vec![0u8; 10000];
    roundtrip_both(&data, crate::encoding::CompressionLevel::Level(1));
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    roundtrip_both(&data, crate::encoding::CompressionLevel::Level(1));
}

#[test]
fn test_level_3_roundtrip() {
    use alloc::vec;
    use alloc::vec::Vec;
    let data = vec![0u8; 10000];
    roundtrip_both(&data, crate::encoding::CompressionLevel::Default);
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    roundtrip_both(&data, crate::encoding::CompressionLevel::Default);
}

#[test]
fn test_level_7_roundtrip() {
    use alloc::vec;
    use alloc::vec::Vec;
    let data = vec![0u8; 10000];
    roundtrip_both(&data, crate::encoding::CompressionLevel::Better);
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    roundtrip_both(&data, crate::encoding::CompressionLevel::Better);
}

#[test]
fn test_level_11_roundtrip() {
    use alloc::vec;
    use alloc::vec::Vec;
    let data = vec![0u8; 10000];
    roundtrip_both(&data, crate::encoding::CompressionLevel::Best);
    let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    roundtrip_both(&data, crate::encoding::CompressionLevel::Best);
}

#[test]
fn test_all_levels_roundtrip_zeros() {
    use alloc::vec;
    let data = vec![0u8; 5000];
    for level in 1..=22 {
        roundtrip_both(&data, crate::encoding::CompressionLevel::Level(level));
    }
}

#[test]
fn test_all_levels_roundtrip_pattern() {
    use alloc::vec::Vec;
    let mut data = Vec::new();
    for _ in 0..100 {
        data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
    }
    for level in 1..=22 {
        roundtrip_both(&data, crate::encoding::CompressionLevel::Level(level));
    }
}

#[test]
fn test_all_levels_roundtrip_mixed() {
    use alloc::vec;
    use alloc::vec::Vec;
    let mut data = vec![0u8; 2000];
    data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    for i in 0..2000u32 {
        data.push(((i * 7 + 13) % 256) as u8);
    }
    data.extend(vec![0xAA; 1000]);
    for level in 1..=22 {
        roundtrip_both(&data, crate::encoding::CompressionLevel::Level(level));
    }
}

#[test]
fn test_all_levels_empty_input() {
    for level in 1..=22 {
        roundtrip_both(&[], crate::encoding::CompressionLevel::Level(level));
    }
}

#[test]
fn test_all_levels_tiny_input() {
    for level in 1..=22 {
        roundtrip_both(&[42], crate::encoding::CompressionLevel::Level(level));
        roundtrip_both(&[1, 2, 3], crate::encoding::CompressionLevel::Level(level));
    }
}

#[test]
fn test_compression_actually_compresses() {
    use alloc::vec::Vec;
    let mut data = Vec::new();
    for _ in 0..200 {
        data.extend_from_slice(b"Hello World! This is a test of the zstd compression. ");
    }
    for level in [1, 3, 7, 11] {
        let compressed =
            crate::encoding::compress_to_vec(data.as_slice(), crate::encoding::CompressionLevel::Level(level));
        assert!(
            compressed.len() < data.len(),
            "Level {} failed to compress: {} -> {} bytes",
            level,
            data.len(),
            compressed.len(),
        );
    }
}

// Helper functions used by all tests above
#[cfg(test)]
fn roundtrip_both(data: &[u8], level: crate::encoding::CompressionLevel) {
    roundtrip_our_decoder(data, level);
    roundtrip_zstd_decoder(data, level);
}

#[cfg(test)]
fn roundtrip_our_decoder(data: &[u8], level: crate::encoding::CompressionLevel) {
    use alloc::vec::Vec;
    let compressed = crate::encoding::compress_to_vec(data, level);
    let mut decoder = crate::decoding::FrameDecoder::new();
    let mut decoded = Vec::with_capacity(data.len() + 1024);
    decoder
        .decode_all_to_vec(&compressed, &mut decoded)
        .unwrap();
    assert_eq!(data, &decoded[..]);
}

#[cfg(test)]
fn roundtrip_zstd_decoder(data: &[u8], level: crate::encoding::CompressionLevel) {
    use alloc::vec::Vec;
    let compressed = crate::encoding::compress_to_vec(data, level);
    let mut decoded = Vec::new();
    zstd::stream::copy_decode(compressed.as_slice(), &mut decoded).unwrap();
    assert_eq!(data, &decoded[..]);
}

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use zenzstd::decoding::dictionary::Dictionary;
use zenzstd::decoding::scratch::{FSEScratch, HuffmanScratch};
use zenzstd::decoding::FrameDecoder;
use zenzstd::encoding::{compress_to_vec_with_dict, CompressionLevel, EncoderDictionary};

#[derive(Debug, Arbitrary)]
struct DictInput {
    /// Dictionary content (will be used as raw dictionary)
    dict_content: Vec<u8>,
    /// Dictionary ID (will be clamped to 1.. to avoid zero)
    dict_id: u32,
    /// Compression level 0-22
    level: u8,
    /// The data to compress
    data: Vec<u8>,
}

impl DictInput {
    fn compression_level(&self) -> CompressionLevel {
        match self.level % 23 {
            0 => CompressionLevel::Uncompressed,
            1 => CompressionLevel::Fastest,
            3 => CompressionLevel::Default,
            7 => CompressionLevel::Better,
            11 => CompressionLevel::Best,
            n => CompressionLevel::Level(n as i32),
        }
    }
}

// Compress with a raw dictionary, then decompress and verify.
// The dictionary is arbitrary content (not a formatted zstd dict),
// so we use EncoderDictionary::new_raw for encoding and construct
// a matching decoder Dictionary manually.
fuzz_target!(|input: DictInput| {
    // Limit sizes to avoid OOM
    if input.data.len() > 256 * 1024 {
        return;
    }
    if input.dict_content.len() > 64 * 1024 {
        return;
    }
    // Dictionary content must be non-empty for meaningful test
    if input.dict_content.is_empty() {
        return;
    }

    // Dictionary ID must be non-zero
    let dict_id = if input.dict_id == 0 { 1 } else { input.dict_id };

    let enc_dict = EncoderDictionary::new_raw(dict_id, input.dict_content.clone());
    let level = input.compression_level();

    let compressed = compress_to_vec_with_dict(input.data.as_slice(), level, &enc_dict);

    // Construct a raw content decoder dictionary with default entropy tables.
    // For raw content dicts the encoder doesn't reference the dict's entropy
    // tables, so default (empty) tables are correct for the decoder too.
    let dec_dict = Dictionary {
        id: dict_id,
        fse: FSEScratch::new(),
        huf: HuffmanScratch::new(),
        dict_content: input.dict_content.clone(),
        offset_hist: [1, 4, 8], // same defaults as EncoderDictionary::new_raw
    };

    let mut decoder = FrameDecoder::new();
    decoder.add_dict(dec_dict).expect("adding dictionary must succeed");

    let mut decompressed = Vec::with_capacity(input.data.len());
    decoder
        .decode_all_to_vec(&compressed, &mut decompressed)
        .expect("decompression with dictionary must succeed");

    assert_eq!(
        input.data, decompressed,
        "dictionary roundtrip mismatch at level {:?} with dict_id {}",
        level, dict_id
    );
});

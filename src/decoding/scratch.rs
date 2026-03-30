//! Structures that wrap around various decoders to make decoding easier.

use super::super::blocks::sequence_section::Sequence;
use super::decode_buffer::DecodeBuffer;
use crate::decoding::dictionary::Dictionary;
use crate::fse::{FSETable, SeqEntry};
use crate::huff0::HuffmanTable;
use alloc::vec::Vec;

use crate::blocks::sequence_section::{
    MAX_LITERAL_LENGTH_CODE, MAX_MATCH_LENGTH_CODE, MAX_OFFSET_CODE,
};

/// A block level decoding buffer.
pub struct DecoderScratch {
    /// The decoder used for Huffman blocks.
    pub huf: HuffmanScratch,
    /// The decoder used for FSE blocks.
    pub fse: FSEScratch,

    pub buffer: DecodeBuffer,
    pub offset_hist: [u32; 3],

    pub literals_buffer: Vec<u8>,
    pub sequences: Vec<Sequence>,
    pub block_content_buffer: Vec<u8>,
}

impl DecoderScratch {
    pub fn new(window_size: usize) -> DecoderScratch {
        DecoderScratch {
            huf: HuffmanScratch {
                table: HuffmanTable::new(),
            },
            fse: FSEScratch::new(),
            buffer: DecodeBuffer::new(window_size),
            offset_hist: [1, 4, 8],

            block_content_buffer: Vec::new(),
            literals_buffer: Vec::new(),
            sequences: Vec::new(),
        }
    }

    pub fn reset(&mut self, window_size: usize) {
        self.offset_hist = [1, 4, 8];
        self.literals_buffer.clear();
        self.sequences.clear();
        // Don't clear block_content_buffer — it is fully overwritten by
        // read_exact on each block. Keeping its length avoids a memset
        // (resize from 0→128KB) on the first compressed block of each frame.

        self.buffer.reset(window_size);

        self.fse.literal_lengths.reset();
        self.fse.match_lengths.reset();
        self.fse.offsets.reset();
        self.fse.ll_rle = None;
        self.fse.ml_rle = None;
        self.fse.of_rle = None;

        self.huf.table.reset();
    }

    pub fn init_from_dict(&mut self, dict: &Dictionary) {
        self.fse.reinit_from(&dict.fse);
        self.huf.table.reinit_from(&dict.huf.table);
        self.offset_hist = dict.offset_hist;
        self.buffer.dict_content.clear();
        self.buffer
            .dict_content
            .extend_from_slice(&dict.dict_content);
    }
}

pub struct HuffmanScratch {
    pub table: HuffmanTable,
}

impl HuffmanScratch {
    pub fn new() -> HuffmanScratch {
        HuffmanScratch {
            table: HuffmanTable::new(),
        }
    }
}

impl Default for HuffmanScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FSEScratch {
    pub offsets: FSETable,
    pub of_rle: Option<u8>,
    pub literal_lengths: FSETable,
    pub ll_rle: Option<u8>,
    pub match_lengths: FSETable,
    pub ml_rle: Option<u8>,

    /// Pre-resolved sequence tables: combine FSE state transition info
    /// with the LL/ML/OF code table lookups into a single entry.
    /// Built by `rebuild_seq_tables()` after any FSE table update.
    pub seq_ll: Vec<SeqEntry>,
    pub seq_ml: Vec<SeqEntry>,
    pub seq_of: Vec<SeqEntry>,

    /// Cached predefined decode tables. Built once on first use, then
    /// `restore_from_prebuilt` copies the decode array instead of recomputing.
    predefined_ll: Option<PredefinedCache>,
    predefined_ml: Option<PredefinedCache>,
    predefined_of: Option<PredefinedCache>,
}

/// Cached prebuilt FSE decode array for a predefined distribution.
#[derive(Clone)]
struct PredefinedCache {
    accuracy_log: u8,
    decode: Vec<crate::fse::Entry>,
}

impl FSEScratch {
    pub fn new() -> FSEScratch {
        FSEScratch {
            offsets: FSETable::new(MAX_OFFSET_CODE),
            of_rle: None,
            literal_lengths: FSETable::new(MAX_LITERAL_LENGTH_CODE),
            ll_rle: None,
            match_lengths: FSETable::new(MAX_MATCH_LENGTH_CODE),
            ml_rle: None,
            seq_ll: Vec::new(),
            seq_ml: Vec::new(),
            seq_of: Vec::new(),
            predefined_ll: None,
            predefined_ml: None,
            predefined_of: None,
        }
    }

    pub fn reinit_from(&mut self, other: &Self) {
        self.offsets.reinit_from(&other.offsets);
        self.literal_lengths.reinit_from(&other.literal_lengths);
        self.match_lengths.reinit_from(&other.match_lengths);
        self.of_rle = other.of_rle;
        self.ll_rle = other.ll_rle;
        self.ml_rle = other.ml_rle;
        // Copy pre-resolved seq tables
        self.seq_ll.clear();
        self.seq_ll.extend_from_slice(&other.seq_ll);
        self.seq_ml.clear();
        self.seq_ml.extend_from_slice(&other.seq_ml);
        self.seq_of.clear();
        self.seq_of.extend_from_slice(&other.seq_of);
        // Don't copy predefined caches — they are decoder-lifetime, not per-frame.
    }

    /// Build the predefined literal-lengths table, using a cached decode array
    /// if available. Returns `Ok(())` on success.
    pub fn build_predefined_ll(
        &mut self,
        acc_log: u8,
        probs: &[i32],
    ) -> Result<(), crate::decoding::errors::FSETableError> {
        if let Some(ref cache) = self.predefined_ll {
            self.literal_lengths
                .restore_from_prebuilt(cache.accuracy_log, &cache.decode);
            return Ok(());
        }
        self.literal_lengths
            .build_from_probabilities(acc_log, probs)?;
        self.predefined_ll = Some(PredefinedCache {
            accuracy_log: self.literal_lengths.accuracy_log,
            decode: self.literal_lengths.decode.clone(),
        });
        Ok(())
    }

    /// Build the predefined match-lengths table, using a cached decode array
    /// if available. Returns `Ok(())` on success.
    pub fn build_predefined_ml(
        &mut self,
        acc_log: u8,
        probs: &[i32],
    ) -> Result<(), crate::decoding::errors::FSETableError> {
        if let Some(ref cache) = self.predefined_ml {
            self.match_lengths
                .restore_from_prebuilt(cache.accuracy_log, &cache.decode);
            return Ok(());
        }
        self.match_lengths
            .build_from_probabilities(acc_log, probs)?;
        self.predefined_ml = Some(PredefinedCache {
            accuracy_log: self.match_lengths.accuracy_log,
            decode: self.match_lengths.decode.clone(),
        });
        Ok(())
    }

    /// Build the predefined offsets table, using a cached decode array
    /// if available. Returns `Ok(())` on success.
    pub fn build_predefined_of(
        &mut self,
        acc_log: u8,
        probs: &[i32],
    ) -> Result<(), crate::decoding::errors::FSETableError> {
        if let Some(ref cache) = self.predefined_of {
            self.offsets
                .restore_from_prebuilt(cache.accuracy_log, &cache.decode);
            return Ok(());
        }
        self.offsets.build_from_probabilities(acc_log, probs)?;
        self.predefined_of = Some(PredefinedCache {
            accuracy_log: self.offsets.accuracy_log,
            decode: self.offsets.decode.clone(),
        });
        Ok(())
    }

    /// Rebuild the pre-resolved SeqEntry tables from the current FSE decode tables.
    /// Must be called after any FSE table update (FSECompressed, Predefined, or Repeat
    /// when the underlying table changed).
    ///
    /// `ll_code_table` and `ml_code_table` map symbol code -> (base_value, extra_bits).
    pub fn rebuild_seq_tables(&mut self, ll_code_table: &[(u32, u8)], ml_code_table: &[(u32, u8)]) {
        if self.ll_rle.is_none() {
            self.seq_ll = crate::fse::build_seq_table(&self.literal_lengths, ll_code_table);
        }
        if self.ml_rle.is_none() {
            self.seq_ml = crate::fse::build_seq_table(&self.match_lengths, ml_code_table);
        }
        if self.of_rle.is_none() {
            self.seq_of = crate::fse::build_seq_table_offset(&self.offsets);
        }
    }
}

impl Default for FSEScratch {
    fn default() -> Self {
        Self::new()
    }
}

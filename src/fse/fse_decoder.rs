use crate::bit_io::{BitReader, BitReaderReversed};
use crate::decoding::errors::{FSEDecoderError, FSETableError};
use alloc::vec::Vec;

pub struct FSEDecoder<'table> {
    /// An FSE state value represents an index in the FSE table.
    pub state: Entry,
    /// A reference to the table used for decoding.
    table: &'table FSETable,
}

impl<'t> FSEDecoder<'t> {
    /// Initialize a new Finite State Entropy decoder.
    pub fn new(table: &'t FSETable) -> FSEDecoder<'t> {
        FSEDecoder {
            state: table.decode.first().copied().unwrap_or(Entry {
                base_line: 0,
                num_bits: 0,
                symbol: 0,
            }),
            table,
        }
    }

    /// Returns the byte associated with the symbol the internal cursor is pointing at.
    #[inline(always)]
    pub fn decode_symbol(&self) -> u8 {
        self.state.symbol
    }

    /// Initialize internal state and prepare for decoding. After this, `decode_symbol` can be called
    /// to read the first symbol and `update_state` can be called to prepare to read the next symbol.
    pub fn init_state(&mut self, bits: &mut BitReaderReversed<'_>) -> Result<(), FSEDecoderError> {
        if self.table.accuracy_log == 0 {
            return Err(FSEDecoderError::TableIsUninitialized);
        }
        let new_state = bits.get_bits(self.table.accuracy_log);
        let table_mask = self.table.decode.len() - 1;
        self.state = self.table.decode[new_state as usize & table_mask];

        Ok(())
    }

    /// Advance the internal state to decode the next symbol in the bitstream.
    #[inline(always)]
    pub fn update_state(&mut self, bits: &mut BitReaderReversed<'_>) {
        let num_bits = self.state.num_bits;
        let add = bits.get_bits(num_bits);
        let base_line = self.state.base_line;
        let new_state = base_line + add as u32;
        let table_mask = self.table.decode.len() - 1;
        self.state = self.table.decode[new_state as usize & table_mask];
    }

    /// Read the number of bits needed for the state update and return (num_bits, base_line).
    /// The caller can batch the bit reads.
    #[allow(dead_code)] // kept for public API / fuzz harness compatibility
    #[inline(always)]
    pub fn state_update_params(&self) -> (u8, u32) {
        (self.state.num_bits, self.state.base_line)
    }

    /// Apply a pre-read bit value to update state. `add` is the value previously read
    /// using `state_update_params().0` bits.
    #[inline(always)]
    pub fn apply_state_update(&mut self, add: u64) {
        let new_state = self.state.base_line + add as u32;
        // Table size is always (1 << accuracy_log), a power of 2.
        // Masking helps LLVM prove the index is in-bounds and elide the bounds check.
        let table_mask = self.table.decode.len() - 1;
        self.state = self.table.decode[new_state as usize & table_mask];
    }

    /// Combined decode + state-update-params: returns (symbol, num_bits_for_update, baseline).
    /// Avoids redundant Entry field loads by doing both operations from the same cached state.
    #[inline(always)]
    pub fn decode_and_params(&self) -> (u8, u8, u32) {
        let entry = self.state;
        (entry.symbol, entry.num_bits, entry.base_line)
    }
}

/// FSE decoding involves a decoding table that describes the probabilities of
/// all literals from 0 to the highest present one
///
/// <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#fse-table-description>
#[derive(Debug, Clone)]
pub struct FSETable {
    /// The maximum symbol in the table (inclusive). Limits the probabilities length to max_symbol + 1.
    max_symbol: u8,
    /// The actual table containing the decoded symbol and the compression data
    /// connected to that symbol.
    pub decode: Vec<Entry>, //used to decode symbols, and calculate the next state
    /// The size of the table is stored in logarithm base 2 format,
    /// with the **size of the table** being equal to `(1 << accuracy_log)`.
    /// This value is used so that the decoder knows how many bits to read from the bitstream.
    pub accuracy_log: u8,
    /// In this context, probability refers to the likelihood that a symbol occurs in the given data.
    /// Given this info, the encoder can assign shorter codes to symbols that appear more often,
    /// and longer codes that appear less often, then the decoder can use the probability
    /// to determine what code was assigned to what symbol.
    ///
    /// The probability of a single symbol is a value representing the proportion of times the symbol
    /// would fall within the data.
    ///
    /// If a symbol probability is set to `-1`, it means that the probability of a symbol
    /// occurring in the data is less than one.
    pub symbol_probabilities: Vec<i32>, //used while building the decode Vector
    /// The number of times each symbol occurs (The first entry being 0x0, the second being 0x1) and so on
    /// up until the highest possible symbol (255).
    symbol_counter: Vec<u32>,
}

impl FSETable {
    /// Initialize a new empty Finite State Entropy decoding table.
    pub fn new(max_symbol: u8) -> FSETable {
        FSETable {
            max_symbol,
            symbol_probabilities: Vec::with_capacity(256), //will never be more than 256 symbols because u8
            symbol_counter: Vec::with_capacity(256), //will never be more than 256 symbols because u8
            decode: Vec::new(),                      //depending on acc_log.
            accuracy_log: 0,
        }
    }

    /// Reset `self` and update `self`'s state to mirror the provided table.
    pub fn reinit_from(&mut self, other: &Self) {
        self.reset();
        self.symbol_counter.extend_from_slice(&other.symbol_counter);
        self.symbol_probabilities
            .extend_from_slice(&other.symbol_probabilities);
        self.decode.extend_from_slice(&other.decode);
        self.accuracy_log = other.accuracy_log;
    }

    /// Empty the table and clear all internal state.
    pub fn reset(&mut self) {
        self.symbol_counter.clear();
        self.symbol_probabilities.clear();
        self.decode.clear();
        self.accuracy_log = 0;
    }

    /// returns how many BYTEs (not bits) were read while building the decoder
    pub fn build_decoder(&mut self, source: &[u8], max_log: u8) -> Result<usize, FSETableError> {
        self.accuracy_log = 0;

        let bytes_read = self.read_probabilities(source, max_log)?;
        self.build_decoding_table()?;

        Ok(bytes_read)
    }

    /// Given the provided accuracy log, build a decoding table from that log.
    pub fn build_from_probabilities(
        &mut self,
        acc_log: u8,
        probs: &[i32],
    ) -> Result<(), FSETableError> {
        if acc_log == 0 {
            return Err(FSETableError::AccLogIsZero);
        }
        self.symbol_probabilities = probs.to_vec();
        self.accuracy_log = acc_log;
        self.build_decoding_table()
    }

    /// Restore this table from a pre-built decode array and accuracy log.
    /// This skips the `build_decoding_table` computation entirely.
    pub fn restore_from_prebuilt(&mut self, acc_log: u8, decode: &[Entry]) {
        self.accuracy_log = acc_log;
        self.decode.clear();
        self.decode.extend_from_slice(decode);
        self.symbol_probabilities.clear();
        self.symbol_counter.clear();
    }

    /// Build the actual decoding table after probabilities have been read into the table.
    /// After this function is called, the decoding process can begin.
    fn build_decoding_table(&mut self) -> Result<(), FSETableError> {
        if self.symbol_probabilities.len() > self.max_symbol as usize + 1 {
            return Err(FSETableError::TooManySymbols {
                got: self.symbol_probabilities.len(),
            });
        }

        self.decode.clear();

        let table_size = 1 << self.accuracy_log;
        if self.decode.len() < table_size {
            self.decode.reserve(table_size - self.decode.len());
        }
        //fill with dummy entries
        self.decode.resize(
            table_size,
            Entry {
                base_line: 0,
                num_bits: 0,
                symbol: 0,
            },
        );

        let mut negative_idx = table_size; //will point to the highest index with is already occupied by a negative-probability-symbol

        //first scan for all -1 probabilities and place them at the top of the table
        for symbol in 0..self.symbol_probabilities.len() {
            if self.symbol_probabilities[symbol] == -1 {
                negative_idx -= 1;
                let entry = &mut self.decode[negative_idx];
                entry.symbol = symbol as u8;
                entry.base_line = 0;
                entry.num_bits = self.accuracy_log;
            }
        }

        //then place in a semi-random order all of the other symbols
        let mut position = 0;
        for idx in 0..self.symbol_probabilities.len() {
            let symbol = idx as u8;
            if self.symbol_probabilities[idx] <= 0 {
                continue;
            }

            //for each probability point the symbol gets on slot
            let prob = self.symbol_probabilities[idx];
            for _ in 0..prob {
                let entry = &mut self.decode[position];
                entry.symbol = symbol;

                position = next_position(position, table_size);
                while position >= negative_idx {
                    position = next_position(position, table_size);
                    //everything above negative_idx is already taken
                }
            }
        }

        // baselines and num_bits can only be calculated when all symbols have been spread
        self.symbol_counter.clear();
        self.symbol_counter
            .resize(self.symbol_probabilities.len(), 0);
        for idx in 0..negative_idx {
            let entry = &mut self.decode[idx];
            let symbol = entry.symbol;
            let prob = self.symbol_probabilities[symbol as usize];

            let symbol_count = self.symbol_counter[symbol as usize];
            let (bl, nb) = calc_baseline_and_numbits(table_size as u32, prob as u32, symbol_count);

            //println!("symbol: {:2}, table: {}, prob: {:3}, count: {:3}, bl: {:3}, nb: {:2}", symbol, table_size, prob, symbol_count, bl, nb);

            assert!(nb <= self.accuracy_log);
            self.symbol_counter[symbol as usize] += 1;

            entry.base_line = bl;
            entry.num_bits = nb;
        }
        Ok(())
    }

    /// Read the accuracy log and the probability table from the source and return the number of bytes
    /// read. If the size of the table is larger than the provided `max_log`, return an error.
    fn read_probabilities(&mut self, source: &[u8], max_log: u8) -> Result<usize, FSETableError> {
        self.symbol_probabilities.clear(); //just clear, we will fill a probability for each entry anyways. No need to force new allocs here

        let mut br = BitReader::new(source);
        self.accuracy_log = ACC_LOG_OFFSET + (br.get_bits(4)? as u8);
        if self.accuracy_log > max_log {
            return Err(FSETableError::AccLogTooBig {
                got: self.accuracy_log,
                max: max_log,
            });
        }
        if self.accuracy_log == 0 {
            return Err(FSETableError::AccLogIsZero);
        }

        let probability_sum = 1 << self.accuracy_log;
        let mut probability_counter = 0;

        while probability_counter < probability_sum {
            let max_remaining_value = probability_sum - probability_counter + 1;
            let bits_to_read = highest_bit_set(max_remaining_value);

            let unchecked_value = br.get_bits(bits_to_read as usize)? as u32;

            let low_threshold = ((1 << bits_to_read) - 1) - (max_remaining_value);
            let mask = (1 << (bits_to_read - 1)) - 1;
            let small_value = unchecked_value & mask;

            let value = if small_value < low_threshold {
                br.return_bits(1);
                small_value
            } else if unchecked_value > mask {
                unchecked_value - low_threshold
            } else {
                unchecked_value
            };
            //println!("{}, {}, {}", self.symbol_probablilities.len(), unchecked_value, value);

            let prob = (value as i32) - 1;

            self.symbol_probabilities.push(prob);
            if prob != 0 {
                if prob > 0 {
                    probability_counter += prob as u32;
                } else {
                    // probability -1 counts as 1
                    assert!(prob == -1);
                    probability_counter += 1;
                }
            } else {
                //fast skip further zero probabilities
                loop {
                    let skip_amount = br.get_bits(2)? as usize;

                    self.symbol_probabilities
                        .resize(self.symbol_probabilities.len() + skip_amount, 0);
                    if skip_amount != 3 {
                        break;
                    }
                }
            }
        }

        if probability_counter != probability_sum {
            return Err(FSETableError::ProbabilityCounterMismatch {
                got: probability_counter,
                expected_sum: probability_sum,
                symbol_probabilities: self.symbol_probabilities.clone(),
            });
        }
        if self.symbol_probabilities.len() > self.max_symbol as usize + 1 {
            return Err(FSETableError::TooManySymbols {
                got: self.symbol_probabilities.len(),
            });
        }

        let bytes_read = if br.bits_read() % 8 == 0 {
            br.bits_read() / 8
        } else {
            (br.bits_read() / 8) + 1
        };

        Ok(bytes_read)
    }
}

/// A single entry in an FSE table.
#[derive(Copy, Clone, Debug)]
pub struct Entry {
    /// This value is used as an offset value, and it is added
    /// to a value read from the stream to determine the next state value.
    pub base_line: u32,
    /// How many bits should be read from the stream when decoding this entry.
    pub num_bits: u8,
    /// The byte that should be put in the decode output when encountering this state.
    pub symbol: u8,
}

/// This value is added to the first 4 bits of the stream to determine the
/// `Accuracy_Log`
const ACC_LOG_OFFSET: u8 = 5;

fn highest_bit_set(x: u32) -> u32 {
    assert!(x > 0);
    u32::BITS - x.leading_zeros()
}

//utility functions for building the decoding table from probabilities
/// Calculate the position of the next entry of the table given the current
/// position and size of the table.
fn next_position(mut p: usize, table_size: usize) -> usize {
    p += (table_size >> 1) + (table_size >> 3) + 3;
    p &= table_size - 1;
    p
}

/// A pre-resolved FSE table entry for sequence decoding that eliminates
/// the separate LL_CODE_TABLE / ML_CODE_TABLE lookups.
///
/// Mirrors C zstd's `ZSTD_seqSymbol` layout: 8 bytes, packed so that a
/// single table load gives everything needed to decode a sequence component.
///
/// Fields:
/// - `next_state`: base state for the FSE state transition (Entry::base_line)
/// - `state_bits`: bits to read for the state transition (Entry::num_bits)
/// - `extra_bits`: additional bits to read for the decoded value
/// - `base_value`: baseline value for the decoded output
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SeqEntry {
    pub base_value: u32,
    pub next_state: u16,
    pub state_bits: u8,
    pub extra_bits: u8,
}

/// Decoder for pre-resolved sequence FSE tables.
///
/// Unlike `FSEDecoder` which returns a raw symbol code, this decoder
/// gives direct access to `base_value` and `extra_bits` without a
/// second lookup through LL_CODE_TABLE or ML_CODE_TABLE.
///
/// The table mask is precomputed to avoid a length load on every state update.
pub struct SeqFSEDecoder<'table> {
    pub state: usize,
    table: &'table [SeqEntry],
    /// Precomputed `table.len() - 1`. Table sizes are always powers of 2.
    mask: usize,
}

impl<'t> SeqFSEDecoder<'t> {
    #[inline(always)]
    pub fn new(table: &'t [SeqEntry]) -> SeqFSEDecoder<'t> {
        SeqFSEDecoder {
            state: 0,
            table,
            mask: table.len().wrapping_sub(1),
        }
    }

    /// Initialize state from the bitstream (reads accuracy_log bits).
    #[inline(always)]
    pub fn init_state(
        &mut self,
        bits: &mut crate::bit_io::BitReaderReversed<'_>,
        accuracy_log: u8,
    ) -> Result<(), FSEDecoderError> {
        if accuracy_log == 0 {
            return Err(FSEDecoderError::TableIsUninitialized);
        }
        let new_state = bits.get_bits(accuracy_log) as usize;
        self.state = new_state & self.mask;
        Ok(())
    }

    /// Read all decode parameters from the current state in one shot.
    /// Returns (base_value, extra_bits, next_state, state_bits).
    #[inline(always)]
    pub fn decode_params(&self) -> (u32, u8, u16, u8) {
        let entry = self.table[self.state];
        (
            entry.base_value,
            entry.extra_bits,
            entry.next_state,
            entry.state_bits,
        )
    }

    /// Apply a pre-read bit value to update state.
    #[inline(always)]
    pub fn apply_state_update(&mut self, next_state: u16, add: u64) {
        let new_state = next_state as usize + add as usize;
        self.state = new_state & self.mask;
    }
}

/// Build a pre-resolved SeqEntry table from an FSE decode table and a
/// code-to-(baseValue, extraBits) mapping.
///
/// `code_table` maps symbol code -> (base_value, extra_bits).
/// For LL this is LL_CODE_TABLE, for ML this is ML_CODE_TABLE.
/// For OF, base_value = (1 << code) and extra_bits = code.
pub fn build_seq_table(
    fse_table: &FSETable,
    code_table: &[(u32, u8)],
) -> alloc::vec::Vec<SeqEntry> {
    let mut seq = alloc::vec::Vec::with_capacity(fse_table.decode.len());
    for entry in &fse_table.decode {
        let (base_value, extra_bits) = if (entry.symbol as usize) < code_table.len() {
            code_table[entry.symbol as usize]
        } else {
            (0, 0)
        };
        seq.push(SeqEntry {
            base_value,
            extra_bits,
            next_state: entry.base_line as u16,
            state_bits: entry.num_bits,
        });
    }
    seq
}

/// Build a pre-resolved SeqEntry table for offset codes.
/// Offset codes have base_value = (1 << code) and extra_bits = code.
pub fn build_seq_table_offset(fse_table: &FSETable) -> alloc::vec::Vec<SeqEntry> {
    let mut seq = alloc::vec::Vec::with_capacity(fse_table.decode.len());
    for entry in &fse_table.decode {
        let code = entry.symbol;
        seq.push(SeqEntry {
            base_value: 1u32 << code,
            extra_bits: code,
            next_state: entry.base_line as u16,
            state_bits: entry.num_bits,
        });
    }
    seq
}

fn calc_baseline_and_numbits(
    num_states_total: u32,
    num_states_symbol: u32,
    state_number: u32,
) -> (u32, u8) {
    if num_states_symbol == 0 {
        return (0, 0);
    }
    let num_state_slices = if 1 << (highest_bit_set(num_states_symbol) - 1) == num_states_symbol {
        num_states_symbol
    } else {
        1 << (highest_bit_set(num_states_symbol))
    }; //always power of two

    let num_double_width_state_slices = num_state_slices - num_states_symbol; //leftovers to the power of two need to be distributed
    let num_single_width_state_slices = num_states_symbol - num_double_width_state_slices; //these will not receive a double width slice of states
    let slice_width = num_states_total / num_state_slices; //size of a single width slice of states
    let num_bits = highest_bit_set(slice_width) - 1; //number of bits needed to read for one slice

    if state_number < num_double_width_state_slices {
        let baseline = num_single_width_state_slices * slice_width + state_number * slice_width * 2;
        (baseline, num_bits as u8 + 1)
    } else {
        let index_shifted = state_number - num_double_width_state_slices;
        ((index_shifted * slice_width), num_bits as u8)
    }
}

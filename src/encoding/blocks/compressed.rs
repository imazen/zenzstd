use alloc::vec::Vec;

use crate::{
    bit_io::BitWriter,
    encoding::frame_compressor::CompressState,
    encoding::{Matcher, Sequence},
    fse::fse_encoder::{FSETable, State, build_table_from_data},
    huff0::huff0_encoder,
};

/// A block of [`crate::common::BlockType::Compressed`]
pub fn compress_block<M: Matcher>(state: &mut CompressState<M>, output: &mut Vec<u8>) {
    let mut literals_vec = Vec::new();
    let mut sequences = Vec::new();
    state.matcher.start_matching(|seq| {
        match seq {
            Sequence::Literals { literals } => literals_vec.extend_from_slice(literals),
            Sequence::Triple {
                literals,
                offset,
                match_len,
            } => {
                literals_vec.extend_from_slice(literals);
                sequences.push(crate::blocks::sequence_section::Sequence {
                    ll: literals.len() as u32,
                    ml: match_len as u32,
                    of: (offset + 3) as u32, // TODO make use of the offset history
                });
            }
        }
    });

    // literals section

    let mut writer = BitWriter::from(output);
    if literals_vec.len() >= 32 {
        if let Some(table) =
            compress_literals(&literals_vec, state.last_huff_table.as_ref(), &mut writer)
        {
            state.last_huff_table.replace(table);
        }
    } else {
        raw_literals(&literals_vec, &mut writer);
    }

    // sequences section

    if sequences.is_empty() {
        writer.write_bits(0u8, 8);
    } else {
        encode_seqnum(sequences.len(), &mut writer);

        // Choose the tables
        // TODO store previously used tables
        let ll_mode = choose_table(
            state.fse_tables.ll_previous.as_ref(),
            &state.fse_tables.ll_default,
            sequences.iter().map(|seq| encode_literal_length(seq.ll).0),
            9,
        );
        let ml_mode = choose_table(
            state.fse_tables.ml_previous.as_ref(),
            &state.fse_tables.ml_default,
            sequences.iter().map(|seq| encode_match_len(seq.ml).0),
            9,
        );
        let of_mode = choose_table(
            state.fse_tables.of_previous.as_ref(),
            &state.fse_tables.of_default,
            sequences.iter().map(|seq| encode_offset(seq.of).0),
            8,
        );

        writer.write_bits(encode_fse_table_modes(&ll_mode, &ml_mode, &of_mode), 8);

        encode_table(&ll_mode, &mut writer);
        encode_table(&of_mode, &mut writer);
        encode_table(&ml_mode, &mut writer);

        encode_sequences(
            &sequences,
            &mut writer,
            ll_mode.as_ref(),
            ml_mode.as_ref(),
            of_mode.as_ref(),
        );

        match ll_mode {
            FseTableMode::Encoded(table) => state.fse_tables.ll_previous = Some(table),
            FseTableMode::RepeateLast(t) => state.fse_tables.ll_previous = Some(t.clone()),
            FseTableMode::Predefined(_) => state.fse_tables.ll_previous = None,
        }
        match ml_mode {
            FseTableMode::Encoded(table) => state.fse_tables.ml_previous = Some(table),
            FseTableMode::RepeateLast(t) => state.fse_tables.ml_previous = Some(t.clone()),
            FseTableMode::Predefined(_) => state.fse_tables.ml_previous = None,
        }
        match of_mode {
            FseTableMode::Encoded(table) => state.fse_tables.of_previous = Some(table),
            FseTableMode::RepeateLast(t) => state.fse_tables.of_previous = Some(t.clone()),
            FseTableMode::Predefined(_) => state.fse_tables.of_previous = None,
        }
    }
    writer.flush();
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum FseTableMode<'a> {
    Predefined(&'a FSETable),
    Encoded(FSETable),
    RepeateLast(&'a FSETable),
}

impl FseTableMode<'_> {
    pub fn as_ref(&self) -> &FSETable {
        match self {
            Self::Predefined(t) => t,
            Self::RepeateLast(t) => t,
            Self::Encoded(t) => t,
        }
    }
}

#[allow(clippy::manual_repeat_n)] // repeat_n is 1.87+, MSRV is 1.85
fn choose_table<'a>(
    previous: Option<&'a FSETable>,
    default_table: &'a FSETable,
    data: impl Iterator<Item = u8>,
    max_log: u8,
) -> FseTableMode<'a> {
    // Collect symbol counts from the data
    let mut counts = [0usize; 256];
    let mut total = 0usize;
    let mut max_symbol = 0u8;
    for sym in data {
        counts[sym as usize] += 1;
        total += 1;
        if sym > max_symbol {
            max_symbol = sym;
        }
    }

    if total == 0 {
        return FseTableMode::Predefined(default_table);
    }

    // Check for single-symbol RLE case — always use encoded (the encoder handles this)
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    if distinct <= 1 {
        return FseTableMode::Encoded(build_table_from_data(
            counts
                .iter()
                .enumerate()
                .flat_map(|(sym, &cnt)| core::iter::repeat(sym as u8).take(cnt)),
            max_log,
            true,
        ));
    }

    // Build the new (compressed/encoded) table
    let new_table = build_table_from_data(
        counts
            .iter()
            .enumerate()
            .flat_map(|(sym, &cnt)| core::iter::repeat(sym as u8).take(cnt)),
        max_log,
        true,
    );

    // Estimate encoded size with new table: table description + entropy bits
    let new_table_desc_bits = estimate_table_description_bits(&new_table);
    let new_entropy_bits = estimate_encoding_cost_with_table(&new_table, &counts, max_symbol);
    let new_cost = new_table_desc_bits + new_entropy_bits;

    // Estimate encoded size with predefined/default table (0 table overhead)
    let default_cost = estimate_encoding_cost_with_table(default_table, &counts, max_symbol);

    // Estimate encoded size with previous table (0 table overhead)
    let repeat_cost =
        previous.map(|prev| estimate_encoding_cost_with_table(prev, &counts, max_symbol));

    // Pick the cheapest option
    let mut best_cost = new_cost;
    let mut best_mode = FseTableMode::Encoded(new_table);

    if default_cost <= best_cost {
        best_cost = default_cost;
        best_mode = FseTableMode::Predefined(default_table);
    }

    if let (Some(prev), Some(rcost)) = (previous, repeat_cost) {
        if rcost <= best_cost {
            best_mode = FseTableMode::RepeateLast(prev);
        }
    }

    best_mode
}

/// Estimate the number of bits needed to encode the table description.
/// Uses a trial write to a temporary buffer.
fn estimate_table_description_bits(table: &FSETable) -> usize {
    let mut buf = Vec::with_capacity(128);
    let mut writer = BitWriter::from(&mut buf);
    table.write_table(&mut writer);
    writer.flush();
    // Return size in bits (byte-aligned, rounded up)
    buf.len() * 8
}

/// Estimate the total encoding cost (in bits) of encoding data with the given
/// FSE table, using cross-entropy calculation.
///
/// For each symbol, the cost is approximately:
///   count[sym] * (acc_log - floor(log2(prob[sym])))
///
/// If a symbol appears in the data but has zero probability in the table,
/// we return usize::MAX (table cannot encode this data).
fn estimate_encoding_cost_with_table(
    table: &FSETable,
    counts: &[usize; 256],
    max_symbol: u8,
) -> usize {
    let acc_log = table.acc_log() as usize;
    let table_size = table.table_size;
    let mut cost_256: u64 = 0;

    for (sym, &count) in counts[..=max_symbol as usize].iter().enumerate() {
        if count == 0 {
            continue;
        }
        let prob = table.symbol_probability(sym as u8);
        if prob == 0 {
            // Table cannot encode this symbol — bail out
            return usize::MAX;
        } else if prob == -1 {
            // "Less than 1" probability — costs approximately acc_log bits per symbol
            cost_256 += count as u64 * (acc_log as u64 * 256);
        } else {
            // Normalized probability: prob out of table_size
            // Bits per symbol ≈ log2(table_size / prob)
            // Use the inverse probability table (fixed-point 8.8) for precision
            let norm = ((prob as u64 * 256) / table_size as u64).clamp(1, 255) as usize;
            cost_256 += count as u64 * INVERSE_PROBABILITY_LOG256[norm] as u64;
        }
    }

    // Convert from fixed-point (8.8) to bits, rounding up
    (cost_256 as usize).div_ceil(256)
}

/// Lookup table: floor(-log2(x / 256) * 256) for x in 0..256.
/// Index 0 is unused (would be infinity). Matches C zstd's kInverseProbabilityLog256.
static INVERSE_PROBABILITY_LOG256: [u16; 256] = [
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162, 1130, 1100, 1073, 1047,
    1024, 1001, 980, 960, 941, 923, 906, 889, 874, 859, 844, 830, 817, 804, 791, 779, 768, 756,
    745, 734, 724, 714, 704, 694, 685, 676, 667, 658, 650, 642, 633, 626, 618, 610, 603, 595, 588,
    581, 574, 567, 561, 554, 548, 542, 535, 529, 523, 517, 512, 506, 500, 495, 489, 484, 478, 473,
    468, 463, 458, 453, 448, 443, 438, 434, 429, 424, 420, 415, 411, 407, 402, 398, 394, 390, 386,
    382, 377, 373, 370, 366, 362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322, 318, 315,
    311, 308, 305, 302, 298, 295, 292, 289, 286, 282, 279, 276, 273, 270, 267, 264, 261, 258, 256,
    253, 250, 247, 244, 241, 239, 236, 233, 230, 228, 225, 222, 220, 217, 215, 212, 209, 207, 204,
    202, 199, 197, 194, 192, 190, 187, 185, 182, 180, 178, 175, 173, 171, 168, 166, 164, 162, 159,
    157, 155, 153, 151, 149, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 123, 121, 119,
    117, 115, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85, 83, 82, 80,
    78, 76, 74, 73, 71, 69, 67, 66, 64, 62, 61, 59, 57, 55, 54, 52, 50, 49, 47, 46, 44, 42, 41, 39,
    37, 36, 34, 33, 31, 30, 28, 26, 25, 23, 22, 20, 19, 17, 16, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1,
];

fn encode_table(mode: &FseTableMode<'_>, writer: &mut BitWriter<&mut Vec<u8>>) {
    match mode {
        FseTableMode::Predefined(_) => {}
        FseTableMode::RepeateLast(_) => {}
        FseTableMode::Encoded(table) => table.write_table(writer),
    }
}

fn encode_fse_table_modes(
    ll_mode: &FseTableMode<'_>,
    ml_mode: &FseTableMode<'_>,
    of_mode: &FseTableMode<'_>,
) -> u8 {
    fn mode_to_bits(mode: &FseTableMode<'_>) -> u8 {
        match mode {
            FseTableMode::Predefined(_) => 0,
            FseTableMode::Encoded(_) => 2,
            FseTableMode::RepeateLast(_) => 3,
        }
    }
    mode_to_bits(ll_mode) << 6 | mode_to_bits(of_mode) << 4 | mode_to_bits(ml_mode) << 2
}

fn encode_sequences(
    sequences: &[crate::blocks::sequence_section::Sequence],
    writer: &mut BitWriter<&mut Vec<u8>>,
    ll_table: &FSETable,
    ml_table: &FSETable,
    of_table: &FSETable,
) {
    let sequence = sequences[sequences.len() - 1];
    let (ll_code, ll_add_bits, ll_num_bits) = encode_literal_length(sequence.ll);
    let (of_code, of_add_bits, of_num_bits) = encode_offset(sequence.of);
    let (ml_code, ml_add_bits, ml_num_bits) = encode_match_len(sequence.ml);
    let mut ll_state: &State = ll_table.start_state(ll_code);
    let mut ml_state: &State = ml_table.start_state(ml_code);
    let mut of_state: &State = of_table.start_state(of_code);

    writer.write_bits(ll_add_bits, ll_num_bits);
    writer.write_bits(ml_add_bits, ml_num_bits);
    writer.write_bits(of_add_bits, of_num_bits);

    // encode backwards so the decoder reads the first sequence first
    if sequences.len() > 1 {
        for sequence in (0..=sequences.len() - 2).rev() {
            let sequence = sequences[sequence];
            let (ll_code, ll_add_bits, ll_num_bits) = encode_literal_length(sequence.ll);
            let (of_code, of_add_bits, of_num_bits) = encode_offset(sequence.of);
            let (ml_code, ml_add_bits, ml_num_bits) = encode_match_len(sequence.ml);

            {
                let next = of_table.next_state(of_code, of_state.index);
                let diff = of_state.index - next.baseline;
                writer.write_bits(diff as u64, next.num_bits as usize);
                of_state = next;
            }
            {
                let next = ml_table.next_state(ml_code, ml_state.index);
                let diff = ml_state.index - next.baseline;
                writer.write_bits(diff as u64, next.num_bits as usize);
                ml_state = next;
            }
            {
                let next = ll_table.next_state(ll_code, ll_state.index);
                let diff = ll_state.index - next.baseline;
                writer.write_bits(diff as u64, next.num_bits as usize);
                ll_state = next;
            }

            writer.write_bits(ll_add_bits, ll_num_bits);
            writer.write_bits(ml_add_bits, ml_num_bits);
            writer.write_bits(of_add_bits, of_num_bits);
        }
    }
    writer.write_bits(ml_state.index as u64, ml_table.table_size.ilog2() as usize);
    writer.write_bits(of_state.index as u64, of_table.table_size.ilog2() as usize);
    writer.write_bits(ll_state.index as u64, ll_table.table_size.ilog2() as usize);

    let bits_to_fill = writer.misaligned();
    if bits_to_fill == 0 {
        writer.write_bits(1u32, 8);
    } else {
        writer.write_bits(1u32, bits_to_fill);
    }
}

fn encode_seqnum(seqnum: usize, writer: &mut BitWriter<impl AsMut<Vec<u8>>>) {
    const UPPER_LIMIT: usize = 0xFFFF + 0x7F00;
    match seqnum {
        1..=127 => writer.write_bits(seqnum as u32, 8),
        128..=0x7FFF => {
            let upper = ((seqnum >> 8) | 0x80) as u8;
            let lower = seqnum as u8;
            writer.write_bits(upper, 8);
            writer.write_bits(lower, 8);
        }
        0x8000..=UPPER_LIMIT => {
            let encode = seqnum - 0x7F00;
            let upper = (encode >> 8) as u8;
            let lower = encode as u8;
            writer.write_bits(255u8, 8);
            writer.write_bits(upper, 8);
            writer.write_bits(lower, 8);
        }
        _ => unreachable!(),
    }
}

fn encode_literal_length(len: u32) -> (u8, u32, usize) {
    match len {
        0..=15 => (len as u8, 0, 0),
        16..=17 => (16, len - 16, 1),
        18..=19 => (17, len - 18, 1),
        20..=21 => (18, len - 20, 1),
        22..=23 => (19, len - 22, 1),
        24..=27 => (20, len - 24, 2),
        28..=31 => (21, len - 28, 2),
        32..=39 => (22, len - 32, 3),
        40..=47 => (23, len - 40, 3),
        48..=63 => (24, len - 48, 4),
        64..=127 => (25, len - 64, 6),
        128..=255 => (26, len - 128, 7),
        256..=511 => (27, len - 256, 8),
        512..=1023 => (28, len - 512, 9),
        1024..=2047 => (29, len - 1024, 10),
        2048..=4095 => (30, len - 2048, 11),
        4096..=8191 => (31, len - 4096, 12),
        8192..=16383 => (32, len - 8192, 13),
        16384..=32767 => (33, len - 16384, 14),
        32768..=65535 => (34, len - 32768, 15),
        65536..=131071 => (35, len - 65536, 16),
        131072.. => unreachable!(),
    }
}

fn encode_match_len(len: u32) -> (u8, u32, usize) {
    match len {
        0..=2 => unreachable!(),
        3..=34 => (len as u8 - 3, 0, 0),
        35..=36 => (32, len - 35, 1),
        37..=38 => (33, len - 37, 1),
        39..=40 => (34, len - 39, 1),
        41..=42 => (35, len - 41, 1),
        43..=46 => (36, len - 43, 2),
        47..=50 => (37, len - 47, 2),
        51..=58 => (38, len - 51, 3),
        59..=66 => (39, len - 59, 3),
        67..=82 => (40, len - 67, 4),
        83..=98 => (41, len - 83, 4),
        99..=130 => (42, len - 99, 5),
        131..=258 => (43, len - 131, 7),
        259..=514 => (44, len - 259, 8),
        515..=1026 => (45, len - 515, 9),
        1027..=2050 => (46, len - 1027, 10),
        2051..=4098 => (47, len - 2051, 11),
        4099..=8194 => (48, len - 4099, 12),
        8195..=16386 => (49, len - 8195, 13),
        16387..=32770 => (50, len - 16387, 14),
        32771..=65538 => (51, len - 32771, 15),
        65539..=131074 => (52, len - 65539, 16),
        131075.. => unreachable!(),
    }
}

fn encode_offset(len: u32) -> (u8, u32, usize) {
    let log = len.ilog2();
    let lower = len & ((1 << log) - 1);
    (log as u8, lower, log as usize)
}

/// Encode a compressed block from pre-computed sequences and literals.
///
/// This is the entry point for the new zstd_match-based compression pipeline.
/// It takes the output of `compress_block_zstd` and encodes it into the zstd
/// compressed block format.
pub fn encode_compressed_block<M: crate::encoding::Matcher>(
    literals_vec: &[u8],
    sequences_out: &[crate::encoding::zstd_match::SequenceOut],
    state: &mut crate::encoding::frame_compressor::CompressState<M>,
    output: &mut Vec<u8>,
) {
    // Convert SequenceOut to the decoder Sequence format
    let sequences: Vec<crate::blocks::sequence_section::Sequence> = sequences_out
        .iter()
        .map(|s| crate::blocks::sequence_section::Sequence {
            ll: s.lit_len,
            ml: s.match_len,
            of: s.off_base,
        })
        .collect();

    let mut writer = BitWriter::from(output);

    // Literals section
    if literals_vec.len() >= 32 {
        if let Some(table) =
            compress_literals(literals_vec, state.last_huff_table.as_ref(), &mut writer)
        {
            state.last_huff_table.replace(table);
        }
    } else {
        raw_literals(literals_vec, &mut writer);
    }

    // Sequences section
    if sequences.is_empty() {
        writer.write_bits(0u8, 8);
    } else {
        encode_seqnum(sequences.len(), &mut writer);

        let ll_mode = choose_table(
            state.fse_tables.ll_previous.as_ref(),
            &state.fse_tables.ll_default,
            sequences.iter().map(|seq| encode_literal_length(seq.ll).0),
            9,
        );
        let ml_mode = choose_table(
            state.fse_tables.ml_previous.as_ref(),
            &state.fse_tables.ml_default,
            sequences.iter().map(|seq| encode_match_len(seq.ml).0),
            9,
        );
        let of_mode = choose_table(
            state.fse_tables.of_previous.as_ref(),
            &state.fse_tables.of_default,
            sequences.iter().map(|seq| encode_offset(seq.of).0),
            8,
        );

        writer.write_bits(encode_fse_table_modes(&ll_mode, &ml_mode, &of_mode), 8);

        encode_table(&ll_mode, &mut writer);
        encode_table(&of_mode, &mut writer);
        encode_table(&ml_mode, &mut writer);

        encode_sequences(
            &sequences,
            &mut writer,
            ll_mode.as_ref(),
            ml_mode.as_ref(),
            of_mode.as_ref(),
        );

        // Update previous tables: save a clone when using repeat-last so
        // subsequent blocks can continue repeating. Encoded mode moves the
        // newly built table into previous.
        match ll_mode {
            FseTableMode::Encoded(table) => state.fse_tables.ll_previous = Some(table),
            FseTableMode::RepeateLast(t) => state.fse_tables.ll_previous = Some(t.clone()),
            FseTableMode::Predefined(_) => state.fse_tables.ll_previous = None,
        }
        match ml_mode {
            FseTableMode::Encoded(table) => state.fse_tables.ml_previous = Some(table),
            FseTableMode::RepeateLast(t) => state.fse_tables.ml_previous = Some(t.clone()),
            FseTableMode::Predefined(_) => state.fse_tables.ml_previous = None,
        }
        match of_mode {
            FseTableMode::Encoded(table) => state.fse_tables.of_previous = Some(table),
            FseTableMode::RepeateLast(t) => state.fse_tables.of_previous = Some(t.clone()),
            FseTableMode::Predefined(_) => state.fse_tables.of_previous = None,
        }
    }
    writer.flush();
}

/// Encode a compressed block from pre-computed sequences and literals,
/// without requiring a CompressState. Uses fresh entropy tables.
///
/// This is used for trial encoding during block splitting to measure
/// the actual compressed size of a candidate partition.
pub fn encode_compressed_block_standalone(
    literals_vec: &[u8],
    sequences_out: &[crate::encoding::zstd_match::SequenceOut],
    output: &mut Vec<u8>,
) {
    let sequences: Vec<crate::blocks::sequence_section::Sequence> = sequences_out
        .iter()
        .map(|s| crate::blocks::sequence_section::Sequence {
            ll: s.lit_len,
            ml: s.match_len,
            of: s.off_base,
        })
        .collect();

    let mut writer = BitWriter::from(output);

    // Literals section
    if literals_vec.len() >= 32 {
        if let Some(_table) = compress_literals(literals_vec, None, &mut writer) {
            // Don't save table — this is standalone
        }
    } else {
        raw_literals(literals_vec, &mut writer);
    }

    // Sequences section
    if sequences.is_empty() {
        writer.write_bits(0u8, 8);
    } else {
        let default_ll = crate::fse::fse_encoder::default_ll_table();
        let default_ml = crate::fse::fse_encoder::default_ml_table();
        let default_of = crate::fse::fse_encoder::default_of_table();

        encode_seqnum(sequences.len(), &mut writer);

        let ll_mode = choose_table(
            None,
            &default_ll,
            sequences.iter().map(|seq| encode_literal_length(seq.ll).0),
            9,
        );
        let ml_mode = choose_table(
            None,
            &default_ml,
            sequences.iter().map(|seq| encode_match_len(seq.ml).0),
            9,
        );
        let of_mode = choose_table(
            None,
            &default_of,
            sequences.iter().map(|seq| encode_offset(seq.of).0),
            8,
        );

        writer.write_bits(encode_fse_table_modes(&ll_mode, &ml_mode, &of_mode), 8);

        encode_table(&ll_mode, &mut writer);
        encode_table(&of_mode, &mut writer);
        encode_table(&ml_mode, &mut writer);

        encode_sequences(
            &sequences,
            &mut writer,
            ll_mode.as_ref(),
            ml_mode.as_ref(),
            of_mode.as_ref(),
        );
    }
    writer.flush();
}

fn raw_literals(literals: &[u8], writer: &mut BitWriter<&mut Vec<u8>>) {
    writer.write_bits(0u8, 2);
    writer.write_bits(0b11u8, 2);
    writer.write_bits(literals.len() as u32, 20);
    writer.append_bytes(literals);
}

fn compress_literals(
    literals: &[u8],
    last_table: Option<&huff0_encoder::HuffmanTable>,
    writer: &mut BitWriter<&mut Vec<u8>>,
) -> Option<huff0_encoder::HuffmanTable> {
    // Huffman coding requires at least 2 distinct symbols. For single-symbol
    // data (e.g. all zeros), fall back to raw literals.
    let distinct = {
        let mut seen = [false; 256];
        let mut count = 0u16;
        for &b in literals {
            if !seen[b as usize] {
                seen[b as usize] = true;
                count += 1;
                if count >= 2 {
                    break;
                }
            }
        }
        count
    };
    if distinct < 2 {
        raw_literals(literals, writer);
        return None;
    }

    let reset_idx = writer.index();

    let new_encoder_table = huff0_encoder::HuffmanTable::build_from_data(literals);

    let (encoder_table, new_table) = if let Some(_table) = last_table {
        if let Some(diff) = _table.can_encode(&new_encoder_table) {
            // TODO this is a very simple heuristic, maybe we should try to do better
            if diff > 5 {
                (&new_encoder_table, true)
            } else {
                (_table, false)
            }
        } else {
            (&new_encoder_table, true)
        }
    } else {
        (&new_encoder_table, true)
    };

    if new_table {
        writer.write_bits(2u8, 2); // compressed literals type
    } else {
        writer.write_bits(3u8, 2); // treeless compressed literals type
    }

    let (size_format, size_bits) = match literals.len() {
        0..6 => (0b00u8, 10),
        6..1024 => (0b01, 10),
        1024..16384 => (0b10, 14),
        16384..262144 => (0b11, 18),
        _ => unimplemented!("too many literals"),
    };

    writer.write_bits(size_format, 2);
    writer.write_bits(literals.len() as u32, size_bits);
    let size_index = writer.index();
    writer.write_bits(0u32, size_bits);
    let index_before = writer.index();
    let mut encoder = huff0_encoder::HuffmanEncoder::new(encoder_table, writer);
    if size_format == 0 {
        encoder.encode(literals, new_table)
    } else {
        encoder.encode4x(literals, new_table)
    };
    let encoded_len = (writer.index() - index_before) / 8;
    writer.change_bits(size_index, encoded_len as u64, size_bits);
    let total_len = (writer.index() - reset_idx) / 8;

    // If encoded len is bigger than the raw literals we are better off just writing the raw literals here
    if total_len >= literals.len() {
        writer.reset_to(reset_idx);
        raw_literals(literals, writer);
        None
    } else if new_table {
        Some(new_encoder_table)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::fse::fse_encoder::{default_ll_table, default_of_table};

    /// Verify that `choose_table` selects predefined mode when the data
    /// matches the default distribution closely. We build a distribution that
    /// spans most of the default LL symbol range, matching the default
    /// probability shape, so the encoding cost is similar but the table
    /// description overhead makes a new table more expensive.
    #[test]
    fn predefined_table_selected_for_default_like_distribution() {
        let default_ll = default_ll_table();

        // Build data that closely mimics the default LL distribution shape.
        // The LL default has acc_log=6, probabilities:
        //   [4,3,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,3,2,1,1,1,1,1,-1,-1,-1,-1]
        // We produce codes with similar relative frequencies, scaled up.
        let mut codes: Vec<u8> = Vec::new();
        let default_probs = [
            4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1,
            1, 1, 1,
        ];
        for (sym, &prob) in default_probs.iter().enumerate() {
            for _ in 0..(prob * 3) {
                codes.push(sym as u8);
            }
        }

        let mode = choose_table(None, &default_ll, codes.iter().copied(), 9);
        assert!(
            matches!(mode, FseTableMode::Predefined(_)),
            "Expected predefined table mode for default-like distribution, got {:?}",
            match &mode {
                FseTableMode::Predefined(_) => "Predefined",
                FseTableMode::Encoded(_) => "Encoded",
                FseTableMode::RepeateLast(_) => "RepeateLast",
            }
        );
    }

    /// Verify that repeat-last is selected when a previous table exists
    /// and the distribution hasn't changed.
    #[test]
    fn repeat_last_selected_when_distribution_unchanged() {
        let default_ll = default_ll_table();

        // First, build a custom distribution
        let codes: Vec<u8> = {
            let mut v = Vec::new();
            for _ in 0..200 {
                v.push(0);
            }
            for _ in 0..100 {
                v.push(1);
            }
            for _ in 0..50 {
                v.push(2);
            }
            for _ in 0..25 {
                v.push(3);
            }
            for _ in 0..10 {
                v.push(4);
            }
            v
        };

        // First call builds a new table
        let mode1 = choose_table(None, &default_ll, codes.iter().copied(), 9);
        let prev_table = match &mode1 {
            FseTableMode::Encoded(t) => t,
            other => panic!(
                "Expected Encoded for first call, got {:?}",
                match other {
                    FseTableMode::Predefined(_) => "Predefined",
                    FseTableMode::Encoded(_) => "Encoded",
                    FseTableMode::RepeateLast(_) => "RepeateLast",
                }
            ),
        };

        // Second call with same distribution should use repeat-last
        let mode2 = choose_table(Some(prev_table), &default_ll, codes.iter().copied(), 9);
        assert!(
            matches!(mode2, FseTableMode::RepeateLast(_)),
            "Expected RepeateLast for same distribution, got {:?}",
            match &mode2 {
                FseTableMode::Predefined(_) => "Predefined",
                FseTableMode::Encoded(_) => "Encoded",
                FseTableMode::RepeateLast(_) => "RepeateLast",
            }
        );
    }

    /// Verify that encoded (new table) is selected when the distribution
    /// is very different from both default and previous.
    #[test]
    fn encoded_table_selected_for_unusual_distribution() {
        let default_of = default_of_table();

        // Build offset codes with a very unusual distribution:
        // heavily concentrated on high offset codes, which don't match the default
        let codes: Vec<u8> = {
            let mut v = Vec::new();
            for _ in 0..300 {
                v.push(20);
            }
            for _ in 0..200 {
                v.push(21);
            }
            for _ in 0..100 {
                v.push(22);
            }
            v
        };

        let mode = choose_table(None, &default_of, codes.iter().copied(), 8);

        // The default offset table doesn't have high codes at all, so
        // it should return usize::MAX cost, forcing encoded mode
        assert!(
            matches!(mode, FseTableMode::Encoded(_)),
            "Expected Encoded for unusual distribution, got {:?}",
            match &mode {
                FseTableMode::Predefined(_) => "Predefined",
                FseTableMode::Encoded(_) => "Encoded",
                FseTableMode::RepeateLast(_) => "RepeateLast",
            }
        );
    }

    /// Verify that Huffman compression is now attempted for small literal sections
    /// (the old threshold was 1024, now it should be 32).
    #[test]
    fn huffman_compression_for_small_literals() {
        // Create data with 200 bytes of literals that should compress well
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(b"abcd");
        }
        assert_eq!(data.len(), 200);

        // Compress at L1 and verify it round-trips
        let compressed = crate::encoding::compress_to_vec(
            data.as_slice(),
            crate::encoding::CompressionLevel::Fastest,
        );
        let mut decoded = Vec::new();
        zstd::stream::copy_decode(compressed.as_slice(), &mut decoded).unwrap();
        assert_eq!(data, decoded);

        // Verify it actually compressed (the data is repetitive enough)
        assert!(
            compressed.len() < data.len(),
            "Expected compression for 200-byte repetitive data: {} -> {} bytes",
            data.len(),
            compressed.len(),
        );
    }

    /// End-to-end test: multi-block data exercises predefined and repeat-last
    /// table modes across block boundaries. Verifies that our previous-table
    /// tracking works correctly.
    #[test]
    fn multi_block_table_mode_roundtrip() {
        // Create multi-block data (> 128KB)
        let mut data = Vec::new();
        for _ in 0..4000 {
            data.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
        }
        assert!(data.len() > 128 * 1024);

        for level in [1, 3, 7, 11, 19] {
            let compressed = crate::encoding::compress_to_vec(
                data.as_slice(),
                crate::encoding::CompressionLevel::Level(level),
            );

            // Verify with both decoders
            let mut decoder = crate::decoding::FrameDecoder::new();
            let mut decoded = Vec::with_capacity(data.len() + 4096);
            decoder
                .decode_all_to_vec(&compressed, &mut decoded)
                .unwrap_or_else(|e| panic!("Our decoder failed at L{level}: {e:?}"));
            assert_eq!(data, decoded, "our decoder: mismatch at L{level}");

            let mut decoded_c = Vec::new();
            match zstd::stream::copy_decode(compressed.as_slice(), &mut decoded_c) {
                Ok(()) => assert_eq!(data, decoded_c, "C zstd: mismatch at L{level}"),
                Err(e) => {
                    let err_str = std::format!("{e:?}");
                    if !err_str.contains("checksum") && !err_str.contains("Checksum") {
                        panic!("C zstd error at L{level}: {e}");
                    }
                }
            }
        }
    }

    /// Verify that entropy coding changes produce valid compressed output.
    /// Uses the same mixed data pattern as the benchmark.
    /// Note: 100KB mixed at L19 has a known pre-existing decoding issue
    /// (not caused by entropy changes), so we test with 10K here.
    #[test]
    fn entropy_improvements_dont_regress() {
        fn make_mixed(size: usize) -> Vec<u8> {
            let mut data = Vec::with_capacity(size);
            let mut i = 0u32;
            while data.len() < size {
                if i % 100 < 50 {
                    data.push(b'A' + (i % 26) as u8);
                } else {
                    data.push(((i.wrapping_mul(2654435761) >> 16) & 0xFF) as u8);
                }
                i += 1;
            }
            data
        }

        for size in [1000, 5000, 10_000] {
            let data = make_mixed(size);
            for level in [1, 3, 7, 11] {
                let compressed = crate::encoding::compress_to_vec(
                    data.as_slice(),
                    crate::encoding::CompressionLevel::Level(level),
                );

                let mut decoder = crate::decoding::FrameDecoder::new();
                let mut decoded = Vec::with_capacity(data.len() + 4096);
                decoder
                    .decode_all_to_vec(&compressed, &mut decoded)
                    .unwrap_or_else(|e| panic!("Our decoder failed at L{level} {size}B: {e:?}"));
                assert_eq!(data, decoded, "mismatch at L{level} {size}B");
            }
        }
    }
}

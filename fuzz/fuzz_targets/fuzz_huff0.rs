#![no_main]

use libfuzzer_sys::fuzz_target;
use zenzstd::huff0::round_trip;

// Feed arbitrary data through the Huffman encoder/decoder roundtrip.
// Must never panic.
fuzz_target!(|data: &[u8]| {
    round_trip(data);
});

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenzstd::fse::round_trip;

// Feed arbitrary data through the FSE encoder/decoder roundtrip.
// Must never panic.
fuzz_target!(|data: &[u8]| {
    round_trip(data);
});

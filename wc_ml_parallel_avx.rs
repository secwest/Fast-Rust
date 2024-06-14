// wc -mw optimized in Rust using Rayon for parallelization, and AVX SIMD optimization on each core if AVX2 or AVX512 are available.
// please note: at this time using avx2 requires using the nightly rustc tool chain: 
//                                                    rustup install nightly
//                                                    rustup default nightly
//                                                    rustup component add rust-src --toolchain nightly
//
// Build with: RUSTFLAGS="-C target-cpu=native"  cargo build --release
// Use lscpu to verify presence of avx2/avx512.
//
//Cargo.toml:
//
//[package]
//name = "wc_mw_parallel_avx"
//version = "0.1.0"
//edition = "2021"
//
//[[bin]]
//name = "wc_mw_parallel_avx"
//
//[dependencies]
//memmap = "0.7"
//rayon = "1.5"
//num_cpus = "1.16.0"

// (C) Copyright 2024 Dragos Ruiu
//
// Dedicated to Rob "@ErrataRob" Graham

// Import necessary crates and modules

eextern crate memmap;
extern crate rayon;
extern crate num_cpus;

use memmap::Mmap;
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::{
    _mm512_loadu_si512, _mm512_set1_epi8, _mm512_set1_epi16, 
    _mm512_cmpeq_epi8_mask, _mm512_cmpeq_epi16_mask, _mm512_movemask_epi8, _mm512_movemask_epi16, __m512i
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    _mm256_loadu_si256, _mm256_set1_epi8, _mm256_set1_epi16, 
    _mm256_cmpeq_epi8, _mm256_cmpeq_epi16, _mm256_movemask_epi8, __m256i
};

// 8-bit patterns to match against for ASCII whitespace characters
const ASCII_WHITESPACE_PATTERNS: [u8; 6] = [
    0x09, // Tab (U+0009)
    0x0A, // Line Feed (U+000A)
    0x0B, // Vertical Tab (U+000B)
    0x0C, // Form Feed (U+000C)
    0x0D, // Carriage Return (U+000D)
    0x20, // Space (U+0020)
];

// 16-bit patterns to match against for Unicode whitespace characters
const UNICODE_WHITESPACE_PATTERNS: [u16; 17] = [
    0x00A0, // Non-breaking Space (U+00A0)
    0x1680, // Ogham Space Mark (U+1680)
    0x180E, // Mongolian Vowel Separator (U+180E)
    0x2000, // En Quad (U+2000)
    0x2001, // Em Quad (U+2001)
    0x2002, // En Space (U+2002)
    0x2003, // Em Space (U+2003)
    0x2004, // Three-per-em Space (U+2004)
    0x2005, // Four-per-em Space (U+2005)
    0x2006, // Six-per-em Space (U+2006)
    0x2007, // Figure Space (U+2007)
    0x2008, // Punctuation Space (U+2008)
    0x2009, // Thin Space (U+2009)
    0x200A, // Hair Space (U+200A)
    0x2028, // Line Separator (U+2028)
    0x2029, // Paragraph Separator (U+2029)
    0x205F, // Medium Mathematical Space (U+205F)
    0x3000  // Ideographic Space (U+3000)
];

#[derive(Default)]
struct ChunkResult {
    ascii_count: usize,
    two_byte_count: usize,
    three_byte_count: usize,
    four_byte_count: usize,
    ascii_whitespace_count: usize,
    unicode_whitespace_count: usize,
    word_count: usize,
    ending_in_word: bool,
    ending_in_utf32: bool,
    ending_in_utf16: bool,
    ending_in_utf8: bool,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe fn count_patterns_avx512_chunk(chunk: &[u8]) -> ChunkResult {
    let mut result = ChunkResult::default();

    // Create SIMD patterns for leading bytes of UTF-8 sequences
    let two_byte_utf_mask = _mm512_set1_epi8(0xC0 as i8);  // 110xxxxx
    let three_byte_utf_mask = _mm512_set1_epi8(0xE0 as i8); // 1110xxxx
    let four_byte_utf_mask = _mm512_set1_epi8(0xF0 as i8);  // 11110xxx

    // Create an array and populate it with the repeated ASCII whitespace patterns
    let mut ascii_pattern_array = [0u8; 64];
    for i in 0..64 {
        ascii_pattern_array[i] = ASCII_WHITESPACE_PATTERNS[i % ASCII_WHITESPACE_PATTERNS.len()];
    }

    // Load the array into an AVX-512 register
    let ascii_whitespace_patterns = _mm512_loadu_si512(ascii_pattern_array.as_ptr() as *const __m512i);

    // Create an array and populate it with the repeated Unicode whitespace patterns
    let mut unicode_pattern_array = [0u16; 32];
    for i in 0..32 {
        unicode_pattern_array[i] = UNICODE_WHITESPACE_PATTERNS[i % UNICODE_WHITESPACE_PATTERNS.len()];
    }

    // Load the array into an AVX-512 register
    let unicode_whitespace_patterns = _mm512_loadu_si512(unicode_pattern_array.as_ptr() as *const __m512i);

    // Load the chunk into an AVX-512 register
    let mut chunk_data = [0u8; 64];
    chunk_data[..chunk.len()].copy_from_slice(chunk);
    let chunk_data = _mm512_loadu_si512(chunk_data.as_ptr() as *const __m512i);

    // Initialize exclusion masks
    let mut exclusion_mask = 0u64;
    let mut whitespace_mask = 0u64;

    // Perform all comparisons simultaneously
    let is_two_byte_utf = _mm512_cmpeq_epi8_mask(_mm512_and_si512(chunk_data, two_byte_utf_mask), two_byte_utf_mask);
    let is_three_byte_utf = _mm512_cmpeq_epi8_mask(_mm512_and_si512(chunk_data, three_byte_utf_mask), three_byte_utf_mask);
    let is_four_byte_utf = _mm512_cmpeq_epi8_mask(_mm512_and_si512(chunk_data, four_byte_utf_mask), four_byte_utf_mask);

    // Combine the results into exclusion masks
    let is_two_byte_utf_mask = _mm512_movemask_epi8(is_two_byte_utf) as u64;
    let is_three_byte_utf_mask = _mm512_movemask_epi8(is_three_byte_utf) as u64;
    let is_four_byte_utf_mask = _mm512_movemask_epi8(is_four_byte_utf) as u64;

    exclusion_mask |= is_four_byte_utf_mask | (is_four_byte_utf_mask << 1) | (is_four_byte_utf_mask << 2) | (is_four_byte_utf_mask << 3);
    exclusion_mask |= is_three_byte_utf_mask | (is_three_byte_utf_mask << 1) | (is_three_byte_utf_mask << 2);

    // Identify and mask out Unicode whitespace
    let unicode_whitespace_cmp = _mm512_cmpeq_epi16_mask(chunk_data, unicode_whitespace_patterns);
    let unicode_whitespace_mask = _mm512_movemask_epi16(unicode_whitespace_cmp) as u64;
    whitespace_mask |= unicode_whitespace_mask & !exclusion_mask;

    exclusion_mask |= is_two_byte_utf_mask | (is_two_byte_utf_mask << 1);

    // Identify and count ASCII whitespace
    let ascii_whitespace_cmp = _mm512_cmpeq_epi8_mask(chunk_data, ascii_whitespace_patterns);
    let ascii_whitespace_mask = _mm512_movemask_epi8(ascii_whitespace_cmp) as u64;
    whitespace_mask |= ascii_whitespace_mask & !exclusion_mask;

    // Use the masks to count words and character types
    let mut in_whitespace = true;
    let mut j = 0;

    while j < chunk.len() {
        let bit = 1 << j;

        if (is_four_byte_utf_mask & bit) != 0 {
            if in_whitespace {
                result.word_count += 1;
                in_whitespace = false;
            }

            result.four_byte_count += 1;
            if j >= chunk.len() - 4 {
                result.ending_in_utf32 = true;
                break;
            }
            j += 4;
            continue;
        }

        if (is_three_byte_utf_mask & bit) != 0 {
            if in_whitespace {
                result.word_count += 1;
                in_whitespace = false;
            }

            result.three_byte_count += 1;
            if j >= chunk.len() - 3 {
                result.ending_in_utf16 = true;
                break;
            }
            j += 3;
            continue;
        }

        if (whitespace_mask & bit) != 0 {
            if !in_whitespace {
                in_whitespace = true;
                if (ascii_whitespace_mask & bit) != 0 {
                    result.ascii_whitespace_count += 1;
                } else {
                    result.unicode_whitespace_count += 1;
                    if j >= chunk.len() - 1 {
                        result.ending_in_utf16 = true;
                        break;
                    }
                    j += 2;
                    continue;
                }
            }
        } else {
            if in_whitespace {
                result.word_count += 1;
                in_whitespace = false;
            }

            if (exclusion_mask & bit) == 0 {
                result.ascii_count += 1;
            } else if (is_two_byte_utf_mask & bit) != 0 {
                result.two_byte_count += 1;
                if j >= chunk.len() - 2 {
                    result.ending_in_utf8 = true;
                    break;
                }
                j += 2;
                continue;
            }
        }

        j += 1;
    }

    if !in_whitespace {
        result.ending_in_word = true;
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn count_patterns_avx2_chunk(chunk: &[u8]) -> ChunkResult {
    let mut result = ChunkResult::default();

    // Create SIMD patterns for leading bytes of UTF-8 sequences
    let two_byte_utf_mask = _mm256_set1_epi8(0xC0 as i8);  // 110xxxxx
    let three_byte_utf_mask = _mm256_set1_epi8(0xE0 as i8); // 1110xxxx
    let four_byte_utf_mask = _mm256_set1_epi8(0xF0 as i8);  // 11110xxx

    // Create an array and populate it with the repeated ASCII whitespace patterns
    let mut ascii_pattern_array = [0u8; 32];
    for i in 0..32 {
        ascii_pattern_array[i] = ASCII_WHITESPACE_PATTERNS[i % ASCII_WHITESPACE_PATTERNS.len()];
    }

    // Load the array into an AVX2 register
    let ascii_whitespace_patterns = _mm256_loadu_si256(ascii_pattern_array.as_ptr() as *const __m256i);

    // Create an array and populate it with the first 16 Unicode whitespace patterns
    let mut unicode_pattern_array1 = [0u16; 16];
    for i in 0..16 {
        unicode_pattern_array1[i] = UNICODE_WHITESPACE_PATTERNS[i];
    }

    // Load the array into an AVX2 register
    let unicode_whitespace_patterns1 = _mm256_loadu_si256(unicode_pattern_array1.as_ptr() as *const __m256i);

    // Load the 17th Unicode whitespace pattern into an AVX2 register, repeated
    let unicode_whitespace_patterns2 = _mm256_set1_epi16(UNICODE_WHITESPACE_PATTERNS[16] as i16);

    // Load the chunk into an AVX2 register
    let mut chunk_data = [0u8; 32];
    chunk_data[..chunk.len()].copy_from_slice(chunk);
    let chunk_data = _mm256_loadu_si256(chunk_data.as_ptr() as *const __m256i);

    // Initialize exclusion masks
    let mut exclusion_mask = 0u32;
    let mut whitespace_mask = 0u32;

    // Perform all comparisons simultaneously
    let is_two_byte_utf = _mm256_cmpeq_epi8(_mm256_and_si256(chunk_data, two_byte_utf_mask), two_byte_utf_mask);
    let is_three_byte_utf = _mm256_cmpeq_epi8(_mm256_and_si256(chunk_data, three_byte_utf_mask), three_byte_utf_mask);
    let is_four_byte_utf = _mm256_cmpeq_epi8(_mm256_and_si256(chunk_data, four_byte_utf_mask), four_byte_utf_mask);

    // Combine the results into exclusion masks
    let is_two_byte_utf_mask = _mm256_movemask_epi8(is_two_byte_utf) as u32;
    let is_three_byte_utf_mask = _mm256_movemask_epi8(is_three_byte_utf) as u32;
    let is_four_byte_utf_mask = _mm256_movemask_epi8(is_four_byte_utf) as u32;

    exclusion_mask |= is_four_byte_utf_mask | (is_four_byte_utf_mask << 1) | (is_four_byte_utf_mask << 2) | (is_four_byte_utf_mask << 3);
    exclusion_mask |= is_three_byte_utf_mask | (is_three_byte_utf_mask << 1) | (is_three_byte_utf_mask << 2);

    // Identify and mask out Unicode whitespace
    let unicode_whitespace_cmp1 = _mm256_cmpeq_epi16(chunk_data, unicode_whitespace_patterns1);
    let unicode_whitespace_cmp2 = _mm256_cmpeq_epi16(chunk_data, unicode_whitespace_patterns2);
    let unicode_whitespace_mask1 = _mm256_movemask_epi16(unicode_whitespace_cmp1) as u32;
    let unicode_whitespace_mask2 = _mm256_movemask_epi16(unicode_whitespace_cmp2) as u32;
    let unicode_whitespace_mask = unicode_whitespace_mask1 | (unicode_whitespace_mask2 << 16);
    whitespace_mask |= unicode_whitespace_mask & !exclusion_mask;

    exclusion_mask |= is_two_byte_utf_mask | (is_two_byte_utf_mask << 1);

    // Identify and count ASCII whitespace
    let ascii_whitespace_cmp = _mm256_cmpeq_epi8(chunk_data, ascii_whitespace_patterns);
    let ascii_whitespace_mask = _mm256_movemask_epi8(ascii_whitespace_cmp) as u32;
    whitespace_mask |= ascii_whitespace_mask & !exclusion_mask;

   unsafe fn count_patterns_fallback_chunk(chunk: &[u8]) -> ChunkResult {
    let mut result = ChunkResult::default();

    let mut i = 0;
    while i < chunk.len() {
        let byte = chunk[i];

        // Check for ASCII whitespace
        if ASCII_WHITESPACE_PATTERNS.contains(&byte) {
            if !result.ending_in_word {
                result.ending_in_word = true;
                result.ascii_whitespace_count += 1;
            }
        } else if byte & 0x80 == 0 { // Check for ASCII characters
            if result.ending_in_word {
                result.word_count += 1;
                result.ending_in_word = false;
            }
            result.ascii_count += 1;
        } else if byte & 0xE0 == 0xC0 { // Check for 2-byte UTF-8 characters
            if i + 1 < chunk.len() {
                if result.ending_in_word {
                    result.word_count += 1;
                    result.ending_in_word = false;
                }
                result.two_byte_count += 1;
                i += 1;
            } else {
                result.ending_in_utf8 = true;
                break;
            }
        } else if byte & 0xF0 == 0xE0 { // Check for 3-byte UTF-8 characters
            if i + 2 < chunk.len() {
                if result.ending_in_word {
                    result.word_count += 1;
                    result.ending_in_word = false;
                }
                result.three_byte_count += 1;
                i += 2;
            } else {
                result.ending_in_utf16 = true;
                break;
            }
        } else if byte & 0xF8 == 0xF0 { // Check for 4-byte UTF-8 characters
            if i + 3 < chunk.len() {
                if result.ending_in_word {
                    result.word_count += 1;
                    result.ending_in_word = false;
                }
                result.four_byte_count += 1;
                i += 3;
            } else {
                result.ending_in_utf32 = true;
                break;
            }
        } else if let Ok(unicode_char) = std::str::from_utf8(&chunk[i..i + 4]).map(|s| s.chars().next().unwrap_or('\0')) {
            if UNICODE_WHITESPACE_PATTERNS.contains(&(unicode_char as u16)) {
                if !result.ending_in_word {
                    result.ending_in_word = true;
                    result.unicode_whitespace_count += 1;
                }
            } else {
                if result.ending_in_word {
                    result.word_count += 1;
                    result.ending_in_word = false;
                }
            }
        }

        i += 1;
    }

    if !result.ending_in_word {
        result.ending_in_word = true;
    }

    result
}


unsafe fn count_patterns_fallback_chunk(chunk: &[u8]) -> ChunkResult {
    let mut result = ChunkResult::default();
    let mut in_whitespace = true;
    let mut j = 0;

    while j < chunk.len() {
        let byte = chunk[j];

        // Check for 4-byte UTF-8 characters first
        if byte & 0xF8 == 0xF0 {
            if in_whitespace {
                result.word_count += 1;
                in_whitespace = false;
            }
            result.four_byte_count += 1;
            if j >= chunk.len() - 4 {
                result.ending_in_utf32 = true;
                break;
            }
            j += 4;
            continue;
        }

        // Check for 3-byte UTF-8 characters
        if byte & 0xF0 == 0xE0 {
            if in_whitespace {
                result.word_count += 1;
                in_whitespace = false;
            }
            result.three_byte_count += 1;
            if j >= chunk.len() - 3 {
                result.ending_in_utf16 = true;
                break;
            }
            j += 3;
            continue;
        }

        // Check for whitespace
        if ASCII_WHITESPACE_PATTERNS.contains(&byte) {
            if !in_whitespace {
                in_whitespace = true;
                result.ascii_whitespace_count += 1;
            }
        } else if UNICODE_WHITESPACE_PATTERNS.contains(&(byte as u16)) {
            if !in_whitespace {
                in_whitespace = true;
                result.unicode_whitespace_count += 1;
            }
            if j >= chunk.len() - 1 {
                result.ending_in_utf16 = true;
                break;
            }
            j += 2;
            continue;
        } else {
            if in_whitespace {
                result.word_count += 1;
                in_whitespace = false;
            }

            // Check for ASCII characters
            if byte & 0x80 == 0 {
                result.ascii_count += 1;
            } else if byte & 0xE0 == 0xC0 {
                // Check for 2-byte UTF-8 characters
                result.two_byte_count += 1;
                if j >= chunk.len() - 2 {
                    result.ending_in_utf8 = true;
                    break;
                }
                j += 2;
                continue;
            }
        }

        j += 1;
    }

    if !in_whitespace {
        result.ending_in_word = true;
    }

    result
}

fn adjust_word_count(results: &mut Vec<ChunkResult>, bytes: &[u8]) {
    for i in 1..results.len() {
        let prev = &results[i - 1];
        let curr = &mut results[i];

        // Only check for unicode whitespace the case where the previous chunk ends in a UTF-8 sequence
        if prev.ending_in_utf8 {
            let prev_last_byte_index = i * 64 - 1;
            if prev_last_byte_index < bytes.len() {
                let prev_last_byte = bytes[prev_last_byte_index];

                // Check for straddling UTF-8 whitespace
                if prev_last_byte == 0x16 || prev_last_byte == 0x18 || prev_last_byte == 0x20 {
                    let first_byte_index = i * 64;
                    if first_byte_index < bytes.len() {
                        let first_byte = bytes[first_byte_index];

                        // Check if the combination of last byte and first byte forms 0x2009 (Thin Space) or 0x200A (Hair Space)
                        // for which the straggler byte will be erroneously flagged as ASCII whitespace
                        if (prev_last_byte == 0x20 && first_byte == 0x09) || (prev_last_byte == 0x20 && first_byte == 0x0A) {
                            // If so, adjust the ASCII whitespace count down
                            curr.ascii_whitespace_count -= 1;
                        } else {
                            // Combine the last byte of the previous chunk with the first byte of the current chunk
                            let combined_char = ((prev_last_byte as u16) << 8) | (first_byte as u16);

                            // Check if the combined character is a Unicode whitespace character
                            if UNICODE_WHITESPACE_PATTERNS.contains(&combined_char) {
                                let next_byte_index = first_byte_index + 1;
                                if next_byte_index < bytes.len() {
                                    let next_byte = bytes[next_byte_index];

                                    // Check if the next byte is either ASCII whitespace or Unicode whitespace
                                    if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                                        UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                        // If so, adjust the word count down by one
                                        curr.word_count -= 1;
                                    }
                                }
                            }
                        }
                    }
                }

                // Handle straddling UTF-8 non-whitespace sequences
                let first_byte_index = i * 64;
                if first_byte_index < bytes.len() {
                    let first_byte = bytes[first_byte_index];

                    // The first byte in the current chunk must be a continuation byte (10xxxxxx)
                    if first_byte & 0xC0 == 0x80 {
                        // Check if the next byte is either ASCII whitespace or Unicode whitespace
                        let next_byte_index = first_byte_index + 1;
                        if next_byte_index < bytes.len() {
                            let next_byte = bytes[next_byte_index];
                            if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                               UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                // This is a valid continuation byte followed by whitespace, so adjust the word count
                                curr.word_count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Check for straddling UTF-16 sequences
        if prev.ending_in_utf16 {
            for offset in 1..=2 {
                let first_byte_index = i * 64;
                if first_byte_index + offset < bytes.len() {
                    let first_bytes = &bytes[first_byte_index..first_byte_index + offset];

                    // UTF-16 can straddle by 1 or 2 bytes, adjust ASCII count for misinterpreted characters
                    if first_bytes.iter().all(|&b| b & 0xC0 == 0x80) {
                        curr.ascii_count += offset;

                        // Check if the next byte is a whitespace character
                        let next_byte_index = first_byte_index + offset;
                        if next_byte_index < bytes.len() {
                            let next_byte = bytes[next_byte_index];
                            if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                               UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                curr.word_count -= 1;
                            }
                        }
                    }
                }
            }
        }

        // Check for straddling UTF-32 sequences
        if prev.ending_in_utf32 {
            for offset in 1..=3 {
                let first_byte_index = i * 64;
                if first_byte_index + offset < bytes.len() {
                    let first_bytes = &bytes[first_byte_index..first_byte_index + offset];

                    // UTF-32 can straddle by 1, 2, or 3 bytes, adjust ASCII count for misinterpreted characters
                    if first_bytes.iter().all(|&b| b & 0xC0 == 0x80) {
                        curr.ascii_count += offset;

                        // Check if the next byte is a whitespace character
                        let next_byte_index = first_byte_index + offset;
                        if next_byte_index < bytes.len() {
                            let next_byte = bytes[next_byte_index];
                            if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                               UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                curr.word_count -= 1;
                            }
                        }
                    }
                }
            }
        }


        // Adjust the word count if the previous chunk ended in a word and the current chunk starts with a non-whitespace character
        if prev.ending_in_word {
            let first_byte_index = i * 64;
            if first_byte_index < bytes.len() {
                let first_byte = bytes[first_byte_index];
                if !ASCII_WHITESPACE_PATTERNS.contains(&first_byte) &&
                   !UNICODE_WHITESPACE_PATTERNS.contains(&(first_byte as u16)) {
                    curr.word_count -= 1;
                }
            }
        }
    }
}

fn count_patterns_parallel(filename: &str) -> io::Result<ChunkResult> {
    let file = File::open(filename)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes = mmap.as_ref();

    let chunk_size = 1_048_576; // Process 1MB chunks

    let total_chunks = (bytes.len() + chunk_size - 1) / chunk_size;
    let mut results: Vec<ChunkResult> = vec![ChunkResult::default(); total_chunks];

    // Process each chunk in parallel using Rayon
    results.par_iter_mut().enumerate().for_each(|(i, result)| {
        let start = i * chunk_size;
        let end = usize::min(start + chunk_size, bytes.len());

        let chunk = &bytes[start..end];

        let counts = if std::is_x86_feature_detected!("avx512f") {
            unsafe { count_patterns_avx512_chunk(chunk) }
        } else if std::is_x86_feature_detected!("avx2") {
            unsafe { count_patterns_avx2_chunk(chunk) }
        } else {
            unsafe { count_patterns_fallback_chunk(chunk) }
        };
        *result = counts;
    });

    adjust_word_count(&mut results, bytes);

    let total_result = results.iter().fold(ChunkResult::default(), |mut acc, r| {
        acc.ascii_count += r.ascii_count;
        acc.two_byte_count += r.two_byte_count;
        acc.three_byte_count += r.three_byte_count;
        acc.four_byte_count += r.four_byte_count;
        acc.ascii_whitespace_count += r.ascii_whitespace_count;
        acc.unicode_whitespace_count += r.unicode_whitespace_count;
        acc.word_count += r.word_count;
        acc
    });

    Ok(total_result)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }
    let filename = &args[1];

    let num_cpus = num_cpus::get();
    let avx512_enabled = std::is_x86_feature_detected!("avx512f");
    let avx2_enabled = std::is_x86_feature_detected!("avx2");

    println!("Number of processors: {}", num_cpus);
    println!("AVX-512 enabled: {}", avx512_enabled);
    println!("AVX2 enabled: {}", avx2_enabled);

    match count_patterns_parallel(filename) {
        Ok(result) => {
            println!("Number of ASCII characters: {}", result.ascii_count);
            println!("Number of 2-byte Unicode characters: {}", result.two_byte_count);
            println!("Number of 3-byte Unicode characters: {}", result.three_byte_count);
            println!("Number of 4-byte Unicode characters: {}", result.four_byte_count);
            println!("Number of ASCII whitespace characters: {}", result.ascii_whitespace_count);
            println!("Number of Unicode whitespace characters: {}", result.unicode_whitespace_count);
            println!("Number of words: {}", result.word_count);
        },
        Err(err) => eprintln!("Error: {}", err),
    }
}


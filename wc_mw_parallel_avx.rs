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
// ....and my dear mom, who is struggling, as I type. You really were the one who taught me algorithmic numerical analysis.

// Import necessary crates and modules

extern crate memmap;
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
    _mm256_loadu_si256, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_and_si256, _mm256_srli_si256,
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
const UNICODE_WHITESPACE_PATTERNS: [u16; 18] = [
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


#[derive(Default, Clone, Debug)]  // Added Debug trait
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
    block_start: usize,
    block_length: usize,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe fn compare_ascii_whitespace_avx512(chunk_data: __m512i) -> u64 {
    let mut result_mask = 0u64;

    // Unroll comparisons for each pattern
    result_mask |= _mm512_movemask_epi8(_mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(0x09))) as u64; // Tab (U+0009)
    result_mask |= _mm512_movemask_epi8(_mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(0x0A))) as u64; // Line Feed (U+000A)
    result_mask |= _mm512_movemask_epi8(_mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(0x0B))) as u64; // Vertical Tab (U+000B)
    result_mask |= _mm512_movemask_epi8(_mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(0x0C))) as u64; // Form Feed (U+000C)
    result_mask |= _mm512_movemask_epi8(_mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(0x0D))) as u64; // Carriage Return (U+000D)
    result_mask |= _mm512_movemask_epi8(_mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(0x20))) as u64; // Space (U+0020)

    result_mask
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn compare_ascii_whitespace_avx2(chunk_data: __m256i) -> u32 {
    let mut result_mask = 0u32;

    // Unroll comparisons for each pattern
    result_mask |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(0x09))) as u32; // Tab (U+0009)
    result_mask |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(0x0A))) as u32; // Line Feed (U+000A)
    result_mask |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(0x0B))) as u32; // Vertical Tab (U+000B)
    result_mask |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(0x0C))) as u32; // Form Feed (U+000C)
    result_mask |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(0x0D))) as u32; // Carriage Return (U+000D)
    result_mask |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(0x20))) as u32; // Space (U+0020)

    result_mask
}




#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe fn compare_unicode_whitespace_avx512(chunk_data: __m512i, shifted_chunk_data: __m512i) -> u64 {
    let mut result_mask = 0u64;
    let mut shifted_result_mask = 0u64;

    macro_rules! compare_and_mask {
        ($data:expr, $shifted_data:expr, $value:expr) => {
            {
                // Compare original chunk data
                result_mask |= _mm512_movemask_epi16(_mm512_cmpeq_epi16_mask($data, _mm512_set1_epi16($value as i16))) as u64;
                // Compare shifted chunk data
                shifted_result_mask |= (_mm512_movemask_epi16(_mm512_cmpeq_epi16_mask($shifted_data, _mm512_set1_epi16($value as i16))) as u64) >> 1;
            }
        };
    }

    // Apply comparisons for each whitespace character
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x00A0); // Non-breaking Space (U+00A0)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x1680); // Ogham Space Mark (U+1680)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x180E); // Mongolian Vowel Separator (U+180E)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2000); // En Quad (U+2000)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2001); // Em Quad (U+2001)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2002); // En Space (U+2002)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2003); // Em Space (U+2003)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2004); // Three-per-em Space (U+2004)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2005); // Four-per-em Space (U+2005)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2006); // Six-per-em Space (U+2006)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2007); // Figure Space (U+2007)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2008); // Punctuation Space (U+2008)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2009); // Thin Space (U+2009)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x200A); // Hair Space (U+200A)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2028); // Line Separator (U+2028)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2029); // Paragraph Separator (U+2029)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x205F); // Medium Mathematical Space (U+205F)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x3000); // Ideographic Space (U+3000)

    // Combine the result masks for chunk_data and shifted_chunk_data
    result_mask | shifted_result_mask
}



#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn compare_unicode_whitespace_avx2(chunk_data: __m256i, shifted_chunk_data: __m256i) -> u32 {
    let mut result_mask = 0u32; // Initialize the result mask for chunk_data
    let mut shifted_result_mask = 0u32; // Initialize the result mask for shifted_chunk_data

    macro_rules! compare_and_mask {
        ($data:expr, $shifted_data:expr, $value:expr) => {
            {
                // Compare original chunk data
                result_mask |= (_mm256_movemask_epi8(_mm256_cmpeq_epi16($data, _mm256_set1_epi16($value))) & 0x55555555) as u32;
                // Compare shifted chunk data
                shifted_result_mask |= (_mm256_movemask_epi8(_mm256_cmpeq_epi16($shifted_data, _mm256_set1_epi16($value))) & 0x55555555) as u32 >> 1;
            }
        };
    }

    // Apply comparisons for each whitespace character
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x00A0); // Non-breaking Space (U+00A0)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x1680); // Ogham Space Mark (U+1680)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x180E); // Mongolian Vowel Separator (U+180E)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2000); // En Quad (U+2000)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2001); // Em Quad (U+2001)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2002); // En Space (U+2002)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2003); // Em Space (U+2003)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2004); // Three-per-em Space (U+2004)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2005); // Four-per-em Space (U+2005)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2006); // Six-per-em Space (U+2006)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2007); // Figure Space (U+2007)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2008); // Punctuation Space (U+2008)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2009); // Thin Space (U+2009)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x200A); // Hair Space (U+200A)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2028); // Line Separator (U+2028)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x2029); // Paragraph Separator (U+2029)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x205F); // Medium Mathematical Space (U+205F)
    compare_and_mask!(chunk_data, shifted_chunk_data, 0x3000); // Ideographic Space (U+3000)

    // Combine the result masks for chunk_data and shifted_chunk_data
    result_mask | shifted_result_mask
}



#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn count_patterns_avx2_chunk(chunk_ptr: *const u8) -> ChunkResult {
    let chunk_len = 64; // Define the chunk length
    // Load 32-byte chunks and their shifted versions
    let chunk_data1 = _mm256_loadu_si256(chunk_ptr as *const __m256i);
    let shifted_chunk_data1 = _mm256_loadu_si256(chunk_ptr.add(1) as *const __m256i);
    let chunk_data2 = _mm256_loadu_si256(chunk_ptr.add(32) as *const __m256i);
    let shifted_chunk_data2 = _mm256_loadu_si256(chunk_ptr.add(33) as *const __m256i);

    // Apply masks and perform UTF sequence comparisons
    let is_two_byte_utf_mask = (u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk_data2, _mm256_set1_epi8(0b11000000)), _mm256_set1_epi8(0b11000000))))
        << 32) | u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk_data1, _mm256_set1_epi8(0b11000000)), _mm256_set1_epi8(0b11000000))));
    let is_three_byte_utf_mask = (u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk_data2, _mm256_set1_epi8(0b11100000)), _mm256_set1_epi8(0b11100000))))
        << 32) | u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk_data1, _mm256_set1_epi8(0b11100000)), _mm256_set1_epi8(0b11100000))));
    let is_four_byte_utf_mask = (u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk_data2, _mm256_set1_epi8(0b11110000)), _mm256_set1_epi8(0b11110000))))
        << 32) | u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk_data1, _mm256_set1_epi8(0b11110000)), _mm256_set1_epi8(0b11110000))));

    // Identify and mask out ASCII whitespace
    let ascii_whitespace_mask = u64::from(compare_ascii_whitespace_avx2(chunk_data1)) | (u64::from(compare_ascii_whitespace_avx2(chunk_data2)) << 32);

    // Identify and mask out Unicode whitespace
    let unicode_whitespace_mask = ((u64::from(compare_unicode_whitespace_avx2(chunk_data2, shifted_chunk_data2))) << 32)
        | u64::from(compare_unicode_whitespace_avx2(chunk_data1, shifted_chunk_data1));

    // Boundary check for straddling UTF-8 sequences
    if is_two_byte_utf_mask & 1 != 0 {
        let last_byte = *chunk_ptr.add(31);
        if last_byte == 0x20 || last_byte == 0x18 || last_byte == 0x16 {
            let combined_char = ptr::read_unaligned(chunk_ptr.add(31) as *const u16);
            if combined_char == 0x2009 || combined_char == 0x200A {
                // Correct the masks for ASCII and Unicode whitespace
                ascii_whitespace_mask &= !(1 << 31);
                unicode_whitespace_mask |= 1 << 32;
            } else if UNICODE_WHITESPACE_PATTERNS.contains(&combined_char) {
                // Correct the mask for Unicode whitespace
                unicode_whitespace_mask |= 1 << 32;
            }
        }
    }

    // Combine the masks
    let whitespace_mask = ascii_whitespace_mask | unicode_whitespace_mask;

    // Use the masks to count words and character types
    count_words_and_chars(chunk_len, is_two_byte_utf_mask, is_three_byte_utf_mask, is_four_byte_utf_mask, ascii_whitespace_mask, whitespace_mask)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe fn count_patterns_avx512_chunk(chunk_ptr: *const u8) -> ChunkResult {
    let chunk_data = _mm512_loadu_si512(chunk_ptr as *const __m512i);
    let shifted_chunk_data = _mm512_loadu_si512(chunk_ptr.add(1) as *const __m512i);
    let chunk_len = 64; // Define the chunk length
	
    // Apply masks and perform UTF sequence comparisons
    let is_two_byte_utf_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(chunk_data, _mm512_set1_epi8(0b11000000)), _mm512_set1_epi8(0b11000000)) as u64;
    let is_three_byte_utf_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(chunk_data, _mm512_set1_epi8(0b11100000)), _mm512_set1_epi8(0b11100000)) as u64;
    let is_four_byte_utf_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(chunk_data, _mm512_set1_epi8(0b11110000)), _mm512_set1_epi8(0b11110000)) as u64;

    // Identify and mask out ASCII whitespace
    let ascii_whitespace_mask = compare_ascii_whitespace_avx512(chunk_data);

    // Identify and mask out Unicode whitespace
    let unicode_whitespace_mask = compare_unicode_whitespace_avx512(chunk_data, shifted_chunk_data);

    // Combine the masks
    let whitespace_mask = ascii_whitespace_mask | unicode_whitespace_mask;

    // Use the masks to count words and character types
    count_words_and_chars(chunk_len, is_two_byte_utf_mask, is_three_byte_utf_mask, is_four_byte_utf_mask, ascii_whitespace_mask, whitespace_mask)
}


fn count_words_and_chars(
    chunk_len: usize,
    is_two_byte_utf_mask: u64,
    is_three_byte_utf_mask: u64,
    is_four_byte_utf_mask: u64,
    ascii_whitespace_mask: u64,
    whitespace_mask: u64,
) -> ChunkResult {
    let mut result = ChunkResult::default();
    let mut in_whitespace = true;
    let mut j = 0;

    while j < chunk_len {
        let bit = 1 << j;

        // Check if the current position is the start of a 4-byte UTF-8 character
        if (is_four_byte_utf_mask & bit) != 0 {
            if in_whitespace { 
                // Start of a new word
                result.word_count += 1; 
            }
            in_whitespace = false;
            result.four_byte_count += 1;
            if j >= chunk_len - 4 {
                // Handle edge case: 4-byte character at the end of the chunk
                result.ending_in_utf32 = true;
                break;
            }
            j += 4; // Move forward by 4 bytes
            continue;
        }

        // Check if the current position is the start of a 3-byte UTF-8 character
        if (is_three_byte_utf_mask & bit) != 0 {
            if in_whitespace {
                // Start of a new word
                result.word_count += 1; 
            }
            in_whitespace = false;
            result.three_byte_count += 1;
            if j >= chunk_len - 3 {
                // Handle edge case: 3-byte character at the end of the chunk
                result.ending_in_utf16 = true;
                break;
            }
            j += 3; // Move forward by 3 bytes
            continue;
        }

        // Check if the current position is a whitespace character
        if (whitespace_mask & bit) != 0 {
            in_whitespace = true;
            if (ascii_whitespace_mask & bit) != 0 {
            // ASCII whitespace
                result.ascii_whitespace_count += 1;
            } else {
                // Unicode whitespace
                result.unicode_whitespace_count += 1;
                if j >= chunk_len - 1 {
                    // Handle edge case: Unicode whitespace at the end of the chunk
                    result.ending_in_utf8 = true;
                    break;
                }
                j += 2; // Move forward by 2 bytes for Unicode whitespace
                continue;
            }
            
        } else {
            if in_whitespace {
                // Start of a new word
                result.word_count += 1; 
            }
            in_whitespace = false;
            
            // Check if the current position is the start of a 2-byte UTF-8 character
            if (is_two_byte_utf_mask & bit) != 0 {
                result.two_byte_count += 1;
                if j >= chunk_len - 2 {
                    // Handle edge case: 2-byte character at the end of the chunk
                    result.ending_in_utf8 = true;
                    break;
                }
                j += 2; // Move forward by 2 bytes
                continue;
            } else {
                // ASCII character
                result.ascii_count += 1;
            }
        }

        j += 1; // Move to the next byte
    }

    if !in_whitespace {
        result.ending_in_word = true;
    }

    result
}


// #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
fn count_patterns_fallback_chunk(chunk: &[u8]) -> ChunkResult {
    let mut result = ChunkResult::default();
    let mut in_whitespace = true;
    let mut j = 0;

    while j < chunk.len() {
        let byte = chunk[j];

        // Check for 4-byte UTF-32 characters first
        if byte & 0xF8 == 0xF0 {
            if in_whitespace {
                result.word_count += 1;
	    }
            in_whitespace = false;
            result.four_byte_count += 1;
            if j >= chunk.len() - 4 {
                result.ending_in_utf32 = true;
                break;
            }
            j += 4;
            continue;
        }

        // Check for 3-byte UTF-16 characters
        if byte & 0xF0 == 0xE0 {
            if in_whitespace {
                result.word_count += 1;
	    }
            in_whitespace = false;
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
            in_whitespace = true;
            result.ascii_whitespace_count += 1;
        } else if UNICODE_WHITESPACE_PATTERNS.contains(&(byte as u16)) {
            in_whitespace = true;
            result.unicode_whitespace_count += 1;
            if j >= chunk.len() - 1 {
                result.ending_in_utf8 = true;
                break;
            }
            j += 2;
            continue;
        } else {
            if in_whitespace {
                result.word_count += 1;
	    }
            in_whitespace = false;
            
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


fn adjust_word_count(results: &[ChunkResult], bytes: &[u8]) -> ChunkResult {
    if results.len() < 2 {
        return results[0].clone();
    }

    let mut total_result = ChunkResult {
        block_start: results[0].block_start,
        ..ChunkResult::default()
    };

    total_result.ascii_count = 0;

    for i in 0..(results.len() - 1) {
        let curr = &results[i];
        let next = &results[i + 1];

        // Only check for unicode whitespace the case where the previous chunk ends in a UTF-8 sequence
        if curr.ending_in_utf8 {
            let curr_last_byte_index = curr.block_start + curr.block_length - 1;
            let curr_last_byte = bytes[curr_last_byte_index];

            // Check for straddling UTF-8 whitespace
            if curr_last_byte == 0x16 || curr_last_byte == 0x18 || curr_last_byte == 0x20 {
                let next_first_byte_index = next.block_start;
                if next_first_byte_index < bytes.len() {
                    let next_first_byte = bytes[next_first_byte_index];

                    // Check if the combination of last byte and first byte forms 0x2009 (Thin Space) or 0x200A (Hair Space)
                    // for which the straggler byte will be erroneously flagged as ASCII whitespace
                    if (curr_last_byte == 0x20 && next_first_byte == 0x09) || (curr_last_byte == 0x20 && next_first_byte == 0x0A) {
                        // If so, adjust the ASCII whitespace count down
                        total_result.ascii_whitespace_count -= 1;
                        total_result.unicode_whitespace_count += 1;
                    } else {
                        // Combine the last byte of the previous chunk with the first byte of the current chunk
                        let combined_char = ((curr_last_byte as u16) << 8) | (next_first_byte as u16);

                        // Check if the combined character is a Unicode whitespace character
                        if UNICODE_WHITESPACE_PATTERNS.contains(&combined_char) {
                            total_result.unicode_whitespace_count += 1;
                            let next_byte_index = next_first_byte_index + 1;
                            if next_byte_index < bytes.len() {
                                let next_byte = bytes[next_byte_index];
                                // Check if the next byte is either ASCII whitespace or Unicode whitespace
                                if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                                    UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                    // If so, adjust the word count down by one
                                    total_result.word_count -= 1;
                                }
                            }
                        }
                    }
                }
            }

            let next_first_byte_index = next.block_start;
            if next_first_byte_index < bytes.len() {
                let next_first_byte = bytes[next_first_byte_index];

                // The first byte in the current chunk must be a continuation byte (10xxxxxx)
                if next_first_byte & 0xC0 == 0x80 {
                    // Check if the next byte is either ASCII whitespace or Unicode whitespace
                    let next_byte_index = next_first_byte_index + 1;
                    if next_byte_index < bytes.len() {
                        let next_byte = bytes[next_byte_index];
                        if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                            UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                            // This is a valid continuation byte followed by whitespace, so adjust the word count
                            total_result.word_count += 1;
                        }
                    }
                }
            }
        }

        // Check for straddling UTF-16 sequences
        if curr.ending_in_utf16 {
            for offset in 1..=2 {
                let next_first_byte_index = next.block_start;
                if next_first_byte_index + offset < bytes.len() {
                    let next_first_bytes = &bytes[next_first_byte_index..next_first_byte_index + offset];

                    // UTF-16 can straddle by 1 or 2 bytes, adjust ASCII count for misinterpreted characters
                    if next_first_bytes.iter().all(|&b| b & 0xC0 == 0x80) {
                        total_result.ascii_count += offset;

                        // Check if the next byte is a whitespace character
                        let next_byte_index = next_first_byte_index + offset;
                        if next_byte_index < bytes.len() {
                            let next_byte = bytes[next_byte_index];
                            if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                                UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                total_result.word_count -= 1;
                            }
                        }
                    }
                }
            }
        }

        // Check for straddling UTF-32 sequences
        if curr.ending_in_utf32 {
            for offset in 1..=3 {
                let next_first_byte_index = next.block_start;
                if next_first_byte_index + offset < bytes.len() {
                    let next_first_bytes = &bytes[next_first_byte_index..next_first_byte_index + offset];

                    // UTF-32 can straddle by 1, 2, or 3 bytes, adjust ASCII count for misinterpreted characters
                    if next_first_bytes.iter().all(|&b| b & 0xC0 == 0x80) {
                        total_result.ascii_count += offset;

                        // Check if the next byte is a whitespace character
                        let next_byte_index = next_first_byte_index + offset;
                        if next_byte_index < bytes.len() {
                            let next_byte = bytes[next_byte_index];
                            if ASCII_WHITESPACE_PATTERNS.contains(&next_byte) ||
                                UNICODE_WHITESPACE_PATTERNS.contains(&(next_byte as u16)) {
                                total_result.word_count -= 1;
                            }
                        }
                    }
                }
            }
        }

        // Adjust the word count if the previous chunk ended in a word and the current chunk starts with a non-whitespace character
        if curr.ending_in_word {
            let next_first_byte_index = next.block_start;
            if next_first_byte_index < bytes.len() {
                let next_first_byte = bytes[next_first_byte_index];
                if !ASCII_WHITESPACE_PATTERNS.contains(&next_first_byte) &&
                    !UNICODE_WHITESPACE_PATTERNS.contains(&(next_first_byte as u16)) {
                    total_result.word_count -= 1;
                }
            }
        }

        // Accumulate the results
        total_result.ascii_count += curr.ascii_count;
        total_result.two_byte_count += curr.two_byte_count;
        total_result.three_byte_count += curr.three_byte_count;
        total_result.four_byte_count += curr.four_byte_count;
        total_result.ascii_whitespace_count += curr.ascii_whitespace_count;
        total_result.unicode_whitespace_count += curr.unicode_whitespace_count;
        total_result.word_count += curr.word_count;
    }

    // Propagate flags from the last result
    let last_result = &results[results.len() - 1];
    total_result.ascii_count += last_result.ascii_count;
    total_result.ending_in_word = last_result.ending_in_word;
    total_result.ending_in_utf32 = last_result.ending_in_utf32;
    total_result.ending_in_utf16 = last_result.ending_in_utf16;
    total_result.ending_in_utf8 = last_result.ending_in_utf8;

    // Add the last block counts to the total result
    total_result.two_byte_count += last_result.two_byte_count;
    total_result.three_byte_count += last_result.three_byte_count;
    total_result.four_byte_count += last_result.four_byte_count;
    total_result.ascii_whitespace_count += last_result.ascii_whitespace_count;
    total_result.unicode_whitespace_count += last_result.unicode_whitespace_count;
    total_result.word_count += last_result.word_count;

    total_result
}


fn count_patterns_parallel(filename: &str) -> io::Result<ChunkResult> {
    let file = File::open(filename)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes = mmap.as_ref();

    let chunk_size = 1_048_576; // Process 1MB chunks

    // Determine block size based on AVX2 or AVX512 support
    let block_size = if std::is_x86_feature_detected!("avx512f") {
        64 // Use 64-byte blocks for AVX-512
    } else if std::is_x86_feature_detected!("avx2") {
        64 // Use 64-byte blocks for AVX2
    } else {
        1_048_576 // Use 1MB blocks if AVX is not supported
    };

    let total_chunks = (bytes.len() + chunk_size - 1) / chunk_size; // Calculate total number of chunks

    // Determine padding size based on AVX2 or AVX512 support
    let padding_size = if std::is_x86_feature_detected!("avx512f") {
        64 // Use 64-byte padding for AVX-512
    } else if std::is_x86_feature_detected!("avx2") {
        64 // Use 64-byte padding for AVX2
    } else {
        0 // No padding needed if AVX is not supported
    };

    // Process each chunk in parallel using Rayon
    let mut chunk_results: Vec<ChunkResult> = (0..total_chunks).into_par_iter().map(|chunk_idx| {
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end = usize::min(chunk_start + chunk_size, bytes.len());
        let chunk_len = chunk_end - chunk_start;

        // Create a local results vector to store block results
        let mut local_results = vec![];

        for block_idx in 0..(chunk_len + block_size - 1) / block_size {
            let block_start = chunk_start + block_idx * block_size;
            let block_end = usize::min(block_start + block_size, chunk_end);
            let block = if padding_size > 0 && block_end - block_start < padding_size {
                // If the block is smaller than the padding size, pad it with zeros
                let mut padded_block = vec![0u8; padding_size];
                let real_size = block_end - block_start;
                padded_block[..real_size].copy_from_slice(&bytes[block_start..block_end]);
                padded_block
            } else {
                // Otherwise, just use the block as is
                bytes[block_start..block_end].to_vec()
            };

            // Determine which SIMD version to use based on CPU features
            let block_result = if std::is_x86_feature_detected!("avx512f") {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                {
                    unsafe { count_patterns_avx512_chunk(block.as_ptr()) }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                {
                    eprintln!("Warning: count_patterns_avx512_chunk routine not compiled in");
                    count_patterns_fallback_chunk(&block)
                }
            } else if std::is_x86_feature_detected!("avx2") {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    unsafe { count_patterns_avx2_chunk(block.as_ptr()) }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    eprintln!("Warning: count_patterns_avx2_chunk routine not compiled in");
                    count_patterns_fallback_chunk(&block)
                }
            } else {
                count_patterns_fallback_chunk(&block)
            };

            let mut result = block_result;
            result.block_start = block_start; // Set absolute block start index
            result.block_length = block_end - block_start; // Set block length

            // Adjust the total result for the last partial chunk
            if block_end - block_start < block_size {
                let actual_padding = block_size - (block_end - block_start);
                result.ascii_count = result.ascii_count.saturating_sub(actual_padding);
                // Check if the last byte of the real data is a whitespace character
                let last_byte = bytes[bytes.len() - 1];
                if ASCII_WHITESPACE_PATTERNS.contains(&last_byte) || UNICODE_WHITESPACE_PATTERNS.contains(&((last_byte-1) as u16)) {
                    result.word_count -= 1;
                }
            }

            local_results.push(result);
        }

        // Adjust word count within the chunk
        let chunk_result = adjust_word_count(&mut local_results, bytes);

        chunk_result

    }).collect();

    // Adjust word count for boundaries between chunks
    let final_result = adjust_word_count(&mut chunk_results, bytes);

    Ok(final_result)
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
            println!("ASCII characters: {}", result.ascii_count);
            println!("2-byte Unicode characters: {}", result.two_byte_count);
            println!("3-byte Unicode characters: {}", result.three_byte_count);
            println!("4-byte Unicode characters: {}", result.four_byte_count);
            println!("ASCII whitespace characters: {}", result.ascii_whitespace_count);
            println!("Unicode whitespace characters: {}", result.unicode_whitespace_count);
            println!("Total characters: {}", result.ascii_count + result.two_byte_count + result.three_byte_count + result.four_byte_count + result.ascii_whitespace_count + result.unicode_whitespace_count);
            println!("Broken wc -mw compatibility character count: {}", result.ascii_count + result.two_byte_count + result.three_byte_count + result.four_byte_count + result.ascii_whitespace_count + result.unicode_whitespace_count *2);
            println!("File Length(bytes): {}", result.ascii_count + result.two_byte_count*2 + result.three_byte_count*3 + result.four_byte_count*4 + result.ascii_whitespace_count + result.unicode_whitespace_count *2);
            println!("Number of words: {}", result.word_count);
        },
        Err(err) => eprintln!("Error: {}", err),
    }
}

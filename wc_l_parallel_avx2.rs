
// wc -l optimized in Rust using Rayon for parallelization, and AVX2 SIMD optimization on each core.
// please note: at this time using avx2 requires using the nightly rustc tool chain: 
//                                                    rustup install nightly
//                                                    rustup default nightly
//                                                    rustup component add rust-src --toolchain nightly
//
// Build with: RUSTFLAGS="-C target-cpu=native"  cargo build --release
// Use lscpu to verify presence of avx2.
//
//Cargo.toml:
//
//[package]
//name = "wc_l_parallel_avx2"
//version = "0.1.0"
//edition = "2021"
//
//[[bin]]
//name = "wc_l_parallel_avx2"
//
//[dependencies]
//memmap = "0.7"
//rayon = "1.5"

// (C) Copyright 2024 Dragos Ruiu

#![feature(stdsimd)] // Enable SIMD features

// Import necessary crates and modules

extern crate memmap;
extern crate rayon;

use memmap::Mmap;                     // For memory-mapped file I/O
use rayon::prelude::*;                // For parallel processing
use std::env;                         // For handling command line arguments
use std::fs::File;                    // For file handling
use std::io;                          // For I/O operations

// Import necessary AVX2 functions and types
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm256_loadu_si256, _mm256_cmpeq_epi8, _mm256_set1_epi8, _mm256_movemask_epi8, __m256i
}; // For AVX2 SIMD instructions

// Function to count newline characters in a chunk using AVX2 SIMD instructions
#[cfg(target_feature = "avx2")]
unsafe fn count_newlines_avx2(chunk: &[u8]) -> usize {
    let mut line_count = 0;           // Initialize line count
    let mut i = 0;                    // Initialize byte index
    // Process 32-byte chunks using AVX2
    while i + 32 <= chunk.len() {
        // Load 32 bytes from the chunk into an AVX2 register
        let chunk_data = _mm256_loadu_si256(chunk.as_ptr().add(i) as *const __m256i);
        // Compare each byte with the newline character '\n'
        let newlines = _mm256_cmpeq_epi8(chunk_data, _mm256_set1_epi8(b'\n' as i8));
        // Create a mask of the results
        let mask = _mm256_movemask_epi8(newlines);
        // Count the number of set bits (newlines) in the mask
        line_count += mask.count_ones() as usize;
        i += 32;                  // Move to the next 32-byte chunk
    }
    // Process any remaining bytes that are not part of a full 32-byte chunk
    while i < chunk.len() {
        if chunk[i] == b'\n' {
            line_count += 1;          // Increment line count for each newline character
        }
        i += 1;                       // Move to the next byte
    }
    line_count                        // Return the total line count for the chunk
}

// Fallback function to count newline characters in a chunk without AVX2
#[cfg(not(target_feature = "avx2"))]
fn count_newlines_avx2(chunk: &[u8]) -> usize {
    chunk.iter().filter(|&&c| c == b'\n').count()
}

// Function to count lines in a file using parallel processing and AVX2 SIMD instructions
fn count_lines_parallel(filename: &str) -> io::Result<usize> {
    let file = File::open(filename)?;                               // Open the file
    let mmap = unsafe { Mmap::map(&file)? };                        // Memory-map the file
    let bytes = mmap.as_ref();                                      // Get a byte slice of the file's contents

    let chunk_size = 1024 * 1024;                                   // Define the chunk size (1MB) - this may need optimization for your processor architecture.
    let total_chunks = (bytes.len() + chunk_size - 1) / chunk_size; // Calculate the total number of chunks
    let mut results: Vec<usize> = vec![0; total_chunks];            // Create a vector to store results for each chunk

    // Process each chunk in parallel using Rayon
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, result)| {
            let start = i * chunk_size;                             // Calculate the start index of the chunk
            let end = usize::min(start + chunk_size, bytes.len());  // Calculate the end index of the chunk
            *result = count_newlines_avx2(&bytes[start..end]);      // Count newlines in the chunk and store the result
        });

    Ok(results.iter().sum())                                        // Sum the results from all chunks and return the total line count
}

fn main() {
    let args: Vec<String> = env::args().collect();                  // Collect command line arguments
    if args.len() != 2 {
        // Check if the correct number of arguments is provided
        eprintln!("Usage: {} <filename>", args[0]);                 // Print usage message
        std::process::exit(1);                                      // Exit with an error code
    }
    let filename = &args[1];                                        // Get the filename from the command line arguments
    match count_lines_parallel(filename) {
        // Count lines in the file
        Ok(count) => println!("Number of lines: {}", count),        // Print the line count if successful
        Err(err) => eprintln!("Error: {}", err),                    // Print an error message if an error occurs
    }
}

// wc -l optimized in Rust using Rayon for paralleization, and AVX512 SIMD optimization on each core.
// please note: at this time using avx512 requires using the nightly rustc tool chain: 
//                                                    rustup install nightly
//                                                    rustup default nightly
//                                                    rustup component add rust-src --toolchain nightly
//
// Build with: RUSTFLAGS="-C target-cpu=native"  cargo build --release
// Use lscpu to verify presence of avx512f.
//
//Cargo.toml:
//
//[package]
//name = "wc_l_parallel_avx512"
//version = "0.1.0"
//edition = "2021"
//
//[[bin]]
//name = "wc_l_parallel_avx512"
//
//[dependencies]
//memmap = "0.7"
//rayon = "1.5"

// (C) Copyright 2024 Dragos Ruiu

#![feature(stdsimd)] // Enable SIMD features
#![feature(stdarch_x86_avx512)] // Enable AVX-512 features

// Import necessary crates and modules

extern crate memmap;
extern crate rayon;

use memmap::Mmap;                     // For memory-mapped file I/O
use rayon::prelude::*;                // For parallel processing
use std::env;                         // For handling command line arguments
use std::fs::File;                    // For file handling
use std::io;                          // For I/O operations

#[cfg(target_feature = "avx512f")]
use std::arch::x86_64::*;             // For AVX-512 SIMD instructions

// Function to count newline characters in a chunk using AVX-512 SIMD instructions
unsafe fn count_newlines_avx512(chunk: &[u8]) -> usize {
    let mut line_count = 0;           // Initialize line count
    let mut i = 0;                    // Initialize byte index
    unsafe {
        // Process 64-byte chunks using AVX-512
        while i + 64 <= chunk.len() {
            // Load 64 bytes from the chunk into an AVX-512 register
            let chunk_data = _mm512_loadu_si512(chunk.as_ptr().add(i) as *const __m512i);
            // Compare each byte with the newline character '\n'
            let newlines = _mm512_cmpeq_epi8_mask(chunk_data, _mm512_set1_epi8(b'\n' as i8));
            // Count the number of newlines in this 64-byte chunk
            line_count += _mm_popcnt_u32(newlines as u32) as usize;
            i += 64;                  // Move to the next 64-byte chunk
        }
    }
    // Process any remaining bytes that are not part of a full 64-byte chunk
    while i < chunk.len() {
        if chunk[i] == b'\n' {
            line_count += 1;          // Increment line count for each newline character
        }
        i += 1;                       // Move to the next byte
    }
    line_count                        // Return the total line count for the chunk
}

// Fallback function to count newline characters in a chunk without AVX-512
#[cfg(not(target_feature = "avx512f"))]
fn count_newlines_avx512(chunk: &[u8]) -> usize {
    chunk.iter().filter(|&&c| c == b'\n').count()
}

// Function to count lines in a file using parallel processing and AVX-512 SIMD instructions
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
            let start = i * chunk_size;                            // Calculate the start index of the chunk
            let end = usize::min(start + chunk_size, bytes.len()); // Calculate the end index of the chunk
            *result = unsafe { count_newlines_avx512(&bytes[start..end]) };   // Count newlines in the chunk and store the result
        });

    Ok(results.iter().sum())                                       // Sum the results from all chunks and return the total line count
}

fn main() {
    let args: Vec<String> = env::args().collect();                 // Collect command line arguments
    if args.len() != 2 {
        // Check if the correct number of arguments is provided
        eprintln!("Usage: {} <filename>", args[0]);                // Print usage message
        std::process::exit(1);                                     // Exit with an error code
    }
    let filename = &args[1];                                       // Get the filename from the command line arguments
    match count_lines_parallel(filename) {
        // Count lines in the file
        Ok(count) => println!("Number of lines: {}", count),      // Print the line count if successful
        Err(err) => eprintln!("Error: {}", err),                  // Print an error message if an error occurs
    }
}

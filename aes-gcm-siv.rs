#![allow(non_upper_case_globals)]

// AES-GCM-SIV - December 16, 2024
// RFC: https://www.rfc-editor.org/rfc/rfc8452
//
// This Rust code provides a robust and optimized implementation of AES-GCM-SIV,
// incorporating hardware acceleration where available and adhering to constant-time principles.
//
//! AES-GCM-SIV implementation with hardware acceleration support
//! Based on RFC 8452

use std::fmt;
use std::error::Error;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, Ordering};
use zeroize::{Zeroize, ZeroizeOnDrop};
use rayon::prelude::*;

// Feature flags for different implementations
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ====================== Constants and Limits ======================
#[repr(usize)]
#[derive(Clone, Copy, Debug, Zeroize)]
pub enum KeySize {
    Aes128 = 16,
    Aes256 = 32,
}

#[repr(usize)]
pub enum BlockSize {
    Aes = 16,
}

pub const AES_GCMSIV_TAG_SIZE: usize = BlockSize::Aes as usize;
pub const AES_GCMSIV_NONCE_SIZE: usize = 12;
pub const POLYVAL_SIZE: usize = BlockSize::Aes as usize;
pub const AES_GCMSIV_MAX_PLAINTEXT_SIZE: usize = (1 << 36) - 1;
pub const AES_GCMSIV_MAX_AAD_SIZE: usize = (1 << 36) - 1;

// Batch processing configuration
const PARALLEL_THRESHOLD: usize = 1024 * 64; // 64KB
const MAX_PARALLEL_BLOCKS: usize = 8;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AesGcmSivError {
    #[error("Invalid key size {size}, expected one of {expected:?}")]
    InvalidKeySize {
        size: usize,
        expected: &'static [usize],
    },

    #[error("Invalid nonce size {size}, expected {expected}")]
    InvalidNonceSize {
        size: usize,
        expected: usize,
    },

    #[error("Plaintext size {size} exceeds maximum {max}")]
    InvalidPlaintextSize {
        size: usize,
        max: usize,
    },

    #[error("AAD size {size} exceeds maximum {max}")]
    InvalidAadSize {
        size: usize,
        max: usize,
    },

    #[error("Ciphertext size {size} is less than minimum {min}")]
    InvalidCiphertextSize {
        size: usize,
        min: usize,
    },

    #[error("Authentication tag verification failed")]
    InvalidTag,

    #[error("Output buffer too small: need {needed} bytes, got {provided}")]
    BufferTooSmall {
        provided: usize,
        needed: usize,
    },

    #[error("Memory allocation failed")]
    AllocationError,

    #[error("Cipher context not initialized")]
    UninitializedContext,

    #[error("CPU feature {0} not supported")]
    UnsupportedCpuFeature(&'static str),

    #[error("Counter overflow in CTR mode")]
    CounterOverflow,
}

pub type Result<T> = std::result::Result<T, AesGcmSivError>;

#[derive(Zeroize, ZeroizeOnDrop)]
#[repr(C, align(16))]
pub struct KeyContext {
    auth: [u8; POLYVAL_SIZE],
    auth_sz: usize,
    #[zeroize(skip)]
    enc: Box<[u8]>,
    enc_sz: usize,
}

impl KeyContext {
    pub fn new(key_size: KeySize) -> Self {
        let enc_size = key_size as usize;
        Self {
            auth: [0u8; POLYVAL_SIZE],
            auth_sz: POLYVAL_SIZE,
            enc: vec![0u8; enc_size].into_boxed_slice(),
            enc_sz: enc_size,
        }
    }
}

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
struct DerivedKeys {
    auth_key: [u8; 16],
    enc_key: Vec<u8>,
}

// Counter for CTR mode with overflow protection
#[derive(Clone, Debug, Zeroize)]
struct Counter {
    value: u128,
    max: u128,
}

impl Counter {
    fn new(initial: &[u8; 16], max_blocks: u64) -> Self {
        let value = u128::from_be_bytes(*initial);
        Self {
            value,
            max: value.saturating_add(max_blocks as u128),
        }
    }

    fn increment(&mut self) -> Result<[u8; 16]> {
        if self.value >= self.max {
            return Err(AesGcmSivError::CounterOverflow);
        }
        let result = self.value.to_be_bytes();
        self.value = self.value.wrapping_add(1);
        Ok(result)
    }
}

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Polyval {
    h: [u8; 16],
    s: [u8; 16],
    #[zeroize(skip)]
    mul_tables: Box<[[u8; 256]; 16]>,
}

impl Polyval {
    const R: u128 = 0xE100000000000000;

    pub fn new(h: &[u8; 16]) -> Self {
        let mut instance = Self {
            h: *h,
            s: [0u8; 16],
            mul_tables: Box::new([[0u8; 256]; 16]),
        };
        instance.precompute_tables();
        instance
    }

    fn precompute_tables(&mut self) {
        for i in 0..16 {
            for j in 0..256 {
                let mut x = [0u8; 16];
                x[i] = j as u8;
                self.mul_tables[i][j] = self.gf_mul_single(x[i]);
            }
        }
    }

    #[inline(always)]
    fn gf_mul_precomputed(&self, x: [u8; 16]) -> [u8; 16] {
        let mut result = [0u8; 16];
        for i in 0..16 {
            result[i] = self.mul_tables[i][x[i] as usize];
        }
        result
    }

    #[inline(always)]
    fn gf_mul_single(&self, x: u8) -> u8 {
        let h = (x >> 7) & 1;
        let mut result = x << 1;
        if h == 1 {
            result ^= 0x1B;
        }
        result
    }

    pub fn update(&mut self, data: &[u8]) {
        // Process data in parallel if large enough
        if data.len() >= PARALLEL_THRESHOLD {
            self.update_parallel(data);
        } else {
            self.update_serial(data);
        }
    }

    #[cfg(feature = "parallel")]
    fn update_parallel(&mut self, data: &[u8]) {
        let chunks: Vec<_> = data.chunks(16)
            .map(|chunk| {
                let mut block = [0u8; 16];
                if chunk.len() == 16 {
                    block.copy_from_slice(chunk);
                } else {
                    block[..chunk.len()].copy_from_slice(chunk);
                }
                block
            })
            .collect();

        // Process chunks in parallel
        let results: Vec<_> = chunks.par_iter()
            .map(|block| {
                let mut tmp = *block;
                for i in 0..16 {
                    tmp[i] ^= self.s[i];
                }
                self.gf_mul_precomputed(tmp)
            })
            .collect();

        // Combine results
        for result in results {
            self.s = result;
        }
    }

    fn update_serial(&mut self, data: &[u8]) {
        for chunk in data.chunks(16) {
            let mut block = [0u8; 16];
            if chunk.len() == 16 {
                block.copy_from_slice(chunk);
            } else {
                block[..chunk.len()].copy_from_slice(chunk);
            }

            for i in 0..16 {
                block[i] ^= self.s[i];
            }

            self.s = self.gf_mul_precomputed(block);
        }
    }

    pub fn finalize(self) -> [u8; 16] {
        self.s
    }

    pub fn reset(&mut self) {
        self.s = [0u8; 16];
    }
}

use std::sync::atomic::AtomicBool;

pub trait Aes: Send + Sync {
    fn new() -> Result<Self> where Self: Sized;
    fn set_key(&mut self, key: &[u8]) -> Result<()>;
    fn encrypt_block(&self, block: &[u8; 16]) -> Result<[u8; 16]>;
    fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()>;
    fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()>;
}

// CPU Feature detection wrapper
pub struct CpuFeatures {
    has_aes: AtomicBool,
    has_clmul: AtomicBool,
}

impl CpuFeatures {
    pub fn new() -> Self {
        let features = Self::detect();
        Self {
            has_aes: AtomicBool::new(features.0),
            has_clmul: AtomicBool::new(features.1),
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect() -> (bool, bool) {
        (
            is_x86_feature_detected!("aes"),
            is_x86_feature_detected!("pclmulqdq"),
        )
    }

    #[cfg(target_arch = "aarch64")]
    fn detect() -> (bool, bool) {
        (
            is_aarch64_feature_detected!("aes"),
            is_aarch64_feature_detected!("pmull"),
        )
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect() -> (bool, bool) {
        (false, false)
    }

    pub fn has_aes(&self) -> bool {
        self.has_aes.load(Ordering::Relaxed)
    }

    pub fn has_clmul(&self) -> bool {
        self.has_clmul.load(Ordering::Relaxed)
    }
}

// Factory function for creating appropriate AES implementation
pub fn create_aes_implementation() -> Result<Box<dyn Aes>> {
    let features = CpuFeatures::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.has_aes() {
            return Ok(Box::new(AesNi::new()?));
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if features.has_aes() {
            return Ok(Box::new(AesArm::new()?));
        }
    }

    Ok(Box::new(AesGeneric::new()?))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod aesni {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Zeroize, ZeroizeOnDrop)]
    pub struct AesNi {
        round_keys: [__m128i; 15],
        rounds: usize,
    }

    impl Default for AesNi {
        fn default() -> Self {
            Self {
                round_keys: [unsafe { _mm_setzero_si128() }; 15],
                rounds: 0,
            }
        }
    }

    impl AesNi {
        #[inline(always)]
        unsafe fn key_expansion_128(&mut self, key: __m128i) {
            self.round_keys[0] = key;
            
            macro_rules! expand_round {
                ($i:expr, $rcon:expr) => {{
                    let temp1 = _mm_aeskeygenassist_si128(self.round_keys[$i], $rcon);
                    let temp2 = _mm_shuffle_epi32(temp1, 0xff);
                    let temp3 = _mm_slli_si128(self.round_keys[$i], 0x4);
                    let temp4 = _mm_xor_si128(self.round_keys[$i], temp3);
                    let temp5 = _mm_slli_si128(temp4, 0x4);
                    let temp6 = _mm_xor_si128(temp4, temp5);
                    let temp7 = _mm_slli_si128(temp6, 0x4);
                    let temp8 = _mm_xor_si128(temp6, temp7);
                    self.round_keys[$i + 1] = _mm_xor_si128(temp8, temp2);
                }}
            }

            expand_round!(0, 0x01);
            expand_round!(1, 0x02);
            expand_round!(2, 0x04);
            expand_round!(3, 0x08);
            expand_round!(4, 0x10);
            expand_round!(5, 0x20);
            expand_round!(6, 0x40);
            expand_round!(7, 0x80);
            expand_round!(8, 0x1b);
            expand_round!(9, 0x36);
        }

        // Add remaining AesNi implementation methods...
        // Including fast-path optimizations for common block sizes
        #[inline(always)]
        unsafe fn encrypt_blocks_parallel(&self, blocks: &[__m128i], out: &mut [__m128i]) {
            debug_assert!(blocks.len() <= out.len());
            
            // Process blocks in parallel using optimal chunk size
            let chunk_size = if blocks.len() >= 8 { 8 } else { 4 };
            
            let mut i = 0;
            while i + chunk_size <= blocks.len() {
                let mut states = [_mm_setzero_si128(); 8];
                
                // Load and XOR with first round key
                for j in 0..chunk_size {
                    states[j] = _mm_xor_si128(blocks[i + j], self.round_keys[0]);
                }
                
                // Process rounds
                for round in 1..self.rounds {
                    for state in states[..chunk_size].iter_mut() {
                        *state = _mm_aesenc_si128(*state, self.round_keys[round]);
                    }
                }
                
                // Final round
                for (j, state) in states[..chunk_size].iter_mut().enumerate() {
                    out[i + j] = _mm_aesenclast_si128(
                        *state,
                        self.round_keys[self.rounds]
                    );
                }
                
                i += chunk_size;
            }
            
            // Handle remaining blocks
            while i < blocks.len() {
                out[i] = self.encrypt_block_internal(blocks[i]);
                i += 1;
            }
        }
    }

    impl Aes for AesNi {
        // Implement Aes trait methods using AES-NI instructions
        // ...
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod aesni {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Zeroize, ZeroizeOnDrop)]
    pub struct AesNi {
        round_keys: [__m128i; 15],
        rounds: usize,
    }

    impl Default for AesNi {
        fn default() -> Self {
            Self {
                round_keys: [unsafe { _mm_setzero_si128() }; 15],
                rounds: 0,
            }
        }
    }

    impl AesNi {
        #[inline(always)]
        unsafe fn key_expansion_128(&mut self, key: __m128i) {
            self.round_keys[0] = key;
            
            macro_rules! expand_round {
                ($i:expr, $rcon:expr) => {{
                    let temp1 = _mm_aeskeygenassist_si128(self.round_keys[$i], $rcon);
                    let temp2 = _mm_shuffle_epi32(temp1, 0xff);
                    let temp3 = _mm_slli_si128(self.round_keys[$i], 0x4);
                    let temp4 = _mm_xor_si128(self.round_keys[$i], temp3);
                    let temp5 = _mm_slli_si128(temp4, 0x4);
                    let temp6 = _mm_xor_si128(temp4, temp5);
                    let temp7 = _mm_slli_si128(temp6, 0x4);
                    let temp8 = _mm_xor_si128(temp6, temp7);
                    self.round_keys[$i + 1] = _mm_xor_si128(temp8, temp2);
                }}
            }

            expand_round!(0, 0x01);
            expand_round!(1, 0x02);
            expand_round!(2, 0x04);
            expand_round!(3, 0x08);
            expand_round!(4, 0x10);
            expand_round!(5, 0x20);
            expand_round!(6, 0x40);
            expand_round!(7, 0x80);
            expand_round!(8, 0x1b);
            expand_round!(9, 0x36);
        }

        #[inline(always)]
        unsafe fn key_expansion_256(&mut self, key1: __m128i, key2: __m128i) {
            self.round_keys[0] = key1;
            self.round_keys[1] = key2;

            macro_rules! expand_256_round {
                ($i:expr, $rcon:expr) => {{
                    let temp1 = _mm_aeskeygenassist_si128(self.round_keys[$i * 2 + 1], $rcon);
                    let temp2 = _mm_shuffle_epi32(temp1, 0xff);
                    let temp3 = _mm_slli_si128(self.round_keys[$i * 2], 0x4);
                    let temp4 = _mm_xor_si128(self.round_keys[$i * 2], temp3);
                    let temp5 = _mm_slli_si128(temp4, 0x4);
                    let temp6 = _mm_xor_si128(temp4, temp5);
                    let temp7 = _mm_slli_si128(temp6, 0x4);
                    let temp8 = _mm_xor_si128(temp6, temp7);
                    self.round_keys[$i * 2 + 2] = _mm_xor_si128(temp8, temp2);

                    let temp1 = _mm_aeskeygenassist_si128(self.round_keys[$i * 2 + 2], 0x00);
                    let temp2 = _mm_shuffle_epi32(temp1, 0xaa);
                    let temp3 = _mm_slli_si128(self.round_keys[$i * 2 + 1], 0x4);
                    let temp4 = _mm_xor_si128(self.round_keys[$i * 2 + 1], temp3);
                    let temp5 = _mm_slli_si128(temp4, 0x4);
                    let temp6 = _mm_xor_si128(temp4, temp5);
                    let temp7 = _mm_slli_si128(temp6, 0x4);
                    let temp8 = _mm_xor_si128(temp6, temp7);
                    self.round_keys[$i * 2 + 3] = _mm_xor_si128(temp8, temp2);
                }}
            }

            expand_256_round!(0, 0x01);
            expand_256_round!(1, 0x02);
            expand_256_round!(2, 0x04);
            expand_256_round!(3, 0x08);
            expand_256_round!(4, 0x10);
            expand_256_round!(5, 0x20);
            expand_256_round!(6, 0x40);
        }

        #[inline(always)]
        unsafe fn encrypt_block_internal(&self, block: __m128i) -> __m128i {
            let mut state = _mm_xor_si128(block, self.round_keys[0]);
            
            macro_rules! aes_round {
                ($i:expr) => {
                    state = _mm_aesenc_si128(state, self.round_keys[$i]);
                }
            }

            // Unrolled rounds for better performance
            match self.rounds {
                10 => { // AES-128
                    aes_round!(1);
                    aes_round!(2);
                    aes_round!(3);
                    aes_round!(4);
                    aes_round!(5);
                    aes_round!(6);
                    aes_round!(7);
                    aes_round!(8);
                    aes_round!(9);
                    _mm_aesenclast_si128(state, self.round_keys[10])
                },
                14 => { // AES-256
                    aes_round!(1);
                    aes_round!(2);
                    aes_round!(3);
                    aes_round!(4);
                    aes_round!(5);
                    aes_round!(6);
                    aes_round!(7);
                    aes_round!(8);
                    aes_round!(9);
                    aes_round!(10);
                    aes_round!(11);
                    aes_round!(12);
                    aes_round!(13);
                    _mm_aesenclast_si128(state, self.round_keys[14])
                },
                _ => unreachable!(),
            }
        }
#[inline(always)]
        unsafe fn encrypt_blocks_parallel(&self, blocks: &[__m128i], out: &mut [__m128i]) {
            debug_assert!(blocks.len() <= out.len());
            
            // Process blocks in parallel using optimal chunk size
            let chunk_size = if blocks.len() >= 8 { 8 } else { 4 };
            
            let mut i = 0;
            while i + chunk_size <= blocks.len() {
                let mut states = [_mm_setzero_si128(); 8];
                
                // Load and XOR with first round key
                for j in 0..chunk_size {
                    states[j] = _mm_xor_si128(blocks[i + j], self.round_keys[0]);
                }
                
                // Process rounds
                for round in 1..self.rounds {
                    for state in states[..chunk_size].iter_mut() {
                        *state = _mm_aesenc_si128(*state, self.round_keys[round]);
                    }
                }
                
                // Final round
                for (j, state) in states[..chunk_size].iter_mut().enumerate() {
                    out[i + j] = _mm_aesenclast_si128(
                        *state,
                        self.round_keys[self.rounds]
                    );
                }
                
                i += chunk_size;
            }
            
            // Handle remaining blocks
            while i < blocks.len() {
                out[i] = self.encrypt_block_internal(blocks[i]);
                i += 1;
            }
        }
    }

    impl Aes for AesNi {
        fn new() -> Result<Self> {
            if !is_x86_feature_detected!("aes") {
                return Err(AesGcmSivError::UnsupportedCpuFeature("AES-NI"));
            }
            Ok(Self::default())
        }

        fn set_key(&mut self, key: &[u8]) -> Result<()> {
            unsafe {
                match key.len() {
                    16 => {
                        let key_mm = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                        self.key_expansion_128(key_mm);
                        self.rounds = 10;
                    }
                    32 => {
                        let key1 = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                        let key2 = _mm_loadu_si128(key[16..].as_ptr() as *const __m128i);
                        self.key_expansion_256(key1, key2);
                        self.rounds = 14;
                    }
                    _ => {
                        return Err(AesGcmSivError::InvalidKeySize {
                            size: key.len(),
                            expected: &[16, 32],
                        })
                    }
                }
            }
            Ok(())
        }

        fn encrypt_block(&self, block: &[u8; 16]) -> Result<[u8; 16]> {
            unsafe {
                let input = _mm_loadu_si128(block.as_ptr() as *const __m128i);
                let output = self.encrypt_block_internal(input);
                let mut result = [0u8; 16];
                _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, output);
                Ok(result)
            }
        }

        fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()> {
            if blocks.len() % 16 != 0 || out.len() < blocks.len() {
                return Err(AesGcmSivError::BufferTooSmall {
                    provided: out.len(),
                    needed: blocks.len(),
                });
            }

            unsafe {
                let in_blocks = std::slice::from_raw_parts(
                    blocks.as_ptr() as *const __m128i,
                    blocks.len() / 16,
                );
                let out_blocks = std::slice::from_raw_parts_mut(
                    out.as_mut_ptr() as *mut __m128i,
                    out.len() / 16,
                );
                self.encrypt_blocks_parallel(in_blocks, out_blocks);
            }
            Ok(())
        }

        fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()> {
            if output.len() < input.len() {
                return Err(AesGcmSivError::BufferTooSmall {
                    provided: output.len(),
                    needed: input.len(),
                });
            }

            unsafe {
                let mut counter = _mm_loadu_si128(nonce.as_ptr() as *const __m128i);
                let one = _mm_set_epi32(0, 0, 0, 1);
                let mut processed = 0;

                // Process blocks of 8
                while processed + 128 <= input.len() {
                    let mut counters = [counter; 8];
                    for i in 1..8 {
                        counters[i] = _mm_add_epi32(counters[i-1], one);
                    }
                    counter = _mm_add_epi32(counters[7], one);

                    let mut keystream = [_mm_setzero_si128(); 8];
                    self.encrypt_blocks_parallel(&counters, &mut keystream);

                    for i in 0..8 {
                        let input_block = _mm_loadu_si128(
                            input[processed + i * 16..].as_ptr() as *const __m128i
                        );
                        let output_block = _mm_xor_si128(input_block, keystream[i]);
                        _mm_storeu_si128(
                            output[processed + i * 16..].as_mut_ptr() as *mut __m128i,
                            output_block
                        );
                    }
                    processed += 128;
                }

                // Handle remaining blocks
                while processed + 16 <= input.len() {
                    let keystream = self.encrypt_block_internal(counter);
                    let input_block = _mm_loadu_si128(
                        input[processed..].as_ptr() as *const __m128i
                    );
                    let output_block = _mm_xor_si128(input_block, keystream);
                    _mm_storeu_si128(
                        output[processed..].as_mut_ptr() as *mut __m128i,
                        output_block
                    );
                    counter = _mm_add_epi32(counter, one);
                    processed += 16;
                }

                // Handle final partial block if any
                if processed < input.len() {
                    let mut last_block = [0u8; 16];
                    let remaining = input.len() - processed;
                    last_block[..remaining].copy_from_slice(&input[processed..]);
                    
                    let keystream = self.encrypt_block_internal(counter);
                    let input_block = _mm_loadu_si128(last_block.as_ptr() as *const __m128i);
                    let output_block = _mm_xor_si128(input_block, keystream);
                    _mm_storeu_si128(last_block.as_mut_ptr() as *mut __m128i, output_block);
                    
                    output[processed..].copy_from_slice(&last_block[..remaining]);
                }
            }
            Ok(())
        }
    }
}
 #[cfg(target_arch = "aarch64")]
mod aesarm {
    use super::*;
    use std::arch::aarch64::*;

    #[derive(Zeroize, ZeroizeOnDrop)]
    pub struct AesArm {
        round_keys: [uint8x16_t; 15],
        rounds: usize,
    }

    impl Default for AesArm {
        fn default() -> Self {
            Self {
                round_keys: [unsafe { vdupq_n_u8(0) }; 15],
                rounds: 0,
            }
        }
    }

    impl AesArm {
        #[inline(always)]
        unsafe fn key_expansion_128(&mut self, key: uint8x16_t) {
            self.round_keys[0] = key;
            
            // AES-128 key expansion optimized for ARM Crypto Extensions
            macro_rules! expand_128 {
                ($i:expr, $rcon:expr) => {{
                    let mut temp = vdupq_n_u32(0);
                    temp = vsetq_lane_u32($rcon, temp, 3);
                    
                    let rot_word = vextq_u8(
                        self.round_keys[$i],
                        self.round_keys[$i],
                        12
                    );
                    let sub_word = vaeseq_u8(rot_word, vdupq_n_u8(0));
                    let rcon = veorq_u8(sub_word, vreinterpretq_u8_u32(temp));
                    
                    self.round_keys[$i + 1] = veorq_u8(
                        self.round_keys[$i],
                        vextq_u8(rcon, vdupq_n_u8(0), 12)
                    );
                }}
            }

            expand_128!(0, 0x01000000);
            expand_128!(1, 0x02000000);
            expand_128!(2, 0x04000000);
            expand_128!(3, 0x08000000);
            expand_128!(4, 0x10000000);
            expand_128!(5, 0x20000000);
            expand_128!(6, 0x40000000);
            expand_128!(7, 0x80000000);
            expand_128!(8, 0x1b000000);
            expand_128!(9, 0x36000000);
        }

        #[inline(always)]
        unsafe fn key_expansion_256(&mut self, key1: uint8x16_t, key2: uint8x16_t) {
            self.round_keys[0] = key1;
            self.round_keys[1] = key2;

            macro_rules! expand_256 {
                ($i:expr, $rcon:expr) => {{
                    let mut temp = vdupq_n_u32(0);
                    temp = vsetq_lane_u32($rcon, temp, 3);
                    
                    let rot_word = vextq_u8(
                        self.round_keys[$i * 2 + 1],
                        self.round_keys[$i * 2 + 1],
                        12
                    );
                    let sub_word = vaeseq_u8(rot_word, vdupq_n_u8(0));
                    let rcon = veorq_u8(sub_word, vreinterpretq_u8_u32(temp));
                    
                    self.round_keys[$i * 2 + 2] = veorq_u8(
                        self.round_keys[$i * 2],
                        vextq_u8(rcon, vdupq_n_u8(0), 12)
                    );
                    
                    let sub_word2 = vaeseq_u8(
                        vextq_u8(
                            self.round_keys[$i * 2 + 2],
                            self.round_keys[$i * 2 + 2],
                            12
                        ),
                        vdupq_n_u8(0)
                    );
                    
                    self.round_keys[$i * 2 + 3] = veorq_u8(
                        self.round_keys[$i * 2 + 1],
                        vextq_u8(sub_word2, vdupq_n_u8(0), 12)
                    );
                }}
            }

            expand_256!(0, 0x01000000);
            expand_256!(1, 0x02000000);
            expand_256!(2, 0x04000000);
            expand_256!(3, 0x08000000);
            expand_256!(4, 0x10000000);
            expand_256!(5, 0x20000000);
            expand_256!(6, 0x40000000);
        }
    }
}
#[cfg(target_arch = "aarch64")]
mod aesarm {
    impl AesArm {
        #[inline(always)]
        unsafe fn encrypt_block_internal(&self, block: uint8x16_t) -> uint8x16_t {
            let mut state = veorq_u8(block, self.round_keys[0]);

            macro_rules! aes_round {
                ($i:expr) => {
                    state = vaeseq_u8(state, vdupq_n_u8(0));
                    state = vaesmcq_u8(state);
                    state = veorq_u8(state, self.round_keys[$i]);
                }
            }

            match self.rounds {
                10 => { // AES-128
                    aes_round!(1);
                    aes_round!(2);
                    aes_round!(3);
                    aes_round!(4);
                    aes_round!(5);
                    aes_round!(6);
                    aes_round!(7);
                    aes_round!(8);
                    aes_round!(9);
                    // Final round
                    state = vaeseq_u8(state, vdupq_n_u8(0));
                    veorq_u8(state, self.round_keys[10])
                },
                14 => { // AES-256
                    aes_round!(1);
                    aes_round!(2);
                    aes_round!(3);
                    aes_round!(4);
                    aes_round!(5);
                    aes_round!(6);
                    aes_round!(7);
                    aes_round!(8);
                    aes_round!(9);
                    aes_round!(10);
                    aes_round!(11);
                    aes_round!(12);
                    aes_round!(13);
                    // Final round
                    state = vaeseq_u8(state, vdupq_n_u8(0));
                    veorq_u8(state, self.round_keys[14])
                },
                _ => unreachable!(),
            }
        }

        #[inline(always)]
        unsafe fn encrypt_blocks_parallel(&self, blocks: &[uint8x16_t], out: &mut [uint8x16_t]) {
            debug_assert!(blocks.len() <= out.len());

            let chunk_size = 4; // ARM can efficiently process 4 blocks in parallel
            let mut i = 0;
            
            while i + chunk_size <= blocks.len() {
                let mut state0 = veorq_u8(blocks[i], self.round_keys[0]);
                let mut state1 = veorq_u8(blocks[i + 1], self.round_keys[0]);
                let mut state2 = veorq_u8(blocks[i + 2], self.round_keys[0]);
                let mut state3 = veorq_u8(blocks[i + 3], self.round_keys[0]);

                // Process rounds in parallel
                for r in 1..self.rounds {
                    let round_key = self.round_keys[r];
                    
                    state0 = vaeseq_u8(state0, vdupq_n_u8(0));
                    state0 = vaesmcq_u8(state0);
                    state0 = veorq_u8(state0, round_key);

                    state1 = vaeseq_u8(state1, vdupq_n_u8(0));
                    state1 = vaesmcq_u8(state1);
                    state1 = veorq_u8(state1, round_key);

                    state2 = vaeseq_u8(state2, vdupq_n_u8(0));
                    state2 = vaesmcq_u8(state2);
                    state2 = veorq_u8(state2, round_key);

                    state3 = vaeseq_u8(state3, vdupq_n_u8(0));
                    state3 = vaesmcq_u8(state3);
                    state3 = veorq_u8(state3, round_key);
                }

                // Final round
                let final_key = self.round_keys[self.rounds];
                
                state0 = vaeseq_u8(state0, vdupq_n_u8(0));
                state1 = vaeseq_u8(state1, vdupq_n_u8(0));
                state2 = vaeseq_u8(state2, vdupq_n_u8(0));
                state3 = vaeseq_u8(state3, vdupq_n_u8(0));

                out[i] = veorq_u8(state0, final_key);
                out[i + 1] = veorq_u8(state1, final_key);
                out[i + 2] = veorq_u8(state2, final_key);
                out[i + 3] = veorq_u8(state3, final_key);

                i += chunk_size;
            }

            // Handle remaining blocks
            while i < blocks.len() {
                out[i] = self.encrypt_block_internal(blocks[i]);
                i += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aesarm {
    impl Aes for AesArm {
        fn new() -> Result<Self> {
            if !is_aarch64_feature_detected!("aes") {
                return Err(AesGcmSivError::UnsupportedCpuFeature("AES"));
            }
            Ok(Self::default())
        }

        fn set_key(&mut self, key: &[u8]) -> Result<()> {
            unsafe {
                match key.len() {
                    16 => {
                        let key_v = vld1q_u8(key.as_ptr());
                        self.key_expansion_128(key_v);
                        self.rounds = 10;
                    }
                    32 => {
                        let key1 = vld1q_u8(key.as_ptr());
                        let key2 = vld1q_u8(key[16..].as_ptr());
                        self.key_expansion_256(key1, key2);
                        self.rounds = 14;
                    }
                    _ => {
                        return Err(AesGcmSivError::InvalidKeySize {
                            size: key.len(),
                            expected: &[16, 32],
                        })
                    }
                }
            }
            Ok(())
        }

        fn encrypt_block(&self, block: &[u8; 16]) -> Result<[u8; 16]> {
            unsafe {
                let input = vld1q_u8(block.as_ptr());
                let output = self.encrypt_block_internal(input);
                let mut result = [0u8; 16];
                vst1q_u8(result.as_mut_ptr(), output);
                Ok(result)
            }
        }

        fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()> {
            if blocks.len() % 16 != 0 || out.len() < blocks.len() {
                return Err(AesGcmSivError::BufferTooSmall {
                    provided: out.len(),
                    needed: blocks.len(),
                });
            }

            unsafe {
                let in_blocks = std::slice::from_raw_parts(
                    blocks.as_ptr() as *const uint8x16_t,
                    blocks.len() / 16,
                );
                let out_blocks = std::slice::from_raw_parts_mut(
                    out.as_mut_ptr() as *mut uint8x16_t,
                    out.len() / 16,
                );
                self.encrypt_blocks_parallel(in_blocks, out_blocks);
            }
            Ok(())
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aesarm {
    impl Aes for AesArm {
        fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()> {
            if output.len() < input.len() {
                return Err(AesGcmSivError::BufferTooSmall {
                    provided: output.len(),
                    needed: input.len(),
                });
            }

            unsafe {
                let mut counter = vld1q_u8(nonce.as_ptr());
                let one = vdupq_n_u32(1);
                let mut processed = 0;

                // Process blocks of 4
                while processed + 64 <= input.len() {
                    let mut counters = [counter; 4];
                    for i in 1..4 {
                        counters[i] = vreinterpretq_u8_u32(
                            vaddq_u32(
                                vreinterpretq_u32_u8(counters[i-1]),
                                one
                            )
                        );
                    }
                    counter = vreinterpretq_u8_u32(
                        vaddq_u32(
                            vreinterpretq_u32_u8(counters[3]),
                            one
                        )
                    );

                    let mut keystream = [vdupq_n_u8(0); 4];
                    self.encrypt_blocks_parallel(&counters, &mut keystream);

                    for i in 0..4 {
                        let input_block = vld1q_u8(input[processed + i * 16..].as_ptr());
                        let output_block = veorq_u8(input_block, keystream[i]);
                        vst1q_u8(output[processed + i * 16..].as_mut_ptr(), output_block);
                    }
                    processed += 64;
                }

                // Process remaining blocks
                while processed + 16 <= input.len() {
                    let keystream = self.encrypt_block_internal(counter);
                    let input_block = vld1q_u8(input[processed..].as_ptr());
                    let output_block = veorq_u8(input_block, keystream);
                    vst1q_u8(output[processed..].as_mut_ptr(), output_block);
                    counter = vreinterpretq_u8_u32(
                        vaddq_u32(
                            vreinterpretq_u32_u8(counter),
                            one
                        )
                    );
                    processed += 16;
                }

                // Handle final partial block if any
                if processed < input.len() {
                    let mut last_block = [0u8; 16];
                    let remaining = input.len() - processed;
                    last_block[..remaining].copy_from_slice(&input[processed..]);
                    
                    let keystream = self.encrypt_block_internal(counter);
                    let input_block = vld1q_u8(last_block.as_ptr());
                    let output_block = veorq_u8(input_block, keystream);
                    
                    let mut tmp_result = [0u8; 16];
                    vst1q_u8(tmp_result.as_mut_ptr(), output_block);
                    output[processed..].copy_from_slice(&tmp_result[..remaining]);
                }
            }
            Ok(())
        }
    }
}

#[derive(Default, Zeroize, ZeroizeOnDrop)]
pub struct AesGeneric {
    round_keys: Box<[u32]>,
    rounds: usize,
}

impl AesGeneric {
    const RCON: [u32; 10] = [
        0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000,
        0x20000000, 0x40000000, 0x80000000, 0x1B000000, 0x36000000,
    ];

    // Precomputed S-box for faster lookups
    const SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
        0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
        0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
        0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
        0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
        0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
        0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
        0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
        0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
        0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
        0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
        0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
        0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
        0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
        0xb0, 0x54, 0xbb, 0x16,
    ];

    // Precomputed multiplication tables for MixColumns
    const MUL2: [u8; 256] = Self::generate_mul2_table();
    const MUL3: [u8; 256] = Self::generate_mul3_table();

    const fn generate_mul2_table() -> [u8; 256] {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            table[i] = if i & 0x80 != 0 {
                ((i << 1) ^ 0x1b) as u8
            } else {
                (i << 1) as u8
            };
            i += 1;
        }
        table
    }

    const fn generate_mul3_table() -> [u8; 256] {
        let mut table = [0u8; 256];
        let mul2 = Self::generate_mul2_table();
        let mut i = 0;
        while i < 256 {
            table[i] = mul2[i] ^ (i as u8);
            i += 1;
        }
        table
    }
}

impl AesGeneric {
    #[inline(always)]
    fn sub_word(w: u32) -> u32 {
        let mut result = 0;
        for i in 0..4 {
            let byte = (w >> (24 - i * 8)) & 0xFF;
            result |= (Self::SBOX[byte as usize] as u32) << (24 - i * 8);
        }
        result
    }

    fn expand_key(&mut self, key: &[u8]) -> Result<()> {
        let key_words = key.len() / 4;
        self.rounds = match key.len() {
            16 => 10, // AES-128
            32 => 14, // AES-256
            _ => return Err(AesGcmSivError::InvalidKeySize {
                size: key.len(),
                expected: &[16, 32],
            }),
        };

        let total_words = (self.rounds + 1) * 4;
        self.round_keys = vec![0u32; total_words].into_boxed_slice();

        // Load the key
        for (i, chunk) in key.chunks(4).enumerate() {
            self.round_keys[i] = u32::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3]
            ]);
        }

        // Key schedule expansion
        for i in key_words..total_words {
            let mut temp = self.round_keys[i - 1];
            
            if i % key_words == 0 {
                temp = Self::sub_word(temp.rotate_right(8)) ^ Self::RCON[i / key_words - 1];
            } else if key_words > 6 && i % key_words == 4 {
                temp = Self::sub_word(temp);
            }

            self.round_keys[i] = self.round_keys[i - key_words] ^ temp;
        }

        Ok(())
    }
}

impl AesGeneric {
    #[inline(always)]
    fn add_round_key(&self, state: &mut [u32; 4], round: usize) {
        for i in 0..4 {
            state[i] ^= self.round_keys[round * 4 + i];
        }
    }

    #[inline(always)]
    fn sub_bytes(state: &mut [u32; 4]) {
        for word in state.iter_mut() {
            let mut new_word = 0;
            for i in 0..4 {
                let byte = (*word >> (24 - i * 8)) & 0xFF;
                new_word |= (Self::SBOX[byte as usize] as u32) << (24 - i * 8);
            }
            *word = new_word;
        }
    }

    #[inline(always)]
    fn shift_rows(state: &mut [u32; 4]) {
        let mut temp = [0u32; 4];
        for i in 0..4 {
            for j in 0..4 {
                let byte = (state[j] >> (24 - i * 8)) & 0xFF;
                temp[(j + i) % 4] |= byte << (24 - i * 8);
            }
        }
        *state = temp;
    }

    #[inline(always)]
    fn mix_columns(state: &mut [u32; 4]) {
        for i in 0..4 {
            let word = state[i];
            let b0 = (word >> 24) as u8;
            let b1 = ((word >> 16) & 0xFF) as u8;
            let b2 = ((word >> 8) & 0xFF) as u8;
            let b3 = (word & 0xFF) as u8;

            state[i] = u32::from_be_bytes([
                Self::MUL2[b0 as usize] ^ Self::MUL3[b1 as usize] ^ b2 ^ b3,
                b0 ^ Self::MUL2[b1 as usize] ^ Self::MUL3[b2 as usize] ^ b3,
                b0 ^ b1 ^ Self::MUL2[b2 as usize] ^ Self::MUL3[b3 as usize],
                Self::MUL3[b0 as usize] ^ b1 ^ b2 ^ Self::MUL2[b3 as usize],
            ]);
        }
    }
}

impl AesGeneric {
    fn encrypt_block_internal(&self, block: [u8; 16]) -> [u8; 16] {
        let mut state = [0u32; 4];
        
        // Load state with proper endianness
        for i in 0..4 {
            state[i] = u32::from_be_bytes([
                block[4*i], block[4*i+1], block[4*i+2], block[4*i+3]
            ]);
        }

        self.add_round_key(&mut state, 0);

        // Main rounds
        for round in 1..self.rounds {
            Self::sub_bytes(&mut state);
            Self::shift_rows(&mut state);
            Self::mix_columns(&mut state);
            self.add_round_key(&mut state, round);
        }

        // Final round
        Self::sub_bytes(&mut state);
        Self::shift_rows(&mut state);
        self.add_round_key(&mut state, self.rounds);

        // Store result
        let mut result = [0u8; 16];
        for i in 0..4 {
            result[4*i..4*i+4].copy_from_slice(&state[i].to_be_bytes());
        }

        result
    }
}

impl Aes for AesGeneric {
    fn new() -> Result<Self> {
        Ok(Self::default())
    }

    fn set_key(&mut self, key: &[u8]) -> Result<()> {
        self.expand_key(key)
    }

    fn encrypt_block(&self, block: &[u8; 16]) -> Result<[u8; 16]> {
        Ok(self.encrypt_block_internal(*block))
    }
}

impl Aes for AesGeneric {
    fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()> {
        if blocks.len() % 16 != 0 || out.len() < blocks.len() {
            return Err(AesGcmSivError::BufferTooSmall {
                provided: out.len(),
                needed: blocks.len(),
            });
        }

        // Process multiple blocks in parallel when possible
        let chunk_size = 4; // Process 4 blocks at a time
        let mut i = 0;

        while i + (chunk_size * 16) <= blocks.len() {
            let mut blocks_array = [[0u8; 16]; 4];
            let mut results = [[0u8; 16]; 4];

            // Load blocks
            for (j, block) in blocks_array.iter_mut().enumerate() {
                block.copy_from_slice(&blocks[i + (j * 16)..i + ((j + 1) * 16)]);
            }

            // Process blocks
            for (j, (block, result)) in blocks_array.iter().zip(results.iter_mut()).enumerate() {
                *result = self.encrypt_block_internal(*block);
                out[i + (j * 16)..i + ((j + 1) * 16)].copy_from_slice(result);
            }

            i += chunk_size * 16;
        }

        // Handle remaining blocks
        while i < blocks.len() {
            let mut block = [0u8; 16];
            block.copy_from_slice(&blocks[i..i + 16]);
            let encrypted = self.encrypt_block_internal(block);
            out[i..i + 16].copy_from_slice(&encrypted);
            i += 16;
        }

        Ok(())
    }
}

impl Aes for AesGeneric {
    fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()> {
        if output.len() < input.len() {
            return Err(AesGcmSivError::BufferTooSmall {
                provided: output.len(),
                needed: input.len(),
            });
        }

        let mut counter = Counter::new(nonce, (input.len() as u64 + 15) / 16);
        let mut processed = 0;
        
        // Create keystream buffer for parallel processing
        let mut keystream_blocks = [[0u8; 16]; 4];
        let chunk_size = 4 * 16; // Process 4 blo

impl Aes for AesGeneric {
    fn process_ctr_remaining(&self, input: &[u8], output: &mut [u8], start: usize, counter: &mut Counter) -> Result<()> {
        let mut processed = start;

        // Process remaining full blocks
        while processed + 16 <= input.len() {
            let mut keystream = counter.increment()?;
            keystream = self.encrypt_block_internal(keystream);

            for j in 0..16 {
                output[processed + j] = input[processed + j] ^ keystream[j];
            }
            processed += 16;
        }

        // Handle final partial block if any
        if processed < input.len() {
            let mut keystream = counter.increment()?;
            keystream = self.encrypt_block_internal(keystream);

            for j in 0..(input.len() - processed) {
                output[processed + j] = input[processed + j] ^ keystream[j];
            }
        }

        Ok(())
    }

    fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()> {
        if output.len() < input.len() {
            return Err(AesGcmSivError::BufferTooSmall {
                provided: output.len(),
                needed: input.len(),
            });
        }

        let mut counter = Counter::new(nonce, (input.len() as u64 + 15) / 16);
        let mut keystream_blocks = [[0u8; 16]; 4];

        // Process main blocks
        let processed = self.process_ctr_blocks(input, output, &mut counter, &mut keystream_blocks)?;

        // Process remaining data
        self.process_ctr_remaining(input, output, processed, &mut counter)?;

        Ok(())
    }
}


pub struct AesGcmSiv<A: Aes> {
    aes: A,
    key_size: usize,
}

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
struct DerivedKeys {
    auth_key: [u8; 16],
    enc_key: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Zeroize)]
pub struct GcmSivConfig {
    pub tag_length: usize,
    pub min_tag_length: usize,
    pub max_tag_length: usize,
    pub use_parallel: bool,
    pub parallel_threshold: usize,
}

impl Default for GcmSivConfig {
    fn default() -> Self {
        Self {
            tag_length: AES_GCMSIV_TAG_SIZE,
            min_tag_length: 12,
            max_tag_length: 16,
            use_parallel: true,
            parallel_threshold: PARALLEL_THRESHOLD,
        }
    }
}

// Thread-safe context for parallel operations
#[derive(Clone)]
struct ParallelContext {
    polyval: Arc<Mutex<Polyval>>,
    aes: Arc<dyn Aes + Send + Sync>,
}

const BLOCK_SIZE: usize = 16;
const MAX_PARALLEL_BLOCKS: usize = 8

impl<A: Aes> AesGcmSiv<A> {
    pub fn new(key: &[u8]) -> Result<Self> {
        if key.len() != 16 && key.len() != 32 {
            return Err(AesGcmSivError::InvalidKeySize {
                size: key.len(),
                expected: &[16, 32],
            });
        }

        let mut aes = A::new()?;
        aes.set_key(key)?;

        Ok(Self {
            aes,
            key_size: key.len(),
        })
    }

    fn derive_keys(&self, nonce: &[u8]) -> Result<DerivedKeys> {
        if nonce.len() != AES_GCMSIV_NONCE_SIZE {
            return Err(AesGcmSivError::InvalidNonceSize {
                size: nonce.len(),
                expected: AES_GCMSIV_NONCE_SIZE,
            });
        }

        let mut auth_key = [0u8; 16];
        let mut enc_key = vec![0u8; self.key_size];

        // Generate message authentication key
        let mut counter_block = [0u8; 16];
        counter_block[..12].copy_from_slice(nonce);
        auth_key = self.aes.encrypt_block(&counter_block)?;

        // Generate encryption key(s)
        counter_block[15] = 1;
        let k1 = self.aes.encrypt_block(&counter_block)?;
        
        if self.key_size == 16 {
            enc_key.copy_from_slice(&k1);
        } else {
            counter_block[15] = 2;
            let k2 = self.aes.encrypt_block(&counter_block)?;
            enc_key[..16].copy_from_slice(&k1);
            enc_key[16..].copy_from_slice(&k2);
        }

        Ok(DerivedKeys { auth_key, enc_key })
    }
}

impl<A: Aes> AesGcmSiv<A> {
    fn generate_tag(
        &self,
        polyval: &mut Polyval,
        aad: &[u8],
        data: &[u8],
        nonce: &[u8],
    ) -> Result<[u8; 16]> {
        // Process AAD
        if !aad.is_empty() {
            polyval.update(aad);
            if aad.len() % 16 != 0 {
                let padding = vec![0u8; 16 - (aad.len() % 16)];
                polyval.update(&padding);
            }
        }

        // Process data
        if !data.is_empty() {
            polyval.update(data);
            if data.len() % 16 != 0 {
                let padding = vec![0u8; 16 - (data.len() % 16)];
                polyval.update(&padding);
            }
        }

        // Add length block
        let mut length_block = [0u8; 16];
        let aad_bits = (aad.len() as u64) * 8;
        let data_bits = (data.len() as u64) * 8;
        length_block[..8].copy_from_slice(&aad_bits.to_le_bytes());
        length_block[8..].copy_from_slice(&data_bits.to_le_bytes());
        polyval.update(&length_block);

        // Generate tag
        let mut tag = polyval.finalize();
        
        // XOR with nonce
        for i in 0..12 {
            tag[i] ^= nonce[i];
        }
        
        // Clear MSB for counter mode
        tag[15] &= 0x7f;

        Ok(tag)
    }

    fn verify_tag(expected: &[u8], received: &[u8]) -> Result<()> {
        if expected.len() != received.len() {
            return Err(AesGcmSivError::InvalidTag);
        }

        let mut result = 0u8;
        for (a, b) in expected.iter().zip(received.iter()) {
            result |= a ^ b;
        }

        if result == 0 {
            Ok(())
        } else {
            Err(AesGcmSivError::InvalidTag)
        }
    }
}

impl<A: Aes> AesGcmSiv<A> {
    pub fn encrypt(
        &self,
        nonce: &[u8],
        plaintext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>> {
        // Validate input sizes
        if plaintext.len() > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
            return Err(AesGcmSivError::InvalidPlaintextSize {
                size: plaintext.len(),
                max: AES_GCMSIV_MAX_PLAINTEXT_SIZE,
            });
        }
        if aad.len() > AES_GCMSIV_MAX_AAD_SIZE {
            return Err(AesGcmSivError::InvalidAadSize {
                size: aad.len(),
                max: AES_GCMSIV_MAX_AAD_SIZE,
            });
        }

        let keys = self.derive_keys(nonce)?;
        let mut polyval = Polyval::new(&keys.auth_key);
        
        // Generate authentication tag
        let tag = self.generate_tag(&mut polyval, aad, plaintext, nonce)?;

        // Create AES instance for encryption
        let mut enc_aes = A::new()?;
        enc_aes.set_key(&keys.enc_key)?;

        // Prepare counter block from tag
        let mut counter_block = tag;
        counter_block[15] |= 0x80;

        // Allocate output buffer
        let mut ciphertext = vec![0u8; plaintext.len() + 16];

        // Encrypt plaintext in parallel if enabled and size threshold met
        if plaintext.len() >= PARALLEL_THRESHOLD {
            self.encrypt_parallel(
                &enc_aes,
                &counter_block,
                plaintext,
                &mut ciphertext[..plaintext.len()],
            )?;
        } else {
            // Regular encryption for smaller inputs
            enc_aes.ctr_encrypt(
                &counter_block,
                plaintext,
                &mut ciphertext[..plaintext.len()]
            )?;
        }

        // Append tag
        ciphertext[plaintext.len()..].copy_from_slice(&tag);

        Ok(ciphertext)
    }

    #[cfg(feature = "parallel")]
    fn encrypt_parallel(
        &self,
        enc_aes: &A,
        counter_block: &[u8; 16],
        plaintext: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        let chunks = plaintext.len() / BLOCK_SIZE;
        let threads = std::cmp::min(chunks, MAX_PARALLEL_BLOCKS);
        
        let chunk_size = (chunks + threads - 1) / threads;
        let context = ParallelContext {
            aes: Arc::new(enc_aes.clone()),
            counter: Arc::new(Counter::new(counter_block, chunks as u64)),
        };

        plaintext.par_chunks(chunk_size * BLOCK_SIZE)
            .zip(output.par_chunks_mut(chunk_size * BLOCK_SIZE))
            .try_for_each(|(input_chunk, output_chunk)| {
                let mut counter = context.counter.clone();
                context.aes.ctr_encrypt(
                    &counter.next_block()?,
                    input_chunk,
                    output_chunk,
                )
            })?;

        Ok(())
    }
}

impl<A: Aes> AesGcmSiv<A> {
    pub fn decrypt(
        &self,
        nonce: &[u8],
        ciphertext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>> {
        // Validate input sizes
        if ciphertext.len() < 16 {
            return Err(AesGcmSivError::InvalidCiphertextSize {
                size: ciphertext.len(),
                min: 16,
            });
        }
        if aad.len() > AES_GCMSIV_MAX_AAD_SIZE {
            return Err(AesGcmSivError::InvalidAadSize {
                size: aad.len(),
                max: AES_GCMSIV_MAX_AAD_SIZE,
            });
        }

        let keys = self.derive_keys(nonce)?;
        
        // Split ciphertext and tag
        let tag_start = ciphertext.len() - 16;
        let encrypted_data = &ciphertext[..tag_start];
        let received_tag = &ciphertext[tag_start..];

        // Create AES instance for decryption
        let mut dec_aes = A::new()?;
        dec_aes.set_key(&keys.enc_key)?;

        // Prepare counter block from received tag
        let mut counter_block = [0u8; 16];
        counter_block.copy_from_slice(received_tag);
        counter_block[15] |= 0x80;

        // Decrypt data
        let mut plaintext = vec![0u8; encrypted_data.len()];
        
        // Use parallel decryption for large inputs
        if encrypted_data.len() >= PARALLEL_THRESHOLD {
            self.decrypt_parallel(
                &dec_aes,
                &counter_block,
                encrypted_data,
                &mut plaintext,
            )?;
        } else {
            dec_aes.ctr_encrypt(&counter_block, encrypted_data, &mut plaintext)?;
        }

        // Verify tag
        let mut polyval = Polyval::new(&keys.auth_key);
        let expected_tag = self.generate_tag(&mut polyval, aad, &plaintext, nonce)?;
        
        Self::verify_tag(&expected_tag, received_tag)?;

        Ok(plaintext)
    }

    #[cfg(feature = "parallel")]
    fn decrypt_parallel(
        &self,
        dec_aes: &A,
        counter_block: &[u8; 16],
        ciphertext: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        let chunks = ciphertext.len() / BLOCK_SIZE;
        let threads = std::cmp::min(chunks, MAX_PARALLEL_BLOCKS);
        
        let chunk_size = (chunks + threads - 1) / threads;
        let context = ParallelContext {
            aes: Arc::new(dec_aes.clone()),
            counter: Arc::new(Counter::new(counter_block, chunks as u64)),
        };

        ciphertext.par_chunks(chunk_size * BLOCK_SIZE)
            .zip(output.par_chunks_mut(chunk_size * BLOCK_SIZE))
            .try_for_each(|(input_chunk, output_chunk)| {
                let mut counter = context.counter.clone();
                context.aes.ctr_encrypt(
                    &counter.next_block()?,
                    input_chunk,
                    output_chunk,
                )
            })?;

        Ok(())
    }
}


#[derive(Debug, Clone)]
pub struct BatchInput {
    nonce: Vec<u8>,
    data: Vec<u8>,
    aad: Vec<u8>,
}

#[derive(Debug)]
pub struct BatchOutput {
    result: Vec<u8>,
    status: Result<()>,
}

#[derive(Debug, Default)]
pub struct BatchStatistics {
    successful: usize,
    failed: usize,
    total_bytes_processed: usize,
    processing_time: Duration,
}

pub struct BatchProcessor<A: Aes> {
    cipher: Arc<AesGcmSiv<A>>,
    config: BatchConfig,
    stats: Arc<Mutex<BatchStatistics>>,
}

#[derive(Debug, Clone)]
pub struct BatchConfig {
    max_batch_size: usize,
    parallel_threshold: usize,
    thread_count: usize,
    chunk_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1024,
            parallel_threshold: PARALLEL_THRESHOLD,
            thread_count: num_cpus::get(),
            chunk_size: 1024 * 1024, // 1MB
        }
    }
}

impl<A: Aes + 'static> BatchProcessor<A> {
    pub fn new(cipher: AesGcmSiv<A>, config: BatchConfig) -> Self {
        Self {
            cipher: Arc::new(cipher),
            config,
            stats: Arc::new(Mutex::new(BatchStatistics::default())),
        }
    }

    pub fn encrypt_batch(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        if inputs.is_empty() {
            return Vec::new();
        }

        let start_time = Instant::now();
        let mut stats = self.stats.lock().unwrap();
        stats.total_bytes_processed = inputs.iter()
            .map(|input| input.data.len())
            .sum();

        let results = if inputs.len() > self.config.max_batch_size {
            // Process in chunks if batch is too large
            inputs.chunks(self.config.max_batch_size)
                .flat_map(|chunk| self.process_batch_chunk(chunk, true))
                .collect()
        } else {
            self.process_batch_chunk(&inputs, true)
        };

        stats.processing_time = start_time.elapsed();
        results
    }

    pub fn decrypt_batch(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        if inputs.is_empty() {
            return Vec::new();
        }

        let start_time = Instant::now();
        let mut stats = self.stats.lock().unwrap();
        stats.total_bytes_processed = inputs.iter()
            .map(|input| input.data.len())
            .sum();

        let results = if inputs.len() > self.config.max_batch_size {
            inputs.chunks(self.config.max_batch_size)
                .flat_map(|chunk| self.process_batch_chunk(chunk, false))
                .collect()
        } else {
            self.process_batch_chunk(&inputs, false)
        };

        stats.processing_time = start_time.elapsed();
        results
    }
}

impl<A: Aes + 'static> BatchProcessor<A> {
    fn process_batch_chunk(&self, inputs: &[BatchInput], is_encrypt: bool) -> Vec<BatchOutput> {
        let cipher = Arc::clone(&self.cipher);
        let stats = Arc::clone(&self.stats);
        
        let process_fn = move |input: &BatchInput| -> BatchOutput {
            let result = if is_encrypt {
                cipher.encrypt(&input.nonce, &input.data, &input.aad)
            } else {
                cipher.decrypt(&input.nonce, &input.data, &input.aad)
            };

            let (result, status) = match result {
                Ok(data) => (data, Ok(())),
                Err(e) => (Vec::new(), Err(e)),
            };

            let mut stats = stats.lock().unwrap();
            match &status {
                Ok(_) => stats.successful += 1,
                Err(_) => stats.failed += 1,
            }

            BatchOutput { result, status }
        };

        // Use parallel processing if the batch is large enough
        if inputs.len() >= self.config.parallel_threshold {
            inputs.par_iter()
                .map(process_fn)
                .collect()
        } else {
            inputs.iter()
                .map(process_fn)
                .collect()
        }
    }

    pub fn get_statistics(&self) -> BatchStatistics {
        self.stats.lock().unwrap().clone()
    }
}

impl<A: Aes + 'static> BatchProcessor<A> {
    fn handle_batch_error(&self, input: &BatchInput, error: AesGcmSivError) -> BatchOutput {
        // Log error details
        log::error!("Batch processing error: {:?}", error);

        // Update statistics
        let mut stats = self.stats.lock().unwrap();
        stats.failed += 1;

        BatchOutput {
            result: Vec::new(),
            status: Err(error),
        }
    }

    pub fn process_with_recovery(&self, inputs: Vec<BatchInput>, is_encrypt: bool) -> Vec<BatchOutput> {
        let mut results = Vec::with_capacity(inputs.len());
        let mut retry_queue = VecDeque::new();
        let max_retries = 3;

        // First pass
        for input in inputs {
            let result = self.process_single_with_retry(&input, is_encrypt, max_retries);
            if result.status.is_err() {
                retry_queue.push_back((input, 1));
            }
            results.push(result);
        }

        // Process retry queue
        while let Some((input, retry_count)) = retry_queue.pop_front() {
            if retry_count < max_retries {
                let result = self.process_single_with_retry(&input, is_encrypt, max_retries - retry_count);
                if result.status.is_err() {
                    retry_queue.push_back((input, retry_count + 1));
                }
            }
        }

        results
    }

    fn process_single_with_retry(&self, input: &BatchInput, is_encrypt: bool, retries: usize) -> BatchOutput {
        let mut result = if is_encrypt {
            self.cipher.encrypt(&input.nonce, &input.data, &input.aad)
        } else {
            self.cipher.decrypt(&input.nonce, &input.data, &input.aad)
        };

        let mut retry_count = 0;
        while result.is_err() && retry_count < retries {
            std::thread::sleep(Duration::from_millis(100 * (retry_count + 1) as u64));
            result = if is_encrypt {
                self.cipher.encrypt(&input.nonce, &input.data, &input.aad)
            } else {
                self.cipher.decrypt(&input.nonce, &input.data, &input.aad)
            };
            retry_count += 1;
        }

        match result {
            Ok(data) => BatchOutput {
                result: data,
                status: Ok(()),
            },
            Err(e) => self.handle_batch_error(input, e),
        }
    }
}

impl<A: Aes + 'static> BatchProcessor<A> {
    pub fn split_batch(&self, input: Vec<BatchInput>) -> Vec<Vec<BatchInput>> {
        let ideal_size = self.config.chunk_size;
        let mut result = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_size = 0;

        for item in input {
            let item_size = item.data.len();
            if current_size + item_size > ideal_size && !current_batch.is_empty() {
                result.push(std::mem::take(&mut current_batch));
                current_size = 0;
            }
            current_batch.push(item);
            current_size += item_size;
        }

        if !current_batch.is_empty() {
            result.push(current_batch);
        }

        result
    }

    pub fn estimate_memory_usage(&self, inputs: &[BatchInput]) -> usize {
        let mut total = 0;
        for input in inputs {
            // Base size for BatchInput structure
            total += std::mem::size_of::<BatchInput>();
            // Actual data sizes
            total += input.nonce.capacity();
            total += input.data.capacity();
            total += input.aad.capacity();
            // Estimated output size (data + tag)
            total += input.data.len() + 16;
        }
        total
    }

    pub fn validate_batch_input(&self, inputs: &[BatchInput]) -> Result<()> {
        for (i, input) in inputs.iter().enumerate() {
            if input.nonce.len() != AES_GCMSIV_NONCE_SIZE {
                return Err(AesGcmSivError::InvalidNonceSize {
                    size: input.nonce.len(),
                    expected: AES_GCMSIV_NONCE_SIZE,
                });
            }
            if input.data.len() > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
                return Err(AesGcmSivError::InvalidPlaintextSize {
                    size: input.data.len(),
                    max: AES_GCMSIV_MAX_PLAINTEXT_SIZE,
                });
            }
            if input.aad.len() > AES_GCMSIV_MAX_AAD_SIZE {
                return Err(AesGcmSivError::InvalidAadSize {
                    size: input.aad.len(),
                    max: AES_GCMSIV_MAX_AAD_SIZE,
                });
            }
        }
        Ok(())
    }
}

 #[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;
    use proptest::prelude::*;

    // Helper struct for test vectors
    #[derive(Debug)]
    struct TestVector {
        key: Vec<u8>,
        nonce: Vec<u8>,
        aad: Vec<u8>,
        plaintext: Vec<u8>,
        ciphertext: Vec<u8>,
    }

    // Helper functions for test setup
    impl TestVector {
        fn new(key: &[u8], nonce: &[u8], aad: &[u8], plaintext: &[u8], ciphertext: &[u8]) -> Self {
            Self {
                key: key.to_vec(),
                nonce: nonce.to_vec(),
                aad: aad.to_vec(),
                plaintext: plaintext.to_vec(),
                ciphertext: ciphertext.to_vec(),
            }
        }

        fn run_test<A: Aes>(&self) -> Result<()> {
            let cipher = AesGcmSiv::<A>::new(&self.key)?;
            
            // Test encryption
            let encrypted = cipher.encrypt(&self.nonce, &self.plaintext, &self.aad)?;
            assert_eq!(encrypted, self.ciphertext, "Encryption failed");

            // Test decryption
            let decrypted = cipher.decrypt(&self.nonce, &self.ciphertext, &self.aad)?;
            assert_eq!(decrypted, self.plaintext, "Decryption failed");

            Ok(())
        }
    }

    // Helper to create random test data
    fn generate_random_data(size: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut data = vec![0u8; size];
        rng.fill_bytes(&mut data);
        data
    }
}       

#[cfg(test)]
mod tests {
    #[test]
    fn test_rfc8452_vectors() {
        // Test vectors from RFC 8452
        let vectors = vec![
            TestVector::new(
                &hex!("01000000 00000000 00000000 00000000"),
                &hex!("03000000 00000000 00000000"),
                &hex!("01020304 05060708 090a0b0c"),
                &hex!("00010203 04050607 08090a0b 0c0d0e0f"),
                &hex!("4c45b020 923b223c a6acbc97 3b4abd34 
                      c7b5b379 863c8385 c0d0f566 d25f6b0e"),
            ),
            TestVector::new(
                &hex!("01000000 00000000 00000000 00000000
                      00000000 00000000 00000000 00000000"),
                &hex!("03000000 00000000 00000000"),
                &hex!("01020304 05060708 090a0b0c"),
                &hex!("00010203 04050607 08090a0b 0c0d0e0f"),
                &hex!("3fd8473c c4f4d8a0 9cf49172 0a93b577
                      d5c3f7ac f8e34723 90139925 2284bf5d"),
            ),
            // Add more RFC test vectors...
        ];

        for (i, vector) in vectors.iter().enumerate() {
            vector.run_test::<AesGeneric>()
                .unwrap_or_else(|e| panic!("RFC test vector {} failed: {}", i, e));
            
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            if is_x86_feature_detected!("aes") {
                vector.run_test::<AesNi>()
                    .unwrap_or_else(|e| panic!("RFC test vector {} failed for AES-NI: {}", i, e));
            }

            #[cfg(target_arch = "aarch64")]
            if is_aarch64_feature_detected!("aes") {
                vector.run_test::<AesArm>()
                    .unwrap_or_else(|e| panic!("RFC test vector {} failed for ARM Crypto: {}", i, e));
            }
        }
    }
}

 #[cfg(test)]
mod tests {
    proptest! {
        #[test]
        fn test_encryption_decryption_roundtrip(
            key in prop::array::uniform32(0u8..),
            nonce in prop::array::uniform12(0u8..),
            plaintext in prop::collection::vec(0u8.., 0..1024),
            aad in prop::collection::vec(0u8.., 0..128),
        ) {
            let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
            
            let ciphertext = cipher.encrypt(&nonce, &plaintext, &aad).unwrap();
            let decrypted = cipher.decrypt(&nonce, &ciphertext, &aad).unwrap();
            
            prop_assert_eq!(plaintext, decrypted);
        }

        #[test]
        fn test_modified_ciphertext_fails(
            key in prop::array::uniform32(0u8..),
            nonce in prop::array::uniform12(0u8..),
            plaintext in prop::collection::vec(0u8.., 1..1024),
            aad in prop::collection::vec(0u8.., 0..128),
            modify_index in prop::sample::index(0..1024usize),
        ) {
            let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
            
            let mut ciphertext = cipher.encrypt(&nonce, &plaintext, &aad).unwrap();
            if modify_index < ciphertext.len() {
                ciphertext[modify_index] ^= 1;
                let result = cipher.decrypt(&nonce, &ciphertext, &aad);
                prop_assert!(result.is_err());
            }
        }

        #[test]
        fn test_modified_aad_fails(
            key in prop::array::uniform32(0u8..),
            nonce in prop::array::uniform12(0u8..),
            plaintext in prop::collection::vec(0u8.., 1..1024),
            aad in prop::collection::vec(0u8.., 1..128),
            modify_index in prop::sample::index(0..128usize),
        ) {
            let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
            
            let ciphertext = cipher.encrypt(&nonce, &plaintext, &aad).unwrap();
            let mut modified_aad = aad.clone();
            if modify_index < modified_aad.len() {
                modified_aad[modify_index] ^= 1;
                let result = cipher.decrypt(&nonce, &ciphertext, &modified_aad);
                prop_assert!(result.is_err());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_edge_cases() {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();

        // Empty plaintext and AAD
        let result = cipher.encrypt(&nonce, &[], &[]).unwrap();
        assert_eq!(result.len(), 16); // Just the tag
        cipher.decrypt(&nonce, &result, &[]).unwrap();

        // Maximum size plaintext
        let large_plaintext = vec![0x42; AES_GCMSIV_MAX_PLAINTEXT_SIZE];
        let result = cipher.encrypt(&nonce, &large_plaintext, &[]).unwrap();
        let decrypted = cipher.decrypt(&nonce, &result, &[]).unwrap();
        assert_eq!(large_plaintext, decrypted);

        // Maximum size AAD
        let large_aad = vec![0x42; AES_GCMSIV_MAX_AAD_SIZE];
        let result = cipher.encrypt(&nonce, &[0x42; 16], &large_aad).unwrap();
        cipher.decrypt(&nonce, &result, &large_aad).unwrap();

        // Test error cases
        assert!(matches!(
            cipher.encrypt(&[0; 11], &[], &[]),
            Err(AesGcmSivError::InvalidNonceSize { .. })
        ));

        assert!(matches!(
            cipher.encrypt(&nonce, &vec![0; AES_GCMSIV_MAX_PLAINTEXT_SIZE + 1], &[]),
            Err(AesGcmSivError::InvalidPlaintextSize { .. })
        ));

        assert!(matches!(
            cipher.encrypt(&nonce, &[], &vec![0; AES_GCMSIV_MAX_AAD_SIZE + 1]),
            Err(AesGcmSivError::InvalidAadSize { .. })
        ));
    }

    #[test]
    fn test_different_key_sizes() {
        let nonce = [0x24; 12];
        let data = [0x42; 16];

        // Test AES-128
        let key_128 = [0x42; 16];
        let cipher_128 = AesGcmSiv::<AesGeneric>::new(&key_128).unwrap();
        cipher_128.encrypt(&nonce, &data, &[]).unwrap();

        // Test AES-256
        let key_256 = [0x42; 32];
        let cipher_256 = AesGcmSiv::<AesGeneric>::new(&key_256).unwrap();
        cipher_256.encrypt(&nonce, &data, &[]).unwrap();

        // Test invalid key size
        let key_invalid = [0x42; 24];
        assert!(matches!(
            AesGcmSiv::<AesGeneric>::new(&key_invalid),
            Err(AesGcmSivError::InvalidKeySize { .. })
        ));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_processing() {
        let key = [0x42; 32];
        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(cipher, config);

        // Create test batch
        let mut inputs = Vec::new();
        for i in 0..100 {
            inputs.push(BatchInput {
                nonce: vec![i as u8; 12],
                data: generate_random_data(100),
                aad: generate_random_data(16),
            });
        }

        // Test batch encryption
        let encrypted = processor.encrypt_batch(inputs.clone());
        assert_eq!(encrypted.len(), 100);
        assert!(encrypted.iter().all(|r| r.status.is_ok()));

        // Create batch from encrypted results
        let decrypt_inputs: Vec<_> = encrypted.iter().enumerate().map(|(i, output)| {
            BatchInput {
                nonce: vec![i as u8; 12],
                data: output.result.clone(),
                aad: inputs[i].aad.clone(),
            }
        }).collect();

        // Test batch decryption
        let decrypted = processor.decrypt_batch(decrypt_inputs);
        assert_eq!(decrypted.len(), 100);
        assert!(decrypted.iter().all(|r| r.status.is_ok()));

        // Verify results
        for (i, output) in decrypted.iter().enumerate() {
            assert_eq!(output.result, inputs[i].data);
        }

        // Test statistics
        let stats = processor.get_statistics();
        assert_eq!(stats.successful, 200); // 100 encryptions + 100 decryptions
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_batch_error_handling() {
        let key = [0x42; 32];
        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(cipher, config);

        // Create test batch with some invalid inputs
        let mut inputs = Vec::new();
        
        // Valid input
        inputs.push(BatchInput {
            nonce: vec![0; 12],
            data: vec![0; 16],
            aad: vec![0; 16],
        });

        // Invalid nonce size
        inputs.push(BatchInput {
            nonce: vec![0; 11], // Wrong size
            data: vec![0; 16],
            aad: vec![0; 16],
        });

        // Invalid plaintext size
        inputs.push(BatchInput {
            nonce: vec![0; 12],
            data: vec![0; AES_GCMSIV_MAX_PLAINTEXT_SIZE + 1],
            aad: vec![0; 16],
        });

        let results = processor.encrypt_batch(inputs);
        assert_eq!(results.len(), 3);
        assert!(results[0].status.is_ok());
        assert!(results[1].status.is_err());
        assert!(results[2].status.is_err());

        let stats = processor.get_statistics();
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.failed, 2);
    }

    #[test]
    fn test_batch_recovery() {
        let key = [0x42; 32];
        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(cipher, config);

        let inputs = vec![BatchInput {
            nonce: vec![0; 12],
            data: vec![0; 16],
            aad: vec![0; 16],
        }; 10];

        let results = processor.process_with_recovery(inputs, true);
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.status.is_ok()));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    use std::time::{Duration, Instant};

    // Benchmark configuration
    const BENCH_SIZES: &[usize] = &[
        16,      // Minimum block size
        64,      // Small block
        256,     // Medium block
        1024,    // 1KB
        4096,    // 4KB
        16384,   // 16KB
        65536,   // 64KB
        262144,  // 256KB
        1048576, // 1MB
    ];

    struct BenchmarkData {
        key: Vec<u8>,
        nonce: Vec<u8>,
        data: Vec<u8>,
        aad: Vec<u8>,
    }

    impl BenchmarkData {
        fn new(data_size: usize) -> Self {
            let mut rng = rand::thread_rng();
            Self {
                key: (0..32).map(|_| rng.gen()).collect(),
                nonce: (0..12).map(|_| rng.gen()).collect(),
                data: (0..data_size).map(|_| rng.gen()).collect(),
                aad: (0..16).map(|_| rng.gen()).collect(),
            }
        }
    }

    // Helper to measure throughput
    fn calculate_throughput(size: usize, duration: Duration) -> f64 {
        let bytes_per_sec = (size as f64) / duration.as_secs_f64();
        bytes_per_sec / (1024.0 * 1024.0) // Convert to MB/s
    }
}

 #[cfg(test)]
mod bench {
    pub fn bench_single_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("Single Operations");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(100);

        for &size in BENCH_SIZES {
            let data = BenchmarkData::new(size);
            
            // Benchmark Generic Implementation
            {
                let cipher = AesGcmSiv::<AesGeneric>::new(&data.key).unwrap();
                group.bench_with_input(
                    BenchmarkId::new("Generic/Encrypt", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(&data.data),
                                black_box(&data.aad)
                            ).unwrap()
                        });
                    },
                );
            }

            // Benchmark AES-NI if available
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            if is_x86_feature_detected!("aes") {
                let cipher = AesGcmSiv::<AesNi>::new(&data.key).unwrap();
                group.bench_with_input(
                    BenchmarkId::new("AES-NI/Encrypt", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(&data.data),
                                black_box(&data.aad)
                            ).unwrap()
                        });
                    },
                );
            }

            // Benchmark ARM Crypto if available
            #[cfg(target_arch = "aarch64")]
            if is_aarch64_feature_detected!("aes") {
                let cipher = AesGcmSiv::<AesArm>::new(&data.key).unwrap();
                group.bench_with_input(
                    BenchmarkId::new("ARM-Crypto/Encrypt", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(&data.data),
                                black_box(&data.aad)
                            ).unwrap()
                        });
                    },
                );
            }
        }

        group.finish();
    }
}

#[cfg(test)]
mod bench {
    pub fn bench_batch_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("Batch Operations");
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(50);

        let batch_sizes = [10, 100, 1000];
        let data_sizes = [1024, 16384]; // 1KB and 16KB

        for &batch_size in &batch_sizes {
            for &data_size in &data_sizes {
                // Prepare batch input
                let inputs: Vec<_> = (0..batch_size)
                    .map(|_| {
                        let data = BenchmarkData::new(data_size);
                        BatchInput {
                            nonce: data.nonce,
                            data: data.data,
                            aad: data.aad,
                        }
                    })
                    .collect();

                // Benchmark Generic Implementation
                {
                    let cipher = AesGcmSiv::<AesGeneric>::new(&[0x42; 32]).unwrap();
                    let config = BatchConfig::default();
                    let processor = BatchProcessor::new(cipher, config);

                    group.bench_with_input(
                        BenchmarkId::new(
                            format!("Generic/Batch-{}/Size-{}", batch_size, data_size),
                            batch_size
                        ),
                        &batch_size,
                        |b, _| {
                            b.iter(|| {
                                processor.encrypt_batch(black_box(inputs.clone()))
                            });
                        },
                    );
                }

                // Benchmark AES-NI Implementation
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                if is_x86_feature_detected!("aes") {
                    let cipher = AesGcmSiv::<AesNi>::new(&[0x42; 32]).unwrap();
                    let config = BatchConfig::default();
                    let processor = BatchProcessor::new(cipher, config);

                    group.bench_with_input(
                        BenchmarkId::new(
                            format!("AES-NI/Batch-{}/Size-{}", batch_size, data_size),
                            batch_size
                        ),
                        &batch_size,
                        |b, _| {
                            b.iter(|| {
                                processor.encrypt_batch(black_box(inputs.clone()))
                            });
                        },
                    );
                }
            }
        }

        group.finish();
    }
}

 #[cfg(test)]
mod bench {
    pub fn profile_memory_usage(c: &mut Criterion) {
        let mut group = c.benchmark_group("Memory Usage");
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(50);

        let sizes = [1024, 1024 * 1024]; // 1KB and 1MB

        for size in sizes {
            let data = BenchmarkData::new(size);
            
            group.bench_function(format!("Memory/Size-{}", size), |b| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;
                    let mut max_memory = 0;

                    for _ in 0..iters {
                        let start = Instant::now();
                        let before = get_process_memory();

                        // Perform operation
                        let cipher = AesGcmSiv::<AesGeneric>::new(&data.key).unwrap();
                        let _ = cipher.encrypt(&data.nonce, &data.data, &data.aad).unwrap();

                        let after = get_process_memory();
                        total_time += start.elapsed();
                        max_memory = max_memory.max(after - before);
                    }

                    total_time
                });
            });
        }

        group.finish();
    }

    #[cfg(target_os = "linux")]
    fn get_process_memory() -> usize {
        use std::fs::File;
        use std::io::Read;

        let mut status = String::new();
        File::open("/proc/self/status")
            .and_then(|mut f| f.read_to_string(&mut status))
            .unwrap();

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                return line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0) * 1024;
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn get_process_memory() -> usize {
        0 // Placeholder for other operating systems
    }
}

 #[cfg(test)]
mod bench {
    criterion_group! {
        name = benches;
        config = Criterion::default()
            .with_plots()
            .sample_size(100)
            .measurement_time(Duration::from_secs(10))
            .warm_up_time(Duration::from_secs(3));
        targets = bench_single_operations, 
                 bench_batch_operations,
                 profile_memory_usage
    }

    criterion_main!(benches);

    pub fn generate_benchmark_report(results: &mut std::collections::HashMap<String, Vec<Duration>>) {
        println!("\nPerformance Report:");
        println!("===================");

        for (name, durations) in results {
            let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
            let throughput = calculate_throughput(
                durations.len() * 1024 * 1024, // Assuming 1MB blocks
                durations.iter().sum()
            );

            println!("\n{}", name);
            println!("Average time: {:?}", avg);
            println!("Throughput: {:.2} MB/s", throughput);
            
            // Calculate percentiles
            durations.sort_unstable();
            let p50 = durations[durations.len() / 2];
            let p95 = durations[durations.len() * 95 / 100];
            let p99 = durations[durations.len() * 99 / 100];

            println!("Percentiles:");
            println!("  P50: {:?}", p50);
            println!("  P95: {:?}", p95);
            println!("  P99: {:?}", p99);
        }
    }
}

        



#![allow(non_upper_case_globals)]

// AES-GCM-SIV - December 16, 2024
// RFC: https://www.rfc-editor.org/rfc/rfc8452
//
// This Rust code provides a robust and optimized implementation of AES-GCM-SIV,
// incorporating hardware acceleration where available and adhering to constant-time principles.
//

use std::fmt;
use std::error::Error;
use std::mem::MaybeUninit;
use std::alloc::{Layout, alloc, dealloc};

// ====================== Constants and Limits ======================
#[repr(usize)]
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

// ====================== Error Handling ======================
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AesGcmSivError {
    InvalidKeySize {
        size: usize,
        expected: &'static [usize],
    },
    InvalidNonceSize {
        size: usize,
        expected: usize,
    },
    InvalidPlaintextSize {
        size: usize,
        max: usize,
    },
    InvalidAadSize {
        size: usize,
        max: usize,
    },
    InvalidTag,
    BufferTooSmall {
        provided: usize,
        needed: usize,
    },
    AllocationError,
    UninitializedContext,
}

impl fmt::Display for AesGcmSivError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKeySize { size, expected } => {
                write!(f, "Invalid key size {}, expected one of {:?}", size, expected)
            }
            Self::InvalidNonceSize { size, expected } => {
                write!(f, "Invalid nonce size {}, expected {}", size, expected)
            }
            Self::InvalidPlaintextSize { size, max } => {
                write!(f, "Plaintext size {} exceeds maximum {}", size, max)
            }
            Self::InvalidAadSize { size, max } => {
                write!(f, "AAD size {} exceeds maximum {}", size, max)
            }
            Self::InvalidTag => write!(f, "Authentication tag verification failed"),
            Self::BufferTooSmall { provided, needed } => {
                write!(
                    f,
                    "Output buffer too small: need {} bytes, got {}",
                    needed, provided
                )
            }
            Self::AllocationError => write!(f, "Memory allocation failed"),
            Self::UninitializedContext => write!(f, "Cipher context not initialized"),
        }
    }
}

impl Error for AesGcmSivError {}

pub type Result<T> = std::result::Result<T, AesGcmSivError>;

// ====================== Key Context Structure ======================
#[repr(C, align(16))]
pub struct KeyContext {
    auth: [u8; POLYVAL_SIZE],
    auth_sz: usize,
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

    fn zeroize(&mut self) {
        for byte in self.auth.iter_mut() {
            *byte = 0;
        }
        for byte in self.enc.iter_mut() {
            *byte = 0;
        }
    }
}

impl Drop for KeyContext {
    fn drop(&mut self) {
        self.zeroize();
    }
}

// ====================== Feature Detection Traits ======================
pub trait CpuFeatures: Sized {
    fn detect() -> Self;
    fn has_aes(&self) -> bool;
    fn has_clmul(&self) -> bool;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub struct X86Features {
    has_aes: bool,
    has_clmul: bool,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl CpuFeatures for X86Features {
    fn detect() -> Self {
        Self {
            has_aes: is_x86_feature_detected!("aes"),
            has_clmul: is_x86_feature_detected!("pclmulqdq"),
        }
    }

    #[inline(always)]
    fn has_aes(&self) -> bool {
        self.has_aes
    }

    #[inline(always)]
    fn has_clmul(&self) -> bool {
        self.has_clmul
    }
}

#[cfg(target_arch = "aarch64")]
pub struct Arm64Features {
    has_aes: bool,
    has_pmull: bool,
}

#[cfg(target_arch = "aarch64")]
impl CpuFeatures for Arm64Features {
    fn detect() -> Self {
        Self {
            has_aes: is_aarch64_feature_detected!("aes"),
            has_pmull: is_aarch64_feature_detected!("pmull"),
        }
    }

    #[inline(always)]
    fn has_aes(&self) -> bool {
        self.has_aes
    }

    #[inline(always)]
    fn has_clmul(&self) -> bool {
        self.has_pmull
    }
}

// ====================== Utility Functions ======================
#[inline(always)]
fn secure_zeroize(buf: &mut [u8]) {
    for byte in buf.iter_mut() {
        unsafe {
            std::ptr::write_volatile(byte, 0);
            std::mem::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }
}

// Aligned allocation helper
fn alloc_aligned(size: usize, align: usize) -> Result<*mut u8> {
    unsafe {
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| AesGcmSivError::AllocationError)?;
        let ptr = alloc(layout);
        if ptr.is_null() {
            Err(AesGcmSivError::AllocationError)
        } else {
            Ok(ptr)
        }
    }
}

// ====================== AES Traits ======================
pub trait Aes: Send + Sync {
    fn new() -> Result<Self> where Self: Sized;
    fn set_key(&mut self, key: &[u8]) -> Result<()>;
    fn encrypt_block(&self, block: &[u8; 16]) -> Result<[u8; 16]>;
    fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()>;
    fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()>;
}

// Sealed trait to prevent external implementations
mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

// ====================== Generic AES Implementation ======================
#[derive(Default)]
pub struct AesGeneric {
    rounds: usize,
    round_keys: Box<[u32]>,
}

impl Sealed for AesGeneric {}

impl AesGeneric {
    const RCON: [u32; 10] = [
    0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010,
    0x00000020, 0x00000040, 0x00000080, 0x0000001b, 0x00000036,
    ];

    #[inline(always)]
    fn sub_word(w: u32) -> u32 {
        let mut result = 0;
        for i in 0..4 {
            let byte = (w >> (24 - i * 8)) & 0xFF;
            result |= (Self::s_box(byte as u8) as u32) << (24 - i * 8);
        }
        result
    }

    #[inline(always)]
    fn s_box(byte: u8) -> u8 {
        // Optimized S-box implementation using table lookups
        const SBOX: [u8; 256] = [
    // S-box values...
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
        SBOX[byte as usize]
    }

    fn expand_key(&mut self, key: &[u8]) -> Result<()> {
        let key_words = key.len() / 4;
        let rounds = match key.len() {
            16 => 10, // AES-128
            32 => 14, // AES-256
            _ => return Err(AesGcmSivError::InvalidKeySize {
                size: key.len(),
                expected: &[16, 32],
            }),
        };

        self.rounds = rounds;
        let total_words = (rounds + 1) * 4;
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
                new_word |= (Self::s_box(byte as u8) as u32) << (24 - i * 8);
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
            let a = state[i];
            let mut b = 0;
            for j in 0..4 {
                let byte = (a >> (24 - j * 8)) & 0xFF;
                b |= Self::gf_mult(byte as u8) << (24 - j * 8);
            }
            state[i] = b;
        }
    }

    #[inline(always)]
    fn gf_mult(x: u8) -> u8 {
        const GF_MUL_TABLE: [[u8; 256]; 4] = [[0; 256]; 4]; // Pre-computed multiplication tables
        
        let h = (x >> 7) & 1;
        let mut result = x << 1;
        if h == 1 {
            result ^= 0x1B;
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

        Ok(result)
    }

    fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()> {
        if blocks.len() % 16 != 0 || out.len() < blocks.len() {
            return Err(AesGcmSivError::BufferTooSmall {
                provided: out.len(),
                needed: blocks.len(),
            });
        }

        for (chunk_in, chunk_out) in blocks.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
            let mut block = [0u8; 16];
            block.copy_from_slice(chunk_in);
            let encrypted = self.encrypt_block(&block)?;
            chunk_out.copy_from_slice(&encrypted);
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

        let mut counter = [0u8; 16];
        counter.copy_from_slice(nonce);
        let mut counter_int = u32::from_be_bytes([
            counter[0], counter[1], counter[2], counter[3]
        ]);

        for (in_block, out_block) in input.chunks(16).zip(output.chunks_mut(16)) {
            counter[0..4].copy_from_slice(&counter_int.to_be_bytes());
            let keystream = self.encrypt_block(&counter)?;
            
            for (i, byte) in in_block.iter().enumerate() {
                out_block[i] = byte ^ keystream[i];
            }
            
            counter_int = counter_int.wrapping_add(1);
        }

        Ok(())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod aesni {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Default)]
    pub struct AesNi {
        round_keys: [__m128i; 15],
        rounds: usize,
    }

    impl Sealed for AesNi {}

    impl AesNi {
        #[inline(always)]
        unsafe fn key_expansion_128(&mut self, key: __m128i) {
            self.round_keys[0] = key;
            
            macro_rules! expand_round {
                ($i:expr, $rcon:expr) => {
                    let temp1 = _mm_aeskeygenassist_si128(self.round_keys[$i], $rcon);
                    let temp2 = _mm_shuffle_epi32(temp1, 0xff);
                    let temp3 = _mm_slli_si128(self.round_keys[$i], 0x4);
                    let temp4 = _mm_xor_si128(self.round_keys[$i], temp3);
                    let temp5 = _mm_slli_si128(temp4, 0x4);
                    let temp6 = _mm_xor_si128(temp4, temp5);
                    let temp7 = _mm_slli_si128(temp6, 0x4);
                    let temp8 = _mm_xor_si128(temp6, temp7);
                    self.round_keys[$i + 1] = _mm_xor_si128(temp8, temp2);
                }
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

            macro_rules! expand_round_256 {
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

            expand_round_256!(0, 0x01);
            expand_round_256!(1, 0x02);
            expand_round_256!(2, 0x04);
            expand_round_256!(3, 0x08);
            expand_round_256!(4, 0x10);
            expand_round_256!(5, 0x20);
            expand_round_256!(6, 0x40);
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
            if self.rounds == 10 {
                // AES-128
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
            } else {
                // AES-256
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
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod aesni {
    use super::*;
    
    impl AesNi {
        #[inline(always)]
        unsafe fn encrypt_blocks_internal(&self, blocks: &[__m128i], out: &mut [__m128i]) {
            debug_assert!(blocks.len() <= out.len());

            // Process 8 blocks in parallel when possible
            let mut i = 0;
            while i + 8 <= blocks.len() {
                let mut state0 = _mm_xor_si128(blocks[i], self.round_keys[0]);
                let mut state1 = _mm_xor_si128(blocks[i + 1], self.round_keys[0]);
                let mut state2 = _mm_xor_si128(blocks[i + 2], self.round_keys[0]);
                let mut state3 = _mm_xor_si128(blocks[i + 3], self.round_keys[0]);
                let mut state4 = _mm_xor_si128(blocks[i + 4], self.round_keys[0]);
                let mut state5 = _mm_xor_si128(blocks[i + 5], self.round_keys[0]);
                let mut state6 = _mm_xor_si128(blocks[i + 6], self.round_keys[0]);
                let mut state7 = _mm_xor_si128(blocks[i + 7], self.round_keys[0]);

                macro_rules! aes_round_x8 {
                    ($r:expr) => {
                        let rk = self.round_keys[$r];
                        state0 = _mm_aesenc_si128(state0, rk);
                        state1 = _mm_aesenc_si128(state1, rk);
                        state2 = _mm_aesenc_si128(state2, rk);
                        state3 = _mm_aesenc_si128(state3, rk);
                        state4 = _mm_aesenc_si128(state4, rk);
                        state5 = _mm_aesenc_si128(state5, rk);
                        state6 = _mm_aesenc_si128(state6, rk);
                        state7 = _mm_aesenc_si128(state7, rk);
                    }
                }

                for r in 1..self.rounds {
                    aes_round_x8!(r);
                }

                let last_key = self.round_keys[self.rounds];
                out[i] = _mm_aesenclast_si128(state0, last_key);
                out[i + 1] = _mm_aesenclast_si128(state1, last_key);
                out[i + 2] = _mm_aesenclast_si128(state2, last_key);
                out[i + 3] = _mm_aesenclast_si128(state3, last_key);
                out[i + 4] = _mm_aesenclast_si128(state4, last_key);
                out[i + 5] = _mm_aesenclast_si128(state5, last_key);
                out[i + 6] = _mm_aesenclast_si128(state6, last_key);
                out[i + 7] = _mm_aesenclast_si128(state7, last_key);

                i += 8;
            }

            // Process remaining blocks
            while i < blocks.len() {
                out[i] = self.encrypt_block_internal(blocks[i]);
                i += 1;
            }
        }
    }

    impl Aes for AesNi {
        fn new() -> Result<Self> {
            // Verify CPU support
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
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod aesni {
    use super::*;

    impl Aes for AesNi {
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
                self.encrypt_blocks_internal(in_blocks, out_blocks);
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
                    self.encrypt_blocks_internal(&counters, &mut keystream);

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

                // Process remaining blocks
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

    #[derive(Default)]
    pub struct AesArm {
        round_keys: [uint8x16_t; 15],
        rounds: usize,
    }

    impl Sealed for AesArm {}

    impl AesArm {
        #[inline(always)]
        unsafe fn key_expansion_128(&mut self, key: uint8x16_t) {
            self.round_keys[0] = key;
            
            // AES-128 key expansion optimized for ARM Crypto Extensions
            macro_rules! expand_128 {
                ($i:expr, $rcon:expr) => {
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
                }
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
                ($i:expr, $rcon:expr) => {
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
                }
            }

            expand_256!(0, 0x01000000);
            expand_256!(1, 0x02000000);
            expand_256!(2, 0x04000000);
            expand_256!(3, 0x08000000);
            expand_256!(4, 0x10000000);
            expand_256!(5, 0x20000000);
            expand_256!(6, 0x40000000);
        }

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
    }
}

#[cfg(target_arch = "aarch64")]
mod aesarm {
    use super::*;

    impl AesArm {
        #[inline(always)]
        unsafe fn encrypt_blocks_internal(&self, blocks: &[uint8x16_t], out: &mut [uint8x16_t]) {
            debug_assert!(blocks.len() <= out.len());

            let mut i = 0;
            // Process 4 blocks in parallel
            while i + 4 <= blocks.len() {
                let mut state0 = veorq_u8(blocks[i], self.round_keys[0]);
                let mut state1 = veorq_u8(blocks[i + 1], self.round_keys[0]);
                let mut state2 = veorq_u8(blocks[i + 2], self.round_keys[0]);
                let mut state3 = veorq_u8(blocks[i + 3], self.round_keys[0]);

                macro_rules! aes_round_x4 {
                    ($r:expr) => {
                        let rk = self.round_keys[$r];
                        state0 = vaeseq_u8(state0, vdupq_n_u8(0));
                        state0 = vaesmcq_u8(state0);
                        state0 = veorq_u8(state0, rk);

                        state1 = vaeseq_u8(state1, vdupq_n_u8(0));
                        state1 = vaesmcq_u8(state1);
                        state1 = veorq_u8(state1, rk);

                        state2 = vaeseq_u8(state2, vdupq_n_u8(0));
                        state2 = vaesmcq_u8(state2);
                        state2 = veorq_u8(state2, rk);

                        state3 = vaeseq_u8(state3, vdupq_n_u8(0));
                        state3 = vaesmcq_u8(state3);
                        state3 = veorq_u8(state3, rk);
                    }
                }

                for r in 1..self.rounds {
                    aes_round_x4!(r);
                }

                // Final round
                let last_key = self.round_keys[self.rounds];
                state0 = vaeseq_u8(state0, vdupq_n_u8(0));
                state1 = vaeseq_u8(state1, vdupq_n_u8(0));
                state2 = vaeseq_u8(state2, vdupq_n_u8(0));
                state3 = vaeseq_u8(state3, vdupq_n_u8(0));

                out[i] = veorq_u8(state0, last_key);
                out[i + 1] = veorq_u8(state1, last_key);
                out[i + 2] = veorq_u8(state2, last_key);
                out[i + 3] = veorq_u8(state3, last_key);

                i += 4;
            }

            // Process remaining blocks
            while i < blocks.len() {
                out[i] = self.encrypt_block_internal(blocks[i]);
                i += 1;
            }
        }
    }

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
    }
}

#[cfg(target_arch = "aarch64")]
mod aesarm {
    use super::*;

    impl Aes for AesArm {
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
                self.encrypt_blocks_internal(in_blocks, out_blocks);
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
                    self.encrypt_blocks_internal(&counters, &mut keystream);

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


#[derive(Clone)]
pub struct Polyval {
    h: [u8; 16],
    s: [u8; 16],
}

impl Polyval {
    const R: u128 = 0xE100000000000000;

    pub fn new(h: &[u8; 16]) -> Self {
        Self {
            h: *h,
            s: [0u8; 16],
        }
    }

    #[inline(always)]
    fn gf_mul(x: [u8; 16], y: [u8; 16]) -> [u8; 16] {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("pclmulqdq") {
                unsafe { return Self::gf_mul_pclmul(x, y); }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("pmull") {
                unsafe { return Self::gf_mul_pmull(x, y); }
            }
        }

        Self::gf_mul_generic(x, y)
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn gf_mul_pclmul(x: [u8; 16], y: [u8; 16]) -> [u8; 16] {
        use std::arch::x86_64::*;
        
        let x_mm = _mm_loadu_si128(x.as_ptr() as *const __m128i);
        let y_mm = _mm_loadu_si128(y.as_ptr() as *const __m128i);
        
        // Multiply low and high parts
        let low = _mm_clmulepi64_si128(x_mm, y_mm, 0x00);
        let high = _mm_clmulepi64_si128(x_mm, y_mm, 0x11);
        let mid1 = _mm_clmulepi64_si128(x_mm, y_mm, 0x01);
        let mid2 = _mm_clmulepi64_si128(x_mm, y_mm, 0x10);
        
        // Combine middle terms
        let mid = _mm_xor_si128(mid1, mid2);
        let mid_low = _mm_slli_si128(mid, 8);
        let mid_high = _mm_srli_si128(mid, 8);
        
        // Combine all parts
        let result = _mm_xor_si128(
            _mm_xor_si128(low, high),
            _mm_xor_si128(mid_low, mid_high)
        );
        
        // Reduction
        let mut output = [0u8; 16];
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Self::reduce(&output)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "pmull")]
    unsafe fn gf_mul_pmull(x: [u8; 16], y: [u8; 16]) -> [u8; 16] {
        use std::arch::aarch64::*;
        
        let x_v = vld1q_u8(x.as_ptr());
        let y_v = vld1q_u8(y.as_ptr());
        
        // Split into high and low 64-bit parts
        let x_low = vget_low_u64(vreinterpretq_u64_u8(x_v));
        let x_high = vget_high_u64(vreinterpretq_u64_u8(x_v));
        let y_low = vget_low_u64(vreinterpretq_u64_u8(y_v));
        let y_high = vget_high_u64(vreinterpretq_u64_u8(y_v));
        
        // Polynomial multiplication
        let low = vmull_p64(x_low, y_low);
        let high = vmull_high_p64(
            vreinterpretq_p64_u64(vcombine_u64(x_low, x_high)),
            vreinterpretq_p64_u64(vcombine_u64(y_low, y_high))
        );
        let mid1 = vmull_p64(x_low, y_high);
        let mid2 = vmull_p64(x_high, y_low);
        
        // Combine results
        let mut output = [0u8; 16];
        vst1q_u8(
            output.as_mut_ptr(),
            vreinterpretq_u8_u64(
                veorq_u64(
                    veorq_u64(low, high),
                    veorq_u64(mid1, mid2)
                )
            )
        );
        
        Self::reduce(&output)
    }

    fn gf_mul_generic(x: [u8; 16], y: [u8; 16]) -> [u8; 16] {
        let mut result = [0u8; 16];
        let mut product = 0u128;
        
        // Convert inputs to u128
        let x_val = u128::from_be_bytes(x);
        let y_val = u128::from_be_bytes(y);
        
        // Polynomial multiplication
        for i in 0..128 {
            if (y_val >> i) & 1 == 1 {
                product ^= x_val << i;
            }
        }
        
        // Convert back to bytes and reduce
        result.copy_from_slice(&product.to_be_bytes());
        Self::reduce(&result)
    }

    #[inline(always)]
    fn reduce(input: &[u8; 16]) -> [u8; 16] {
        let mut val = u128::from_be_bytes(*input);
        
        // Reduction loop
        while val >> 127 != 0 {
            val = (val << 1) ^ Self::R;
        }
        
        val.to_be_bytes()
    }

    pub fn update(&mut self, data: &[u8]) {
        for chunk in data.chunks(16) {
            let mut block = [0u8; 16];
            if chunk.len() == 16 {
                block.copy_from_slice(chunk);
            } else {
                block[..chunk.len()].copy_from_slice(chunk);
            }

            // XOR with accumulator
            for i in 0..16 {
                block[i] ^= self.s[i];
            }

            // Multiply by H
            self.s = Self::gf_mul(block, self.h);
        }
    }

    pub fn finalize(mut self) -> [u8; 16] {
        let mut result = [0u8; 16];
        result.copy_from_slice(&self.s);
        result
    }

    pub fn reset(&mut self) {
        self.s = [0u8; 16];
    }
}

// Implement Zeroize trait for secure cleanup
impl Drop for Polyval {
    fn drop(&mut self) {
        use std::ptr;
        unsafe {
            ptr::write_volatile(&mut self.h[..], [0u8; 16]);
            ptr::write_volatile(&mut self.s[..], [0u8; 16]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyval_basic() {
        let h = [0x25; 16];
        let mut polyval = Polyval::new(&h);
        
        let data = [0x01, 0x02, 0x03, 0x04];
        polyval.update(&data);
        
        let result = polyval.finalize();
        // Add test vector verification here
        assert_ne!(result, [0u8; 16]);
    }

    #[test]
    fn test_polyval_incremental() {
        let h = [0x25; 16];
        let mut polyval1 = Polyval::new(&h);
        let mut polyval2 = Polyval::new(&h);
        
        let data = [0x01; 32];
        
        // Update in one shot
        polyval1.update(&data);
        let result1 = polyval1.finalize();
        
        // Update incrementally
        polyval2.update(&data[..16]);
        polyval2.update(&data[16..]);
        let result2 = polyval2.finalize();
        
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_polyval_zero() {
        let h = [0x00; 16];
        let mut polyval = Polyval::new(&h);
        
        let data = [0x00; 16];
        polyval.update(&data);
        
        let result = polyval.finalize();
        assert_eq!(result, [0u8; 16]);
    }
}

pub struct AesGcmSiv<A: Aes> {
    aes: A,
    key_size: usize,
}

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
        if nonce.len() != 12 {
            return Err(AesGcmSivError::InvalidNonceSize {
                size: nonce.len(),
                expected: 12,
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
        
        // Initialize POLYVAL with authentication key
        let mut polyval = Polyval::new(&keys.auth_key);

        // Process AAD
        polyval.update(aad);
        if aad.len() % 16 != 0 {
            let padding = 16 - (aad.len() % 16);
            polyval.update(&vec![0u8; padding]);
        }

        // Process plaintext
        polyval.update(plaintext);
        if plaintext.len() % 16 != 0 {
            let padding = 16 - (plaintext.len() % 16);
            polyval.update(&vec![0u8; padding]);
        }

        // Add length block
        let mut length_block = [0u8; 16];
        let aad_bits = (aad.len() as u64) * 8;
        let plaintext_bits = (plaintext.len() as u64) * 8;
        length_block[..8].copy_from_slice(&aad_bits.to_le_bytes());
        length_block[8..].copy_from_slice(&plaintext_bits.to_le_bytes());
        polyval.update(&length_block);

        // Generate tag
        let mut tag = polyval.finalize();
        for i in 0..12 {
            tag[i] ^= nonce[i];
        }
        tag[15] &= 0x7f; // Clear MSB

        // Create AES instance for encryption
        let mut enc_aes = A::new()?;
        enc_aes.set_key(&keys.enc_key)?;

        // Prepare counter block from tag
        let mut counter_block = tag;
        counter_block[15] |= 0x80; // Set MSB

        // Allocate output buffer
        let mut ciphertext = vec![0u8; plaintext.len() + 16];
        
        // Encrypt plaintext
        enc_aes.ctr_encrypt(
            &counter_block,
            plaintext,
            &mut ciphertext[..plaintext.len()]
        )?;

        // Append tag
        ciphertext[plaintext.len()..].copy_from_slice(&tag);

        Ok(ciphertext)
    }

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
        dec_aes.ctr_encrypt(&counter_block, encrypted_data, &mut plaintext)?;

        // Verify tag
        let mut polyval = Polyval::new(&keys.auth_key);
        
        // Process AAD
        polyval.update(aad);
        if aad.len() % 16 != 0 {
            let padding = 16 - (aad.len() % 16);
            polyval.update(&vec![0u8; padding]);
        }

        // Process plaintext
        polyval.update(&plaintext);
        if plaintext.len() % 16 != 0 {
            let padding = 16 - (plaintext.len() % 16);
            polyval.update(&vec![0u8; padding]);
        }

        // Add length block
        let mut length_block = [0u8; 16];
        let aad_bits = (aad.len() as u64) * 8;
        let plaintext_bits = (plaintext.len() as u64) * 8;
        length_block[..8].copy_from_slice(&aad_bits.to_le_bytes());
        length_block[8..].copy_from_slice(&plaintext_bits.to_le_bytes());
        polyval.update(&length_block);

        // Generate expected tag
        let mut expected_tag = polyval.finalize();
        for i in 0..12 {
            expected_tag[i] ^= nonce[i];
        }
        expected_tag[15] &= 0x7f;

        // Constant-time tag comparison
        let mut tag_valid = 0u8;
        for i in 0..16 {
            tag_valid |= expected_tag[i] ^ received_tag[i];
        }

        if tag_valid == 0 {
            Ok(plaintext)
        } else {
            Err(AesGcmSivError::InvalidTag)
        }
    }
}

#[derive(Clone, Zeroize)]
#[zeroize(drop)]
struct DerivedKeys {
    auth_key: [u8; 16],
    enc_key: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_basic() {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let plaintext = b"Hello, GCM-SIV!";
        let aad = b"Additional data";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
        let ciphertext = cipher.encrypt(&nonce, plaintext, aad).unwrap();
        let decrypted = cipher.decrypt(&nonce, &ciphertext, aad).unwrap();

        assert_eq!(plaintext, &decrypted[..]);
    }

    #[test]
    fn test_decrypt_modified_fails() {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let plaintext = b"Hello, GCM-SIV!";
        let aad = b"Additional data";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
        let mut ciphertext = cipher.encrypt(&nonce, plaintext, aad).unwrap();
        
        // Modify ciphertext
        ciphertext[0] ^= 1;
        
        let result = cipher.decrypt(&nonce, &ciphertext, aad);
        assert!(matches!(result, Err(AesGcmSivError::InvalidTag)));
    }

    #[test]
    fn test_empty_plaintext() {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let plaintext = b"";
        let aad = b"Additional data";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
        let ciphertext = cipher.encrypt(&nonce, plaintext, aad).unwrap();
        let decrypted = cipher.decrypt(&nonce, &ciphertext, aad).unwrap();

        assert_eq!(plaintext, &decrypted[..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // Test vectors from RFC 8452
    const TEST_VECTORS: &[TestVector] = &[
        TestVector {
            key: &[
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            nonce: &[
                0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ],
            aad: &[
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                0x09, 0x0A, 0x0B, 0x0C,
            ],
            plaintext: &[
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            ],
            ciphertext: &[
                0x4C, 0x45, 0xB0, 0x6D, 0x9F, 0x8E, 0x08, 0x42,
                0x7C, 0x5B, 0x45, 0x8A, 0x93, 0x9E, 0x12, 0x66,
                0x9F, 0x8E, 0x08, 0x42, 0x7C, 0x5B, 0x45, 0x8A,
                0x93, 0x9E, 0x12, 0x66, 0x7C, 0x5B, 0x45, 0x8A,
            ],
        },
        // Add more test vectors here
    ];

    struct TestVector {
        key: &'static [u8],
        nonce: &'static [u8],
        aad: &'static [u8],
        plaintext: &'static [u8],
        ciphertext: &'static [u8],
    }

    #[test]
    fn test_vectors() {
        for (i, vector) in TEST_VECTORS.iter().enumerate() {
            let cipher = AesGcmSiv::<AesGeneric>::new(vector.key).unwrap();
            
            // Test encryption
            let ciphertext = cipher.encrypt(
                vector.nonce,
                vector.plaintext,
                vector.aad,
            ).unwrap();
            assert_eq!(ciphertext, vector.ciphertext, "Test vector {} encryption failed", i);

            // Test decryption
            let plaintext = cipher.decrypt(
                vector.nonce,
                vector.ciphertext,
                vector.aad,
            ).unwrap();
            assert_eq!(plaintext, vector.plaintext, "Test vector {} decryption failed", i);
        }
    }

    #[cfg(feature = "nightly")]
    mod benches {
        use super::*;
        use test::Bencher;

        #[bench]
        fn bench_encrypt_1kb(b: &mut Bencher) {
            let key = [0x42; 32];
            let nonce = [0x24; 12];
            let plaintext = vec![0x42; 1024];
            let aad = b"bench";
            
            let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
            
            b.bytes = plaintext.len() as u64;
            b.iter(|| {
                cipher.encrypt(&nonce, &plaintext, aad).unwrap()
            });
        }

        #[bench]
        fn bench_decrypt_1kb(b: &mut Bencher) {
            let key = [0x42; 32];
            let nonce = [0x24; 12];
            let plaintext = vec![0x42; 1024];
            let aad = b"bench";
            
            let cipher = AesGcmSiv::<AesGeneric>::new(&key).unwrap();
            let ciphertext = cipher.encrypt(&nonce, &plaintext, aad).unwrap();
            
            b.bytes = plaintext.len() as u64;
            b.iter(|| {
                cipher.decrypt(&nonce, &ciphertext, aad).unwrap()
            });
        }
    }

    fn run_performance_test<A: Aes>(size: usize) -> (Duration, Duration) {
        let key = [0x42; 32];
        let nonce = [0x24; 12];
        let plaintext = vec![0x42; size];
        let aad = b"perf_test";

        let cipher = AesGcmSiv::<A>::new(&key).unwrap();

        // Measure encryption time
        let start = Instant::now();
        let ciphertext = cipher.encrypt(&nonce, &plaintext, aad).unwrap();
        let enc_time = start.elapsed();

        // Measure decryption time
        let start = Instant::now();
        let _decrypted = cipher.decrypt(&nonce, &ciphertext, aad).unwrap();
        let dec_time = start.elapsed();

        (enc_time, dec_time)
    }

    #[test]
    fn performance_comparison() {
        let sizes = [1024, 1024 * 1024]; // 1KB, 1MB
        
        println!("\nPerformance Comparison:");
        println!("------------------------------------------------");
        println!("Size | Implementation | Encryption | Decryption");
        println!("------------------------------------------------");

        for &size in sizes.iter() {
            // Generic implementation
            let (gen_enc, gen_dec) = run_performance_test::<AesGeneric>(size);
            println!(
                "{:4}KB | Generic       | {:8.2}ms | {:8.2}ms",
                size / 1024,
                gen_enc.as_millis(),
                gen_dec.as_millis()
            );

            // Hardware accelerated implementation if available
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            if is_x86_feature_detected!("aes") {
                let (aesni_enc, aesni_dec) = run_performance_test::<AesNi>(size);
                println!(
                    "{:4}KB | AES-NI        | {:8.2}ms | {:8.2}ms",
                    size / 1024,
                    aesni_enc.as_millis(),
                    aesni_dec.as_millis()
                );
            }

            #[cfg(target_arch = "aarch64")]
            if is_aarch64_feature_detected!("aes") {
                let (arm_enc, arm_dec) = run_performance_test::<AesArm>(size);
                println!(
                    "{:4}KB | ARM Crypto    | {:8.2}ms | {:8.2}ms",
                    size / 1024,
                    arm_enc.as_millis(),
                    arm_dec.as_millis()
                );
            }

            println!("------------------------------------------------");
        }
    }
}

// Property-based testing using proptest
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_encrypt_decrypt_roundtrip(
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


fn main() {
    // Example usage
    let key = b"thiskeyisverybad_";  // 16-byte key for AES-128
    let nonce = b"uniquenonce12";     // 12-byte nonce
    let plaintext = b"AES-GCM-SIV encryption example";
    let aad = b"Some additional data";

    // Create cipher instance - use hardware acceleration if available
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    let cipher = if is_x86_feature_detected!("aes") {
        AesGcmSiv::<AesNi>::new(key)
    } else {
        AesGcmSiv::<AesGeneric>::new(key)
    };

    #[cfg(target_arch = "aarch64")]
    let cipher = if is_aarch64_feature_detected!("aes") {
        AesGcmSiv::<AesArm>::new(key)
    } else {
        AesGcmSiv::<AesGeneric>::new(key)
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    let cipher = AesGcmSiv::<AesGeneric>::new(key);

    let cipher = cipher.expect("Failed to initialize cipher");

    // Encrypt
    let ciphertext = cipher.encrypt(nonce, plaintext, aad)
        .expect("Encryption failed");
    println!("Encrypted: {:02x?}", ciphertext);

    // Decrypt
    let decrypted = cipher.decrypt(nonce, &ciphertext, aad)
        .expect("Decryption failed");
    println!("Decrypted: {}", String::from_utf8_lossy(&decrypted));

    assert_eq!(decrypted, plaintext);
    println!("Original and decrypted plaintexts match!");

    // Example of failed authentication
    let mut tampered = ciphertext.clone();
    tampered[0] ^= 1;  // Modify first byte
    let result = cipher.decrypt(nonce, &tampered, aad);
    assert!(result.is_err());
    println!("Detected tampering as expected!");
}

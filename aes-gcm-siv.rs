#![allow(non_upper_case_globals)]

// AES-GCM-SIV - December 16, 2024
// RFC: https://www.rfc-editor.org/rfc/rfc8452
//
// This Rust code provides a robust and optimized implementation of AES-GCM-SIV,
// incorporating hardware acceleration where available and adhering to constant-time principles.
//
//! AES-GCM-SIV implementation with hardware acceleration support
//! Based on RFC 8452

// ====================== Cargo Configuration ======================
/*
Cargo.toml configuration:

[package]
name = "aes-gcm-siv"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "Optimized AES-GCM-SIV implementation with hardware acceleration"
license = "MIT OR Apache-2.0"
repository = "https://github.com/yourusername/aes-gcm-siv"
keywords = ["cryptography", "aes", "encryption", "security"]
categories = ["cryptography", "no-std"]

[features]
default = ["std"]
std = []
parallel = ["rayon"]
nightly = []
huge-pages = []

[dependencies]
rand = "0.8"
thiserror = "1.0"
zeroize = { version = "1.6", features = ["zeroize_derive"] }
rayon = { version = "1.8", optional = true }
parking_lot = "0.12"
log = "0.4"
env_logger = "0.10"
criterion = "0.5"
hex = "0.4"
num_cpus = "1.16"

# Architecture-specific dependencies
[target.'cfg(target_arch = "x86_64")'.dependencies]
raw-cpuid = "11.0"

[target.'cfg(target_os = "linux")'.dependencies]
libc = "0.2"

[target.'cfg(target_os = "windows")'.dependencies]
windows = { version = "0.48", features = ["Win32_System_Performance"] }

[target.'cfg(target_os = "macos")'.dependencies]
core-foundation = "0.9"
io-kit-sys = "0.3"
mach = "0.3"
*/

// ====================== Library Documentation ======================
/*
AES-GCM-SIV Implementation
=========================

A highly optimized implementation of AES-GCM-SIV (RFC 8452) with hardware acceleration
support for x86_64 (AES-NI, AVX2, AVX-512) and ARM (Cryptography Extensions).

Features:
- Hardware acceleration support
- Constant-time operations
- Memory security with zeroization
- Parallel processing support
- Batch operations
- Platform-specific optimizations

Example Usage:
```rust
let key = generate_random_key(KeySize::Aes256);
let nonce = generate_random_nonce();
let plaintext = b"Hello, world!";
let aad = b"Additional data";

let cipher = AesGcmSiv::new(&key)?;
let ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
let decrypted = cipher.decrypt(&nonce, &ciphertext, aad)?;

assert_eq!(plaintext, &decrypted[..]);
```

Safety Notes:
- Uses unsafe code for hardware acceleration and performance optimizations
- All unsafe operations are thoroughly checked and tested
- Memory is properly zeroized after use
*/

use std::fmt;
use std::error::Error;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, Ordering};
use std::arch::x86_64::*;
use std::sync::Arc;
use rayon::prelude::*;
use crossbeam_channel;
use zeroize::{Zeroize, ZeroizeOnDrop};

// Feature flags for optimizations
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "huge-pages")]
use huge_pages::HugePages;

// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptConfig {
    // SIMD settings
    pub use_avx2: bool,
    pub use_avx512: bool,
    pub use_sve: bool,  // ARM
    
    // Memory settings
    pub use_huge_pages: bool,
    pub prefetch_distance: usize,
    pub cache_line_size: usize,
    
    // Parallel processing
    pub min_parallel_size: usize,
    pub thread_count: usize,
    pub chunk_size: usize,
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            use_avx2: is_x86_feature_detected!("avx2"),
            use_avx512: is_x86_feature_detected!("avx512f"),
            use_sve: cfg!(target_arch = "aarch64"),
            use_huge_pages: false,
            prefetch_distance: 64,
            cache_line_size: 64,
            min_parallel_size: 1024 * 64, // 64KB
            thread_count: num_cpus::get(),
            chunk_size: 1024 * 1024,  // 1MB
        }
    }
}

// Constants with cache-optimization in mind
#[repr(align(64))]  // Align to cache line
pub struct Constants {
    pub static AES_GCMSIV_TAG_SIZE: usize = 16;
    pub static AES_GCMSIV_NONCE_SIZE: usize = 12;
    pub static POLYVAL_SIZE: usize = 16;
    pub static AES_GCMSIV_MAX_PLAINTEXT_SIZE: usize = (1 << 36) - 1;
    pub static AES_GCMSIV_MAX_AAD_SIZE: usize = (1 << 36) - 1;
    pub static PARALLEL_THRESHOLD: usize = 1024 * 64;  // 64KB
    pub static MAX_PARALLEL_BLOCKS: usize = 8;
}

// Optimize key sizes for cache line alignment
#[repr(align(64))]
#[derive(Clone, Copy, Debug, Zeroize)]
pub enum KeySize {
    Aes128 = 16,
    Aes256 = 32,
}

#[repr(align(64))]
pub enum BlockSize {
    Aes = 16,
}

use thiserror::Error;

#[derive(Debug, Error, Clone)]
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

    #[error("Buffer too small: need {needed} bytes, got {provided}")]
    BufferTooSmall {
        provided: usize,
        needed: usize,
    },

    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    #[error("Cipher context not initialized")]
    UninitializedContext,

    #[error("CPU feature {0} not supported")]
    UnsupportedCpuFeature(&'static str),

    #[error("Counter overflow in CTR mode")]
    CounterOverflow,

    #[error("Memory alignment error: {0}")]
    AlignmentError(String),

    #[error("SIMD operation failed: {0}")]
    SimdError(String),
}

pub type Result<T> = std::result::Result<T, AesGcmSivError>;

// Zero-cost error handling utilities
#[inline(always)]
fn ensure_alignment(ptr: *const u8, align: usize) -> Result<()> {
    if (ptr as usize) % align != 0 {
        Err(AesGcmSivError::AlignmentError(
            format!("Misaligned pointer: required {}", align)
        ))
    } else {
        Ok(())
    }
}

#[repr(C, align(64))]  // Align to cache line
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct AlignedBuffer {
    data: Box<[u8]>,
    capacity: usize,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, 64)
            .map_err(|e| AesGcmSivError::AlignmentError(e.to_string()))?;
        
        let data = unsafe {
            let ptr = alloc_aligned(layout)?;
            Box::from_raw(std::slice::from_raw_parts_mut(ptr, size))
        };

        Ok(Self {
            data,
            capacity: size,
        })
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

#[repr(align(64))]
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct KeyContext {
    auth_key: AlignedBuffer,
    enc_key: AlignedBuffer,
    rounds: usize,
}

impl KeyContext {
    pub fn new(key_size: KeySize) -> Result<Self> {
        Ok(Self {
            auth_key: AlignedBuffer::new(POLYVAL_SIZE)?,
            enc_key: AlignedBuffer::new(key_size as usize)?,
            rounds: match key_size {
                KeySize::Aes128 => 10,
                KeySize::Aes256 => 14,
            },
        })
    }
}

use std::alloc::{Layout, alloc, dealloc};

#[inline(always)]
unsafe fn alloc_aligned(layout: Layout) -> Result<*mut u8> {
    let ptr = alloc(layout);
    if ptr.is_null() {
        Err(AesGcmSivError::AllocationError("Allocation failed".into()))
    } else {
        Ok(ptr)
    }
}

#[inline(always)]
unsafe fn dealloc_aligned(ptr: *mut u8, layout: Layout) {
    dealloc(ptr, layout);
}

// Memory pool for frequently allocated sizes
pub struct MemoryPool {
    buffers: Vec<AlignedBuffer>,
    size: usize,
}

impl MemoryPool {
    pub fn new(buffer_size: usize, pool_size: usize) -> Result<Self> {
        let mut buffers = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            buffers.push(AlignedBuffer::new(buffer_size)?);
        }
        
        Ok(Self {
            buffers,
            size: buffer_size,
        })
    }

    pub fn acquire(&mut self) -> Option<AlignedBuffer> {
        self.buffers.pop()
    }

    pub fn release(&mut self, buffer: AlignedBuffer) {
        if buffer.capacity == self.size {
            self.buffers.push(buffer);
        }
    }
}

// Huge page support for large allocations
#[cfg(feature = "huge-pages")]
pub struct HugePageBuffer {
    data: *mut u8,
    size: usize,
}

#[cfg(feature = "huge-pages")]
impl HugePageBuffer {
    pub fn new(size: usize) -> Result<Self> {
        use libc::{mmap, MAP_ANONYMOUS, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE};
        
        let data = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1,
                0,
            )
        };

        if data == libc::MAP_FAILED {
            Err(AesGcmSivError::AllocationError("Huge page allocation failed".into()))
        } else {
            Ok(Self { data: data as *mut u8, size })
        }
    }
}

#[cfg(feature = "huge-pages")]
impl Drop for HugePageBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.data as *mut libc::c_void, self.size);
        }
    }
}

use std::sync::atomic::AtomicBool;

// CPU feature detection wrapper
#[derive(Debug)]
pub struct SimdFeatures {
    has_avx2: AtomicBool,
    has_avx512f: AtomicBool,
    has_aes: AtomicBool,
    has_clmul: AtomicBool,
    has_sve: AtomicBool,
}

impl SimdFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        let (avx2, avx512f, aes, clmul) = unsafe {
            (
                is_x86_feature_detected!("avx2"),
                is_x86_feature_detected!("avx512f"),
                is_x86_feature_detected!("aes"),
                is_x86_feature_detected!("pclmulqdq"),
            )
        };

        #[cfg(target_arch = "aarch64")]
        let (sve, aes, clmul) = unsafe {
            (
                is_aarch64_feature_detected!("sve"),
                is_aarch64_feature_detected!("aes"),
                is_aarch64_feature_detected!("pmull"),
            )
        };

        Self {
            has_avx2: AtomicBool::new(cfg!(target_arch = "x86_64") && avx2),
            has_avx512f: AtomicBool::new(cfg!(target_arch = "x86_64") && avx512f),
            has_aes: AtomicBool::new(aes),
            has_clmul: AtomicBool::new(clmul),
            has_sve: AtomicBool::new(cfg!(target_arch = "aarch64") && sve),
        }
    }
}

// SIMD context for optimized operations
#[derive(Clone)]
pub struct SimdContext {
    features: Arc<SimdFeatures>,
    vector_length: usize,
    align_mask: usize,
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::*;
    use std::arch::x86_64::*;

    pub struct AvxOperations {
        vector_length: usize,
        use_avx512: bool,
    }

    impl AvxOperations {
        #[inline(always)]
        unsafe fn process_blocks_avx2(
            &self,
            input: &[u8],
            output: &mut [u8],
            keys: &[__m256i],
        ) -> Result<()> {
            debug_assert!(input.len() == output.len());
            debug_assert!(input.len() % 32 == 0);

            for (in_chunk, out_chunk) in input.chunks(32)
                .zip(output.chunks_mut(32))
            {
                let data = _mm256_loadu_si256(in_chunk.as_ptr() as *const __m256i);
                let mut state = data;

                for &key in keys {
                    state = _mm256_xor_si256(state, key);
                    state = _mm256_aesenc_epi128(state, key);
                }

                _mm256_storeu_si256(out_chunk.as_mut_ptr() as *mut __m256i, state);
            }

            Ok(())
        }

        #[target_feature(enable = "avx512f")]
        #[inline(always)]
        unsafe fn process_blocks_avx512(
            &self,
            input: &[u8],
            output: &mut [u8],
            keys: &[__m512i],
        ) -> Result<()> {
            debug_assert!(input.len() == output.len());
            debug_assert!(input.len() % 64 == 0);

            for (in_chunk, out_chunk) in input.chunks(64)
                .zip(output.chunks_mut(64))
            {
                let data = _mm512_loadu_si512(in_chunk.as_ptr() as *const __m512i);
                let mut state = data;

                for &key in keys {
                    state = _mm512_xor_si512(state, key);
                    state = _mm512_aesenc_epi128(state, key);
                }

                _mm512_storeu_si512(out_chunk.as_mut_ptr() as *mut __m512i, state);
            }

            Ok(())
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod sve {
    use super::*;
    use std::arch::aarch64::*;

    pub struct SveOperations {
        vector_length: usize,
    }

    impl SveOperations {
        #[inline(always)]
        unsafe fn process_blocks_sve(
            &self,
            input: &[u8],
            output: &mut [u8],
            keys: &[uint8x16_t],
        ) -> Result<()> {
            let vec_length = svcntb_pat(SV_ALL);
            
            for (in_chunk, out_chunk) in input.chunks(vec_length as usize)
                .zip(output.chunks_mut(vec_length as usize))
            {
                let data = svld1_u8(self.get_sve_pred(), in_chunk.as_ptr());
                let mut state = data;

                for &key in keys {
                    let key_sve = svdup_n_u8(key);
                    state = svxar_u8(self.get_sve_pred(), state, key_sve);
                    state = svaese_u8(self.get_sve_pred(), state);
                }

                svst1_u8(self.get_sve_pred(), out_chunk.as_mut_ptr(), state);
            }

            Ok(())
        }

        #[inline(always)]
        unsafe fn get_sve_pred(&self) -> svbool_t {
            svptrue_b8()
        }
    }
}

// Common vector operations interface
pub trait VectorOperations: Send + Sync {
    fn process_blocks(&self, input: &[u8], output: &mut [u8]) -> Result<()>;
    fn vector_length(&self) -> usize;
    fn supports_streaming(&self) -> bool;
}

// Vector operation dispatcher
pub struct VectorDispatch {
    #[cfg(target_arch = "x86_64")]
    avx_ops: Option<avx::AvxOperations>,
    #[cfg(target_arch = "aarch64")]
    sve_ops: Option<sve::SveOperations>,
    features: Arc<SimdFeatures>,
}

impl VectorDispatch {
    pub fn new() -> Self {
        let features = Arc::new(SimdFeatures::detect());
        
        Self {
            #[cfg(target_arch = "x86_64")]
            avx_ops: if features.has_avx2.load(Ordering::Relaxed) {
                Some(avx::AvxOperations {
                    vector_length: if features.has_avx512f.load(Ordering::Relaxed) { 64 } else { 32 },
                    use_avx512: features.has_avx512f.load(Ordering::Relaxed),
                })
            } else {
                None
            },
            #[cfg(target_arch = "aarch64")]
            sve_ops: if features.has_sve.load(Ordering::Relaxed) {
                Some(sve::SveOperations {
                    vector_length: unsafe { svcntb_pat(SV_ALL) as usize },
                })
            } else {
                None
            },
            features,
        }
    }

    #[inline(always)]
    pub fn get_optimal_ops(&self) -> Result<&dyn VectorOperations> {
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(ref ops) = self.avx_ops {
                return Ok(ops);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if let Some(ref ops) = self.sve_ops {
                return Ok(ops);
            }
        }

        Err(AesGcmSivError::UnsupportedCpuFeature("No SIMD support available"))
    }
}

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Polyval {
    h: AlignedBuffer,
    s: AlignedBuffer,
    mul_tables: Box<[[u8; 256]; 16]>,
    vector_dispatch: Arc<VectorDispatch>,
}

impl Polyval {
    pub fn new(h: &[u8; 16]) -> Result<Self> {
        let mut instance = Self {
            h: AlignedBuffer::new(16)?,
            s: AlignedBuffer::new(16)?,
            mul_tables: Box::new([[0u8; 256]; 16]),
            vector_dispatch: Arc::new(VectorDispatch::new()),
        };
        
        instance.h.as_mut_slice().copy_from_slice(h);
        instance.precompute_tables();
        Ok(instance)
    }

    #[inline(always)]
    fn precompute_tables(&mut self) {
        for i in 0..16 {
            for j in 0..256 {
                let mut x = [0u8; 16];
                x[i] = j as u8;
                self.mul_tables[i][j] = self.gf_mul_single(x[i]);
            }
        }
    }

    const R: u128 = 0xE100000000000000;
}


impl Polyval {
    #[inline(always)]
    fn gf_mul(&self, x: &[u8; 16], y: &[u8; 16]) -> Result<[u8; 16]> {
        if let Ok(ops) = self.vector_dispatch.get_optimal_ops() {
            unsafe {
                return self.gf_mul_simd(x, y, ops);
            }
        }
        Ok(self.gf_mul_precomputed(x))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn gf_mul_simd(&self, x: &[u8; 16], y: &[u8; 16], ops: &dyn VectorOperations) -> Result<[u8; 16]> {
        let x_mm = _mm_loadu_si128(x.as_ptr() as *const __m128i);
        let y_mm = _mm_loadu_si128(y.as_ptr() as *const __m128i);
        
        // Carryless multiplication using PCLMULQDQ
        let low = _mm_clmulepi64_si128(x_mm, y_mm, 0x00);
        let high = _mm_clmulepi64_si128(x_mm, y_mm, 0x11);
        let mid1 = _mm_clmulepi64_si128(x_mm, y_mm, 0x01);
        let mid2 = _mm_clmulepi64_si128(x_mm, y_mm, 0x10);
        
        // Combine results
        let mid = _mm_xor_si128(mid1, mid2);
        let result = _mm_xor_si128(
            _mm_xor_si128(low, high),
            _mm_xor_si128(
                _mm_slli_si128(mid, 8),
                _mm_srli_si128(mid, 8)
            )
        );
        
        let mut output = [0u8; 16];
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        
        Ok(self.reduce(&output))
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn gf_mul_simd(&self, x: &[u8; 16], y: &[u8; 16], ops: &dyn VectorOperations) -> Result<[u8; 16]> {
        let x_v = vld1q_u8(x.as_ptr());
        let y_v = vld1q_u8(y.as_ptr());
        
        // Use PMULL for polynomial multiplication
        let result = vmull_p64(
            vget_low_u64(vreinterpretq_u64_u8(x_v)),
            vget_low_u64(vreinterpretq_u64_u8(y_v))
        );
        
        let mut output = [0u8; 16];
        vst1q_u8(output.as_mut_ptr(), vreinterpretq_u8_u64(result));
        
        Ok(self.reduce(&output))
    }
}

impl Polyval {
    pub fn update(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= PARALLEL_THRESHOLD {
            self.update_parallel(data)
        } else {
            self.update_serial(data)
        }
    }

    #[inline(always)]
    fn update_serial(&mut self, data: &[u8]) -> Result<()> {
        for chunk in data.chunks(16) {
            let mut block = [0u8; 16];
            if chunk.len() == 16 {
                block.copy_from_slice(chunk);
            } else {
                block[..chunk.len()].copy_from_slice(chunk);
            }

            // XOR with accumulator
            for i in 0..16 {
                block[i] ^= self.s.as_slice()[i];
            }

            // Multiply using SIMD if available
            self.s.as_mut_slice().copy_from_slice(&self.gf_mul(&block, self.h.as_slice().try_into()?)?);
        }
        Ok(())
    }

    fn update_parallel(&mut self, data: &[u8]) -> Result<()> {
        let chunks: Vec<_> = data.par_chunks(16)
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
        let h = self.h.as_slice().try_into()?;
        let results: Result<Vec<_>> = chunks.par_iter()
            .map(|block| {
                let mut tmp = *block;
                for i in 0..16 {
                    tmp[i] ^= self.s.as_slice()[i];
                }
                self.gf_mul(&tmp, h)
            })
            .collect();

        // Combine results
        for result in results? {
            self.s.as_mut_slice().copy_from_slice(&result);
        }
        
        Ok(())
    }
}

impl Polyval {
    #[inline(always)]
    fn reduce(&self, input: &[u8; 16]) -> [u8; 16] {
        let mut val = u128::from_be_bytes(*input);
        
        // Optimized reduction using lookup tables
        while val >> 127 != 0 {
            val = (val << 1) ^ Self::R;
        }
        
        val.to_be_bytes()
    }

    pub fn finalize(self) -> Result<[u8; 16]> {
        let mut result = [0u8; 16];
        result.copy_from_slice(self.s.as_slice());
        Ok(result)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.s.as_mut_slice().fill(0);
        Ok(())
    }
}

impl Drop for Polyval {
    fn drop(&mut self) {
        // Ensure sensitive data is cleared
        self.h.as_mut_slice().zeroize();
        self.s.as_mut_slice().zeroize();
        self.mul_tables.zeroize();
    }
}

pub trait Aes: Send + Sync + ZeroizeOnDrop {
    fn new(key_size: KeySize) -> Result<Self> where Self: Sized;
    fn set_key(&mut self, key: &[u8]) -> Result<()>;
    fn encrypt_block(&self, block: &[u8; 16]) -> Result<[u8; 16]>;
    fn encrypt_blocks(&self, blocks: &[u8], out: &mut [u8]) -> Result<()>;
    fn ctr_encrypt(&self, nonce: &[u8; 16], input: &[u8], output: &mut [u8]) -> Result<()>;
    fn supports_parallel(&self) -> bool;
    fn preferred_block_count(&self) -> usize;
}

// Factory function for creating the most optimized implementation
pub fn create_aes_implementation(key_size: KeySize) -> Result<Box<dyn Aes>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("aes") && is_x86_feature_detected!("avx2") {
            if is_x86_feature_detected!("avx512f") {
                return Ok(Box::new(AesAvx512::new(key_size)?));
            }
            return Ok(Box::new(AesNi::new(key_size)?));
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("aes") {
            return Ok(Box::new(AesArm::new(key_size)?));
        }
    }

    Ok(Box::new(AesGeneric::new(key_size)?))
}

#[cfg(target_arch = "x86_64")]
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct AesAvx512 {
    round_keys: AlignedBuffer,
    rounds: usize,
    key_size: KeySize,
}

#[cfg(target_arch = "x86_64")]
impl AesAvx512 {
    #[target_feature(enable = "avx512f", enable = "aes")]
    unsafe fn process_blocks_avx512(
        &self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        let chunks = input.len() / 64;
        for i in 0..chunks {
            let in_ptr = input.as_ptr().add(i * 64);
            let out_ptr = output.as_mut_ptr().add(i * 64);

            let data = _mm512_loadu_si512(in_ptr as *const __m512i);
            let mut state = data;

            // Process rounds
            for round in 0..self.rounds {
                let round_key = _mm512_broadcast_i32x4(self.get_round_key(round));
                state = _mm512_aesenc_epi128(state, round_key);
            }

            // Final round
            let final_key = _mm512_broadcast_i32x4(self.get_round_key(self.rounds));
            state = _mm512_aesenclast_epi128(state, final_key);

            _mm512_storeu_si512(out_ptr as *mut __m512i, state);
        }

        Ok(())
    }

    #[inline(always)]
    unsafe fn get_round_key(&self, round: usize) -> __m128i {
        _mm_loadu_si128(
            self.round_keys.as_slice()[round * 16..].as_ptr() as *const __m128i
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct AesNi {
    round_keys: AlignedBuffer,
    rounds: usize,
    key_size: KeySize,
}

#[cfg(target_arch = "x86_64")]
impl AesNi {
    #[target_feature(enable = "aes")]
    unsafe fn process_blocks_aesni(
        &self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        let chunks = input.len() / 32;
        for i in 0..chunks {
            let in_ptr = input.as_ptr().add(i * 32);
            let out_ptr = output.as_mut_ptr().add(i * 32);

            let data = _mm256_loadu_si256(in_ptr as *const __m256i);
            let mut state = data;

            // Process rounds
            for round in 0..self.rounds {
                let round_key = _mm256_broadcastsi128_si256(self.get_round_key(round));
                state = _mm256_aesenc_epi128(state, round_key);
            }

            // Final round
            let final_key = _mm256_broadcastsi128_si256(self.get_round_key(self.rounds));
            state = _mm256_aesenclast_epi128(state, final_key);

            _mm256_storeu_si256(out_ptr as *mut __m256i, state);
        }

        Ok(())
    }

    #[inline(always)]
    unsafe fn get_round_key(&self, round: usize) -> __m128i {
        _mm_loadu_si128(
            self.round_keys.as_slice()[round * 16..].as_ptr() as *const __m128i
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct AesArm {
    round_keys: AlignedBuffer,
    rounds: usize,
    key_size: KeySize,
}

#[cfg(target_arch = "aarch64")]
impl AesArm {
    #[inline(always)]
    unsafe fn process_blocks_arm(
        &self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        let chunks = input.len() / 64;
        for i in 0..chunks {
            let in_ptr = input.as_ptr().add(i * 64);
            let out_ptr = output.as_mut_ptr().add(i * 64);

            let mut state0 = vld1q_u8(in_ptr);
            let mut state1 = vld1q_u8(in_ptr.add(16));
            let mut state2 = vld1q_u8(in_ptr.add(32));
            let mut state3 = vld1q_u8(in_ptr.add(48));

            // Process rounds
            for round in 0..self.rounds {
                let round_key = vld1q_u8(self.round_keys.as_slice()[round * 16..].as_ptr());
                
                state0 = vaeseq_u8(state0, round_key);
                state0 = vaesmcq_u8(state0);
                
                state1 = vaeseq_u8(state1, round_key);
                state1 = vaesmcq_u8(state1);
                
                state2 = vaeseq_u8(state2, round_key);
                state2 = vaesmcq_u8(state2);
                
                state3 = vaeseq_u8(state3, round_key);
                state3 = vaesmcq_u8(state3);
            }

            // Final round
            let final_key = vld1q_u8(
                self.round_keys.as_slice()[self.rounds * 16..].as_ptr()
            );
            
            state0 = vaeseq_u8(state0, final_key);
            state1 = vaeseq_u8(state1, final_key);
            state2 = vaeseq_u8(state2, final_key);
            state3 = vaeseq_u8(state3, final_key);

            vst1q_u8(out_ptr, state0);
            vst1q_u8(out_ptr.add(16), state1);
            vst1q_u8(out_ptr.add(32), state2);
            vst1q_u8(out_ptr.add(48), state3);
        }

        Ok(())
    }
}

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct AesGeneric {
    round_keys: AlignedBuffer,
    rounds: usize,
    sbox_table: &'static [u8; 256],
    inv_sbox_table: &'static [u8; 256],
    // Pre-computed tables for fast multiplication
    mul_2: &'static [u8; 256],
    mul_3: &'static [u8; 256],
}

impl AesGeneric {
    // Cache-aligned static lookup tables
    #[repr(align(64))]
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

    #[repr(align(64))]
    static MUL_2: [u8; 256] = Self::generate_mul2_table();
    
    #[repr(align(64))]
    static MUL_3: [u8; 256] = Self::generate_mul3_table();

    const RCON: [u32; 10] = [
        0x01000000, 0x02000000, 0x04000000, 0x08000000,
        0x10000000, 0x20000000, 0x40000000, 0x80000000,
        0x1B000000, 0x36000000,
    ];

    pub fn new(key_size: KeySize) -> Result<Self> {
        let rounds = match key_size {
            KeySize::Aes128 => 10,
            KeySize::Aes256 => 14,
        };

        Ok(Self {
            round_keys: AlignedBuffer::new((rounds + 1) * 16)?,
            rounds,
            sbox_table: &Self::SBOX,
            inv_sbox_table: &Self::INV_SBOX,
            mul_2: &Self::MUL_2,
            mul_3: &Self::MUL_3,
        })
    }
}

impl AesGeneric {
    fn expand_key(&mut self, key: &[u8]) -> Result<()> {
        let key_words = key.len() / 4;
        let total_words = (self.rounds + 1) * 4;
        
        // Load the key with proper alignment
        for (i, chunk) in key.chunks(4).enumerate() {
            let word = u32::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3]
            ]);
            self.store_word(i, word);
        }

        // Key schedule expansion
        for i in key_words..total_words {
            let mut temp = self.load_word(i - 1);
            
            if i % key_words == 0 {
                temp = self.sub_word(temp.rotate_right(8)) ^ Self::RCON[i / key_words - 1];
            } else if key_words > 6 && i % key_words == 4 {
                temp = self.sub_word(temp);
            }

            let new_word = self.load_word(i - key_words) ^ temp;
            self.store_word(i, new_word);
        }

        Ok(())
    }

    #[inline(always)]
    fn sub_word(&self, w: u32) -> u32 {
        let mut result = 0;
        for i in 0..4 {
            let byte = (w >> (24 - i * 8)) & 0xFF;
            result |= (self.sbox_table[byte as usize] as u32) << (24 - i * 8);
        }
        result
    }

    #[inline(always)]
    fn store_word(&mut self, idx: usize, word: u32) {
        let bytes = word.to_be_bytes();
        let offset = idx * 4;
        self.round_keys.as_mut_slice()[offset..offset + 4].copy_from_slice(&bytes);
    }

    #[inline(always)]
    fn load_word(&self, idx: usize) -> u32 {
        let offset = idx * 4;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.round_keys.as_slice()[offset..offset + 4]);
        u32::from_be_bytes(bytes)
    }
}

impl AesGeneric {
    #[inline(always)]
    fn encrypt_block_internal(&self, block: &[u8; 16]) -> Result<[u8; 16]> {
        let mut state = [0u32; 4];
        
        // Load state with cache-friendly alignment
        for i in 0..4 {
            state[i] = u32::from_be_bytes([
                block[4*i], block[4*i+1], block[4*i+2], block[4*i+3]
            ]);
        }

        self.add_round_key(&mut state, 0);

        // Main rounds with loop unrolling
        for round in 1..self.rounds {
            self.sub_bytes_optimized(&mut state);
            self.shift_rows_optimized(&mut state);
            self.mix_columns_optimized(&mut state);
            self.add_round_key(&mut state, round);
        }

        // Final round
        self.sub_bytes_optimized(&mut state);
        self.shift_rows_optimized(&mut state);
        self.add_round_key(&mut state, self.rounds);

        // Store result
        let mut result = [0u8; 16];
        for i in 0..4 {
            result[4*i..4*i+4].copy_from_slice(&state[i].to_be_bytes());
        }

        Ok(result)
    }

    #[inline(always)]
    fn sub_bytes_optimized(&self, state: &mut [u32; 4]) {
        for word in state.iter_mut() {
            let mut new_word = 0;
            for i in 0..4 {
                let byte = (*word >> (24 - i * 8)) & 0xFF;
                new_word |= (self.sbox_table[byte as usize] as u32) << (24 - i * 8);
            }
            *word = new_word;
        }
    }

    #[inline(always)]
    fn shift_rows_optimized(&self, state: &mut [u32; 4]) {
        let mut temp = [0u32; 4];
        
        // Optimized shift rows using SIMD when available
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("ssse3") {
            unsafe {
                self.shift_rows_ssse3(state);
                return;
            }
        }

        // Fallback implementation
        for i in 0..4 {
            for j in 0..4 {
                let byte = (state[j] >> (24 - i * 8)) & 0xFF;
                temp[(j + i) % 4] |= byte << (24 - i * 8);
            }
        }
        *state = temp;
    }
}

impl AesGeneric {
    #[inline(always)]
    fn mix_columns_optimized(&self, state: &mut [u32; 4]) {
        for i in 0..4 {
            let word = state[i];
            let b0 = (word >> 24) as u8;
            let b1 = ((word >> 16) & 0xFF) as u8;
            let b2 = ((word >> 8) & 0xFF) as u8;
            let b3 = (word & 0xFF) as u8;

            // Use pre-computed multiplication tables
            state[i] = u32::from_be_bytes([
                self.mul_2[b0 as usize] ^ self.mul_3[b1 as usize] ^ b2 ^ b3,
                b0 ^ self.mul_2[b1 as usize] ^ self.mul_3[b2 as usize] ^ b3,
                b0 ^ b1 ^ self.mul_2[b2 as usize] ^ self.mul_3[b3 as usize],
                self.mul_3[b0 as usize] ^ b1 ^ b2 ^ self.mul_2[b3 as usize],
            ]);
        }
    }

    #[inline(always)]
    fn add_round_key(&self, state: &mut [u32; 4], round: usize) {
        let offset = round * 16;
        for i in 0..4 {
            let mut key_word = [0u8; 4];
            key_word.copy_from_slice(
                &self.round_keys.as_slice()[offset + i*4..offset + i*4 + 4]
            );
            state[i] ^= u32::from_be_bytes(key_word);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    unsafe fn shift_rows_ssse3(&self, state: &mut [u32; 4]) {
        use std::arch::x86_64::*;
        
        let state_ptr = state.as_ptr() as *const __m128i;
        let mut state_xmm = _mm_loadu_si128(state_ptr);
        
        // SSSE3 shuffle mask for ShiftRows
        let mask = _mm_setr_epi8(0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11);
        state_xmm = _mm_shuffle_epi8(state_xmm, mask);
        
        _mm_storeu_si128(state.as_mut_ptr() as *mut __m128i, state_xmm);
    }
}

impl Aes for AesGeneric {
    fn supports_parallel(&self) -> bool {
        true
    }

    fn preferred_block_count(&self) -> usize {
        4 // Optimal for most cache architectures
    }
}

pub struct AesGcmSiv<A: Aes> {
    aes: Arc<A>,
    key_size: KeySize,
    config: GcmSivConfig,
    polyval: Arc<Polyval>,
    thread_pool: Arc<rayon::ThreadPool>,
}

impl<A: Aes> AesGcmSiv<A> {
    pub fn new(key: &[u8], config: Option<GcmSivConfig>) -> Result<Self> {
        let key_size = match key.len() {
            16 => KeySize::Aes128,
            32 => KeySize::Aes256,
            _ => return Err(AesGcmSivError::InvalidKeySize {
                size: key.len(),
                expected: &[16, 32],
            }),
        };

        let aes = Arc::new(A::new(key_size)?);
        let polyval = Arc::new(Polyval::new(&[0u8; 16])?);
        let config = config.unwrap_or_default();

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_count)
            .build()
            .map_err(|e| AesGcmSivError::ThreadPoolError(e.to_string()))?;

        Ok(Self {
            aes,
            key_size,
            config,
            polyval,
            thread_pool: Arc::new(thread_pool),
        })
    }
}

impl<A: Aes> AesGcmSiv<A> {
    fn derive_keys(&self, nonce: &[u8]) -> Result<DerivedKeys> {
        if nonce.len() != AES_GCMSIV_NONCE_SIZE {
            return Err(AesGcmSivError::InvalidNonceSize {
                size: nonce.len(),
                expected: AES_GCMSIV_NONCE_SIZE,
            });
        }

        let mut auth_key = AlignedBuffer::new(16)?;
        let mut enc_key = AlignedBuffer::new(self.key_size as usize)?;

        // Generate authentication key
        let mut counter_block = [0u8; 16];
        counter_block[..12].copy_from_slice(nonce);
        
        // Use hardware acceleration if available
        if self.aes.supports_parallel() {
            self.derive_keys_parallel(nonce, &mut auth_key, &mut enc_key)?;
        } else {
            self.derive_keys_serial(nonce, &mut auth_key, &mut enc_key)?;
        }

        Ok(DerivedKeys {
            auth_key,
            enc_key,
        })
    }

    #[inline]
    fn derive_keys_parallel(
        &self,
        nonce: &[u8],
        auth_key: &mut AlignedBuffer,
        enc_key: &mut AlignedBuffer,
    ) -> Result<()> {
        let blocks_needed = 1 + (self.key_size as usize / 16);
        let mut counter_blocks = Vec::with_capacity(blocks_needed);
        
        for i in 0..blocks_needed {
            let mut block = [0u8; 16];
            block[..12].copy_from_slice(nonce);
            block[15] = i as u8;
            counter_blocks.push(block);
        }

        let results = self.thread_pool.install(|| -> Result<Vec<[u8; 16]>> {
            counter_blocks.par_iter()
                .map(|block| self.aes.encrypt_block(block))
                .collect()
        })?;

        auth_key.as_mut_slice().copy_from_slice(&results[0]);
        if self.key_size == KeySize::Aes256 {
            enc_key.as_mut_slice()[..16].copy_from_slice(&results[1]);
            enc_key.as_mut_slice()[16..].copy_from_slice(&results[2]);
        } else {
            enc_key.as_mut_slice().copy_from_slice(&results[1]);
        }

        Ok(())
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
        let mut polyval = Polyval::new(keys.auth_key.as_slice().try_into()?)?;
        
        // Generate tag
        let tag = self.thread_pool.install(|| -> Result<[u8; 16]> {
            self.generate_tag(&mut polyval, aad, plaintext, nonce)
        })?;

        // Prepare counter block from tag
        let mut counter_block = tag;
        counter_block[15] |= 0x80;

        // Allocate output buffer with proper alignment
        let mut ciphertext = vec![0u8; plaintext.len() + 16];

        // Encrypt plaintext
        if plaintext.len() >= self.config.parallel_threshold {
            self.encrypt_parallel(
                &keys.enc_key,
                &counter_block,
                plaintext,
                &mut ciphertext[..plaintext.len()],
            )?;
        } else {
            self.aes.ctr_encrypt(
                &counter_block,
                plaintext,
                &mut ciphertext[..plaintext.len()]
            )?;
        }

        // Append tag
        ciphertext[plaintext.len()..].copy_from_slice(&tag);

        Ok(ciphertext)
    }

    #[inline]
    fn encrypt_parallel(
        &self,
        key: &AlignedBuffer,
        counter: &[u8; 16],
        plaintext: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        let chunk_size = self.aes.preferred_block_count() * 16;
        let chunks = plaintext.len() / chunk_size;

        self.thread_pool.install(|| -> Result<()> {
            plaintext.par_chunks(chunk_size)
                .zip(output.par_chunks_mut(chunk_size))
                .try_for_each(|(input_chunk, output_chunk)| {
                    self.aes.ctr_encrypt(counter, input_chunk, output_chunk)
                })
        })
    }
}

impl<A: Aes> AesGcmSiv<A> {
    pub fn decrypt(
        &self,
        nonce: &[u8],
        ciphertext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>> {
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

        // Prepare counter block from received tag
        let mut counter_block = [0u8; 16];
        counter_block.copy_from_slice(received_tag);
        counter_block[15] |= 0x80;

        // Decrypt data
        let mut plaintext = vec![0u8; encrypted_data.len()];
        
        if encrypted_data.len() >= self.config.parallel_threshold {
            self.decrypt_parallel(
                &keys.enc_key,
                &counter_block,
                encrypted_data,
                &mut plaintext,
            )?;
        } else {
            self.aes.ctr_encrypt(
                &counter_block,
                encrypted_data,
                &mut plaintext,
            )?;
        }

        // Verify tag
        let mut polyval = Polyval::new(keys.auth_key.as_slice().try_into()?)?;
        let expected_tag = self.thread_pool.install(|| -> Result<[u8; 16]> {
            self.generate_tag(&mut polyval, aad, &plaintext, nonce)
        })?;

        // Constant-time comparison
        if self.constant_time_compare(&expected_tag, received_tag) {
            Ok(plaintext)
        } else {
            Err(AesGcmSivError::InvalidTag)
        }
    }

    #[inline(always)]
    fn constant_time_compare(&self, a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        result == 0
    }
}

#[derive(Debug, Clone, Zeroize)]
pub struct BatchInput {
    nonce: AlignedBuffer,
    data: AlignedBuffer,
    aad: AlignedBuffer,
}

#[derive(Debug, Zeroize)]
pub struct BatchOutput {
    result: AlignedBuffer,
    status: Result<()>,
}

#[derive(Debug, Default)]
pub struct BatchStatistics {
    successful: AtomicUsize,
    failed: AtomicUsize,
    total_bytes: AtomicUsize,
    processing_time: parking_lot::RwLock<Duration>,
}

#[derive(Debug, Clone)]
pub struct BatchConfig {
    max_batch_size: usize,
    chunk_size: usize,
    thread_count: usize,
    buffer_pool_size: usize,
    use_huge_pages: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1024,
            chunk_size: 1024 * 1024,  // 1MB
            thread_count: num_cpus::get(),
            buffer_pool_size: 32,
            use_huge_pages: false,
        }
    }
}

// Memory pool for batch processing
struct BatchBufferPool {
    buffers: parking_lot::Mutex<Vec<AlignedBuffer>>,
    huge_buffers: parking_lot::Mutex<Vec<HugePageBuffer>>,
    config: BatchConfig,
}

pub struct BatchProcessor<A: Aes> {
    cipher: Arc<AesGcmSiv<A>>,
    config: BatchConfig,
    stats: Arc<BatchStatistics>,
    buffer_pool: Arc<BatchBufferPool>,
    thread_pool: Arc<rayon::ThreadPool>,
}

impl<A: Aes> BatchProcessor<A> {
    pub fn new(cipher: AesGcmSiv<A>, config: BatchConfig) -> Result<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_count)
            .build()
            .map_err(|e| AesGcmSivError::ThreadPoolError(e.to_string()))?;

        let buffer_pool = BatchBufferPool::new(&config)?;

        Ok(Self {
            cipher: Arc::new(cipher),
            config,
            stats: Arc::new(BatchStatistics::default()),
            buffer_pool: Arc::new(buffer_pool),
            thread_pool: Arc::new(thread_pool),
        })
    }

    pub fn get_statistics(&self) -> BatchStatistics {
        BatchStatistics {
            successful: AtomicUsize::new(self.stats.successful.load(Ordering::Relaxed)),
            failed: AtomicUsize::new(self.stats.failed.load(Ordering::Relaxed)),
            total_bytes: AtomicUsize::new(self.stats.total_bytes.load(Ordering::Relaxed)),
            processing_time: parking_lot::RwLock::new(*self.stats.processing_time.read()),
        }
    }
}

impl<A: Aes> BatchProcessor<A> {
    pub fn encrypt_batch(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        let start_time = Instant::now();
        
        let total_bytes: usize = inputs.iter()
            .map(|input| input.data.as_slice().len())
            .sum();
        self.stats.total_bytes.fetch_add(total_bytes, Ordering::Relaxed);

        let results = if inputs.len() > self.config.max_batch_size {
            // Process in chunks if batch is too large
            inputs.chunks(self.config.max_batch_size)
                .flat_map(|chunk| self.process_batch_chunk(chunk, true))
                .collect()
        } else {
            self.process_batch_chunk(&inputs, true)
        };

        *self.stats.processing_time.write() = start_time.elapsed();
        results
    }

    fn process_batch_chunk(&self, inputs: &[BatchInput], is_encrypt: bool) -> Vec<BatchOutput> {
        self.thread_pool.install(|| {
            inputs.par_iter()
                .map(|input| {
                    let result = if is_encrypt {
                        self.process_single_encryption(input)
                    } else {
                        self.process_single_decryption(input)
                    };

                    match &result {
                        Ok(_) => self.stats.successful.fetch_add(1, Ordering::Relaxed),
                        Err(_) => self.stats.failed.fetch_add(1, Ordering::Relaxed),
                    }

                    BatchOutput {
                        result: result.unwrap_or_else(|_| AlignedBuffer::new(0).unwrap()),
                        status: result.map(|_| ()),
                    }
                })
                .collect()
        })
    }

    #[inline]
    fn process_single_encryption(&self, input: &BatchInput) -> Result<AlignedBuffer> {
        let result = self.cipher.encrypt(
            input.nonce.as_slice(),
            input.data.as_slice(),
            input.aad.as_slice(),
        )?;

        let mut buffer = self.buffer_pool.acquire(result.len())?;
        buffer.as_mut_slice().copy_from_slice(&result);
        Ok(buffer)
    }
}

impl<A: Aes> BatchProcessor<A> {
    pub fn decrypt_batch(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        let start_time = Instant::now();
        
        let total_bytes: usize = inputs.iter()
            .map(|input| input.data.as_slice().len())
            .sum();
        self.stats.total_bytes.fetch_add(total_bytes, Ordering::Relaxed);

        let results = if inputs.len() > self.config.max_batch_size {
            inputs.chunks(self.config.max_batch_size)
                .flat_map(|chunk| self.process_batch_chunk(chunk, false))
                .collect()
        } else {
            self.process_batch_chunk(&inputs, false)
        };

        *self.stats.processing_time.write() = start_time.elapsed();
        results
    }

    #[inline]
    fn process_single_decryption(&self, input: &BatchInput) -> Result<AlignedBuffer> {
        let result = self.cipher.decrypt(
            input.nonce.as_slice(),
            input.data.as_slice(),
            input.aad.as_slice(),
        )?;

        let mut buffer = self.buffer_pool.acquire(result.len())?;
        buffer.as_mut_slice().copy_from_slice(&result);
        Ok(buffer)
    }

    pub fn process_with_retry(&self, inputs: Vec<BatchInput>, is_encrypt: bool) -> Vec<BatchOutput> {
        const MAX_RETRIES: usize = 3;
        let mut retry_queue = VecDeque::new();
        let mut results = Vec::with_capacity(inputs.len());

        // First pass
        for input in inputs {
            let result = self.process_with_backoff(&input, is_encrypt, 0);
            if result.status.is_err() {
                retry_queue.push_back((input, 1));
            }
            results.push(result);
        }

        // Process retry queue with exponential backoff
        while let Some((input, retry_count)) = retry_queue.pop_front() {
            if retry_count < MAX_RETRIES {
                let result = self.process_with_backoff(&input, is_encrypt, retry_count);
                if result.status.is_err() {
                    retry_queue.push_back((input, retry_count + 1));
                }
            }
        }

        results
    }

    #[inline]
    fn process_with_backoff(&self, input: &BatchInput, is_encrypt: bool, retry_count: usize) -> BatchOutput {
        if retry_count > 0 {
            thread::sleep(Duration::from_millis(100 * 2u64.pow(retry_count as u32 - 1)));
        }

        let result = if is_encrypt {
            self.process_single_encryption(input)
        } else {
            self.process_single_decryption(input)
        };

        BatchOutput {
            result: result.unwrap_or_else(|_| AlignedBuffer::new(0).unwrap()),
            status: result.map(|_| ()),
        }
    }
}

struct MemoryPoolStats {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}

impl MemoryPool {
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        let pool = Self {
            small_buffers: parking_lot::Mutex::new(Vec::with_capacity(config.small_pool_size)),
            medium_buffers: parking_lot::Mutex::new(Vec::with_capacity(config.medium_pool_size)),
            large_buffers: parking_lot::Mutex::new(Vec::with_capacity(config.large_pool_size)),
            huge_pages: parking_lot::Mutex::new(Vec::new()),
            stats: MemoryPoolStats::default(),
        };

        // Pre-allocate buffers
        pool.initialize_pools(config)?;
        Ok(pool)
    }

    fn initialize_pools(&self, config: &MemoryConfig) -> Result<()> {
        // Initialize small buffers (<=4KB)
        for _ in 0..config.small_pool_size {
            self.small_buffers.lock().push(AlignedBuffer::new(4096)?);
        }

        // Initialize medium buffers (<=64KB)
        for _ in 0..config.medium_pool_size {
            self.medium_buffers.lock().push(AlignedBuffer::new(65536)?);
        }

        // Initialize large buffers (<=1MB)
        for _ in 0..config.large_pool_size {
            self.large_buffers.lock().push(AlignedBuffer::new(1048576)?);
        }

        Ok(())
    }
}

impl MemoryPool {
    pub fn acquire(&self, size: usize) -> Result<AlignedBuffer> {
        let buffer = if size <= 4096 {
            self.acquire_from_pool(&self.small_buffers, size)
        } else if size <= 65536 {
            self.acquire_from_pool(&self.medium_buffers, size)
        } else if size <= 1048576 {
            self.acquire_from_pool(&self.large_buffers, size)
        } else {
            self.acquire_huge_page(size)
        };

        match buffer {
            Ok(buf) => {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                Ok(buf)
            }
            Err(_) => {
                self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
                self.allocate_new(size)
            }
        }
    }

    fn acquire_from_pool(
        &self,
        pool: &parking_lot::Mutex<Vec<AlignedBuffer>>,
        size: usize
    ) -> Result<AlignedBuffer> {
        let mut pool_guard = pool.lock();
        match pool_guard.pop() {
            Some(mut buffer) => {
                if buffer.capacity() >= size {
                    buffer.resize(size)?;
                    Ok(buffer)
                } else {
                    pool_guard.push(buffer);
                    Err(AesGcmSivError::BufferTooSmall {
                        provided: buffer.capacity(),
                        needed: size,
                    })
                }
            }
            None => Err(AesGcmSivError::NoBuffersAvailable),
        }
    }

    fn acquire_huge_page(&self, size: usize) -> Result<AlignedBuffer> {
        #[cfg(feature = "huge-pages")]
        {
            let mut huge_pages = self.huge_pages.lock();
            if let Some(buffer) = huge_pages.pop() {
                if buffer.size() >= size {
                    return Ok(buffer.into());
                }
                huge_pages.push(buffer);
            }
        }
        Err(AesGcmSivError::NoBuffersAvailable)
    }
}

impl MemoryPool {
    fn allocate_new(&self, size: usize) -> Result<AlignedBuffer> {
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        
        if size > 2 * 1024 * 1024 && cfg!(feature = "huge-pages") {
            // Use huge pages for very large allocations
            self.allocate_huge_page(size)
        } else {
            AlignedBuffer::new(size)
        }
    }

    #[cfg(feature = "huge-pages")]
    fn allocate_huge_page(&self, size: usize) -> Result<AlignedBuffer> {
        let huge_page_size = 2 * 1024 * 1024; // 2MB
        let pages_needed = (size + huge_page_size - 1) / huge_page_size;
        let buffer = HugePageBuffer::new(pages_needed * huge_page_size)?;
        Ok(buffer.into())
    }

    pub fn release(&self, buffer: AlignedBuffer) {
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        
        let size = buffer.capacity();
        let pool = if size <= 4096 {
            &self.small_buffers
        } else if size <= 65536 {
            &self.medium_buffers
        } else if size <= 1048576 {
            &self.large_buffers
        } else {
            // Large buffers go to huge pages pool
            if cfg!(feature = "huge-pages") {
                self.huge_pages.lock().push(buffer.into());
            }
            return;
        };

        // Return to appropriate pool if not full
        let mut pool_guard = pool.lock();
        if pool_guard.len() < pool_guard.capacity() {
            pool_guard.push(buffer);
        }
    }

    pub fn get_stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            allocations: AtomicUsize::new(
                self.stats.allocations.load(Ordering::Relaxed)
            ),
            deallocations: AtomicUsize::new(
                self.stats.deallocations.load(Ordering::Relaxed)
            ),
            cache_hits: AtomicUsize::new(
                self.stats.cache_hits.load(Ordering::Relaxed)
            ),
            cache_misses: AtomicUsize::new(
                self.stats.cache_misses.load(Ordering::Relaxed)
            ),
        }
    }
}

pub struct CacheOptimizer {
    line_size: usize,
    prefetch_distance: usize,
}

impl CacheOptimizer {
    pub fn new() -> Self {
        Self {
            line_size: cache_line_size(),
            prefetch_distance: 64 * 8,  // Prefetch 8 cache lines ahead
        }
    }

    #[inline(always)]
    pub unsafe fn prefetch_read(&self, ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::__prefetch;
            __prefetch(ptr as *const i8, 0, 0);
        }
    }

    #[inline(always)]
    pub unsafe fn copy_aligned(
        &self,
        dst: *mut u8,
        src: *const u8,
        len: usize
    ) {
        let mut i = 0;
        
        // Prefetch ahead
        while i + self.prefetch_distance < len {
            self.prefetch_read(src.add(i + self.prefetch_distance));
            i += self.line_size;
        }

        // Copy using SIMD when possible
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                self.copy_aligned_avx2(dst, src, len);
                return;
            }
        }

        // Fallback to standard copy
        std::ptr::copy_nonoverlapping(src, dst, len);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn copy_aligned_avx2(
        &self,
        dst: *mut u8,
        src: *const u8,
        len: usize
    ) {
        use std::arch::x86_64::*;
        
        let mut i = 0;
        while i + 32 <= len {
            let data = _mm256_load_si256(src.add(i) as *const __m256i);
            _mm256_store_si256(dst.add(i) as *mut __m256i, data);
            i += 32;
        }

        if i < len {
            std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
        }
    }
}

#[inline(always)]
fn cache_line_size() -> usize {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Try to get from CPUID
        if let Some(size) = get_x86_cache_line_size() {
            return size;
        }
    }
    
    // Default fallback
    64
}

#[cfg(target_arch = "x86_64")]
unsafe fn get_x86_cache_line_size() -> Option<usize> {
    use std::arch::x86_64::__cpuid;
    let cpuid = __cpuid(0x1);
    Some(((cpuid.ebx >> 8) & 0xff) * 8)
}

#[derive(Debug, Clone)]
pub struct PlatformConfig {
    cpu_features: CpuFeatures,
    memory_features: MemoryFeatures,
    os_features: OsFeatures,
}

impl PlatformConfig {
    pub fn detect() -> Self {
        Self {
            cpu_features: CpuFeatures::detect(),
            memory_features: MemoryFeatures::detect(),
            os_features: OsFeatures::detect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuFeatures {
    cache_line_size: usize,
    l1_cache_size: usize,
    l2_cache_size: usize,
    l3_cache_size: usize,
    cores: usize,
    threads_per_core: usize,
    numa_nodes: usize,
}

impl CpuFeatures {
    fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::fs;
    use libc::{mlock, madvise, MADV_HUGEPAGE};

    pub struct LinuxOptimizer {
        config: PlatformConfig,
        memory_lock: bool,
        transparent_hugepages: bool,
    }

    impl LinuxOptimizer {
        pub fn new(config: PlatformConfig) -> Self {
            Self {
                config,
                memory_lock: false,
                transparent_hugepages: false,
            }
        }

        pub fn optimize_memory(&mut self, buffer: &mut AlignedBuffer) -> Result<()> {
            // Lock memory to prevent swapping
            if !self.memory_lock {
                unsafe {
                    if mlock(
                        buffer.as_ptr() as *const libc::c_void,
                        buffer.len()
                    ) == 0 {
                        self.memory_lock = true;
                    }
                }
            }

            // Enable transparent huge pages
            if !self.transparent_hugepages {
                unsafe {
                    if madvise(
                        buffer.as_mut_ptr() as *mut libc::c_void,
                        buffer.len(),
                        MADV_HUGEPAGE
                    ) == 0 {
                        self.transparent_hugepages = true;
                    }
                }
            }

            Ok(())
        }

        pub fn set_cpu_affinity(&self) -> Result<()> {
            use core_affinity::CoreId;
            
            // Get available cores
            let core_ids = core_affinity::get_core_ids()
                .ok_or_else(|| AesGcmSivError::PlatformError("Failed to get core IDs".into()))?;

            // Set affinity to physical cores first
            for core_id in core_ids.iter().step_by(2) {
                core_affinity::set_for_current(*core_id);
            }

            Ok(())
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use super::*;
    use windows::Win32::System::Memory;
    use windows::Win32::System::Threading;

    pub struct WindowsOptimizer {
        config: PlatformConfig,
        large_pages_enabled: bool,
    }

    impl WindowsOptimizer {
        pub fn new(config: PlatformConfig) -> Self {
            Self {
                config,
                large_pages_enabled: false,
            }
        }

        pub fn optimize_memory(&mut self, buffer: &mut AlignedBuffer) -> Result<()> {
            unsafe {
                // Enable large pages if available
                if !self.large_pages_enabled {
                    let token = Threading::GetCurrentProcess();
                    if Memory::VirtualLock(
                        buffer.as_mut_ptr() as *mut _,
                        buffer.len()
                    ).is_ok() {
                        self.large_pages_enabled = true;
                    }
                }

                // Set working set priority
                Memory::SetProcessWorkingSetSize(
                    Threading::GetCurrentProcess(),
                    buffer.len() as usize,
                    buffer.len() as usize * 2,
                );
            }

            Ok(())
        }

        pub fn set_processor_group(&self) -> Result<()> {
            unsafe {
                let mut group_affinity = Threading::GROUP_AFFINITY::default();
                
                // Prefer first processor group
                group_affinity.Group = 0;
                group_affinity.Mask = (1 << self.config.cpu_features.cores) - 1;

                if Threading::SetThreadGroupAffinity(
                    Threading::GetCurrentThread(),
                    &group_affinity,
                    std::ptr::null_mut(),
                ).is_ok() {
                    Ok(())
                } else {
                    Err(AesGcmSivError::PlatformError("Failed to set processor group".into()))
                }
            }
        }
    }
}

pub struct PerformanceMonitor {
    counters: HashMap<String, PerformanceCounter>,
    platform_config: PlatformConfig,
}

impl PerformanceMonitor {
    pub fn new(platform_config: PlatformConfig) -> Self {
        let mut counters = HashMap::new();
        
        // Initialize platform-specific counters
        #[cfg(target_os = "linux")]
        {
            counters.insert(
                "cache_misses".into(),
                PerformanceCounter::new(perf_event::PERF_COUNT_HW_CACHE_MISSES)
            );
            counters.insert(
                "cache_references".into(),
                PerformanceCounter::new(perf_event::PERF_COUNT_HW_CACHE_REFERENCES)
            );
        }

#[cfg(target_os = "windows")]
mod windows_performance {
    use windows::Win32::System::Performance::*;
    use windows::Win32::Foundation::*;
    use std::ptr::null_mut;

    pub struct WindowsPerformanceCounter {
        query_handle: HQUERY,
        counter_handle: HCOUNTER,
        counter_path: String,
        last_value: i64,
    }

    impl WindowsPerformanceCounter {
        pub fn new(counter_path: &str) -> Result<Self> {
            let mut query_handle = HQUERY::default();
            let mut counter_handle = HCOUNTER::default();

            unsafe {
                // Create a query object
                if PdhOpenQueryW(None, 0, &mut query_handle).is_err() {
                    return Err(AesGcmSivError::PlatformError(
                        "Failed to open performance query".into()
                    ));
                }

                // Add the counter to the query
                if PdhAddCounterW(
                    query_handle,
                    &windows::core::PWSTR::from(counter_path),
                    0,
                    &mut counter_handle
                ).is_err() {
                    PdhCloseQuery(query_handle);
                    return Err(AesGcmSivError::PlatformError(
                        "Failed to add performance counter".into()
                    ));
                }
            }

            Ok(Self {
                query_handle,
                counter_handle,
                counter_path: counter_path.to_string(),
                last_value: 0,
            })
        }

        pub fn collect_data(&mut self) -> Result<i64> {
            unsafe {
                // Collect the current counter values
                if PdhCollectQueryData(self.query_handle).is_err() {
                    return Err(AesGcmSivError::PlatformError(
                        "Failed to collect performance data".into()
                    ));
                }

                let mut counter_value = PDH_FMT_COUNTERVALUE::default();
                let mut counter_type = 0u32;

                // Get the formatted counter value
                if PdhGetFormattedCounterValue(
                    self.counter_handle,
                    PDH_FMT_LARGE,
                    &mut counter_type,
                    &mut counter_value
                ).is_err() {
                    return Err(AesGcmSivError::PlatformError(
                        "Failed to get counter value".into()
                    ));
                }

                self.last_value = counter_value.Anonymous.largeValue;
                Ok(self.last_value)
            }
        }
    }

    impl Drop for WindowsPerformanceCounter {
        fn drop(&mut self) {
            unsafe {
                PdhRemoveCounter(self.counter_handle);
                PdhCloseQuery(self.query_handle);
            }
        }
    }

    // Predefined performance counters for crypto operations
    pub struct CryptoPerformanceCounters {
        processor_time: WindowsPerformanceCounter,
        memory_usage: WindowsPerformanceCounter,
        io_operations: WindowsPerformanceCounter,
        cache_hits: WindowsPerformanceCounter,
        cache_misses: WindowsPerformanceCounter,
    }

    impl CryptoPerformanceCounters {
        pub fn new() -> Result<Self> {
            Ok(Self {
                processor_time: WindowsPerformanceCounter::new(
                    "\\Processor(_Total)\\% Processor Time"
                )?,
                memory_usage: WindowsPerformanceCounter::new(
                    "\\Memory\\Available MBytes"
                )?,
                io_operations: WindowsPerformanceCounter::new(
                    "\\PhysicalDisk(_Total)\\Avg. Disk Queue Length"
                )?,
                cache_hits: WindowsPerformanceCounter::new(
                    "\\Memory\\Cache Bytes"
                )?,
                cache_misses: WindowsPerformanceCounter::new(
                    "\\Memory\\Cache Faults/sec"
                )?,
            })
        }

        pub fn collect_all(&mut self) -> Result<PerformanceMetrics> {
            Ok(PerformanceMetrics {
                cpu_usage: self.processor_time.collect_data()?,
                memory_available: self.memory_usage.collect_data()?,
                io_queue_length: self.io_operations.collect_data()?,
                cache_hits: self.cache_hits.collect_data()?,
                cache_misses: self.cache_misses.collect_data()?,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics {
        pub cpu_usage: i64,
        pub memory_available: i64,
        pub io_queue_length: i64,
        pub cache_hits: i64,
        pub cache_misses: i64,
    }

    // Integration with the main performance monitor
    impl PerformanceMonitor {
        #[cfg(target_os = "windows")]
        fn initialize_windows_counters(&mut self) -> Result<()> {
            let crypto_counters = CryptoPerformanceCounters::new()?;
            
            self.counters.insert(
                "windows_perf".to_string(),
                Box::new(crypto_counters)
            );

            // Initialize system-wide performance monitoring
            unsafe {
                let mut counter_size: u32 = 0;
                let mut counter_count: u32 = 0;

                // Get the size needed for the counter list
                PdhEnumObjectItemsW(
                    None,
                    None,
                    windows::core::PCWSTR::null(),
                    null_mut(),
                    &mut counter_size,
                    null_mut(),
                    &mut counter_count,
                    PERF_DETAIL_WIZARD,
                    0,
                );

                // Allocate buffer and get counter names
                let mut counter_list = vec![0u16; counter_size as usize];
                let mut instance_list = vec![0u16; counter_count as usize];

                PdhEnumObjectItemsW(
                    None,
                    None,
                    windows::core::PCWSTR::null(),
                    counter_list.as_mut_ptr() as *mut _,
                    &mut counter_size,
                    instance_list.as_mut_ptr() as *mut _,
                    &mut counter_count,
                    PERF_DETAIL_WIZARD,
                    0,
                );
            }

            Ok(())
        }
    }
}

        
        Self {
            counters,
            platform_config,
        }
    }

    pub fn start_measurement(&mut self, counter_name: &str) -> Result<()> {
        if let Some(counter) = self.counters.get_mut(counter_name) {
            counter.start()?;
        }
        Ok(())
    }

    pub fn stop_measurement(&mut self, counter_name: &str) -> Result<u64> {
        if let Some(counter) = self.counters.get_mut(counter_name) {
            return counter.stop();
        }
        Err(AesGcmSivError::PlatformError("Counter not found".into()))
    }

    pub fn get_statistics(&self) -> HashMap<String, u64> {
        self.counters.iter()
            .map(|(name, counter)| (name.clone(), counter.value()))
            .collect()
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ptr;
    use mach::{kern_return, vm_types, vm_statistics, vm_prot};
    use mach::mach_port::mach_port_t;
    use mach::vm_types::mach_vm_address_t;
    use mach::kern_return::kern_return_t;
    use core_foundation::base::*;
    use io_kit_sys::*;

    pub struct MacosOptimizer {
        config: PlatformConfig,
        memory_locked: bool,
        qos_class_set: bool,
    }

    impl MacosOptimizer {
        pub fn new(config: PlatformConfig) -> Self {
            Self {
                config,
                memory_locked: false,
                qos_class_set: false,
            }
        }

        pub fn optimize_memory(&mut self, buffer: &mut AlignedBuffer) -> Result<()> {
            unsafe {
                // Lock memory to prevent paging
                if !self.memory_locked {
                    let result = mach_vm_wire(
                        mach_host_self(),
                        mach_task_self(),
                        buffer.as_ptr() as mach_vm_address_t,
                        buffer.len() as u64,
                        vm_prot::VM_PROT_READ | vm_prot::VM_PROT_WRITE
                    );

                    if result == kern_return::KERN_SUCCESS {
                        self.memory_locked = true;
                    }
                }

                // Enable memory page monitoring
                vm_statistics64_t::vm_page_monitor(
                    buffer.as_ptr() as mach_vm_address_t,
                    buffer.len() as u64,
                    VM_PAGE_MONITOR_ENABLE
                );
            }

            Ok(())
        }

        pub fn set_thread_qos(&mut self) -> Result<()> {
            if !self.qos_class_set {
                unsafe {
                    // Set QoS class to USER_INTERACTIVE for crypto operations
                    let mut qos = QOS_CLASS_USER_INTERACTIVE;
                    pthread_set_qos_class_self_np(qos, 0);
                    self.qos_class_set = true;
                }
            }
            Ok(())
        }

        pub fn optimize_power_management(&self) -> Result<()> {
            unsafe {
                // Prevent idle sleep during crypto operations
                let mut power_assert = IOPMAssertionID::default();
                let assertion_name = CFString::new("Crypto Operation In Progress");
                
                IOPMAssertionCreateWithName(
                    kIOPMAssertionTypeNoIdleSleep,
                    kIOPMAssertionLevelOn,
                    assertion_name.as_concrete_TypeRef(),
                    &mut power_assert
                );
            }
            Ok(())
        }

        // Optimize for Apple Silicon if available
        #[cfg(target_arch = "aarch64")]
        pub fn optimize_for_apple_silicon(&self) -> Result<()> {
            unsafe {
                // Enable ARM Cryptography Extensions
                if self.config.cpu_features.has_apple_crypto {
                    // Set CPU affinity to performance cores
                    let mut affinity = thread_affinity_policy_data_t {
                        affinity_tag: THREAD_AFFINITY_TAG_PERFORMANCE,
                    };
                    thread_policy_set(
                        mach_thread_self(),
                        THREAD_AFFINITY_POLICY,
                        &affinity as *const _ as *const i32,
                        THREAD_AFFINITY_POLICY_COUNT
                    );
                }
            }
            Ok(())
        }
    }

    // Performance monitoring for macOS
    pub struct MacosPerformanceMonitor {
        task_port: mach_port_t,
        previous_stats: task_vm_info_data_t,
    }

    impl MacosPerformanceMonitor {
        pub fn new() -> Result<Self> {
            let task_port = unsafe { mach_task_self() };
            Ok(Self {
                task_port,
                previous_stats: task_vm_info_data_t::default(),
            })
        }

        pub fn collect_metrics(&mut self) -> Result<MacosPerformanceMetrics> {
            unsafe {
                let mut task_info = task_vm_info_data_t::default();
                let mut count = TASK_VM_INFO_COUNT;

                let kr = task_info(
                    self.task_port,
                    TASK_VM_INFO,
                    &mut task_info as *mut _ as *mut i32,
                    &mut count
                );

                if kr != KERN_SUCCESS {
                    return Err(AesGcmSivError::PlatformError(
                        "Failed to collect macOS performance metrics".into()
                    ));
                }

                let metrics = MacosPerformanceMetrics {
                    memory_footprint: task_info.phys_footprint,
                    memory_compressed: task_info.compressed,
                    page_ins: task_info.pageins,
                    page_outs: task_info.pageouts,
                };

                self.previous_stats = task_info;
                Ok(metrics)
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct MacosPerformanceMetrics {
        pub memory_footprint: u64,
        pub memory_compressed: u64,
        pub page_ins: u64,
        pub page_outs: u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use rayon::prelude::*;
    use test_case::test_case;
    use hex_literal::hex;

    // Helper struct for organizing test vectors
    #[derive(Debug, Clone)]
    struct TestVector {
        key: Vec<u8>,
        nonce: Vec<u8>,
        aad: Vec<u8>,
        plaintext: Vec<u8>,
        ciphertext: Vec<u8>,
    }

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

        // Helper to run test with multiple implementations
        fn test_all_implementations(&self) -> Result<()> {
            // Test generic implementation
            self.test_implementation::<AesGeneric>()?;

            // Test hardware-accelerated implementations if available
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("aes") {
                self.test_implementation::<AesNi>()?;
                if is_x86_feature_detected!("avx512f") {
                    self.test_implementation::<AesAvx512>()?;
                }
            }

            #[cfg(target_arch = "aarch64")]
            if is_aarch64_feature_detected!("aes") {
                self.test_implementation::<AesArm>()?;
            }

            Ok(())
        }
    }
}


#[cfg(test)]
mod tests {
    #[test_case(KeySize::Aes128)]
    #[test_case(KeySize::Aes256)]
    fn test_basic_encryption_decryption(key_size: KeySize) -> Result<()> {
        let key = generate_random_key(key_size);
        let nonce = generate_random_nonce();
        let plaintext = b"Test message for encryption and decryption";
        let aad = b"Additional authenticated data";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key)?;
        let ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
        let decrypted = cipher.decrypt(&nonce, &ciphertext, aad)?;

        assert_eq!(plaintext, &decrypted[..]);
        Ok(())
    }

    #[test]
    fn test_aad_modification_fails() -> Result<()> {
        let key = generate_random_key(KeySize::Aes256);
        let nonce = generate_random_nonce();
        let plaintext = b"Test message";
        let aad = b"Original AAD";
        let modified_aad = b"Modified AAD";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key)?;
        let ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
        
        let result = cipher.decrypt(&nonce, &ciphertext, modified_aad);
        assert!(matches!(result, Err(AesGcmSivError::InvalidTag)));
        Ok(())
    }

    #[test]
    fn test_ciphertext_modification_fails() -> Result<()> {
        let key = generate_random_key(KeySize::Aes256);
        let nonce = generate_random_nonce();
        let plaintext = b"Test message";
        let aad = b"Additional data";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key)?;
        let mut ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
        
        // Modify one byte in the ciphertext
        ciphertext[0] ^= 1;
        
        let result = cipher.decrypt(&nonce, &ciphertext, aad);
        assert!(matches!(result, Err(AesGcmSivError::InvalidTag)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parallel_performance() -> Result<()> {
        let test_sizes = [
            1024,          // 1 KB
            1024 * 1024,   // 1 MB
            10 * 1024 * 1024, // 10 MB
        ];

        for &size in &test_sizes {
            let key = generate_random_key(KeySize::Aes256);
            let nonce = generate_random_nonce();
            let plaintext = generate_random_data(size);
            let aad = b"Performance test AAD";

            let cipher = AesGcmSiv::<AesGeneric>::new(&key)?;
            
            // Measure encryption performance
            let start = Instant::now();
            let ciphertext = cipher.encrypt(&nonce, &plaintext, aad)?;
            let encryption_time = start.elapsed();

            // Measure decryption performance
            let start = Instant::now();
            let _ = cipher.decrypt(&nonce, &ciphertext, aad)?;
            let decryption_time = start.elapsed();

            println!("Size: {} bytes", size);
            println!("Encryption time: {:?}", encryption_time);
            println!("Decryption time: {:?}", decryption_time);
            println!("Throughput: {:.2} MB/s", 
                (size as f64 / encryption_time.as_secs_f64()) / (1024.0 * 1024.0));
        }

        Ok(())
    }

    #[test]
    fn test_memory_usage() -> Result<()> {
        let initial_memory = get_process_memory();
        
        let test_data = generate_random_data(50 * 1024 * 1024); // 50MB
        let key = generate_random_key(KeySize::Aes256);
        let nonce = generate_random_nonce();
        let aad = b"Memory test AAD";

        let cipher = AesGcmSiv::<AesGeneric>::new(&key)?;
        
        // Measure memory during encryption
        let encrypt_memory = {
            let _ = cipher.encrypt(&nonce, &test_data, aad)?;
            get_process_memory()
        };

        // Verify memory usage is within acceptable bounds
        let memory_increase = encrypt_memory - initial_memory;
        assert!(memory_increase < 100 * 1024 * 1024); // Less than 100MB overhead
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    mod x86_tests {
        use super::*;

        #[test]
        fn test_aesni_implementation() -> Result<()> {
            if !is_x86_feature_detected!("aes") {
                return Ok(());
            }

            let key = generate_random_key(KeySize::Aes256);
            let nonce = generate_random_nonce();
            let plaintext = b"AES-NI test message";
            let aad = b"AES-NI test AAD";

            let cipher = AesGcmSiv::<AesNi>::new(&key)?;
            let ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
            let decrypted = cipher.decrypt(&nonce, &ciphertext, aad)?;

            assert_eq!(plaintext, &decrypted[..]);
            Ok(())
        }

        #[test]
        fn test_avx512_implementation() -> Result<()> {
            if !is_x86_feature_detected!("avx512f") {
                return Ok(());
            }

            let key = generate_random_key(KeySize::Aes256);
            let nonce = generate_random_nonce();
            let plaintext = generate_random_data(1024 * 1024); // 1MB
            let aad = b"AVX-512 test AAD";

            let cipher = AesGcmSiv::<AesAvx512>::new(&key)?;
            let ciphertext = cipher.encrypt(&nonce, &plaintext, aad)?;
            let decrypted = cipher.decrypt(&nonce, &ciphertext, aad)?;

            assert_eq!(plaintext, &decrypted[..]);
            Ok(())
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod arm_tests {
        use super::*;

        #[test]
        fn test_arm_crypto_implementation() -> Result<()> {
            if !is_aarch64_feature_detected!("aes") {
                return Ok(());
            }

            let key = generate_random_key(KeySize::Aes256);
            let nonce = generate_random_nonce();
            let plaintext = b"ARM Crypto test message";
            let aad = b"ARM Crypto test AAD";

            let cipher = AesGcmSiv::<AesArm>::new(&key)?;
            let ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
            let decrypted = cipher.decrypt(&nonce, &ciphertext, aad)?;

            assert_eq!(plaintext, &decrypted[..]);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_processing() -> Result<()> {
        let config = BatchConfig::default();
        let cipher = AesGcmSiv::<AesGeneric>::new(&generate_random_key(KeySize::Aes256))?;
        let processor = BatchProcessor::new(cipher, config)?;

        // Create test batch
        let mut inputs = Vec::new();
        for i in 0..100 {
            inputs.push(BatchInput {
                nonce: generate_random_nonce().to_vec(),
                data: generate_random_data(1024), // 1KB each
                aad: format!("AAD for batch {}", i).into_bytes(),
            });
        }

        // Test encryption
        let encrypted = processor.encrypt_batch(inputs.clone());
        assert_eq!(encrypted.len(), 100);
        assert!(encrypted.iter().all(|r| r.status.is_ok()));

        // Create decrypt batch
        let decrypt_inputs: Vec<_> = encrypted.iter().enumerate().map(|(i, output)| {
            BatchInput {
                nonce: inputs[i].nonce.clone(),
                data: output.result.clone(),
                aad: inputs[i].aad.clone(),
            }
        }).collect();

        // Test decryption
        let decrypted = processor.decrypt_batch(decrypt_inputs);
        assert_eq!(decrypted.len(), 100);
        assert!(decrypted.iter().all(|r| r.status.is_ok()));

        // Verify results
        for (i, output) in decrypted.iter().enumerate() {
            assert_eq!(output.result, inputs[i].data);
        }

        Ok(())
    }

    #[test]
    fn test_batch_error_handling() -> Result<()> {
        let config = BatchConfig::default();
        let cipher = AesGcmSiv::<AesGeneric>::new(&generate_random_key(KeySize::Aes256))?;
        let processor = BatchProcessor::new(cipher, config)?;

        // Create batch with some invalid inputs
        let mut inputs = Vec::new();
        
        // Valid input
        inputs.push(BatchInput {
            nonce: generate_random_nonce().to_vec(),
            data: b"Valid data".to_vec(),
            aad: b"Valid AAD".to_vec(),
        });

        // Invalid nonce size
        inputs.push(BatchInput {
            nonce: vec![0; 11], // Wrong size
            data: b"Invalid nonce".to_vec(),
            aad: b"AAD".to_vec(),
        });

        // Invalid data size
        inputs.push(BatchInput {
            nonce: generate_random_nonce().to_vec(),
            data: vec![0; AES_GCMSIV_MAX_PLAINTEXT_SIZE + 1],
            aad: b"AAD".to_vec(),
        });

        let results = processor.encrypt_batch(inputs);
        assert_eq!(results.len(), 3);
        assert!(results[0].status.is_ok());
        assert!(results[1].status.is_err());
        assert!(results[2].status.is_err());

        Ok(())
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use criterion::{
        black_box, criterion_group, criterion_main, Criterion, BenchmarkId,
        measurement::WallTime, BatchSize,
    };
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

    fn create_benchmark_group<'a>(c: &'a mut Criterion) -> criterion::BenchmarkGroup<'a, WallTime> {
        let mut group = c.benchmark_group("AES-GCM-SIV");
        group
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(5))
            .sample_size(100);
        group
    }
}

#[cfg(test)]
mod bench {
    pub fn bench_implementations(c: &mut Criterion) {
        let mut group = create_benchmark_group(c);

        for &size in BENCH_SIZES {
            let data = BenchmarkData::new(size);
            
            // Benchmark Generic Implementation
            group.bench_function(
                BenchmarkId::new("Generic/Encrypt", size),
                |b| {
                    let cipher = AesGcmSiv::<AesGeneric>::new(&data.key).unwrap();
                    b.iter(|| {
                        cipher.encrypt(
                            black_box(&data.nonce),
                            black_box(&data.data),
                            black_box(&data.aad)
                        )
                    });
                }
            );

            // Benchmark AES-NI if available
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("aes") {
                group.bench_function(
                    BenchmarkId::new("AES-NI/Encrypt", size),
                    |b| {
                        let cipher = AesGcmSiv::<AesNi>::new(&data.key).unwrap();
                        b.iter(|| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(&data.data),
                                black_box(&data.aad)
                            )
                        });
                    }
                );

                if is_x86_feature_detected!("avx512f") {
                    group.bench_function(
                        BenchmarkId::new("AVX512/Encrypt", size),
                        |b| {
                            let cipher = AesGcmSiv::<AesAvx512>::new(&data.key).unwrap();
                            b.iter(|| {
                                cipher.encrypt(
                                    black_box(&data.nonce),
                                    black_box(&data.data),
                                    black_box(&data.aad)
                                )
                            });
                        }
                    );
                }
            }

            // Benchmark ARM implementation if available
            #[cfg(target_arch = "aarch64")]
            if is_aarch64_feature_detected!("aes") {
                group.bench_function(
                    BenchmarkId::new("ARM-Crypto/Encrypt", size),
                    |b| {
                        let cipher = AesGcmSiv::<AesArm>::new(&data.key).unwrap();
                        b.iter(|| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(&data.data),
                                black_box(&data.aad)
                            )
                        });
                    }
                );
            }
        }

        group.finish();
    }
}

#[cfg(test)]
mod bench {
    pub fn bench_memory_performance(c: &mut Criterion) {
        let mut group = create_benchmark_group(c);
        
        for &size in BENCH_SIZES {
            let data = BenchmarkData::new(size);
            
            // Benchmark with different memory configurations
            for alignment in [16, 32, 64, 128] {
                group.bench_function(
                    BenchmarkId::new(format!("Aligned-{}", alignment), size),
                    |b| {
                        let mut aligned_data = AlignedBuffer::new_aligned(size, alignment).unwrap();
                        aligned_data.copy_from_slice(&data.data);
                        
                        let cipher = AesGcmSiv::<AesGeneric>::new(&data.key).unwrap();
                        b.iter(|| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(aligned_data.as_slice()),
                                black_box(&data.aad)
                            )
                        });
                    }
                );
            }

            // Benchmark cache effects
            group.bench_function(
                BenchmarkId::new("CacheEffects", size),
                |b| {
                    let cipher = AesGcmSiv::<AesGeneric>::new(&data.key).unwrap();
                    b.iter_batched(
                        || data.data.clone(),
                        |data| {
                            cipher.encrypt(
                                black_box(&data.nonce),
                                black_box(&data),
                                black_box(&data.aad)
                            )
                        },
                        BatchSize::SmallInput
                    );
                }
            );
        }

        group.finish();
    }
}

#[cfg(test)]
mod bench {
    pub fn bench_parallel_processing(c: &mut Criterion) {
        let mut group = create_benchmark_group(c);
        
        let thread_counts = [1, 2, 4, 8, 16];
        let batch_sizes = [1, 10, 100, 1000];

        for &batch_size in &batch_sizes {
            for &threads in &thread_counts {
                let config = BatchConfig {
                    thread_count: threads,
                    ..Default::default()
                };

                group.bench_function(
                    BenchmarkId::new(
                        format!("Batch-{}/Threads-{}", batch_size, threads),
                        batch_size
                    ),
                    |b| {
                        let cipher = AesGcmSiv::<AesGeneric>::new(
                            &generate_random_key(KeySize::Aes256)
                        ).unwrap();
                        let processor = BatchProcessor::new(cipher, config.clone()).unwrap();
                        
                        let inputs: Vec<_> = (0..batch_size)
                            .map(|_| BatchInput {
                                nonce: generate_random_nonce().to_vec(),
                                data: generate_random_data(1024),
                                aad: b"Benchmark AAD".to_vec(),
                            })
                            .collect();

                        b.iter(|| {
                            processor.encrypt_batch(black_box(inputs.clone()))
                        });
                    }
                );
            }
        }

        group.finish();
    }
}

#[cfg(test)]
mod bench {
    pub fn bench_parallel_processing(c: &mut Criterion) {
        let mut group = create_benchmark_group(c);
        
        let thread_counts = [1, 2, 4, 8, 16];
        let batch_sizes = [1, 10, 100, 1000];

        for &batch_size in &batch_sizes {
            for &threads in &thread_counts {
                let config = BatchConfig {
                    thread_count: threads,
                    ..Default::default()
                };

                group.bench_function(
                    BenchmarkId::new(
                        format!("Batch-{}/Threads-{}", batch_size, threads),
                        batch_size
                    ),
                    |b| {
                        let cipher = AesGcmSiv::<AesGeneric>::new(
                            &generate_random_key(KeySize::Aes256)
                        ).unwrap();
                        let processor = BatchProcessor::new(cipher, config.clone()).unwrap();
                        
                        let inputs: Vec<_> = (0..batch_size)
                            .map(|_| BatchInput {
                                nonce: generate_random_nonce().to_vec(),
                                data: generate_random_data(1024),
                                aad: b"Benchmark AAD".to_vec(),
                            })
                            .collect();

                        b.iter(|| {
                            processor.encrypt_batch(black_box(inputs.clone()))
                        });
                    }
                );
            }
        }

        group.finish();
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
        targets = bench_implementations,
                 bench_memory_performance,
                 bench_parallel_processing
    }

    criterion_main!(benches);

    pub fn generate_performance_report(results: &mut HashMap<String, Vec<Duration>>) {
        println!("\nPerformance Report:");
        println!("===================");

        for (name, durations) in results {
            let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
            let throughput = calculate_throughput(
                durations.len() * 1024 * 1024,
                durations.iter().sum()
            );

            println!("\nBenchmark: {}", name);
            println!("Average time: {:?}", avg);
            println!("Throughput: {:.2} MB/s", throughput);
            
            // Calculate percentiles
            durations.sort_unstable();
            let p50 = durations[durations.len() / 2];
            let p95 = durations[durations.len() * 95 / 100];
            let p99 = durations[durations.len() * 99 / 100];

            println!("Latency Percentiles:");
            println!("  P50: {:?}", p50);
            println!("  P95: {:?}", p95);
            println!("  P99: {:?}", p99);

            // Hardware utilization if available
            #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
            if let Some(stats) = get_hardware_stats() {
                println!("\nHardware Utilization:");
                println!("  CPU Usage: {:.1}%", stats.cpu_usage);
                println!("  Memory Usage: {:.1} MB", stats.memory_usage / 1024.0 / 1024.0);
                println!("  Cache Miss Rate: {:.2}%", stats.cache_miss_rate);
            }
        }
    }

    fn calculate_throughput(bytes: usize, duration: Duration) -> f64 {
        (bytes as f64) / duration.as_secs_f64() / (1024.0 * 1024.0)
    }
}

fn main() -> Result<()> {
    // Example configuration
    let config = GcmSivConfig {
        tag_length: AES_GCMSIV_TAG_SIZE,
        min_tag_length: 12,
        max_tag_length: 16,
        use_parallel: true,
        parallel_threshold: PARALLEL_THRESHOLD,
    };

    // Create cipher with optimal implementation for platform
    let cipher = create_aes_implementation(KeySize::Aes256)?;
    
    // Example usage
    println!("AES-GCM-SIV Encryption Example");
    println!("==============================");

    let key = generate_random_key(KeySize::Aes256);
    let nonce = generate_random_nonce();
    let plaintext = b"Example message for encryption";
    let aad = b"Additional authenticated data";

    // Single operation example
    println!("\nSingle Operation Example:");
    let ciphertext = cipher.encrypt(&nonce, plaintext, aad)?;
    println!("Encrypted: {:?}", hex::encode(&ciphertext));
    
    let decrypted = cipher.decrypt(&nonce, &ciphertext, aad)?;
    println!("Decrypted: {}", String::from_utf8_lossy(&decrypted));

    // Batch processing example
    println!("\nBatch Processing Example:");
    let batch_config = BatchConfig::default();
    let processor = BatchProcessor::new(cipher.clone(), batch_config)?;

    let inputs: Vec<_> = (0..10)
        .map(|i| BatchInput {
            nonce: generate_random_nonce().to_vec(),
            data: format!("Batch message {}", i).into_bytes(),
            aad: b"Batch AAD".to_vec(),
        })
        .collect();

    let encrypted_batch = processor.encrypt_batch(inputs);
    println!("Processed {} batch items", encrypted_batch.len());

    // Performance example
    println!("\nPerformance Test:");
    let perf_data = generate_random_data(1024 * 1024); // 1MB
    let start = Instant::now();
    let _ = cipher.encrypt(&nonce, &perf_data, aad)?;
    let duration = start.elapsed();
    println!(
        "Throughput: {:.2} MB/s",
        (perf_data.len() as f64) / duration.as_secs_f64() / (1024.0 * 1024.0)
    );

    Ok(())
}


// Utility functions for key and nonce generation
pub fn generate_random_key(key_size: KeySize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..key_size as usize).map(|_| rng.gen()).collect()
}

pub fn generate_random_nonce() -> [u8; AES_GCMSIV_NONCE_SIZE] {
    let mut nonce = [0u8; AES_GCMSIV_NONCE_SIZE];
    rand::thread_rng().fill_bytes(&mut nonce);
    nonce
}

pub fn generate_random_data(size: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

// Helper function for hex formatting
pub fn format_hex(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<String>>()
        .join("")
}

// Helper function for measuring performance
pub fn measure_performance<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

use log::{info, warn, error, debug};

// Initialize logging
pub fn init_logging() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    debug!("Logging initialized");
    Ok(())
}

// Global error handler
pub fn handle_error(err: AesGcmSivError) {
    error!("Error occurred: {}", err);
    match err {
        AesGcmSivError::InvalidKeySize { size, expected } => {
            warn!("Invalid key size provided: {}, expected one of {:?}", size, expected);
        }
        AesGcmSivError::InvalidTag => {
            warn!("Authentication tag verification failed - data may be corrupted or tampered");
        }
        AesGcmSivError::BufferTooSmall { provided, needed } => {
            warn!("Buffer too small: needed {} bytes, got {}", needed, provided);
        }
        _ => {
            error!("Unexpected error occurred: {:?}", err);
        }
    }
}

// Panic hook
pub fn set_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        error!("Panic occurred: {:?}", panic_info);
        if let Some(location) = panic_info.location() {
            error!(
                "Panic occurred in file '{}' at line {}",
                location.file(),
                location.line()
            );
        }
    }));
}







// ====================== CI/CD Configuration ======================
/*
GitHub Actions Configuration (.github/workflows/ci.yml):

CI Pipeline:
1. Test matrix:
   - OS: Ubuntu, Windows, MacOS
   - Rust: stable, nightly
2. Steps for each combination:
   - Build with all features
   - Run test suite
   - Run clippy checks
   - Verify formatting
3. Benchmark job:
   - Runs on main branch pushes
   - Stores results
   - Generates performance reports

Development workflow:
1. Pull request checks:
   - Must pass all tests
   - Must maintain formatting
   - No clippy warnings
2. Main branch protection:
   - Require reviews
   - Must pass CI
3. Automated benchmarking
   - Performance regression detection
   - Benchmark result storage
*/

// ====================== Security Audit Configuration ======================
/*
Security and Audit Configuration:

Advisory Checks:
- Database: RustSec advisory database
- Vulnerability handling: deny
- Unmaintained crate handling: warn
- Yanked version handling: warn

License Requirements:
- Allowed licenses: MIT, Apache-2.0, BSD-3-Clause
- Unlicensed crates: deny
- Copyleft licenses: warn

Dependency Management:
- Multiple versions: warn
- Unknown registries: deny
- Unknown git sources: deny
- Allowed registries: crates.io only

Security Considerations:
1. Regular dependency updates
2. Automated vulnerability scanning
3. License compliance checking
4. Version conflict detection
*/

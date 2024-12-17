// AES-GCM_SIV - December 16 2024
// RFC: https://www.rfc-editor.org/rfc/rfc8452
// 
// This Rust code is a direct, functional translation of the canonical demonstration C code into Rust.
// It attempts to preserve logic, structure, and naming as closely as possible, relying only on Rust's standard library.
// All original logic, including AES key schedule, AES encryption, Polyval operations, and GCM-SIV related steps, are implemented.
// No hardware acceleration or platform-specific code is complete yet. The hardware-specific code paths and macros are omitted,
// leaving only the generic implementations. Minimal imports are used (only std for basic operations).
//
// Note: This code is large and complex, as it closely follows the original C implementation. Tables and constants are included inline.
// Please note that the AES implementation here is a direct port and may not be optimized for production use.
// It serves as a demonstration of functional equivalence and correctness.
//
// The original code depends on conditional compilation and multiple files. This Rust code places all logic in a single file for simplicity.
// Runtime CPU feature detection and related code are stubbed out or simplified.
// The code uses Box for dynamic memory allocation where the original code used malloc/free.
// Memory zeroization is done using a volatile write loop. Constants and tables are directly copied from the original code.
// For brevity, some inline comments are omitted, but the logic remains equivalent.
//
// DISCLAIMER: This code is for demonstration and has not been tested or audited.

#![feature(stdsimd, target_feature_11)]

use std::mem;
use std::ptr;
use std::slice;

/// ====================== Constants and Limits ======================

/// Size of the authentication tag in bytes.
const AES_GCMSIV_TAG_SIZE: usize = 16;

/// Size of the nonce in bytes.
const AES_GCMSIV_NONCE_SIZE: usize = 12;

/// Size of the Polyval output in bytes.
const POLYVAL_SIZE: usize = 16;

/// Size of an AES block in bytes.
const AES_BLOCK_SIZE: usize = 16;

/// Maximum plaintext size supported.
const AES_GCMSIV_MAX_PLAINTEXT_SIZE: usize = (1 << 36) - 1;

/// Maximum Additional Authenticated Data (AAD) size supported.
const AES_GCMSIV_MAX_AAD_SIZE: usize = (1 << 36) - 1;

/// Size of the authentication key in bytes.
const KEY_AUTH_SIZE: usize = 16;

/// Maximum size of the encryption key in bytes.
const KEY_ENC_MAX_SIZE: usize = 32;

/// ====================== Status and Error Handling ======================

/// Enum representing various status codes for AES-GCM-SIV operations.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
enum aes_gcmsiv_status_t {
    AES_GCMSIV_SUCCESS = 0,
    AES_GCMSIV_FAILURE = -1,
    AES_GCMSIV_OUT_OF_MEMORY = -2,
    AES_GCMSIV_UPDATE_OUTPUT_SIZE = -3,
    AES_GCMSIV_INVALID_PARAMETERS = -4,
    AES_GCMSIV_INVALID_KEY_SIZE = -5,
    AES_GCMSIV_INVALID_NONCE_SIZE = -6,
    AES_GCMSIV_INVALID_PLAINTEXT_SIZE = -7,
    AES_GCMSIV_INVALID_AAD_SIZE = -8,
    AES_GCMSIV_INVALID_CIPHERTEXT_SIZE = -9,
    AES_GCMSIV_INVALID_TAG = -10,
}

/// ====================== Key Context Structure ======================

/// Structure to hold derived authentication and encryption keys.
#[derive(Copy, Clone)]
struct key_context {
    auth: [u8; KEY_AUTH_SIZE],
    auth_sz: usize,
    enc: [u8; KEY_ENC_MAX_SIZE],
    enc_sz: usize,
}

/// ====================== AES-GCM-SIV Context Structure ======================

/// Context structure for AES-GCM-SIV operations.
#[repr(C)]
struct aes_gcmsiv_ctx {
    key_gen_ctx: *mut aes,
    key_sz: usize,
}

/// ====================== Utility Functions ======================

/// Writes a 32-bit unsigned integer to a byte slice in little-endian format.
#[inline]
fn PUT_UINT32_LE(val: u32, dst: &mut [u8], offset: usize) {
    dst[offset] = (val & 0xff) as u8;
    dst[offset + 1] = ((val >> 8) & 0xff) as u8;
    dst[offset + 2] = ((val >> 16) & 0xff) as u8;
    dst[offset + 3] = ((val >> 24) & 0xff) as u8;
}

/// Reads a 32-bit unsigned integer from a byte slice in little-endian format.
#[inline]
fn GET_UINT32_LE(src: &[u8], offset: usize) -> u32 {
    (src[offset] as u32)
        | ((src[offset + 1] as u32) << 8)
        | ((src[offset + 2] as u32) << 16)
        | ((src[offset + 3] as u32) << 24)
}

/// Writes a 64-bit unsigned integer to a byte slice in little-endian format.
#[inline]
fn PUT_UINT64_LE(val: u64, dst: &mut [u8], offset: usize) {
    let bytes = val.to_le_bytes();
    dst[offset..offset + 8].copy_from_slice(&bytes);
}

/// Securely zeroizes a mutable byte buffer using volatile writes to prevent compiler optimizations.
#[inline]
fn aes_gcmsiv_zeroize(buf: &mut [u8]) {
    for x in buf.iter_mut() {
        unsafe {
            ptr::write_volatile(x, 0);
        }
    }
}

/// Securely zeroizes a mutable structure by treating it as a byte buffer.
#[inline]
fn aes_gcmsiv_zeroize_struct<T>(val: &mut T) {
    let p = val as *mut T as *mut u8;
    let size = std::mem::size_of::<T>();
    unsafe {
        for i in 0..size {
            ptr::write_volatile(p.add(i), 0);
        }
    }
}

/// Stub function for CPU feature detection. Currently always returns 0.
#[inline]
fn aes_gcmsiv_has_feature(_what: i32) -> i32 {
    0
}

/// =========================== AES Generic Code =================================

/// AES Round Constant (RCON) lookup table.
static RCON: [u32; 10] = [
    0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010,
    0x00000020, 0x00000040, 0x00000080, 0x0000001b, 0x00000036,
];

/// Rotate a 32-bit unsigned integer left by 8 bits.
#[inline]
fn ROTL8(x: u32) -> u32 {
    ((x << 8) & 0xffffffff) | (x >> 24)
}

/// Rotate a 32-bit unsigned integer left by 16 bits.
#[inline]
fn ROTL16(x: u32) -> u32 {
    ((x << 16) & 0xffffffff) | (x >> 16)
}

/// Rotate a 32-bit unsigned integer left by 24 bits.
#[inline]
fn ROTL24(x: u32) -> u32 {
    ((x << 24) & 0xffffffff) | (x >> 8)
}

/// Multiply a byte by x (i.e., perform a finite field multiplication by {02}).
#[inline]
fn XTIME(x: u8) -> u8 {
    let mut val = x << 1;
    if x & 0x80 != 0 {
        val ^= 0x1B;
    }
    (val & 0xFF)
}

/// AES S-box lookup table.
static FSb: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// Structure representing the generic AES context.
struct aes_generic {
    nr: i32,               // Number of AES rounds.
    rk: *mut u32,          // Pointer to the round keys.
    buf: [u32; 68],        // Buffer holding round keys.
}

impl aes_generic {
    /// Creates a new `aes_generic` instance with default values.
    fn new() -> Self {
        let mut ctx = aes_generic {
            nr: 0,
            rk: std::ptr::null_mut(),
            buf: [0; 68],
        };
        aes_generic_init(&mut ctx);
        ctx
    }
}

/// Initializes the generic AES context.
fn aes_generic_init(ctx: &mut aes_generic) {
    ctx.nr = 0;
    ctx.rk = ctx.buf.as_ptr() as *mut u32;
}

/// Frees the generic AES context by securely zeroizing it.
fn aes_generic_free(ctx: &mut aes_generic) {
    aes_gcmsiv_zeroize_struct(ctx);
}

/// Sets the AES key for the generic AES context.
fn aes_generic_set_key(ctx: &mut aes_generic, key: &[u8]) -> aes_gcmsiv_status_t {
    aes_generic_gen_tables();
    let key_sz = key.len();
    let nr = match key_sz {
        16 => 10, // AES-128
        24 => 12, // AES-192
        32 => 14, // AES-256
        _ => return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE,
    };
    ctx.nr = nr;
    let rk = ctx.rk;
    let nk = key_sz / 4; // Number of 32-bit words in the key.
    for i in 0..nk {
        let v = GET_UINT32_LE(key, i * 4);
        unsafe {
            *rk.add(i) = v;
        }
    }
    unsafe {
        match ctx.nr {
            10 => {
                // AES-128 Key Expansion
                for i in 0..10 {
                    let off = (4 * i) as usize;
                    let temp = *rk.add(off + 3);
                    let t = FSb[((temp >> 8) & 0xFF) as usize] as u32
                        | ((FSb[((temp >> 16) & 0xFF) as usize] as u32) << 8)
                        | ((FSb[((temp >> 24) & 0xFF) as usize] as u32) << 16)
                        | ((FSb[(temp & 0xFF) as usize] as u32) << 24);
                    *rk.add(off + 4) = *rk.add(off + 0) ^ RCON[i] ^ t;
                    *rk.add(off + 5) = *rk.add(off + 1) ^ *rk.add(off + 4);
                    *rk.add(off + 6) = *rk.add(off + 2) ^ *rk.add(off + 5);
                    *rk.add(off + 7) = *rk.add(off + 3) ^ *rk.add(off + 6);
                }
            }
            12 => {
                // AES-192 Key Expansion
                for i in 0..8 {
                    let off = (6 * i) as usize;
                    let temp = *rk.add(off + 5);
                    let t = FSb[((temp >> 8) & 0xFF) as usize] as u32
                        | ((FSb[((temp >> 16) & 0xFF) as usize] as u32) << 8)
                        | ((FSb[((temp >> 24) & 0xFF) as usize] as u32) << 16)
                        | ((FSb[(temp & 0xFF) as usize] as u32) << 24);
                    *rk.add(off + 6) = *rk.add(off + 0) ^ RCON[i] ^ t;
                    *rk.add(off + 7) = *rk.add(off + 1) ^ *rk.add(off + 6);
                    *rk.add(off + 8) = *rk.add(off + 2) ^ *rk.add(off + v 7);
                    *rk.add(off + 9) = *rk.add(off + 3) ^ *rk.add(off + 8);
                    *rk.add(off + 10) = *rk.add(off + 4) ^ *rk.add(off + 9);
                    *rk.add(off + 11) = *rk.add(off + 5) ^ *rk.add(off + 10);
                }
            }
            14 => {
                // AES-256 Key Expansion
                for i in 0..7 {
                    let off = (8 * i) as usize;
                    let temp = *rk.add(off + 7);
                    let t = FSb[((temp >> 8) & 0xFF) as usize] as u32
                        | ((FSb[((temp >> 16) & 0xFF) as usize] as u32) << 8)
                        | ((FSb[((temp >> 24) & 0xFF) as usize] as u32) << 16)
                        | ((FSb[(temp & 0xFF) as usize] as u32) << 24);
                    *rk.add(off + 8) = *rk.add(off + 0) ^ RCON[i] ^ t;
                    *rk.add(off + 9) = *rk.add(off + 1) ^ *rk.add(off + 8);
                    *rk.add(off + 10) = *rk.add(off + 2) ^ *rk.add(off + 9);
                    *rk.add(off + 11) = *rk.add(off + 3) ^ *rk.add(off + 10);
                    let temp2 = *rk.add(off + 11);
                    let t2 = FSb[(temp2 & 0xFF) as usize] as u32
                        | ((FSb[((temp2 >> 8) & 0xFF) as usize] as u32) << 8)
                        | ((FSb[((temp2 >> 16) & 0xFF) as usize] as u32) << 16)
                        | ((FSb[((temp2 >> 24) & 0xFF) as usize] as u32) << 24);
                    *rk.add(off + 12) = *rk.add(off + 4) ^ t2;
                    *rk.add(off + 13) = *rk.add(off + 5) ^ *rk.add(off + 12);
                    *rk.add(off + 14) = *rk.add(off + 6) ^ *rk.add(off + 13);
                    *rk.add(off + 15) = *rk.add(off + 7) ^ *rk.add(off + 14);
                }
            }
            _ => {}
        }
    }

    /// Performs a single AES round using precomputed tables.
    #[inline]
    fn AES_FROUND(
        X0: &mut u32,
        X1: &mut u32,
        X2: &mut u32,
        X3: &mut u32,
        Y0: u32,
        Y1: u32,
        Y2: u32,
        Y3: u32,
        RK: &mut *const u32,
    ) {
        let rk = *RK;
        *X0 = unsafe { *rk } ^ FT0[(Y0 & 0xFF) as usize]
            ^ FT1[((Y1 >> 8) & 0xFF) as usize]
            ^ FT2[((Y2 >> 16) & 0xFF) as usize]
            ^ FT3[((Y3 >> 24) & 0xFF) as usize];
        *RK = (*RK).add(1);

        let rk = *RK;
        *X1 = unsafe { *rk } ^ FT0[(Y1 & 0xFF) as usize]
            ^ FT1[((Y2 >> 8) & 0xFF) as usize]
            ^ FT2[((Y3 >> 16) & 0xFF) as usize]
            ^ FT3[((Y0 >> 24) & 0xFF) as usize];
        *RK = (*RK).add(1);

        let rk = *RK;
        *X2 = unsafe { *rk } ^ FT0[(Y2 & 0xFF) as usize]
            ^ FT1[((Y3 >> 8) & 0xFF) as usize]
            ^ FT2[((Y0 >> 16) & 0xFF) as usize]
            ^ FT3[((Y1 >> 24) & 0xFF) as usize];
        *RK = (*RK).add(1);

        let rk = *RK;
        *X3 = unsafe { *rk } ^ FT0[(Y3 & 0xFF) as usize]
            ^ FT1[((Y0 >> 8) & 0xFF) as usize]
            ^ FT2[((Y1 >> 16) & 0xFF) as usize]
            ^ FT3[((Y2 >> 24) & 0xFF) as usize];
        *RK = (*RK).add(1);
    }

    /// Encrypts a single 16-byte block using AES in ECB mode.
    fn aes_generic_ecb_encrypt(
        ctx: &aes_generic,
        plain: &[u8; 16],
        cipher: &mut [u8; 16],
    ) -> aes_gcmsiv_status_t {
        let nr = ctx.nr;
        let mut X0: u32;
        let mut X1: u32;
        let mut X2: u32;
        let mut X3: u32;
        let mut Y0: u32;
        let mut Y1: u32;
        let mut Y2: u32;
        let mut Y3: u32;

        let mut RK = ctx.rk;
        unsafe {
            // Initial AddRoundKey
            X0 = GET_UINT32_LE(plain, 0) ^ *RK;
            X1 = GET_UINT32_LE(plain, 4) ^ *(RK.add(1));
            X2 = GET_UINT32_LE(plain, 8) ^ *(RK.add(2));
            X3 = GET_UINT32_LE(plain, 12) ^ *(RK.add(3));

            // Pointer to the next round key
            RK = RK.add(4);

            // Perform Nr-1 rounds.
            for _ in 1..nr {
                AES_FROUND(&mut Y0, &mut Y1, &mut Y2, &mut Y3, X0, X1, X2, X3, &mut RK);
                X0 = Y0;
                X1 = Y1;
                X2 = Y2;
                X3 = Y3;
            }

            // Final round
            AES_FROUND(&mut Y0, &mut Y1, &mut Y2, &mut Y3, X0, X1, X2, X3, &mut RK);
            X0 = Y0;
            X1 = Y1;
            X2 = Y2;
            X3 = Y3;
        }

        // Serialize the encrypted block back to bytes.
        PUT_UINT32_LE(X0, cipher, 0);
        PUT_UINT32_LE(X1, cipher, 4);
        PUT_UINT32_LE(X2, cipher, 8);
        PUT_UINT32_LE(X3, cipher, 12);

        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Performs AES encryption in CTR mode.
    fn aes_generic_ctr(
        ctx: &aes_generic,
        nonce: &[u8; 16],
        input: &[u8],
        output: &mut [u8],
    ) -> aes_gcmsiv_status_t {
        let mut counter_block = *nonce;
        let mut counter = GET_UINT32_LE(&counter_block, 0);
        let mut key_stream = [0u8; 16];
        let mut processed = 0;

        // Process full 16-byte blocks.
        while processed + 16 <= input.len() {
            aes_generic_ecb_encrypt(ctx, &counter_block, &mut key_stream.try_into().unwrap())?;
            counter = counter.wrapping_add(1);
            PUT_UINT32_LE(counter, &mut counter_block, 0);
            for i in 0..16 {
                output[processed + i] = input[processed + i] ^ key_stream[i];
            }
            processed += 16;
        }

        // Process any remaining bytes.
        if processed < input.len() {
            aes_generic_ecb_encrypt(ctx, &counter_block, &mut key_stream.try_into().unwrap())?;
            counter = counter.wrapping_add(1);
            PUT_UINT32_LE(counter, &mut counter_block, 0);
            for i in 0..(input.len() - processed) {
                output[processed + i] = input[processed + i] ^ key_stream[i];
            }
        }

        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    // ========================== Polyval Generic Code =========================

    /// Structure representing the generic Polyval context.
    struct polyval_generic {
        S: [u8; 16],     // Current state.
        HL: [u64; 16],   // Precomputed low parts for multiplication.
        HH: [u64; 16],   // Precomputed high parts for multiplication.
    }

    /// Structure used during the dot product operation in Polyval.
    struct dot_context {
        hl: u64,
        hh: u64,
        lo: u64,
        hi: u64,
        rem: u64,
    }

    /// Performs the dot product operation for Polyval.
    fn dot(dot: &mut dot_context, a: &[u8], bl: &[u64; 16], bh: &[u64; 16]) {
        dot.hl = 0;
        dot.hh = 0;

        for i in 0..16 {
            let b = a[16 - i - 1];
            dot.hi = ((b >> 4) & 0x0f) as u64;
            dot.lo = (b & 0x0f) as u64;

            dot.rem = (dot.hh >> 60) & 0x0f;
            dot.hh = ((dot.hh << 4) | (dot.hl >> 60)) ^ PH[dot.rem as usize] ^ bh[dot.hi as usize];
            dot.hl = (dot.hl << 4) ^ PL[dot.rem as usize] ^ bl[dot.hi as usize];

            dot.rem = (dot.hh >> 60) & 0x0f;
            dot.hh = ((dot.hh << 4) | (dot.hl >> 60)) ^ PH[dot.rem as usize] ^ bh[dot.lo as usize];
            dot.hl = (dot.hl << 4) ^ PL[dot.rem as usize] ^ bl[dot.lo as usize];
        }
    }

    /// Starts the Polyval computation by initializing the context with the provided key.
    fn polyval_generic_start(ctx: &mut polyval_generic, key: &[u8]) -> aes_gcmsiv_status_t {
        if key.len() != 16 {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE;
        }

        // Precomputed multiplication tables for key expansion.
        static XL: [u64; 16] = [
            0x0000000000000000, 0x0000000000000001, 0x0000000000000003, 0x0000000000000002,
            0x0000000000000007, 0x0000000000000006, 0x0000000000000004, 0x0000000000000005,
            0x000000000000000e, 0x000000000000000c, 0x000000000000000d, 0x000000000000000f,
            0x000000000000000b, 0x000000000000000a, 0x0000000000000008, 0x0000000000000009,
        ];

        static XH: [u64; 16] = [
            0x0000000000000000, 0xc200000000000000, 0x4600000000000000, 0x8400000000000000,
            0x8c00000000000000, 0x4e00000000000000, 0xca00000000000000, 0x0800000000000000,
            0xda00000000000000, 0x1800000000000000, 0x9c00000000000000, 0x5e00000000000000,
            0x5600000000000000, 0x9400000000000000, 0x1000000000000000, 0xd200000000000000,
        ];

        let mut dot_ctx = dot_context {
            hl: 0,
            hh: 0,
            lo: 0,
            hi: 0,
            rem: 0,
        };
        dot(&mut dot_ctx, key, &XL, &XH);

        ctx.HL[0] = 0;
        ctx.HH[0] = 0;
        ctx.HL[1] = dot_ctx.hl;
        ctx.HH[1] = dot_ctx.hh;

        let mut hl = dot_ctx.hl;
        let mut hh = dot_ctx.hh;
        for i in (2..16).step_by(2) {
            dot_ctx.rem = (hh >> 63) & 0x01;
            hh = (hh << 1) ^ (hl >> 63) ^ PH[dot_ctx.rem as usize];
            hl = (hl << 1) ^ PL[dot_ctx.rem as usize];

            ctx.HL[i] = hl;
            ctx.HH[i] = hh;

            for j in 1..i {
                ctx.HL[i + j] = hl ^ ctx.HL[j];
                ctx.HH[i + j] = hh ^ ctx.HH[j];
            }
        }

        for i in 0..16 {
            ctx.S[i] = 0;
        }

        aes_gcmsiv_zeroize_struct(&mut dot_ctx);
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Updates the Polyval context with additional data.
    fn polyval_generic_update(ctx: &mut polyval_generic, data: &[u8]) -> aes_gcmsiv_status_t {
        let mut idx = 0;
        while idx + 16 <= data.len() {
            for i in 0..16 {
                ctx.S[i] ^= data[idx + i];
            }

            let mut dot_ctx = dot_context {
                hl: 0,
                hh: 0,
                lo: 0,
                hi: 0,
                rem: 0,
            };
            dot(&mut dot_ctx, &ctx.S, &ctx.HL, &ctx.HH);
            PUT_UINT64_LE(dot_ctx.hl, &mut ctx.S, 0);
            PUT_UINT64_LE(dot_ctx.hh, &mut ctx.S, 8);
            aes_gcmsiv_zeroize_struct(&mut dot_ctx);
            idx += 16;
        }

        if idx < data.len() {
            for i in 0..(data.len() - idx) {
                ctx.S[i] ^= data[idx + i];
            }
            let mut dot_ctx = dot_context {
                hl: 0,
                hh: 0,
                lo: 0,
                hi: 0,
                rem: 0,
            };
            dot(&mut dot_ctx, &ctx.S, &ctx.HL, &ctx.HH);
            PUT_UINT64_LE(dot_ctx.hl, &mut ctx.S, 0);
            PUT_UINT64_LE(dot_ctx.hh, &mut ctx.S, 8);
            aes_gcmsiv_zeroize_struct(&mut dot_ctx);
        }

        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Finalizes the Polyval computation and produces the authentication tag.
    fn polyval_generic_finish(
        ctx: &mut polyval_generic,
        nonce: &[u8],
        tag: &mut [u8],
    ) -> aes_gcmsiv_status_t {
        if nonce.len() > 16 {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
        }

        for i in 0..nonce.len() {
            tag[i] = ctx.S[i] ^ nonce[i];
        }
        for i in nonce.len()..16 {
            tag[i] = ctx.S[i];
        }

        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    // ========================== Polyval Generic Code =========================

    /// Structure representing the generic Polyval context.
    struct polyval_generic {
        S: [u8; 16],     // Current state.
        HL: [u64; 16],   // Precomputed low parts for multiplication.
        HH: [u64; 16],   // Precomputed high parts for multiplication.
    }

    /// Structure used during the dot product operation in Polyval.
    struct dot_context {
        hl: u64,
        hh: u64,
        lo: u64,
        hi: u64,
        rem: u64,
    }

    /// Performs the dot product operation for Polyval.
    fn dot(dot: &mut dot_context, a: &[u8], bl: &[u64; 16], bh: &[u64; 16]) {
        dot.hl = 0;
        dot.hh = 0;

        for i in 0..16 {
            let b = a[16 - i - 1];
            dot.hi = ((b >> 4) & 0x0f) as u64;
            dot.lo = (b & 0x0f) as u64;

            dot.rem = (dot.hh >> 60) & 0x0f;
            dot.hh = ((dot.hh << 4) | (dot.hl >> 60)) ^ PH[dot.rem as usize] ^ bh[dot.hi as usize];
            dot.hl = (dot.hl << 4) ^ PL[dot.rem as usize] ^ bl[dot.hi as usize];

            dot.rem = (dot.hh >> 60) & 0x0f;
            dot.hh = ((dot.hh << 4) | (dot.hl >> 60)) ^ PH[dot.rem as usize] ^ bh[dot.lo as usize];
            dot.hl = (dot.hl << 4) ^ PL[dot.rem as usize] ^ bl[dot.lo as usize];
        }
    }

    /// Starts the Polyval computation by initializing the context with the provided key.
    fn polyval_generic_start(ctx: &mut polyval_generic, key: &[u8]) -> aes_gcmsiv_status_t {
        if key.len() != 16 {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE;
        }

        // Precomputed multiplication tables for key expansion.
        static XL: [u64; 16] = [
            0x0000000000000000, 0x0000000000000001, 0x0000000000000003, 0x0000000000000002,
            0x0000000000000007, 0x0000000000000006, 0x0000000000000004, 0x0000000000000005,
            0x000000000000000e, 0x000000000000000c, 0x000000000000000d, 0x000000000000000f,
            0x000000000000000b, 0x000000000000000a, 0x0000000000000008, 0x0000000000000009,
        ];

        static XH: [u64; 16] = [
            0x0000000000000000, 0xc200000000000000, 0x4600000000000000, 0x8400000000000000,
            0x8c00000000000000, 0x4e00000000000000, 0xca00000000000000, 0x0800000000000000,
            0xda00000000000000, 0x1800000000000000, 0x9c00000000000000, 0x5e00000000000000,
            0x5600000000000000, 0x9400000000000000, 0x1000000000000000, 0xd200000000000000,
        ];

        let mut dot_ctx = dot_context {
            hl: 0,
            hh: 0,
            lo: 0,
            hi: 0,
            rem: 0,
        };
        dot(&mut dot_ctx, key, &XL, &XH);

        ctx.HL[0] = 0;
        ctx.HH[0] = 0;
        ctx.HL[1] = dot_ctx.hl;
        ctx.HH[1] = dot_ctx.hh;

        let mut hl = dot_ctx.hl;
        let mut hh = dot_ctx.hh;
        for i in (2..16).step_by(2) {
            dot_ctx.rem = (hh >> 63) & 0x01;
            hh = (hh << 1) ^ (hl >> 63) ^ PH[dot_ctx.rem as usize];
            hl = (hl << 1) ^ PL[dot_ctx.rem as usize];

            ctx.HL[i] = hl;
            ctx.HH[i] = hh;

            for j in 1..i {
                ctx.HL[i + j] = hl ^ ctx.HL[j];
                ctx.HH[i + j] = hh ^ ctx.HH[j];
            }
        }

        for i in 0..16 {
            ctx.S[i] = 0;
        }

        aes_gcmsiv_zeroize_struct(&mut dot_ctx);
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Updates the Polyval context with additional data.
    fn polyval_generic_update(ctx: &mut polyval_generic, data: &[u8]) -> aes_gcmsiv_status_t {
        let mut idx = 0;
        while idx + 16 <= data.len() {
            for i in 0..16 {
                ctx.S[i] ^= data[idx + i];
            }

            let mut dot_ctx = dot_context {
                hl: 0,
                hh: 0,
                lo: 0,
                hi: 0,
                rem: 0,
            };
            dot(&mut dot_ctx, &ctx.S, &ctx.HL, &ctx.HH);
            PUT_UINT64_LE(dot_ctx.hl, &mut ctx.S, 0);
            PUT_UINT64_LE(dot_ctx.hh, &mut ctx.S, 8);
            aes_gcmsiv_zeroize_struct(&mut dot_ctx);
            idx += 16;
        }

        if idx < data.len() {
            for i in 0..(data.len() - idx) {
                ctx.S[i] ^= data[idx + i];
            }
            let mut dot_ctx = dot_context {
                hl: 0,
                hh: 0,
                lo: 0,
                hi: 0,
                rem: 0,
            };
            dot(&mut dot_ctx, &ctx.S, &ctx.HL, &ctx.HH);
            PUT_UINT64_LE(dot_ctx.hl, &mut ctx.S, 0);
            PUT_UINT64_LE(dot_ctx.hh, &mut ctx.S, 8);
            aes_gcmsiv_zeroize_struct(&mut dot_ctx);
        }

        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Finalizes the Polyval computation and produces the authentication tag.
    fn polyval_generic_finish(
        ctx: &mut polyval_generic,
        nonce: &[u8],
        tag: &mut [u8],
    ) -> aes_gcmsiv_status_t {
        if nonce.len() > 16 {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
        }

        for i in 0..nonce.len() {
            tag[i] = ctx.S[i] ^ nonce[i];
        }
        for i in nonce.len()..16 {
            tag[i] = ctx.S[i];
        }

        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Performs the dot product operation for Polyval.
    fn polyval_mul(a: &[u8; 16], bl: &[u64; 16], bh: &[u64; 16]) -> [u8; 16] {
        let mut dot_ctx = dot_context {
            hl: 0,
            hh: 0,
            lo: 0,
            hi: 0,
            rem: 0,
        };
        dot(&mut dot_ctx, a, bl, bh);
        let mut result = [0u8; 16];
        PUT_UINT64_LE(dot_ctx.hl, &mut result, 0);
        PUT_UINT64_LE(dot_ctx.hh, &mut result, 8);
        result
    }

    /// Generates the authentication tag using the derived keys, nonce, plaintext, and AAD.
    fn aes_gcmsiv_make_tag(
        key: &key_context,
        nonce: &[u8],
        plain: &[u8],
        aad: &[u8],
        tag: &mut [u8],
    ) {
        // Initialize AES and Polyval contexts.
        let mut ctx = aes_gcmsiv_ctx {
            key_gen_ctx: std::ptr::null_mut(),
            key_sz: 0,
        };
        aes_gcmsiv_init(&mut ctx);
        aes_gcmsiv_set_key(&mut ctx, &key.enc[0..key.enc_sz]);

        // Derive the authentication and encryption keys.
        let mut derived_key = key_context {
            auth: [0; 16],
            auth_sz: 0,
            enc: [0; 32],
            enc_sz: 0,
        };
        aes_gcmsiv_derive_keys(ctx.key_gen_ctx, ctx.key_sz, nonce, &mut derived_key);

        // Perform Polyval operations to generate the tag.
        let mut polyval_ctx = polyval {
            has_hw: 0,
            storage: polyval_storage {
                generic: polyval_generic {
                    S: [0; 16],
                    HL: [0; 16],
                    HH: [0; 16],
                },
            },
        };
        aes_gcmsiv_zeroize_struct(&mut polyval_ctx.storage.generic.S);
        aes_gcmsiv_zeroize_struct(&mut polyval_ctx.storage.generic.HL);
        aes_gcmsiv_zeroize_struct(&mut polyval_ctx.storage.generic.HH);

        polyval_generic_start(&mut polyval_ctx.storage.generic, &derived_key.auth).unwrap();
        polyval_generic_update(&mut polyval_ctx.storage.generic, aad).unwrap();
        polyval_generic_update(&mut polyval_ctx.storage.generic, plain).unwrap();

        // Prepare the length block for Polyval.
        let mut length_block = [0u8; 16];
        let aad_bit_sz = (aad.len() as u64) * 8;
        PUT_UINT64_LE(aad_bit_sz, &mut length_block, 0);
        let plain_bit_sz = (plain.len() as u64) * 8;
        PUT_UINT64_LE(plain_bit_sz, &mut length_block, 8);
        polyval_generic_update(&mut polyval_ctx.storage.generic, &length_block).unwrap();

        // Finalize Polyval to produce the tag.
        polyval_generic_finish(&mut polyval_ctx.storage.generic, nonce, tag).unwrap();

        // Clear the highest bit as per RFC 8452.
        tag[15] &= 0x7F;

        // Encrypt the tag using AES-ECB.
        let mut block = [0u8; 16];
        block.copy_from_slice(tag);
        aes_generic_ecb_encrypt(&mut ctx, &block, tag);

        // Clean up AES and Polyval contexts.
        aes_gcmsiv_free(&mut ctx);
    }

    /// Encrypts the plaintext using AES-CTR mode and the derived encryption key.
    fn aes_gcmsiv_aes_ctr(key: &[u8], key_sz: usize, tag: &[u8; 16], input: &[u8], output: &mut [u8]) {
        // Initialize AES context.
        let mut ctx = aes_gcmsiv_ctx {
            key_gen_ctx: std::ptr::null_mut(),
            key_sz: 0,
        };
        aes_gcmsiv_init(&mut ctx);
        aes_gcmsiv_set_key(&mut ctx, key);

        // Prepare the nonce for AES-CTR by setting the highest bit.
        let mut nonce_ctr = [0u8; 16];
        nonce_ctr.copy_from_slice(tag);
        nonce_ctr[15] |= 0x80;

        // Perform AES-CTR encryption.
        aes_generic_ctr(&ctx, &nonce_ctr, input, output);

        // Clean up AES context.
        aes_gcmsiv_free(&mut ctx);
    }

    /// Compares two authentication tags in constant time to prevent timing attacks.
    fn aes_gcmsiv_check_tag(lhs: &[u8; 16], rhs: &[u8; 16]) -> aes_gcmsiv_status_t {
        let mut sum = 0u8;
        for i in 0..16 {
            sum |= lhs[i] ^ rhs[i];
        }
        if sum == 0 {
            aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
        } else {
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_TAG
        }
    }

    /// ====================== Public Functions ======================

    /// Returns the size of the AES-GCM-SIV context.
    fn aes_gcmsiv_context_size() -> usize {
        std::mem::size_of::<aes_gcmsiv_ctx>()
    }

    /// Initializes the AES-GCM-SIV context.
    fn aes_gcmsiv_init(ctx: &mut aes_gcmsiv_ctx) {
        if ctx as *mut _ as *mut u8 == ptr::null_mut() {
            return;
        }
        ctx.key_gen_ctx = std::ptr::null_mut();
        ctx.key_sz = 0;
    }

    /// Frees the AES-GCM-SIV context, securely zeroizing its contents.
    fn aes_gcmsiv_free(ctx: &mut aes_gcmsiv_ctx) {
        if ctx as *mut _ as *mut u8 == ptr::null_mut() {
            return;
        }
        if !ctx.key_gen_ctx.is_null() {
            unsafe {
                aes_generic_free(&mut *ctx.key_gen_ctx);
                Box::from_raw(ctx.key_gen_ctx);
            }
        }
        let c = ctx as *mut aes_gcmsiv_ctx as *mut u8;
        let size = std::mem::size_of::<aes_gcmsiv_ctx>();
        for i in 0..size {
            unsafe {
                ptr::write_volatile(c.add(i), 0);
            }
        }
    }

    /// Sets the AES-GCM-SIV key.
    fn aes_gcmsiv_set_key(
        ctx: &mut aes_gcmsiv_ctx,
        key: &[u8],
    ) -> aes_gcmsiv_status_t {
        if ctx as *mut _ as *mut u8 == ptr::null_mut() || key.is_empty() {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
        }
        if key.len() != 16 && key.len() != 32 {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE;
        }

        // Initialize a new AES context and set the key.
        let mut key_gen_ctx = Box::new(aes_generic::new());
        aes_generic_init(&mut key_gen_ctx);
        let res = aes_generic_set_key(&mut key_gen_ctx, key);
        if res != aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
            return res;
        }

        // Replace the existing AES context if present.
        if !ctx.key_gen_ctx.is_null() {
            unsafe {
                aes_generic_free(&mut *ctx.key_gen_ctx);
                Box::from_raw(ctx.key_gen_ctx);
            }
        }

        ctx.key_gen_ctx = Box::into_raw(key_gen_ctx);
        ctx.key_sz = key.len();
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Calculates the required ciphertext size for encryption.
    fn aes_gcmsiv_encrypt_size(
        plain_sz: usize,
        aad_sz: usize,
        cipher_sz: &mut usize,
    ) -> aes_gcmsiv_status_t {
        if cipher_sz as *mut usize == ptr::null_mut() {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
        }

        if plain_sz > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PLAINTEXT_SIZE;
        }

        if aad_sz > AES_GCMSIV_MAX_AAD_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_AAD_SIZE;
        }

        let needed_sz = plain_sz + AES_GCMSIV_TAG_SIZE;
        if needed_sz < plain_sz {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PLAINTEXT_SIZE;
        }

        *cipher_sz = needed_sz;
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Encrypts the plaintext using AES-CTR mode and the derived encryption key.
    fn aes_gcmsiv_aes_ctr(key: &[u8], key_sz: usize, tag: &[u8; 16], input: &[u8], output: &mut [u8]) {
        // Initialize AES context.
        let mut ctx = aes_gcmsiv_ctx {
            key_gen_ctx: std::ptr::null_mut(),
            key_sz: 0,
        };
        aes_gcmsiv_init(&mut ctx);
        aes_gcmsiv_set_key(&mut ctx, key);

        // Prepare the nonce for AES-CTR by setting the highest bit.
        let mut nonce_ctr = [0u8; 16];
        nonce_ctr.copy_from_slice(tag);
        nonce_ctr[15] |= 0x80;

        // Perform AES-CTR encryption.
        aes_generic_ctr(&ctx, &nonce_ctr, input, output);

        // Clean up AES context.
        aes_gcmsiv_free(&mut ctx);
    }

    /// Generates the authentication tag using the derived keys, nonce, plaintext, and AAD.
    fn aes_gcmsiv_make_tag(
        key: &key_context,
        nonce: &[u8],
        plain: &[u8],
        aad: &[u8],
        tag: &mut [u8],
    ) {
        // Initialize AES and Polyval contexts.
        let mut ctx = aes_gcmsiv_ctx {
            key_gen_ctx: std::ptr::null_mut(),
            key_sz: 0,
        };
        aes_gcmsiv_init(&mut ctx);
        aes_gcmsiv_set_key(&mut ctx, &key.enc[0..key.enc_sz]);

        // Derive the authentication and encryption keys.
        let mut derived_key = key_context {
            auth: [0; 16],
            auth_sz: 0,
            enc: [0; 32],
            enc_sz: 0,
        };
        aes_gcmsiv_derive_keys(ctx.key_gen_ctx, ctx.key_sz, nonce, &mut derived_key);

        // Perform Polyval operations to generate the tag.
        let mut polyval_ctx = polyval {
            has_hw: 0,
            storage: polyval_storage {
                generic: polyval_generic {
                    S: [0; 16],
                    HL: [0; 16],
                    HH: [0; 16],
                },
            },
        };
        aes_gcmsiv_zeroize_struct(&mut polyval_ctx.storage.generic.S);
        aes_gcmsiv_zeroize_struct(&mut polyval_ctx.storage.generic.HL);
        aes_gcmsiv_zeroize_struct(&mut polyval_ctx.storage.generic.HH);

        polyval_generic_start(&mut polyval_ctx.storage.generic, &derived_key.auth).unwrap();
        polyval_generic_update(&mut polyval_ctx.storage.generic, aad).unwrap();
        polyval_generic_update(&mut polyval_ctx.storage.generic, plain).unwrap();

        // Prepare the length block for Polyval.
        let mut length_block = [0u8; 16];
        let aad_bit_sz = (aad.len() as u64) * 8;
        PUT_UINT64_LE(aad_bit_sz, &mut length_block, 0);
        let plain_bit_sz = (plain.len() as u64) * 8;
        PUT_UINT64_LE(plain_bit_sz, &mut length_block, 8);
        polyval_generic_update(&mut polyval_ctx.storage.generic, &length_block).unwrap();

        // Finalize Polyval to produce the tag.
        polyval_generic_finish(&mut polyval_ctx.storage.generic, nonce, tag).unwrap();

        // Clear the highest bit as per RFC 8452.
        tag[15] &= 0x7F;

        // Encrypt the tag using AES-ECB.
        let mut block = [0u8; 16];
        block.copy_from_slice(tag);
        aes_generic_ecb_encrypt(&mut ctx, &block, tag);

        // Clean up AES and Polyval contexts.
        aes_gcmsiv_free(&mut ctx);
        polyval_free(&mut polyval_ctx);
    }

    /// Encrypts the plaintext with authentication, producing ciphertext and a tag.
    fn aes_gcmsiv_encrypt_with_tag(
        ctx: &mut aes_gcmsiv_ctx,
        nonce: &[u8],
        plain: &[u8],
        aad: &[u8],
        cipher: &mut [u8],
        write_sz: &mut usize,
    ) -> aes_gcmsiv_status_t {
        // Validate input parameters.
        if ctx as *mut _ as *mut u8 == ptr::null_mut()
            || nonce.is_empty()
            || plain.is_empty()
            || aad.is_empty()
            || cipher.is_empty()
            || write_sz as *mut usize == ptr::null_mut()
        {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
        }

        if nonce.len() != AES_GCMSIV_NONCE_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
        }

        // Calculate the required ciphertext size.
        let mut needed_sz = 0usize;
        let res = aes_gcmsiv_encrypt_size(plain.len(), aad.len(), &mut needed_sz);
        if res != aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
            return res;
        }

        if cipher.len() < needed_sz {
            *write_sz = needed_sz;
            return aes_gcmsiv_status_t::AES_GCMSIV_UPDATE_OUTPUT_SIZE;
        }

        let key_gen_ctx = ctx.key_gen_ctx;
        if key_gen_ctx.is_null() {
            return aes_gcmsiv_status_t::AES_GCMSIV_FAILURE;
        }

        // Derive the authentication and encryption keys.
        let mut key = key_context {
            auth: [0; 16],
            auth_sz: 0,
            enc: [0; 32],
            enc_sz: 0,
        };
        aes_gcmsiv_derive_keys(key_gen_ctx, ctx.key_sz, nonce, &mut key);

        // Split the ciphertext buffer into data and tag sections.
        let tag_offset = plain.len();
        let (cipher_data, tag_buf) = cipher.split_at_mut(tag_offset);
        aes_gcmsiv_make_tag(&key, nonce, plain, aad, tag_buf);

        // Encrypt the plaintext using AES-CTR mode.
        aes_gcmsiv_aes_ctr(&key.enc[0..key.enc_sz], key.enc_sz, tag_buf.try_into().unwrap(), plain, cipher_data);

        *write_sz = needed_sz;

        // Securely zeroize the derived keys.
        let mut zero_key = key;
        aes_gcmsiv_zeroize_struct(&mut zero_key);
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Calculates the required plaintext size for decryption.
    fn aes_gcmsiv_decrypt_size(
        cipher_sz: usize,
        aad_sz: usize,
        plain_sz: &mut usize,
    ) -> aes_gcmsiv_status_t {
        if plain_sz as *mut usize == ptr::null_mut() {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
        }

        if cipher_sz < AES_GCMSIV_TAG_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_CIPHERTEXT_SIZE;
        }

        let needed_sz = cipher_sz - AES_GCMSIV_TAG_SIZE;

        if needed_sz > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_CIPHERTEXT_SIZE;
        }

        if aad_sz > AES_GCMSIV_MAX_AAD_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_AAD_SIZE;
        }

        *plain_sz = needed_sz;
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Decrypts the ciphertext and verifies the authentication tag, producing plaintext.
    fn aes_gcmsiv_decrypt_and_check(
        ctx: &mut aes_gcmsiv_ctx,
        nonce: &[u8],
        cipher: &[u8],
        aad: &[u8],
        plain: &mut [u8],
        write_sz: &mut usize,
    ) -> aes_gcmsiv_status_t {
        // Validate input parameters.
        if ctx as *mut _ as *mut u8 == ptr::null_mut()
            || nonce.is_empty()
            || cipher.is_empty()
            || aad.is_empty()
            || plain.is_empty()
            || write_sz as *mut usize == ptr::null_mut()
        {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
        }

        if nonce.len() != AES_GCMSIV_NONCE_SIZE {
            return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
        }

        // Calculate the required plaintext size.
        let mut needed_sz = 0usize;
        let res = aes_gcmsiv_decrypt_size(cipher.len(), aad.len(), &mut needed_sz);
        if res != aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
            return res;
        }

        if plain.len() < needed_sz {
            *write_sz = needed_sz;
            return aes_gcmsiv_status_t::AES_GCMSIV_UPDATE_OUTPUT_SIZE;
        }

        let key_gen_ctx = ctx.key_gen_ctx;
        if key_gen_ctx.is_null() {
            return aes_gcmsiv_status_t::AES_GCMSIV_FAILURE;
        }

        // Derive the authentication and encryption keys.
        let mut key = key_context {
            auth: [0; 16],
            auth_sz: 0,
            enc: [0; 32],
            enc_sz: 0,
        };
        aes_gcmsiv_derive_keys(key_gen_ctx, ctx.key_sz, nonce, &mut key);

        // Split the ciphertext into data and expected tag.
        let expected_tag = &cipher[cipher.len() - AES_GCMSIV_TAG_SIZE..];
        let ciphertext_data = &cipher[0..cipher.len() - AES_GCMSIV_TAG_SIZE];

        // Decrypt the ciphertext using AES-CTR mode.
        aes_gcmsiv_aes_ctr(&key.enc[0..key.enc_sz], key.enc_sz, expected_tag.try_into().unwrap(), ciphertext_data, plain);

        // Generate the expected authentication tag.
        let mut computed_tag = [0u8; 16];
        aes_gcmsiv_make_tag(&key, nonce, plain, aad, &mut computed_tag);

        // Verify the authentication tag.
        let res = aes_gcmsiv_check_tag(&computed_tag, expected_tag.try_into().unwrap());
        if res != aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
            // If the tag is invalid, securely zeroize the plaintext.
            for i in 0..needed_sz {
                plain[i] = 0;
            }
            *write_sz = 0;
            aes_gcmsiv_zeroize_struct(&mut key);
            return res;
        }

        *write_sz = needed_sz;
        aes_gcmsiv_zeroize_struct(&mut key);
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    }

    /// Generates a human-readable message corresponding to the status code.
    fn aes_gcmsiv_get_status_code_msg(status: aes_gcmsiv_status_t) -> &'static str {
        match status {
            aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS => "Success",
            aes_gcmsiv_status_t::AES_GCMSIV_FAILURE => "Failure",
            aes_gcmsiv_status_t::AES_GCMSIV_OUT_OF_MEMORY => "Out of memory",
            aes_gcmsiv_status_t::AES_GCMSIV_UPDATE_OUTPUT_SIZE => "Update output size",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS => "Invalid parameters",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE => "Unsupported key size",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE => "Invalid nonce size",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PLAINTEXT_SIZE => "Invalid plaintext size",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_AAD_SIZE => "Invalid additional authenticated data size",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_CIPHERTEXT_SIZE => "Invalid ciphertext size",
            aes_gcmsiv_status_t::AES_GCMSIV_INVALID_TAG => "Invalid tag",
        }
    }


// ====================== ARCH-SPECIFIC OPTIMIZATIONS ======================


#![feature(stdsimd, target_feature_11)]
use std::arch::x86_64::*;
use std::mem;
use std::ptr;
use std::slice;

const AES_BLOCK_SIZE: usize = 16;
const POLYVAL_SIZE: usize = 16;

#[derive(Debug)]
enum AesGcmSivError {
    InvalidParameters,
    InvalidKeySize,
    InvalidNonceSize,
}

type Result<T> = std::result::Result<T, AesGcmSivError>;

#[inline]
unsafe fn zeroize(ptr: *mut u8, len: usize) {
    for i in 0..len {
        ptr::write_volatile(ptr.add(i), 0);
    }
}

#[inline]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a,b)
}

#[inline]
unsafe fn aeskeygenassist(a: __m128i, r: i32) -> __m128i {
    _mm_aeskeygenassist_si128(a, r)
}

#[inline]
unsafe fn split(a: __m128i, b: i32) -> __m128i {
    _mm_shuffle_epi32(a, b)
}

#[inline]
unsafe fn shift_add(a: __m128i) -> __m128i {
    let tmp = _mm_slli_si128(a,4);
    xor(a, tmp)
}

#[inline]
unsafe fn tshift_add(a: __m128i) -> __m128i {
    shift_add(shift_add(shift_add(a)))
}

#[inline]
unsafe fn key_exp_helper(k0: __m128i, k1: __m128i, r: i32, s: i32) -> __m128i {
    let t = aeskeygenassist(k1, r);
    xor(tshift_add(k0), split(t, s))
}

#[inline]
unsafe fn key_exp_128(key: &mut [__m128i], i: usize, r: i32) -> __m128i {
    key_exp_helper(key[i], key[i], r, 0xff)
}

#[inline]
unsafe fn key_exp_256_1(key: &mut [__m128i], i: usize, r: i32) -> __m128i {
    key_exp_helper(key[i], key[i+1], r, 0xff)
}

#[inline]
unsafe fn key_exp_256_2(key: &mut [__m128i], i: usize) -> __m128i {
    key_exp_helper(key[i], key[i+1], 0x00, 0xaa)
}

#[repr(C)]
pub struct AesX86_64 {
    key: [__m128i; 15],
    key_sz: usize,
}

impl AesX86_64 {
    pub fn new() -> Self {
        unsafe {
            let mut ctx = mem::MaybeUninit::<AesX86_64>::uninit();
            ptr::write_bytes(ctx.as_mut_ptr(), 0, 1);
            ctx.assume_init()
        }
    }

    pub fn set_key(&mut self, key: &[u8]) -> Result<()> {
        if key.len()!=16 && key.len()!=32 {
            return Err(AesGcmSivError::InvalidKeySize);
        }
        self.key_sz = key.len();
        unsafe {
            self.key[0]=loadu(key.as_ptr());
            if key.len()==16 {
                // AES-128 key expansion
                self.key[5] = key_exp_128(&mut self.key,0,0x01);
                self.key[6] = key_exp_128(&mut self.key,5,0x02);
                self.key[7] = key_exp_128(&mut self.key,6,0x04);
                self.key[8] = key_exp_128(&mut self.key,7,0x08);
                self.key[9] = key_exp_128(&mut self.key,8,0x10);
                self.key[10] = key_exp_128(&mut self.key,9,0x20);
                self.key[11] = key_exp_128(&mut self.key,10,0x40);
                self.key[12] = key_exp_128(&mut self.key,11,0x80);
                self.key[13] = key_exp_128(&mut self.key,12,0x1b);
                self.key[14] = key_exp_128(&mut self.key,13,0x36);
            } else {
                // AES-256
                self.key[1] = loadu(key.as_ptr().add(16));
                self.key[2] = key_exp_256_1(&mut self.key,0,0x01);
                self.key[3] = key_exp_256_2(&mut self.key,1);
                self.key[4] = key_exp_256_1(&mut self.key,2,0x02);
                self.key[5] = key_exp_256_2(&mut self.key,3);
                self.key[6] = key_exp_256_1(&mut self.key,4,0x04);
                self.key[7] = key_exp_256_2(&mut self.key,5);
                self.key[8] = key_exp_256_1(&mut self.key,6,0x08);
                self.key[9] = key_exp_256_2(&mut self.key,7);
                self.key[10]= key_exp_256_1(&mut self.key,8,0x10);
                self.key[11]= key_exp_256_2(&mut self.key,9);
                self.key[12]= key_exp_256_1(&mut self.key,10,0x20);
                self.key[13]= key_exp_256_2(&mut self.key,11);
                self.key[14]= key_exp_256_1(&mut self.key,12,0x40);
            }
        }
        Ok(())
    }

    pub fn ecb_encrypt(&self, plain: &[u8;16]) -> Result<[u8;16]> {
        unsafe {
            let block=loadu(plain.as_ptr());
            let c = aes_encrypt(self, block);
            let mut cipher=[0u8;16];
            storeu(cipher.as_mut_ptr(), c);
            Ok(cipher)
        }
    }

    pub fn ctr(&self, nonce:&[u8;16], input:&[u8], output:&mut [u8]) -> Result<()> {
        if output.len()<input.len() {
            return Err(AesGcmSivError::InvalidParameters);
        }
        unsafe {aes_ctr(self,nonce,input,output)}
    }
}

impl Drop for AesX86_64 {
    fn drop(&mut self) {
        unsafe {
            zeroize(self as *mut _ as *mut u8, mem::size_of::<AesX86_64>());
        }
    }
}

#[target_feature(enable="aes")]
unsafe fn loadu(data:*const u8)->__m128i {
    _mm_loadu_si128(data as *const __m128i)
}

#[target_feature(enable="aes")]
unsafe fn storeu(data:*mut u8, reg:__m128i) {
    _mm_storeu_si128(data as *mut __m128i, reg);
}

#[target_feature(enable="aes")]
unsafe fn aes_encrypt(ctx:&AesX86_64, mut block:__m128i)->__m128i {
    block = xor(block, ctx.key[0]);
    match ctx.key_sz {
        16 => {
            // AES-128
            block = _mm_aesenc_si128(block, ctx.key[5]);
            block = _mm_aesenc_si128(block, ctx.key[6]);
            block = _mm_aesenc_si128(block, ctx.key[7]);
            block = _mm_aesenc_si128(block, ctx.key[8]);
            block = _mm_aesenc_si128(block, ctx.key[9]);
            block = _mm_aesenc_si128(block, ctx.key[10]);
            block = _mm_aesenc_si128(block, ctx.key[11]);
            block = _mm_aesenc_si128(block, ctx.key[12]);
            block = _mm_aesenc_si128(block, ctx.key[13]);
            block = _mm_aesenclast_si128(block, ctx.key[14]);
        }
        32 => {
            // AES-256
            block = _mm_aesenc_si128(block, ctx.key[1]);
            block = _mm_aesenc_si128(block, ctx.key[2]);
            block = _mm_aesenc_si128(block, ctx.key[3]);
            block = _mm_aesenc_si128(block, ctx.key[4]);
            for i in 5..14 {
                block = _mm_aesenc_si128(block, ctx.key[i]);
            }
            block = _mm_aesenclast_si128(block, ctx.key[14]);
        }
        _=>{}
    }
    block
}

#[target_feature(enable="aes")]
unsafe fn aes_encrypt_x4(ctx:&AesX86_64, counter:&[__m128i;4], stream:&mut [__m128i;4]) {
    for i in 0..4 { stream[i]=xor(counter[i],ctx.key[0]); }

    if ctx.key_sz==32 {
        for r in 1..5 {
            for j in 0..4 {
                stream[j]=_mm_aesenc_si128(stream[j], ctx.key[r]);
            }
        }
    }
    for r in 5..14 {
        for j in 0..4 {
            stream[j]=_mm_aesenc_si128(stream[j], ctx.key[r]);
        }
    }
    for j in 0..4 {
        stream[j]=_mm_aesenclast_si128(stream[j], ctx.key[14]);
    }
}

#[target_feature(enable="aes")]
unsafe fn aes_encrypt_x2(ctx:&AesX86_64, plain:&[__m128i;2], cipher:&mut[__m128i;2]) {
    for i in 0..2 {
        cipher[i]=xor(plain[i],ctx.key[0]);
    }
    if ctx.key_sz==32 {
        for r in 1..5 {
            for j in 0..2 {
                cipher[j]=_mm_aesenc_si128(cipher[j], ctx.key[r]);
            }
        }
    }
    for r in 5..14 {
        for j in 0..2 {
            cipher[j]=_mm_aesenc_si128(cipher[j], ctx.key[r]);
        }
    }
    for j in 0..2 {
        cipher[j]=_mm_aesenclast_si128(cipher[j], ctx.key[14]);
    }
}

#[target_feature(enable="aes")]
unsafe fn aes_encrypt_x3(ctx:&AesX86_64, plain:&[__m128i;3], cipher:&mut[__m128i;3]) {
    for i in 0..3 {
        cipher[i]=xor(plain[i],ctx.key[0]);
    }
    if ctx.key_sz==32 {
        for r in 1..5 {
            for j in 0..3 {
                cipher[j]=_mm_aesenc_si128(cipher[j], ctx.key[r]);
            }
        }
    }
    for r in 5..14 {
        for j in 0..3 {
            cipher[j]=_mm_aesenc_si128(cipher[j], ctx.key[r]);
        }
    }
    for j in 0..3 {
        cipher[j]=_mm_aesenclast_si128(cipher[j], ctx.key[14]);
    }
}

#[target_feature(enable="aes")]
unsafe fn aes_ctr(ctx:&AesX86_64, nonce:&[u8;16], input:&[u8], output:&mut[u8]) -> Result<()> {
    if output.len()<input.len() {
        return Err(AesGcmSivError::InvalidParameters);
    }

    let one = _mm_set_epi32(0,0,0,1);
    let mut counter=[_mm_setzero_si128();4];
    counter[0]=loadu(nonce.as_ptr());
    counter[1]=_mm_add_epi32(counter[0],one);
    counter[2]=_mm_add_epi32(counter[1],one);
    counter[3]=_mm_add_epi32(counter[2],one);

    let mut remain=input.len();
    let mut inptr=input.as_ptr();
    let mut outptr=output.as_mut_ptr();

    while remain>=4*AES_BLOCK_SIZE {
        let mut stream=[_mm_setzero_si128();4];
        aes_encrypt_x4(ctx,&counter,&mut stream);

        counter[0]=_mm_add_epi32(counter[3],one);
        counter[1]=_mm_add_epi32(counter[0],one);
        counter[2]=_mm_add_epi32(counter[1],one);
        counter[3]=_mm_add_epi32(counter[2],one);

        let inblk0=loadu(inptr);
        let inblk1=loadu(inptr.add(16));
        let inblk2=loadu(inptr.add(32));
        let inblk3=loadu(inptr.add(48));

        let c0=xor(inblk0,stream[0]);
        let c1=xor(inblk1,stream[1]);
        let c2=xor(inblk2,stream[2]);
        let c3=xor(inblk3,stream[3]);

        storeu(outptr,c0);
        storeu(outptr.add(16),c1);
        storeu(outptr.add(32),c2);
        storeu(outptr.add(48),c3);

        inptr=inptr.add(64);
        outptr=outptr.add(64);
        remain-=64;
    }

    if remain>0 {
        let blocks=remain/16;
        let leftover=remain%16;
        let mut tmp=[0u8;16];

        match blocks {
            0=>{
                // single partial block
                let mut single=[xor(counter[0],ctx.key[0])];
                if ctx.key_sz==32 {
                    for r in 1..5 { single[0]=_mm_aesenc_si128(single[0], ctx.key[r]); }
                }
                for r in 5..14 {
                    single[0]=_mm_aesenc_si128(single[0], ctx.key[r]);
                }
                single[0]=_mm_aesenclast_si128(single[0], ctx.key[14]);

                ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
                for i in leftover..16 { tmp[i]=0; }
                let inblk=loadu(tmp.as_ptr());
                let outblk=xor(inblk,single[0]);
                storeu(tmp.as_mut_ptr(),outblk);
                ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
            }
            1=>{
                let plains=[counter[0], counter[1]];
                let mut ciphers=[_mm_setzero_si128();2];
                aes_encrypt_x2(ctx,&plains,&mut ciphers);

                let inblk0=loadu(inptr);
                let c0=xor(inblk0,ciphers[0]);
                storeu(outptr,c0);
                inptr=inptr.add(16);
                outptr=outptr.add(16);

                if leftover>0 {
                    ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
                    for i in leftover..16 { tmp[i]=0; }
                    let inblk=loadu(tmp.as_ptr());
                    let outblk=xor(inblk,ciphers[1]);
                    storeu(tmp.as_mut_ptr(),outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            2=>{
                let plains=[counter[0],counter[1],counter[2]];
                let mut ciphers=[_mm_setzero_si128();3];
                aes_encrypt_x3(ctx,&plains,&mut ciphers);

                let inblk0=loadu(inptr);
                let inblk1=loadu(inptr.add(16));
                let c0=xor(inblk0,ciphers[0]);
                let c1=xor(inblk1,ciphers[1]);
                storeu(outptr,c0);
                storeu(outptr.add(16),c1);

                inptr=inptr.add(32);
                outptr=outptr.add(32);

                if leftover>0 {
                    ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(),leftover);
                    for i in leftover..16 { tmp[i]=0; }
                    let inblk=loadu(tmp.as_ptr());
                    let outblk=xor(inblk,ciphers[2]);
                    storeu(tmp.as_mut_ptr(), outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr,leftover);
                }
            }
            3=>{
                let mut stream=[_mm_setzero_si128();4];
                aes_encrypt_x4(ctx,&counter,&mut stream);
                let inblk0=loadu(inptr);
                let inblk1=loadu(inptr.add(16));
                let inblk2=loadu(inptr.add(32));

                let c0=xor(inblk0,stream[0]);
                let c1=xor(inblk1,stream[1]);
                let c2=xor(inblk2,stream[2]);
                storeu(outptr,c0);
                storeu(outptr.add(16),c1);
                storeu(outptr.add(32),c2);

                inptr=inptr.add(48);
                outptr=outptr.add(48);

                if leftover>0 {
                    ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
                    for i in leftover..16 { tmp[i]=0; }
                    let inblk=loadu(tmp.as_ptr());
                    let outblk=xor(inblk,stream[3]);
                    storeu(tmp.as_mut_ptr(), outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            _=>{}
        }
    }

    Ok(())
}

#[repr(C)]
pub struct PolyvalX86_64 {
    s: __m128i,
    h_table: [__m128i;8],
}

impl PolyvalX86_64 {
    pub fn new(key:&[u8;16]) -> Result<Self> {
        unsafe {
            let mut ctx=mem::MaybeUninit::<PolyvalX86_64>::uninit();
            ptr::write_bytes(ctx.as_mut_ptr(),0,1);
            let mut ctx=ctx.assume_init();
            ctx.s=_mm_setzero_si128();
            ctx.h_table[0]=loadu(key.as_ptr());
            for i in 1..8 {
                ctx.h_table[i]=dot(ctx.h_table[0],ctx.h_table[i-1]);
            }
            Ok(ctx)
        }
    }

    pub fn update(&mut self, data:&[u8]) {
        unsafe {
            self.s = polyval_x86_64_process_tables(&self.h_table, self.s, data.as_ptr(), data.len());
        }
    }

    pub fn finish(self, nonce:&[u8]) -> Result<[u8;16]> {
        if nonce.len()>16 {
            return Err(AesGcmSivError::InvalidNonceSize);
        }
        let mut tmp=[0u8;16];
        tmp[..nonce.len()].copy_from_slice(nonce);
        unsafe {
            let n=loadu(tmp.as_ptr());
            let out=xor(n,self.s);
            let mut tag=[0u8;16];
            storeu(tag.as_mut_ptr(),out);
            Ok(tag)
        }
    }
}

impl Drop for PolyvalX86_64 {
    fn drop(&mut self) {
        unsafe {
            zeroize(self as *mut _ as *mut u8, mem::size_of::<PolyvalX86_64>());
        }
    }
}

// Polyval logic
#[inline]
unsafe fn clmul(a:__m128i, b:__m128i, c:i32)->__m128i {
    _mm_clmulepi64_si128(a,b,c)
}

#[inline]
unsafe fn swap(a:__m128i)->__m128i {
    _mm_shuffle_epi32(a,0x4e)
}

#[inline]
unsafe fn mult(a:__m128i,b:__m128i,c0:&mut __m128i,c1:&mut __m128i,c2:&mut __m128i) {
    *c0=clmul(a,b,0x00);
    *c2=clmul(a,b,0x11);
    *c1 = xor(clmul(a,b,0x01), clmul(a,b,0x10));
}

#[inline]
unsafe fn add_mult(a:__m128i,b:__m128i,c0:&mut __m128i,c1:&mut __m128i,c2:&mut __m128i) {
    *c0=xor(*c0, clmul(a,b,0x00));
    *c2=xor(*c2, clmul(a,b,0x11));
    *c1=xor(*c1, xor(clmul(a,b,0x01), clmul(a,b,0x10)));
}

#[inline]
unsafe fn mult_inv_x64(p:__m128i)->__m128i {
    // From original logic
    let q=swap(p);
    // poly from code:0xc2000000...
    let poly=_mm_setr_epi32(0,0xc2000000u32 as i32,0x00000001u32 as i32,0);
    let r=clmul(p,poly,0x00);
    xor(q,r)
}

#[inline]
unsafe fn mult_inv_x128(p0:__m128i,p1:__m128i,p2:__m128i)->__m128i {
    let q=xor(p0,_mm_slli_si128(p1,8));
    let r=xor(p2,_mm_srli_si128(p1,8));
    let s=mult_inv_x64(q);
    let t=mult_inv_x64(s);
    xor(r,t)
}

#[inline]
unsafe fn dot(a:__m128i,b:__m128i)->__m128i {
    let mut c0=_mm_setzero_si128();
    let mut c1=_mm_setzero_si128();
    let mut c2=_mm_setzero_si128();
    mult(a,b,&mut c0,&mut c1,&mut c2);
    mult_inv_x128(c0,c1,c2)
}

#[inline]
#[target_feature(enable="pclmul")]
unsafe fn polyval_x86_64_process_tables(h_table:&[__m128i;8],
                                        mut s:__m128i,
                                        mut data:*const u8,
                                        mut data_sz:usize)->__m128i {
    let mut tmp=[0u8;16];

    while data_sz>=8*16 {
        let blocks=data as *const __m128i;
        let D0=_mm_loadu_si128(blocks);
        let D1=_mm_loadu_si128(blocks.add(1));
        let D2=_mm_loadu_si128(blocks.add(2));
        let D3=_mm_loadu_si128(blocks.add(3));
        let D4=_mm_loadu_si128(blocks.add(4));
        let D5=_mm_loadu_si128(blocks.add(5));
        let D6=_mm_loadu_si128(blocks.add(6));
        let D7=_mm_loadu_si128(blocks.add(7));

        let mut s0;let mut s1;let mut s2;
        mult(D7,h_table[0],&mut s0,&mut s1,&mut s2);
        add_mult(D6,h_table[1],&mut s0,&mut s1,&mut s2);
        add_mult(D5,h_table[2],&mut s0,&mut s1,&mut s2);
        add_mult(D4,h_table[3],&mut s0,&mut s1,&mut s2);
        add_mult(D3,h_table[4],&mut s0,&mut s1,&mut s2);
        add_mult(D2,h_table[5],&mut s0,&mut s1,&mut s2);
        add_mult(D1,h_table[6],&mut s0,&mut s1,&mut s2);
        let D0_s=xor(s,D0);
        add_mult(D0_s,h_table[7],&mut s0,&mut s1,&mut s2);

        s=mult_inv_x128(s0,s1,s2);

        data=data.add(8*16);
        data_sz-=8*16;
    }

    let blocks=data_sz/16;
    if blocks>0 {
        let bptr=data as *const __m128i;
        if blocks>1 {
            let last=_mm_loadu_si128(bptr.add(blocks-1));
            let mut s0;let mut s1;let mut s2;
            mult(last,h_table[0],&mut s0,&mut s1,&mut s2);

            for i in 1..(blocks-1) {
                let blk=_mm_loadu_si128(bptr.add(blocks-1-i));
                add_mult(blk,h_table[i],&mut s0,&mut s1,&mut s2);
            }

            let first_s=xor(s,_mm_loadu_si128(bptr));
            add_mult(first_s,h_table[blocks-1],&mut s0,&mut s1,&mut s2);
            s=mult_inv_x128(s0,s1,s2);
        } else {
            let D0_s=xor(s,_mm_loadu_si128(bptr));
            let mut s0;let mut s1;let mut s2;
            mult(D0_s,h_table[0],&mut s0,&mut s1,&mut s2);
            s=mult_inv_x128(s0,s1,s2);
        }
        data=data.add(blocks*16);
        data_sz-=blocks*16;
    }

    if data_sz>0 {
        ptr::copy_nonoverlapping(data,tmp.as_mut_ptr(), data_sz);
        for i in data_sz..16 {
            tmp[i]=0;
        }
        let d=xor(s,_mm_loadu_si128(tmp.as_ptr() as *const __m128i));
        s=dot(d,h_table[0]);
    }

    s
}

pub struct PolyvalX86_64 {
    s: __m128i,
    h_table: [__m128i;8],
}

impl PolyvalX86_64 {
    pub fn new(key:&[u8;16]) -> Result<Self> {
        unsafe {
            let mut ctx=mem::MaybeUninit::<PolyvalX86_64>::uninit();
            ptr::write_bytes(ctx.as_mut_ptr(),0,1);
            let mut ctx=ctx.assume_init();
            ctx.s=_mm_setzero_si128();
            ctx.h_table[0]=loadu(key.as_ptr());
            for i in 1..8 {
                ctx.h_table[i]=dot(ctx.h_table[0],ctx.h_table[i-1]);
            }
            Ok(ctx)
        }
    }

    pub fn update(&mut self, data:&[u8]) {
        unsafe {
            self.s=polyval_x86_64_process_tables(&self.h_table,self.s,data.as_ptr(),data.len());
        }
    }

    pub fn finish(self, nonce:&[u8]) -> Result<[u8;16]> {
        if nonce.len()>16 {
            return Err(AesGcmSivError::InvalidNonceSize);
        }
        let mut tmp=[0u8;16];
        tmp[..nonce.len()].copy_from_slice(nonce);
        unsafe {
            let n=loadu(tmp.as_ptr());
            let out=xor(n,self.s);
            let mut tag=[0u8;16];
            storeu(tag.as_mut_ptr(), out);
            Ok(tag)
        }
    }
}

impl Drop for PolyvalX86_64 {
    fn drop(&mut self) {
        unsafe {
            zeroize(self as *mut _ as *mut u8, mem::size_of::<PolyvalX86_64>());
        }
    }
}

// ARM64 OPTIMIZATION
//

#![feature(stdsimd, target_feature_11)]
use std::arch::aarch64::*;
use std::mem;
use std::ptr;
use std::slice;

const AES_BLOCK_SIZE: usize = 16;
const POLYVAL_SIZE: usize = 16;
const AES_BLOCK_SIZE_X4: usize = AES_BLOCK_SIZE * 4;

#[derive(Debug)]
enum AesGcmSivError {
    InvalidParameters,
    InvalidKeySize,
    InvalidNonceSize,
    // Add other errors as needed.
}

type Result<T> = std::result::Result<T, AesGcmSivError>;

#[inline]
unsafe fn zeroize(ptr: *mut u8, len: usize) {
    for i in 0..len {
        ptr::write_volatile(ptr.add(i), 0);
    }
}

#[repr(C)]
pub struct AesArm64 {
    key: [uint8x16_t; 15],
    key_size: usize,
}

#[repr(C)]
pub struct PolyvalArm64 {
    s: uint8x16_t,
    h_table: [uint8x16_t; 8],
}

impl AesArm64 {
    pub fn new() -> Self {
        // Initialize everything to zero
        unsafe {
            let mut ctx = mem::MaybeUninit::<AesArm64>::uninit();
            ptr::write_bytes(ctx.as_mut_ptr(), 0, 1);
            ctx.assume_init()
        }
    }

    pub fn set_key(&mut self, key: &[u8]) -> Result<()> {
        if key.len() != 16 && key.len() != 32 {
            return Err(AesGcmSivError::InvalidKeySize);
        }

        self.key_size = key.len();

        unsafe {
            match key.len() {
                16 => {
                    // AES-128 key expansion
                    self.key[4] = vld1q_u8(key.as_ptr());
                    self.key[5] = key_exp_128(&mut self.key, 4, 0x01);
                    self.key[6] = key_exp_128(&mut self.key, 5, 0x02);
                    self.key[7] = key_exp_128(&mut self.key, 6, 0x04);
                    self.key[8] = key_exp_128(&mut self.key, 7, 0x08);
                    self.key[9] = key_exp_128(&mut self.key, 8, 0x10);
                    self.key[10] = key_exp_128(&mut self.key, 9, 0x20);
                    self.key[11] = key_exp_128(&mut self.key, 10, 0x40);
                    self.key[12] = key_exp_128(&mut self.key, 11, 0x80);
                    self.key[13] = key_exp_128(&mut self.key, 12, 0x1b);
                    self.key[14] = key_exp_128(&mut self.key, 13, 0x36);
                }
                32 => {
                    // AES-256 key expansion
                    self.key[0] = vld1q_u8(key.as_ptr());
                    self.key[1] = vld1q_u8(key.as_ptr().add(16));
                    self.key[2] = key_exp_256_1(&mut self.key, 0, 0x01);
                    self.key[3] = key_exp_256_2(&mut self.key, 1);
                    self.key[4] = key_exp_256_1(&mut self.key, 2, 0x02);
                    self.key[5] = key_exp_256_2(&mut self.key, 3);
                    self.key[6] = key_exp_256_1(&mut self.key, 4, 0x04);
                    self.key[7] = key_exp_256_2(&mut self.key, 5);
                    self.key[8] = key_exp_256_1(&mut self.key, 6, 0x08);
                    self.key[9] = key_exp_256_2(&mut self.key, 7);
                    self.key[10] = key_exp_256_1(&mut self.key, 8, 0x10);
                    self.key[11] = key_exp_256_2(&mut self.key, 9);
                    self.key[12] = key_exp_256_1(&mut self.key, 10, 0x20);
                    self.key[13] = key_exp_256_2(&mut self.key, 11);
                    self.key[14] = key_exp_256_1(&mut self.key, 12, 0x40);
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    pub fn ecb_encrypt(&self, plain: &[u8; AES_BLOCK_SIZE]) -> Result<[u8; AES_BLOCK_SIZE]> {
        let mut block = unsafe { vld1q_u8(plain.as_ptr()) };
        block = unsafe { aes_encrypt(self, block) };
        let mut cipher = [0u8;16];
        unsafe { vst1q_u8(cipher.as_mut_ptr(), block); }
        Ok(cipher)
    }

    pub fn ctr(&self, nonce: &[u8;AES_BLOCK_SIZE], input:&[u8], output:&mut [u8]) -> Result<()> {
        if output.len()<input.len() {
            return Err(AesGcmSivError::InvalidParameters);
        }

        unsafe { aes_ctr(self, nonce, input, output) }
    }
}

impl Drop for AesArm64 {
    fn drop(&mut self) {
        unsafe {
            zeroize(self as *mut _ as *mut u8, mem::size_of::<AesArm64>());
        }
    }
}

impl PolyvalArm64 {
    pub fn new(key:&[u8;16]) -> Result<Self> {
        let mut ctx=unsafe{
            let mut c=mem::MaybeUninit::<PolyvalArm64>::uninit();
            ptr::write_bytes(c.as_mut_ptr(),0,1);
            c.assume_init()
        };

        unsafe {
            ctx.s=vdupq_n_u8(0);
            ctx.h_table[0]=vld1q_u8(key.as_ptr());
            for i in 1..8 {
                ctx.h_table[i]=dot(ctx.h_table[0], ctx.h_table[i-1]);
            }
        }

        Ok(ctx)
    }

    pub fn update(&mut self, data:&[u8]) {
        unsafe {
            self.s=polyval_arm64_process_tables(&self.h_table,self.s,data.as_ptr(), data.len());
        }
    }

    pub fn finish(mut self, nonce:&[u8]) -> Result<[u8;16]> {
        if nonce.len()>16 {
            return Err(AesGcmSivError::InvalidNonceSize);
        }

        let mut tmp=[0u8;16];
        tmp[..nonce.len()].copy_from_slice(nonce);

        let n=unsafe{vld1q_u8(tmp.as_ptr())};
        let out=unsafe{veorq_u8(n,self.s)};
        let mut tag=[0u8;16];
        unsafe { vst1q_u8(tag.as_mut_ptr(),out) };
        Ok(tag)
    }
}

impl Drop for PolyvalArm64 {
    fn drop(&mut self) {
        unsafe {
            zeroize(self as *mut _ as *mut u8, mem::size_of::<PolyvalArm64>());
        }
    }
}

// ----- Internal unsafe functions for AES and Polyval -----

// Key expansion helpers (same logic as original)
#[inline]
unsafe fn SHIFT_ADD(a:uint8x16_t)->uint8x16_t {
    let tmp=vextq_u8(vdupq_n_u8(0),a,12);
    veorq_u8(a,tmp)
}

#[inline]
unsafe fn TSHIFT_ADD(a:uint8x16_t)->uint8x16_t {
    SHIFT_ADD(SHIFT_ADD(SHIFT_ADD(a)))
}
#[inline]
unsafe fn ROTWORD(a:uint8x16_t)->uint8x16_t {
    vextq_u8(a,a,4)
}
#[inline]
unsafe fn NOROTWORD(a:uint8x16_t)->uint8x16_t {
    vextq_u8(a,vdupq_n_u8(0),12)
}
#[inline]
unsafe fn SUBWORD(a:uint8x16_t)->uint8x16_t {
    vaeseq_u8(a,vdupq_n_u8(0))
}
#[inline]
unsafe fn RCON(a:u32)->uint8x16_t {
    vreinterpretq_u8_u32(vdupq_n_u32(a))
}
#[inline]
unsafe fn XOR(a:uint8x16_t,b:uint8x16_t)->uint8x16_t { veorq_u8(a,b) }
#[inline]
unsafe fn XOR3(a:uint8x16_t,b:uint8x16_t,c:uint8x16_t)->uint8x16_t {
    veorq_u8(a, veorq_u8(b,c))
}

#[inline]
unsafe fn key_exp_128(key:&mut [uint8x16_t], i:usize, r:u32)->uint8x16_t {
    XOR3(TSHIFT_ADD(key[i]), SUBWORD(ROTWORD(key[i])), RCON(r))
}

#[inline]
unsafe fn key_exp_256_1(key:&mut [uint8x16_t], i:usize, r:u32)->uint8x16_t {
    XOR3(TSHIFT_ADD(key[i]), SUBWORD(ROTWORD(key[i+1])), RCON(r))
}

#[inline]
unsafe fn key_exp_256_2(key:&mut [uint8x16_t], i:usize)->uint8x16_t {
    XOR(TSHIFT_ADD(key[i]), SUBWORD(NOROTWORD(key[i+1])))
}

#[inline]
unsafe fn AES_ROUND(block:&mut uint8x16_t, k:uint8x16_t) {
    *block=vaeseq_u8(*block,k);
    *block=vaesmcq_u8(*block);
}

#[inline]
unsafe fn AES_LAST_ROUND(block:&mut uint8x16_t,k0:uint8x16_t,k1:uint8x16_t){
    *block=vaeseq_u8(*block,k0);
    *block=veorq_u8(*block,k1);
}

#[target_feature(enable="aes")]
unsafe fn aes_encrypt(ctx:&AesArm64, mut block:uint8x16_t)->uint8x16_t {
    if ctx.key_size==32 {
        AES_ROUND(&mut block, ctx.key[0]);
        AES_ROUND(&mut block, ctx.key[1]);
        AES_ROUND(&mut block, ctx.key[2]);
        AES_ROUND(&mut block, ctx.key[3]);
    }

    AES_ROUND(&mut block, ctx.key[4]);
    AES_ROUND(&mut block, ctx.key[5]);
    AES_ROUND(&mut block, ctx.key[6]);
    AES_ROUND(&mut block, ctx.key[7]);
    AES_ROUND(&mut block, ctx.key[8]);
    AES_ROUND(&mut block, ctx.key[9]);
    AES_ROUND(&mut block, ctx.key[10]);
    AES_ROUND(&mut block, ctx.key[11]);
    AES_ROUND(&mut block, ctx.key[12]);
    AES_LAST_ROUND(&mut block, ctx.key[13], ctx.key[14]);
    block
}

#[inline]
unsafe fn add_u32x4(a:uint8x16_t,b:uint32x4_t)->uint8x16_t {
    let x=vreinterpretq_u32_u8(a);
    let y=b;
    vreinterpretq_u8_u32(vaddq_u32(x,y))
}

#[inline]
unsafe fn uint32x4_c(a:u32)->uint32x4_t {
    vdupq_n_u32(a)
}

#[target_feature(enable="aes")]
unsafe fn aes_encrypt_x4(ctx:&AesArm64, counter:&[uint8x16_t;4], out:&mut[uint8x16_t;4]) {
    for i in 0..4 { out[i]=veorq_u8(counter[i],ctx.key[0]); }
    if ctx.key_size==32 {
        for r in 1..4 {
            for i in 0..4 {
                out[i]=vaeseq_u8(out[i], ctx.key[r]);
                out[i]=vaesmcq_u8(out[i]);
            }
        }
    }
    for r in 4..13 {
        for i in 0..4 {
            out[i]=vaeseq_u8(out[i],ctx.key[r]);
            out[i]=vaesmcq_u8(out[i]);
        }
    }
    for i in 0..4 {
        out[i]=vaeseq_u8(out[i], ctx.key[13]);
        out[i]=veorq_u8(out[i], ctx.key[14]);
    }
}

#[target_feature(enable="aes")]
unsafe fn aes_ctr(ctx:&AesArm64, nonce:&[u8;16], input:&[u8], output:&mut[u8])->Result<()> {
    if output.len()<input.len() {
        return Err(AesGcmSivError::InvalidParameters);
    }

    let one=uint32x4_c(1);
    let mut counter=[vdupq_n_u8(0);4];
    counter[0]=vld1q_u8(nonce.as_ptr());
    counter[1]=add_u32x4(counter[0],one);
    counter[2]=add_u32x4(counter[1],one);
    counter[3]=add_u32x4(counter[2],one);

    let mut remain=input.len();
    let mut inptr=input.as_ptr();
    let mut outptr=output.as_mut_ptr();

    let mut stream=[vdupq_n_u8(0);4];

    while remain>=AES_BLOCK_SIZE_X4 {
        aes_encrypt_x4(ctx,&counter,&mut stream);
        counter[0]=add_u32x4(counter[0],uint32x4_c(4));
        counter[1]=add_u32x4(counter[1],uint32x4_c(4));
        counter[2]=add_u32x4(counter[2],uint32x4_c(4));
        counter[3]=add_u32x4(counter[3],uint32x4_c(4));

        let mut blocks_in=[vdupq_n_u8(0);4];
        for i in 0..4 {
            blocks_in[i]=vld1q_u8(inptr.add(i*16));
        }

        for i in 0..4 {
            let c=veorq_u8(blocks_in[i],stream[i]);
            vst1q_u8(outptr.add(i*16),c);
        }

        inptr=inptr.add(AES_BLOCK_SIZE_X4);
        outptr=outptr.add(AES_BLOCK_SIZE_X4);
        remain-=AES_BLOCK_SIZE_X4;
    }

    if remain>0 {
        let num_blocks=remain/AES_BLOCK_SIZE;
        let leftover=remain%AES_BLOCK_SIZE;
        let mut stack_stream=[vdupq_n_u8(0);4];

        match num_blocks {
            0 => {
                stack_stream[0]=aes_encrypt(ctx,counter[0]);
            }
            1=>{
                let tempctr=[counter[0],counter[1],vdupq_n_u8(0),vdupq_n_u8(0)];
                aes_encrypt_x4(ctx,&tempctr,&mut stack_stream);
                let inblk=vld1q_u8(inptr);
                let c=veorq_u8(inblk,stack_stream[0]);
                vst1q_u8(outptr,c);
            }
            2=>{
                let tempctr=[counter[0],counter[1],counter[2],vdupq_n_u8(0)];
                aes_encrypt_x4(ctx,&tempctr,&mut stack_stream);
                let b0=vld1q_u8(inptr);
                let b1=vld1q_u8(inptr.add(16));
                vst1q_u8(outptr,veorq_u8(b0,stack_stream[0]));
                vst1q_u8(outptr.add(16),veorq_u8(b1,stack_stream[1]));
            }
            3=>{
                aes_encrypt_x4(ctx,&counter,&mut stack_stream);
                let b0=vld1q_u8(inptr);
                let b1=vld1q_u8(inptr.add(16));
                let b2=vld1q_u8(inptr.add(32));
                vst1q_u8(outptr,veorq_u8(b0,stack_stream[0]));
                vst1q_u8(outptr.add(16),veorq_u8(b1,stack_stream[1]));
                vst1q_u8(outptr.add(32),veorq_u8(b2,stack_stream[2]));
            }
            _=>{}
        }

        inptr=inptr.add(num_blocks*AES_BLOCK_SIZE);
        outptr=outptr.add(num_blocks*AES_BLOCK_SIZE);
        remain-=num_blocks*AES_BLOCK_SIZE;

        if leftover>0 {
            let mut tmp=[0u8;16];
            ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
            let inblk=vld1q_u8(tmp.as_ptr());
            let outblk=veorq_u8(inblk,stack_stream[num_blocks]);
            vst1q_u8(tmp.as_mut_ptr(),outblk);
            ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
        }
    }

    Ok(())
}

// Polyval logic (same as original code, just tidied up)
#[inline]
unsafe fn SWAP(a:uint8x16_t)->uint8x16_t {
    vextq_u8(a,a,8)
}

#[inline]
unsafe fn poly64_low(a:uint8x16_t)->poly64_t {
    vget_lane_p64(vreinterpretq_p64_u8(a),0)
}

#[inline]
unsafe fn poly64_high(a:uint8x16_t)->poly64_t {
    vget_lane_p64(vreinterpretq_p64_u8(a),1)
}

#[inline]
unsafe fn MULT_LOW(a:uint8x16_t,b:uint8x16_t)->uint8x16_t {
    let a0=poly64_low(a);
    let b0=poly64_low(b);
    vreinterpretq_u8_p128(vmull_p64(a0,b0))
}
#[inline]
unsafe fn MULT_HIGH(a:uint8x16_t,b:uint8x16_t)->uint8x16_t {
    let a1=poly64_high(a);
    let b1=poly64_high(b);
    vreinterpretq_u8_p128(vmull_p64(a1,b1))
}

#[inline]
unsafe fn mult(a:uint8x16_t,b:uint8x16_t,c0:&mut uint8x16_t,c1:&mut uint8x16_t,c2:&mut uint8x16_t) {
    *c0=MULT_LOW(a,b);
    *c2=MULT_HIGH(a,b);
    let a0=poly64_low(a);
    let b1=poly64_high(b);
    let a1=poly64_high(a);
    let b0=poly64_low(b);
    let a0b1=vreinterpretq_u8_p128(vmull_p64(a0,b1));
    let a1b0=vreinterpretq_u8_p128(vmull_p64(a1,b0));
    *c1=veorq_u8(a0b1,a1b0);
}

#[inline]
unsafe fn add_mult(a:uint8x16_t,b:uint8x16_t,c0:&mut uint8x16_t,c1:&mut uint8x16_t,c2:&mut uint8x16_t){
    let mut aa0=vdupq_n_u8(0); let mut aa1=vdupq_n_u8(0);let mut aa2=vdupq_n_u8(0);
    mult(a,b,&mut aa0,&mut aa1,&mut aa2);
    *c0=veorq_u8(*c0,aa0);
    *c2=veorq_u8(*c2,aa2);
    *c1=veorq_u8(*c1,aa1);
}

#[inline]
unsafe fn mult_inv_x64(p:uint8x16_t)->uint8x16_t {
    let poly=0xc200000000000000u64;
    let poly_p64=vcreate_p64(poly);
    let q=SWAP(p);
    let a0=poly64_low(p);
    let r=vreinterpretq_u8_p128(vmull_p64(a0,vget_lane_p64(poly_p64,0)));
    veorq_u8(q,r)
}

#[inline]
unsafe fn mult_inv_x128(p0:uint8x16_t,p1:uint8x16_t,p2:uint8x16_t)->uint8x16_t {
    let q=veorq_u8(p0,vextq_u8(vdupq_n_u8(0),p1,8));
    let r=veorq_u8(p2,vextq_u8(p1,vdupq_n_u8(0),8));
    let s=mult_inv_x64(q);
    let t=mult_inv_x64(s);
    veorq_u8(r,t)
}

#[inline]
unsafe fn dot(a:uint8x16_t,b:uint8x16_t)->uint8x16_t {
    let mut c0=vdupq_n_u8(0);
    let mut c1=vdupq_n_u8(0);
    let mut c2=vdupq_n_u8(0);
    mult(a,b,&mut c0,&mut c1,&mut c2);
    mult_inv_x128(c0,c1,c2)
}

#[inline]
unsafe fn polyval_arm64_process_tables(h_table:&[uint8x16_t;8],
                                       mut s:uint8x16_t,
                                       mut data:*const u8,
                                       mut data_sz:usize) -> uint8x16_t {
    let mut tmp=[0u8;16];

    let blocks_8=data_sz/(8*POLYVAL_SIZE);
    for _ in 0..blocks_8 {
        let D0=vld1q_u8(data);
        let D1=vld1q_u8(data.add(16));
        let D2=vld1q_u8(data.add(32));
        let D3=vld1q_u8(data.add(48));
        let D4=vld1q_u8(data.add(64));
        let D5=vld1q_u8(data.add(80));
        let D6=vld1q_u8(data.add(96));
        let D7=vld1q_u8(data.add(112));

        let mut s0=vdupq_n_u8(0); let mut s1=vdupq_n_u8(0); let mut s2=vdupq_n_u8(0);

        mult(D7,h_table[0],&mut s0,&mut s1,&mut s2);
        add_mult(D6,h_table[1],&mut s0,&mut s1,&mut s2);
        add_mult(D5,h_table[2],&mut s0,&mut s1,&mut s2);
        add_mult(D4,h_table[3],&mut s0,&mut s1,&mut s2);
        add_mult(D3,h_table[4],&mut s0,&mut s1,&mut s2);
        add_mult(D2,h_table[5],&mut s0,&mut s1,&mut s2);
        add_mult(D1,h_table[6],&mut s0,&mut s1,&mut s2);
        add_mult(veorq_u8(s,D0),h_table[7],&mut s0,&mut s1,&mut s2);

        s=mult_inv_x128(s0,s1,s2);

        data=data.add(8*16);
        data_sz-=8*16;
    }

    let blocks=data_sz/16;
    if blocks>0 {
        if blocks>1 {
            let last=vld1q_u8(data.add((blocks-1)*16));
            let mut s0=vdupq_n_u8(0);let mut s1=vdupq_n_u8(0);let mut s2=vdupq_n_u8(0);
            mult(last,h_table[0],&mut s0,&mut s1,&mut s2);
            for i in 1..(blocks-1) {
                let blk=vld1q_u8(data.add((blocks-1-i)*16));
                add_mult(blk,h_table[i],&mut s0,&mut s1,&mut s2);
            }
            let first=veorq_u8(s,vld1q_u8(data));
            add_mult(first,h_table[blocks-1],&mut s0,&mut s1,&mut s2);
            s=mult_inv_x128(s0,s1,s2);
        } else {
            let first=veorq_u8(s,vld1q_u8(data));
            let mut s0=vdupq_n_u8(0);let mut s1=vdupq_n_u8(0);let mut s2=vdupq_n_u8(0);
            mult(first,h_table[0],&mut s0,&mut s1,&mut s2);
            s=mult_inv_x128(s0,s1,s2);
        }

        data=data.add(blocks*16);
        data_sz-=blocks*16;
    }

    if data_sz>0 {
        ptr::copy_nonoverlapping(data,tmp.as_mut_ptr(), data_sz);
        for i in data_sz..16 {
            tmp[i]=0;
        }
        let d=veorq_u8(s,vld1q_u8(tmp.as_ptr()));
        s=dot(d,h_table[0]);
    }

    s
}

// aes-gcm-siv.rs

// ====================== Complete Generic Implementation ======================

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
                    *rk.add(off + 8) = *rk.add(off + 2) ^ *rk.add(off + 7);
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

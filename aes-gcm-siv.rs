#![allow(non_upper_case_globals)]

// AES-GCM-SIV - December 16, 2024
// RFC: https://www.rfc-editor.org/rfc/rfc8452
//
// This Rust code provides a robust and optimized implementation of AES-GCM-SIV,
// incorporating hardware acceleration where available and adhering to constant-time principles.
//

use std::mem;
use std::ptr;

// ====================== Constants and Limits ======================

const AES_GCMSIV_TAG_SIZE: usize = 16;
const AES_GCMSIV_NONCE_SIZE: usize = 12;
const POLYVAL_SIZE: usize = 16;
const AES_BLOCK_SIZE: usize = 16;
const AES_GCMSIV_MAX_PLAINTEXT_SIZE: usize = (1 << 36) - 1;
const AES_GCMSIV_MAX_AAD_SIZE: usize = (1 << 36) - 1;
const KEY_AUTH_SIZE: usize = 16;
const KEY_ENC_MAX_SIZE: usize = 32;

// ====================== Status and Error Handling ======================

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
enum AesGcmSivStatus {
    Success = 0,
    Failure = -1,
    OutOfMemory = -2,
    UpdateOutputSize = -3,
    InvalidParameters = -4,
    InvalidKeySize = -5,
    InvalidNonceSize = -6,
    InvalidPlaintextSize = -7,
    InvalidAadSize = -8,
    InvalidCiphertextSize = -9,
    InvalidTag = -10,
}

// ====================== Key Context Structure ======================

#[derive(Copy, Clone, Default)]
struct KeyContext {
    auth: [u8; KEY_AUTH_SIZE],
    auth_sz: usize,
    enc: [u8; KEY_ENC_MAX_SIZE],
    enc_sz: usize,
}

// ====================== AES-GCM-SIV Context Structure ======================

#[repr(C)]
struct AesGcmSivCtx {
    key_gen_ctx: *mut Aes,
    key_sz: usize,
}

// ====================== Utility Functions ======================

#[inline]
fn put_u32_le(val: u32, dst: &mut [u8], offset: usize) {
    dst[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn get_u32_le(src: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(src[offset..offset + 4].try_into().unwrap())
}

#[inline]
fn put_u64_le(val: u64, dst: &mut [u8], offset: usize) {
    dst[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn secure_zeroize(buf: &mut [u8]) {
    for b in buf {
        unsafe { ptr::write_volatile(b, 0) };
    }
}

#[inline]
fn secure_zeroize_struct<T>(val: &mut T) {
    let ptr = val as *mut T as *mut u8;
    let len = mem::size_of::<T>();
    unsafe { ptr::write_bytes(ptr, 0, len) };
}
// ====================== Feature Detection ======================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_x86_features() -> (bool, bool) {
    let aes = is_x86_feature_detected!("aes");
    let pclmul = is_x86_feature_detected!("pclmulqdq");
    (aes, pclmul)
}

#[cfg(target_arch = "aarch64")]
fn detect_arm64_features() -> (bool, bool) {
    let aes = is_aarch64_feature_detected!("aes");
    let pmull = is_aarch64_feature_detected!("pmull");
    (aes, pmull)
}

// ====================== AES Implementation ======================

trait Aes {
    fn new() -> Self;
    fn set_key(&mut self, key: &[u8]) -> AesGcmSivStatus;
    fn ecb_encrypt(&self, block: &[u8; AES_BLOCK_SIZE]) -> [u8; AES_BLOCK_SIZE];
    fn ctr(&self, nonce: &[u8; AES_BLOCK_SIZE], input: &[u8], output: &mut [u8]) -> AesGcmSivStatus;
}

// Generic AES Implementation
#[derive(Default)]
struct AesGeneric {
    nr: i32,
    rk: [u32; 68],
}

const RCON: [u32; 10] = [
    0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x0000001b, 0x00000036,
];

const FSb: [u8; 256] = [
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

#[inline]
fn rotr8(x: u32) -> u32 {
    (x >> 8) | (x << 24)
}

#[inline]
fn sub_word(x: u32) -> u32 {
    let b0 = FSb[(x >> 24) as u8] as u32;
    let b1 = FSb[((x >> 16) & 0xff) as u8] as u32;
    let b2 = FSb[((x >> 8) & 0xff) as u8] as u32;
    let b3 = FSb[(x & 0xff) as u8] as u32;
    b0 << 24 | b1 << 16 | b2 << 8 | b3
}

impl Aes for AesGeneric {
    fn new() -> Self {
        Self::default()
    }

    fn set_key(&mut self, key: &[u8]) -> AesGcmSivStatus {
        let key_sz = key.len();
        let nr = match key_sz {
            16 => 10,
            24 => 12,
            32 => 14,
            _ => return AesGcmSivStatus::InvalidKeySize,
        };
        self.nr = nr;

        for i in 0..key_sz / 4 {
            self.rk[i] = get_u32_le(key, i * 4);
        }

        for i in key_sz / 4..self.rk.len() {
            let mut temp = self.rk[i - 1];
            if i % (key_sz / 4) == 0 {
                temp = sub_word(rotr8(temp)) ^ RCON[i / (key_sz / 4) - 1];
            } else if key_sz == 32 && i % (key_sz / 4) == 4 {
                temp = sub_word(temp);
            }
            self.rk[i] = self.rk[i - key_sz / 4] ^ temp;
        }

        AesGcmSivStatus::Success
    }

    fn ecb_encrypt(&self, block: &[u8; AES_BLOCK_SIZE]) -> [u8; AES_BLOCK_SIZE] {
        let mut state = [0u32; 4];
        for i in 0..4 {
            state[i] = get_u32_le(block, i * 4);
        }

        self.add_round_key(&mut state, 0);

        for round in 1..self.nr {
            self.sub_bytes(&mut state);
            self.shift_rows(&mut state);
            self.mix_columns(&mut state);
            self.add_round_key(&mut state, round as usize * 4);
        }

        self.sub_bytes(&mut state);
        self.shift_rows(&mut state);
        self.add_round_key(&mut state, self.nr as usize * 4);

        let mut result = [0u8; AES_BLOCK_SIZE];
        for i in 0..4 {
            put_u32_le(state[i], &mut result, i * 4);
        }
        result
    }

    fn ctr(&self, nonce: &[u8; AES_BLOCK_SIZE], input: &[u8], output: &mut [u8]) -> AesGcmSivStatus {
        if output.len() < input.len() {
            return AesGcmSivStatus::InvalidParameters;
        }

        let mut counter_block = *nonce;
        let mut counter = get_u32_le(&counter_block, 0);
        let mut key_stream = [0u8; AES_BLOCK_SIZE];
        let mut processed = 0;

        while processed + AES_BLOCK_SIZE <= input.len() {
            key_stream = self.ecb_encrypt(&counter_block);
            counter = counter.wrapping_add(1);
            put_u32_le(counter, &mut counter_block, 0);
            for i in 0..AES_BLOCK_SIZE {
                output[processed + i] = input[processed + i] ^ key_stream[i];
            }
            processed += AES_BLOCK_SIZE;
        }

        if processed < input.len() {
            key_stream = self.ecb_encrypt(&counter_block);
            for i in 0..(input.len() - processed) {
                output[processed + i] = input[processed + i] ^ key_stream[i];
            }
        }

        AesGcmSivStatus::Success
    }
}

impl AesGeneric {
    #[inline]
    fn sub_bytes(&self, state: &mut [u32; 4]) {
        for i in 0..4 {
            let b0 = FSb[(state[i] >> 24) as u8] as u32;
            let b1 = FSb[((state[i] >> 16) & 0xff) as u8] as u32;
            let b2 = FSb[((state[i] >> 8) & 0xff) as u8] as u32;
            let b3 = FSb[(state[i] & 0xff) as u8] as u32;
            state[i] = b0 << 24 | b1 << 16 | b2 << 8 | b3;
        }
    }

    #[inline]
    fn shift_rows(&self, state: &mut [u32; 4]) {
        state[1] = rotr8(state[1]);
        state[2] = rotr8(rotr8(state[2]));
        state[3] = rotr8(rotr8(rotr8(state[3])));
    }

    #[inline]
    fn mix_columns(&self, state: &mut [u32; 4]) {
        for i in 0..4 {
            let s0 = state[i];
            let s1 = rotr8(s0);
            let s2 = rotr8(s1);
            let s3 = rotr8(s2);
            state[i] = (g2(s0) ^ g3(s1) ^ s2 ^ s3)
                ^ (s0 ^ g2(s1) ^ g3(s2) ^ s3)
                ^ (s0 ^ s1 ^ g2(s2) ^ g3(s3))
                ^ (g3(s0) ^ s1 ^ s2 ^ g2(s3));
        }
    }

    #[inline]
    fn add_round_key(&self, state: &mut [u32; 4], offset: usize) {
        for i in 0..4 {
            state[i] ^= self.rk[i + offset];
        }
    }
}

#[inline]
fn g2(x: u32) -> u32 {
    let b0 = XTIME((x >> 24) as u8) as u32;
    let b1 = XTIME(((x >> 16) & 0xff) as u8) as u32;
    let b2 = XTIME(((x >> 8) & 0xff) as u8) as u32;
    let b3 = XTIME((x & 0xff) as u8) as u32;
    b0 << 24 | b1 << 16 | b2 << 8 | b3
}

#[inline]
fn g3(x: u32) -> u32 {
    g2(x) ^ x
}

#[inline]
fn XTIME(x: u8) -> u8 {
    let mut val = x << 1;
    if x & 0x80 != 0 {
        val ^= 0x1B;
    }
    (val & 0xFF)
}

// x86-64 Optimized AES Implementation (using AES-NI)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Default)]
struct AesX86_64 {
    key: [__m128i; 15],
    key_sz: usize,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Aes for AesX86_64 {
    fn new() -> Self {
        Self::default()
    }

    fn set_key(&mut self, key: &[u8]) -> AesGcmSivStatus {
        if key.len() != 16 && key.len() != 32 {
            return AesGcmSivStatus::InvalidKeySize;
        }
        self.key_sz = key.len();
        unsafe {
            self.key[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
            if key.len() == 16 {
                // AES-128 key expansion
                self.key[1] = _mm_aeskeygenassist_si128(self.key[0], 0x01);
                self.key[2] = _mm_aeskeygenassist_si128(self.key[1], 0x02);
                self.key[3] = _mm_aeskeygenassist_si128(self.key[2], 0x04);
                self.key[4] = _mm_aeskeygenassist_si128(self.key[3], 0x08);
                self.key[5] = _mm_aeskeygenassist_si128(self.key[4], 0x10);
                self.key[6] = _mm_aeskeygenassist_si128(self.key[5], 0x20);
                self.key[7] = _mm_aeskeygenassist_si128(self.key[6], 0x40);
                self.key[8] = _mm_aeskeygenassist_si128(self.key[7], 0x80);
                self.key[9] = _mm_aeskeygenassist_si128(self.key[8], 0x1b);
                self.key[10] = _mm_aeskeygenassist_si128(self.key[9], 0x36);
                for i in 0..10 {
                    let t = _mm_shuffle_epi32(self.key[i + 1], 0xff);
                    self.key[i + 1] = _mm_xor_si128(
                        _mm_slli_si128(
                            _mm_xor_si128(
                                _mm_xor_si128(
                                    _mm_slli_si128(self.key[i], 0x4),
                                    _mm_slli_si128(self.key[i], 0x8),
                                ),
                                _mm_slli_si128(self.key[i], 0xc),
                            ),
                            0x4,
                        ),
                        t,
                    );
                    self.key[i + 1] = _mm_xor_si128(self.key[i + 1], self.key[i]);
                }
            } else {
                // AES-256
                self.key[1] = _mm_loadu_si128(key.as_ptr().add(16) as *const __m128i);
                self.key[2] = _mm_aeskeygenassist_si128(self.key[1], 0x01);
                self.key[3] = _mm_aeskeygenassist_si128(self.key[2], 0x02);
                self.key[4] = _mm_aeskeygenassist_si128(self.key[3], 0x04);
                self.key[5] = _mm_aeskeygenassist_si128(self.key[4], 0x08);
                self.key[6] = _mm_aeskeygenassist_si128(self.key[5], 0x10);
                self.key[7] = _mm_aeskeygenassist_si128(self.key[6], 0x20);
                self.key[8] = _mm_aeskeygenassist_si128(self.key[7], 0x40);
                self.key[9] = _mm_aeskeygenassist_si128(self.key[8], 0x80);
                self.key[10] = _mm_aeskeygenassist_si128(self.key[9], 0x1b);
                self.key[11] = _mm_aeskeygenassist_si128(self.key[10], 0x36);
                for i in (0..10).step_by(2) {
                    let t1 = _mm_shuffle_epi32(self.key[i + 2], 0xff);
                    self.key[i + 2] = _mm_xor_si128(
                        _mm_slli_si128(
                            _mm_xor_si128(
                                _mm_xor_si128(
                                    _mm_slli_si128(self.key[i], 0x4),
                                    _mm_slli_si128(self.key[i], 0x8),
                                ),
                                _mm_slli_si128(self.key[i], 0xc),
                            ),
                            0x4,
                        ),
                        t1,
                    );
                    self.key[i + 2] = _mm_xor_si128(self.key[i + 2], self.key[i]);

                    let t2 = _mm_shuffle_epi32(self.key[i + 3], 0xaa);
                    self.key[i + 3] = _mm_xor_si128(
                        _mm_slli_si128(
                            _mm_xor_si128(
                                _mm_xor_si128(
                                    _mm_slli_si128(self.key[i + 1], 0x4),
                                    _mm_slli_si128(self.key[i + 1], 0x8),
                                ),
                                _mm_slli_si128(self.key[i + 1], 0xc),
                            ),
                            0x4,
                        ),
                        t2,
                    );
                    self.key[i + 3] = _mm_xor_si128(self.key[i + 3], self.key[i + 1]);
                }
            }
        }
        AesGcmSivStatus::Success
    }

    fn ecb_encrypt(&self, block: &[u8; AES_BLOCK_SIZE]) -> [u8; AES_BLOCK_SIZE] {
        unsafe {
            let mut block = _mm_loadu_si128(block.as_ptr() as *const __m128i);
            block = _mm_xor_si128(block, self.key[0]);
            match self.key_sz {
                16 => {
                    // AES-128
                    block = _mm_aesenc_si128(block, self.key[1]);
                    block = _mm_aesenc_si128(block, self.key[2]);
                    block = _mm_aesenc_si128(block, self.key[3]);
                    block = _mm_aesenc_si128(block, self.key[4]);
                    block = _mm_aesenc_si128(block, self.key[5]);
                    block = _mm_aesenc_si128(block, self.key[6]);
                    block = _mm_aesenc_si128(block, self.key[7]);
                    block = _mm_aesenc_si128(block, self.key[8]);
                    block = _mm_aesenc_si128(block, self.key[9]);
                    block = _mm_aesenclast_si128(block, self.key[10]);
                }
                32 => {
                    // AES-256
                    block = _mm_aesenc_si128(block, self.key[1]);
                    block = _mm_aesenc_si128(block, self.key[2]);
                    block = _mm_aesenc_si128(block, self.key[3]);
                    block = _mm_aesenc_si128(block, self.key[4]);
                    block = _mm_aesenc_si128(block, self.key[5]);
                    block = _mm_aesenc_si128(block, self.key[6]);
                    block = _mm_aesenc_si128(block, self.key[7]);
                    block = _mm_aesenc_si128(block, self.key[8]);
                    block = _mm_aesenc_si128(block, self.key[9]);
                    block = _mm_aesenc_si128(block, self.key[10]);
                    block = _mm_aesenc_si128(block, self.key[11]);
                    block = _mm_aesenc_si128(block, self.key[12]);
                    block = _mm_aesenc_si128(block, self.key[13]);
                    block = _mm_aesenclast_si128(block, self.key[14]);
                }
                _ => {}
            }
            let mut result = [0u8; AES_BLOCK_SIZE];
            _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, block);
            result
        }
    }

    fn ctr(&self, nonce: &[u8; AES_BLOCK_SIZE], input: &[u8], output: &mut [u8]) -> AesGcmSivStatus {
        if output.len() < input.len() {
            return AesGcmSivStatus::InvalidParameters;
        }
        unsafe { aes_ctr_x86_64(self, nonce, input, output) }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn aes_ctr_x86_64(
    ctx: &AesX86_64,
    nonce: &[u8; AES_BLOCK_SIZE],
    input: &[u8],
    output: &mut [u8],
) -> AesGcmSivStatus {
    let one = _mm_set_epi32(0, 0, 0, 1);
    let mut counter = [_mm_setzero_si128(); 4];
    counter[0] = _mm_loadu_si128(nonce.as_ptr() as *const __m128i);
    counter[1] = _mm_add_epi32(counter[0], one);
    counter[2] = _mm_add_epi32(counter[1], one);
    counter[3] = _mm_add_epi32(counter[2], one);

    let mut remain = input.len();
    let mut inptr = input.as_ptr();
    let mut outptr = output.as_mut_ptr();

    while remain >= 4 * AES_BLOCK_SIZE {
        let mut stream = [_mm_setzero_si128(); 4];
        aes_encrypt_x4_x86_64(ctx, &counter, &mut stream);

        counter[0] = _mm_add_epi32(counter[3], one);
        counter[1] = _mm_add_epi32(counter[0], one);
        counter[2] = _mm_add_epi32(counter[1], one);
        counter[3] = _mm_add_epi32(counter[2], one);

        for i in 0..4 {
            let inblk = _mm_loadu_si128(inptr.add(i * AES_BLOCK_SIZE) as *const __m128i);
            let outblk = _mm_xor_si128(inblk, stream[i]);
            _mm_storeu_si128(outptr.add(i * AES_BLOCK_SIZE) as *mut __m128i, outblk);
        }

        inptr = inptr.add(4 * AES_BLOCK_SIZE);
        outptr = outptr.add(4 * AES_BLOCK_SIZE);
        remain -= 4 * AES_BLOCK_SIZE;
    }

    if remain > 0 {
        let blocks = remain / AES_BLOCK_SIZE;
        let leftover = remain % AES_BLOCK_SIZE;
        let mut tmp = [0u8; AES_BLOCK_SIZE];

        match blocks {
            0 => {
                // single partial block
                let mut single = [_mm_xor_si128(counter[0], ctx.key[0])];
                if ctx.key_sz == 32 {
                    for r in 1..5 {
                        single[0] = _mm_aesenc_si128(single[0], ctx.key[r]);
                    }
                }
                for r in 5..14 {
                    single[0] = _mm_aesenc_si128(single[0], ctx.key[r]);
                }
                single[0] = _mm_aesenclast_si128(single[0], ctx.key[14]);

                ptr::copy_nonoverlapping(inptr, tmp.as_mut_ptr(), leftover);
                for i in leftover..AES_BLOCK_SIZE {
                    tmp[i] = 0;
                }
                let inblk = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                let outblk = _mm_xor_si128(inblk, single[0]);
                _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, outblk);
                ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
            }
            1 => {
                let mut stream = [_mm_setzero_si128(); 2];
                aes_encrypt_x2_x86_64(ctx, &counter, &mut stream);

                let inblk = _mm_loadu_si128(inptr as *const __m128i);
                let outblk = _mm_xor_si128(inblk, stream[0]);
                _mm_storeu_si128(outptr as *mut __m128i, outblk);
                inptr = inptr.add(AES_BLOCK_SIZE);
                outptr = outptr.add(AES_BLOCK_SIZE);

                if leftover > 0 {
                    ptr::copy_nonoverlapping(inptr, tmp.as_mut_ptr(), leftover);
                    for i in leftover..AES_BLOCK_SIZE {
                        tmp[i] = 0;
                    }
                    let inblk = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                    let outblk = _mm_xor_si128(inblk, stream[1]);
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            2 => {
                let mut stream = [_mm_setzero_si128(); 3];
                aes_encrypt_x3_x86_64(ctx, &counter, &mut stream);

                for i in 0..2 {
                    let inblk = _mm_loadu_si128(inptr.add(i * AES_BLOCK_SIZE) as *const __m128i);
                    let outblk = _mm_xor_si128(inblk, stream[i]);
                    _mm_storeu_si128(outptr.add(i * AES_BLOCK_SIZE) as *mut __m128i, outblk);
                }

                inptr = inptr.add(2 * AES_BLOCK_SIZE);
                outptr = outptr.add(2 * AES_BLOCK_SIZE);

                if leftover > 0 {
                    ptr::copy_nonoverlapping(inptr, tmp.as_mut_ptr(), leftover);
                    for i in leftover..AES_BLOCK_SIZE {
                        tmp[i] = 0;
                    }
                    let inblk = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                    let outblk = _mm_xor_si128(inblk, stream[2]);
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            3 => {
                let mut stream = [_mm_setzero_si128(); 4];
                aes_encrypt_x4_x86_64(ctx, &counter, &mut stream);

                for i in 0..3 {
                    let inblk = _mm_loadu_si128(inptr.add(i * AES_BLOCK_SIZE) as *const __m128i);
                    let outblk = _mm_xor_si128(inblk, stream[i]);
                    _mm_storeu_si128(outptr.add(i * AES_BLOCK_SIZE) as *mut __m128i, outblk);
                }

                inptr = inptr.add(3 * AES_BLOCK_SIZE);
                outptr = outptr.add(3 * AES_BLOCK_SIZE);

                if leftover > 0 {
                    ptr::copy_nonoverlapping(inptr, tmp.as_mut_ptr(), leftover);
                    for i in leftover..AES_BLOCK_SIZE {
                        tmp[i] = 0;
                    }
                    let inblk = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
                    let outblk = _mm_xor_si128(inblk, stream[3]);
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            _ => {}
        }
    }

    AesGcmSivStatus::Success
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn aes_encrypt_x4_x86_64(ctx: &AesX86_64, counter: &[__m128i; 4], stream: &mut [__m128i; 4]) {
    for i in 0..4 {
        stream[i] = _mm_xor_si128(counter[i], ctx.key[0]);
    }

    if ctx.key_sz == 32 {
        for r in 1..5 {
            for j in 0..4 {
                stream[j] = _mm_aesenc_si128(stream[j], ctx.key[r]);
            }
        }
    }
    for r in 5..14 {
        for j in 0..4 {
            stream[j] = _mm_aesenc_si128(stream[j], ctx.key[r]);
        }
    }
    for j in 0..4 {
        stream[j] = _mm_aesenclast_si128(stream[j], ctx.key[14]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn aes_encrypt_x2_x86_64(ctx: &AesX86_64, counter: &[__m128i; 2], stream: &mut [__m128i; 2]) {
    for i in 0..2 {
        stream[i] = _mm_xor_si128(counter[i], ctx.key[0]);
    }
    if ctx.key_sz == 32 {
        for r in 1..5 {
            for j in 0..2 {
                stream[j] = _mm_aesenc_si128(stream[j], ctx.key[r]);
            }
        }
    }
    for r in 5..14 {
        for j in 0..2 {
            stream[j] = _mm_aesenc_si128(stream[j], ctx.key[r]);
        }
    }
    for j in 0..2 {
        stream[j] = _mm_aesenclast_si128(stream[j], ctx.key[14]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn aes_encrypt_x3_x86_64(ctx: &AesX86_64, counter: &[__m128i; 3], stream: &mut [__m128i; 3]) {
    for i in 0..3 {
        stream[i] = _mm_xor_si128(counter[i], ctx.key[0]);
    }
    if ctx.key_sz == 32 {
        for r in 1..5 {
            for j in 0..3 {
                stream[j] = _mm_aesenc_si128(stream[j], ctx.key[r]);
            }
        }
    }
    for r in 5..14 {
        for j in 0..3 {
            stream[j] = _mm_aesenc_si128(stream[j], ctx.key[r]);
        }
    }
    for j in 0..3 {
        stream[j] = _mm_aesenclast_si128(stream[j], ctx.key[14]);
    }
}

// ARM64 Optimized AES Implementation
#[cfg(target_arch = "aarch64")]
#[derive(Default)]
struct AesArm64 {
    key: [uint8x16_t; 15],
    key_size: usize,
}

#[cfg(target_arch = "aarch64")]
impl Aes for AesArm64 {
    fn new() -> Self {
        Self::default()
    }

    fn set_key(&mut self, key: &[u8]) -> AesGcmSivStatus {
        if key.len() != 16 && key.len() != 32 {
            return AesGcmSivStatus::InvalidKeySize;
        }

        self.key_size = key.len();

        unsafe {
            match key.len() {
                16 => {
                    // AES-128 key expansion
                    self.key[4] = vld1q_u8(key.as_ptr());
                    self.key[5] = key_exp_128_arm64(&mut self.key, 4, 0x01);
                    self.key[6] = key_exp_128_arm64(&mut self.key, 5, 0x02);
                    self.key[7] = key_exp_128_arm64(&mut self.key, 6, 0x04);
                    self.key[8] = key_exp_128_arm64(&mut self.key, 7, 0x08);
                    self.key[9] = key_exp_128_arm64(&mut self.key, 8, 0x10);
                    self.key[10] = key_exp_128_arm64(&mut self.key, 9, 0x20);
                    self.key[11] = key_exp_128_arm64(&mut self.key, 10, 0x40);
                    self.key[12] = key_exp_128_arm64(&mut self.key, 11, 0x80);
                    self.key[13] = key_exp_128_arm64(&mut self.key, 12, 0x1b);
                    self.key[14] = key_exp_128_arm64(&mut self.key, 13, 0x36);
                }
                32 => {
                    // AES-256 key expansion
                    self.key[0] = vld1q_u8(key.as_ptr());
                    self.key[1] = vld1q_u8(key.as_ptr().add(16));
                    self.key[2] = key_exp_256_1_arm64(&mut self.key, 0, 0x01);
                    self.key[3] = key_exp_256_2_arm64(&mut self.key, 1);
                    self.key[4] = key_exp_256_1_arm64(&mut self.key, 2, 0x02);
                    self.key[5] = key_exp_256_2_arm64(&mut self.key, 3);
                    self.key[6] = key_exp_256_1_arm64(&mut self.key, 4, 0x04);
                    self.key[7] = key_exp_256_2_arm64(&mut self.key, 5);
                    self.key[8] = key_exp_256_1_arm64(&mut self.key, 6, 0x08);
                    self.key[9] = key_exp_256_2_arm64(&mut self.key, 7);
                    self.key[10] = key_exp_256_1_arm64(&mut self.key, 8, 0x10);
                    self.key[11] = key_exp_256_2_arm64(&mut self.key, 9);
                    self.key[12] = key_exp_256_1_arm64(&mut self.key, 10, 0x20);
                    self.key[13] = key_exp_256_2_arm64(&mut self.key, 11);
                    self.key[14] = key_exp_256_1_arm64(&mut self.key, 12, 0x40);
                }
                _ => unreachable!(),
            }
        }

        AesGcmSivStatus::Success
    }

    fn ecb_encrypt(&self, plain: &[u8; AES_BLOCK_SIZE]) -> [u8; AES_BLOCK_SIZE] {
        let mut block = unsafe { vld1q_u8(plain.as_ptr()) };
        block = unsafe { aes_encrypt_arm64(self, block) };
        let mut cipher = [0u8; 16];
        unsafe { vst1q_u8(cipher.as_mut_ptr(), block) };
        cipher
    }

    fn ctr(&self, nonce: &[u8; AES_BLOCK_SIZE], input: &[u8], output: &mut [u8]) -> AesGcmSivStatus {
        if output.len() < input.len() {
            return AesGcmSivStatus::InvalidParameters;
        }

        unsafe { aes_ctr_arm64(self, nonce, input, output) }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn aes_ctr_arm64(
    ctx: &AesArm64,
    nonce: &[u8; AES_BLOCK_SIZE],
    input: &[u8],
    output: &mut [u8],
) -> AesGcmSivStatus {
    let one = vdupq_n_u32(1);
    let mut counter = [vdupq_n_u8(0); 4];
    counter[0] = vld1q_u8(nonce.as_ptr());
    counter[1] = add_u32x4_arm64(counter[0], one);
    counter[2] = add_u32x4_arm64(counter[1], one);
    counter[3] = add_u32x4_arm64(counter[2], one);

    let mut remain = input.len();
    let mut inptr = input.as_ptr();
    let mut outptr = output.as_mut_ptr();

    const AES_BLOCK_SIZE_X4: usize = AES_BLOCK_SIZE * 4;
    let mut stream = [vdupq_n_u8(0); 4];

    while remain >= AES_BLOCK_SIZE_X4 {
        aes_encrypt_x4_arm64(ctx, &counter, &mut stream);
        counter[0] = add_u32x4_arm64(counter[0], vdupq_n_u32(4));
        counter[1] = add_u32x4_arm64(counter[1], vdupq_n_u32(4));
        counter[2] = add_u32x4_arm64(counter[2], vdupq_n_u32(4));
        counter[3] = add_u32x4_arm64(counter[3], vdupq_n_u32(4));

        for i in 0..4 {
            let inblk = vld1q_u8(inptr.add(i * AES_BLOCK_SIZE));
            let outblk = veorq_u8(inblk, stream[i]);
            vst1q_u8(outptr.add(i * AES_BLOCK_SIZE), outblk);
        }

        inptr = inptr.add(AES_BLOCK_SIZE_X4);
        outptr = outptr.add(AES_BLOCK_SIZE_X4);
        remain -= AES_BLOCK_SIZE_X4;
    }

    if remain > 0 {
        let num_blocks = remain / AES_BLOCK_SIZE;
        let leftover = remain % AES_BLOCK_SIZE;
        let mut stack_stream = [vdupq_n_u8(0); 4];

        match num_blocks {
            0 => {
                stack_stream[0] = aes_encrypt_arm64(ctx, counter[0]);
            }
            1 => {
                let tempctr = [counter[0], counter[1], vdupq_n_u8(0), vdupq_n_u8(0)];
                aes_encrypt_x4_arm64(ctx, &tempctr, &mut stack_stream);
                let inblk = vld1q_u8(inptr);
                let c = veorq_u8(inblk, stack_stream[0]);
                vst1q_u8(outptr, c);
            }
            2 => {
                let tempctr = [counter[0], counter[1], counter[2], vdupq_n_u8(0)];
                aes_encrypt_x4_arm64(ctx, &tempctr, &mut stack_stream);
                let b0 = vld1q_u8(inptr);
                let b1 = vld1q_u8(inptr.add(16));
                vst1q_u8(outptr, veorq_u8(b0, stack_stream[0]));
                vst1q_u8(outptr.add(16), veorq_u8(b1, stack_stream[1]));
            }
            3 => {
                aes_encrypt_x4_arm64(ctx, &counter, &mut stack_stream);
                let b0 = vld1q_u8(inptr);
                let b1 = vld1q_u8(inptr.add(16));
                let b2 = vld1q_u8(inptr.add(32));
                vst1q_u8(outptr, veorq_u8(b0, stack_stream[0]));
                vst1q_u8(outptr.add(16), veorq_u8(b1, stack_stream[1]));
                vst1q_u8(outptr.add(32), veorq_u8(b2, stack_stream[2]));
            }
            _ => {}
        }

        inptr = inptr.add(num_blocks * AES_BLOCK_SIZE);
        outptr = outptr.add(num_blocks * AES_BLOCK_SIZE);
        remain -= num_blocks * AES_BLOCK_SIZE;

        if leftover > 0 {
            let mut tmp = [0u8; 16];
            ptr::copy_nonoverlapping(inptr, tmp.as_mut_ptr(), leftover);
            let inblk = vld1q_u8(tmp.as_ptr());
            let outblk = veorq_u8(inblk, stack_stream[num_blocks]);
            vst1q_u8(tmp.as_mut_ptr(), outblk);
            ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
        }
    }

    AesGcmSivStatus::Success
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn key_exp_128_arm64(key: &mut [uint8x16_t], i: usize, r: u32) -> uint8x16_t {
    let temp = vextq_u8(key[i], key[i], 4);
    let temp = vaeseq_u8(temp, vdupq_n_u8(0));
    let temp = veorq_u8(temp, vreinterpretq_u8_u32(vdupq_n_u32(r)));
    let temp = veorq_u8(
        temp,
        vextq_u8(
            vdupq_n_u8(0),
            veorq_u8(
                veorq_u8(vld1q_u8(&[0u8; 16]), vextq_u8(key[i], vdupq_n_u8(0), 12)),
                vextq_u8(vdupq_n_u8(0), key[i], 8),
            ),
            4,
        ),
    );
    veorq_u8(temp, key[i])
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn key_exp_256_1_arm64(key: &mut [uint8x16_t], i: usize, r: u32) -> uint8x16_t {
    let temp = vextq_u8(key[i + 1], key[i + 1], 4);
    let temp = vaeseq_u8(temp, vdupq_n_u8(0));
    let temp = veorq_u8(temp, vreinterpretq_u8_u32(vdupq_n_u32(r)));
    let temp = veorq_u8(
        temp,
        vextq_u8(
            vdupq_n_u8(0),
            veorq_u8(
                veorq_u8(vld1q_u8(&[0u8; 16]), vextq_u8(key[i], vdupq_n_u8(0), 12)),
                vextq_u8(vdupq_n_u8(0), key[i], 8),
            ),
            4,
        ),
    );
    veorq_u8(temp, key[i])
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn key_exp_256_2_arm64(key: &mut [uint8x16_t], i: usize) -> uint8x16_t {
    let temp = vextq_u8(key[i + 1], vdupq_n_u8(0), 12);
    let temp = vaeseq_u8(temp, vdupq_n_u8(0));
    let temp = veorq_u8(
        temp,
        vextq_u8(
            vdupq_n_u8(0),
            veorq_u8(
                veorq_u8(vld1q_u8(&[0u8; 16]), vextq_u8(key[i], vdupq_n_u8(0), 12)),
                vextq_u8(vdupq_n_u8(0), key[i], 8),
            ),
            4,
        ),
    );
    veorq_u8(temp, key[i])
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn aes_encrypt_arm64(ctx: &AesArm64, mut block: uint8x16_t) -> uint8x16_t {
    if ctx.key_size == 32 {
        block = vaeseq_u8(block, ctx.key[0]);
        block = vaesmcq_u8(block);
        block = vaeseq_u8(block, ctx.key[1]);
        block = vaesmcq_u8(block);
        block = vaeseq_u8(block, ctx.key[2]);
        block = vaesmcq_u8(block);
        block = vaeseq_u8(block, ctx.key[3]);
        block = vaesmcq_u8(block);
    }

    block = vaeseq_u8(block, ctx.key[4]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[5]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[6]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[7]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[8]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[9]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[10]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[11]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[12]);
    block = vaesmcq_u8(block);
    block = vaeseq_u8(block, ctx.key[13]);
    block = veorq_u8(block, ctx.key[14]);
    block
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn add_u32x4_arm64(a: uint8x16_t, b: uint32x4_t) -> uint8x16_t {
    let x = vreinterpretq_u32_u8(a);
    let y = b;
    vreinterpretq_u8_u32(vaddq_u32(x, y))
}

#[cfg(target_arch = "aarch64")]
unsafe fn aes_encrypt_x4_arm64(ctx: &AesArm64, counter: &[uint8x16_t; 4], out: &mut [uint8x16_t; 4]) {
    for i in 0..4 {
        out[i] = veorq_u8(counter[i], ctx.key[0]);
    }
    if ctx.key_size == 32 {
        for r in 1..4 {
            for i in 0..4 {
                out[i] = vaeseq_u8(out[i], ctx.key[r]);
                out[i] = vaesmcq_u8(out[i]);
            }
        }
    }
    for r in 4..13 {
        for i in 0..4 {
            out[i] = vaeseq_u8(out[i], ctx.key[r]);
            out[i] = vaesmcq_u8(out[i]);
        }
    }
    for i in 0..4 {
        out[i] = vaeseq_u8(out[i], ctx.key[13]);
        out[i] = veorq_u8(out[i], ctx.key[14]);
    }
}

// ====================== Polyval Implementation ======================

trait Polyval {
    fn new(key: &[u8; POLYVAL_SIZE]) -> Self;
    fn update(&mut self, data: &[u8]);
    fn finish(&mut self, nonce: &[u8]) -> [u8; POLYVAL_SIZE];
}

// Generic Polyval Implementation
struct PolyvalGeneric {
    s: [u8; POLYVAL_SIZE],
    hl: [u64; 16],
    hh: [u64; 16],
}

const PL: [u64; 16] = [
    0x0000000000000000, 0x1c20000000000000, 0x3840000000000000, 0x2460000000000000,
    0x7080000000000000, 0x6c20000000000000, 0x4840000000000000, 0x5460000000000000,
    0xe100000000000000, 0xf840000000000000, 0xd460000000000000, 0xc200000000000000,
    0x9100000000000000, 0x8840000000000000, 0xa460000000000000, 0xb200000000000000,
];
const PH: [u64; 16] = [
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x1c20000000000000, 0x3840000000000000, 0x2460000000000000,
    0x7080000000000000, 0x6c20000000000000, 0x4840000000000000, 0x5460000000000000,
];

impl Polyval for PolyvalGeneric {
    fn new(key: &[u8; POLYVAL_SIZE]) -> Self {
        let mut ctx = Self {
            s: [0u8; POLYVAL_SIZE],
            hl: [0u64; 16],
            hh: [0u64; 16],
        };
        polyval_generic_start(&mut ctx, key);
        ctx
    }

    fn update(&mut self, data: &[u8]) {
        polyval_generic_update(self, data);
    }

    fn finish(&mut self, nonce: &[u8]) -> [u8; POLYVAL_SIZE] {
        let mut tag = [0u8; POLYVAL_SIZE];
        polyval_generic_finish(self, nonce, &mut tag);
        tag
    }
}

fn polyval_generic_start(ctx: &mut PolyvalGeneric, key: &[u8; POLYVAL_SIZE]) {
    let mut dot_ctx = DotContext {
        hl: 0,
        hh: 0,
        lo: 0,
        hi: 0,
        rem: 0,
    };
    dot(&mut dot_ctx, key, &XL, &XH);

    ctx.hl[0] = 0;
    ctx.hh[0] = 0;
    ctx.hl[1] = dot_ctx.hl;
    ctx.hh[1] = dot_ctx.hh;

    let mut hl = dot_ctx.hl;
    let mut hh = dot_ctx.hh;
    for i in (2..16).step_by(2) {
        dot_ctx.rem = (hh >> 63) & 0x01;
        hh = (hh << 1) ^ (hl >> 63) ^ PH[dot_ctx.rem as usize];
        hl = (hl << 1) ^ PL[dot_ctx.rem as usize];

        ctx.hl[i] = hl;
        ctx.hh[i] = hh;

        for j in 1..i {
            ctx.hl[i + j] = hl ^ ctx.hl[j];
            ctx.hh[i + j] = hh ^ ctx.hh[j];
        }
    }

    for i in 0..16 {
        ctx.s[i] = 0;
    }
}

fn polyval_generic_update(ctx: &mut PolyvalGeneric, data: &[u8]) {
    let mut idx = 0;
    while idx + 16 <= data.len() {
        for i in 0..16 {
            ctx.s[i] ^= data[idx + i];
        }

        let mut dot_ctx = DotContext {
            hl: 0,
            hh: 0,
            lo: 0,
            hi: 0,
            rem: 0,
        };
        dot(&mut dot_ctx, &ctx.s, &ctx.hl, &ctx.hh);
        put_u64_le(dot_ctx.hl, &mut ctx.s, 0);
        put_u64_le(dot_ctx.hh, &mut ctx.s, 8);
        idx += 16;
    }

    if idx < data.len() {
        for i in 0..(data.len() - idx) {
            ctx.s[i] ^= data[idx + i];
        }
        let mut dot_ctx = DotContext {
            hl: 0,
            hh: 0,
            lo: 0,
            hi: 0,
            rem: 0,
        };
        dot(&mut dot_ctx, &ctx.s, &ctx.hl, &ctx.hh);
        put_u64_le(dot_ctx.hl, &mut ctx.s, 0);
        put_u64_le(dot_ctx.hh, &mut ctx.s, 8);
    }
}

fn polyval_generic_finish(ctx: &mut PolyvalGeneric, nonce: &[u8], tag: &mut [u8]) {
    for i in 0..nonce.len() {
        tag[i] = ctx.s[i] ^ nonce[i];
    }
    for i in nonce.len()..16 {
        tag[i] = ctx.s[i];
    }
}

struct DotContext {
    hl: u64,
    hh: u64,
    lo: u64,
    hi: u64,
    rem: u64,
}

fn dot(dot: &mut DotContext, a: &[u8], bl: &[u64; 16], bh: &[u64; 16]) {
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

const XL: [u64; 16] = [
    0x0000000000000000, 0x0000000000000001, 0x0000000000000003, 0x0000000000000002,
    0x0000000000000007, 0x0000000000000006, 0x0000000000000004, 0x0000000000000005,
    0x000000000000000e, 0x000000000000000d, 0x000000000000000f, 0x000000000000000c,
    0x000000000000000b, 0x000000000000000a, 0x0000000000000009, 0x0000000000000008,
];
const XH: [u64; 16] = [
    0x0000000000000000, 0xc200000000000000, 0x4600000000000000, 0x8400000000000000,
    0x8c00000000000000, 0x4e00000000000000, 0xca00000000000000, 0x0800000000000000,
    0xda00000000000000, 0x1800000000000000, 0x9c00000000000000, 0x5e00000000000000,
    0x5600000000000000, 0x9400000000000000, 0x1000000000000000, 0xd200000000000000,
];

// x86-64 Optimized Polyval Implementation (using CLMUL)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Default)]
struct PolyvalX86_64 {
    s: __m128i,
    h_table: [__m128i; 8],
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Polyval for PolyvalX86_64 {
    fn new(key: &[u8; POLYVAL_SIZE]) -> Self {
        let mut ctx = Self::default();
        unsafe {
            ctx.h_table[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
            for i in 1..8 {
                ctx.h_table[i] = dot_x86_64(ctx.h_table[0], ctx.h_table[i - 1]);
            }
        }
        ctx
    }

    fn update(&mut self, data: &[u8]) {
        unsafe {
            self.s = polyval_x86_64_process_tables(&self.h_table, self.s, data.as_ptr(), data.len());
        }
    }

    fn finish(&mut self, nonce: &[u8]) -> [u8; POLYVAL_SIZE] {
        let mut tmp = [0u8; 16];
        tmp[..nonce.len()].copy_from_slice(nonce);
        unsafe {
            let n = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
            let out = _mm_xor_si128(n, self.s);
            let mut tag = [0u8; 16];
            _mm_storeu_si128(tag.as_mut_ptr() as *mut __m128i, out);
            tag
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn dot_x86_64(a: __m128i, b: __m128i) -> __m128i {
    let mut c0 = _mm_setzero_si128();
    let mut c1 = _mm_setzero_si128();
    let mut c2 = _mm_setzero_si128();
    mult_x86_64(a, b, &mut c0, &mut c1, &mut c2);
    mult_inv_x128_x86_64(c0, c1, c2)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn mult_x86_64(a: __m128i, b: __m128i, c0: &mut __m128i, c1: &mut __m128i, c2: &mut __m128i) {
    *c0 = _mm_clmulepi64_si128(a, b, 0x00);
    *c2 = _mm_clmulepi64_si128(a, b, 0x11);
    *c1 = _mm_xor_si128(
        _mm_clmulepi64_si128(a, b, 0x01),
        _mm_clmulepi64_si128(a, b, 0x10),
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn mult_inv_x64_x86_64(p: __m128i) -> __m128i {
    let q = _mm_shuffle_epi32(p, 0x4e);
    let poly = _mm_setr_epi32(0, 0xc2000000u32 as i32, 0x00000001u32 as i32, 0);
    let r = _mm_clmulepi64_si128(p, poly, 0x00);
    _mm_xor_si128(q, r)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn mult_inv_x128_x86_64(p0: __m128i, p1: __m128i, p2: __m128i) -> __m128i {
    let q = _mm_xor_si128(p0, _mm_slli_si128(p1, 8));
    let r = _mm_xor_si128(p2, _mm_srli_si128(p1, 8));
    let s = mult_inv_x64_x86_64(q);
    let t = mult_inv_x64_x86_64(s);
    _mm_xor_si128(r, t)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "pclmul")]
unsafe fn polyval_x86_64_process_tables(
    h_table: &[__m128i; 8],
    mut s: __m128i,
    mut data: *const u8,
    mut data_sz: usize,
) -> __m128i {
    let mut tmp = [0u8; 16];

    while data_sz >= 8 * 16 {
        let blocks = data as *const __m128i;
        let d0 = _mm_loadu_si128(blocks);
        let d1 = _mm_loadu_si128(blocks.add(1));
        let d2 = _mm_loadu_si128(blocks.add(2));
        let d3 = _mm_loadu_si128(blocks.add(3));
        let d4 = _mm_loadu_si128(blocks.add(4));
        let d5 = _mm_loadu_si128(blocks.add(5));
        let d6 = _mm_loadu_si128(blocks.add(6));
        let d7 = _mm_loadu_si128(blocks.add(7));

        let mut s0;
        let mut s1;
        let mut s2;
        mult_x86_64(d7, h_table[0], &mut s0, &mut s1, &mut s2);
        add_mult_x86_64(d6, h_table[1], &mut s0, &mut s1, &mut s2);
        add_mult_x86_64(d5, h_table[2], &mut s0, &mut s1, &mut s2);
        add_mult_x86_64(d4, h_table[3], &mut s0, &mut s1, &mut s2);
        add_mult_x86_64(d3, h_table[4], &mut s0, &mut s1, &mut s2);
        add_mult_x86_64(d2, h_table[5], &mut s0, &mut s1, &mut s2);
        add_mult_x86_64(d1, h_table[6], &mut s0, &mut s1, &mut s2);
        let d0_s = _mm_xor_si128(s, d0);
        add_mult_x86_64(d0_s, h_table[7], &mut s0, &mut s1, &mut s2);

        s = mult_inv_x128_x86_64(s0, s1, s2);

        data = data.add(8 * 16);
        data_sz -= 8 * 16;
    }

    let blocks = data_sz / 16;
    if blocks > 0 {
        let bptr = data as *const __m128i;
        if blocks > 1 {
            let last = _mm_loadu_si128(bptr.add(blocks - 1));
            let mut s0;
            let mut s1;
            let mut s2;
            mult_x86_64(last, h_table[0], &mut s0, &mut s1, &mut s2);

            for i in 1..(blocks - 1) {
                let blk = _mm_loadu_si128(bptr.add(blocks - 1 - i));
                add_mult_x86_64(blk, h_table[i], &mut s0, &mut s1, &mut s2);
            }

            let first_s = _mm_xor_si128(s, _mm_loadu_si128(bptr));
            add_mult_x86_64(first_s, h_table[blocks - 1], &mut s0, &mut s1, &mut s2);
            s = mult_inv_x128_x86_64(s0, s1, s2);
        } else {
            let d0_s = _mm_xor_si128(s, _mm_loadu_si128(bptr));
            let mut s0;
            let mut s1;
            let mut s2;
            mult_x86_64(d0_s, h_table[0], &mut s0, &mut s1, &mut s2);
            s = mult_inv_x128_x86_64(s0, s1, s2);
        }
        data = data.add(blocks * 16);
        data_sz -= blocks * 16;
    }

    if data_sz > 0 {
        ptr::copy_nonoverlapping(data, tmp.as_mut_ptr(), data_sz);
        for i in data_sz..16 {
            tmp[i] = 0;
        }
        let d = _mm_xor_si128(s, _mm_loadu_si128(tmp.as_ptr() as *const __m128i));
        s = dot_x86_64(d, h_table[0]);
    }

    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn add_mult_x86_64(
    a: __m128i,
    b: __m128i,
    c0: &mut __m128i,
    c1: &mut __m128i,
    c2: &mut __m128i,
) {
    *c0 = _mm_xor_si128(*c0, _mm_clmulepi64_si128(a, b, 0x00));
    *c2 = _mm_xor_si128(*c2, _mm_clmulepi64_si128(a, b, 0x11));
    *c1 = _mm_xor_si128(
        *c1,
        _mm_xor_si128(
            _mm_clmulepi64_si128(a, b, 0x01),
            _mm_clmulepi64_si128(a, b, 0x10),
        ),
    );
}

// ARM64 Optimized Polyval Implementation
#[cfg(target_arch = "aarch64")]
#[derive(Default)]
struct PolyvalArm64 {
    s: uint8x16_t,
    h_table: [uint8x16_t; 8],
}

#[cfg(target_arch = "aarch64")]
impl Polyval for PolyvalArm64 {
    fn new(key: &[u8; POLYVAL_SIZE]) -> Self {
        let mut ctx = Self::default();
        unsafe {
            ctx.h_table[0] = vld1q_u8(key.as_ptr());
            for i in 1..8 {
                ctx.h_table[i] = dot_arm64(ctx.h_table[0], ctx.h_table[i - 1]);
            }
        }
        ctx
    }

    fn update(&mut self, data: &[u8]) {
        unsafe {
            self.s = polyval_arm64_process_tables(&self.h_table, self.s, data.as_ptr(), data.len());
        }
    }

    fn finish(&mut self, nonce: &[u8]) -> [u8; POLYVAL_SIZE] {
        let mut tmp = [0u8; 16];
        tmp[..nonce.len()].copy_from_slice(nonce);
        unsafe {
            let n = vld1q_u8(tmp.as_ptr());
            let out = veorq_u8(n, self.s);
            let mut tag = [0u8; 16];
            vst1q_u8(tag.as_mut_ptr(), out);
            tag
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_arm64(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut c0 = vdupq_n_u8(0);
    let mut c1 = vdupq_n_u8(0);
    let mut c2 = vdupq_n_u8(0);
    mult_arm64(a, b, &mut c0, &mut c1, &mut c2);
    mult_inv_x128_arm64(c0, c1, c2)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mult_arm64(a: uint8x16_t, b: uint8x16_t, c0: &mut uint8x16_t, c1: &mut uint8x16_t, c2: &mut uint8x16_t) {
    *c0 = vmull_p64(vget_low_p64(vreinterpretq_p64_u8(a)), vget_low_p64(vreinterpretq_p64_u8(b)));
    *c2 = vmull_p64(vget_high_p64(vreinterpretq_p64_u8(a)), vget_high_p64(vreinterpretq_p64_u8(b)));
    let a0b1 = vmull_p64(vget_low_p64(vreinterpretq_p64_u8(a)), vget_high_p64(vreinterpretq_p64_u8(b)));
    let a1b0 = vmull_p64(vget_high_p64(vreinterpretq_p64_u8(a)), vget_low_p64(vreinterpretq_p64_u8(b)));
    *c1 = veorq_u8(a0b1, a1b0);
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mult_inv_x64_arm64(p: uint8x16_t) -> uint8x16_t {
    let poly = 0xc200000000000000u64;
    let poly_p64 = vcreate_p64(poly);
    let q = vextq_u8(p, p, 8);
    let a0 = vget_low_p64(vreinterpretq_p64_u8(p));
    let r = vmull_p64(a0, vget_lane_p64(poly_p64, 0));
    veorq_u8(q, r)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mult_inv_x128_arm64(p0: uint8x16_t, p1: uint8x16_t, p2: uint8x16_t) -> uint8x16_t {
    let q = veorq_u8(p0, vextq_u8(vdupq_n_u8(0), p1, 8));
    let r = veorq_u8(p2, vextq_u8(p1, vdupq_n_u8(0), 8));
    let s = mult_inv_x64_arm64(q);
    let t = mult_inv_x64_arm64(s);
    veorq_u8(r, t)
}

#[cfg(target_arch = "aarch64")]
unsafe fn polyval_arm64_process_tables(
    h_table: &[uint8x16_t; 8],
    mut s: uint8x16_t,
    mut data: *const u8,
    mut data_sz: usize,
) -> uint8x16_t {
    let mut tmp = [0u8; 16];

    let blocks_8 = data_sz / (8 * POLYVAL_SIZE);
    for _ in 0..blocks_8 {
        let d0 = vld1q_u8(data);
        let d1 = vld1q_u8(data.add(16));
        let d2 = vld1q_u8(data.add(32));
        let d3 = vld1q_u8(data.add(48));
        let d4 = vld1q_u8(data.add(64));
        let d5 = vld1q_u8(data.add(80));
        let d6 = vld1q_u8(data.add(96));
        let d7 = vld1q_u8(data.add(112));

        let mut s0 = vdupq_n_u8(0);
        let mut s1 = vdupq_n_u8(0);
        let mut s2 = vdupq_n_u8(0);

        mult_arm64(d7, h_table[0], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(d6, h_table[1], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(d5, h_table[2], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(d4, h_table[3], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(d3, h_table[4], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(d2, h_table[5], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(d1, h_table[6], &mut s0, &mut s1, &mut s2);
        add_mult_arm64(veorq_u8(s, d0), h_table[7], &mut s0, &mut s1, &mut s2);

        s = mult_inv_x128_arm64(s0, s1, s2);

        data = data.add(8 * 16);
        data_sz -= 8 * 16;
    }

    let blocks = data_sz / 16;
    if blocks > 0 {
        if blocks > 1 {
            let last = vld1q_u8(data.add((blocks - 1) * 16));
            let mut s0 = vdupq_n_u8(0);
            let mut s1 = vdupq_n_u8(0);
            let mut s2 = vdupq_n_u8(0);
            mult_arm64(last, h_table[0], &mut s0, &mut s1, &mut s2);
            for i in 1..(blocks - 1) {
                let blk = vld1q_u8(data.add((blocks - 1 - i) * 16));
                add_mult_arm64(blk, h_table[i], &mut s0, &mut s1, &mut s2);
            }
            let first = veorq_u8(s, vld1q_u8(data));
            add_mult_arm64(first, h_table[blocks - 1], &mut s0, &mut s1, &mut s2);
            s = mult_inv_x128_arm64(s0, s1, s2);
        } else {
            let first = veorq_u8(s, vld1q_u8(data));
            let mut s0 = vdupq_n_u8(0);
            let mut s1 = vdupq_n_u8(0);
            let mut s2 = vdupq_n_u8(0);
            mult_arm64(first, h_table[0], &mut s0, &mut s1, &mut s2);
            s = mult_inv_x128_arm64(s0, s1, s2);
        }

        data = data.add(blocks * 16);
        data_sz -= blocks * 16;
    }

    if data_sz > 0 {
        ptr::copy_nonoverlapping(data, tmp.as_mut_ptr(), data_sz);
        for i in data_sz..16 {
            tmp[i] = 0;
        }
        let d = veorq_u8(s, vld1q_u8(tmp.as_ptr()));
        s = dot_arm64(d, h_table[0]);
    }

    s
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn add_mult_arm64(
    a: uint8x16_t,
    b: uint8x16_t,
    c0: &mut uint8x16_t,
    c1: &mut uint8x16_t,
    c2: &mut uint8x16_t,
) {
    let mut aa0 = vdupq_n_u8(0);
    let mut aa1 = vdupq_n_u8(0);
    let mut aa2 = vdupq_n_u8(0);
    mult_arm64(a, b, &mut aa0, &mut aa1, &mut aa2);
    *c0 = veorq_u8(*c0, aa0);
    *c2 = veorq_u8(*c2, aa2);
    *c1 = veorq_u8(*c1, aa1);
}

// ====================== AES-GCM-SIV Functions ======================

fn aes_gcmsiv_derive_keys(aes_ctx: *mut Aes, key_size: usize, nonce: &[u8], key: &mut KeyContext) {
    let mut h = [0u8; AES_BLOCK_SIZE];
    unsafe {
        let aes = &mut *aes_ctx;
        h = aes.ecb_encrypt(&[0u8; AES_BLOCK_SIZE]);
    }

    let h_poly = PolyvalGeneric::new(&h);
    let mut polyval_ctx = PolyvalGeneric::new(&[0u8; POLYVAL_SIZE]);
    polyval_generic_update(&mut polyval_ctx, nonce);

    let mut tag = [0u8; POLYVAL_SIZE];
    polyval_generic_finish(&mut polyval_ctx, &[], &mut tag);
    tag[15] |= 0x80;

    let mut result = [0u8; POLYVAL_SIZE];
    polyval_generic_start(&mut polyval_ctx, &tag);
    polyval_generic_finish(&mut polyval_ctx, &[], &mut result);
    result = polyval_mul(&result, &h_poly.hl, &h_poly.hh);

    key.auth_sz = KEY_AUTH_SIZE;
    key.auth.copy_from_slice(&result);

    if key_size == 32 {
        key.enc_sz = 32;
        polyval_generic_start(&mut polyval_ctx, &tag);
        polyval_generic_finish(&mut polyval_ctx, &[1u8], &mut result);
        result = polyval_mul(&result, &h_poly.hl, &h_poly.hh);
        key.enc[0..16].copy_from_slice(&result);

        polyval_generic_start(&mut polyval_ctx, &tag);
        polyval_generic_finish(&mut polyval_ctx, &[2u8], &mut result);
        result = polyval_mul(&result, &h_poly.hl, &h_poly.hh);
        key.enc[16..32].copy_from_slice(&result);
    } else {
        key.enc_sz = 16;
        key.enc[0..16].copy_from_slice(&key.auth);
    }
}

fn aes_gcmsiv_make_tag(
    key: &KeyContext,
    nonce: &[u8],
    plain: &[u8],
    aad: &[u8],
    tag: &mut [u8],
) {
    let mut ctx = AesGcmSivCtx::new();
    ctx.set_key(&key.enc[0..key.enc_sz]);

    let mut derived_key = KeyContext::default();
    aes_gcmsiv_derive_keys(ctx.key_gen_ctx, ctx.key_sz, nonce, &mut derived_key);

    let mut polyval_ctx = PolyvalGeneric::new(&derived_key.auth);
    polyval_generic_update(&mut polyval_ctx, aad);
    polyval_generic_update(&mut polyval_ctx, plain);

    let mut length_block = [0u8; 16];
    let aad_bit_sz = (aad.len() as u64) * 8;
    put_u64_le(aad_bit_sz, &mut length_block, 0);
    let plain_bit_sz = (plain.len() as u64) * 8;
    put_u64_le(plain_bit_sz, &mut length_block, 8);
    polyval_generic_update(&mut polyval_ctx, &length_block);

    polyval_generic_finish(&mut polyval_ctx, nonce, tag);

    tag[15] &= 0x7F;

    let mut block = [0u8; 16];
    block.copy_from_slice(tag);
    unsafe {
        let aes = &mut *ctx.key_gen_ctx;
        *tag = aes.ecb_encrypt(&block);
    }
}

fn aes_gcmsiv_aes_ctr(
    key: &[u8],
    key_sz: usize,
    tag: &[u8; 16],
    input: &[u8],
    output: &mut [u8],
) {
    let mut ctx = AesGcmSivCtx::new();
    ctx.set_key(key);

    let mut nonce_ctr = [0u8; 16];
    nonce_ctr.copy_from_slice(tag);
    nonce_ctr[15] |= 0x80;

    unsafe {
        let aes = &mut *ctx.key_gen_ctx;
        aes.ctr(&nonce_ctr, input, output);
    }
}

fn aes_gcmsiv_check_tag(lhs: &[u8; 16], rhs: &[u8; 16]) -> AesGcmSivStatus {
    let mut result = 0;
    for i in 0..16 {
        result |= lhs[i] ^ rhs[i];
    }
    if result == 0 {
        AesGcmSivStatus::Success
    } else {
        AesGcmSivStatus::InvalidTag
    }
}

fn aes_gcmsiv_encrypt_size(
    plain_sz: usize,
    _aad_sz: usize,
    cipher_sz: &mut usize,
) -> AesGcmSivStatus {
    if plain_sz > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
        return AesGcmSivStatus::InvalidPlaintextSize;
    }

    *cipher_sz = plain_sz + AES_GCMSIV_TAG_SIZE;
    AesGcmSivStatus::Success
}

fn aes_gcmsiv_decrypt_size(
    cipher_sz: usize,
    _aad_sz: usize,
    plain_sz: &mut usize,
) -> AesGcmSivStatus {
    if cipher_sz < AES_GCMSIV_TAG_SIZE {
        return AesGcmSivStatus::InvalidCiphertextSize;
    }

    *plain_sz = cipher_sz - AES_GCMSIV_TAG_SIZE;
    AesGcmSivStatus::Success
}

impl AesGcmSivCtx {
    fn new() -> Self {
        Self {
            key_gen_ctx: ptr::null_mut(),
            key_sz: 0,
        }
    }

    fn set_key(&mut self, key: &[u8]) -> AesGcmSivStatus {
        let key_gen_ctx = match key.len() {
            16 | 32 => {
                // Determine the appropriate AES implementation based on CPU features
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("aes") {
                        let mut ctx = AesX86_64::new();
                        if ctx.set_key(key) != AesGcmSivStatus::Success {
                            return AesGcmSivStatus::InvalidKeySize;
                        }
                        Box::into_raw(Box::new(ctx))
                    } else {
                        let mut ctx = AesGeneric::new();
                        if ctx.set_key(key) != AesGcmSivStatus::Success {
                            return AesGcmSivStatus::InvalidKeySize;
                        }
                        Box::into_raw(Box::new(ctx))
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if is_aarch64_feature_detected!("aes") {
                        let mut ctx = AesArm64::new();
                        if ctx.set_key(key) != AesGcmSivStatus::Success {
                            return AesGcmSivStatus::InvalidKeySize;
                        }
                        Box::into_raw(Box::new(ctx))
                    } else {
                        let mut ctx = AesGeneric::new();
                        if ctx.set_key(key) != AesGcmSivStatus::Success {
                            return AesGcmSivStatus::InvalidKeySize;
                        }
                        Box::into_raw(Box::new(ctx))
                    }
                }
                #[cfg(not(any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64"
                )))]
                {
                    let mut ctx = AesGeneric::new();
                    if ctx.set_key(key) != AesGcmSivStatus::Success {
                        return AesGcmSivStatus::InvalidKeySize;
                    }
                    Box::into_raw(Box::new(ctx))
                }
            }
            _ => return AesGcmSivStatus::InvalidKeySize,
        };

        if !self.key_gen_ctx.is_null() {
            unsafe {
                drop(Box::from_raw(self.key_gen_ctx));
            }
        }

        self.key_gen_ctx = key_gen_ctx;
        self.key_sz = key.len();
        AesGcmSivStatus::Success
    }

    fn encrypt_with_tag(
        &mut self,
        nonce: &[u8],
        plain: &[u8],
        aad: &[u8],
        cipher: &mut [u8],
    ) -> AesGcmSivStatus {
        if nonce.len() != AES_GCMSIV_NONCE_SIZE {
            return AesGcmSivStatus::InvalidNonceSize;
        }

        let mut needed_sz = 0;
        match aes_gcmsiv_encrypt_size(plain.len(), aad.len(), &mut needed_sz) {
            AesGcmSivStatus::Success => (),
            e => return e
        }

        if cipher.len() < needed_sz {
            return AesGcmSivStatus::UpdateOutputSize;
        }

        let key_gen_ctx = self.key_gen_ctx;
        if key_gen_ctx.is_null() {
            return AesGcmSivStatus::Failure;
        }

        let mut key = KeyContext::default();
        aes_gcmsiv_derive_keys(key_gen_ctx, self.key_sz, nonce, &mut key);

        let tag_offset = plain.len();
        let (cipher_data, tag_buf) = cipher.split_at_mut(tag_offset);
        aes_gcmsiv_make_tag(&key, nonce, plain, aad, tag_buf);

        aes_gcmsiv_aes_ctr(
            &key.enc[0..key.enc_sz],
            key.enc_sz,
            tag_buf.try_into().unwrap(),
            plain,
            cipher_data,
        );

        secure_zeroize_struct(&mut key);
        AesGcmSivStatus::Success
    }

    fn decrypt_and_check(
        &mut self,
        nonce: &[u8],
        cipher: &[u8],
        aad: &[u8],
        plain: &mut [u8],
    ) -> AesGcmSivStatus {
        if nonce.len() != AES_GCMSIV_NONCE_SIZE {
            return AesGcmSivStatus::InvalidNonceSize;
        }

        let mut needed_sz = 0;
        match aes_gcmsiv_decrypt_size(cipher.len(), aad.len(), &mut needed_sz) {
            AesGcmSivStatus::Success => (),
            e => return e
        }

        if plain.len() < needed_sz {
            return AesGcmSivStatus::UpdateOutputSize;
        }

        let key_gen_ctx = self.key_gen_ctx;
        if key_gen_ctx.is_null() {
            return AesGcmSivStatus::Failure;
        }

        let mut key = KeyContext::default();
        aes_gcmsiv_derive_keys(key_gen_ctx, self.key_sz, nonce, &mut key);

        let expected_tag = &cipher[cipher.len() - AES_GCMSIV_TAG_SIZE..];
        let ciphertext_data = &cipher[0..cipher.len() - AES_GCMSIV_TAG_SIZE];

        aes_gcmsiv_aes_ctr(
            &key.enc[0..key.enc_sz],
            key.enc_sz,
            expected_tag.try_into().unwrap(),
            ciphertext_data,
            plain,
        );

        let mut computed_tag = [0u8; 16];
        aes_gcmsiv_make_tag(&key, nonce, plain, aad, &mut computed_tag);

        let res = aes_gcmsiv_check_tag(&computed_tag, expected_tag.try_into().unwrap());
        if res != AesGcmSivStatus::Success {
            for i in 0..needed_sz {
                plain[i] = 0;
            }
            secure_zeroize_struct(&mut key);
            return res;
        }

        secure_zeroize_struct(&mut key);
        AesGcmSivStatus::Success
    }
}

fn main() {
    // Example usage
    let key = b"thiskeyisverybad";
    let nonce = b"abcdefghijkl";
    let plaintext = b"AES-GCM-SIV encryption example";
    let aad = b"Some additional data";

    let mut ctx = AesGcmSivCtx::new();
    assert_eq!(ctx.set_key(key), AesGcmSivStatus::Success);

    let mut ciphertext = vec![0u8; plaintext.len() + AES_GCMSIV_TAG_SIZE];
    assert_eq!(
        ctx.encrypt_with_tag(nonce, plaintext, aad, &mut ciphertext),
        AesGcmSivStatus::Success
    );

    let mut decrypted = vec![0u8; plaintext.len()];
    assert_eq!(
        ctx.decrypt_and_check(nonce, &ciphertext, aad, &mut decrypted),
        AesGcmSivStatus::Success
    );

    assert_eq!(decrypted, plaintext);
}

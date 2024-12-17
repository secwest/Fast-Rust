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

use std::ptr;

const AES_GCMSIV_TAG_SIZE: usize = 16;
const AES_GCMSIV_NONCE_SIZE: usize = 12;
const POLYVAL_SIZE: usize = 16;
const AES_BLOCK_SIZE: usize = 16;

const AES_GCMSIV_MAX_PLAINTEXT_SIZE: usize = (1 << 36) - 1;
const AES_GCMSIV_MAX_AAD_SIZE: usize = (1 << 36) - 1;

const KEY_AUTH_SIZE: usize = 16;
const KEY_ENC_MAX_SIZE: usize = 32;

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

#[derive(Copy, Clone)]
struct key_context {
    auth: [u8; KEY_AUTH_SIZE],
    auth_sz: usize,
    enc: [u8; KEY_ENC_MAX_SIZE],
    enc_sz: usize,
}

#[repr(C)]
struct aes_gcmsiv_ctx {
    key_gen_ctx: *mut aes,
    key_sz: usize,
}

#[inline]
fn PUT_UINT32_LE(val: u32, dst: &mut [u8], offset: usize) {
    dst[offset] = (val & 0xff) as u8;
    dst[offset+1] = ((val >> 8) & 0xff) as u8;
    dst[offset+2] = ((val >> 16) & 0xff) as u8;
    dst[offset+3] = ((val >> 24) & 0xff) as u8;
}

#[inline]
fn GET_UINT32_LE(src: &[u8], offset: usize) -> u32 {
    (src[offset] as u32)
        | ((src[offset+1] as u32) << 8)
        | ((src[offset+2] as u32) << 16)
        | ((src[offset+3] as u32) << 24)
}

#[inline]
fn PUT_UINT64_LE(val: u64, dst: &mut [u8], offset: usize) {
    let v = val.to_le_bytes();
    dst[offset..offset+8].copy_from_slice(&v);
}

#[inline]
fn aes_gcmsiv_zeroize(buf: &mut [u8]) {
    for x in buf.iter_mut() {
        unsafe {
            ptr::write_volatile(x, 0);
        }
    }
}

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

#[inline]
fn aes_gcmsiv_has_feature(_what: i32) -> i32 {
    0
}

// =========================== AES generic code =================================

// The following code is translated from aes_generic.c and aes_generic_tables.h
// We include the forward S-box and tables as static constants.

static FSb: [u8;256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

static RCON: [u32;10] = [
    0x00000001,0x00000002,0x00000004,0x00000008,0x00000010,
    0x00000020,0x00000040,0x00000080,0x0000001b,0x00000036,
];

#[inline]
fn ROTL8(x: u32) -> u32 {
    ((x << 8) & 0xffffffff) | (x >> 24)
}

#[inline]
fn ROTL16(x: u32) -> u32 {
    ((x << 16) & 0xffffffff) | (x >> 16)
}

#[inline]
fn ROTL24(x: u32) -> u32 {
    ((x << 24) & 0xffffffff) | (x >> 8)
}

#[inline]
fn XTIME(x: u8) -> u8 {
    let mut val = x<<1;
    if (x &0x80)!=0 { val ^=0x1B; }
    (val &0xFF)
}

// We must generate FT0, FT1, FT2, FT3 tables. The code in C does it at runtime if AES_GENERIC_ROM_TABLES is not defined.
// Here we replicate that code at runtime.

static mut FT0: [u32;256] = [0;256];
static mut FT1: [u32;256] = [0;256];
static mut FT2: [u32;256] = [0;256];
static mut FT3: [u32;256] = [0;256];

static mut AES_GENERIC_GEN_TABLES_IS_INIT: bool = false;

#[inline]
fn aes_generic_gen_tables() {
    unsafe {
        if AES_GENERIC_GEN_TABLES_IS_INIT {
            return;
        }
        let mut pow = [0u8;256];
        let mut log = [0u8;256];

        let mut x:u8=1;
        for i in 0..256 {
            pow[i]=x;
            log[x as usize]=i as u8;
            x = ((x as u32)^(XTIME(x))) as u8;
        }

        // Generate forward tables
        for i in 0..256 {
            let x = FSb[i];
            let y = XTIME(x);
            let z = y ^ x;

            let t = ((y as u32) )
                  ^ ((x as u32)<<8)
                  ^ ((x as u32)<<16)
                  ^ ((z as u32)<<24);

            FT0[i] = t;
        }

        // If AES_GENERIC_FEWER_TABLES not defined, we precompute FT1,FT2,FT3
        for i in 0..256 {
            FT1[i] = ROTL8(FT0[i]);
            FT2[i] = ROTL16(FT0[i]);
            FT3[i] = ROTL24(FT0[i]);
        }

        AES_GENERIC_GEN_TABLES_IS_INIT=true;
    }
}

struct aes_generic {
    nr: i32,
    rk: *mut u32,
    buf: [u32;68],
}

impl aes_generic {
    fn new() -> Self {
        let mut ctx = aes_generic {
            nr: 0,
            rk: std::ptr::null_mut(),
            buf: [0;68],
        };
        ctx.rk = ctx.buf.as_ptr() as *mut u32;
        ctx
    }
}

fn aes_generic_init(ctx: &mut aes_generic) {
    ctx.nr=0;
    ctx.rk=ctx.buf.as_ptr() as *mut u32;
}

fn aes_generic_free(ctx: &mut aes_generic) {
    aes_gcmsiv_zeroize_struct(ctx);
}

fn aes_generic_set_key(ctx: &mut aes_generic, key: &[u8]) -> aes_gcmsiv_status_t {
    aes_generic_gen_tables();

    let key_sz = key.len();
    let nr = match key_sz {
        16 => 10,
        24 => 12,
        32 => 14,
        _ => return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE,
    };
    ctx.nr = nr;
    let RK = ctx.buf.as_mut_ptr();

    let nk = key_sz >> 2;
    for i in 0..nk {
        let v = GET_UINT32_LE(key,i*4);
        unsafe {
            *RK.add(i)=v;
        }
    }

    unsafe {
        match ctx.nr {
        10 => {
            for i in 0..10 {
                let off=(4*i) as usize;
                let temp = *RK.add(off+3);
                let t = ((FSb[((temp >>8)&0xFF)as usize] as u32))
                      |((FSb[((temp>>16)&0xFF)as usize]as u32)<<8)
                      |((FSb[((temp>>24)&0xFF)as usize]as u32)<<16)
                      |((FSb[((temp)&0xFF)as usize]as u32)<<24);
                *RK.add(off+4)= *RK.add(off+0)^ RCON[i]^t;
                *RK.add(off+5)= *RK.add(off+1)^ *RK.add(off+4);
                *RK.add(off+6)= *RK.add(off+2)^ *RK.add(off+5);
                *RK.add(off+7)= *RK.add(off+3)^ *RK.add(off+6);
            }
        }
        12 => {
            for i in 0..8 {
                let off=(6*i) as usize;
                let temp = *RK.add(off+5);
                let t = ((FSb[((temp >>8)&0xFF)as usize] as u32))
                      |((FSb[((temp>>16)&0xFF)as usize]<<8)as u32)
                      |((FSb[((temp>>24)&0xFF)as usize]<<16)as u32)
                      |((FSb[((temp)&0xFF)as usize]<<24)as u32);
                *RK.add(off+6)= *RK.add(off+0)^RCON[i]^t;
                *RK.add(off+7)= *RK.add(off+1)^*RK.add(off+6);
                *RK.add(off+8)= *RK.add(off+2)^*RK.add(off+7);
                *RK.add(off+9)= *RK.add(off+3)^*RK.add(off+8);
                *RK.add(off+10)= *RK.add(off+4)^*RK.add(off+9);
                *RK.add(off+11)= *RK.add(off+5)^*RK.add(off+10);
            }
        }
        14 => {
            for i in 0..7 {
                let off=(8*i) as usize;
                let temp = *RK.add(off+7);
                let t = ((FSb[((temp >>8)&0xFF)as usize] as u32))
                      |((FSb[((temp>>16)&0xFF)as usize]<<8)as u32)
                      |((FSb[((temp>>24)&0xFF)as usize]<<16)as u32)
                      |((FSb[((temp)&0xFF)as usize]<<24)as u32);
                *RK.add(off+8)= *RK.add(off+0)^RCON[i]^t;
                *RK.add(off+9)= *RK.add(off+1)^*RK.add(off+8);
                *RK.add(off+10)= *RK.add(off+2)^*RK.add(off+9);
                *RK.add(off+11)= *RK.add(off+3)^*RK.add(off+10);
                let temp2 = *RK.add(off+11);
                let t2 = ((FSb[((temp2)&0xFF)as usize]) as u32)
                        |((FSb[((temp2>>8)&0xFF)as usize] as u32)<<8)
                        |((FSb[((temp2>>16)&0xFF)as usize]<<16)as u32)
                        |((FSb[((temp2>>24)&0xFF)as usize]<<24)as u32);
                *RK.add(off+12)= *RK.add(off+4)^t2;
                *RK.add(off+13)= *RK.add(off+5)^*RK.add(off+12);
                *RK.add(off+14)= *RK.add(off+6)^*RK.add(off+13);
                *RK.add(off+15)= *RK.add(off+7)^*RK.add(off+14);
            }
        }
        _=>{}
        }
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

#[inline]
fn AES_FT0(idx: usize) -> u32 { unsafe{FT0[idx]} }
#[inline]
fn AES_FT1(idx: usize) -> u32 { unsafe{FT1[idx]} }
#[inline]
fn AES_FT2(idx: usize) -> u32 { unsafe{FT2[idx]} }
#[inline]
fn AES_FT3(idx: usize) -> u32 { unsafe{FT3[idx]} }

#[inline]
fn AES_FROUND(X0:&mut u32, X1:&mut u32, X2:&mut u32, X3:&mut u32, Y0:u32, Y1:u32, Y2:u32, Y3:u32, RK:&mut *const u32) {
    let rk = *RK;
    *X0 = unsafe{*rk} ^ AES_FT0((Y0 & 0xFF)as usize) ^ AES_FT1(((Y1 >>8)&0xFF)as usize)
          ^AES_FT2(((Y2>>16)&0xFF)as usize)^AES_FT3(((Y3>>24)&0xFF)as usize);
    *RK = (*RK).add(1);
    let rk = *RK;
    *X1 = unsafe{*rk} ^ AES_FT0((Y1 & 0xFF)as usize) ^ AES_FT1(((Y2 >>8)&0xFF)as usize)
          ^AES_FT2(((Y3>>16)&0xFF)as usize)^AES_FT3(((Y0>>24)&0xFF)as usize);
    *RK = (*RK).add(1);
    let rk = *RK;
    *X2 = unsafe{*rk} ^ AES_FT0((Y2 &0xFF)as usize) ^ AES_FT1(((Y3>>8)&0xFF)as usize)
          ^AES_FT2(((Y0>>16)&0xFF)as usize)^AES_FT3(((Y1>>24)&0xFF)as usize);
    *RK=(*RK).add(1);
    let rk = *RK;
    *X3 = unsafe{*rk} ^ AES_FT0((Y3 &0xFF)as usize) ^ AES_FT1(((Y0>>8)&0xFF)as usize)
          ^AES_FT2(((Y1>>16)&0xFF)as usize)^AES_FT3(((Y2>>24)&0xFF)as usize);
    *RK=(*RK).add(1);
}

fn aes_generic_ecb_encrypt(ctx: &aes_generic, plain: &[u8;16], cipher: &mut [u8;16]) -> aes_gcmsiv_status_t {
    let nr = ctx.nr;
    let mut X0:u32; let mut X1:u32; let mut X2:u32; let mut X3:u32;
    let mut Y0:u32; let mut Y1:u32; let mut Y2:u32; let mut Y3:u32;

    let RK = ctx.rk;
    unsafe {
        X0=GET_UINT32_LE(plain,0)^*RK;
        X1=GET_UINT32_LE(plain,4)^*RK.add(1);
        X2=GET_UINT32_LE(plain,8)^*RK.add(2);
        X3=GET_UINT32_LE(plain,12)^*RK.add(3);

        let mut p = RK.add(4);

        for _i in  (nr>>1)-1 .. (nr>>1)-1 { } // no-op
        // Actually do rounds
        for _ in 1..(nr>>1) {
            AES_FROUND(&mut Y0,&mut Y1,&mut Y2,&mut Y3,X0,X1,X2,X3,&mut p);
            AES_FROUND(&mut X0,&mut X1,&mut X2,&mut X3,Y0,Y1,Y2,Y3,&mut p);
        }

        // Last two rounds
        AES_FROUND(&mut Y0,&mut Y1,&mut Y2,&mut Y3,X0,X1,X2,X3,&mut p);

        // final round
        let mut r = p;
        X0 = *r ^ (FSb[(Y0 &0xFF)as usize] as u32)
             ^((FSb[((Y1>>8)&0xFF)as usize] as u32)<<8)
             ^((FSb[((Y2>>16)&0xFF)as usize] <<16)as u32)
             ^((FSb[((Y3>>24)&0xFF)as usize]<<24)as u32); r=r.add(1);

        X1 = *r ^ (FSb[(Y1 &0xFF)as usize] as u32)
             ^((FSb[((Y2>>8)&0xFF)as usize]<<8)as u32)
             ^((FSb[((Y3>>16)&0xFF)as usize]<<16)as u32)
             ^((FSb[((Y0>>24)&0xFF)as usize]<<24)as u32); r=r.add(1);

        X2 = *r ^ (FSb[(Y2&0xFF)as usize] as u32)
             ^((FSb[((Y3>>8)&0xFF)as usize]<<8)as u32)
             ^((FSb[((Y0>>16)&0xFF)as usize]<<16)as u32)
             ^((FSb[((Y1>>24)&0xFF)as usize]<<24)as u32); r=r.add(1);

        X3 = *r ^ (FSb[(Y3&0xFF)as usize] as u32)
             ^((FSb[((Y0>>8)&0xFF)as usize]<<8)as u32)
             ^((FSb[((Y1>>16)&0xFF)as usize]<<16)as u32)
             ^((FSb[((Y2>>24)&0xFF)as usize]<<24)as u32);

    }

    PUT_UINT32_LE(X0,cipher,0);
    PUT_UINT32_LE(X1,cipher,4);
    PUT_UINT32_LE(X2,cipher,8);
    PUT_UINT32_LE(X3,cipher,12);

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn aes_generic_ctr(ctx: &aes_generic, nonce: &[u8;16], input: &[u8], output: &mut [u8]) -> aes_gcmsiv_status_t {
    let mut counter_block = *nonce;
    let mut counter = GET_UINT32_LE(&counter_block,0);
    let mut key_stream = [0u8;16];

    let mut processed=0;

    while processed+16 <= input.len() {
        aes_generic_ecb_encrypt(ctx, &counter_block, &mut key_stream);
        counter=counter.wrapping_add(1);
        PUT_UINT32_LE(counter,&mut counter_block,0);
        for i in 0..16 {
            output[processed+i]=input[processed+i]^key_stream[i];
        }
        processed+=16;
    }

    if processed < input.len() {
        aes_generic_ecb_encrypt(ctx, &counter_block, &mut key_stream);
        counter=counter.wrapping_add(1);
        PUT_UINT32_LE(counter,&mut counter_block,0);
        for i in 0..(input.len()-processed) {
            output[processed+i]=input[processed+i]^key_stream[i];
        }
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

// AES wrapper struct
#[derive(Copy, Clone)]
union aes_storage {
    generic: aes_generic,
}

#[derive(Copy, Clone)]
struct aes {
    has_hw: i32,
    storage: aes_storage,
}

fn aes_init(ctx: &mut aes) {
    ctx.has_hw = aes_gcmsiv_has_feature(0);
    unsafe { aes_generic_init(&mut ctx.storage.generic); }
}

fn aes_free(ctx: &mut aes) {
    unsafe { aes_generic_free(&mut ctx.storage.generic); }
    aes_gcmsiv_zeroize_struct(ctx);
}

fn aes_set_key(ctx: &mut aes, key: &[u8]) -> aes_gcmsiv_status_t {
    unsafe { aes_generic_set_key(&mut ctx.storage.generic, key) }
}

fn aes_ecb_encrypt(ctx: &mut aes, plain: &[u8;16], cipher: &mut [u8;16]) -> aes_gcmsiv_status_t {
    unsafe { aes_generic_ecb_encrypt(&ctx.storage.generic, plain, cipher) }
}

fn aes_ctr(ctx: &mut aes, nonce: &[u8;16], input: &[u8], output: &mut [u8]) -> aes_gcmsiv_status_t {
    unsafe { aes_generic_ctr(&ctx.storage.generic, nonce, input, output) }
}

// ========================== Polyval generic code =========================

struct polyval_generic {
    S: [u8;16],
    HL: [u64;16],
    HH: [u64;16],
}

#[derive(Copy, Clone)]
union polyval_storage {
    generic: polyval_generic,
}

#[derive(Copy, Clone)]
struct polyval {
    has_hw: i32,
    storage: polyval_storage,
}

fn polyval_init(ctx: &mut polyval) {
    ctx.has_hw=0;
    unsafe {
        for i in 0..16 {
            ctx.storage.generic.S[i]=0;
        }
    }
}

fn polyval_free(ctx: &mut polyval) {
    aes_gcmsiv_zeroize_struct(ctx);
}

// Dot function from original code

static PL: [u64;16] = [
    0x0000000000000000,0x0000000000000001,0x0000000000000003,0x0000000000000002,
    0x0000000000000006,0x0000000000000007,0x0000000000000005,0x0000000000000004,
    0x000000000000000d,0x000000000000000c,0x000000000000000e,0x000000000000000f,
    0x000000000000000b,0x000000000000000a,0x0000000000000008,0x0000000000000009,
];

static PH: [u64;16] = [
    0x0000000000000000,0xc200000000000000,0x4600000000000000,0x8400000000000000,
    0x8c00000000000000,0x4e00000000000000,0xca00000000000000,0x0800000000000000,
    0xda00000000000000,0x1800000000000000,0x9c00000000000000,0x5e00000000000000,
    0x5600000000000000,0x9400000000000000,0x1000000000000000,0xd200000000000000,
];

struct dot_context {
    hl:u64,
    hh:u64,
    lo:u64,
    hi:u64,
    rem:u64,
}

fn dot(dot:&mut dot_context, a:&[u8], bl:&[u64;16], bh:&[u64;16]) {
    dot.hl=0;
    dot.hh=0;

    for i in 0..16 {
        let b = a[16-i-1];
        dot.hi = ((b>>4)&0x0f)as u64;
        dot.lo = (b&0x0f)as u64;

        dot.rem=(dot.hh>>60)&0x0f;
        dot.hh=((dot.hh<<4)| (dot.hl>>60)) ^ PH[dot.rem as usize]^ bh[dot.hi as usize];
        dot.hl=(dot.hl<<4)^PL[dot.rem as usize]^ bl[dot.hi as usize];

        dot.rem=(dot.hh>>60)&0x0f;
        dot.hh=((dot.hh<<4)|(dot.hl>>60))^PH[dot.rem as usize]^ bh[dot.lo as usize];
        dot.hl=(dot.hl<<4)^PL[dot.rem as usize]^ bl[dot.lo as usize];
    }
}

fn polyval_generic_start(ctx:&mut polyval_generic, key:&[u8]) -> aes_gcmsiv_status_t {
    if key.len()!=16 {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE;
    }

    // Compute H * X^-128
  
    static XL: [u64;16] = [
        0x0000000000000000,0x0000000000000001,0x0000000000000003,0x0000000000000002,
        0x0000000000000007,0x0000000000000006,0x0000000000000004,0x0000000000000005,
        0x000000000000000e,0x000000000000000f,0x000000000000000d,0x000000000000000c,
        0x0000000000000009,0x0000000000000008,0x000000000000000a,0x000000000000000b,
    ];

    static XH: [u64;16] = [
        0x0000000000000000,0x9204000000000000,0xe608000000000000,0x740c000000000000,
        0x0e10000000000000,0x9c14000000000000,0xe818000000000000,0x7a1c000000000000,
        0x1c20000000000000,0x8e24000000000000,0xfa28000000000000,0x682c000000000000,
        0x1230000000000000,0x8034000000000000,0xf438000000000000,0x663c000000000000,
    ];

    let mut dot_ctx = dot_context{hl:0,hh:0,lo:0,hi:0,rem:0};
    dot(&mut dot_ctx, key, &XL,&XH);

    ctx.HL[0]=0; ctx.HH[0]=0;
    ctx.HL[1]=dot_ctx.hl; ctx.HH[1]=dot_ctx.hh;

    // Compute HX,HX^2,... 
    let mut hl=dot_ctx.hl;
    let mut hh=dot_ctx.hh;
    for i in (2..16).step_by(2) {
        dot_ctx.rem=(hh>>63)&0x01;
        hh=(hh<<1)^(hl>>63)^(PH[dot_ctx.rem as usize]);
        hl=(hl<<1)^PL[dot_ctx.rem as usize];

        ctx.HL[i]=hl; ctx.HH[i]=hh;

        for j in 1..i {
            ctx.HL[i+j]=hl^ctx.HL[j];
            ctx.HH[i+j]=hh^ctx.HH[j];
        }
    }

    for i in 0..16 {
        ctx.S[i]=0;
    }

    aes_gcmsiv_zeroize_struct(&mut dot_ctx);
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn polyval_generic_update(ctx:&mut polyval_generic, data:&[u8]) -> aes_gcmsiv_status_t {
    let mut idx=0;
    while idx+16 <= data.len() {
        for i in 0..16 {
            ctx.S[i]^=data[idx+i];
        }

        let mut dot_ctx=dot_context{hl:0,hh:0,lo:0,hi:0,rem:0};
        dot(&mut dot_ctx,&ctx.S,&ctx.HL,&ctx.HH);
        PUT_UINT64_LE(dot_ctx.hl,&mut ctx.S,0);
        PUT_UINT64_LE(dot_ctx.hh,&mut ctx.S,8);
        aes_gcmsiv_zeroize_struct(&mut dot_ctx);
        idx+=16;
    }

    if idx<data.len() {
        for i in 0..(data.len()-idx) {
            ctx.S[i]^=data[idx+i];
        }
        let mut dot_ctx=dot_context{hl:0,hh:0,lo:0,hi:0,rem:0};
        dot(&mut dot_ctx,&ctx.S,&ctx.HL,&ctx.HH);
        PUT_UINT64_LE(dot_ctx.hl,&mut ctx.S,0);
        PUT_UINT64_LE(dot_ctx.hh,&mut ctx.S,8);
        aes_gcmsiv_zeroize_struct(&mut dot_ctx);
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn polyval_generic_finish(ctx:&mut polyval_generic, nonce:&[u8], tag:&mut [u8]) -> aes_gcmsiv_status_t {
    if nonce.len()>16 {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
    }
    for i in 0..nonce.len() {
        tag[i]=ctx.S[i]^nonce[i];
    }
    for i in nonce.len()..16 {
        tag[i]=ctx.S[i];
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn polyval_start(ctx:&mut polyval, key:&[u8]) -> aes_gcmsiv_status_t {
    unsafe { polyval_generic_start(&mut ctx.storage.generic,key) }
}
fn polyval_update(ctx:&mut polyval, data:&[u8]) -> aes_gcmsiv_status_t {
    unsafe { polyval_generic_update(&mut ctx.storage.generic,data) }
}
fn polyval_finish(ctx:&mut polyval, nonce:&[u8], tag:&mut [u8]) -> aes_gcmsiv_status_t {
    unsafe { polyval_generic_finish(&mut ctx.storage.generic,nonce,tag) }
}

// ========================== GCM-SIV code ===================================

fn aes_gcmsiv_derive_keys(ctx: *mut aes, key_sz: usize, nonce: &[u8], key:&mut key_context) {
    let mut stack_input=[0u8;16];
    let mut stack_output=[0u8;16];

    key.auth_sz=KEY_AUTH_SIZE;
    key.enc_sz=key_sz;

    stack_input[4..4+12].copy_from_slice(nonce);

    // Derive auth key
    PUT_UINT32_LE(0,&mut stack_input,0);
    unsafe { aes_ecb_encrypt(&mut *ctx, &stack_input, &mut stack_output); }
    key.auth[0..8].copy_from_slice(&stack_output[0..8]);

    PUT_UINT32_LE(1,&mut stack_input,0);
    unsafe { aes_ecb_encrypt(&mut *ctx, &stack_input, &mut stack_output); }
    key.auth[8..16].copy_from_slice(&stack_output[0..8]);

    // Derive enc key
    PUT_UINT32_LE(2,&mut stack_input,0);
    unsafe { aes_ecb_encrypt(&mut *ctx,&stack_input,&mut stack_output); }
    key.enc[0..8].copy_from_slice(&stack_output[0..8]);

    PUT_UINT32_LE(3,&mut stack_input,0);
    unsafe { aes_ecb_encrypt(&mut *ctx,&stack_input,&mut stack_output); }
    key.enc[8..16].copy_from_slice(&stack_output[0..8]);

    if key_sz==32 {
        PUT_UINT32_LE(4,&mut stack_input,0);
        unsafe { aes_ecb_encrypt(&mut *ctx,&stack_input,&mut stack_output); }
        key.enc[16..24].copy_from_slice(&stack_output[0..8]);

        PUT_UINT32_LE(5,&mut stack_input,0);
        unsafe { aes_ecb_encrypt(&mut *ctx,&stack_input,&mut stack_output); }
        key.enc[24..32].copy_from_slice(&stack_output[0..8]);
    }

    aes_gcmsiv_zeroize(&mut stack_input);
    aes_gcmsiv_zeroize(&mut stack_output);
}

fn aes_gcmsiv_make_tag(key:&key_context, nonce:&[u8], plain:&[u8], aad:&[u8], tag:&mut [u8]) {
    let mut ctx = aes{has_hw:0,storage:aes_storage{generic:aes_generic::new()}};
    let mut polyval_ctx = polyval{has_hw:0,storage:polyval_storage{generic: polyval_generic{S:[0;16],HL:[0;16],HH:[0;16]}}};

    aes_init(&mut ctx);
    aes_set_key(&mut ctx,&key.enc[0..key.enc_sz]);
    polyval_init(&mut polyval_ctx);
    polyval_start(&mut polyval_ctx,&key.auth[0..key.auth_sz]);
    polyval_update(&mut polyval_ctx,aad);
    polyval_update(&mut polyval_ctx,plain);

    let mut length_block=[0u8;16];
    let aad_bit_sz=(aad.len() as u64)*8;
    PUT_UINT64_LE(aad_bit_sz,&mut length_block,0);
    let plain_bit_sz=(plain.len() as u64)*8;
    PUT_UINT64_LE(plain_bit_sz,&mut length_block,8);
    polyval_update(&mut polyval_ctx,&length_block);

    polyval_finish(&mut polyval_ctx,nonce,tag);
    tag[15]&=0x7f;

    let mut block=[0u8;16];
    block.copy_from_slice(tag);
    aes_ecb_encrypt(&mut ctx,&block,tag);
    aes_free(&mut ctx);
    polyval_free(&mut polyval_ctx);
}

fn aes_gcmsiv_aes_ctr(key:&[u8], key_sz:usize, tag:&[u8;16], input:&[u8], output:&mut [u8]) {
    let mut ctx = aes{has_hw:0,storage:aes_storage{generic:aes_generic::new()}};
    aes_init(&mut ctx);
    aes_set_key(&mut ctx, key);
    let mut nonce=[0u8;16];
    nonce.copy_from_slice(tag);
    nonce[15]|=0x80;
    aes_ctr(&mut ctx,&nonce,input,output);
    aes_free(&mut ctx);
}

fn aes_gcmsiv_check_tag(lhs:&[u8;16], rhs:&[u8;16]) -> aes_gcmsiv_status_t {
    let mut sum=0u8;
    for i in 0..16 {
        sum|=lhs[i]^rhs[i];
    }
    if sum==0 {
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
    } else {
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_TAG
    }
}

// Public functions
fn aes_gcmsiv_context_size() -> usize {
    std::mem::size_of::<aes_gcmsiv_ctx>()
}

fn aes_gcmsiv_init(ctx: &mut aes_gcmsiv_ctx) {
    if ctx as *mut _ as *mut u8 == ptr::null_mut() {
        return;
    }
    ctx.key_gen_ctx=std::ptr::null_mut();
    ctx.key_sz=0;
}

fn aes_gcmsiv_free(ctx:&mut aes_gcmsiv_ctx) {
    if ctx as *mut _ as *mut u8 == ptr::null_mut() {
        return;
    }
    if !ctx.key_gen_ctx.is_null() {
        unsafe {
            aes_free(&mut *ctx.key_gen_ctx);
            Box::from_raw(ctx.key_gen_ctx);
        }
    }
    let c=ctx as *mut aes_gcmsiv_ctx as *mut u8;
    let size=std::mem::size_of::<aes_gcmsiv_ctx>();
    for i in 0..size {
        unsafe { ptr::write_volatile(c.add(i),0) };
    }
}

fn aes_gcmsiv_set_key(ctx:&mut aes_gcmsiv_ctx,key:&[u8]) -> aes_gcmsiv_status_t {
    if ctx as *mut _ as *mut u8 == ptr::null_mut() || key.is_empty(){
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
    }
    if key.len()!=16 && key.len()!=32 {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE;
    }

    let mut key_gen_ctx=Box::new(aes{has_hw:0,storage:aes_storage{generic:aes_generic::new()}});
    aes_init(&mut key_gen_ctx);
    let res = aes_set_key(&mut key_gen_ctx,key);
    if res!=aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
        return res;
    }

    if !ctx.key_gen_ctx.is_null() {
        unsafe {
            aes_free(&mut *ctx.key_gen_ctx);
            Box::from_raw(ctx.key_gen_ctx);
        }
    }

    ctx.key_gen_ctx=Box::into_raw(key_gen_ctx);
    ctx.key_sz=key.len();
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn aes_gcmsiv_encrypt_size(plain_sz:usize,aad_sz:usize,cipher_sz:&mut usize)-> aes_gcmsiv_status_t {
    if cipher_sz as *mut usize == ptr::null_mut() {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
    }

    if plain_sz > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PLAINTEXT_SIZE;
    }

    if aad_sz > AES_GCMSIV_MAX_AAD_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_AAD_SIZE;
    }

    let needed_sz=plain_sz+AES_GCMSIV_TAG_SIZE;
    if needed_sz<plain_sz {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PLAINTEXT_SIZE;
    }

    *cipher_sz=needed_sz;
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn aes_gcmsiv_encrypt_with_tag(ctx:&mut aes_gcmsiv_ctx,
                               nonce:&[u8],
                               plain:&[u8],
                               aad:&[u8],
                               cipher:&mut [u8],
                               write_sz:&mut usize)-> aes_gcmsiv_status_t {
    if ctx as *mut _ as *mut u8 == ptr::null_mut() ||
       (nonce.is_empty() && AES_GCMSIV_NONCE_SIZE!=0) ||
       (plain.is_empty() && !plain.is_empty()) ||
       (aad.is_empty() && !aad.is_empty()) ||
       (cipher.is_empty() && (!plain.is_empty())) ||
       write_sz as *mut usize == ptr::null_mut(){
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
    }

    if nonce.len()!=AES_GCMSIV_NONCE_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
    }

    let mut needed_sz=0usize;
    let res=aes_gcmsiv_encrypt_size(plain.len(),aad.len(),&mut needed_sz);
    if res!=aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
        return res;
    }

    if cipher.len()<needed_sz {
        *write_sz=needed_sz;
        return aes_gcmsiv_status_t::AES_GCMSIV_UPDATE_OUTPUT_SIZE;
    }

    let key_gen_ctx= ctx.key_gen_ctx;
    if key_gen_ctx.is_null() {
        return aes_gcmsiv_status_t::AES_GCMSIV_FAILURE;
    }

    let mut key=key_context{auth:[0;16],auth_sz:0,enc:[0;32],enc_sz:0};
    aes_gcmsiv_derive_keys(key_gen_ctx, ctx.key_sz, nonce, &mut key);

    let tag_offset = plain.len();
    let (cipher_data, tag_buf) = cipher.split_at_mut(tag_offset);
    aes_gcmsiv_make_tag(&key, nonce, plain, aad, tag_buf);

    aes_gcmsiv_aes_ctr(&key.enc[0..key.enc_sz], key.enc_sz, tag_buf.try_into().unwrap(), plain, cipher_data);

    *write_sz=needed_sz;

    let mut zero_key=key;
    aes_gcmsiv_zeroize_struct(&mut zero_key);
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn aes_gcmsiv_decrypt_size(cipher_sz:usize, aad_sz:usize, plain_sz:&mut usize)-> aes_gcmsiv_status_t {
    if plain_sz as *mut usize == ptr::null_mut() {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
    }

    if cipher_sz< AES_GCMSIV_TAG_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_CIPHERTEXT_SIZE;
    }

    let needed_sz = cipher_sz - AES_GCMSIV_TAG_SIZE;

    if needed_sz > AES_GCMSIV_MAX_PLAINTEXT_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_CIPHERTEXT_SIZE;
    }

    if aad_sz > AES_GCMSIV_MAX_AAD_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_AAD_SIZE;
    }

    *plain_sz=needed_sz;
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn aes_gcmsiv_decrypt_and_check(ctx:&mut aes_gcmsiv_ctx,
                                nonce:&[u8],
                                cipher:&[u8],
                                aad:&[u8],
                                plain:&mut [u8],
                                write_sz:&mut usize)-> aes_gcmsiv_status_t {
    if ctx as *mut _ as *mut u8 == ptr::null_mut() ||
       (nonce.is_empty() && AES_GCMSIV_NONCE_SIZE!=0) ||
       (cipher.is_empty() && !cipher.is_empty()) ||
       (aad.is_empty() && !aad.is_empty()) ||
       (plain.is_empty() && !plain.is_empty()) ||
       write_sz as *mut usize == ptr::null_mut() {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS;
    }

    if nonce.len()!=AES_GCMSIV_NONCE_SIZE {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE;
    }

    let mut needed_sz=0usize;
    let res=aes_gcmsiv_decrypt_size(cipher.len(),aad.len(),&mut needed_sz);
    if res!=aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
        return res;
    }

    if plain.len()<needed_sz {
        *write_sz=needed_sz;
        return aes_gcmsiv_status_t::AES_GCMSIV_UPDATE_OUTPUT_SIZE;
    }

    let key_gen_ctx= ctx.key_gen_ctx;
    if key_gen_ctx.is_null() {
        return aes_gcmsiv_status_t::AES_GCMSIV_FAILURE;
    }

    let mut key=key_context{auth:[0;16],auth_sz:0,enc:[0;32],enc_sz:0};
    aes_gcmsiv_derive_keys(key_gen_ctx, ctx.key_sz, nonce, &mut key);

    let expected_tag=&cipher[cipher.len()-AES_GCMSIV_TAG_SIZE..];
    let ciphertext_data=&cipher[0..cipher.len()-AES_GCMSIV_TAG_SIZE];

    aes_gcmsiv_aes_ctr(&key.enc[0..key.enc_sz], key.enc_sz, expected_tag.try_into().unwrap(), ciphertext_data, plain);

    let mut computed_tag=[0u8;16];
    aes_gcmsiv_make_tag(&key,nonce,plain,aad,&mut computed_tag);

    let res=aes_gcmsiv_check_tag(&computed_tag, expected_tag.try_into().unwrap());
    if res!=aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS {
        for i in 0..needed_sz {
            plain[i]=0;
        }
        *write_sz=0;
        aes_gcmsiv_zeroize_struct(&mut key);
        return res;
    }

    *write_sz=needed_sz;
    aes_gcmsiv_zeroize_struct(&mut key);
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS
}

fn aes_gcmsiv_get_status_code_msg(status:aes_gcmsiv_status_t)-> &'static str {
    match status {
        aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS=>"Success",
        aes_gcmsiv_status_t::AES_GCMSIV_FAILURE=>"Failure",
        aes_gcmsiv_status_t::AES_GCMSIV_OUT_OF_MEMORY=>"Out of memory",
        aes_gcmsiv_status_t::AES_GCMSIV_UPDATE_OUTPUT_SIZE=>"Update output size",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS=>"Invalid parameters",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE=>"Unsupported key size",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE=>"Invalid nonce size",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PLAINTEXT_SIZE=>"Invalid plaintext size",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_AAD_SIZE=>"Invalid additional authenticated data size",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_CIPHERTEXT_SIZE=>"Invalid ciphertext size",
        aes_gcmsiv_status_t::AES_GCMSIV_INVALID_TAG=>"Invalid tag",
    }
}

// X86-64 Optimizations: 
//

#![feature(stdsimd, target_feature_11)]
use std::arch::x86_64::*;
use std::mem;
use std::ptr;
use std::slice;

const AES_BLOCK_SIZE: usize = 16;
const POLYVAL_SIZE: usize = 16;

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

#[inline]
unsafe fn aes_gcmsiv_zeroize(ptr: *mut u8, len: usize) {
    for i in 0..len {
        ptr::write_volatile(ptr.add(i), 0);
    }
}

#[inline]
unsafe fn XOR(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a,b)
}

#[inline]
unsafe fn AESKEYGENASSIST(a: __m128i, r: i32) -> __m128i {
    _mm_aeskeygenassist_si128(a, r)
}

#[inline]
unsafe fn SPLIT(a: __m128i, b: i32) -> __m128i {
    _mm_shuffle_epi32(a, b)
}

#[inline]
unsafe fn SHIFT_ADD(a: __m128i) -> __m128i {
    let tmp = _mm_slli_si128(a,4);
    XOR(a, tmp)
}

#[inline]
unsafe fn TSHIFT_ADD(a: __m128i) -> __m128i {
    SHIFT_ADD(SHIFT_ADD(SHIFT_ADD(a)))
}

#[inline]
unsafe fn KEY_EXP_HELPER(k0: __m128i, k1: __m128i, r: i32, s: i32) -> __m128i {
    let t = AESKEYGENASSIST(k1, r);
    XOR(TSHIFT_ADD(k0), SPLIT(t, s))
}

#[inline]
unsafe fn KEY_EXP_128(key: &mut [__m128i], i: usize, r: i32) -> __m128i {
    KEY_EXP_HELPER(key[i], key[i], r, 0xff)
}

#[inline]
unsafe fn KEY_EXP_256_1(key: &mut [__m128i], i: usize, r: i32) -> __m128i {
    KEY_EXP_HELPER(key[i], key[i+1], r, 0xff)
}

#[inline]
unsafe fn KEY_EXP_256_2(key: &mut [__m128i], i: usize) -> __m128i {
    KEY_EXP_HELPER(key[i], key[i+1], 0x00, 0xaa)
}

#[repr(C)]
struct aes_x86_64 {
    key: [__m128i; 15],
    key_sz: usize,
}

#[target_feature(enable = "aes")]
unsafe fn loadu(data: *const u8) -> __m128i {
    _mm_loadu_si128(data as *const __m128i)
}

#[target_feature(enable = "aes")]
unsafe fn storeu(data: *mut u8, reg: __m128i) {
    _mm_storeu_si128(data as *mut __m128i, reg);
}

#[no_mangle]
#[target_feature(enable = "aes")]
unsafe extern "C" fn aes_x86_64_init(ctx: *mut aes_x86_64) {
    if ctx.is_null() {
        return;
    }
    ptr::write_bytes(ctx, 0, 1);
}

#[no_mangle]
#[target_feature(enable = "aes")]
unsafe extern "C" fn aes_x86_64_free(ctx: *mut aes_x86_64) {
    if ctx.is_null() {
        return;
    }
    aes_gcmsiv_zeroize(ctx as *mut u8, mem::size_of::<aes_x86_64>());
}

#[no_mangle]
#[target_feature(enable = "aes")]
unsafe extern "C" fn aes_x86_64_set_key(ctx: *mut aes_x86_64, key_ptr: *const u8, key_sz: usize) -> i32 {
    if ctx.is_null() || (key_ptr.is_null() && key_sz !=0 ) {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS as i32;
    }
    if key_sz!=16 && key_sz!=32 {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE as i32;
    }

    let ctx_ref = &mut *ctx;
    ctx_ref.key_sz=key_sz;
    let keyslice = slice::from_raw_parts(key_ptr,key_sz);

    ctx_ref.key[0]= loadu(keyslice.as_ptr());

    if key_sz==16 {
        // AES-128 key expansion
        ctx_ref.key[5] = KEY_EXP_128(&mut ctx_ref.key,0,0x01);
        ctx_ref.key[6] = KEY_EXP_128(&mut ctx_ref.key,5,0x02);
        ctx_ref.key[7] = KEY_EXP_128(&mut ctx_ref.key,6,0x04);
        ctx_ref.key[8] = KEY_EXP_128(&mut ctx_ref.key,7,0x08);
        ctx_ref.key[9] = KEY_EXP_128(&mut ctx_ref.key,8,0x10);
        ctx_ref.key[10] = KEY_EXP_128(&mut ctx_ref.key,9,0x20);
        ctx_ref.key[11] = KEY_EXP_128(&mut ctx_ref.key,10,0x40);
        ctx_ref.key[12] = KEY_EXP_128(&mut ctx_ref.key,11,0x80);
        ctx_ref.key[13] = KEY_EXP_128(&mut ctx_ref.key,12,0x1b);
        ctx_ref.key[14] = KEY_EXP_128(&mut ctx_ref.key,13,0x36);
    } else {
        // AES-256
        ctx_ref.key[1] = loadu(keyslice.as_ptr().add(16));
        ctx_ref.key[2] = KEY_EXP_256_1(&mut ctx_ref.key,0,0x01);
        ctx_ref.key[3] = KEY_EXP_256_2(&mut ctx_ref.key,1);
        ctx_ref.key[4] = KEY_EXP_256_1(&mut ctx_ref.key,2,0x02);
        ctx_ref.key[5] = KEY_EXP_256_2(&mut ctx_ref.key,3);
        ctx_ref.key[6] = KEY_EXP_256_1(&mut ctx_ref.key,4,0x04);
        ctx_ref.key[7] = KEY_EXP_256_2(&mut ctx_ref.key,5);
        ctx_ref.key[8] = KEY_EXP_256_1(&mut ctx_ref.key,6,0x08);
        ctx_ref.key[9] = KEY_EXP_256_2(&mut ctx_ref.key,7);
        ctx_ref.key[10] = KEY_EXP_256_1(&mut ctx_ref.key,8,0x10);
        ctx_ref.key[11] = KEY_EXP_256_2(&mut ctx_ref.key,9);
        ctx_ref.key[12] = KEY_EXP_256_1(&mut ctx_ref.key,10,0x20);
        ctx_ref.key[13] = KEY_EXP_256_2(&mut ctx_ref.key,11);
        ctx_ref.key[14] = KEY_EXP_256_1(&mut ctx_ref.key,12,0x40);
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS as i32
}

#[target_feature(enable = "aes")]
unsafe fn aes_encrypt(ctx: &aes_x86_64, mut block: __m128i) -> __m128i {
    block = XOR(block, ctx.key[0]);

    match ctx.key_sz {
        16 => {
            // AES-128 = 10 rounds total
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
            // AES-256 = 14 rounds total
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

#[no_mangle]
#[target_feature(enable = "aes")]
unsafe extern "C" fn aes_x86_64_ecb_encrypt(ctx: *const aes_x86_64,
                                            input: *const u8,
                                            output: *mut u8) -> i32 {
    if ctx.is_null() || input.is_null() || output.is_null() {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS as i32;
    }

    let block = loadu(input);
    let c = aes_encrypt(&*ctx, block);
    storeu(output,c);
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS as i32
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn aes_encrypt_x4(ctx: &aes_x86_64, counter: &[__m128i;4], stream: &mut [__m128i;4]) {
    for i in 0..4 {
        stream[i]=XOR(counter[i],ctx.key[0]);
    }
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

#[inline]
#[target_feature(enable = "aes")]
unsafe fn aes_encrypt_x2(ctx:&aes_x86_64, plain:&[__m128i;2], cipher:&mut[__m128i;2]) {
    for i in 0..2 {
        cipher[i]=XOR(plain[i],ctx.key[0]);
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

#[inline]
#[target_feature(enable = "aes")]
unsafe fn aes_encrypt_x3(ctx:&aes_x86_64, plain:&[__m128i;3], cipher:&mut[__m128i;3]) {
    for i in 0..3 {
        cipher[i]=XOR(plain[i],ctx.key[0]);
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


#[no_mangle]
#[target_feature(enable = "aes")]
unsafe extern "C" fn aes_x86_64_ctr(ctx: *const aes_x86_64,
                                    nonce: *const u8,
                                    input: *const u8,
                                    input_sz: usize,
                                    output: *mut u8) -> i32 {
    if ctx.is_null() || ((input.is_null()||output.is_null()) && input_sz!=0) {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS as i32;
    }

    let ctx_ref = &*ctx;
    let one = _mm_set_epi32(0,0,0,1);
    let mut counter = [ _mm_setzero_si128();4];
    counter[0]=loadu(nonce);
    counter[1]=_mm_add_epi32(counter[0],one);
    counter[2]=_mm_add_epi32(counter[1],one);
    counter[3]=_mm_add_epi32(counter[2],one);

    let mut inptr = input;
    let mut outptr = output;
    let mut remain = input_sz;

    while remain >= 4*AES_BLOCK_SIZE {
        let mut stream = [_mm_setzero_si128();4];
        aes_encrypt_x4(ctx_ref,&counter,&mut stream);

        counter[0] = _mm_add_epi32(counter[3], one);
        counter[1] = _mm_add_epi32(counter[0], one);
        counter[2] = _mm_add_epi32(counter[1], one);
        counter[3] = _mm_add_epi32(counter[2], one);

        let inblk0 = loadu(inptr);
        let inblk1 = loadu(inptr.add(16));
        let inblk2 = loadu(inptr.add(32));
        let inblk3 = loadu(inptr.add(48));

        let c0=XOR(inblk0,stream[0]);
        let c1=XOR(inblk1,stream[1]);
        let c2=XOR(inblk2,stream[2]);
        let c3=XOR(inblk3,stream[3]);

        storeu(outptr, c0);
        storeu(outptr.add(16), c1);
        storeu(outptr.add(32), c2);
        storeu(outptr.add(48), c3);

        inptr = inptr.add(64);
        outptr = outptr.add(64);
        remain -=64;
    }

    // Handle remainder fully:
    if remain > 0 {
        let mut tmp=[0u8;16];
        let blocks = remain / AES_BLOCK_SIZE;
        let leftover = remain % AES_BLOCK_SIZE;

        match blocks {
            0 => {
                // No full block, just a partial block
                let mut single=[XOR(counter[0], ctx_ref.key[0])];
                if ctx_ref.key_sz==32 {
                    for r in 1..5 {
                        single[0]=_mm_aesenc_si128(single[0], ctx_ref.key[r]);
                    }
                }
                for r in 5..14 {
                    single[0]=_mm_aesenc_si128(single[0], ctx_ref.key[r]);
                }
                single[0]=_mm_aesenclast_si128(single[0], ctx_ref.key[14]);

                ptr::copy_nonoverlapping(inptr, tmp.as_mut_ptr(), leftover);
                for i in leftover..16 {
                    tmp[i]=0;
                }
                let inblk=loadu(tmp.as_ptr());
                let outblk=XOR(inblk,single[0]);
                storeu(tmp.as_mut_ptr(),outblk);
                ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
            }
            1 => {
                // 1 full block and maybe partial remainder
                let plains=[counter[0], counter[1]];
                let mut ciphers=[_mm_setzero_si128();2];
                aes_encrypt_x2(ctx_ref,&plains,&mut ciphers);

                // First full block
                let inblk0=loadu(inptr);
                let c0=XOR(inblk0,ciphers[0]);
                storeu(outptr,c0);
                inptr=inptr.add(16);
                outptr=outptr.add(16);

                if leftover>0 {
                    ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
                    for i in leftover..16 {
                        tmp[i]=0;
                    }
                    let inblk=loadu(tmp.as_ptr());
                    let outblk=XOR(inblk,ciphers[1]);
                    storeu(tmp.as_mut_ptr(),outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            2 => {
                // 2 full blocks and maybe partial remainder
                let plains=[counter[0], counter[1], counter[2]];
                let mut ciphers=[_mm_setzero_si128();3];
                aes_encrypt_x3(ctx_ref,&plains,&mut ciphers);

                let inblk0=loadu(inptr);
                let inblk1=loadu(inptr.add(16));

                let c0=XOR(inblk0,ciphers[0]);
                let c1=XOR(inblk1,ciphers[1]);
                storeu(outptr,c0);
                storeu(outptr.add(16),c1);

                inptr=inptr.add(32);
                outptr=outptr.add(32);

                if leftover>0 {
                    ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
                    for i in leftover..16 {
                        tmp[i]=0;
                    }
                    let inblk=loadu(tmp.as_ptr());
                    let outblk=XOR(inblk,ciphers[2]);
                    storeu(tmp.as_mut_ptr(),outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            3 => {
                // 3 full blocks and maybe partial remainder = actually 4 blocks needed for x4
                let mut stream=[_mm_setzero_si128();4];
                aes_encrypt_x4(ctx_ref,&counter,&mut stream);

                let inblk0=loadu(inptr);
                let inblk1=loadu(inptr.add(16));
                let inblk2=loadu(inptr.add(32));

                let c0=XOR(inblk0,stream[0]);
                let c1=XOR(inblk1,stream[1]);
                let c2=XOR(inblk2,stream[2]);

                storeu(outptr,c0);
                storeu(outptr.add(16),c1);
                storeu(outptr.add(32),c2);

                inptr=inptr.add(48);
                outptr=outptr.add(48);

                if leftover>0 {
                    ptr::copy_nonoverlapping(inptr,tmp.as_mut_ptr(), leftover);
                    for i in leftover..16 {
                        tmp[i]=0;
                    }
                    let inblk=loadu(tmp.as_ptr());
                    let outblk=XOR(inblk,stream[3]);
                    storeu(tmp.as_mut_ptr(),outblk);
                    ptr::copy_nonoverlapping(tmp.as_ptr(), outptr, leftover);
                }
            }
            _=>{}
        }
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS as i32
}

// POLYVAL x86_64 optimization

#[repr(C)]
struct polyval_x86_64 {
    s: __m128i,
    h_table: [__m128i;8],
}

#[inline]
unsafe fn CLMUL(a: __m128i, b: __m128i, c: i32) -> __m128i {
    _mm_clmulepi64_si128(a, b, c)
}

#[inline]
unsafe fn SWAP(a: __m128i) -> __m128i {
    _mm_shuffle_epi32(a,0x4e)
}

#[inline]
unsafe fn mult(a: __m128i, b: __m128i, c0: &mut __m128i, c1:&mut __m128i, c2:&mut __m128i) {
    *c0 = CLMUL(a,b,0x00);
    *c2 = CLMUL(a,b,0x11);
    *c1 = XOR(CLMUL(a,b,0x01), CLMUL(a,b,0x10));
}

#[inline]
unsafe fn add_mult(a: __m128i, b: __m128i, c0:&mut __m128i, c1:&mut __m128i, c2:&mut __m128i) {
    *c0 = XOR(*c0, CLMUL(a,b,0x00));
    *c2 = XOR(*c2, CLMUL(a,b,0x11));
    *c1 = XOR(*c1, XOR(CLMUL(a,b,0x01), CLMUL(a,b,0x10)));
}

#[inline]
unsafe fn mult_inv_x64(p: __m128i) -> __m128i {
    // Derived from original polynomial logic in the C code
    const POLY: __m128i = __m128i::from_ne_bytes([0x00,0x00,0x00,0x00,0xc2,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00]);
    let q = SWAP(p);
    let r = CLMUL(p,POLY,0x00);
    XOR(q,r)
}

#[inline]
unsafe fn mult_inv_x128(p0: __m128i, p1: __m128i, p2: __m128i) -> __m128i {
    let q = XOR(p0, _mm_slli_si128(p1,8));
    let r = XOR(p2,_mm_srli_si128(p1,8));
    let s = mult_inv_x64(q);
    let t = mult_inv_x64(s);
    XOR(r,t)
}

#[inline]
unsafe fn dot(a: __m128i, b: __m128i) -> __m128i {
    let mut c0=_mm_setzero_si128();
    let mut c1=_mm_setzero_si128();
    let mut c2=_mm_setzero_si128();
    mult(a,b,&mut c0,&mut c1,&mut c2);
    mult_inv_x128(c0,c1,c2)
}

#[inline]
#[target_feature(enable = "pclmul")]
unsafe fn polyval_x86_64_process_tables(h_table:&[__m128i;8],
                                        mut s: __m128i,
                                        mut data: *const u8,
                                        mut data_sz: usize) -> __m128i {
    let mut tmp=[0u8;16];

    // Process 8-block chunks
    while data_sz >= 8*POLYVAL_SIZE {
        // load 8 blocks: D0..D7
        // order: last loaded is D7, as per original code pattern
        // It performs:
        // d = D7
        // mult(d,h_table[0])
        // d = D6 add_mult ...
        // ...
        // finally add s

        let blocks = data as *const __m128i;
        let D0 = _mm_loadu_si128(blocks);
        let D1 = _mm_loadu_si128(blocks.add(1));
        let D2 = _mm_loadu_si128(blocks.add(2));
        let D3 = _mm_loadu_si128(blocks.add(3));
        let D4 = _mm_loadu_si128(blocks.add(4));
        let D5 = _mm_loadu_si128(blocks.add(5));
        let D6 = _mm_loadu_si128(blocks.add(6));
        let D7 = _mm_loadu_si128(blocks.add(7));

        let mut s0; let mut s1; let mut s2;

        // d0 = D7 * H0
        mult(D7,h_table[0], &mut s0,&mut s1,&mut s2);
        // d1 = d0 + D6 * H1
        add_mult(D6,h_table[1], &mut s0,&mut s1,&mut s2);
        // d2 = d1 + D5 * H2
        add_mult(D5,h_table[2], &mut s0,&mut s1,&mut s2);
        // d3 = d2 + D4 * H3
        add_mult(D4,h_table[3], &mut s0,&mut s1,&mut s2);
        // d4 = d3 + D3 * H4
        add_mult(D3,h_table[4], &mut s0,&mut s1,&mut s2);
        // d5 = d4 + D2 * H5
        add_mult(D2,h_table[5], &mut s0,&mut s1,&mut s2);
        // d6 = d5 + D1 * H6
        add_mult(D1,h_table[6], &mut s0,&mut s1,&mut s2);

        // d7 = d6 + (D0 + s) * H7
        let D0_s=XOR(s,D0);
        add_mult(D0_s,h_table[7], &mut s0,&mut s1,&mut s2);

        // s = d7 * X^-128
        s=mult_inv_x128(s0,s1,s2);

        data = data.add(8*POLYVAL_SIZE);
        data_sz -= 8*POLYVAL_SIZE;
    }

    // process full blocks <8
    let blocks = data_sz / POLYVAL_SIZE;
    if blocks>0 {
        let bptr = data as *const __m128i;
        // Similar logic as above but for fewer blocks:
        // For n blocks:
        // d start from last block * H0
        // then add_mult with previous block * H1, ...
        // finally add_mult with (s + first_block)*H(n-1)

        // Example if blocks=1:
        // d = (s+ D0)*H0
        // s = d * X^-128

        if blocks ==1 {
            let D0 = _mm_loadu_si128(bptr);
            let D0_s = XOR(s,D0);
            let mut s0; let mut s1; let mut s2;
            mult(D0_s,h_table[0], &mut s0,&mut s1,&mut s2);
            s=mult_inv_x128(s0,s1,s2);
        } else {
            // blocks >1:
            // general form:
            // d0 = lastblock * H0
            // d1 = d0 + second_last_block * H1
            // ...
            // dn = d(n-1) + (s+first_block)*H(n-1)
            // s = dn * X^-128
            let last_block = _mm_loadu_si128(bptr.add(blocks-1));
            let mut s0; let mut s1; let mut s2;
            mult(last_block,h_table[0], &mut s0,&mut s1,&mut s2);

            for i in 1..(blocks-1) {
                let blk = _mm_loadu_si128(bptr.add(blocks-1-i));
                add_mult(blk,h_table[i], &mut s0,&mut s1,&mut s2);
            }
            let first_blk = _mm_loadu_si128(bptr);
            let first_s=XOR(s,first_blk);
            add_mult(first_s,h_table[blocks-1], &mut s0,&mut s1,&mut s2);
            s=mult_inv_x128(s0,s1,s2);
        }

        data = data.add(blocks*POLYVAL_SIZE);
        data_sz -= blocks*POLYVAL_SIZE;
    }

    // partial leftover
    if data_sz >0 {
        ptr::copy_nonoverlapping(data, tmp.as_mut_ptr(), data_sz);
        for i in data_sz..16 {
            tmp[i]=0;
        }
        let d=XOR(s, loadu(tmp.as_ptr()));
        s=dot(d,h_table[0]);
    }

    s
}

#[no_mangle]
#[target_feature(enable = "pclmul")]
unsafe extern "C" fn polyval_x86_64_start(ctx: *mut polyval_x86_64, key:*const u8, key_sz: usize) -> i32 {
    if ctx.is_null() || (key.is_null() && key_sz!=0) {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS as i32;
    }
    if key_sz!=16 {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_KEY_SIZE as i32;
    }

    let ctx_ref = &mut *ctx;
    ctx_ref.s = _mm_setzero_si128();

    let h = loadu(key);
    ctx_ref.h_table[0]=h;
    for i in 1..8 {
        ctx_ref.h_table[i]= dot(ctx_ref.h_table[0],ctx_ref.h_table[i-1]);
    }

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS as i32
}

#[no_mangle]
#[target_feature(enable = "pclmul")]
unsafe extern "C" fn polyval_x86_64_update(ctx: *mut polyval_x86_64, data:*const u8, data_sz:usize) -> i32 {
    if ctx.is_null() || (data.is_null() && data_sz!=0) {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS as i32;
    }

    let ctx_ref = &mut *ctx;
    ctx_ref.s = polyval_x86_64_process_tables(&ctx_ref.h_table,ctx_ref.s,data,data_sz);
    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS as i32
}

#[no_mangle]
#[target_feature(enable = "pclmul")]
unsafe extern "C" fn polyval_x86_64_finish(ctx:*mut polyval_x86_64,
                                           nonce:*const u8,
                                           nonce_sz:usize,
                                           tag:*mut u8) -> i32 {
    if ctx.is_null() || (nonce.is_null() && nonce_sz!=0) || tag.is_null() {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_PARAMETERS as i32;
    }
    if nonce_sz>16 {
        return aes_gcmsiv_status_t::AES_GCMSIV_INVALID_NONCE_SIZE as i32;
    }

    let ctx_ref = &mut *ctx;
    let mut tmp=[0u8;16];
    ptr::copy_nonoverlapping(nonce, tmp.as_mut_ptr(), nonce_sz);
    for i in nonce_sz..16 {
        tmp[i]=0;
    }

    let n = loadu(tmp.as_ptr());
    let out = XOR(n, ctx_ref.s);
    storeu(tag, out);

    aes_gcmsiv_status_t::AES_GCMSIV_SUCCESS as i32
}


// aes-gcm-siv.rs

#ifndef QWEN36_DEQUANT_CUH
#define QWEN36_DEQUANT_CUH

/* Exact ggml-compatible dequantization for the four quant types present in the
 * Qwen3.6-35B-A3B MXFP4 GGUF.  MXFP4 is the headline routed-expert format; the
 * attention/shared/embed/output tensors are Q8_0, and a handful of dynamically
 * up-quantized expert tensors are Q5_K / Q6_K.
 *
 * Every constant here was lifted verbatim from llama.cpp's ggml-quants.c /
 * ggml-common.h / ggml-impl.h so the numbers match bit-for-bit.  The functions
 * are host+device dual-compilable (Q36_HD) so the math can be unit-tested on a
 * CPU before it ever touches a GPU.
 *
 * --- Attribution & License ---
 * This file is a mixed-license source file:
 *
 * 1. The custom integration scaffolding, CUDA/C++ dual-compilation wrappers,
 *    and target framework bindings are licensed under the GNU Affero General
 *    Public License version 3 (AGPL-3.0).
 *
 *    Copyright (C) 2026 Ambud Sharma
 *
 * 2. The quantization constants, block layouts, and dequantization math formulas
 *    are adapted from llama.cpp / ggml (licensed under the MIT License):
 *
 *    Copyright (c) 2023-2026 Georgi Gerganov and llama.cpp contributors
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all
 *    copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
  #define Q36_HD __host__ __device__ __forceinline__
#else
  #define Q36_HD static inline
#endif

#include "q36.h"   /* block_q8_0, block_mxfp4, block_q5_K, block_q6_K, QK_K */

/* ---- fp16 -> fp32 (IEEE half, matches ggml scalar path) ----------------- */
Q36_HD float q36_fp16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

/* ---- MXFP4: E8M0 shared scale (halved) + E2M1 code table ----------------- */
/* kvalues_mxfp4 from ggml-common.h; note it is the *halved* convention paired
 * with GGML_E8M0_TO_FP32_HALF, so the product reproduces the true weight. */
#if defined(__CUDACC__)
__device__ __constant__ static const int8_t q36_mxfp4_k_dev[16] =
    { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
#endif
static const int8_t q36_mxfp4_k_host[16] =
    { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
Q36_HD const int8_t *q36_mxfp4_table(void) {
#if defined(__CUDA_ARCH__)
    return q36_mxfp4_k_dev;   /* device-side constant memory */
#else
    return q36_mxfp4_k_host;  /* host (also the CPU unit-test path) */
#endif
}

/* GGML_E8M0_TO_FP32_HALF: 0.5 * 2^(e-127), with denormal handling for e<2. */
Q36_HD float q36_e8m0_half(uint8_t e) {
    uint32_t bits;
    if (e < 2) bits = 0x00200000u << e;         /* 2^-128, 2^-127 */
    else       bits = (uint32_t)(e - 1) << 23;   /* 2^(e-128) */
    float f; memcpy(&f, &bits, 4); return f;
}

/* Dequantize one MXFP4 block (32 values) into out[0..31]. */
Q36_HD void q36_deq_mxfp4(const block_mxfp4 *b, float *out) {
    const float d = q36_e8m0_half(b->e);
    const int8_t *k = q36_mxfp4_table();
    for (int j = 0; j < QK_MXFP4/2; ++j) {
        out[j]              = k[b->qs[j] & 0x0F] * d;
        out[j + QK_MXFP4/2] = k[b->qs[j] >> 4]   * d;
    }
}

/* Dequantize one Q8_0 block (32 values). */
Q36_HD void q36_deq_q8_0(const block_q8_0 *b, float *out) {
    const float d = q36_fp16(b->d);
    for (int j = 0; j < 32; ++j) out[j] = d * b->qs[j];
}

/* 6-bit packed scale/min extraction (ggml get_scale_min_k4). */
Q36_HD void q36_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4)  | ((q[j-0] >> 6) << 4); }
}

/* Dequantize one Q5_K super-block (256 values). */
Q36_HD void q36_deq_q5_K(const block_q5_K *x, float *y) {
    const uint8_t *ql = x->qs;
    const uint8_t *qh = x->qh;
    const float d   = q36_fp16(x->d);
    const float min = q36_fp16(x->dmin);
    int is = 0; uint8_t sc, m, u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        q36_scale_min_k4(is+0, x->scales, &sc, &m); float d1 = d*sc, m1 = min*m;
        q36_scale_min_k4(is+1, x->scales, &sc, &m); float d2 = d*sc, m2 = min*m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

/* Dequantize one Q6_K super-block (256 values). */
Q36_HD void q36_deq_q6_K(const block_q6_K *x, float *y) {
    const float d = q36_fp16(x->d);
    const uint8_t *ql = x->ql;
    const uint8_t *qh = x->qh;
    const int8_t  *sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l/16;
            int8_t q1 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l+ 0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l+32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l+ 0] = d * sc[is+0] * q1;
            y[l+32] = d * sc[is+2] * q2;
            y[l+64] = d * sc[is+4] * q3;
            y[l+96] = d * sc[is+6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

/* Generic block dispatch: dequantize one contiguous row of `n` values of the
 * given ggml type into `out`. Used by the reference path and unit tests. */
Q36_HD void q36_deq_row(uint32_t type, const void *blocks, int n, float *out) {
    switch (type) {
        case Q36_GGML_Q8_0: { const block_q8_0 *b = (const block_q8_0*)blocks;
            for (int i = 0; i < n/32; i++) q36_deq_q8_0(b+i, out+i*32); break; }
        case Q36_GGML_MXFP4: { const block_mxfp4 *b = (const block_mxfp4*)blocks;
            for (int i = 0; i < n/32; i++) q36_deq_mxfp4(b+i, out+i*32); break; }
        case Q36_GGML_Q5_K: { const block_q5_K *b = (const block_q5_K*)blocks;
            for (int i = 0; i < n/QK_K; i++) q36_deq_q5_K(b+i, out+i*QK_K); break; }
        case Q36_GGML_Q6_K: { const block_q6_K *b = (const block_q6_K*)blocks;
            for (int i = 0; i < n/QK_K; i++) q36_deq_q6_K(b+i, out+i*QK_K); break; }
        case Q36_GGML_F32: memcpy(out, blocks, (size_t)n*4); break;
        default: for (int i = 0; i < n; i++) out[i] = 0.0f;
    }
}

#endif /* QWEN36_DEQUANT_CUH */

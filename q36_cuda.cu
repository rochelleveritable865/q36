/* This file is part of q36, a dedicated CUDA inference engine for
 * Qwen3.6-35B-A3B.
 *
 * Copyright (C) 2026 Ambud Sharma
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public
 * License along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

/* Qwen3.6-35B-A3B CUDA hot path.
 *
 * Design: batch=1 decode is memory-bandwidth bound (2.544 GiB active/token),
 * so every hot kernel is structured to stream each quantized weight byte from
 * global memory exactly once and dequantize in-register.  There is no cuBLAS
 * dependency: the weights never exist in fp16/fp32 in memory, only the packed
 * quant blocks do, and we dequantize on the fly inside the dot product.  That
 * is the whole point of a model-specific engine -- the quant format is part of
 * the kernel, not a pre-pass.
 *
 * This file holds the fully-determined kernels (dequant-matvec, RMSNorm, RoPE,
 * sampling).  Attention, the gated-DeltaNet SSM scan, and MoE routing live in
 * their own files so each can be validated against a reference independently.
 */

#include "q36_dequant.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define Q36_CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    abort(); } } while (0)

#define WARP 32

/* ---- per-warp dequant dot product --------------------------------------
 * Each warp computes one output element y[row] = <W_row, x>.  W_row is stored
 * as contiguous quant blocks (ggml row-major); the 32 lanes stride over the
 * blocks so consecutive lanes read consecutive blocks -> coalesced loads.  The
 * activation x is f32 in shared/global memory (K is small: 2048 or 4096).     */

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP/2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

/* Dot of one Q8_0 row (K elems -> K/32 blocks) with x[K]. */
__device__ float dot_q8_0_row(const block_q8_0 *row, const float *x, int K, int lane) {
    float acc = 0.f;
    int nb = K / 32;
    for (int b = lane; b < nb; b += WARP) {
        const block_q8_0 *blk = row + b;
        float d = q36_fp16(blk->d);
        const float *xb = x + b*32;
        #pragma unroll
        for (int j = 0; j < 32; j++) acc += d * blk->qs[j] * xb[j];
    }
    return warp_reduce_sum(acc);
}

/* Dot of one MXFP4 row with x[K]. */
__device__ float dot_mxfp4_row(const block_mxfp4 *row, const float *x, int K, int lane) {
    const int8_t *kt = q36_mxfp4_table();
    float acc = 0.f;
    int nb = K / 32;
    for (int b = lane; b < nb; b += WARP) {
        const block_mxfp4 *blk = row + b;
        float d = q36_e8m0_half(blk->e);
        const float *xb = x + b*32;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            acc += d * kt[blk->qs[j] & 0xF] * xb[j];
            acc += d * kt[blk->qs[j] >> 4]  * xb[j + 16];
        }
    }
    return warp_reduce_sum(acc);
}

/* Dot of one Q5_K row (K/256 super-blocks) with x[K]. Dequant then MAC. */
__device__ float dot_q5_K_row(const block_q5_K *row, const float *x, int K, int lane) {
    float acc = 0.f; int nb = K / QK_K;
    for (int sb = lane; sb < nb; sb += WARP) {
        float tmp[QK_K];
        q36_deq_q5_K(row + sb, tmp);           /* 256 vals; per-lane, no shared */
        const float *xb = x + sb*QK_K;
        #pragma unroll
        for (int j = 0; j < QK_K; j++) acc += tmp[j] * xb[j];
    }
    return warp_reduce_sum(acc);
}
__device__ float dot_q6_K_row(const block_q6_K *row, const float *x, int K, int lane) {
    float acc = 0.f; int nb = K / QK_K;
    for (int sb = lane; sb < nb; sb += WARP) {
        float tmp[QK_K];
        q36_deq_q6_K(row + sb, tmp);
        const float *xb = x + sb*QK_K;
        #pragma unroll
        for (int j = 0; j < QK_K; j++) acc += tmp[j] * xb[j];
    }
    return warp_reduce_sum(acc);
}

/* Generic matvec y[M] = W[M,K] @ x[K], one warp per row, dispatched on quant
 * type.  bytes(W_row) is read once.  Grid: M warps.  x lives in global memory
 * (callers may stage it in shared for the hottest projections). */
__global__ void q36_matvec(uint32_t type, const void *W, const float *x,
                           float *y, int M, int K, uint64_t row_bytes) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane = threadIdx.x & (WARP-1);
    if (warp >= M) return;
    const uint8_t *row = (const uint8_t *)W + (uint64_t)warp * row_bytes;
    float r;
    switch (type) {
        case Q36_GGML_Q8_0:  r = dot_q8_0_row((const block_q8_0*)row, x, K, lane); break;
        case Q36_GGML_MXFP4: r = dot_mxfp4_row((const block_mxfp4*)row, x, K, lane); break;
        case Q36_GGML_Q5_K:  r = dot_q5_K_row((const block_q5_K*)row, x, K, lane); break;
        case Q36_GGML_Q6_K:  r = dot_q6_K_row((const block_q6_K*)row, x, K, lane); break;
        default: r = 0.f;
    }
    if (lane == 0) y[warp] = r;
}

/* ---- RMSNorm: y = x / rms(x) * w  (w is F32) --------------------------- */
__global__ void q36_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    /* single block, blockDim.x threads reduce over n */
    __shared__ float ssum;
    float local = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i]*x[i];
    local = warp_reduce_sum(local);
    __shared__ float partials[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) partials[wid] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0; int nw = (blockDim.x+31)/32;
        for (int i = 0; i < nw; i++) s += partials[i];
        ssum = rsqrtf(s/n + eps);
    }
    __syncthreads();
    float scale = ssum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i]*scale*w[i];
}

/* Per-head RMSNorm over head_dim (256), used on q and k before RoPE. w is the
 * shared [256] gain vector applied to every head. */
__global__ void q36_head_rmsnorm(float *x, const float *w, int n_head, int head_dim, float eps) {
    int h = blockIdx.x;                 /* one block per head */
    float *xh = x + h*head_dim;
    __shared__ float s;
    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) local += xh[i]*xh[i];
    local = warp_reduce_sum(local);
    if ((threadIdx.x & 31) == 0) atomicAdd(&s, local); /* head_dim<=256 -> few warps */
    if (threadIdx.x == 0) {}
    __syncthreads();
    float scale = rsqrtf(s/head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xh[i] = xh[i]*scale*w[i];
}

/* ---- partial mRoPE (temporal axis for text) ----------------------------
 * Rotate the first ROT_DIM (64) dims of each head with NeoX-style pairing.
 * Text-only decode uses the temporal position `pos` for all rope sections. */
__global__ void q36_rope(float *x, int n_head, int head_dim, int rot_dim,
                         int pos, float freq_base) {
    int h = blockIdx.x;
    int i = threadIdx.x;                /* pair index within [0, rot_dim/2) */
    if (i >= rot_dim/2) return;
    float *xh = x + h*head_dim;
    float inv = powf(freq_base, -2.0f*i/rot_dim);
    float ang = pos * inv;
    float c = cosf(ang), s = sinf(ang);
    float a = xh[i], b = xh[i + rot_dim/2];
    xh[i]            = a*c - b*s;
    xh[i + rot_dim/2]= a*s + b*c;
}

/* ---- sampling: argmax + temperature/top-k/top-p ------------------------- */
__global__ void q36_argmax(const float *logits, int n, int *out) {
    /* Race-free block argmax: warp-reduce (value,index) pairs, stage one pair
     * per warp in shared memory, final scan by thread 0. */
    __shared__ float sv[32]; __shared__ int si[32];
    float lv = -1e30f; int li = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > lv) { lv = logits[i]; li = i; }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, lv, o);
        int   oi = __shfl_down_sync(0xffffffff, li, o);
        if (ov > lv) { lv = ov; li = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = lv; si[wid] = li; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) / 32;
        float bv = sv[0]; int bi = si[0];
        for (int w = 1; w < nw; w++) if (sv[w] > bv) { bv = sv[w]; bi = si[w]; }
        *out = bi;
    }
}

/* Host-visible thin wrappers (extern "C") let the C driver call kernels without
 * seeing CUDA types.  Full attention/SSM/MoE orchestration is added alongside. */
extern "C" void q36_cuda_matvec(uint32_t type, const void *W, const float *x, float *y,
                                int M, int K, uint64_t row_bytes) {
    int warps_per_block = 8;                 /* 256 threads */
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    q36_matvec<<<blocks, warps_per_block*WARP>>>(type, W, x, y, M, K, row_bytes);
}
extern "C" void q36_cuda_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    q36_rmsnorm<<<1, 256>>>(x, w, y, n, eps);
}
extern "C" void q36_cuda_rope(float *x, int n_head, int head_dim, int rot_dim, int pos, float fb) {
    q36_rope<<<n_head, rot_dim/2>>>(x, n_head, head_dim, rot_dim, pos, fb);
}
/* two-stage argmax: 128 blocks of partials, then a single-warp finish.
 * The single-block version left 169 of 170 SMs idle (74us for a 1MB scan). */
__global__ void q36_argmax_p1(const float *logits, int n, float *pv, int *pi) {
    __shared__ float sv[32]; __shared__ int si[32];
    float lv = -1e30f; int li = 0;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
        if (logits[i] > lv) { lv = logits[i]; li = i; }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, lv, o);
        int   oi = __shfl_down_sync(0xffffffff, li, o);
        if (ov > lv) { lv = ov; li = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = lv; si[wid] = li; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) / 32;
        for (int w = 1; w < nw; w++) if (sv[w] > sv[0]) { sv[0] = sv[w]; si[0] = si[w]; }
        pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0];
    }
}
__global__ void q36_argmax_p2(const float *pv, const int *pi, int nparts, int *out) {
    int lane = threadIdx.x;
    float lv = (lane < nparts) ? pv[lane] : -1e30f;
    int   li = (lane < nparts) ? pi[lane] : 0;
    for (int i = lane + 32; i < nparts; i += 32)
        if (pv[i] > lv) { lv = pv[i]; li = pi[i]; }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, lv, o);
        int   oi = __shfl_down_sync(0xffffffff, li, o);
        if (ov > lv) { lv = ov; li = oi; }
    }
    if (lane == 0) *out = li;
}

/* graph-capturable form: launch only.  Scratch must be allocated eagerly
 * (q36_cuda_argmax_init from engine create) -- cudaMalloc during graph
 * capture is illegal. */
static float *g_amax_pv = NULL; static int *g_amax_pi = NULL;
extern "C" void q36_cuda_argmax_init(void) {
    if (!g_amax_pv) { cudaMalloc(&g_amax_pv, 128*sizeof(float)); cudaMalloc(&g_amax_pi, 128*sizeof(int)); }
}
extern "C" void q36_cuda_argmax_async(const float *logits, int n, int *d_out) {
    q36_argmax_p1<<<128, 256>>>(logits, n, g_amax_pv, g_amax_pi);
    q36_argmax_p2<<<1, 32>>>(g_amax_pv, g_amax_pi, 128, d_out);
}

extern "C" int q36_cuda_argmax(const float *logits, int n) {
    static int *d_out = NULL;   /* cached: per-call cudaMalloc can sync the device */
    int h;
    if (!d_out) Q36_CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
    q36_argmax<<<1, 1024>>>(logits, n, d_out);
    Q36_CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    return h;
}

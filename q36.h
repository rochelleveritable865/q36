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

#ifndef QWEN36_H
#define QWEN36_H

/* Qwen3.6-35B-A3B ("qwen3_5_moe") dedicated inference engine.
 *
 * This is NOT a generic GGUF runner.  It targets exactly one model shape,
 * validated against the published Unsloth MXFP4 GGUF, so every hot loop can
 * read compile-time-known dimensions and the CUDA path can specialize kernels
 * to this architecture.  The design bet (see README) is that a single-model
 * engine can beat generic engines (vLLM/SGLang) on the axes they treat as a
 * long tail: hybrid SSM/attention scheduling, MoE expert-gather efficiency,
 * and quant-format-specific bandwidth.
 *
 * Architecture (from GGUF metadata, arch=qwen35moe):
 *
 *   - 40 blocks, d_model 2048, RMSNorm eps 1e-6, vocab 248320, ctx 262144.
 *   - HYBRID layer schedule: full_attention_interval = 4.  One block in four is
 *     a full softmax-attention block; the other three are linear-attention /
 *     state-space (SSM, gated-DeltaNet style) blocks.  Attention blocks carry
 *     attn_* tensors; SSM blocks carry ssm_* tensors.
 *   - Full attention: GQA 16 query heads / 2 KV heads, head_dim 256, per-head
 *     q/k RMSNorm, an output gate (attn_gate), and partial mRoPE (rot dim 64,
 *     sections [11,11,10,0], freq_base 1e7).
 *   - SSM block: causal conv1d (kernel 4), selective state-space scan with
 *     state_size 128, 16 groups, dt rank 32, inner_size 4096, ssm_norm, ssm_out.
 *   - MoE FFN on every block: F32 router -> top-8 of 256 experts, expert FFN
 *     width 512 with SwiGLU (gate/up MXFP4, down Q5_K), plus one always-on
 *     shared expert (width 512, Q8_0) with its own sigmoid input gate.
 *   - Quant mix: attn/shared/embed/output/ssm_out = Q8_0, routed gate/up =
 *     MXFP4, routed down = Q5_K, norms/router/ssm params = F32.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* -------- Fixed model shape (compile-time known) ------------------------- */

enum {
    Q36_N_LAYER          = 40,
    Q36_D_MODEL          = 2048,
    Q36_N_VOCAB          = 248320,
    Q36_CTX_MAX          = 262144,

    /* Full attention blocks */
    Q36_N_HEAD           = 16,
    Q36_N_HEAD_KV        = 2,
    Q36_HEAD_DIM         = 256,       /* key_length == value_length */
    Q36_Q_DIM            = Q36_N_HEAD    * Q36_HEAD_DIM,   /* 4096 */
    Q36_KV_DIM           = Q36_N_HEAD_KV * Q36_HEAD_DIM,   /* 512  */
    Q36_ROT_DIM          = 64,        /* partial rope over first 64 dims */
    Q36_ATTN_INTERVAL    = 4,         /* full_attention_interval */

    /* SSM (linear attention) blocks */
    Q36_SSM_CONV_K       = 4,
    Q36_SSM_STATE        = 128,
    Q36_SSM_GROUPS       = 16,
    Q36_SSM_DT_RANK      = 32,
    Q36_SSM_INNER        = 4096,

    /* MoE */
    Q36_N_EXPERT         = 256,
    Q36_N_EXPERT_USED    = 8,
    Q36_EXPERT_FF        = 512,
    Q36_SHARED_FF        = 512,
};

#define Q36_RMS_EPS        1e-6f
#define Q36_ROPE_FREQ_BASE 1.0e7f
/* mRoPE per-axis rotary section widths (t, h, w, extra); text-only inference
 * uses the temporal axis, but the split is kept so multimodal positions map. */
#define Q36_ROPE_SECTIONS  { 11, 11, 10, 0 }

/* A block is a full-attention block iff (layer % interval) selects it.  The
 * exact phase is validated at load time against which tensors are present; this
 * predicate is the default schedule and is asserted, not assumed. */
static inline bool q36_layer_is_attention(int layer) {
    /* Qwen3-Next style: the attention block is the last of each group of
     * `interval`, i.e. layers 3,7,11,... are full attention. Confirmed per
     * layer at bind time by tensor presence. */
    return (layer % Q36_ATTN_INTERVAL) == (Q36_ATTN_INTERVAL - 1);
}

/* -------- GGUF quant block layouts (must match ggml on-disk format) ------ */

#define QK_K 256   /* super-block size for k-quants */
#define QK_MXFP4 32 /* MXFP4 block: 32 elems, one shared e8m0 scale */

/* Q8_0: 32 int8 weights + one f16 scale. */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;      /* 34 bytes */

/* MXFP4: OCP micro-scaling FP4 (E2M1) with a shared 8-bit exponent (E8M0)
 * scale per 32-element block.  16 packed bytes hold 32 4-bit codes. */
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4; /* 17 bytes */

/* Q5_K: k-quant, 256 weights/super-block, 8 sub-blocks of 32. */
typedef struct {
    uint16_t d;              /* super-block scale (f16) */
    uint16_t dmin;           /* super-block min   (f16) */
    uint8_t  scales[12];     /* 6-bit scales/mins, packed */
    uint8_t  qh[QK_K/8];     /* high bit of each 5-bit weight */
    uint8_t  qs[QK_K/2];     /* low 4 bits */
} block_q5_K;                /* 176 bytes */

/* Q6_K: 256 weights/super-block. */
typedef struct {
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    uint16_t d;
} block_q6_K;                /* 210 bytes */

typedef enum {
    Q36_GGML_F32   = 0,
    Q36_GGML_F16   = 1,
    Q36_GGML_Q8_0  = 8,
    Q36_GGML_Q5_K  = 13,
    Q36_GGML_Q6_K  = 14,
    Q36_GGML_BF16  = 30,   /* block 40 (nextn) router tensors in the MTP GGUF */
    Q36_GGML_MXFP4 = 39,
} q36_ggml_type;

#endif /* QWEN36_H */

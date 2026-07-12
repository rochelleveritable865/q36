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

#ifndef QWEN36_MODEL_H
#define QWEN36_MODEL_H

/* Weight binding: resolves every GGUF tensor to a typed pointer + quant type.
 * The engine keeps the GGUF mmap alive and refers to weights in place on the
 * host; the CUDA backend uploads them once at load time.  Nothing here copies
 * tensor data. */

#include "gguf.h"
#include "q36.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A bound weight is just (host pointer, ggml type, element count).  The CUDA
 * loader consumes these to allocate + upload; the CPU reference reads directly. */
typedef struct {
    const void *data;   /* into the GGUF mapping */
    uint32_t    type;   /* q36_ggml_type */
    uint64_t    n_elem; /* logical element count (product of dims) */
    uint64_t    nbytes; /* on-disk byte size */
    uint64_t    dims[4];
    uint32_t    n_dims;
} q36_weight;

/* Full-attention block (gated GQA). */
typedef struct {
    q36_weight attn_norm;    /* F32 [2048]        input RMSNorm            */
    q36_weight q;            /* Q8_0 [2048,8192]  4096 query + 4096 gate   */
    q36_weight k;            /* Q8_0 [2048,512]                            */
    q36_weight v;            /* Q8_0 [2048,512]                            */
    q36_weight q_norm;       /* F32 [256]  per-head RMSNorm over head_dim  */
    q36_weight k_norm;       /* F32 [256]                                  */
    q36_weight o;            /* Q8_0 [4096,2048]  output projection        */
} q36_attn_block;

/* Linear-attention / SSM block (gated DeltaNet). */
typedef struct {
    q36_weight attn_norm;    /* F32 [2048]  input RMSNorm                  */
    q36_weight qkv;          /* Q8_0 [2048,8192]  fused in-projection      */
    q36_weight conv1d;       /* F32 [4,8192]  causal depthwise conv (k=4)  */
    q36_weight a;            /* F32 [32]   log-decay per dt-rank channel   */
    q36_weight dt_bias;      /* F32 [32]                                   */
    q36_weight alpha;        /* F32 [2048,32]  input-dependent gate proj   */
    q36_weight beta;         /* F32 [2048,32]                              */
    q36_weight ssm_norm;     /* F32 [128]  group RMSNorm over state        */
    q36_weight gate;         /* Q8_0 [2048,4096]  output gate projection   */
    q36_weight out;          /* Q8_0 [4096,2048]  output projection        */
} q36_ssm_block;

/* MoE FFN, present on every block. */
typedef struct {
    q36_weight router;       /* F32 [2048,256]  gate logits                */
    q36_weight gate_exps;    /* MXFP4 [2048,512,256]  per-expert gate      */
    q36_weight up_exps;      /* MXFP4 [2048,512,256]  per-expert up        */
    q36_weight down_exps;    /* Q5_K  [512,2048,256]  per-expert down      */
    /* Always-on shared expert with its own sigmoid input gate. */
    q36_weight shared_gate_inp; /* F32 [2048]  scalar sigmoid gate proj    */
    q36_weight shared_gate;     /* Q8_0 [2048,512]                         */
    q36_weight shared_up;       /* Q8_0 [2048,512]                         */
    q36_weight shared_down;     /* Q8_0 [512,2048]                         */
    q36_weight post_norm;    /* F32 [2048]  post-attention (pre-MoE) norm  */
} q36_moe_block;

typedef struct {
    bool is_attention;       /* true: q36_attn_block; false: q36_ssm_block */
    q36_attn_block attn;
    q36_ssm_block  ssm;
    q36_moe_block  moe;
} q36_block;

/* Multi-Token-Prediction (nextn) module -- an EXTRA block (index 40) present
 * only in the MTP GGUF variant.  It is a full attention + MoE block whose
 * input is eh_proj(concat(hnorm(prev_hidden), enorm(embed(next_token)))),
 * and whose output goes through shared_head_norm into the SHARED output head
 * to predict token t+2.  Enables lossless self-speculative decoding. */
typedef struct {
    q36_attn_block attn;
    q36_moe_block  moe;
    q36_weight eh_proj;          /* [4096,2048]  fuse concat -> D            */
    q36_weight enorm;            /* [2048]  RMSNorm on the next-token embed  */
    q36_weight hnorm;            /* [2048]  RMSNorm on the previous hidden   */
    q36_weight shared_head_norm; /* [2048]  RMSNorm before the shared head   */
} q36_mtp_block;

typedef struct {
    gguf_file gguf;
    q36_weight token_embd;   /* Q8_0 [2048,248320] */
    q36_weight output_norm;  /* F32  [2048]        */
    q36_weight output;       /* Q8_0 [2048,248320] (untied) */
    q36_block  blocks[Q36_N_LAYER];
    int n_attn, n_ssm;
    bool       has_mtp;      /* MTP GGUF variant: block 40 is the nextn module */
    q36_mtp_block mtp;

    /* mRoPE precompute helpers, filled from metadata. */
    float rope_freq_base;
    int   rope_sections[4];
    uint32_t bos_id, eos_id;
} q36_model;

int  q36_model_open(q36_model *m, const char *path, char *err, size_t errlen);
void q36_model_close(q36_model *m);
uint64_t q36_model_total_weight_bytes(const q36_model *m);
/* Bytes read for one decode step at batch=1 (active weights): the denominator
 * of the memory-bandwidth roofline. */
uint64_t q36_model_active_bytes_per_token(const q36_model *m);

#ifdef __cplusplus
}
#endif

#endif

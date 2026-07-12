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

#include "q36_model.h"

#include <stdio.h>
#include <string.h>

/* Bind one tensor by name into a q36_weight; validate it exists and, if a
 * nonzero expected type is given, that the quant matches.  A single missing or
 * mistyped tensor fails the whole open -- this engine refuses partial models. */
static int bind(q36_model *m, q36_weight *w, const char *name,
                uint32_t want_type, char *err, size_t errlen) {
    const gguf_tensor *t = gguf_tensor_find(&m->gguf, name);
    if (!t) { snprintf(err, errlen, "missing tensor %s", name); return -1; }
    if (want_type && t->ggml_type != want_type) {
        snprintf(err, errlen, "tensor %s is %s, expected %s", name,
                 gguf_type_name(t->ggml_type), gguf_type_name(want_type));
        return -1;
    }
    w->data = t->data; w->type = t->ggml_type; w->nbytes = t->nbytes;
    w->n_dims = t->n_dims; w->n_elem = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) { w->dims[d] = t->dims[d]; w->n_elem *= t->dims[d]; }
    return 0;
}

#define BIND(w, fmt, type) do { \
    char _n[80]; snprintf(_n, sizeof _n, fmt, L); \
    if (bind(m, (w), _n, (type), err, errlen)) return -1; \
} while (0)

/* Accept only quant types the CUDA dequant path implements. */
static bool supported_expert_quant(uint32_t t) {
    return t == Q36_GGML_MXFP4 || t == Q36_GGML_Q5_K ||
           t == Q36_GGML_Q6_K  || t == Q36_GGML_Q8_0;
}

/* NOTE: bind()'s want_type check is skipped for F32 because Q36_GGML_F32==0
 * doubles as "any type" -- so "expect F32" was never enforced.  The MTP GGUF
 * exploits the hole: block 40's ffn_gate_inp / ffn_gate_inp_shexp are BF16.
 * The engine widens BF16 to F32 at upload; validate explicitly here so any
 * OTHER type still fails the open instead of faulting a kernel. */
static bool f32_or_bf16(uint32_t t) {
    return t == Q36_GGML_F32 || t == Q36_GGML_BF16;
}

static int bind_moe(q36_model *m, int L, q36_moe_block *e, char *err, size_t errlen) {
    BIND(&e->post_norm,       "blk.%d.post_attention_norm.weight", Q36_GGML_F32);
    BIND(&e->router,          "blk.%d.ffn_gate_inp.weight",        Q36_GGML_F32);
    /* Expert quant is per-layer dynamic (MXFP4 / Q5_K / Q6_K): bind any
     * supported type and let the kernel dispatch on q36_weight.type. */
    BIND(&e->gate_exps,       "blk.%d.ffn_gate_exps.weight",       0);
    BIND(&e->up_exps,         "blk.%d.ffn_up_exps.weight",         0);
    BIND(&e->down_exps,       "blk.%d.ffn_down_exps.weight",       0);
    if (!supported_expert_quant(e->gate_exps.type) ||
        !supported_expert_quant(e->up_exps.type) ||
        !supported_expert_quant(e->down_exps.type)) {
        snprintf(err, errlen, "block %d has unsupported expert quant", L);
        return -1;
    }
    BIND(&e->shared_gate_inp, "blk.%d.ffn_gate_inp_shexp.weight",  Q36_GGML_F32);
    BIND(&e->shared_gate,     "blk.%d.ffn_gate_shexp.weight",      Q36_GGML_Q8_0);
    BIND(&e->shared_up,       "blk.%d.ffn_up_shexp.weight",        Q36_GGML_Q8_0);
    BIND(&e->shared_down,     "blk.%d.ffn_down_shexp.weight",      Q36_GGML_Q8_0);
    if (!f32_or_bf16(e->router.type) || !f32_or_bf16(e->shared_gate_inp.type)) {
        snprintf(err, errlen, "block %d router/gate_inp type unsupported", L);
        return -1;
    }
    return 0;
}

int q36_model_open(q36_model *m, const char *path, char *err, size_t errlen) {
    memset(m, 0, sizeof(*m));
    if (gguf_open(&m->gguf, path, err, errlen) != 0) return -1;

    /* Validate the architecture is the one this engine is built for. */
    char arch[64] = {0};
    gguf_str_dup(&m->gguf, "general.architecture", arch, sizeof arch);
    if (strcmp(arch, "qwen35moe") != 0) {
        snprintf(err, errlen, "arch '%s' != qwen35moe (this engine is single-model)", arch);
        gguf_close(&m->gguf); return -1;
    }
    uint32_t nl = 0, ne = 0, nu = 0;
    gguf_u32(&m->gguf, "qwen35moe.block_count", &nl);
    gguf_u32(&m->gguf, "qwen35moe.expert_count", &ne);
    gguf_u32(&m->gguf, "qwen35moe.expert_used_count", &nu);
    /* block_count is Q36_N_LAYER (40) for the base model, or Q36_N_LAYER+1
     * (41) for the MTP variant whose extra block 40 is the nextn module. */
    m->has_mtp = (nl == Q36_N_LAYER + 1);
    if ((nl != Q36_N_LAYER && !m->has_mtp) || ne != Q36_N_EXPERT || nu != Q36_N_EXPERT_USED) {
        snprintf(err, errlen, "shape mismatch: layers=%u experts=%u/%u (built for %d/%d/%d)",
                 nl, ne, nu, Q36_N_LAYER, Q36_N_EXPERT, Q36_N_EXPERT_USED);
        gguf_close(&m->gguf); return -1;
    }

    m->rope_freq_base = Q36_ROPE_FREQ_BASE;
    gguf_f32(&m->gguf, "qwen35moe.rope.freq_base", &m->rope_freq_base);
    int secs[4] = Q36_ROPE_SECTIONS;
    memcpy(m->rope_sections, secs, sizeof secs);
    gguf_u32(&m->gguf, "tokenizer.ggml.bos_token_id", &m->bos_id);
    gguf_u32(&m->gguf, "tokenizer.ggml.eos_token_id", &m->eos_id);

    /* Global tensors. */
    if (bind(m, &m->token_embd,  "token_embd.weight",  Q36_GGML_Q8_0, err, errlen)) goto fail;
    if (bind(m, &m->output_norm, "output_norm.weight", Q36_GGML_F32,  err, errlen)) goto fail;
    if (bind(m, &m->output,      "output.weight",      Q36_GGML_Q8_0, err, errlen)) goto fail;

    /* Per-block: detect block type by tensor presence, then bind + validate.
     * The schedule predicate q36_layer_is_attention() is asserted against the
     * actual tensors, so a GGUF that disagrees with our fixed schedule is
     * rejected rather than silently mis-run. */
    for (int L = 0; L < Q36_N_LAYER; L++) {
        q36_block *b = &m->blocks[L];
        char nm[80];
        snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", L);
        bool has_attn = gguf_tensor_find(&m->gguf, nm) != NULL;
        snprintf(nm, sizeof nm, "blk.%d.ssm_conv1d.weight", L);
        bool has_ssm  = gguf_tensor_find(&m->gguf, nm) != NULL;
        if (has_attn == has_ssm) {
            snprintf(err, errlen, "block %d ambiguous (attn=%d ssm=%d)", L, has_attn, has_ssm);
            goto fail;
        }
        b->is_attention = has_attn;
        if (has_attn != q36_layer_is_attention(L)) {
            snprintf(err, errlen, "block %d type %s contradicts fixed schedule",
                     L, has_attn ? "attn" : "ssm");
            goto fail;
        }

        if (b->is_attention) {
            q36_attn_block *a = &b->attn;
            BIND(&a->attn_norm, "blk.%d.attn_norm.weight",   Q36_GGML_F32);
            BIND(&a->q,         "blk.%d.attn_q.weight",      Q36_GGML_Q8_0);
            BIND(&a->k,         "blk.%d.attn_k.weight",      Q36_GGML_Q8_0);
            BIND(&a->v,         "blk.%d.attn_v.weight",      Q36_GGML_Q8_0);
            BIND(&a->q_norm,    "blk.%d.attn_q_norm.weight", Q36_GGML_F32);
            BIND(&a->k_norm,    "blk.%d.attn_k_norm.weight", Q36_GGML_F32);
            BIND(&a->o,         "blk.%d.attn_output.weight", Q36_GGML_Q8_0);
            m->n_attn++;
        } else {
            q36_ssm_block *s = &b->ssm;
            BIND(&s->attn_norm, "blk.%d.attn_norm.weight",   Q36_GGML_F32);
            BIND(&s->qkv,       "blk.%d.attn_qkv.weight",    Q36_GGML_Q8_0);
            BIND(&s->conv1d,    "blk.%d.ssm_conv1d.weight",  Q36_GGML_F32);
            BIND(&s->a,         "blk.%d.ssm_a",              Q36_GGML_F32);
            BIND(&s->dt_bias,   "blk.%d.ssm_dt.bias",        Q36_GGML_F32);
            BIND(&s->alpha,     "blk.%d.ssm_alpha.weight",   Q36_GGML_F32);
            BIND(&s->beta,      "blk.%d.ssm_beta.weight",    Q36_GGML_F32);
            BIND(&s->ssm_norm,  "blk.%d.ssm_norm.weight",    Q36_GGML_F32);
            BIND(&s->gate,      "blk.%d.attn_gate.weight",   Q36_GGML_Q8_0);
            BIND(&s->out,       "blk.%d.ssm_out.weight",     Q36_GGML_Q8_0);
            m->n_ssm++;
        }
        if (bind_moe(m, L, &b->moe, err, errlen)) goto fail;
    }

    if (m->has_mtp) {           /* block 40: nextn / MTP module */
        int L = Q36_N_LAYER;    /* 40 -- BIND() uses L for the tensor name */
        q36_attn_block *a = &m->mtp.attn;
        BIND(&a->attn_norm, "blk.%d.attn_norm.weight",   Q36_GGML_F32);
        BIND(&a->q,         "blk.%d.attn_q.weight",      Q36_GGML_Q8_0);
        BIND(&a->k,         "blk.%d.attn_k.weight",      Q36_GGML_Q8_0);
        BIND(&a->v,         "blk.%d.attn_v.weight",      Q36_GGML_Q8_0);
        BIND(&a->q_norm,    "blk.%d.attn_q_norm.weight", Q36_GGML_F32);
        BIND(&a->k_norm,    "blk.%d.attn_k_norm.weight", Q36_GGML_F32);
        BIND(&a->o,         "blk.%d.attn_output.weight", Q36_GGML_Q8_0);
        if (bind_moe(m, L, &m->mtp.moe, err, errlen)) goto fail;
        BIND(&m->mtp.eh_proj,          "blk.%d.nextn.eh_proj.weight",          0);
        BIND(&m->mtp.enorm,            "blk.%d.nextn.enorm.weight",            Q36_GGML_F32);
        BIND(&m->mtp.hnorm,            "blk.%d.nextn.hnorm.weight",            Q36_GGML_F32);
        BIND(&m->mtp.shared_head_norm, "blk.%d.nextn.shared_head_norm.weight", Q36_GGML_F32);
    }
    return 0;

fail:
    gguf_close(&m->gguf);
    return -1;
}

void q36_model_close(q36_model *m) { gguf_close(&m->gguf); memset(m, 0, sizeof(*m)); }

uint64_t q36_model_total_weight_bytes(const q36_model *m) {
    uint64_t t = 0;
    for (uint64_t i = 0; i < m->gguf.n_tensors; i++) t += m->gguf.tensors[i].nbytes;
    return t;
}

/* Active bytes/token = all always-resident weights read every decode step plus
 * only the top-8 routed experts (not all 256).  This is the roofline
 * denominator that decides the batch=1 decode ceiling. */
uint64_t q36_model_active_bytes_per_token(const q36_model *m) {
    uint64_t bytes = 0;
    /* embedding lookup: one row of token_embd (Q8_0): 2048 elems */
    bytes += (Q36_D_MODEL / 32) * sizeof(block_q8_0);
    bytes += m->output_norm.nbytes + m->output.nbytes; /* output head, full */
    for (int L = 0; L < Q36_N_LAYER; L++) {
        const q36_block *b = &m->blocks[L];
        if (b->is_attention) {
            bytes += b->attn.attn_norm.nbytes + b->attn.q.nbytes + b->attn.k.nbytes +
                     b->attn.v.nbytes + b->attn.o.nbytes;
        } else {
            bytes += b->ssm.attn_norm.nbytes + b->ssm.qkv.nbytes + b->ssm.conv1d.nbytes +
                     b->ssm.alpha.nbytes + b->ssm.beta.nbytes + b->ssm.gate.nbytes + b->ssm.out.nbytes;
        }
        const q36_moe_block *e = &b->moe;
        bytes += e->post_norm.nbytes + e->router.nbytes;
        bytes += e->shared_gate.nbytes + e->shared_up.nbytes + e->shared_down.nbytes;
        /* Only top-8 of 256 routed experts are read per token. */
        uint64_t per_expert = (e->gate_exps.nbytes + e->up_exps.nbytes + e->down_exps.nbytes) / Q36_N_EXPERT;
        bytes += per_expert * Q36_N_EXPERT_USED;
    }
    return bytes;
}

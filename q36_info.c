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

/* Open + fully bind the model, then print the memory-bandwidth roofline that
 * bounds batch=1 decode on various GPUs.  This is the "how much headroom"
 * number: t/s_max = bandwidth / active_bytes_per_token. */
#include "q36_model.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    q36_model m; char err[256];
    if (q36_model_open(&m, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "open failed: %s\n", err); return 1;
    }
    uint64_t total  = q36_model_total_weight_bytes(&m);
    uint64_t active = q36_model_active_bytes_per_token(&m);

    /* Break active bytes/token into what each optimization lever can touch. */
    uint64_t b_mxfp4=0, b_q5k=0, b_q8_dense=0, b_output=0, b_f32=0;
    b_output = m.output.nbytes;
    for (int L=0; L<Q36_N_LAYER; L++){
        const q36_block *b=&m.blocks[L];
        if (b->is_attention){
            b_q8_dense += b->attn.q.nbytes+b->attn.k.nbytes+b->attn.v.nbytes+b->attn.o.nbytes;
        } else {
            b_q8_dense += b->ssm.qkv.nbytes+b->ssm.gate.nbytes+b->ssm.out.nbytes;
            b_f32 += b->ssm.conv1d.nbytes+b->ssm.alpha.nbytes+b->ssm.beta.nbytes;
        }
        const q36_moe_block *e=&b->moe;
        b_q8_dense += e->shared_gate.nbytes+e->shared_up.nbytes+e->shared_down.nbytes;
        b_f32 += e->router.nbytes;
        /* top-8 routed experts, split by quant */
        uint64_t g=e->gate_exps.nbytes/Q36_N_EXPERT, u=e->up_exps.nbytes/Q36_N_EXPERT, d=e->down_exps.nbytes/Q36_N_EXPERT;
        uint64_t used=Q36_N_EXPERT_USED;
        if (e->gate_exps.type==Q36_GGML_MXFP4) b_mxfp4+=g*used; else b_q5k+=g*used;
        if (e->up_exps.type==Q36_GGML_MXFP4)   b_mxfp4+=u*used; else b_q5k+=u*used;
        if (e->down_exps.type==Q36_GGML_MXFP4) b_mxfp4+=d*used; else b_q5k+=d*used;
    }
    printf("model bound OK: %d attention blocks, %d ssm blocks\n", m.n_attn, m.n_ssm);
    printf("active-bytes/token breakdown (what each lever can optimize):\n");
    printf("  Q8_0 dense (attn+ssm+shared): %6.1f MB  %4.1f%%\n", b_q8_dense/1e6, 100.0*b_q8_dense/active);
    printf("  Q8_0 output head            : %6.1f MB  %4.1f%%\n", b_output/1e6, 100.0*b_output/active);
    printf("  MXFP4 routed experts (top-8): %6.1f MB  %4.1f%%  <- the 'optimize MXFP4' lever\n", b_mxfp4/1e6, 100.0*b_mxfp4/active);
    printf("  Q5_K/Q6_K routed experts    : %6.1f MB  %4.1f%%\n", b_q5k/1e6, 100.0*b_q5k/active);
    printf("  F32 (router/conv/ssm gates) : %6.1f MB  %4.1f%%\n", b_f32/1e6, 100.0*b_f32/active);
    printf("bos=%u eos=%u rope_freq_base=%.0f\n", m.bos_id, m.eos_id, m.rope_freq_base);
    printf("total weights      : %.2f GiB\n", total / 1073741824.0);
    printf("active/decode token: %.3f GiB  (top-%d of %d experts + dense)\n",
           active / 1073741824.0, Q36_N_EXPERT_USED, Q36_N_EXPERT);

    struct { const char *n; double bw; } hw[] = {
        {"RTX 5090   (1.79 TB/s)*", 1792e9}, /* <- target: 32GB, sm_120 */
        {"RTX 4090   (1.01 TB/s)",  1008e9}, {"A100 80G   (2.04 TB/s)", 2039e9},
        {"H100 SXM   (3.35 TB/s)",  3350e9}, {"B200       (8.0 TB/s)",  8000e9},
    };
    printf("* target GPU. 20.2 GiB weights fit resident in 32 GiB (no streaming).\n");
    double abt = (double)active;
    printf("\n%-24s %14s %18s\n", "GPU", "roofline t/s", "realistic ~65%");
    for (size_t i = 0; i < sizeof hw/sizeof *hw; i++) {
        double ceil_ts = hw[i].bw / abt;
        printf("%-24s %14.0f %18.0f\n", hw[i].n, ceil_ts, ceil_ts * 0.65);
    }
    printf("\nHeadroom read: a dedicated engine can only close the gap between a\n"
           "generic engine's achieved t/s and the roofline above -- it cannot pass\n"
           "it. The larger wins are in prefill, MoE expert-gather, and the SSM scan.\n");
    q36_model_close(&m);
    return 0;
}

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

/* Standalone loader validation: parse the real GGUF, print shape + a few
 * tensors, and confirm every tensor's data pointer/size fits the mapping.
 * Build: cc -O2 gguf_dump.c gguf.c -o gguf_dump */
#include "gguf.h"
#include "q36.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    gguf_file g; char err[256];
    if (gguf_open(&g, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "load failed: %s\n", err); return 1;
    }
    char arch[64] = {0};
    gguf_str_dup(&g, "general.architecture", arch, sizeof arch);
    uint32_t nl=0, nx=0, nu=0; gguf_u32(&g,"qwen35moe.block_count",&nl);
    gguf_u32(&g,"qwen35moe.expert_count",&nx); gguf_u32(&g,"qwen35moe.expert_used_count",&nu);
    printf("arch=%s version=%u tensors=%llu kv=%llu\n", arch, g.version,
           (unsigned long long)g.n_tensors, (unsigned long long)g.n_kv);
    printf("blocks=%u experts=%u/%u  (header says: %d layers expected)\n",
           nl, nu, nx, Q36_N_LAYER);

    /* Spot-check the tensors the loader must bind, on an attention block (3)
     * and an SSM block (0), and the MoE experts. */
    const char *probe[] = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_gate.weight",
        "blk.3.attn_q_norm.weight", "blk.0.ssm_conv1d.weight", "blk.0.ssm_a",
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_gate_inp.weight", "blk.0.ffn_up_shexp.weight",
    };
    printf("\n%-34s %-6s %-22s %10s\n", "tensor", "type", "dims", "MiB");
    for (size_t i = 0; i < sizeof probe/sizeof *probe; i++) {
        const gguf_tensor *t = gguf_tensor_find(&g, probe[i]);
        if (!t) { printf("%-34s MISSING\n", probe[i]); continue; }
        char dbuf[64]; int o=0;
        for (uint32_t d=0; d<t->n_dims; d++) o+=snprintf(dbuf+o,sizeof dbuf-o,"%s%llu",d?"x":"",(unsigned long long)t->dims[d]);
        printf("%-34s %-6s %-22s %10.2f\n", probe[i], gguf_type_name(t->ggml_type),
               dbuf, t->nbytes/1048576.0);
    }

    /* Total on-disk tensor bytes as a sanity check vs file size. */
    unsigned long long total=0; int attn=0, ssm=0;
    for (uint64_t i=0;i<g.n_tensors;i++){
        total += g.tensors[i].nbytes;
        if (strstr(g.tensors[i].name.ptr ? "" : "", "")) {}
    }
    for (int L=0; L<Q36_N_LAYER; L++){
        char nm[64]; snprintf(nm,sizeof nm,"blk.%d.attn_q.weight",L);
        if (gguf_tensor_find(&g,nm)) attn++;
        snprintf(nm,sizeof nm,"blk.%d.ssm_conv1d.weight",L);
        if (gguf_tensor_find(&g,nm)) ssm++;
    }
    printf("\nsummed tensor bytes = %.2f GiB (file map = %.2f GiB)\n",
           total/1073741824.0, g.map_size/1073741824.0);
    printf("layer census: %d attention blocks, %d ssm blocks (of %d)\n",
           attn, ssm, Q36_N_LAYER);
    gguf_close(&g);
    return 0;
}

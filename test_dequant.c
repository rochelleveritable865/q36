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

/* CPU validation of the dequant core against the real GGUF and against
 * hand-computed reference constants.  Build:
 *   cc -O2 test_dequant.c gguf.c -o test_dequant
 * (q36_dequant.cuh compiles as plain C here via the Q36_HD fallback.) */
#include "gguf.h"
#include "q36.h"
#include "q36_dequant.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(cond, ...) do { if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while(0)
static int approx(float a, float b){ return fabsf(a-b) <= 1e-4f*(1+fabsf(b)); }

static void stats(const char *name, const float *v, int n) {
    float mn=v[0], mx=v[0]; double s=0, s2=0; int bad=0;
    for (int i=0;i<n;i++){ float x=v[i]; if(!isfinite(x))bad++; if(x<mn)mn=x; if(x>mx)mx=x; s+=x; s2+=(double)x*x; }
    double mean=s/n, var=s2/n-mean*mean;
    printf("  %-28s n=%d min=%.4f max=%.4f mean=%.4f std=%.4f nonfinite=%d\n",
           name, n, mn, mx, mean, sqrt(var>0?var:0), bad);
    CHECK(bad==0, "%s has %d non-finite values", name, bad);
}

int main(int argc, char **argv) {
    printf("== constant checks ==\n");
    CHECK(sizeof(block_q8_0)==34, "q8_0 size %zu", sizeof(block_q8_0));
    CHECK(sizeof(block_mxfp4)==17, "mxfp4 size %zu", sizeof(block_mxfp4));
    CHECK(sizeof(block_q5_K)==176, "q5_K size %zu", sizeof(block_q5_K));
    CHECK(sizeof(block_q6_K)==210, "q6_K size %zu", sizeof(block_q6_K));
    CHECK(approx(q36_fp16(0x3C00),1.0f), "fp16 1.0 -> %f", q36_fp16(0x3C00));
    CHECK(approx(q36_fp16(0x4000),2.0f), "fp16 2.0 -> %f", q36_fp16(0x4000));
    CHECK(approx(q36_fp16(0xBC00),-1.0f),"fp16 -1.0 -> %f", q36_fp16(0xBC00));
    CHECK(approx(q36_e8m0_half(127),0.5f),"e8m0(127)=0.5 -> %f", q36_e8m0_half(127));
    CHECK(approx(q36_e8m0_half(128),1.0f),"e8m0(128)=1.0 -> %f", q36_e8m0_half(128));
    CHECK(approx(q36_e8m0_half(129),2.0f),"e8m0(129)=2.0 -> %f", q36_e8m0_half(129));
    /* MXFP4: e=128 (scale 1.0), codes 0..7 -> {0,1,2,3,4,6,8,12} */
    { block_mxfp4 b; b.e=128; for(int i=0;i<16;i++) b.qs[i]=(uint8_t)i; /* low nibbles 0..15 wrap */
      float o[32]; q36_deq_mxfp4(&b,o);
      float want[8]={0,1,2,3,4,6,8,12};
      for(int i=0;i<8;i++) CHECK(approx(o[i],want[i]),"mxfp4 code %d -> %f want %f",i,o[i],want[i]); }
    printf("  constants: %s\n", fails? "SOME FAILED":"OK");

    if (argc < 2) { printf("\n(pass model path to validate real tensors)\n"); return fails?1:0; }

    gguf_file g; char err[256];
    if (gguf_open(&g, argv[1], err, sizeof err)) { fprintf(stderr,"%s\n",err); return 1; }
    printf("\n== real-tensor dequant (block 0) ==\n");
    struct { const char *nm; int n; } probe[] = {
        {"token_embd.weight", 2048},          /* Q8_0, first row (one token embedding) */
        {"blk.0.ffn_gate_exps.weight", 512},  /* MXFP4, first expert first output row-ish */
        {"blk.0.ffn_down_exps.weight", 512},  /* Q5_K */
        {"blk.3.attn_q.weight", 2048},        /* Q8_0 */
    };
    for (size_t i=0;i<sizeof probe/sizeof*probe;i++){
        const gguf_tensor *t = gguf_tensor_find(&g, probe[i].nm);
        if(!t){ printf("  %s MISSING\n", probe[i].nm); continue; }
        int n = probe[i].n;
        float *out = malloc(sizeof(float)*n);
        q36_deq_row(t->ggml_type, t->data, n, out);
        char lbl[64]; snprintf(lbl,sizeof lbl,"%s[%s]",probe[i].nm,gguf_type_name(t->ggml_type));
        stats(lbl, out, n);
        free(out);
    }
    gguf_close(&g);
    printf("\n%s (%d failures)\n", fails? "VALIDATION FAILED":"ALL DEQUANT CHECKS PASSED", fails);
    return fails?1:0;
}

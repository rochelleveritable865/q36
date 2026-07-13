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

/* q36 engine: host orchestration of the clean-room decode forward pass.
 * Weights are uploaded once to the 5090 (20.2 GiB fits in 32 GiB); each decode
 * step runs embedding -> 40 hybrid blocks -> final norm -> logits.  Correctness
 * first: kernels are the straightforward reference form, to be validated
 * against the model's own outputs, then optimized. */

#include "q36_model.h"
#include "q36_ops.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

using namespace nvcuda;

extern "C" void q36_cuda_matvec(uint32_t type,const void*W,const float*x,float*y,int M,int K,uint64_t row_bytes);
extern "C" int  q36_cuda_argmax(const float*logits,int n);
extern "C" void q36_cuda_argmax_async(const float*logits,int n,int*d_out);
extern "C" void q36_cuda_argmax_init(void);

#define CK(x) do{cudaError_t ck_=(x);if(ck_){fprintf(stderr,"cuda %s @%s:%d\n",cudaGetErrorString(ck_),__FILE__,__LINE__);exit(1);}}while(0)
static int gr(int n,int b){return (n+b-1)/b;}

/* per-slot state views (slot 0 == the classic single-sequence pointers) */
#define KCS(e,L,s) ((int8_t*)((uint8_t*)(e)->Kc[L]+(size_t)(s)*(e)->ctx*Q36_KV_DIM*2))
#define KSS(e,L,s) ((__half*)((uint8_t*)(e)->Ks[L]+(size_t)(s)*(e)->ctx*(Q36_KV_DIM/32)*2))
#define VCS(e,L,s) ((uint8_t*)(e)->Vc[L]+(size_t)(s)*(e)->ctx*Q36_KV_DIM*2)
#define VSS(e,L,s) ((uint8_t*)(e)->Vs[L]+(size_t)(s)*(e)->ctx*(Q36_KV_DIM/32))

#define Q36_FA3_TQ 64   /* FA prefill: queries per block (16 per warp) */
#define Q36_FA3_TK 32   /* FA prefill: positions per KV tile           */

/* Expert shard range for multi-GPU expert parallelism: upload() repacks only
 * experts [g_shard_e0, g_shard_e1) of 3D tensors; kernels receive the range
 * and skip non-resident experts.  Full range = single-GPU behavior. */
static int g_shard_e0=0, g_shard_e1=Q36_N_EXPERT;
static int g_no_autoselect=0;
static int g_create_slots=1;   /* q36_engine_create_mt sets before create */

/* device weight: a matvec-ready quantized matrix (or slice of a 3D expert).
 * elayout: 0 = raw gguf blocks, 1 = q8_0-split, 2 = mxfp4-split. */
typedef struct { void*d; uint32_t type; bool split; int elayout; int M,K; uint64_t row_bytes,expert_stride; } dw;

/* host f32 -> f16 (round-to-nearest via cvt) */
static uint16_t h_f16(float f){
    __half h=__float2half(f); uint16_t u; memcpy(&u,&h,2); return u;
}
/* Quantize one row of K f32 values to q8_0-split (qs int8 + per-32 f16 scale). */
static void h_q8_row(const float*xf,int K,int8_t*qs,uint16_t*ds){
    for(int b=0;b<K/32;b++){
        float amax=0;
        for(int j=0;j<32;j++){ float v=fabsf(xf[b*32+j]); if(v>amax)amax=v; }
        float d=amax/127.0f, id=d?1.0f/d:0.0f;
        ds[b]=h_f16(d);
        for(int j=0;j<32;j++){
            int q=(int)lrintf(xf[b*32+j]*id);
            qs[b*32+j]=(int8_t)(q>127?127:(q<-128?-128:q));
        }
    }
}

/* Requantize a Q8_0 2D tensor to e4m3 + per-32 ue8m0 (the FP8 output head):
 * halves the largest decode tensor.  split==2 marks the layout. */
/* NVFP4 repack: Q8_0 -> E2M1 nibbles + per-16 e4m3 scales (4.5 bits/w,
 * 540 -> 286MB for the head).  Scale = amax/12 rounded UP in e4m3 so the
 * max-magnitude code never clips; codes are nearest-value on the halved
 * E2M1 table {0,1,2,3,4,6,8,12} (decode via mxfp4_val). */
static uint8_t h_e2m1_nearest(float v){
    static const float t[8]={0,1,2,3,4,6,8,12};
    float a=fabsf(v); int best=0; float bd=1e30f;
    for(int i=0;i<8;i++){ float d2=fabsf(a-t[i]); if(d2<bd){bd=d2;best=i;} }
    return (uint8_t)(best|((v<0.f&&best)?8:0));
}
static dw upload_nvfp4(const q36_weight*w){
    dw o; memset(&o,0,sizeof o);
    o.type=w->type; o.K=(int)w->dims[0]; o.M=(int)w->dims[1];
    uint64_t nb16=(uint64_t)o.K/16;
    uint64_t qs_bytes=(uint64_t)o.M*(o.K/2), es_bytes=(uint64_t)o.M*nb16;
    uint8_t*buf=(uint8_t*)malloc(qs_bytes+es_bytes);
    const uint8_t*srcb=(const uint8_t*)w->data;
    #pragma omp parallel for schedule(static)
    for(int64_t r=0;r<(int64_t)o.M;r++){
        uint64_t nb32=(uint64_t)o.K/32;
        for(uint64_t b=0;b<nb32;b++){
            const block_q8_0*blk=(const block_q8_0*)(srcb+((uint64_t)r*nb32+b)*34);
            float d=q36_fp16(blk->d), rowf[32];
            for(int j=0;j<32;j++) rowf[j]=d*blk->qs[j];
            for(int h=0;h<2;h++){               /* two 16-blocks per q8 block */
                const float*v16=rowf+h*16;
                float amax=0;
                for(int j=0;j<16;j++){ float a=fabsf(v16[j]); if(a>amax)amax=a; }
                float sf=amax/12.f;
                __nv_fp8_e4m3 s8(sf);
                uint8_t sb=*(uint8_t*)&s8;
                if((float)s8<sf&&sb<0x7e) sb++;          /* round scale UP */
                float sd=(float)*(__nv_fp8_e4m3*)&sb;
                float inv=(sd>0.f)?1.f/sd:0.f;
                uint64_t bi=(uint64_t)r*nb16+b*2+h;
                buf[qs_bytes+bi]=sb;
                for(int j=0;j<8;j++){
                    uint8_t lo=h_e2m1_nearest(v16[j*2]*inv);
                    uint8_t hi=h_e2m1_nearest(v16[j*2+1]*inv);
                    buf[bi*8+j]=(uint8_t)(lo|(hi<<4));
                }
            }
        }
    }
    o.split=true; o.elayout=4;    /* 4 = nvfp4-split */
    CK(cudaMalloc(&o.d,qs_bytes+es_bytes));
    CK(cudaMemcpy(o.d,buf,qs_bytes+es_bytes,cudaMemcpyHostToDevice));
    free(buf);
    return o;
}

/* Frequency-tiered head: BPE ids are ~frequency-ordered (bytes, then
 * merges by rank), so rows [0,R) -- the plausible-candidate tokens whose
 * logit precision matters -- stay FP8-e4m3, and the long tail [R,M) goes
 * NVFP4.  Plain NVFP4 measured ppl +3.5% (rejected); this recovers the
 * frequent rows at a 40% byte cost of the savings. */
static dw upload_head_mixed(const q36_weight*w,int R);
static dw upload_e4m3(const q36_weight*w){
    dw o; memset(&o,0,sizeof o);
    o.type=w->type; o.split=false; o.elayout=0;
    o.K=(int)w->dims[0]; o.M=(int)w->dims[1];
    uint64_t nbr=(uint64_t)o.K/32;
    uint64_t qs_bytes=(uint64_t)o.M*o.K, es_bytes=(uint64_t)o.M*nbr;
    uint8_t*buf=(uint8_t*)malloc(qs_bytes+es_bytes);
    const uint8_t*srcb=(const uint8_t*)w->data;
    #pragma omp parallel for schedule(static)
    for(int64_t r=0;r<(int64_t)o.M;r++){
        float rowf[32];
        for(uint64_t b=0;b<nbr;b++){
            const block_q8_0*blk=(const block_q8_0*)(srcb+((uint64_t)r*nbr+b)*34);
            float d=q36_fp16(blk->d), amax=0;
            for(int j=0;j<32;j++){ rowf[j]=d*blk->qs[j]; float a=fabsf(rowf[j]); if(a>amax)amax=a; }
            int e2=(amax>0.f)?(int)ceilf(log2f(amax/448.f)):-127;
            if(e2<-127)e2=-127;
            buf[qs_bytes+(uint64_t)r*nbr+b]=(uint8_t)(127+e2);
            float inv=exp2f((float)-e2);
            for(int j=0;j<32;j++){
                __nv_fp8_e4m3 q(rowf[j]*inv);
                buf[(uint64_t)r*o.K+b*32+j]=*(uint8_t*)&q;
            }
        }
    }
    o.row_bytes=o.K;              /* unused by split path; kept sane */
    o.split=true; o.elayout=3;    /* 3 = e4m3-split */
    CK(cudaMalloc(&o.d,qs_bytes+es_bytes));
    CK(cudaMemcpy(o.d,buf,qs_bytes+es_bytes,cudaMemcpyHostToDevice));
    free(buf);
    return o;
}

static dw upload_head_mixed(const q36_weight*w,int R){
    dw a=upload_e4m3(w), b=upload_nvfp4(w);   /* full repacks; slice on device */
    /* concatenate: [ e4m3 qs/es of rows 0..R ) | nvfp4 qs/es of rows R..M ) ] */
    int M=a.M, K=a.K;
    uint64_t aq=(uint64_t)R*K, ae=(uint64_t)R*(K/32);
    uint64_t bq=(uint64_t)(M-R)*(K/2), be=(uint64_t)(M-R)*(K/16);
    dw o=a; o.elayout=5; o.row_bytes=(uint64_t)R;
    CK(cudaMalloc(&o.d,aq+ae+bq+be));
    uint8_t*dst=(uint8_t*)o.d;
    const uint8_t*ap=(const uint8_t*)a.d, *bp=(const uint8_t*)b.d;
    CK(cudaMemcpy(dst,ap,aq,cudaMemcpyDeviceToDevice));                       /* e4m3 qs   */
    CK(cudaMemcpy(dst+aq,ap+(uint64_t)M*K,ae,cudaMemcpyDeviceToDevice));      /* e4m3 es   */
    CK(cudaMemcpy(dst+aq+ae,bp+(uint64_t)R*(K/2),bq,cudaMemcpyDeviceToDevice));               /* fp4 qs */
    CK(cudaMemcpy(dst+aq+ae+bq,bp+(uint64_t)M*(K/2)+(uint64_t)R*(K/16),be,cudaMemcpyDeviceToDevice)); /* fp4 es */
    CK(cudaFree(a.d)); CK(cudaFree(b.d));
    return o;
}

static dw upload(const q36_weight*w){
    dw o; o.type=w->type; o.split=false; o.elayout=0;
    /* 3D expert tensors get GPU-friendly layouts:
     *  - MXFP4 -> split nibbles+scales per expert (same bytes, coalesced)
     *  - Q5_K/Q6_K -> requantized to q8_0-split (more bytes, no quality loss,
     *    rides the fast path; the k-quant block layout defeats coalescing)  */
    if(w->n_dims==3){
        int K=(int)w->dims[0], M=(int)w->dims[1];
        o.K=K; o.M=M;
        if(w->type==Q36_GGML_MXFP4){
            o.elayout=2;
            int ne=g_shard_e1-g_shard_e0;
            uint64_t qs_e=(uint64_t)M*(K/2), es_e=(uint64_t)M*(K/32);
            o.expert_stride=qs_e+es_e; o.row_bytes=0;
            uint8_t*buf=(uint8_t*)malloc(o.expert_stride*ne);
            const uint8_t*src=(const uint8_t*)w->data;
            uint64_t nbr=(uint64_t)K/32;
            #pragma omp parallel for schedule(static)
            for(int xi=0;xi<ne;xi++){
                int ex=g_shard_e0+xi;
                uint8_t*qd=buf+(uint64_t)xi*o.expert_stride;
                uint8_t*ed=qd+qs_e;
                for(uint64_t r=0;r<(uint64_t)M;r++) for(uint64_t b=0;b<nbr;b++){
                    const uint8_t*blk=src+(((uint64_t)ex*M+r)*nbr+b)*17;
                    ed[r*nbr+b]=blk[0];
                    memcpy(qd+(r*nbr+b)*16,blk+1,16);
                }
            }
            CK(cudaMalloc(&o.d,o.expert_stride*ne));
            CK(cudaMemcpy(o.d,buf,o.expert_stride*ne,cudaMemcpyHostToDevice));
            free(buf);
            return o;
        }
        if(w->type==Q36_GGML_Q5_K||w->type==Q36_GGML_Q6_K){
            o.elayout=1;
            int ne=g_shard_e1-g_shard_e0;
            uint64_t qs_e=(uint64_t)M*K, ds_e=(uint64_t)M*(K/32)*2;
            o.expert_stride=qs_e+ds_e; o.row_bytes=0;
            uint8_t*buf=(uint8_t*)malloc(o.expert_stride*ne);
            uint64_t src_row=(uint64_t)(K/QK_K)*gguf_type_block_size(w->type);
            const uint8_t*src=(const uint8_t*)w->data;
            #pragma omp parallel for schedule(static)
            for(int xi=0;xi<ne;xi++){
                int ex=g_shard_e0+xi;
                float rowf[4096];                    /* K <= 2048 in this model */
                int8_t*qd=(int8_t*)(buf+(uint64_t)xi*o.expert_stride);
                uint16_t*dd=(uint16_t*)(buf+(uint64_t)xi*o.expert_stride+qs_e);
                for(uint64_t r=0;r<(uint64_t)M;r++){
                    q36_deq_row(w->type,src+((uint64_t)ex*M+r)*src_row,K,rowf);
                    h_q8_row(rowf,K,qd+r*K,dd+r*(K/32));
                }
            }
            CK(cudaMalloc(&o.d,o.expert_stride*ne));
            CK(cudaMemcpy(o.d,buf,o.expert_stride*ne,cudaMemcpyHostToDevice));
            free(buf);
            return o;
        }
    }
    /* 2D: [K,M]; 3D expert: [K,M,E] -> per-expert matrix K x M */
    o.K=(int)w->dims[0]; o.M=(int)w->dims[1];
    uint64_t be=gguf_type_block_elems(w->type), bs=gguf_type_block_size(w->type);
    o.row_bytes=(o.K/be)*bs;
    o.expert_stride=(uint64_t)o.M*o.row_bytes;
    /* Q8_0 2D matrices are repacked into split layout: [all quants][all f16
     * scales], both aligned, so the dot kernel gets coalesced 128-bit loads.
     * Same byte count as the 34-byte-block on-disk form, just reordered. */
    if(w->type==Q36_GGML_Q8_0 && w->n_dims==2){
        o.split=true;
        uint64_t nbr=(uint64_t)o.K/32;
        uint64_t qs_bytes=(uint64_t)o.M*o.K, d_bytes=(uint64_t)o.M*nbr*2;
        uint8_t*buf=(uint8_t*)malloc(qs_bytes+d_bytes);
        const uint8_t*src=(const uint8_t*)w->data;
        #pragma omp parallel for schedule(static)
        for(int64_t r=0;r<(int64_t)o.M;r++)
            for(uint64_t b=0;b<nbr;b++){
                const uint8_t*blk=src+((uint64_t)r*nbr+b)*34;
                memcpy(buf+qs_bytes+((uint64_t)r*nbr+b)*2, blk, 2);     /* f16 scale */
                memcpy(buf+(uint64_t)r*o.K+b*32, blk+2, 32);            /* 32 int8   */
            }
        CK(cudaMalloc(&o.d,qs_bytes+d_bytes));
        CK(cudaMemcpy(o.d,buf,qs_bytes+d_bytes,cudaMemcpyHostToDevice));
        free(buf);
        return o;
    }
    CK(cudaMalloc(&o.d,w->nbytes));
    CK(cudaMemcpy(o.d,w->data,w->nbytes,cudaMemcpyHostToDevice));
    return o;
}
/* Upload an F32 tensor to dst, widening BF16 on the host if needed (the MTP
 * GGUF stores block 40's ffn_gate_inp / ffn_gate_inp_shexp as BF16). */
static void upf32_to(const q36_weight*w,float*dst){
    if(w->type==Q36_GGML_BF16){
        uint64_t n=w->n_elem;
        float*hb=(float*)malloc(n*4);
        const uint16_t*src=(const uint16_t*)w->data;
        for(uint64_t i=0;i<n;i++){ uint32_t u=(uint32_t)src[i]<<16; memcpy(&hb[i],&u,4); }
        CK(cudaMemcpy(dst,hb,n*4,cudaMemcpyHostToDevice));
        free(hb);
    } else
        CK(cudaMemcpy(dst,w->data,w->nbytes,cudaMemcpyHostToDevice));
}
static float* upf32(const q36_weight*w){ /* upload as F32 (widen bf16) */
    float*d; CK(cudaMalloc(&d,w->n_elem*4)); upf32_to(w,d); return d;
}
/* upload two F32 tensors row-concatenated (fused-projection decode path) */
static float* upf32_cat(const q36_weight*a,const q36_weight*b){
    float*d; CK(cudaMalloc(&d,(a->n_elem+b->n_elem)*4));
    upf32_to(a,d); upf32_to(b,d+a->n_elem);
    return d;
}
/* forward decl; kernel defined below with the ops it needs */
static void mv(const dw*W,const float*x,float*y);

/* device mirror of one block */
typedef struct {
    bool is_attn;
    float *attn_norm,*post_norm;
    /* attention */
    dw q,k,v,o; float *q_norm,*k_norm;
    /* ssm */
    dw qkv,gate,ssm_out; float *conv1d,*ssm_a,*dt_bias,*ssm_norm,*alpha,*beta;
    /* moe */
    float *router; dw gate_exps,up_exps,down_exps;
    float *sh_gate_inp; dw sh_gate,sh_up,sh_down;
} dblock;

typedef struct {
    q36_model*m;
    block_q8_0*tok_embd; float*out_norm; dw output;
    dblock blk[Q36_N_LAYER];
    int ctx;
    /* scratch (device) */
    float *h,*x,*tmp,*q,*gate,*kbuf,*vbuf,*attn,*moe,*g512,*u512,*d2048,*shexp,*shg,*rlogits,*logits;
    int *d_topk; float *d_topw;
    /* attention KV cache, asymmetric quantized (split layouts, per-32 scales):
     * K = int8 + fp16 scale (protects RoPE'd retrieval),
     * V = E2M1 nibbles + UE8M0 scale (robust to 4-bit).
     * 2.5x less traffic/VRAM than fp16. */
    int se0,se1;   /* resident expert shard [se0,se1) -- full range single-GPU */
    int inc_shared; /* this device contributes the shared expert (multi-GPU) */
    int kvq;   /* 0 = fp16 KV (default, fastest <=32k), 1 = Q8-K/MXFP4-V */
    int8_t *Kc[Q36_N_LAYER]; __half *Ks[Q36_N_LAYER];
    uint8_t *Vc[Q36_N_LAYER],*Vs[Q36_N_LAYER];
    /* flash-decode partials: [n_head][max_chunks][head_dim] + (m,s) pairs */
    float *pacc; float2 *pms;
    /* CUDA-graph decode state: device-resident token/pos/argmax + exec */
    int *d_tok2,*d_pos,*d_argmax;
    cudaGraphExec_t gexec; int gchunks;
    /* multi-tenant: nslots independent sequences (KV/SSM state per slot).
     * Slot 0 aliases the classic single-sequence pointers, so every solo
     * path is untouched.  Batched decode processes all nslots each step. */
    int nslots, pf_slot;
    cudaGraphExec_t gexec_mt; int gchunks_mt, gcap_mt, gB_mt;
    float *hb,*xb,*tmpb,*qb,*gateb,*kbufb,*vbufb,*attnb,*moeb,
          *g512b,*u512b,*d2048b,*shexpb,*sh_ab,*sh_bb,*rlogitsb,*d_topwb;
    int *d_topkb,*d_tok_b,*d_pos_b,*d_out_b;
    unsigned long long *d_rng_b;
    float *mt_temp,*mt_topp; int *mt_topk;      /* host-side, baked at capture */
    float *hd_pv; int *hd_pi; int hd_nblk;
    /* parallel shared-expert branch (decode graph fork/join) */
    cudaStream_t s2; cudaEvent_t ev_in,ev_sh;
    float *sh_a,*sh_b;
    /* sampling (temp==0 -> greedy argmax path) */
    float s_temp,s_topp; int s_topk;
    unsigned long long *d_rng;
    float *d_pv; int *d_pi;
    float *d_nll;   /* perplexity scoring scratch (lazy) */
    /* chunked-prefill scratch (Q36_PF_CHUNK tokens) */
    int *pf_toks,*pf_topk,*pf_ecount,*pf_elist;
    uint8_t *pfXq,*pfXs; int8_t *pfXq8; float *pfXsf;
    float *pfH,*pfX,*pfQKV,*pfZ,*pfA,*pfB,*pfG,*pfBt,*pfCB,*pfO,*pfOUT,
          *pfQ,*pfGATE,*pfK,*pfV,*pfRL,*pf_topw,*pfGU_g,*pfGU_u,*pfDN,
          *pfMOE,*pfSH,*pfSU,*pfshg;
    /* ssm state: per ssm layer [num_v_heads][hk][hv] + conv history [conv_dim][K-1] */
    float *Sstate[Q36_N_LAYER],*convhist[Q36_N_LAYER];
    float expert_scale;
#define Q36_MTP_MAXB 4            /* max verify width (K<=3 drafts + 1) */
    /* ---- MTP (nextn) self-speculative decode (solo only) ------------------
     * mtp_db = block 40; eh_proj fuses [enorm(embed)|hnorm(prev hidden)];
     * the module has its OWN 1-layer KV cache (fp16, slot = entry index -
     * mtp_base, true rope positions).  Depth K (1..3): K chained draft
     * steps (step k>0 feeds the module its own z output), then ONE B=K+1
     * verify forward.  v* = verify scratch sized for B<=4; Ssnap/convsnap
     * hold K checkpoints (state after verify token j) so a partial accept
     * rolls back exactly to any prefix. */
    int has_mtp;
    int mtp_k;                    /* draft depth K (1..3): verify B = K+1 */
    dblock mtp_db; dw mtp_eh;
    float *mtp_enorm,*mtp_hnorm,*mtp_shn;
    int8_t *mtpKc; __half *mtpKs; uint8_t *mtpVc,*mtpVs;
    float *mtp_h,*mtp_emb,*mtp_cat,*mtp_z;
    /* device int block, ONE H2D per cycle:
     * [0..3]   verify tokens: tok0 + up to 3 drafts (draft argmaxes land here)
     * [4..7]   verify argmax outputs y_0..y_K          (d_vout)
     * [8..15]  4 (rope pos, kv slot) pairs: draft steps 0..2, pair 3 = fill (d_mm)
     * [16..19] verify positions P..P+K                 (d_vpos) */
    int *d_vtok,*d_vout,*d_mm,*d_vpos;
    float *vh,*vx,*vtmp,*vq,*vgate,*vkbuf,*vvbuf,*vattn,*vmoe,
          *vg512,*vu512,*vd2048,*vshexp,*vsh_a,*vsh_b,*vrl,*vtopw,*v_pv;
    int *vtopk,*v_pi,v_nblk;
    float *Ssnap,*convsnap; int ssm_sidx[Q36_N_LAYER];
    cudaGraphExec_t gx_v,gx_mn; int gvc[Q36_MTP_MAXB],gmch;  /* gx_v = full cycle (drafts+verify) */
    int mtp_base,mtp_primed;      /* first nextn entry index this conversation */
    long mtp_cycles,mtp_accepts;
} q36_engine;

/* dims for the delta-net */
enum { NKH=Q36_SSM_GROUPS, NVH=Q36_SSM_DT_RANK, HKD=Q36_SSM_STATE, HVD=Q36_SSM_INNER/Q36_SSM_DT_RANK,
       KEYD=HKD*NKH, VALD=HVD*NVH, CONVD=KEYD*2+VALD };

/* prefill chunk: sized so each MoE expert sees ~16 tokens/GEMM on average */
#define Q36_PF_CHUNK 2048
#define Q36_SAMPLE_KMAX 32

/* ---- gated DeltaNet conv (decode step) --------------------------------
 * causal depthwise conv over CONVD channels, kernel 4, then silu.  history is
 * [CONVD][3] most-recent-last. */
__global__ void k_dn_conv(const float*in,const float*w,float*hist,float*out,int convd,int K){Q36_GDS();
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=convd)return;
    float acc=0;
    /* window = hist[0],hist[1],hist[2], in[c] */
    float win[4]; win[0]=hist[c*3+0]; win[1]=hist[c*3+1]; win[2]=hist[c*3+2]; win[3]=in[c];
    /* conv1d weight is {K, convd} with ne0=K contiguous: element (t,c) at c*K+t. */
    for(int t=0;t<K;t++) acc+=w[c*K+t]*win[t];
    out[c]=silu(acc);
    /* shift history */
    hist[c*3+0]=win[1]; hist[c*3+1]=win[2]; hist[c*3+2]=win[3];
}

/* Path-specialized engine matvec (split-Q8_0 repacked/coalesced, split
 * FP8/NVFP4 heads, or raw row layouts).  The old runtime-dispatched
 * k_matvec_g inlined ALL dot cores into one kernel; the register
 * allocation for the union (255/thread, dominated by the k-quant row
 * dots) capped occupancy at ONE block per SM (16.67%, ncu-verified).
 * The path is known at launch, so each instantiation carries only its
 * own core.  Bit-identical math per path. */
enum { MV_MIXED=4, MV_NVFP4=3, MV_E4M3=2, MV_Q8SPLIT=1,
       MV_Q8ROW=10, MV_MXFP4ROW=11, MV_Q5KROW=12, MV_Q6KROW=13 };
template<int P>
__global__ void k_matvec_t(const void*W,const float*x,float*y,
                           int M,int K,uint64_t row_bytes){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    if(warp>=M)return;
    float r;
    if constexpr(P==MV_MIXED){ /* rows<R e4m3, tail nvfp4 (R in row_bytes) */
        int R=(int)row_bytes;
        if(warp<R){
            const uint8_t*qs=(const uint8_t*)W+(uint64_t)warp*K;
            const uint8_t*es=(const uint8_t*)W+(uint64_t)R*K+(uint64_t)warp*(K/32);
            r=dot_e4m3_split(qs,es,x,K,lane);
        } else {
            const uint8_t*base=(const uint8_t*)W+(uint64_t)R*K+(uint64_t)R*(K/32);
            int w2=warp-R, MT=M-R;
            const uint8_t*qs=base+(uint64_t)w2*(K/2);
            const uint8_t*es=base+(uint64_t)MT*(K/2)+(uint64_t)w2*(K/16);
            r=dot_nvfp4_split(qs,es,x,K,lane);
        }
    } else if constexpr(P==MV_NVFP4){
        const uint8_t*qs=(const uint8_t*)W+(uint64_t)warp*(K/2);
        const uint8_t*es=(const uint8_t*)W+(uint64_t)M*(K/2)+(uint64_t)warp*(K/16);
        r=dot_nvfp4_split(qs,es,x,K,lane);
    } else if constexpr(P==MV_E4M3){
        const uint8_t*qs=(const uint8_t*)W+(uint64_t)warp*K;
        const uint8_t*es=(const uint8_t*)W+(uint64_t)M*K+(uint64_t)warp*(K/32);
        r=dot_e4m3_split(qs,es,x,K,lane);
    } else if constexpr(P==MV_Q8SPLIT){
        const int8_t*qs=(const int8_t*)W+(uint64_t)warp*K;
        const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)M*K)+(uint64_t)warp*(K/32);
        r=dot_q8_0_split(qs,ds,x,K,lane);
    } else {
        const uint8_t*row=(const uint8_t*)W+(uint64_t)warp*row_bytes;
        if constexpr(P==MV_Q8ROW)    r=dot_q8_0_row((const block_q8_0*)row,x,K,lane);
        if constexpr(P==MV_MXFP4ROW) r=dot_mxfp4_row((const block_mxfp4*)row,x,K,lane);
        if constexpr(P==MV_Q5KROW)   r=dot_q5_K_row((const block_q5_K*)row,x,K,lane);
        if constexpr(P==MV_Q6KROW)   r=dot_q6_K_row((const block_q6_K*)row,x,K,lane);
    }
    if(lane==0)y[warp]=r;
}
static void mv_on(cudaStream_t s,const dw*W,const float*x,float*y){
    int sp=(W->elayout==5)?4:(W->elayout==4)?3:(W->elayout==3)?2:(W->split?1:0);
    int p = sp? sp :
        (W->type==Q36_GGML_Q8_0)? MV_Q8ROW :
        (W->type==Q36_GGML_MXFP4)? MV_MXFP4ROW :
        (W->type==Q36_GGML_Q5_K)? MV_Q5KROW : MV_Q6KROW;
    dim3 g(gr(W->M,8));
    switch(p){
        case MV_MIXED:   k_matvec_t<MV_MIXED>   <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        case MV_NVFP4:   k_matvec_t<MV_NVFP4>   <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        case MV_E4M3:    k_matvec_t<MV_E4M3>    <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        case MV_Q8SPLIT: k_matvec_t<MV_Q8SPLIT> <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        case MV_Q8ROW:   k_matvec_t<MV_Q8ROW>   <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        case MV_MXFP4ROW:k_matvec_t<MV_MXFP4ROW><<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        case MV_Q5KROW:  k_matvec_t<MV_Q5KROW>  <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
        default:         k_matvec_t<MV_Q6KROW>  <<<g,256,0,s>>>(W->d,x,y,W->M,W->K,W->row_bytes); break;
    }
}
static void mv(const dw*W,const float*x,float*y){ mv_on(0,W,x,y); }

/* Fused multi-tensor matvec: up to 3 split-Q8 weights sharing input x in one
 * launch.  Kills the per-kernel latency floor of small dependent matvecs
 * (q/k/v; ssm qkv+z; shared gate+up).  Pass M3=0 to use two. */
__global__ void k_matvec_multi3(const void*W1,int M1,float*y1,
                                const void*W2,int M2,float*y2,
                                const void*W3,int M3,float*y3,
                                const float*x,int K){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    const void*W; float*y; int M,r=warp;
    if(r<M1){ W=W1;y=y1;M=M1; }
    else if(r<M1+M2){ W=W2;y=y2;M=M2;r-=M1; }
    else if(r<M1+M2+M3){ W=W3;y=y3;M=M3;r-=M1+M2; }
    else return;
    const int8_t*qs=(const int8_t*)W+(uint64_t)r*K;
    const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)M*K)+(uint64_t)r*(K/32);
    float acc=dot_q8_0_split(qs,ds,x,K,lane);
    if(lane==0)y[r]=acc;
}

/* Fused batched LM head + argmax (greedy).  One pass over the output-head
 * weights serves all B tenants: each warp owns a vocab row, dots it against
 * every tenant's hidden vector (hidden fits in L1, reused), and keeps a
 * per-tenant running (max,idx).  Block-reduces to pv/pi[tenant][block];
 * k_head_argmax_p2 folds blocks.  Turns B serial 0.5GB head reads into ONE.
 * elayout: 3=e4m3, 4=nvfp4, 5=mixed(row<R e4m3 else nvfp4), else split-q8. */
__global__ void k_head_argmax_p1(int elayout,const void*W,const float*x,int Rmix,
                                 int M,int K,int B,uint64_t xs,float*pv,int*pi,int nblk){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31, wid=threadIdx.x>>5;
    int row=warp;
    float lmax[16]; int lidx[16];
    #pragma unroll
    for(int b=0;b<16;b++){ lmax[b]=-1e30f; lidx[b]=0; }
    if(row<M){
        const uint8_t*Wb=(const uint8_t*)W;
        for(int b=0;b<B;b++){
            const float*xv=x+(size_t)b*xs; float r;
            int el=elayout;
            if(el==5) el=(row<Rmix)?3:4;
            if(el==3){
                const uint8_t*qs; const uint8_t*es;
                if(elayout==5){ /* mixed: e4m3 block is rows [0,Rmix) */
                    qs=Wb+(uint64_t)row*K; es=Wb+(uint64_t)Rmix*K+(uint64_t)row*(K/32);
                } else { qs=Wb+(uint64_t)row*K; es=Wb+(uint64_t)M*K+(uint64_t)row*(K/32); }
                r=dot_e4m3_split(qs,es,xv,K,lane);
            } else if(el==4){
                const uint8_t*qs; const uint8_t*es;
                if(elayout==5){ int r2=row-Rmix, MT=M-Rmix;
                    const uint8_t*base=Wb+(uint64_t)Rmix*K+(uint64_t)Rmix*(K/32);
                    qs=base+(uint64_t)r2*(K/2); es=base+(uint64_t)MT*(K/2)+(uint64_t)r2*(K/16);
                } else { qs=Wb+(uint64_t)row*(K/2); es=Wb+(uint64_t)M*(K/2)+(uint64_t)row*(K/16); }
                r=dot_nvfp4_split(qs,es,xv,K,lane);
            } else {
                const int8_t*qs=(const int8_t*)Wb+(uint64_t)row*K;
                const __half*ds=(const __half*)(Wb+(uint64_t)M*K)+(uint64_t)row*(K/32);
                r=dot_q8_0_split(qs,ds,xv,K,lane);
            }
            if(lane==0 && b<16){ lmax[b]=r; lidx[b]=row; }
        }
    }
    /* per-tenant block reduction (lane 0 of each warp holds its row's value) */
    __shared__ float sv[8][16]; __shared__ int si[8][16];
    if(lane==0){
        #pragma unroll
        for(int b=0;b<16;b++){ sv[wid][b]=lmax[b]; si[wid][b]=lidx[b]; }
    }
    __syncthreads();
    if(threadIdx.x<B){
        int b=threadIdx.x; float mv=-1e30f; int mi=0;
        for(int w=0;w<8;w++) if(sv[w][b]>mv){ mv=sv[w][b]; mi=si[w][b]; }
        pv[(size_t)b*nblk+blockIdx.x]=mv; pi[(size_t)b*nblk+blockIdx.x]=mi;
    }
}
/* 256 threads (one warp over 31k partials cost ~3% of decode) and an
 * EXPLICIT lowest-index tie-break: the winner is a pure function of the
 * partial values, independent of B and thread mapping, so plain decode and
 * the MTP verify head agree on exact logit ties by construction. */
__global__ void k_head_argmax_p2(const float*pv,const int*pi,int nblk,int B,int*out){Q36_GDS();
    int b=blockIdx.x;
    float mv=-1e30f; int mi=INT_MAX;
    for(int i=threadIdx.x;i<nblk;i+=blockDim.x){
        float v=pv[(size_t)b*nblk+i]; int ix=pi[(size_t)b*nblk+i];
        if(v>mv||(v==mv&&ix<mi)){ mv=v; mi=ix; }
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1){
        float ov=__shfl_down_sync(0xffffffff,mv,o); int oi=__shfl_down_sync(0xffffffff,mi,o);
        if(ov>mv||(ov==mv&&oi<mi)){ mv=ov; mi=oi; }
    }
    __shared__ float sv[32]; __shared__ int si[32];
    int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    if(lane==0){ sv[wid]=mv; si[wid]=mi; }
    __syncthreads();
    if(threadIdx.x==0){
        int nw=(blockDim.x+31)/32;
        for(int w=1;w<nw;w++) if(sv[w]>sv[0]||(sv[w]==sv[0]&&si[w]<si[0])){ sv[0]=sv[w]; si[0]=si[w]; }
        out[b]=si[0];
    }
}

/* Batched-across-tenants routed expert GEMVs.  grid.y = B*NU: tenant =
 * y/NU, local expert slot = y%NU.  One launch replaces the per-tenant loop
 * so the scheduler fills the GPU and each expert's weight read serves every
 * tenant that routed to it (L2 reuse when experts overlap).  NUp passed so
 * the kernel can decode tenant/slot. */
__global__ void k_expert_gemv2_bt(int elg,const void*Wg,uint64_t sg,
                               int elu,const void*Wu,uint64_t su,int se0,int se1,
                               const int*topk,const float*x,float*yg,float*yu,
                               int M,int K,int NUp,uint64_t xs){Q36_GDS();
    int gy=blockIdx.y, which=blockIdx.z;
    int tenant=gy/NUp, slot=gy;             /* topk/y indexed by GLOBAL slot */
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    if(warp>=M)return;
    int ex=topk[slot];
    float*y=which?yu:yg;
    if(ex<se0||ex>=se1){ if(lane==0)y[(size_t)slot*M+warp]=0.f; return; }
    const uint8_t*base=(const uint8_t*)(which?Wu:Wg)+(uint64_t)(ex-se0)*(which?su:sg);
    const float*xv=x+(size_t)tenant*xs;
    int el=which?elu:elg; float r;
    if(el==2){
        const uint8_t*qs=base+(uint64_t)warp*(K/2);
        const uint8_t*es=base+(uint64_t)M*(K/2)+(uint64_t)warp*(K/32);
        r=dot_mxfp4_split(qs,es,xv,K,lane);
    } else {
        const int8_t*qs=(const int8_t*)base+(uint64_t)warp*K;
        const __half*ds=(const __half*)(base+(uint64_t)M*K)+(uint64_t)warp*(K/32);
        r=dot_q8_0_split(qs,ds,xv,K,lane);
    }
    if(lane==0)y[(size_t)slot*M+warp]=r;
}
/* batched down proj: x is per-(tenant,slot) swiglu output */
__global__ void k_expert_gemv_bt(int elayout,const void*W,uint64_t estride,
                              int se0,int se1,const int*topk,const float*x,
                              uint64_t x_stride,float*y,int M,int K){Q36_GDS();
    int slot=blockIdx.y;                     /* GLOBAL slot 0..B*NU-1 */
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    if(warp>=M)return;
    int ex=topk[slot];
    if(ex<se0||ex>=se1){ if(lane==0)y[(size_t)slot*M+warp]=0.f; return; }
    const float*xv=x+(uint64_t)slot*x_stride;
    const uint8_t*base=(const uint8_t*)W+(uint64_t)(ex-se0)*estride;
    float r;
    if(elayout==2){
        const uint8_t*qs=base+(uint64_t)warp*(K/2);
        const uint8_t*es=base+(uint64_t)M*(K/2)+(uint64_t)warp*(K/32);
        r=dot_mxfp4_split(qs,es,xv,K,lane);
    } else {
        const int8_t*qs=(const int8_t*)base+(uint64_t)warp*K;
        const __half*ds=(const __half*)(base+(uint64_t)M*K)+(uint64_t)warp*(K/32);
        r=dot_q8_0_split(qs,ds,xv,K,lane);
    }
    if(lane==0)y[(size_t)slot*M+warp]=r;
}

/* Batched (multi-tenant) GEMV variants: identical math, but each weight
 * row is dotted against B activation vectors while it sits in L1 -- the
 * weight read (the decode bottleneck) amortizes across tenants.  The
 * measured expert-union curve validated this model. */
__global__ void k_matvec_multi3_b(const void*W1,int M1,float*y1,uint64_t ys1,
                                  const void*W2,int M2,float*y2,uint64_t ys2,
                                  const void*W3,int M3,float*y3,uint64_t ys3,
                                  const float*x,int K,int B,uint64_t xs){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    const void*W; float*y; int M,r=warp; uint64_t ys;
    if(r<M1){ W=W1;y=y1;M=M1;ys=ys1; }
    else if(r<M1+M2){ W=W2;y=y2;M=M2;r-=M1;ys=ys2; }
    else if(r<M1+M2+M3){ W=W3;y=y3;M=M3;r-=M1+M2;ys=ys3; }
    else return;
    const int8_t*qs=(const int8_t*)W+(uint64_t)r*K;
    const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)M*K)+(uint64_t)r*(K/32);
    for(int b=0;b<B;b++){
        float acc=dot_q8_0_split(qs,ds,x+(size_t)b*xs,K,lane);
        if(lane==0)y[(size_t)b*ys+r]=acc;
    }
}
__global__ void k_matvec_gb(int split,const void*W,const float*x,float*y,
                            int M,int K,int B,uint64_t xs,uint64_t ys){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    if(warp>=M)return;
    /* split-q8 only: every batched dense tensor is elayout 1 */
    const int8_t*qs=(const int8_t*)W+(uint64_t)warp*K;
    const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)M*K)+(uint64_t)warp*(K/32);
    (void)split;
    for(int b=0;b<B;b++){
        float r=dot_q8_0_split(qs,ds,x+(size_t)b*xs,K,lane);
        if(lane==0)y[(size_t)b*ys+warp]=r;
    }
}
__global__ void k_matvec_f32_b(const float*W,const float*x,float*y,int M,int K,
                               int B,uint64_t xs,uint64_t ys){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    if(warp>=M)return;
    const float*row=W+(uint64_t)warp*K;
    for(int b=0;b<B;b++){
        const float*xv=x+(size_t)b*xs;
        float acc=0;
        for(int i=lane;i<K;i+=WARP) acc+=row[i]*xv[i];
        acc=warpsum(acc);
        if(lane==0)y[(size_t)b*ys+warp]=acc;
    }
}

/* h += a (residual), then y = rmsnorm(h)*w -- one launch instead of two. */
__global__ void k_add_rmsnorm(float*h,const float*a,const float*w,float*y,int n,float eps){Q36_GDS();
    __shared__ float part[32]; __shared__ float ss;
    float loc=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){ float v=h[i]+a[i]; h[i]=v; loc+=v*v; }
    loc=warpsum(loc);
    int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    if(lane==0)part[wid]=loc; __syncthreads();
    if(threadIdx.x==0){float s=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)s+=part[i];ss=rsqrtf(s/n+eps);}
    __syncthreads();
    for(int i=threadIdx.x;i<n;i+=blockDim.x) y[i]=h[i]*ss*w[i];
}

/* ---- device-side MoE (no host round-trips) ----------------------------
 * Indirect expert GEMV: expert id comes from d_topk[blockIdx.y] on the GPU,
 * so routing never touches the host.  One launch covers all 8 slots. */
__global__ void k_expert_gemv(int elayout,const void*W,uint64_t expert_stride,
                              int se0,int se1,
                              const int*topk,const float*x,uint64_t x_stride,float*y,
                              int M,int K){Q36_GDS();
    /* dual-row warps, same rationale as k_expert_gemv2 (M even: D=2048) */
    int slot=blockIdx.y;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    int Mh=M>>1;
    if(warp>=Mh)return;
    int r0=warp, r1=warp+Mh;
    int ex=topk[slot];
    if(ex<se0||ex>=se1){
        if(lane==0){ y[(size_t)slot*M+r0]=0.f; y[(size_t)slot*M+r1]=0.f; }
        return;
    }
    x+= (uint64_t)slot*x_stride;   /* 0 = same input for all slots (gate/up) */
    const uint8_t*base=(const uint8_t*)W+(uint64_t)(ex-se0)*expert_stride;
    float v0,v1;
    if(elayout==2){        /* mxfp4-split: [qs M*K/2][es M*K/32] per expert */
        const uint8_t*qs=base, *es=base+(uint64_t)M*(K/2);
        dot_mxfp4_split_x2(qs+(uint64_t)r0*(K/2),es+(uint64_t)r0*(K/32),
                           qs+(uint64_t)r1*(K/2),es+(uint64_t)r1*(K/32),
                           x,K,lane,&v0,&v1);
    } else {               /* q8_0-split: [qs M*K][ds M*K/16] per expert */
        const int8_t*qs=(const int8_t*)base;
        const __half*ds=(const __half*)(base+(uint64_t)M*K);
        dot_q8_0_split_x2(qs+(uint64_t)r0*K,ds+(uint64_t)r0*(K/32),
                          qs+(uint64_t)r1*K,ds+(uint64_t)r1*(K/32),
                          x,K,lane,&v0,&v1);
    }
    if(lane==0){ y[(size_t)slot*M+r0]=v0; y[(size_t)slot*M+r1]=v1; }
}
/* silu(g)*u over all slots at once. */
__global__ void k_silu_mul_slots(float*g,const float*u,int n){Q36_GDS();
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) g[i]=silu(g[i])*u[i];
}
/* gate+up expert GEMVs fused into one launch via grid.z (independent).
 * One warp owns rows (w, w+M/2): the dual-row dots keep 4 weight loads in
 * flight per lane, which the ~one-wave GEMV grid needs to reach DRAM
 * bandwidth (occupancy can't provide the parallelism here).  M is even
 * (FF=512).  Results are bit-identical to the single-row form. */
__global__ void k_expert_gemv2(int elg,const void*Wg,uint64_t sg,
                               int elu,const void*Wu,uint64_t su,int se0,int se1,
                               const int*topk,const float*x,float*yg,float*yu,
                               int M,int K){Q36_GDS();
    int slot=blockIdx.y, which=blockIdx.z;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    int Mh=M>>1;
    if(warp>=Mh)return;
    int r0=warp, r1=warp+Mh;
    int ex=topk[slot];
    float*y = which? yu:yg;
    if(ex<se0||ex>=se1){                      /* non-resident: zero contribution */
        if(lane==0){ y[(size_t)slot*M+r0]=0.f; y[(size_t)slot*M+r1]=0.f; }
        return;
    }
    int el = which? elu:elg;
    const uint8_t*base=(const uint8_t*)(which?Wu:Wg)+(uint64_t)(ex-se0)*(which?su:sg);
    float v0,v1;
    if(el==2){
        const uint8_t*qs=base, *es=base+(uint64_t)M*(K/2);
        dot_mxfp4_split_x2(qs+(uint64_t)r0*(K/2),es+(uint64_t)r0*(K/32),
                           qs+(uint64_t)r1*(K/2),es+(uint64_t)r1*(K/32),
                           x,K,lane,&v0,&v1);
    } else {
        const int8_t*qs=(const int8_t*)base;
        const __half*ds=(const __half*)(base+(uint64_t)M*K);
        dot_q8_0_split_x2(qs+(uint64_t)r0*K,ds+(uint64_t)r0*(K/32),
                          qs+(uint64_t)r1*K,ds+(uint64_t)r1*(K/32),
                          x,K,lane,&v0,&v1);
    }
    if(lane==0){ y[(size_t)slot*M+r0]=v0; y[(size_t)slot*M+r1]=v1; }
}
/* moe[i] = sum_slot w[slot]*down[slot][i] + sigmoid(*shgate)*shared[i] */
__global__ void k_moe_combine(float*moe,const float*down,const float*w,
                              const float*shared,const float*shgate,int M,int nslots){Q36_GDS();
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M)return;
    float acc=0;
    for(int s=0;s<nslots;s++) acc+=w[s]*down[(size_t)s*M+i];
    moe[i]=acc+sigm(*shgate)*shared[i];
}

/* ---- flash-decode attention (split-K over the sequence) ----------------
 * Batch-1 decode attention is bandwidth-bound over the KV cache, so the
 * kernel splits the sequence into chunks (grid.x), each block computing a
 * partial online-softmax accumulation; k_attn_merge combines partials with
 * the log-sum-exp trick.  KV lives in fp16.  One warp owns one position at a
 * time (lane = 8 contiguous dims of head_dim 256 -> coalesced half loads). */
#define Q36_ATTN_CHUNK  512   /* per-(q-head) flash-decode (default)       */
#define Q36_ATTN_CHUNK2 256   /* mma/GQA decode need-granularity           */
/* Flash-decode kernel choice.  Default = per-(q-head) k_attn_partial.
 * Q36_ATTN2=1 selects the GQA-shared kernel -- kept as a MEASURED DEAD END:
 * eliminating the 8x KV re-read changed nothing at 90k (154.6 vs 154.5) and
 * lost at mid depth (208 vs 244 @16k, pinned GPU).  Decode attention is
 * INSTRUCTION-ISSUE bound (per-position warpsum/softmax chain), not
 * bandwidth bound -- kv-quant reads 40% of the bytes for only +9% too.
 * The designed fix is tensor-core mma decode (GQA group as the mma M dim),
 * which collapses the per-position instruction count ~10x. */
static int attn_chunk(void){
    /* 256 = need/graph granularity for the mma + GQA kernels (both re-chunk
     * dynamically); Q36_ATTN1=1 restores the fixed-512 scalar-only setup */
    static int c=-1;
    if(c<0) c=getenv("Q36_ATTN1")?Q36_ATTN_CHUNK:Q36_ATTN_CHUNK2;
    return c;
}

/* Decode kernels read position/token from DEVICE memory so a captured CUDA
 * graph replays with fresh values (grids are sized for the bucket maximum;
 * out-of-range chunks exit early). */
__device__ __forceinline__ uint8_t e2m1_encode(float x){
    uint8_t s=(x<0.f)?8:0; float m=fabsf(x);
    uint8_t c;
    if(m<0.25f)c=0; else if(m<0.75f)c=1; else if(m<1.25f)c=2; else if(m<1.75f)c=3;
    else if(m<2.5f)c=4; else if(m<3.5f)c=5; else if(m<5.f)c=6; else c=7;
    return c?(uint8_t)(s|c):0;
}
/* quantize one position's K (int8/f16-scale) and V (e2m1/ue8m0) per 32-block.
 * one warp per 32-dim block: warps 0..15 = K, 16..31 = V. */
__global__ void k_kv_append(const float*k,const float*v,int8_t*Kc,__half*Ks,
                            uint8_t*Vc,uint8_t*Vs,const int*posp,int kv_dim,int kvq){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    int nb=kv_dim/32;
    if(warp>=2*nb)return;
    int pos=*posp;
    if(!kvq){   /* fp16 mode: plain converts */
        if(warp<nb) ((__half*)Kc)[(size_t)pos*kv_dim+warp*32+lane]=__float2half(k[warp*32+lane]);
        else { int b=warp-nb; ((__half*)Vc)[(size_t)pos*kv_dim+b*32+lane]=__float2half(v[b*32+lane]); }
        return;
    }
    if(warp<nb){
        float x=k[warp*32+lane], a=fabsf(x);
        #pragma unroll
        for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
        float d=a/127.f, id=(a>0.f)?127.f/a:0.f;
        int q=(int)lrintf(x*id);
        Kc[(size_t)pos*kv_dim+warp*32+lane]=(int8_t)(q>127?127:(q<-127?-127:q));
        if(lane==0)Ks[(size_t)pos*nb+warp]=__float2half(d);
    } else {
        int b=warp-nb;
        float x=v[b*32+lane], a=fabsf(x);
        #pragma unroll
        for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
        int e2=(a>0.f)?(int)ceilf(log2f(a/6.f)):-127;
        if(e2<-127)e2=-127;
        float inv=__int_as_float((127-e2)<<23);
        uint8_t code=e2m1_encode(x*inv);
        /* byte k holds dims (2k,2k+1) -- matches the consecutive-pair reads */
        uint8_t hi=__shfl_down_sync(0xffffffff,(unsigned)code,1)&0xF;
        if((lane&1)==0) Vc[(size_t)pos*(kv_dim/2)+b*16+(lane>>1)]=(uint8_t)(code|(hi<<4));
        if(lane==0) Vs[(size_t)pos*nb+b]=(uint8_t)(127+e2);
    }
}
__global__ void k_embed_dt(const block_q8_0*emb,const int*tokp,float*h,int D){Q36_GDS();
    const block_q8_0*row=emb+(size_t)(*tokp)*(D/32);
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=D)return;
    const block_q8_0*b=row+i/32; h[i]=q36_fp16(b->d)*b->qs[i%32];
}
__global__ void k_rope_p(float*x,int head_dim,int rot_dim,const int*posp,float base){Q36_GDS();
    int h=blockIdx.x, i=threadIdx.x; if(i>=rot_dim/2)return;
    float*xh=x+(size_t)h*head_dim;
    float inv=__powf(base,-2.0f*i/rot_dim), ang=(float)(*posp)*inv;
    float c=__cosf(ang),s=__sinf(ang);
    float a=xh[i],b=xh[i+rot_dim/2];
    xh[i]=a*c-b*s; xh[i+rot_dim/2]=a*s+b*c;
}

__global__ void k_attn_partial(const float*Q,const int8_t*Kc,const __half*Ks,
                               const uint8_t*Vc,const uint8_t*Vs,
                               float*pacc,float2*pms,float*final_out,
                               int n_head,int n_head_kv,int head_dim,const int*posp,
                               float scale,int kvq){Q36_GDS();
    int h=blockIdx.y, chunk=blockIdx.x, nchunks=gridDim.x;
    int seqlen=*posp+1;
    if(chunk*Q36_ATTN_CHUNK>=seqlen) return;   /* beyond live prefix */
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int nw=blockDim.x>>5;
    /* byte -> (lo,hi) e2m1 value pair LUT in smem: 1 read replaces ~12 ALU */
    __shared__ float2 vlut[256];
    for(int b2=threadIdx.x;b2<256;b2+=blockDim.x)
        vlut[b2]=make_float2(mxfp4_val(b2&0xF),mxfp4_val(b2>>4));
    __syncthreads();
    int t0=chunk*Q36_ATTN_CHUNK;
    int t1=t0+Q36_ATTN_CHUNK; if(t1>seqlen)t1=seqlen;
    int kvh=h/(n_head/n_head_kv), kv_dim=n_head_kv*head_dim;
    const float*q=Q+(size_t)h*head_dim;
    /* per-warp online softmax state; 8 dims per lane in registers */
    float m=-1e30f,s=0.f,racc[8];
    #pragma unroll
    for(int e2=0;e2<8;e2++) racc[e2]=0.f;
    float ql[8];
    #pragma unroll
    for(int e2=0;e2<8;e2++) ql[e2]=q[lane*8+e2];
    int nb=kv_dim/32, blk=kvh*(head_dim/32)+(lane>>2);   /* per-32 scale index */
    for(int t=t0+warp;t<t1;t+=nw){
        float dot=0, mn,f,p;
        if(kvq){
            uint2 kq=*(const uint2*)(Kc+(size_t)t*kv_dim+kvh*head_dim+lane*8);
            const int8_t*k8=(const int8_t*)&kq;
            float kd=__half2float(Ks[(size_t)t*nb+blk]);
            #pragma unroll
            for(int e2=0;e2<8;e2++) dot+=ql[e2]*(float)k8[e2];
            dot=warpsum_xor(dot*kd)*scale;
            mn=fmaxf(m,dot); f=__expf(m-mn); p=__expf(dot-mn);
            unsigned vq=*(const unsigned*)(Vc+(size_t)t*(kv_dim/2)+kvh*(head_dim/2)+lane*4);
            const uint8_t*v4=(const uint8_t*)&vq;
            float vd=q36_e8m0_half(Vs[(size_t)t*nb+blk]);
            #pragma unroll
            for(int e2=0;e2<4;e2++){
                float2 lv=vlut[v4[e2]];
                racc[e2*2]  =racc[e2*2]*f  +p*vd*lv.x;
                racc[e2*2+1]=racc[e2*2+1]*f+p*vd*lv.y;
            }
        } else {
            const __half*krow=(const __half*)Kc+(size_t)t*kv_dim+kvh*head_dim;
            uint4 kq=*(const uint4*)(krow+lane*8);
            const __half*kh=(const __half*)&kq;
            #pragma unroll
            for(int e2=0;e2<8;e2++) dot+=ql[e2]*__half2float(kh[e2]);
            dot=warpsum_xor(dot)*scale;
            mn=fmaxf(m,dot); f=__expf(m-mn); p=__expf(dot-mn);
            const __half*vrow=(const __half*)Vc+(size_t)t*kv_dim+kvh*head_dim;
            uint4 vq=*(const uint4*)(vrow+lane*8);
            const __half*vh=(const __half*)&vq;
            #pragma unroll
            for(int e2=0;e2<8;e2++) racc[e2]=racc[e2]*f+p*__half2float(vh[e2]);
        }
        s=s*f+p; m=mn;
    }
    /* combine the block's warps in shared memory */
    __shared__ float sm[8],ss[8],sa[8][256];
    if(lane==0){ sm[warp]=m; ss[warp]=s; }
    #pragma unroll
    for(int e2=0;e2<8;e2++) sa[warp][lane*8+e2]=racc[e2];
    __syncthreads();
    if(threadIdx.x<head_dim){
        int d=threadIdx.x;
        float bm=-1e30f;
        for(int w=0;w<nw;w++) bm=fmaxf(bm,sm[w]);
        float bs=0,ba=0;
        for(int w=0;w<nw;w++){ float f=__expf(sm[w]-bm); bs+=ss[w]*f; ba+=sa[w][d]*f; }
        if(final_out&&nchunks==1){   /* single chunk: skip the merge kernel */
            final_out[(size_t)h*head_dim+d]=ba/bs;
        } else {
            pacc[((size_t)h*nchunks+chunk)*head_dim+d]=ba;
            if(d==0) pms[(size_t)h*nchunks+chunk]=make_float2(bm,bs);
        }
    }
}

/* GQA-shared flash-decode: block = (64-position chunk, KV head); each of the
 * 8 warps owns ONE q-head of the group, so all warps stream the SAME K/V
 * rows and L1 dedupes them.  The per-(q-head) block version (k_attn_partial,
 * kept behind Q36_ATTN1=1) re-read every KV row up to 8x from L2/DRAM:
 * ~2.9ms/token of depth cost at 90k = the decode-at-depth gap vs llama.cpp.
 * A warp owns its head outright -- no cross-warp combine phase at all. */
__global__ void k_attn_partial2(const float*Q,const int8_t*Kc,const __half*Ks,
                                const uint8_t*Vc,const uint8_t*Vs,
                                float*pacc,float2*pms,float*final_out,
                                int n_head,int n_head_kv,int head_dim,const int*posp,
                                float scale,int kvq){Q36_GDS();
    int kvh=blockIdx.y, chunk=blockIdx.x, nchunks=gridDim.x;
    int seqlen=*posp+1;
    /* dynamic re-chunk: split the LIVE prefix evenly over the captured grid
     * (the graph never shrinks, so a fixed stride would idle most blocks on
     * short prefixes after a long-context capture) */
    int per=(seqlen+nchunks-1)/nchunks;
    if(chunk*per>=seqlen) return;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int h=kvh*(n_head/n_head_kv)+warp;          /* this warp's q-head */
    __shared__ float2 vlut[256];
    if(kvq){
        for(int b2=threadIdx.x;b2<256;b2+=blockDim.x)
            vlut[b2]=make_float2(mxfp4_val(b2&0xF),mxfp4_val(b2>>4));
        __syncthreads();
    }
    int t0=chunk*per;
    int t1=t0+per; if(t1>seqlen)t1=seqlen;
    int kv_dim=n_head_kv*head_dim;
    const float*q=Q+(size_t)h*head_dim;
    /* 4 independent online-softmax streams per warp (positions t0+j, +4, ..)
     * keep 4 KV rows in flight: the 1-stream version had ~1KB of MLP per
     * warp and lost to the old kernel's stride-8 interleave at mid depth */
    enum{NS=4};
    float m[NS],s[NS],racc[NS][8],ql[8];
    #pragma unroll
    for(int j=0;j<NS;j++){ m[j]=-1e30f; s[j]=0.f;
        #pragma unroll
        for(int e2=0;e2<8;e2++) racc[j][e2]=0.f; }
    #pragma unroll
    for(int e2=0;e2<8;e2++) ql[e2]=q[lane*8+e2];
    int nb=kv_dim/32, blk=kvh*(head_dim/32)+(lane>>2);
    for(int t=t0;t<t1;t+=NS){
        #pragma unroll
        for(int j=0;j<NS;j++){
            int tt=t+j;
            if(tt>=t1) continue;
            float dot=0, mn,f,p;
            if(kvq){
                uint2 kq=*(const uint2*)(Kc+(size_t)tt*kv_dim+kvh*head_dim+lane*8);
                const int8_t*k8=(const int8_t*)&kq;
                float kd=__half2float(Ks[(size_t)tt*nb+blk]);
                #pragma unroll
                for(int e2=0;e2<8;e2++) dot+=ql[e2]*(float)k8[e2];
                dot=warpsum_xor(dot*kd)*scale;
                mn=fmaxf(m[j],dot); f=__expf(m[j]-mn); p=__expf(dot-mn);
                unsigned vq=*(const unsigned*)(Vc+(size_t)tt*(kv_dim/2)+kvh*(head_dim/2)+lane*4);
                const uint8_t*v4=(const uint8_t*)&vq;
                float vd=q36_e8m0_half(Vs[(size_t)tt*nb+blk]);
                #pragma unroll
                for(int e2=0;e2<4;e2++){
                    float2 lv=vlut[v4[e2]];
                    racc[j][e2*2]  =racc[j][e2*2]*f  +p*vd*lv.x;
                    racc[j][e2*2+1]=racc[j][e2*2+1]*f+p*vd*lv.y;
                }
            } else {
                const __half*krow=(const __half*)Kc+(size_t)tt*kv_dim+kvh*head_dim;
                uint4 kq=*(const uint4*)(krow+lane*8);
                const __half*kh=(const __half*)&kq;
                #pragma unroll
                for(int e2=0;e2<8;e2++) dot+=ql[e2]*__half2float(kh[e2]);
                dot=warpsum_xor(dot)*scale;
                mn=fmaxf(m[j],dot); f=__expf(m[j]-mn); p=__expf(dot-mn);
                const __half*vrow=(const __half*)Vc+(size_t)tt*kv_dim+kvh*head_dim;
                uint4 vq=*(const uint4*)(vrow+lane*8);
                const __half*vh=(const __half*)&vq;
                #pragma unroll
                for(int e2=0;e2<8;e2++) racc[j][e2]=racc[j][e2]*f+p*__half2float(vh[e2]);
            }
            s[j]=s[j]*f+p; m[j]=mn;
        }
    }
    /* fold the streams (registers only) */
    float mm=fmaxf(fmaxf(m[0],m[1]),fmaxf(m[2],m[3]));
    float ss=0.f, outv[8];
    #pragma unroll
    for(int e2=0;e2<8;e2++) outv[e2]=0.f;
    #pragma unroll
    for(int j=0;j<NS;j++){
        float f=__expf(m[j]-mm); ss+=s[j]*f;
        #pragma unroll
        for(int e2=0;e2<8;e2++) outv[e2]+=racc[j][e2]*f;
    }
    if(final_out&&nchunks==1){                  /* single chunk: no merge pass */
        float inv=1.f/ss;
        #pragma unroll
        for(int e2=0;e2<8;e2++)
            final_out[(size_t)h*head_dim+lane*8+e2]=outv[e2]*inv;
    } else {
        #pragma unroll
        for(int e2=0;e2<8;e2++)
            pacc[((size_t)h*nchunks+chunk)*head_dim+lane*8+e2]=outv[e2];
        if(lane==0) pms[(size_t)h*nchunks+chunk]=make_float2(mm,ss);
    }
}

/* 16B async global->shared copy (.cg: KV streams through L2, skip L1) */
__device__ __forceinline__ void q36_cpa16(void*dst,const void*src){
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
        ::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));
}

/* Tensor-core mma flash-decode (fp16 KV): the per-(position,head) scalar
 * kernels are INSTRUCTION-ISSUE bound (~30 inst/KB: warpsum+softmax chain;
 * measured: GQA-dedup and byte savings change nothing).  Here the 8-head
 * GQA group is the mma M dimension (rows 8..15 idle), positions are N, and
 * the k_attn_pf3 machinery transplants: smem KV tiles, online softmax on
 * the C fragments, P repacked in-register, V B-frags via ldmatrix.trans.
 * (First cut read KV fragments STRAIGHT from global: 121 t/s @90k -- the
 * scattered 2B loads cost up to 8 L1 sectors per instruction.  Staging
 * through smem is not optional for mma operand shapes.)
 * Grid (nchunks, NKV), block = 4 warps sharing each 32-position tile; warp
 * w owns n-tile w (positions tt+w*8..+7) and carries its own online
 * softmax, writing partial slot chunk*4+w; k_attn_merge(chunk_sz=-1)
 * folds slots (a slot is live iff w*8 < the chunk's live length). */
/* MINB=3 forces ~200B of spills but wins when the grid can actually fill 3
 * blocks/SM (>=~512 blocks, i.e. deep contexts); MINB=2 is spill-free and
 * wins below that.  attn_layer dispatches on the grid size. */
template<int MINB>
__global__ void __launch_bounds__(128,MINB) k_attn_dec_mma(
        const float*Q,const __half*Kh,const __half*Vh,
        float*pacc,float2*pms,int n_head,int n_head_kv,int head_dim,
        const int*posp,float scale){Q36_GDS();
    enum{HD=Q36_HEAD_DIM};
    /* KP=HD (no padding) + XOR swizzle: 16B chunk j of row r lives at
     * j^(r&7).  Padding to 264 halves cost 1.8KB/buffer and capped the
     * kernel at 2 blocks/SM; 32KB tiles + the register diet below fit 3.
     * All fragment loads verified conflict-free under the swizzle. */
    __shared__ __align__(16) __half sK[32][HD];
    __shared__ __align__(16) __half sV[32][HD];
    int kvh=blockIdx.y, chunk=blockIdx.x, nch=gridDim.x;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31, g=lane>>2, t4=lane&3;
    int seqlen=*posp+1;
    int per=(seqlen+nch-1)/nch;
    int c0=chunk*per, cend=c0+per; if(cend>seqlen)cend=seqlen;
    if(c0>=seqlen) return;
    int slot=chunk*4+warp, nslots=nch*4;
    int kv_dim=n_head_kv*head_dim;
    /* Register diet (255 -> ~140 regs = 3 blocks/SM): the A-fragment rows
     * 8..15 are architecturally zero, so their operand halves share ONE
     * zero register, and their C/D accumulator halves alias TWO dummies
     * (D_row = 0*B + C_row keeps them 0 forever -- no cross-mma pollution). */
    unsigned zr=0u;
    float dz0=0.f,dz1=0.f;
    unsigned qr[16][2];
    {
        const float*q0=Q+(size_t)(kvh*8+g)*HD;
        #pragma unroll
        for(int ks=0;ks<16;ks++){
            int c=ks*16+t4*2;
            float2 x0=*(const float2*)(q0+c);
            float2 y0=*(const float2*)(q0+c+8);
            __half2 hl=__floats2half2_rn(x0.x*scale,x0.y*scale);
            __half2 hh=__floats2half2_rn(y0.x*scale,y0.y*scale);
            qr[ks][0]=*(unsigned*)&hl;
            qr[ks][1]=*(unsigned*)&hh;
        }
    }
    /* ldmatrix.x4.trans source for PV B-frags: 8 k-rows x 4 8-dim blocks */
    int vrow=lane&7, vcol8=lane>>3;
    float m0=-1e28f,s0=0.f;            /* row g state (this warp's stripe) */
    unsigned pa0=0;                    /* P fragment, carried softmax -> PV */
    float oa[32][2];
    #pragma unroll
    for(int n=0;n<32;n++){oa[n][0]=0.f;oa[n][1]=0.f;}
    /* intra-tile cp.async pipeline (pf3 scheme): V(tt) streams during the
     * S phase, K(tt+32) during PV (sync staging measured 63% memory-SoL at
     * 16% compute).  Chunk j of row p lands swizzled at j^(p&7). */
    #define Q36_DEC_CPA(dst,src,tt_) do{ \
        for(int i2=threadIdx.x;i2<32*(HD/8);i2+=(int)blockDim.x){ \
            int p_=i2>>5, cb_=i2&31; \
            if((tt_)+p_<cend) \
                q36_cpa16(&dst[p_][(cb_^(p_&7))*8],src+(size_t)((tt_)+p_)*kv_dim+kvh*HD+cb_*8); \
        }}while(0)
    Q36_DEC_CPA(sK,Kh,c0);
    asm volatile("cp.async.commit_group;\n");
    for(int tt=c0;tt<cend;tt+=32){
        int tlen=cend-tt; if(tlen>32)tlen=32;
        int have_next=(tt+32<cend);
        Q36_DEC_CPA(sV,Vh,tt);
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n");
        __syncthreads();               /* sK(tt) visible */
        /* ---- S = Q(16x256).K^T for this warp's 8 positions ---- */
        float sf0=0.f,sf1=0.f;
        #pragma unroll
        for(int ks=0;ks<16;ks++){
            const __half*kr0=&sK[warp*8+g][(((ks*2  )^g)<<3)+t4*2];
            const __half*kr1=&sK[warp*8+g][(((ks*2+1)^g)<<3)+t4*2];
            unsigned b0=*(const unsigned*)kr0, b1=*(const unsigned*)kr1;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                :"+f"(sf0),"+f"(sf1),"+f"(dz0),"+f"(dz1)
                :"r"(qr[ks][0]),"r"(zr),"r"(qr[ks][1]),"r"(zr),
                 "r"(b0),"r"(b1));
        }
        {   /* mask cols beyond the live tile, online softmax on row g */
            int p=warp*8+t4*2;
            if(p  >=tlen) sf0=-1e30f;
            if(p+1>=tlen) sf1=-1e30f;
            float lm0=fmaxf(sf0,sf1);
            #pragma unroll
            for(int o2=1;o2<4;o2<<=1) lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,o2));
            float mn0=fmaxf(m0,lm0), f0=__expf(m0-mn0);
            float p00=__expf(sf0-mn0), p01=__expf(sf1-mn0);
            float ps0=p00+p01;
            #pragma unroll
            for(int o2=1;o2<4;o2<<=1) ps0+=__shfl_xor_sync(0xffffffff,ps0,o2);
            s0=s0*f0+ps0;
            __half2 hp=__floats2half2_rn(p00,p01);
            pa0=*(unsigned*)&hp;
            if(__any_sync(0xffffffff,mn0>m0)){
                #pragma unroll
                for(int n=0;n<32;n++){oa[n][0]*=f0;oa[n][1]*=f0;}
            }
            m0=mn0;
        }
        __syncthreads();               /* sK consumed */
        if(have_next){ Q36_DEC_CPA(sK,Kh,tt+32); asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n"); }
        else asm volatile("cp.async.wait_group 0;\n");
        if(tlen<32)                    /* zero stale V rows: P=0 x inf = NaN */
            for(int i2=threadIdx.x;i2<(32-tlen)*(HD/8);i2+=(int)blockDim.x){
                int p_=tlen+(i2>>5), cb_=i2&31;
                *(uint4*)&sV[p_][(cb_^(p_&7))*8]=make_uint4(0,0,0,0);
            }
        __syncthreads();               /* sV(tt) visible */
        {
            /* ---- O += P.V : k=8 (this stripe), ldmatrix.x4.trans B ---- */
            #pragma unroll
            for(int np=0;np<8;np++){
                unsigned d0,d1,d2,d3;
                unsigned sp=(unsigned)__cvta_generic_to_shared(
                    &sV[warp*8+vrow][(((np*4+vcol8)^vrow)<<3)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                    "{%0,%1,%2,%3},[%4];\n"
                    :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(sp));
                #define Q36_PV8(nn,bb) \
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 " \
                        "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};\n" \
                        :"+f"(oa[nn][0]),"+f"(oa[nn][1]),"+f"(dz0),"+f"(dz1) \
                        :"r"(pa0),"r"(zr),"r"(bb))
                Q36_PV8(np*4+0,d0); Q36_PV8(np*4+1,d1);
                Q36_PV8(np*4+2,d2); Q36_PV8(np*4+3,d3);
                #undef Q36_PV8
            }
        }
        __syncthreads();               /* sV consumed */
    }
    #undef Q36_DEC_CPA
    /* a slot is live iff this warp's stripe saw at least one position */
    int clen=cend-c0;
    if(clen>=32 || warp*8<clen){
        int h=kvh*8+g;
        #pragma unroll
        for(int n=0;n<32;n++){
            int d=n*8+t4*2;
            float2 w2=make_float2(oa[n][0],oa[n][1]);
            *(float2*)(pacc+((size_t)h*nslots+slot)*HD+d)=w2;
        }
        if(t4==0) pms[(size_t)h*nslots+slot]=make_float2(m0,s0);
    }
}

/* Two-level merge for the mma decode's stripe slots: block = (head, 32-dim
 * group), 8 slot-stripes accumulate online IN PARALLEL then fold in smem.
 * The single-level merge looped every slot serially per thread: 152us at
 * 32k depth vs 58us for the attention kernel itself. */
__global__ void k_attn_merge2(const float*pacc,const float2*pms,float*out,
                              int nslots,const int*posp){Q36_GDS();
    enum{HD=Q36_HEAD_DIM};
    int h=blockIdx.x;
    int dl=threadIdx.x&31, d=blockIdx.y*32+dl, sl0=threadIdx.x>>5;
    int seqlen=*posp+1;
    int nch=nslots>>2, per=(seqlen+nch-1)/nch;
    float m=-1e30f,sv=0.f,a=0.f;
    for(int sl=sl0;sl<nslots;sl+=8){
        int c=sl>>2,w=sl&3;
        int clen=seqlen-c*per; if(clen>per)clen=per;
        if(clen<=0||(clen<32&&w*8>=clen)) continue;
        float2 msv=pms[(size_t)h*nslots+sl];
        float v=pacc[((size_t)h*nslots+sl)*HD+d];
        float mn=fmaxf(m,msv.x);
        float fo=__expf(m-mn), fn=__expf(msv.x-mn);
        sv=sv*fo+msv.y*fn; a=a*fo+v*fn; m=mn;
    }
    __shared__ float sm[8][33],ss_[8][33],sa[8][33];
    sm[sl0][dl]=m; ss_[sl0][dl]=sv; sa[sl0][dl]=a;
    __syncthreads();
    if(sl0==0){
        float gm=sm[0][dl],gs=ss_[0][dl],ga=sa[0][dl];
        #pragma unroll
        for(int i=1;i<8;i++){
            float mn=fmaxf(gm,sm[i][dl]);
            float fo=__expf(gm-mn), fn=__expf(sm[i][dl]-mn);
            gs=gs*fo+ss_[i][dl]*fn; ga=ga*fo+sa[i][dl]*fn; gm=mn;
        }
        out[(size_t)h*HD+d]=ga/gs;
    }
}

__global__ void k_attn_merge(const float*pacc,const float2*pms,float*out,
                             int head_dim,int nchunks,const int*posp,int chunk_sz){Q36_GDS();
    int h=blockIdx.x, d=threadIdx.x; if(d>=head_dim)return;
    /* chunk_sz>0: fixed chunks (per-q-head kernel); 0: dynamic re-chunk
     * (GQA kernel); -1: dynamic re-chunk with 4 warp sub-slots per chunk
     * (mma decode) -- liveness must mirror the producer kernel exactly */
    int seqlen=*posp+1;
    float gm=-1e30f,gs=0,ga=0;
    if(chunk_sz==-1){                    /* mma decode: 4 stripe-slots/chunk */
        int nch=nchunks>>2;
        int per=(seqlen+nch-1)/nch;
        for(int pass=0;pass<2;pass++){
            for(int sl=0;sl<nchunks;sl++){
                int c=sl>>2, w=sl&3;
                int clen=seqlen-c*per; if(clen>per)clen=per;
                if(clen<=0||(clen<32&&w*8>=clen)) continue;
                if(pass==0){ gm=fmaxf(gm,pms[(size_t)h*nchunks+sl].x); continue; }
                float2 msv=pms[(size_t)h*nchunks+sl];
                float f=__expf(msv.x-gm);
                gs+=msv.y*f; ga+=pacc[((size_t)h*nchunks+sl)*head_dim+d]*f;
            }
        }
        out[(size_t)h*head_dim+d]=ga/gs;
        return;
    }
    int per=chunk_sz?chunk_sz:(seqlen+nchunks-1)/nchunks;
    int live=(seqlen+per-1)/per;                         /* only merge live chunks */
    for(int c=0;c<live;c++) gm=fmaxf(gm,pms[(size_t)h*nchunks+c].x);
    for(int c=0;c<live;c++){
        float2 msv=pms[(size_t)h*nchunks+c];
        float f=__expf(msv.x-gm);
        gs+=msv.y*f; ga+=pacc[((size_t)h*nchunks+c)*head_dim+d]*f;
    }
    out[(size_t)h*head_dim+d]=ga/gs;
}

/* Gated delta rule (inclusive readout, q scaled by 1/sqrt(d_k)): decay the
 * previous state, apply this token's delta update, read o_t = q~.S_t.
 *
 * v-head h pairs with k-head (h % nkh): the 16->32 head expansion in the
 * reference graph is a TILED repeat, not GQA-style interleaving.
 *
 * Layout/parallelism: the state is stored TRANSPOSED, [nvh][hvd][hkd], so one
 * warp owns one state column j and its lanes stream hkd contiguously
 * (coalesced).  The decayed values stay in registers between the kvold pass
 * and the update pass, halving state traffic.  8 warps/block share one head's
 * q/k staged in shared memory. */
__global__ void k_dn_scan(const float*qc,const float*kc,const float*vc,
                          const float*g,const float*beta,float*S,float*out,
                          int nvh,int nkh,int hkd,int hvd){Q36_GDS();
    int h=blockIdx.y;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int j=blockIdx.x*(blockDim.x>>5)+warp;
    int kh=h%nkh;
    __shared__ float sk[128], sq[128];
    const float*q=qc+kh*hkd, *k=kc+kh*hkd;
    for(int i=threadIdx.x;i<hkd;i+=blockDim.x){ sk[i]=k[i]; sq[i]=q[i]; }
    __syncthreads();
    if(j>=hvd)return;
    float*Sj=S+((size_t)h*hvd+j)*hkd;
    float expg=__expf(g[h]), b=beta[h], vj=vc[(size_t)h*hvd+j];
    float sd[4], kvold=0;                      /* hkd==128 -> 4 lane-chunks */
    #pragma unroll
    for(int c=0;c<4;c++){ int i=c*32+lane; float x=Sj[i]*expg; sd[c]=x; kvold+=sk[i]*x; }
    kvold=warpsum_xor(kvold);
    float delta=vj-kvold;
    float o=0;
    #pragma unroll
    for(int c=0;c<4;c++){ int i=c*32+lane; float x=sd[c]+b*sk[i]*delta; Sj[i]=x; o+=sq[i]*x; }
    o=warpsum(o);
    if(lane==0) out[(size_t)h*hvd+j]=o*rsqrtf((float)hkd);
}
/* Batched decode SSM scan: grid.z = tenant.  Inputs are per-slot strided
 * (conv output, gates in the MT scratch; state in Sstate slot).  Same math
 * as k_dn_scan; folds B launches into one to fill the SMs. */
__global__ void k_dn_scan_b(const float*qc0,const float*kc0,const float*vc0,
                          const float*g0,const float*beta0,float*S0,float*out0,
                          int nvh,int nkh,int hkd,int hvd,
                          uint64_t in_s,uint64_t gv_s,uint64_t S_s,uint64_t out_s){Q36_GDS();
    int t=blockIdx.z;
    const float*qc=qc0+(size_t)t*in_s, *kc=kc0+(size_t)t*in_s, *vc=vc0+(size_t)t*in_s;
    const float*g=g0+(size_t)t*gv_s, *beta=beta0+(size_t)t*gv_s;
    float*S=S0+(size_t)t*S_s, *out=out0+(size_t)t*out_s;
    int h=blockIdx.y;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int j=blockIdx.x*(blockDim.x>>5)+warp;
    int kh=h%nkh;
    __shared__ float sk[128], sq[128];
    const float*q=qc+kh*hkd, *k=kc+kh*hkd;
    for(int i=threadIdx.x;i<hkd;i+=blockDim.x){ sk[i]=k[i]; sq[i]=q[i]; }
    __syncthreads();
    if(j>=hvd)return;
    float*Sj=S+((size_t)h*hvd+j)*hkd;
    float expg=__expf(g[h]), b=beta[h], vj=vc[(size_t)h*hvd+j];
    float sd[4], kvold=0;
    #pragma unroll
    for(int c=0;c<4;c++){ int i=c*32+lane; float x=Sj[i]*expg; sd[c]=x; kvold+=sk[i]*x; }
    kvold=warpsum_xor(kvold);
    float delta=vj-kvold, o=0;
    #pragma unroll
    for(int c=0;c<4;c++){ int i=c*32+lane; float x=sd[c]+b*sk[i]*delta; Sj[i]=x; o+=sq[i]*x; }
    o=warpsum(o);
    if(lane==0) out[(size_t)h*hvd+j]=o*rsqrtf((float)hkd);
}

/* gated group RMSNorm: normalized(out over hvd, gain ssm_norm) * silu(z).
 * Reduction is fixed-order smem partials, NOT atomicAdd: with 4 warps the
 * atomic arrival order changes the f32 sum by ~1ulp between scheduling
 * contexts (graph vs eager), which broke MTP-verify bit-parity with solo
 * decode -- every other decode norm is <=2 warps (commutative-safe) or
 * already fixed-order. */
__global__ void k_dn_gnorm(float*out,const float*gain,const float*z,int nvh,int hvd,float eps){Q36_GDS();
    int h=blockIdx.x; float*oh=out+h*hvd; const float*zh=z+h*hvd;
    __shared__ float part[32]; __shared__ float ss;
    float loc=0;
    for(int j=threadIdx.x;j<hvd;j+=blockDim.x) loc+=oh[j]*oh[j];
    loc=warpsum(loc);
    if((threadIdx.x&31)==0) part[threadIdx.x>>5]=loc;
    __syncthreads();
    if(threadIdx.x==0){ float s=0; int nw=(blockDim.x+31)/32; for(int w=0;w<nw;w++)s+=part[w]; ss=s; }
    __syncthreads(); float sc=rsqrtf(ss/hvd+eps);
    for(int j=threadIdx.x;j<hvd;j+=blockDim.x) oh[j]=oh[j]*sc*gain[j]*silu(zh[j]);
}
/* softplus(alpha+dt)*a -> g[nvh]; beta sigmoid.  grid.y = batch row. */
__global__ void k_dn_gates(const float*alpha,const float*dt,const float*a,const float*betain,
                           float*g,float*beta,int nvh){Q36_GDS();
    int t=blockIdx.y;
    alpha+=(size_t)t*nvh; betain+=(size_t)t*nvh; g+=(size_t)t*nvh; beta+=(size_t)t*nvh;
    int h=blockIdx.x*blockDim.x+threadIdx.x; if(h>=nvh)return;
    float sp=log1pf(__expf(alpha[h]+dt[h]));   /* softplus */
    g[h]=sp*a[h];
    beta[h]=sigm(betain[h]);
}

/* ============================ engine ================================== */
extern "C" q36_engine* q36_engine_create(q36_model*m,int ctx){
    /* pick the GPU with the most free VRAM unless the user pinned one
     * (prevents OOM from two processes colliding on device 0) */
    if(!g_no_autoselect && !getenv("CUDA_VISIBLE_DEVICES")){
        int nd=0; cudaGetDeviceCount(&nd);
        int best=0; size_t bestfree=0;
        for(int d=0;d<nd;d++){
            size_t fr=0,tot=0;
            if(cudaSetDevice(d)==cudaSuccess && cudaMemGetInfo(&fr,&tot)==cudaSuccess)
                if(fr>bestfree){bestfree=fr;best=d;}
        }
        cudaSetDevice(best);
        if(bestfree < 27ull<<30)
            fprintf(stderr,"warning: best GPU (%d) has only %.1f GiB free; "
                    "~26 GiB needed. Another process may be using the GPUs.\n",
                    best,bestfree/1073741824.0);
        else fprintf(stderr,"using GPU %d (%.1f GiB free)\n",best,bestfree/1073741824.0);
    }
    q36_engine*e=(q36_engine*)calloc(1,sizeof(q36_engine));
    e->m=m; e->ctx=ctx; e->expert_scale=1.0f;
    e->nslots=(g_create_slots<1)?1:g_create_slots; e->pf_slot=0;
    e->se0=g_shard_e0; e->se1=g_shard_e1; e->inc_shared=1;
    /* upload embedding + output head */
    CK(cudaMalloc(&e->tok_embd,m->token_embd.nbytes));
    CK(cudaMemcpy(e->tok_embd,m->token_embd.data,m->token_embd.nbytes,cudaMemcpyHostToDevice));
    e->out_norm=upf32(&m->output_norm);
    /* output head: FP8 e4m3 (524MB, ppl-validated) by default;
     * Q36_FP4_HEAD=1 selects the NVFP4 experiment (286MB, ppl-gated) */
    {
        const char*h4=getenv("Q36_FP4_HEAD"), *hm=getenv("Q36_MIX_HEAD");
        if(h4)      e->output=upload_nvfp4(&m->output);
        else if(hm) e->output=upload_head_mixed(&m->output,atoi(hm)>0?atoi(hm):32768);
        else        e->output=upload_e4m3(&m->output);
    }
    for(int L=0;L<Q36_N_LAYER;L++){
        const q36_block*b=&m->blocks[L]; dblock*db=&e->blk[L];
        db->is_attn=b->is_attention;
        db->post_norm=upf32(&b->moe.post_norm);
        db->router=upf32_cat(&b->moe.router,&b->moe.shared_gate_inp); /* rows 0..255 + shg row 256 */
        db->gate_exps=upload(&b->moe.gate_exps); db->up_exps=upload(&b->moe.up_exps); db->down_exps=upload(&b->moe.down_exps);
        db->sh_gate_inp=upf32(&b->moe.shared_gate_inp);
        db->sh_gate=upload(&b->moe.shared_gate); db->sh_up=upload(&b->moe.shared_up); db->sh_down=upload(&b->moe.shared_down);
        if(b->is_attention){
            db->attn_norm=upf32(&b->attn.attn_norm);
            db->q=upload(&b->attn.q); db->k=upload(&b->attn.k); db->v=upload(&b->attn.v); db->o=upload(&b->attn.o);
            db->q_norm=upf32(&b->attn.q_norm); db->k_norm=upf32(&b->attn.k_norm);
            /* fp16-sized so either mode fits; quant mode uses ~40% of it */
            CK(cudaMalloc(&e->Kc[L],(size_t)e->nslots*ctx*Q36_KV_DIM*sizeof(__half)));
            CK(cudaMalloc(&e->Ks[L],(size_t)e->nslots*ctx*(Q36_KV_DIM/32)*sizeof(__half)));
            CK(cudaMalloc(&e->Vc[L],(size_t)e->nslots*ctx*Q36_KV_DIM*sizeof(__half)));
            CK(cudaMalloc(&e->Vs[L],(size_t)e->nslots*ctx*(Q36_KV_DIM/32)));
        } else {
            db->attn_norm=upf32(&b->ssm.attn_norm);
            db->qkv=upload(&b->ssm.qkv); db->gate=upload(&b->ssm.gate); db->ssm_out=upload(&b->ssm.out);
            db->conv1d=upf32(&b->ssm.conv1d); db->ssm_a=upf32(&b->ssm.a); db->dt_bias=upf32(&b->ssm.dt_bias); db->ssm_norm=upf32(&b->ssm.ssm_norm);
            db->alpha=upf32_cat(&b->ssm.alpha,&b->ssm.beta); /* rows 0..31 alpha, 32..63 beta */
            db->beta=upf32(&b->ssm.beta);          /* prefill uses the separate copy */
            CK(cudaMalloc(&e->Sstate[L],(size_t)e->nslots*NVH*HKD*HVD*sizeof(float)));
            CK(cudaMemset(e->Sstate[L],0,(size_t)e->nslots*NVH*HKD*HVD*sizeof(float)));
            CK(cudaMalloc(&e->convhist[L],(size_t)e->nslots*CONVD*3*sizeof(float)));
            CK(cudaMemset(e->convhist[L],0,(size_t)e->nslots*CONVD*3*sizeof(float)));
        }
    }
    #define A(p,n) CK(cudaMalloc(&e->p,(n)*sizeof(float)))
    A(h,Q36_D_MODEL);A(x,Q36_D_MODEL);A(tmp,8192);A(q,Q36_Q_DIM);A(gate,Q36_Q_DIM);
    A(kbuf,8192);A(vbuf,VALD);A(attn,Q36_Q_DIM);A(moe,Q36_D_MODEL);
    A(g512,Q36_N_EXPERT_USED*Q36_EXPERT_FF);A(u512,Q36_N_EXPERT_USED*Q36_EXPERT_FF);
    A(d2048,Q36_N_EXPERT_USED*Q36_D_MODEL);A(shexp,Q36_D_MODEL);A(shg,1);
    A(rlogits,Q36_N_EXPERT+1);A(logits,Q36_N_VOCAB);
    CK(cudaMalloc(&e->d_topk,Q36_N_EXPERT_USED*sizeof(int)));
    CK(cudaMalloc(&e->d_topw,Q36_N_EXPERT_USED*sizeof(float)));
    int max_slots=4*((ctx+attn_chunk()-1)/attn_chunk());   /* mma: 4 sub-slots */
    CK(cudaMalloc(&e->pacc,(size_t)Q36_N_HEAD*max_slots*Q36_HEAD_DIM*sizeof(float)));
    CK(cudaMalloc(&e->pms,(size_t)Q36_N_HEAD*max_slots*sizeof(float2)));
    CK(cudaMalloc(&e->d_tok2,4)); CK(cudaMalloc(&e->d_pos,4)); CK(cudaMalloc(&e->d_argmax,4));
    e->gexec=NULL; e->gchunks=0;
    e->s_temp=0.f; e->s_topk=20; e->s_topp=0.95f;
    CK(cudaMalloc(&e->d_rng,8));
    { unsigned long long seed=0x9E3779B97F4A7C15ULL; CK(cudaMemcpy(e->d_rng,&seed,8,cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&e->d_pv,128*Q36_SAMPLE_KMAX*sizeof(float)));
    CK(cudaMalloc(&e->d_pi,128*Q36_SAMPLE_KMAX*sizeof(int)));
    e->v_nblk=gr(Q36_N_VOCAB,8);       /* fused-head partials (greedy solo + verify) */
    CK(cudaMalloc(&e->v_pv,(size_t)Q36_MTP_MAXB*e->v_nblk*sizeof(float)));
    CK(cudaMalloc(&e->v_pi,(size_t)Q36_MTP_MAXB*e->v_nblk*sizeof(int)));
    CK(cudaStreamCreateWithFlags(&e->s2,cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&e->ev_in,cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&e->ev_sh,cudaEventDisableTiming));
    A(sh_a,Q36_EXPERT_FF);A(sh_b,Q36_EXPERT_FF);
    q36_cuda_argmax_init();
    {   /* chunked-prefill scratch */
        const int T=Q36_PF_CHUNK;
        CK(cudaMalloc(&e->pf_toks,T*sizeof(int)));
        A(pfH,T*Q36_D_MODEL);A(pfX,T*Q36_D_MODEL);A(pfOUT,T*Q36_D_MODEL);
        A(pfMOE,T*Q36_D_MODEL);A(pfSH,T*Q36_D_MODEL);
        A(pfQKV,T*8192);A(pfCB,T*8192);
        A(pfZ,T*4096);A(pfO,T*4096);A(pfQ,T*4096);A(pfGATE,T*4096);
        A(pfA,T*32);A(pfB,T*32);A(pfG,T*32);A(pfBt,T*32);
        A(pfK,T*Q36_KV_DIM);A(pfV,T*Q36_KV_DIM);
        A(pfRL,T*Q36_N_EXPERT);A(pf_topw,T*Q36_N_EXPERT_USED);
        A(pfGU_g,(size_t)T*8*Q36_EXPERT_FF);A(pfGU_u,(size_t)T*8*Q36_EXPERT_FF);
        A(pfDN,(size_t)T*8*Q36_D_MODEL);A(pfSU,T*Q36_EXPERT_FF);A(pfshg,T);
        CK(cudaMalloc(&e->pf_topk,(size_t)T*Q36_N_EXPERT_USED*sizeof(int)));
        CK(cudaMalloc(&e->pf_ecount,Q36_N_EXPERT*sizeof(int)));
        CK(cudaMalloc(&e->pf_elist,(size_t)Q36_N_EXPERT*T*sizeof(int)));
        CK(cudaMalloc(&e->pfXq,(size_t)T*Q36_D_MODEL));
        CK(cudaMalloc(&e->pfXq8,(size_t)T*8192));
        CK(cudaMalloc(&e->pfXsf,(size_t)T*256*sizeof(float)));
        CK(cudaMalloc(&e->pfXs,(size_t)T*(Q36_D_MODEL/32)));
    }
    if(m->has_mtp && e->nslots==1 && e->se0==0 && e->se1==Q36_N_EXPERT){
        /* stage 2: upload the nextn module + dedicated KV + verify scratch.
         * Solo single-GPU only: the speculative loop shares the solo decode
         * scratch and the slot-0 KV/SSM state. */
        const q36_mtp_block*mb=&m->mtp; dblock*db=&e->mtp_db;
        db->is_attn=true;
        db->attn_norm=upf32(&mb->attn.attn_norm);
        db->q=upload(&mb->attn.q); db->k=upload(&mb->attn.k);
        db->v=upload(&mb->attn.v); db->o=upload(&mb->attn.o);
        db->q_norm=upf32(&mb->attn.q_norm); db->k_norm=upf32(&mb->attn.k_norm);
        db->post_norm=upf32(&mb->moe.post_norm);
        db->router=upf32_cat(&mb->moe.router,&mb->moe.shared_gate_inp);
        db->gate_exps=upload(&mb->moe.gate_exps); db->up_exps=upload(&mb->moe.up_exps);
        db->down_exps=upload(&mb->moe.down_exps);
        db->sh_gate_inp=upf32(&mb->moe.shared_gate_inp);
        db->sh_gate=upload(&mb->moe.shared_gate); db->sh_up=upload(&mb->moe.shared_up);
        db->sh_down=upload(&mb->moe.shared_down);
        e->mtp_eh=upload(&mb->eh_proj);
        e->mtp_enorm=upf32(&mb->enorm); e->mtp_hnorm=upf32(&mb->hnorm);
        e->mtp_shn=upf32(&mb->shared_head_norm);
        /* MTP KV: 1 attention layer x ctx, always fp16 (tiny vs main KV) */
        CK(cudaMalloc(&e->mtpKc,(size_t)ctx*Q36_KV_DIM*sizeof(__half)));
        CK(cudaMalloc(&e->mtpKs,(size_t)ctx*(Q36_KV_DIM/32)*sizeof(__half)));
        CK(cudaMalloc(&e->mtpVc,(size_t)ctx*Q36_KV_DIM*sizeof(__half)));
        CK(cudaMalloc(&e->mtpVs,(size_t)ctx*(Q36_KV_DIM/32)));
        A(mtp_h,Q36_D_MODEL);A(mtp_emb,Q36_D_MODEL);
        A(mtp_cat,2*Q36_D_MODEL);A(mtp_z,Q36_D_MODEL);
        /* verify scratch, sized for the max width B = Q36_MTP_MAXB */
        enum{MB=Q36_MTP_MAXB};
        A(vh,MB*Q36_D_MODEL);A(vx,MB*Q36_D_MODEL);A(vmoe,MB*Q36_D_MODEL);A(vshexp,MB*Q36_D_MODEL);
        A(vtmp,MB*8192);A(vkbuf,MB*8192);
        A(vq,MB*4096);A(vgate,MB*4096);A(vvbuf,MB*4096);A(vattn,MB*4096);
        A(vg512,MB*Q36_N_EXPERT_USED*Q36_EXPERT_FF);A(vu512,MB*Q36_N_EXPERT_USED*Q36_EXPERT_FF);
        A(vd2048,MB*Q36_N_EXPERT_USED*Q36_D_MODEL);
        A(vsh_a,MB*Q36_EXPERT_FF);A(vsh_b,MB*Q36_EXPERT_FF);
        A(vrl,MB*(Q36_N_EXPERT+1));A(vtopw,MB*Q36_N_EXPERT_USED);
        CK(cudaMalloc(&e->vtopk,MB*Q36_N_EXPERT_USED*sizeof(int)));
        CK(cudaMalloc(&e->d_vtok,20*sizeof(int)));   /* layout: see struct */
        e->d_vout=e->d_vtok+4; e->d_mm=e->d_vtok+8; e->d_vpos=e->d_vtok+16;
        int nssm=0;
        for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn) e->ssm_sidx[L]=nssm++;
        CK(cudaMalloc(&e->Ssnap,(size_t)(MB-1)*nssm*NVH*HKD*HVD*sizeof(float)));
        CK(cudaMalloc(&e->convsnap,(size_t)(MB-1)*nssm*CONVD*3*sizeof(float)));
        e->has_mtp=1; e->mtp_k=1; e->mtp_base=-1;
        fprintf(stderr,"MTP: nextn module uploaded (self-speculative decode available)\n");
    }
    if(e->nslots>1){   /* multi-tenant batched-decode scratch */
        int B=e->nslots; enum{D=Q36_D_MODEL};
        A(hb,B*D);A(xb,B*D);A(tmpb,(size_t)B*8192);A(qb,B*4096);A(gateb,B*4096);
        A(kbufb,(size_t)B*8192);A(vbufb,B*4096);A(attnb,B*4096);A(moeb,B*D);
        A(g512b,B*4096);A(u512b,B*4096);A(d2048b,(size_t)B*Q36_N_EXPERT_USED*D);
        A(shexpb,B*D);A(sh_ab,B*Q36_EXPERT_FF);A(sh_bb,B*Q36_EXPERT_FF);
        A(rlogitsb,B*(Q36_N_EXPERT+1));A(d_topwb,B*Q36_N_EXPERT_USED);
        CK(cudaMalloc(&e->d_topkb,(size_t)B*Q36_N_EXPERT_USED*sizeof(int)));
        CK(cudaMalloc(&e->d_tok_b,B*sizeof(int)));
        CK(cudaMalloc(&e->d_pos_b,B*sizeof(int)));
        CK(cudaMalloc(&e->d_out_b,B*sizeof(int)));
        CK(cudaMalloc(&e->d_rng_b,B*8));
        unsigned long long*seeds=(unsigned long long*)malloc(B*8);
        for(int i=0;i<B;i++)seeds[i]=0x9E3779B97F4A7C15ULL+i;
        CK(cudaMemcpy(e->d_rng_b,seeds,B*8,cudaMemcpyHostToDevice));
        free(seeds);
        e->mt_temp=(float*)calloc(B,4); e->mt_topp=(float*)calloc(B,4);
        e->mt_topk=(int*)calloc(B,4);
        for(int i=0;i<B;i++){e->mt_topk[i]=20;e->mt_topp[i]=0.95f;}
        e->hd_nblk=gr(Q36_N_VOCAB,8);   /* 8 warps -> 8 rows per block */
        CK(cudaMalloc(&e->hd_pv,(size_t)B*e->hd_nblk*sizeof(float)));
        CK(cudaMalloc(&e->hd_pi,(size_t)B*e->hd_nblk*sizeof(int)));
    }
    return e;
}
extern "C" q36_engine* q36_engine_create_mt(q36_model*m,int ctx,int nslots){
    g_create_slots=nslots; q36_engine*e=q36_engine_create(m,ctx);
    g_create_slots=1; return e;
}
extern "C" int q36_engine_nslots(q36_engine*e){ return e->nslots; }

/* Fully device-resident MoE: router, top-8 selection, expert GEMVs, shared
 * expert, and the weighted combine never leave the GPU.  Zero host syncs. */
static void moe_ffn(q36_engine*e,dblock*db,const float*in,int L){
    enum { NU=Q36_N_EXPERT_USED, FF=Q36_EXPERT_FF, D=Q36_D_MODEL };
    static int msen=-1; if(msen<0) msen=getenv("Q36_MTP_SYNC")?1:0;
    int msync=msen&&(L==Q36_N_LAYER);
    #define MOSY(tag) do{ if(msync){ cudaError_t se_=cudaDeviceSynchronize(); \
        fprintf(stderr,"moe[%s]: %s\n",tag,se_?cudaGetErrorString(se_):"ok"); \
        if(se_)exit(1);} }while(0)
    /* router rows 0..255 + shared-gate row 256 in ONE f32 matvec (weights
     * were concatenated at load); shg scalar lands at rlogits[256] */
    k_matvec_f32_row<<<Q36_N_EXPERT+1,256>>>(db->router,in,e->rlogits,Q36_N_EXPERT+1,D);
    MOSY("router");
    k_router_topk<<<1,32>>>(e->rlogits,Q36_N_EXPERT,NU,e->d_topk,e->d_topw,e->expert_scale,1);
    MOSY("topk");
    /* routed gate+up fused in one launch (grid.z), all 8 slots via grid.y.
     * Measured dead ends (keep them dead): W4A8-MMA at T=1 (270->163 t/s,
     * per-tile staging syncs don't amortize at one token) and cp.async
     * double-buffering in the prefill GEMM (5720->4596 t/s, block-level
     * parallelism already hides staging; 2x smem just cut occupancy). */
    dim3 gg(gr(FF/2,8),NU,2);   /* dual-row warps: half the row-blocks */
    k_expert_gemv2<<<gg,256>>>(db->gate_exps.elayout,db->gate_exps.d,db->gate_exps.expert_stride,
                               db->up_exps.elayout,db->up_exps.d,db->up_exps.expert_stride,
                               e->se0,e->se1,e->d_topk,in,e->g512,e->u512,FF,D);
    MOSY("gateup");
    k_silu_mul_slots<<<gr(NU*FF,256),256>>>(e->g512,e->u512,NU*FF);
    dim3 gd(gr(D/2,8),NU);      /* dual-row warps */
    k_expert_gemv<<<gd,256>>>(db->down_exps.elayout,db->down_exps.d,db->down_exps.expert_stride,
                              e->se0,e->se1,e->d_topk,e->g512,FF,e->d2048,D,FF);
    MOSY("down");
    /* shared expert on a forked stream: fully independent of routing, so it
     * overlaps the routed gate/up/down chain (graph capture turns the event
     * wait/record into graph edges).  Own scratch (sh_a/sh_b) -- the routed
     * chain owns g512/u512/tmp. */
    int fork=!getenv("Q36_NOFORK");
    cudaStream_t sh_s=fork?e->s2:cudaStreamPerThread;
    if(fork){ CK(cudaEventRecord(e->ev_in,cudaStreamPerThread));
              CK(cudaStreamWaitEvent(e->s2,e->ev_in,0)); }
    k_matvec_multi3<<<gr(2*FF,8),256,0,sh_s>>>(db->sh_gate.d,FF,e->sh_a,
                                        db->sh_up.d,FF,e->sh_b,NULL,0,NULL,in,D);
    k_silu_mul<<<gr(FF,256),256,0,sh_s>>>(e->sh_a,e->sh_b,FF);
    mv_on(sh_s,&db->sh_down,e->sh_a,e->shexp);
    if(fork){ CK(cudaEventRecord(e->ev_sh,e->s2));
              CK(cudaStreamWaitEvent(cudaStreamPerThread,e->ev_sh,0)); }
    MOSY("shared");
    k_moe_combine<<<gr(D,256),256>>>(e->moe,e->d2048,e->d_topw,e->shexp,e->rlogits+Q36_N_EXPERT,D,NU);
    MOSY("combine");
    #undef MOSY
    if(getenv("Q36_RSTAT")){
        static double cum[9]={0}; static long nsamp=0;
        float tw[NU]; CK(cudaMemcpy(tw,e->d_topw,sizeof tw,cudaMemcpyDeviceToHost));
        double c=0; for(int j=0;j<NU;j++){ c+=tw[j]; cum[j+1]+=c; }
        if(++nsamp%2000==0){
            fprintf(stderr,"[router] avg cumulative weight by k (n=%ld):",nsamp);
            for(int j=1;j<=NU;j++) fprintf(stderr," k%d=%.3f",j,cum[j]/nsamp);
            fprintf(stderr,"\n");
        }
    }
    if(L==0 && getenv("Q36_DEBUG")){ int tk[NU]; float tw[NU],lg[3];
        CK(cudaMemcpy(tk,e->d_topk,sizeof tk,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tw,e->d_topw,sizeof tw,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(lg,e->rlogits,12,cudaMemcpyDeviceToHost));
        fprintf(stderr,"    moe.logits=%.3f %.3f %.3f  topk=%d %d %d..%d %d %d  w=%.4f %.4f..%.4f\n",
            lg[0],lg[1],lg[2],tk[0],tk[1],tk[2],tk[5],tk[6],tk[7],tw[0],tw[1],tw[7]); }
}

/* Decode attention: position comes from device (e->d_pos) so this is graph-
 * capturable; grids sized for `maxch` chunks, live chunks gated in-kernel. */
static void attn_layer(q36_engine*e,dblock*db,int L,const float*x,int maxch){
    /* q(8192)+k(512)+v(512) projections fused into one launch */
    k_matvec_multi3<<<gr(Q36_Q_DIM*2+2*Q36_KV_DIM,8),256>>>(
        db->q.d,Q36_Q_DIM*2,e->tmp, db->k.d,Q36_KV_DIM,e->kbuf,
        db->v.d,Q36_KV_DIM,e->vbuf, x,Q36_D_MODEL);
    k_split_qgate<<<gr(Q36_Q_DIM,256),256>>>(e->tmp,e->q,e->gate,Q36_N_HEAD,Q36_HEAD_DIM);
    k_head_rmsnorm<<<Q36_N_HEAD,64>>>(e->q,db->q_norm,Q36_HEAD_DIM,Q36_RMS_EPS);
    k_head_rmsnorm<<<Q36_N_HEAD_KV,64>>>(e->kbuf,db->k_norm,Q36_HEAD_DIM,Q36_RMS_EPS);
    k_rope_p<<<Q36_N_HEAD,Q36_ROT_DIM/2>>>(e->q,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_pos,Q36_ROPE_FREQ_BASE);
    k_rope_p<<<Q36_N_HEAD_KV,Q36_ROT_DIM/2>>>(e->kbuf,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_pos,Q36_ROPE_FREQ_BASE);
    k_kv_append<<<gr(2*(Q36_KV_DIM/32)*32,256),256>>>(e->kbuf,e->vbuf,
        e->Kc[L],e->Ks[L],e->Vc[L],e->Vs[L],e->d_pos,Q36_KV_DIM,e->kvq);
    float scale=1.f/sqrtf((float)Q36_HEAD_DIM);
    static int nomma=-1;
    if(nomma<0) nomma=getenv("Q36_NOMMA")?1:(getenv("Q36_ATTN2")?2:0);
    int mode; /* 0 = mma, 1 = per-q-head scalar, 2 = GQA scalar (dead end) */
    if(maxch<=1 || attn_chunk()!=Q36_ATTN_CHUNK2) mode=1;
    else if(e->kvq||nomma==1) mode=1;
    else mode=(nomma==2)?2:0;
    if(mode==0){
        if(maxch*Q36_N_HEAD_KV>=512)   /* enough blocks to fill 3/SM */
            k_attn_dec_mma<3><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(e->q,
                (const __half*)e->Kc[L],(const __half*)e->Vc[L],
                e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos,scale);
        else
            k_attn_dec_mma<2><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(e->q,
                (const __half*)e->Kc[L],(const __half*)e->Vc[L],
                e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos,scale);
    }
    else if(mode==2)
        k_attn_partial2<<<dim3(maxch,Q36_N_HEAD_KV),256>>>(e->q,e->Kc[L],e->Ks[L],e->Vc[L],e->Vs[L],
            e->pacc,e->pms,NULL,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos,scale,e->kvq);
    else
        k_attn_partial<<<dim3(maxch,Q36_N_HEAD),256>>>(e->q,e->Kc[L],e->Ks[L],e->Vc[L],e->Vs[L],
            e->pacc,e->pms,(maxch==1)?e->attn:NULL,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos,scale,e->kvq);
    if(maxch>1){
        if(mode==0)
            k_attn_merge2<<<dim3(Q36_N_HEAD,Q36_HEAD_DIM/32),256>>>(
                e->pacc,e->pms,e->attn,maxch*4,e->d_pos);
        else
            k_attn_merge<<<Q36_N_HEAD,Q36_HEAD_DIM>>>(e->pacc,e->pms,e->attn,Q36_HEAD_DIM,
                maxch,e->d_pos,mode==2?0:Q36_ATTN_CHUNK);
    }
    k_sigmoid_mul<<<gr(Q36_Q_DIM,256),256>>>(e->attn,e->gate,Q36_Q_DIM);
    mv(&db->o,e->attn,e->x);             /* -> e->x holds attn contribution [2048] */
}

/* ---- SSM megakernel (decode) --------------------------------------------
 * Fuses conv -> alpha/beta projections -> gates -> q/k L2 norm -> delta scan
 * -> gated group norm into ONE launch per SSM layer.  16 blocks, one per
 * k-head kh; each block owns q/k channels of kh and the v channels + state of
 * v-heads kh and kh+16, so all phases stay block-local (no grid sync).
 * Intermediates never touch VRAM. */
__global__ void k_ssm_mega(const float*qkv,const float*z,const float*x,
                           const float*convw,float*hist,
                           const float*alphabeta,const float*dtb,const float*a_log,
                           const float*gain,float*S,float*out,int nvh,int nkh,int hkd,int hvd){Q36_GDS();
    int kh=blockIdx.x, tid=threadIdx.x, lane=tid&31;
    __shared__ float qc[128],kc[128],vA[128],vB[128];
    __shared__ float red[32];
    __shared__ float gA,gB,bA,bB,nq,nk;
    /* phase A: conv+silu for this block's 512 channels */
    for(int s2=tid;s2<512;s2+=blockDim.x){
        int c;
        float*dst;
        if(s2<128){ c=kh*128+s2; dst=&qc[s2]; }
        else if(s2<256){ c=2048+kh*128+(s2-128); dst=&kc[s2-128]; }
        else if(s2<384){ c=4096+kh*128+(s2-256); dst=&vA[s2-256]; }
        else { c=4096+(kh+16)*128+(s2-384); dst=&vB[s2-384]; }
        float h0=hist[c*3],h1=hist[c*3+1],h2=hist[c*3+2],xt=qkv[c];
        *dst=silu(convw[c*4]*h0+convw[c*4+1]*h1+convw[c*4+2]*h2+convw[c*4+3]*xt);
        hist[c*3]=h1; hist[c*3+1]=h2; hist[c*3+2]=xt;
    }
    __syncthreads();
    /* phase B: block reductions -- q/k sumsq + 4 alpha/beta dots */
    {
        float sq=0,sk=0,da0=0,da1=0,db0=0,db1=0;
        for(int i2=tid;i2<128;i2+=blockDim.x){ sq+=qc[i2]*qc[i2]; sk+=kc[i2]*kc[i2]; }
        for(int i2=tid;i2<2048;i2+=blockDim.x){
            float xv=x[i2];
            da0+=alphabeta[(size_t)kh*2048+i2]*xv;
            da1+=alphabeta[(size_t)(kh+16)*2048+i2]*xv;
            db0+=alphabeta[(size_t)(32+kh)*2048+i2]*xv;
            db1+=alphabeta[(size_t)(32+kh+16)*2048+i2]*xv;
        }
        float v6[6]={sq,sk,da0,da1,db0,db1};
        for(int k2=0;k2<6;k2++){
            float v=warpsum(v6[k2]);
            if(lane==0) red[tid>>5]=v;
            __syncthreads();
            if(tid==0){ float s3=0; for(int w2=0;w2<(int)(blockDim.x>>5);w2++)s3+=red[w2];
                if(k2==0)nq=rsqrtf(s3+Q36_RMS_EPS);
                else if(k2==1)nk=rsqrtf(s3+Q36_RMS_EPS);
                else if(k2==2)gA=log1pf(__expf(s3+dtb[kh]))*a_log[kh];
                else if(k2==3)gB=log1pf(__expf(s3+dtb[kh+16]))*a_log[kh+16];
                else if(k2==4)bA=sigm(s3);
                else bB=sigm(s3);
            }
            __syncthreads();
        }
    }
    /* phase C: L2-normalize q/k in smem */
    for(int i2=tid;i2<128;i2+=blockDim.x){ qc[i2]*=nq; kc[i2]*=nk; }
    __syncthreads();
    /* phase D: delta scan, thread per state column (2 heads x 128 cols) */
    {
        int h=(tid<128)?kh:kh+16, j=tid&127;
        const float*v=(tid<128)?vA:vB;
        float expg=__expf((tid<128)?gA:gB), b=(tid<128)?bA:bB;
        float*Sj=S+((size_t)h*hvd+j)*hkd;
        float kvold=0;
        float sreg[128];
        #pragma unroll 8
        for(int i2=0;i2<128;i2++){ float s3=Sj[i2]*expg; sreg[i2]=s3; kvold+=kc[i2]*s3; }
        float delta=v[j]-kvold, o=0;
        #pragma unroll 8
        for(int i2=0;i2<128;i2++){ float s3=sreg[i2]+b*kc[i2]*delta; Sj[i2]=s3; o+=qc[i2]*s3; }
        /* stash scan output in vA/vB (consumed next phase) */
        __syncthreads();
        if(tid<128) vA[j]=o*rsqrtf((float)hkd); else vB[j]=o*rsqrtf((float)hkd);
    }
    __syncthreads();
    /* phase E: gated group RMS norm per head + write out */
    for(int hh=0;hh<2;hh++){
        float*o=(hh==0)?vA:vB; int h=(hh==0)?kh:kh+16;
        float loc=0;
        for(int i2=tid;i2<128;i2+=blockDim.x) loc+=o[i2]*o[i2];
        loc=warpsum(loc);
        if(lane==0) red[tid>>5]=loc;
        __syncthreads();
        __shared__ float sc2;
        if(tid==0){ float s3=0; for(int w2=0;w2<(int)(blockDim.x>>5);w2++)s3+=red[w2];
            sc2=rsqrtf(s3/128.f+Q36_RMS_EPS); }
        __syncthreads();
        for(int i2=tid;i2<128;i2+=blockDim.x)
            out[(size_t)h*128+i2]=o[i2]*sc2*gain[i2]*silu(z[(size_t)h*128+i2]);
        __syncthreads();
    }
}

static int gg_dbg=-1;
static void d3(const char*t,const float*dp){
    if(gg_dbg<0) gg_dbg=getenv("Q36_DEBUG")?1:0; if(!gg_dbg) return;
    float v[3]; cudaMemcpy(v,dp,12,cudaMemcpyDeviceToHost);
    fprintf(stderr,"    ssm.%s = %.4f %.4f %.4f\n",t,v[0],v[1],v[2]);
}
static void dN(const char*t,const float*dp,int n){
    if(gg_dbg<0) gg_dbg=getenv("Q36_DEBUG")?1:0; if(!gg_dbg) return;
    float*v=(float*)malloc(n*4); cudaMemcpy(v,dp,n*4,cudaMemcpyDeviceToHost);
    double s=0,sum=0; float mx=0; for(int i=0;i<n;i++){sum+=v[i]; s+=(double)v[i]*v[i]; if(fabsf(v[i])>mx)mx=fabsf(v[i]);}
    fprintf(stderr,"    ssm.%s sum=%.6f |.|=%.4f max=%.4f (n=%d)\n",t,sum,sqrt(s),mx,n); free(v);
}
static void ssm_layer(q36_engine*e,dblock*db,int L,const float*x,int pos){
    /* qkv(8192) + z-gate(4096) fused into one launch */
    k_matvec_multi3<<<gr(CONVD+VALD,8),256>>>(db->qkv.d,CONVD,e->tmp,
        db->gate.d,VALD,e->vbuf, NULL,0,NULL, x,Q36_D_MODEL);
    if(L==0 && getenv("Q36_DEBUG")){ cudaDeviceSynchronize(); d3("z[0..2]",e->vbuf); d3("z[4093..]",e->vbuf+4093); d3("qkv",e->tmp); }
    /* Measured dead end #3: fusing the 7-kernel SSM chain into k_ssm_mega is
     * CORRECT but 273 -> 219 t/s.  Under CUDA graphs the launch savings are
     * ~1us/node, while the fused kernel's 16-block grid occupies 16 of 170
     * SMs and the per-thread state array spills registers in the scan phase.
     * Kept behind Q36_MEGA=1 for reference. */
    if(getenv("Q36_MEGA")){
        k_ssm_mega<<<NKH,256>>>(e->tmp,e->vbuf,x,db->conv1d,e->convhist[L],
            db->alpha,db->dt_bias,db->ssm_a,db->ssm_norm,e->Sstate[L],e->attn,
            NVH,NKH,HKD,HVD);
        mv(&db->ssm_out,e->attn,e->x);
        return;
    }
    /* conv over CONVD=8192 channels */
    k_dn_conv<<<gr(CONVD,256),256>>>(e->tmp,db->conv1d,e->convhist[L],e->kbuf,CONVD,Q36_SSM_CONV_K);
    /* split conv output: q[KEYD] k[KEYD] v[VALD] */
    float *qc=e->kbuf, *kc=e->kbuf+KEYD, *vc=e->kbuf+2*KEYD;
    k_head_l2norm<<<NKH,64>>>(qc,HKD,Q36_RMS_EPS);
    k_head_l2norm<<<NKH,64>>>(kc,HKD,Q36_RMS_EPS);
    /* alpha+beta projections: one M=64 matvec over load-time-concatenated
     * weights (alpha rows 0..31, beta rows 32..63) */
    k_matvec_f32_row<<<2*NVH,256>>>(db->alpha,x,e->g512,2*NVH,Q36_D_MODEL);
    float *g=e->q, *beta=e->q+NVH;       /* reuse scratch */
    k_dn_gates<<<gr(NVH,64),64>>>(e->g512,db->dt_bias,db->ssm_a,e->g512+NVH,g,beta,NVH);
    if(L==0 && getenv("Q36_DEBUG")){ cudaDeviceSynchronize();
        /* sum fingerprints to diff against the oracle's per-tensor sums */
        dN("SUM_qkv",e->tmp,CONVD);
        dN("SUM_conv_silu",e->kbuf,CONVD);
        dN("SUM_q_post_l2",qc,KEYD);
        dN("SUM_k_post_l2",kc,KEYD);
        dN("SUM_v",vc,VALD);
        dN("SUM_z",e->vbuf,VALD); }
    /* delta scan: 8 warps/block (one per state column), grid covers all heads */
    k_dn_scan<<<dim3((HVD+7)/8,NVH),256>>>(qc,kc,vc,g,beta,e->Sstate[L],e->attn,NVH,NKH,HKD,HVD);
    if(L==0 && getenv("Q36_DEBUG")){ cudaDeviceSynchronize(); dN("SUM_g",g,NVH); dN("SUM_beta",beta,NVH); dN("SUM_scan",e->attn,VALD); }
    /* gated group norm with z */
    k_dn_gnorm<<<NVH,128>>>(e->attn,db->ssm_norm,e->vbuf,NVH,HVD,Q36_RMS_EPS);
    if(L==0 && getenv("Q36_DEBUG")){ cudaDeviceSynchronize(); dN("SUM_gnorm",e->attn,VALD); }
    /* out proj VALD->2048 */
    mv(&db->ssm_out,e->attn,e->x);
    if(L==0 && getenv("Q36_DEBUG")){ cudaDeviceSynchronize(); dN("SUM_ssm_out",e->x,Q36_D_MODEL); }
}

static int g_dbg=-1;
static void dbg_norm(q36_engine*e,const char*tag,int L){
    if(g_dbg<0) g_dbg=getenv("Q36_DEBUG")?1:0;
    if(!g_dbg) return;
    float hb[Q36_D_MODEL]; CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hb,e->h,sizeof hb,cudaMemcpyDeviceToHost));
    double s=0; int bad=0; float mx=0; for(int i=0;i<Q36_D_MODEL;i++){float v=hb[i]; s+=(double)v*v; if(!isfinite(v))bad++; if(fabsf(v)>mx)mx=fabsf(v);}
    fprintf(stderr,"  [%s L%02d] |h|=%.3f max=%.3f nan=%d  h[0..3]=%.3f %.3f %.3f\n",tag,L,sqrt(s),mx,bad,hb[0],hb[1],hb[2]);
}
/* ---- on-device sampling (graph-compatible) -----------------------------
 * temp/top-k/top-p with persistent xorshift RNG state in device memory, so a
 * captured graph replays with fresh randomness.  Two stages: per-block top-K
 * candidates, then one block merges + softmax(T) + nucleus cut + sample. */
__global__ void k_sample_p1(const float*logits,int n,float*pv,int*pi,int k){Q36_GDS();
    __shared__ float sv[2048]; __shared__ int si[2048];
    int nb=gridDim.x, slice=(n+nb-1)/nb;
    int s0=blockIdx.x*slice, s1=min(s0+slice,n);
    for(int i=threadIdx.x;i<2048;i+=blockDim.x){
        int gi=s0+i;
        sv[i]=(gi<s1)?logits[gi]:-1e30f; si[i]=gi;
    }
    __syncthreads();
    for(int r=0;r<k;r++){
        /* tree argmax over smem */
        for(int o=1024;o>0;o>>=1){
            for(int i=threadIdx.x;i<o;i+=blockDim.x)
                if(sv[i+o]>sv[i]){ sv[i]=sv[i+o]; si[i]=si[i+o]; }
            __syncthreads();
        }
        if(threadIdx.x==0){
            pv[blockIdx.x*k+r]=sv[0]; pi[blockIdx.x*k+r]=si[0];
        }
        __syncthreads();
        /* reload smem minus picked (mark by index) */
        int picked=si[0];
        for(int i=threadIdx.x;i<2048;i+=blockDim.x){
            int gi=s0+i;
            sv[i]=(gi<s1&&gi!=picked)?logits[gi]:-1e30f; si[i]=gi;
        }
        /* remove all previously picked */
        for(int r2=0;r2<r;r2++){
            int p2=pi[blockIdx.x*k+r2];
            if(p2>=s0&&p2<s1&&((p2-s0)%blockDim.x)==(int)threadIdx.x%1) {}
        }
        for(int i=threadIdx.x;i<2048;i+=blockDim.x){
            int gi=s0+i;
            for(int r2=0;r2<=r;r2++) if(pi[blockIdx.x*k+r2]==gi){ sv[i]=-1e30f; }
        }
        __syncthreads();
    }
}
__global__ void k_sample_p2(const float*pv,const int*pi,int nparts,int k,
                            float temp,float topp,unsigned long long*rng,int*out){Q36_GDS();
    __shared__ float sv[4096]; __shared__ int si[4096];
    int n=nparts*k;
    for(int i=threadIdx.x;i<4096;i+=blockDim.x){
        sv[i]=(i<n)?pv[i]:-1e30f; si[i]=(i<n)?pi[i]:0;
    }
    __syncthreads();
    __shared__ float tv[Q36_SAMPLE_KMAX]; __shared__ int ti[Q36_SAMPLE_KMAX];
    for(int r=0;r<k;r++){
        for(int o=2048;o>0;o>>=1){
            for(int i=threadIdx.x;i<o;i+=blockDim.x)
                if(sv[i+o]>sv[i]){ sv[i]=sv[i+o]; si[i]=si[i+o]; }
            __syncthreads();
        }
        if(threadIdx.x==0){ tv[r]=sv[0]; ti[r]=si[0]; }
        __syncthreads();
        for(int i=threadIdx.x;i<4096;i+=blockDim.x){
            int gi2=(i<n)?pi[i]:-1;
            bool dead=false;
            for(int r2=0;r2<=r;r2++) if(ti[r2]==gi2) dead=true;
            sv[i]=(i<n&&!dead)?pv[i]:-1e30f; si[i]=gi2<0?0:gi2;
        }
        __syncthreads();
    }
    if(threadIdx.x==0){
        /* temp softmax over the sorted-desc top-k */
        float mx=tv[0], s=0, p[Q36_SAMPLE_KMAX];
        for(int i=0;i<k;i++){ p[i]=__expf((tv[i]-mx)/temp); s+=p[i]; }
        /* nucleus cut */
        float cum=0; int keep=k;
        for(int i=0;i<k;i++){ cum+=p[i]/s; if(cum>=topp){ keep=i+1; break; } }
        float s2=0; for(int i=0;i<keep;i++) s2+=p[i];
        /* xorshift64* */
        unsigned long long x=*rng;
        x^=x>>12; x^=x<<25; x^=x>>27; *rng=x;
        float u=(float)((x*2685821657736338717ULL)>>40)/16777216.f*s2;
        float acc2=0; int pick=ti[0];
        for(int i=0;i<keep;i++){ acc2+=p[i]; if(u<=acc2){ pick=ti[i]; break; } }
        *out=pick;
    }
}

/* One decode step's kernel sequence; token/pos read from device.  Safe to
 * capture in a CUDA graph when debug dumps are off. */
static void q36_decode_body(q36_engine*e,int maxch){
    k_embed_dt<<<gr(Q36_D_MODEL,256),256>>>(e->tok_embd,e->d_tok2,e->h,Q36_D_MODEL);
    dbg_norm(e,"embed",-1);
    /* residual-add + next-layer norm fused into one kernel per boundary;
     * the final boundary fuses the output norm feeding the LM head. */
    k_rmsnorm<<<1,1024>>>(e->h,e->blk[0].attn_norm,e->x,Q36_D_MODEL,Q36_RMS_EPS);
    for(int L=0;L<Q36_N_LAYER;L++){
        dblock*db=&e->blk[L];
        if(db->is_attn) attn_layer(e,db,L,e->x,maxch); else ssm_layer(e,db,L,e->x,0);
        k_add_rmsnorm<<<1,1024>>>(e->h,e->x,db->post_norm,e->x,Q36_D_MODEL,Q36_RMS_EPS);
        dbg_norm(e,db->is_attn?"attn":"ssm ",L);
        moe_ffn(e,db,e->x,L);
        const float*nw=(L+1<Q36_N_LAYER)?e->blk[L+1].attn_norm:e->out_norm;
        k_add_rmsnorm<<<1,1024>>>(e->h,e->moe,nw,e->x,Q36_D_MODEL,Q36_RMS_EPS);
        dbg_norm(e,"moe ",L);
    }
    if(e->s_temp>0.f){
        mv(&e->output,e->x,e->logits);
        k_sample_p1<<<128,256>>>(e->logits,Q36_N_VOCAB,e->d_pv,e->d_pi,e->s_topk);
        k_sample_p2<<<1,256>>>(e->d_pv,e->d_pi,128,e->s_topk,e->s_temp,e->s_topp,e->d_rng,e->d_argmax);
    } else {
        /* fused dot+argmax (skips the logits round-trip); the SAME kernel
         * serves the MTP verify head at B=2 so greedy tie-breaking is
         * identical between plain and speculative decode. */
        int Rmix=(e->output.elayout==5)?(int)e->output.row_bytes:0;
        k_head_argmax_p1<<<gr(Q36_N_VOCAB,8),256>>>(e->output.elayout,e->output.d,e->x,Rmix,
            e->output.M,e->output.K,1,0,e->v_pv,e->v_pi,e->v_nblk);
        k_head_argmax_p2<<<1,256>>>(e->v_pv,e->v_pi,e->v_nblk,1,e->d_argmax);
    }
}

/* ================= multi-tenant batched decode =========================
 * One step advances ALL nslots sequences.  Weight-bound work (dense q8
 * projections, shared experts) runs through the batched GEMVs (weights
 * read once per step); state-bound work (attention over per-slot KV, SSM
 * scan over per-slot state, routed experts, LM head, samplers) loops per
 * slot -- it is per-sequence traffic regardless, and looping reuses the
 * verified solo kernels with slot-offset pointers.  Token/position come
 * from device arrays, so ONE captured graph serves every step; sampler
 * params are host-baked per slot (recapture on change, like solo). */
static void q36_decode_body_mt(q36_engine*e,int maxch){
    int B=e->nslots;
    enum{D=Q36_D_MODEL,NU=Q36_N_EXPERT_USED,FF=Q36_EXPERT_FF};
    float scale=1.f/sqrtf((float)Q36_HEAD_DIM);
    static int nomma=-1;
    if(nomma<0) nomma=getenv("Q36_NOMMA")?1:0;
    for(int i=0;i<B;i++)
        k_embed_dt<<<gr(D,256),256>>>(e->tok_embd,e->d_tok_b+i,e->hb+(size_t)i*D,D);
    for(int i=0;i<B;i++)
        k_rmsnorm<<<1,1024>>>(e->hb+(size_t)i*D,e->blk[0].attn_norm,e->xb+(size_t)i*D,D,Q36_RMS_EPS);
    for(int L=0;L<Q36_N_LAYER;L++){
        dblock*db=&e->blk[L];
        if(db->is_attn){
            k_matvec_multi3_b<<<gr(Q36_Q_DIM*2+2*Q36_KV_DIM,8),256>>>(
                db->q.d,Q36_Q_DIM*2,e->tmpb,8192, db->k.d,Q36_KV_DIM,e->kbufb,8192,
                db->v.d,Q36_KV_DIM,e->vbufb,4096, e->xb,D,B,D);
            for(int i=0;i<B;i++){
                float*q=e->qb+(size_t)i*4096, *gate=e->gateb+(size_t)i*4096;
                float*kx=e->kbufb+(size_t)i*8192, *vx=e->vbufb+(size_t)i*4096;
                k_split_qgate<<<gr(Q36_Q_DIM,256),256>>>(e->tmpb+(size_t)i*8192,q,gate,Q36_N_HEAD,Q36_HEAD_DIM);
                k_head_rmsnorm<<<Q36_N_HEAD,64>>>(q,db->q_norm,Q36_HEAD_DIM,Q36_RMS_EPS);
                k_head_rmsnorm<<<Q36_N_HEAD_KV,64>>>(kx,db->k_norm,Q36_HEAD_DIM,Q36_RMS_EPS);
                k_rope_p<<<Q36_N_HEAD,Q36_ROT_DIM/2>>>(q,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_pos_b+i,Q36_ROPE_FREQ_BASE);
                k_rope_p<<<Q36_N_HEAD_KV,Q36_ROT_DIM/2>>>(kx,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_pos_b+i,Q36_ROPE_FREQ_BASE);
                k_kv_append<<<gr(2*(Q36_KV_DIM/32)*32,256),256>>>(kx,vx,
                    KCS(e,L,i),KSS(e,L,i),VCS(e,L,i),VSS(e,L,i),e->d_pos_b+i,Q36_KV_DIM,e->kvq);
                float*aout=e->attnb+(size_t)i*4096;
                /* mirror solo attn_layer dispatch EXACTLY so equal-depth
                 * batches are bit-identical: mma+merge only when maxch>1 and
                 * fp16; else the scalar per-q-head kernel (merge-skip at
                 * maxch==1, merge otherwise). */
                if(!e->kvq && !nomma && maxch>1){
                    k_attn_dec_mma<2><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(q,
                        (const __half*)KCS(e,L,i),(const __half*)VCS(e,L,i),
                        e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos_b+i,scale);
                    k_attn_merge2<<<dim3(Q36_N_HEAD,Q36_HEAD_DIM/32),256>>>(
                        e->pacc,e->pms,aout,maxch*4,e->d_pos_b+i);
                } else {
                    k_attn_partial<<<dim3(maxch,Q36_N_HEAD),256>>>(q,
                        KCS(e,L,i),KSS(e,L,i),VCS(e,L,i),VSS(e,L,i),
                        e->pacc,e->pms,(maxch==1)?aout:NULL,
                        Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos_b+i,scale,e->kvq);
                    if(maxch>1)
                        k_attn_merge<<<Q36_N_HEAD,Q36_HEAD_DIM>>>(e->pacc,e->pms,aout,
                            Q36_HEAD_DIM,maxch,e->d_pos_b+i,Q36_ATTN_CHUNK);
                }
                k_sigmoid_mul<<<gr(Q36_Q_DIM,256),256>>>(aout,gate,Q36_Q_DIM);
            }
            k_matvec_gb<<<gr(D,8),256>>>(1,db->o.d,e->attnb,e->xb,D,Q36_Q_DIM,B,4096,D);
        } else {
            k_matvec_multi3_b<<<gr(CONVD+VALD,8),256>>>(
                db->qkv.d,CONVD,e->tmpb,8192, db->gate.d,VALD,e->vbufb,4096,
                NULL,0,NULL,0, e->xb,D,B,D);
            for(int i=0;i<B;i++){          /* per-slot: conv, l2norm, gates (cheap) */
                float*tmp=e->tmpb+(size_t)i*8192, *kb=e->kbufb+(size_t)i*8192;
                k_dn_conv<<<gr(CONVD,256),256>>>(tmp,db->conv1d,
                    e->convhist[L]+(size_t)i*CONVD*3,kb,CONVD,Q36_SSM_CONV_K);
                k_head_l2norm<<<NKH,64>>>(kb,HKD,Q36_RMS_EPS);
                k_head_l2norm<<<NKH,64>>>(kb+KEYD,HKD,Q36_RMS_EPS);
                float*ab=e->g512b+(size_t)i*4096;
                k_matvec_f32_row<<<2*NVH,256>>>(db->alpha,e->xb+(size_t)i*D,ab,2*NVH,D);
                k_dn_gates<<<gr(NVH,64),64>>>(ab,db->dt_bias,db->ssm_a,ab+NVH,
                    e->qb+(size_t)i*4096,e->qb+(size_t)i*4096+NVH,NVH);
            }
            /* scan: ONE launch across all tenants (grid.z) */
            k_dn_scan_b<<<dim3((HVD+7)/8,NVH,B),256>>>(
                e->kbufb,e->kbufb+KEYD,e->kbufb+2*KEYD,
                e->qb,e->qb+NVH,e->Sstate[L],e->attnb,
                NVH,NKH,HKD,HVD, 8192,4096,(uint64_t)NVH*HKD*HVD,4096);
            for(int i=0;i<B;i++)          /* gnorm per-slot (cheap) */
                k_dn_gnorm<<<NVH,128>>>(e->attnb+(size_t)i*4096,db->ssm_norm,
                    e->vbufb+(size_t)i*4096,NVH,HVD,Q36_RMS_EPS);
            k_matvec_gb<<<gr(D,8),256>>>(1,db->ssm_out.d,e->attnb,e->xb,D,VALD,B,4096,D);
        }
        for(int i=0;i<B;i++)
            k_add_rmsnorm<<<1,1024>>>(e->hb+(size_t)i*D,e->xb+(size_t)i*D,db->post_norm,
                                      e->xb+(size_t)i*D,D,Q36_RMS_EPS);
        /* MoE: router batched; routed experts per slot (expert overlap at
         * B<=16 is ~10%: not worth cross-slot dispatch in v1) */
        k_matvec_f32_b<<<gr(Q36_N_EXPERT+1,8),256>>>(db->router,e->xb,e->rlogitsb,
            Q36_N_EXPERT+1,D,B,D,Q36_N_EXPERT+1);
        for(int i=0;i<B;i++)             /* router top-k is tiny; per-slot */
            k_router_topk<<<1,32>>>(e->rlogitsb+(size_t)i*(Q36_N_EXPERT+1),
                Q36_N_EXPERT,NU,e->d_topkb+(size_t)i*NU,e->d_topwb+(size_t)i*NU,
                e->expert_scale,1);
        /* gate+up and down: ONE launch each across all B*NU (tenant,expert)
         * pairs -- fills the GPU and reuses overlapping experts' weights */
        dim3 gg(gr(FF,8),B*NU,2);
        k_expert_gemv2_bt<<<gg,256>>>(db->gate_exps.elayout,db->gate_exps.d,db->gate_exps.expert_stride,
            db->up_exps.elayout,db->up_exps.d,db->up_exps.expert_stride,
            e->se0,e->se1,e->d_topkb,e->xb,e->g512b,e->u512b,FF,D,NU,D);
        k_silu_mul_slots<<<gr(B*NU*FF,256),256>>>(e->g512b,e->u512b,B*NU*FF);
        dim3 gd(gr(D,8),B*NU);
        k_expert_gemv_bt<<<gd,256>>>(db->down_exps.elayout,db->down_exps.d,db->down_exps.expert_stride,
            e->se0,e->se1,e->d_topkb,e->g512b,FF,e->d2048b,D,FF);
        k_matvec_multi3_b<<<gr(2*FF,8),256>>>(db->sh_gate.d,FF,e->sh_ab,FF,
            db->sh_up.d,FF,e->sh_bb,FF, NULL,0,NULL,0, e->xb,D,B,D);
        k_silu_mul<<<gr(B*FF,256),256>>>(e->sh_ab,e->sh_bb,B*FF);
        k_matvec_gb<<<gr(D,8),256>>>(1,db->sh_down.d,e->sh_ab,e->shexpb,D,FF,B,FF,D);
        for(int i=0;i<B;i++)
            k_moe_combine<<<gr(D,256),256>>>(e->moeb+(size_t)i*D,e->d2048b+(size_t)i*NU*D,
                e->d_topwb+(size_t)i*NU,e->shexpb+(size_t)i*D,
                e->rlogitsb+(size_t)i*(Q36_N_EXPERT+1)+Q36_N_EXPERT,D,NU);
        const float*nw=(L+1<Q36_N_LAYER)?e->blk[L+1].attn_norm:e->out_norm;
        for(int i=0;i<B;i++)
            k_add_rmsnorm<<<1,1024>>>(e->hb+(size_t)i*D,e->moeb+(size_t)i*D,nw,
                                      e->xb+(size_t)i*D,D,Q36_RMS_EPS);
    }
    /* final hidden states left in e->xb; the per-slot LM head + sampler run
     * EAGERLY (not captured) in step_mt -- they share e->logits/argmax
     * scratch, which cannot be safely serialized inside one graph. */
}
/* per-slot LM head + greedy/sampled pick; eager, sequential on the stream */
static void q36_mt_heads_n(q36_engine*e,int B){
    enum{D=Q36_D_MODEL};
    const dw*W=&e->output;
    int all_greedy=1; for(int i=0;i<B;i++) if(e->mt_temp[i]>0.f) all_greedy=0;
    if(all_greedy){
        /* fused head reads the 0.5GB weights ONCE per tenant-tile of <=16
         * (the kernel keeps a [16] per-tenant register array); ceil(B/16)
         * passes covers any B while still amortizing the head across tiles */
        int Rmix=(W->elayout==5)?(int)W->row_bytes:0;
        for(int b0=0;b0<B;b0+=16){
            int bt2=(B-b0<16)?(B-b0):16;
            k_head_argmax_p1<<<gr(Q36_N_VOCAB,8),256>>>(W->elayout,W->d,
                e->xb+(size_t)b0*D,Rmix,W->M,W->K,bt2,D,e->hd_pv,e->hd_pi,e->hd_nblk);
            k_head_argmax_p2<<<bt2,256>>>(e->hd_pv,e->hd_pi,e->hd_nblk,bt2,e->d_out_b+b0);
        }
        return;
    }
    for(int i=0;i<B;i++){   /* mixed/sampled: per-slot fallback (shared logits) */
        mv(W,e->xb+(size_t)i*D,e->logits);
        if(e->mt_temp[i]>0.f){
            k_sample_p1<<<128,256>>>(e->logits,Q36_N_VOCAB,e->d_pv,e->d_pi,e->mt_topk[i]);
            k_sample_p2<<<1,256>>>(e->d_pv,e->d_pi,128,e->mt_topk[i],e->mt_temp[i],
                e->mt_topp[i],e->d_rng_b+i,e->d_out_b+i);
        } else {
            q36_cuda_argmax_async(e->logits,Q36_N_VOCAB,e->d_out_b+i);
        }
        CK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

extern "C" int q36_engine_prefill_from(q36_engine*e,const int*toks,int n,int pos0);
extern "C" int q36_engine_prefill_slot(q36_engine*e,int slot,const int*toks,int n,int pos0){
    e->pf_slot=slot;
    int r=q36_engine_prefill_from(e,toks,n,pos0);
    e->pf_slot=0;
    return r;
}
extern "C" void q36_engine_slot_sampler(q36_engine*e,int slot,float temp,int topk,float topp,
                                        unsigned long long seed){
    e->mt_temp[slot]=temp;
    e->mt_topk[slot]=(topk<1)?1:(topk>Q36_SAMPLE_KMAX?Q36_SAMPLE_KMAX:topk);
    e->mt_topp[slot]=topp;
    CK(cudaMemcpy(e->d_rng_b+slot,&seed,8,cudaMemcpyHostToDevice));
    if(e->gexec_mt){ cudaGraphExecDestroy(e->gexec_mt); e->gexec_mt=NULL; }  /* recapture */
}
static void q36_decode_body_bt(q36_engine*e,int B);
extern "C" void q36_engine_slot_move(q36_engine*e,int dst,int src){
    if(dst==src) return;
    size_t kc=(size_t)e->ctx*Q36_KV_DIM*2, ks=(size_t)e->ctx*(Q36_KV_DIM/32)*2;
    size_t vc=(size_t)e->ctx*Q36_KV_DIM*2, vs=(size_t)e->ctx*(Q36_KV_DIM/32);
    size_t ss=(size_t)NVH*HKD*HVD*4, ch=(size_t)CONVD*3*4;
    for(int L=0;L<Q36_N_LAYER;L++){
        if(e->blk[L].is_attn){
            CK(cudaMemcpyAsync(KCS(e,L,dst),KCS(e,L,src),kc,cudaMemcpyDeviceToDevice));
            CK(cudaMemcpyAsync(KSS(e,L,dst),KSS(e,L,src),ks,cudaMemcpyDeviceToDevice));
            CK(cudaMemcpyAsync(VCS(e,L,dst),VCS(e,L,src),vc,cudaMemcpyDeviceToDevice));
            CK(cudaMemcpyAsync(VSS(e,L,dst),VSS(e,L,src),vs,cudaMemcpyDeviceToDevice));
        } else {
            CK(cudaMemcpyAsync(e->Sstate[L]+(size_t)dst*NVH*HKD*HVD,
                               e->Sstate[L]+(size_t)src*NVH*HKD*HVD,ss,cudaMemcpyDeviceToDevice));
            CK(cudaMemcpyAsync(e->convhist[L]+(size_t)dst*CONVD*3,
                               e->convhist[L]+(size_t)src*CONVD*3,ch,cudaMemcpyDeviceToDevice));
        }
    }
}
extern "C" void q36_engine_step_active(q36_engine*e,int na,const int*toks,const int*pos,int*out){
    if(na<=0) return;
    static const int BK[]={8,16,32,48,64}, NBK=5;
    int B=e->nslots; for(int i=0;i<NBK;i++) if(BK[i]>=na && BK[i]<=e->nslots){ B=BK[i]; break; }
    CK(cudaMemcpy(e->d_tok_b,toks,(size_t)na*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(e->d_pos_b,pos,(size_t)na*4,cudaMemcpyHostToDevice));
    if(B>na){ CK(cudaMemsetAsync(e->d_tok_b+na,0,(size_t)(B-na)*4));
              CK(cudaMemsetAsync(e->d_pos_b+na,0,(size_t)(B-na)*4)); }
    int csz=Q36_ATTN_CHUNK2, maxpos=0; for(int i=0;i<na;i++) if(pos[i]>maxpos)maxpos=pos[i];
    int need=(maxpos+1+csz-1)/csz; int cap=(need<=1)?1:((need+7)&~7);
    int maxcap=(e->ctx+csz-1)/csz; if(cap>maxcap)cap=maxcap;
    static int gemv=-1; if(gemv<0) gemv=getenv("Q36_MT_GEMV")?1:0;
    e->gchunks_mt=cap;
    if(getenv("Q36_NOGRAPH")){
        if(gemv){int sv=e->nslots;e->nslots=B; q36_decode_body_mt(e,cap); e->nslots=sv;}
        else q36_decode_body_bt(e,B);
    } else {
        if(!e->gexec_mt || cap>e->gcap_mt || B!=e->gB_mt){
            if(e->gexec_mt){ cudaGraphExecDestroy(e->gexec_mt); e->gexec_mt=NULL; }
            e->gcap_mt=cap; e->gB_mt=B;
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
            if(gemv){int sv=e->nslots;e->nslots=B; q36_decode_body_mt(e,cap); e->nslots=sv;}
            else q36_decode_body_bt(e,B);
            CK(cudaStreamEndCapture(cudaStreamPerThread,&g));
            CK(cudaGraphInstantiate(&e->gexec_mt,g,0));
            cudaGraphDestroy(g);
        }
        CK(cudaGraphLaunch(e->gexec_mt,cudaStreamPerThread));
    }
    q36_mt_heads_n(e,B);
    CK(cudaMemcpy(out,e->d_out_b,(size_t)na*4,cudaMemcpyDeviceToHost));
}

/* advance all nslots sequences one token; toks/pos/out are nslots wide */
extern "C" void q36_engine_step_mt(q36_engine*e,const int*toks,const int*pos,int*out){
    int B=e->nslots;
    CK(cudaMemcpy(e->d_tok_b,toks,(size_t)B*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(e->d_pos_b,pos,(size_t)B*4,cudaMemcpyHostToDevice));
    /* grid = need bucket for the DEEPEST slot, bucketed exactly as solo
     * (cap=1 or multiple of 8) so equal-depth batches match solo bit-for-bit
     * and shallow batches stay cheap; recapture when the bucket grows. */
    int csz=Q36_ATTN_CHUNK2, maxpos=0;
    for(int i=0;i<B;i++) if(pos[i]>maxpos)maxpos=pos[i];
    int need=(maxpos+1+csz-1)/csz; int cap=(need<=1)?1:((need+7)&~7);
    int maxcap=(e->ctx+csz-1)/csz; if(cap>maxcap)cap=maxcap;
    /* GEMV path wins at small B (weight-bound GEMV), tiled tensor-core path
     * wins at B>=~14 (crossover measured); auto-pick, override with env. */
    static int gemv=-1;
    if(gemv<0){ if(getenv("Q36_MT_GEMV"))gemv=1; else if(getenv("Q36_MT_TILED"))gemv=0;
                else gemv=(e->nslots<14)?1:0; }
    e->gchunks_mt=cap;          /* bt_layer reads this for the attention grid */
    if(getenv("Q36_NOGRAPH")){
        if(gemv) q36_decode_body_mt(e,cap); else q36_decode_body_bt(e,cap);
    } else {
        if(!e->gexec_mt || cap>e->gcap_mt){
            if(e->gexec_mt){ cudaGraphExecDestroy(e->gexec_mt); e->gexec_mt=NULL; }
            e->gcap_mt=cap;
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
            if(gemv) q36_decode_body_mt(e,cap); else q36_decode_body_bt(e,cap);
            CK(cudaStreamEndCapture(cudaStreamPerThread,&g));
            CK(cudaGraphInstantiate(&e->gexec_mt,g,0));
            cudaGraphDestroy(g);
        }
        CK(cudaGraphLaunch(e->gexec_mt,cudaStreamPerThread));
    }
    q36_mt_heads_n(e,e->nslots);
    CK(cudaMemcpy(out,e->d_out_b,(size_t)B*4,cudaMemcpyDeviceToHost));
}

extern "C" void q36_engine_set_kvq(q36_engine*e,int on){ e->kvq=on; }
extern "C" void q36_engine_set_sampler(q36_engine*e,float temp,int topk,float topp,unsigned long long seed){
    e->s_temp=temp;
    e->s_topk=(topk<1)?1:(topk>Q36_SAMPLE_KMAX?Q36_SAMPLE_KMAX:topk);
    e->s_topp=topp;
    CK(cudaMemcpy(e->d_rng,&seed,8,cudaMemcpyHostToDevice));
    if(e->gexec){ cudaGraphExecDestroy(e->gexec); e->gexec=NULL; }  /* recapture */
}

/* Programmatic Dependent Launch (default ON; Q36_NOPDL=1 disables):
 * rewrite the captured decode graph's kernel->kernel edges as programmatic,
 * so each grid's setup/scheduling overlaps its predecessor's drain instead
 * of waiting for a hard boundary.  Every kernel begins with Q36_GDS()
 * (grid dependency sync), so memory ordering stays exactly serial -- only
 * launch latency is hidden.  Attacks the exposed-latency share of the
 * decode roofline gap; measured +3.7%/+3.3%/+2.4% tg128 at d0/32k/90k. */
static void pdl_edges(cudaGraph_t g){
    size_t ne=0;
    if(cudaGraphGetEdges(g,NULL,NULL,NULL,&ne)!=cudaSuccess||!ne) return;
    cudaGraphNode_t*from=(cudaGraphNode_t*)malloc(ne*sizeof*from);
    cudaGraphNode_t*to  =(cudaGraphNode_t*)malloc(ne*sizeof*to);
    cudaGraphEdgeData*ed=(cudaGraphEdgeData*)calloc(ne,sizeof*ed);
    CK(cudaGraphGetEdges(g,from,to,ed,&ne));
    size_t swapped=0;
    for(size_t i=0;i<ne;i++){
        cudaGraphNodeType tf,tt;
        cudaGraphNodeGetType(from[i],&tf); cudaGraphNodeGetType(to[i],&tt);
        if(tf!=cudaGraphNodeTypeKernel||tt!=cudaGraphNodeTypeKernel) continue;
        if(ed[i].type!=cudaGraphDependencyTypeDefault) continue;  /* only plain edges */
        cudaGraphEdgeData pe; memset(&pe,0,sizeof pe);
        pe.from_port=cudaGraphKernelNodePortProgrammatic;
        pe.type=cudaGraphDependencyTypeProgrammatic;
        if(cudaGraphRemoveDependencies(g,&from[i],&to[i],&ed[i],1)==cudaSuccess &&
           cudaGraphAddDependencies(g,&from[i],&to[i],&pe,1)==cudaSuccess)
            swapped++;
        else {  /* restore the plain edge if the programmatic form was rejected */
            cudaGetLastError();
            cudaGraphAddDependencies(g,&from[i],&to[i],&ed[i],1);
        }
    }
    static int once=0;
    if(!once){ once=1;
        fprintf(stderr,"[pdl] %zu/%zu kernel edges -> programmatic\n",swapped,ne); }
    free(from); free(to); free(ed);
}

/* Decode step: uploads token+pos to device, then replays a captured graph of
 * the whole 40-layer step (recaptured when the sequence outgrows the current
 * chunk bucket).  Q36_NOGRAPH=1 or Q36_DEBUG=1 fall back to direct launch. */
extern "C" int q36_engine_step(q36_engine*e,int token,int pos){
    if(pos<0||pos>=e->ctx){   /* KV append at pos>=ctx would write OOB */
        fprintf(stderr,"q36: decode step at pos %d exceeds ctx %d -- refused\n",pos,e->ctx);
        return -1;
    }
    CK(cudaMemcpy(e->d_tok2,&token,4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(e->d_pos,&pos,4,cudaMemcpyHostToDevice));
    int csz=attn_chunk();
    int need=(pos+1+csz-1)/csz;
    if(getenv("Q36_DEBUG")||getenv("Q36_NOGRAPH")){
        q36_decode_body(e,need);
    } else {
        if(!e->gexec || need>e->gchunks){
            if(e->gexec){ cudaGraphExecDestroy(e->gexec); e->gexec=NULL; }
            /* single chunk stays exact (enables the merge-skip); beyond that,
             * capture in 8-chunk buckets so recapture stays ~per-512-tokens
             * (dead chunks are gated in-kernel on the live prefix) */
            int cap=(need<=1)?1:((need+7)&~7);
            int maxcap=(e->ctx+csz-1)/csz;
            if(cap>maxcap)cap=maxcap;
            e->gchunks=cap;
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
            q36_decode_body(e,cap);
            CK(cudaStreamEndCapture(cudaStreamPerThread,&g));
            if(!getenv("Q36_NOPDL")) pdl_edges(g);
            CK(cudaGraphInstantiate(&e->gexec,g,0));
            cudaGraphDestroy(g);
        }
        CK(cudaGraphLaunch(e->gexec,cudaStreamPerThread));
    }
    int tok; CK(cudaMemcpy(&tok,e->d_argmax,4,cudaMemcpyDeviceToHost));
    return tok;
}
extern "C" int q36_engine_step_ex(q36_engine*e,int token,int pos,int want_logits){
    (void)want_logits;
    return q36_engine_step(e,token,pos);
}

/* =========================================================================
 * Chunked prefill.
 * =========================================================================
 * Sequential prefill reads all active weights once PER TOKEN; the batched
 * path reads them once per chunk of up to Q36_PF_CHUNK tokens:
 *   - dense projections become tiled GEMMs (weights staged in smem, shared
 *     across a 16-token tile),
 *   - MoE tokens are bucketed by expert on-device and each expert runs one
 *     GEMM over its assigned tokens,
 *   - the DeltaNet scan keeps its state column in registers across the whole
 *     chunk (state traffic ~once per chunk),
 *   - attention runs a batched-Q causal kernel (one launch per layer).
 */

/* element dequant for GEMM smem staging (each element touched once/tile) */
__device__ __forceinline__ float q36_w_at(int wtype,const void*W,int M,int K,int r,int k){
    if(wtype==0){ /* q8split */
        const int8_t*qs=(const int8_t*)W;
        const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)M*K);
        return (float)qs[(uint64_t)r*K+k]*__half2float(ds[(uint64_t)r*(K/32)+k/32]);
    }
    if(wtype==1){ /* mxfp4split */
        const uint8_t*qs=(const uint8_t*)W;
        const uint8_t*es=(const uint8_t*)W+(uint64_t)M*(K/2);
        int b=k/32,j=k%32;
        uint8_t byte=qs[(uint64_t)r*(K/2)+b*16+(j&15)];
        return mxfp4_val((j<16)?(byte&0xF):(byte>>4))*q36_e8m0_half(es[(uint64_t)r*(K/32)+b]);
    }
    return ((const float*)W)[(uint64_t)r*K+k]; /* f32 */
}

/* Tensor-core GEMM: Y[t][r] = X[t][:] . W[r][:].  fp16 multiplicands staged
 * in smem (weights dequantized once per k-tile), f32 accumulate.  Block =
 * 8 warps covering 16 rows x 128 tokens; each warp owns one 16x16 wmma tile
 * per k-subtile.  X viewed col-major in smem gives B(k,t) for free. */
__global__ void k_gemm_tc(int wtype,const void*W,const float*X,float*Y,int T,int M,int K){Q36_GDS();
    __shared__ __align__(32) __half sw[16][72];
    __shared__ __align__(32) __half sx[128][72];
    int rb=blockIdx.x*16, tb=blockIdx.y*128, tid=threadIdx.x, warp=tid>>5;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=64){
        for(int i=tid;i<16*64;i+=256){ int rr=i>>6,kk=i&63;
            sw[rr][kk]=__float2half((rb+rr<M)?q36_w_at(wtype,W,M,K,rb+rr,k0+kk):0.f); }
        for(int i=tid;i<128*64;i+=256){ int tt=i>>6,kk=i&63;
            sx[tt][kk]=__float2half((tb+tt<T)?X[(size_t)(tb+tt)*K+k0+kk]:0.f); }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<4;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
            wmma::load_matrix_sync(fa,&sw[0][ks*16],72);
            wmma::load_matrix_sync(fb,&sx[warp*16][ks*16],72);
            wmma::mma_sync(acc,fa,fb,acc);
        }
        __syncthreads();
    }
    int t0=tb+warp*16;
    if(t0+16<=T && rb+16<=M){
        /* col-major store puts C(r,t) at Y[(t0+t)*M + rb+r] directly */
        wmma::store_matrix_sync(Y+(size_t)t0*M+rb,acc,M,wmma::mem_col_major);
    } else if(t0<T){
        /* ragged tile: stage then guarded scatter.  ldm for f32 wmma stores
         * must be a multiple of 4 (16 bytes) -- 20, not 17. */
        __shared__ __align__(32) float sc[8][16][20];
        wmma::store_matrix_sync(&sc[warp][0][0],acc,20,wmma::mem_row_major);
        int lane=tid&31;
        for(int i2=lane;i2<256;i2+=32){ int r=i2>>4,t=i2&15;
            if(rb+r<M && t0+t<T) Y[(size_t)(t0+t)*M+rb+r]=sc[warp][r][t]; }
    }
}

/* ---- W4A8 block-scaled MMA expert GEMM (sm_120a) ------------------------
 * Hardware-validated mma.sync.m16n8k32.kind::mxf8f6f4 (calibrated on
 * silicon via test_mma_bs.cu): A = MXFP4 expert weights, nibbles expanded to
 * E2M3-aligned byte containers (P(c)=((c&7)<<2)|((c>>3)<<5)) during smem
 * staging; B = activations pre-quantized to e4m3 + per-32 ue8m0 scales;
 * selectors immediate {0,0}; SF_A(row q)<-lane 4q (row q+8 <-lane 4q+1),
 * SF_B(col c)<-lane 4c, all byte0. */
#include <cuda_fp8.h>

/* X[T][K] f32 -> Xq e4m3 bytes + Xs[T][K/32] ue8m0 scales (warp per block) */
__global__ void k_quant_e4m3(const float*X,uint8_t*Xq,uint8_t*Xs,int T,int K){Q36_GDS();
    int nb=K/32;
    int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    if(gwarp>=T*nb)return;
    int t=gwarp/nb, b=gwarp%nb;
    float v=X[(size_t)t*K+b*32+lane];
    float a=fabsf(v);
    #pragma unroll
    for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
    int e2=(a>0.f)?(int)ceilf(log2f(a/448.f)):-127;
    if(e2<-127)e2=-127; if(e2>127)e2=127;
    /* 2^-e2 via exponent bits (e2=127 would hit the denormal edge and yield
     * 0.0, but that needs amax >= 2^126 -- unreachable for real activations) */
    __nv_fp8_e4m3 q(v*__int_as_float((127-e2)<<23));
    Xq[(size_t)t*K+b*32+lane]=*(uint8_t*)&q;
    if(lane==0) Xs[(size_t)t*nb+b]=(uint8_t)(127+e2);
}

/* block = 128 rows x 64 entries of one expert (grid.z); 8 warps; K-tiles of
 * 64 = 2 mma k-steps x 8 n-tiles.  mxfp4-split weights only (elayout==2).
 * 64-entry tiles: at ~64 routed entries/expert (T=2048 x 8/256) most experts
 * take ONE tile, so the A-tile (weights) is read once -- the 32-entry shape
 * measured 60% memory SoL on W re-reads. */
__global__ void k_gemm_expert_mma(const void*W,uint64_t estride,int se0,int se1,
                                  const int*elist,const int*ecount,int cap,
                                  const uint8_t*Xq,const uint8_t*Xs,uint64_t x_stride,
                                  float*Y,uint64_t y_stride,int M,int K){Q36_GDS();
    int ex=blockIdx.z;
    if(ex<se0||ex>=se1) return;              /* non-resident expert (shard) */
    int nt=ecount[ex];
    int tb=blockIdx.y*64;
    if(tb>=nt) return;
    const uint8_t*qsb=(const uint8_t*)W+(uint64_t)(ex-se0)*estride;
    const uint8_t*esb=qsb+(uint64_t)M*(K/2);
    __shared__ __align__(16) uint8_t swb[128][72];
    __shared__ uint8_t sae[128][2];
    __shared__ __align__(16) uint8_t sxb[64][80];
    __shared__ uint8_t sbe[64][2];
    __shared__ int stok[64];
    int rb=blockIdx.x*128, tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    if(tid<64) stok[tid]=(tb+tid<nt)?elist[(size_t)ex*cap+tb+tid]:-1;
    __syncthreads();
    float acc[32];
    #pragma unroll
    for(int i=0;i<32;i++)acc[i]=0.f;
    int g=lane>>2, t4=lane&3;
    int nkb=K/32;
    for(int k0=0;k0<K;k0+=64){
        int kb0=k0/32;
        {   /* stage A: thread expands one 32-code block (16B nibbles -> 32B) */
            int rr=tid>>1, blk=tid&1, grow=rb+rr;
            uint4 nib=(grow<M)?*(const uint4*)(qsb+(uint64_t)grow*(K/2)+(size_t)(kb0+blk)*16)
                              :make_uint4(0,0,0,0);
            const uint8_t*nb8=(const uint8_t*)&nib;
            uint8_t*dst=&swb[rr][blk*32];
            #pragma unroll
            for(int j=0;j<16;j++){
                uint8_t lo=nb8[j]&0xF, hi=nb8[j]>>4;
                dst[j]   =(uint8_t)(((lo&7)<<2)|((lo>>3)<<5));
                dst[j+16]=(uint8_t)(((hi&7)<<2)|((hi>>3)<<5));
            }
            sae[rr][blk]=(grow<M)?esb[(uint64_t)grow*nkb+kb0+blk]:127;
        }
        if(tid<128){   /* stage B: entry, k-block */
            int e2=tid>>1, blk=tid&1, ts=stok[e2];
            uint8_t*dst=&sxb[e2][blk*32];
            if(ts>=0){
                const uint8_t*src=Xq+(size_t)(ts>>3)*x_stride+(size_t)(kb0+blk)*32;
                *(uint4*)dst=*(const uint4*)src; *(uint4*)(dst+16)=*(const uint4*)(src+16);
                sbe[e2][blk]=Xs[(size_t)(ts>>3)*nkb+kb0+blk];
            } else { for(int j=0;j<32;j++)dst[j]=0; sbe[e2][blk]=127; }
        }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<2;ks++){
            unsigned a0=*(const unsigned*)&swb[warp*16+g  ][ks*32+t4*4];
            unsigned a1=*(const unsigned*)&swb[warp*16+g+8][ks*32+t4*4];
            unsigned a2=*(const unsigned*)&swb[warp*16+g  ][ks*32+t4*4+16];
            unsigned a3=*(const unsigned*)&swb[warp*16+g+8][ks*32+t4*4+16];
            unsigned sa=(t4==0)?sae[warp*16+g][ks]:(t4==1)?sae[warp*16+g+8][ks]:0u;
            #pragma unroll
            for(int nt2=0;nt2<8;nt2++){
                int col=nt2*8+g;
                /* 4B-aligned (stride 80, offset multiple of 4): single 32-bit
                 * smem reads are bit-equivalent to the little-endian packing */
                unsigned b0=*(const unsigned*)&sxb[col][ks*32+t4*4];
                unsigned b1=*(const unsigned*)&sxb[col][ks*32+t4*4+16];
                unsigned sb=(t4==0)?sbe[col][ks]:0u;
                float*d=&acc[nt2*4];
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                  ".f32.e2m1.e4m3.f32.ue8m0 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                  "%10, {0, 0}, %11, {0, 0};\n"
                  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
                  : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
            }
        }
        __syncthreads();
    }
    /* direct register scatter via the C fragment layout */
    #pragma unroll
    for(int nt2=0;nt2<8;nt2++){
        int c0=nt2*8+t4*2;
        int r0=rb+warp*16+g, r1=r0+8;
        if(tb+c0<nt){ int ts=stok[c0];
            if(r0<M) Y[(size_t)ts*y_stride+r0]=acc[nt2*4+0];
            if(r1<M) Y[(size_t)ts*y_stride+r1]=acc[nt2*4+2]; }
        if(tb+c0+1<nt){ int ts=stok[c0+1];
            if(r0<M) Y[(size_t)ts*y_stride+r0]=acc[nt2*4+1];
            if(r1<M) Y[(size_t)ts*y_stride+r1]=acc[nt2*4+3]; }
    }
}

/* Expert GEMM on int8 tensor cores (q8split expert weights, e.g. the
 * requantized down-projections).  Structure of k_gemm_expert_mma, math of
 * k_gemm_i8: elist-gathered entry rows, s8 x s8 -> s32, per-32 rescale. */
__global__ void k_gemm_expert_i8(const void*W,uint64_t estride,int se0,int se1,
                                 const int*elist,const int*ecount,int cap,
                                 const int8_t*Xq,const float*Xsf,uint64_t x_stride,
                                 float*Y,uint64_t y_stride,int M,int K){Q36_GDS();
    /* 64-entry tiles: the 16-entry shape re-read each expert's weights
     * ceil(nt/16) ~ 4x per launch and its 597us average matched the pure
     * W-traffic time (fully memory-bound). */
    int ex=blockIdx.z;
    if(ex<se0||ex>=se1) return;
    int nt=ecount[ex];
    int tb=blockIdx.y*64;
    if(tb>=nt) return;
    const int8_t*qs=(const int8_t*)W+(uint64_t)(ex-se0)*estride;
    const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)(ex-se0)*estride+(uint64_t)M*K);
    __shared__ __align__(16) int8_t swb[128][80];
    __shared__ float saf[128][2];
    __shared__ __align__(16) int8_t sxb[64][80];
    __shared__ float sbf[64][2];
    __shared__ int stok[64];
    int rb=blockIdx.x*128, tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    if(tid<64) stok[tid]=(tb+tid<nt)?elist[(size_t)ex*cap+tb+tid]:-1;
    __syncthreads();
    int nkb=K/32;
    float acc[32];
    #pragma unroll
    for(int i=0;i<32;i++)acc[i]=0.f;
    int g=lane>>2, t4=lane&3;
    int rr=tid>>1, blk=tid&1, grow=rb+rr;
    for(int k0=0;k0<K;k0+=64){
        int kb0=k0/32;
        {
            uint4 w=(grow<M)?*(const uint4*)(qs+(size_t)grow*K+k0+blk*32):make_uint4(0,0,0,0);
            *(uint4*)&swb[rr][blk*32]=w;
            uint4 w2=(grow<M)?*(const uint4*)(qs+(size_t)grow*K+k0+blk*32+16):make_uint4(0,0,0,0);
            *(uint4*)&swb[rr][blk*32+16]=w2;
            if(blk==0){
                saf[rr][0]=(grow<M)?__half2float(ds[(size_t)grow*nkb+kb0  ]):0.f;
                saf[rr][1]=(grow<M)?__half2float(ds[(size_t)grow*nkb+kb0+1]):0.f;
            }
        }
        if(tid<128){
            int e2=tid>>1, bl2=tid&1, ts=stok[e2];
            int8_t*dst=&sxb[e2][bl2*32];
            if(ts>=0){   /* entry-indexed rows */
                const int8_t*srcp=Xq+(size_t)ts*x_stride+k0+bl2*32;
                *(uint4*)dst=*(const uint4*)srcp; *(uint4*)(dst+16)=*(const uint4*)(srcp+16);
                sbf[e2][bl2]=Xsf[(size_t)ts*nkb+kb0+bl2];
            } else { *(uint4*)dst=make_uint4(0,0,0,0); *(uint4*)(dst+16)=make_uint4(0,0,0,0); sbf[e2][bl2]=0.f; }
        }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<2;ks++){
            unsigned a0=*(const unsigned*)&swb[warp*16+g  ][ks*32+t4*4];
            unsigned a1=*(const unsigned*)&swb[warp*16+g+8][ks*32+t4*4];
            unsigned a2=*(const unsigned*)&swb[warp*16+g  ][ks*32+t4*4+16];
            unsigned a3=*(const unsigned*)&swb[warp*16+g+8][ks*32+t4*4+16];
            float wsc0=saf[warp*16+g][ks], wsc1=saf[warp*16+g+8][ks];
            #pragma unroll
            for(int nt2=0;nt2<8;nt2++){
                int col=nt2*8+g;
                unsigned b0=*(const unsigned*)&sxb[col][ks*32+t4*4];
                unsigned b1=*(const unsigned*)&sxb[col][ks*32+t4*4+16];
                int d0=0,d1=0,d2=0,d3=0;
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                  : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                int c0=nt2*8+t4*2;
                float x0=sbf[c0][ks], x1=sbf[c0+1][ks];
                acc[nt2*4+0]+=(float)d0*wsc0*x0;
                acc[nt2*4+1]+=(float)d1*wsc0*x1;
                acc[nt2*4+2]+=(float)d2*wsc1*x0;
                acc[nt2*4+3]+=(float)d3*wsc1*x1;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int nt2=0;nt2<8;nt2++){
        int c0=nt2*8+t4*2;
        int r0=rb+warp*16+g, r1=r0+8;
        if(tb+c0<nt){ int ts=stok[c0];
            if(r0<M)Y[(size_t)ts*y_stride+r0]=acc[nt2*4+0];
            if(r1<M)Y[(size_t)ts*y_stride+r1]=acc[nt2*4+2]; }
        if(tb+c0+1<nt){ int ts=stok[c0+1];
            if(r0<M)Y[(size_t)ts*y_stride+r0]=acc[nt2*4+1];
            if(r1<M)Y[(size_t)ts*y_stride+r1]=acc[nt2*4+3]; }
    }
}

/* ---- int8 tensor-core GEMM (dense, split-Q8 weights) --------------------
 * Same m16n8k32 fragment geometry as the W4A8 kernel, but s8 x s8 -> s32 with
 * per-32-block rescale in registers via the calibrated C-fragment mapping.
 * Weights are used exactly as stored (split-Q8 int8 + f16 scales): no quality
 * change vs the CUDA-core path. */

/* X[T][K] f32 -> int8 + per-32 float scales */
__global__ void k_quant_q8b(const float*X,int8_t*Xq,float*Xsf,int T,int K){Q36_GDS();
    int nb=K/32;
    int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    if(gwarp>=T*nb)return;
    int t=gwarp/nb, b=gwarp%nb;
    float v=X[(size_t)t*K+b*32+lane];
    float a=fabsf(v);
    #pragma unroll
    for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
    float d=a/127.f, id=(a>0.f)?__fdividef(127.f,a):0.f;
    int q=(int)lrintf(v*id);
    Xq[(size_t)t*K+b*32+lane]=(int8_t)(q>127?127:(q<-127?-127:q));
    if(lane==0) Xsf[(size_t)t*nb+b]=d;
}

__global__ void k_gemm_i8(const void*W,const int8_t*Xq,const float*Xsf,
                          float*Y,int T,int M,int K){Q36_GDS();
    /* 64-token tiles: each block amortizes its A-tile over more tokens
     * (L2 weight re-reads scale with T/BN; the big prefill shapes measured
     * 73% memory-throughput at BN=32 -- W traffic, not X, is the cost) */
    __shared__ __align__(16) int8_t swb[128][80];
    __shared__ float saf[128][2];
    __shared__ __align__(16) int8_t sxb[64][80];
    __shared__ float sbf[64][2];
    int rb=blockIdx.x*128, tb=blockIdx.y*64, tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int8_t*qs=(const int8_t*)W;
    const __half*ds=(const __half*)((const uint8_t*)W+(uint64_t)M*K);
    int nkb=K/32;
    float acc[32];
    #pragma unroll
    for(int i=0;i<32;i++)acc[i]=0.f;
    int g=lane>>2, t4=lane&3;
    for(int k0=0;k0<K;k0+=64){
        int kb0=k0/32;
        {   /* stage 128 rows x 64 int8 + scales */
            int rr=tid>>1, blk=tid&1, grow=rb+rr;
            uint4 w=(grow<M)?*(const uint4*)(qs+(size_t)grow*K+k0+blk*32):make_uint4(0,0,0,0);
            *(uint4*)&swb[rr][blk*32]=w;
            uint4 w2=(grow<M)?*(const uint4*)(qs+(size_t)grow*K+k0+blk*32+16):make_uint4(0,0,0,0);
            *(uint4*)&swb[rr][blk*32+16]=w2;
            if(blk==0){
                saf[rr][0]=(grow<M)?__half2float(ds[(size_t)grow*nkb+kb0  ]):0.f;
                saf[rr][1]=(grow<M)?__half2float(ds[(size_t)grow*nkb+kb0+1]):0.f;
            }
        }
        if(tid<128){   /* stage 64 token rows x 64 int8 + scales */
            int e2=tid>>1, blk=tid&1, t=tb+e2;
            int8_t*dst=&sxb[e2][blk*32];
            if(t<T){
                const int8_t*src=Xq+(size_t)t*K+k0+blk*32;
                *(uint4*)dst=*(const uint4*)src; *(uint4*)(dst+16)=*(const uint4*)(src+16);
                sbf[e2][blk]=Xsf[(size_t)t*nkb+kb0+blk];
            } else { *(uint4*)dst=make_uint4(0,0,0,0); *(uint4*)(dst+16)=make_uint4(0,0,0,0); sbf[e2][blk]=0.f; }
        }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<2;ks++){
            unsigned a0=*(const unsigned*)&swb[warp*16+g  ][ks*32+t4*4];
            unsigned a1=*(const unsigned*)&swb[warp*16+g+8][ks*32+t4*4];
            unsigned a2=*(const unsigned*)&swb[warp*16+g  ][ks*32+t4*4+16];
            unsigned a3=*(const unsigned*)&swb[warp*16+g+8][ks*32+t4*4+16];
            float wsc0=saf[warp*16+g][ks], wsc1=saf[warp*16+g+8][ks];
            #pragma unroll
            for(int nt2=0;nt2<8;nt2++){
                int col=nt2*8+g;
                unsigned b0=*(const unsigned*)&sxb[col][ks*32+t4*4];
                unsigned b1=*(const unsigned*)&sxb[col][ks*32+t4*4+16];
                int d0=0,d1=0,d2=0,d3=0;
                asm volatile(
                  "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
                  : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                /* rescale via known C layout: cols t4*2, t4*2+1 of this n-tile */
                int c0=nt2*8+t4*2;
                float x0=sbf[c0][ks], x1=sbf[c0+1][ks];
                acc[nt2*4+0]+=(float)d0*wsc0*x0;
                acc[nt2*4+1]+=(float)d1*wsc0*x1;
                acc[nt2*4+2]+=(float)d2*wsc1*x0;
                acc[nt2*4+3]+=(float)d3*wsc1*x1;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int nt2=0;nt2<8;nt2++){
        int c0=tb+nt2*8+t4*2;
        int r0=rb+warp*16+g, r1=r0+8;
        if(c0<T){ if(r0<M)Y[(size_t)c0*M+r0]=acc[nt2*4+0]; if(r1<M)Y[(size_t)c0*M+r1]=acc[nt2*4+2]; }
        if(c0+1<T){ if(r0<M)Y[(size_t)(c0+1)*M+r0]=acc[nt2*4+1]; if(r1<M)Y[(size_t)(c0+1)*M+r1]=acc[nt2*4+3]; }
    }
}

/* Tensor-core expert GEMM: block = 128 rows x 16 entries of one expert
 * (grid.z), warps own 16-row slices; output always scattered through smem
 * because entry->row mapping goes via elist. */
__global__ void k_gemm_expert_tc(int wtype,const void*W,uint64_t estride,int se0,int se1,
                                 const int*elist,const int*ecount,int cap,
                                 const float*X,uint64_t x_stride,int x_by_entry,
                                 float*Y,uint64_t y_stride,int M,int K){Q36_GDS();
    int ex=blockIdx.z;
    if(ex<se0||ex>=se1) return;
    int nt=ecount[ex];
    int tb=blockIdx.y*16;
    if(tb>=nt) return;
    const uint8_t*Wb=(const uint8_t*)W+(uint64_t)(ex-se0)*estride;
    __shared__ __align__(32) __half sw[128][72];
    __shared__ __align__(32) __half sx[16][72];
    __shared__ int stok[16];
    int rb=blockIdx.x*128, tid=threadIdx.x, warp=tid>>5;
    if(tid<16) stok[tid]=(tb+tid<nt)?elist[(size_t)ex*cap+tb+tid]:-1;
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=64){
        for(int i=tid;i<128*64;i+=256){ int rr=i>>6,kk=i&63;
            sw[rr][kk]=__float2half((rb+rr<M)?q36_w_at(wtype,Wb,M,K,rb+rr,k0+kk):0.f); }
        for(int i=tid;i<16*64;i+=256){ int tt=i>>6,kk=i&63; int ts=stok[tt];
            size_t row=(ts<0)?0:(x_by_entry?(size_t)ts:(size_t)(ts>>3));
            sx[tt][kk]=__float2half((ts>=0)?X[row*x_stride+k0+kk]:0.f); }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<4;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
            wmma::load_matrix_sync(fa,&sw[warp*16][ks*16],72);
            wmma::load_matrix_sync(fb,&sx[0][ks*16],72);
            wmma::mma_sync(acc,fa,fb,acc);
        }
        __syncthreads();
    }
    __shared__ __align__(32) float sc[8][16][20];
    wmma::store_matrix_sync(&sc[warp][0][0],acc,20,wmma::mem_row_major);
    int lane=tid&31;
    for(int i2=lane;i2<256;i2+=32){ int r=i2>>4,t=i2&15;
        int rr=rb+warp*16+r;
        if(rr<M && tb+t<nt){ int ts=stok[t];
            Y[(size_t)ts*y_stride+rr]=sc[warp][r][t]; }
    }
}

/* FA-tiled causal prefill attention: block per (16-query tile, head).
 * K/V position-tiles are staged once into smem and reused by all 16 queries
 * (the untiled version re-read the whole prefix per query: ~16x more DRAM
 * traffic, the cause of the long-context prefill collapse).  Each warp owns
 * 2 whole queries -> per-warp online softmax, no cross-warp merge.  Handles
 * both KV modes (fp16 / Q8-K+MXFP4-V). */
#define Q36_FA_TQ 32   /* queries per block  */
#define Q36_FA_TK 32   /* positions per tile */
__global__ void k_attn_pf2(const float*Qb,const int8_t*Kc,const __half*Ks,
                           const uint8_t*Vc,const uint8_t*Vs,float*Ob,
                           int c0,int T,int n_head,int n_head_kv,int head_dim,
                           float scale,int kvq){Q36_GDS();
    int h=blockIdx.y, qt=blockIdx.x;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int kvh=h/(n_head/n_head_kv), kv_dim=n_head_kv*head_dim;
    int nb=kv_dim/32;
    __shared__ __align__(16) __half sK[Q36_FA_TK][256+8];
    __shared__ __align__(16) __half sV[Q36_FA_TK][256+8];
    /* warp owns queries q0,q0+1 (whole-query ownership: no merge phase) */
    int tq0=qt*Q36_FA_TQ;
    int myq[4]={tq0+warp*4, tq0+warp*4+1, tq0+warp*4+2, tq0+warp*4+3};
    float ql[4][8], m[4]={-1e30f,-1e30f,-1e30f,-1e30f}, s[4]={0,0,0,0}, racc[4][8];
    #pragma unroll
    for(int qi=0;qi<4;qi++){
        #pragma unroll
        for(int e2=0;e2<8;e2++){
            racc[qi][e2]=0.f;
            ql[qi][e2]=(myq[qi]<T)?Qb[(size_t)myq[qi]*(n_head*head_dim)+(size_t)h*head_dim+lane*8+e2]:0.f;
        }
    }
    int qmax=tq0+Q36_FA_TQ-1; if(qmax>=T)qmax=T-1;
    int maxpos=c0+qmax;                       /* highest position any query sees */
    for(int kt=0;kt<=maxpos;kt+=Q36_FA_TK){
        int tlen=min(Q36_FA_TK,maxpos+1-kt);
        /* stage K/V tile (dequant if quantized), 256 threads cooperate */
        for(int i2=threadIdx.x;i2<tlen*256;i2+=blockDim.x){
            int p2=i2>>8, d=i2&255;
            size_t pos=(size_t)(kt+p2);
            if(kvq){
                float kd=__half2float(Ks[pos*nb+kvh*8+(d>>5)]);
                sK[p2][d]=__float2half(kd*(float)((const int8_t*)Kc)[pos*kv_dim+kvh*head_dim+d]);
                uint8_t vb=Vc[pos*(kv_dim/2)+kvh*(head_dim/2)+(d>>1)];
                float vd=q36_e8m0_half(Vs[pos*nb+kvh*8+(d>>5)]);
                sV[p2][d]=__float2half(vd*mxfp4_val((d&1)?(vb>>4):(vb&0xF)));
            } else {
                sK[p2][d]=((const __half*)Kc)[pos*kv_dim+kvh*head_dim+d];
                sV[p2][d]=((const __half*)Vc)[pos*kv_dim+kvh*head_dim+d];
            }
        }
        __syncthreads();
        for(int p2=0;p2<tlen;p2++){
            int pos=kt+p2;
            float k8[8];
            #pragma unroll
            for(int e2=0;e2<8;e2++) k8[e2]=__half2float(sK[p2][lane*8+e2]);
            #pragma unroll
            for(int qi=0;qi<4;qi++){
                if(myq[qi]>=T || pos>c0+myq[qi]) continue;   /* causal */
                float dot=0;
                #pragma unroll
                for(int e2=0;e2<8;e2++) dot+=ql[qi][e2]*k8[e2];
                dot=warpsum_xor(dot)*scale;
                float mn=fmaxf(m[qi],dot), f=__expf(m[qi]-mn), pw=__expf(dot-mn);
                #pragma unroll
                for(int e2=0;e2<8;e2++)
                    racc[qi][e2]=racc[qi][e2]*f+pw*__half2float(sV[p2][lane*8+e2]);
                s[qi]=s[qi]*f+pw; m[qi]=mn;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int qi=0;qi<4;qi++){
        if(myq[qi]>=T) continue;
        float inv=1.f/s[qi];
        #pragma unroll
        for(int e2=0;e2<8;e2++)
            Ob[(size_t)myq[qi]*(n_head*head_dim)+(size_t)h*head_dim+lane*8+e2]=racc[qi][e2]*inv;
    }
}

/* Tensor-core FA prefill (FA2 scheme): block = (64-query tile, head), 4 warps
 * owning 16 queries each.  Per 32-position KV tile:
 *   1. S = Q(16x256).K^T on mma.m16n8k16 (f16 in, f32 accum, scale pre-folded
 *      into the register-resident Q fragments).
 *   2. Online softmax directly on the C fragments: a query row lives in one
 *      4-lane quad (rows g/g+8, cols t4*2+{0,1}), so row-max/row-sum are 2
 *      xor-shuffles per TILE -- not a 5-shuffle warpsum per (query,position)
 *      pair like k_attn_pf2 (~1e14 shuffle ops at 90k ctx = the collapse).
 *   3. O += P.V on a second mma; the P A-fragments are exactly the S
 *      C-fragment pairs (n-tiles 2j,2j+1 == k-step j), built in-register;
 *      V B-fragments come straight off row-major sV via ldmatrix.x4.trans.
 * KV staging is a single-buffer intra-tile cp.async pipeline (the register
 * round-trip version measured 3.2 long-scoreboard stalls per issue: DRAM
 * latency serialized with compute): V(kt) streams in during the S phase,
 * K(kt+TK) streams in during the PV phase.  --kv-quant falls back to
 * synchronous dequant staging at the same pipeline points. */
__global__ void __launch_bounds__(128,2) k_attn_pf3(
        const float*Qb,const int8_t*Kc,const __half*Ks,
        const uint8_t*Vc,const uint8_t*Vs,float*Ob,
        int ctx0,int T,float scale,int kvq){Q36_GDS();
    enum{HD=Q36_HEAD_DIM,NH=Q36_N_HEAD,NKV=Q36_N_HEAD_KV,KVD=Q36_KV_DIM,
         TQ=Q36_FA3_TQ,TK=Q36_FA3_TK,KP=HD+8,NB=KVD/32};
    __shared__ __align__(16) __half sK[TK][KP];
    __shared__ __align__(16) __half sV[TK][KP];
    /* --kv-quant: raw quantized tiles land here via cp.async and are
     * dequanted smem->smem (a global-load dequant would stall on DRAM) */
    __shared__ __align__(16) int8_t  rK[TK][HD];
    __shared__ __align__(16) uint8_t rV[TK][HD/2];
    __shared__ __align__(16) __half  rKs[TK][8];
    __shared__ __align__(16) uint8_t rVs[TK][8];
    int h=blockIdx.y, tq0=blockIdx.x*TQ;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31, g=lane>>2, t4=lane&3;
    int kvh=h/(NH/NKV);
    /* ldmatrix.x4.trans source for PV B-frags: lane -> (row, 8-col block) */
    int vrow=lane&15, vcol=(lane>>4)*8;
    int qw0=tq0+warp*16;               /* warp's first query               */
    int q0=qw0+g, q1=qw0+g+8;          /* this lane's two C-fragment rows  */
    /* Q lives in REGISTERS as ready A-fragments (reused every KV tile).
     * With Q in smem the block needed 71KB -> 1 block = 4 warps/SM, and the
     * measured CPI was 7.2 (schedulers idle on every mma/smem stall); regs
     * free the smem for 2 blocks/SM and drop 4 smem loads per S-mma. */
    unsigned qr[16][4];
    {
        const float*Q0=Qb+(size_t)q0*(NH*HD)+(size_t)h*HD;
        const float*Q1=Qb+(size_t)q1*(NH*HD)+(size_t)h*HD;
        #pragma unroll
        for(int ks=0;ks<16;ks++){
            int c=ks*16+t4*2;
            float2 x0=(q0<T)?*(const float2*)(Q0+c  ):make_float2(0.f,0.f);
            float2 x1=(q1<T)?*(const float2*)(Q1+c  ):make_float2(0.f,0.f);
            float2 y0=(q0<T)?*(const float2*)(Q0+c+8):make_float2(0.f,0.f);
            float2 y1=(q1<T)?*(const float2*)(Q1+c+8):make_float2(0.f,0.f);
            __half2 h0=__floats2half2_rn(x0.x*scale,x0.y*scale);
            __half2 h1=__floats2half2_rn(x1.x*scale,x1.y*scale);
            __half2 h2=__floats2half2_rn(y0.x*scale,y0.y*scale);
            __half2 h3=__floats2half2_rn(y1.x*scale,y1.y*scale);
            qr[ks][0]=*(unsigned*)&h0; qr[ks][1]=*(unsigned*)&h1;
            qr[ks][2]=*(unsigned*)&h2; qr[ks][3]=*(unsigned*)&h3;
        }
    }
    /* m floor -1e28 (real logits are far above; masked = -1e30 stays below)
     * keeps exp(masked-m) == 0 even before any real value is seen */
    float m0=-1e28f,m1=-1e28f,s0=0.f,s1=0.f;
    float oa[32][4];                   /* O: 32 8-dim n-tiles of m16n8 frags */
    #pragma unroll
    for(int n=0;n<32;n++){oa[n][0]=0.f;oa[n][1]=0.f;oa[n][2]=0.f;oa[n][3]=0.f;}
    int qmax=min(tq0+TQ-1,T-1), maxpos=ctx0+qmax;
    const __half*Kh=(const __half*)Kc, *Vh=(const __half*)Vc;
    unsigned pA[2][4];                 /* P fragments carried S phase -> PV  */
    /* async staging: 16B chunks, 32 per 256-half row, 8 per thread per tile */
    #define Q36_FA3_CPA(dst,src,kt_) do{ \
        for(int i2=threadIdx.x;i2<TK*(HD/8);i2+=(int)blockDim.x){ \
            int p_=i2>>5, c_=i2&31; \
            if((kt_)+p_<=maxpos) \
                q36_cpa16(&dst[p_][c_*8],src+(size_t)((kt_)+p_)*KVD+kvh*HD+c_*8); \
        }}while(0)
    /* --kv-quant: async-fetch one RAW quantized tile (K int8, V nibbles,
     * scale rows; all rows 16B-multiples except Vs at 8B) */
    #define Q36_FA3_CPARAW(kt_) do{ \
        for(int i2=threadIdx.x;i2<TK*16;i2+=(int)blockDim.x){ \
            int p_=i2>>4, c_=i2&15; \
            if((kt_)+p_<=maxpos) \
                q36_cpa16(&rK[p_][c_*16],Kc+(size_t)((kt_)+p_)*KVD+kvh*HD+c_*16); } \
        for(int i2=threadIdx.x;i2<TK*8;i2+=(int)blockDim.x){ \
            int p_=i2>>3, c_=i2&7; \
            if((kt_)+p_<=maxpos) \
                q36_cpa16(&rV[p_][c_*16],Vc+(size_t)((kt_)+p_)*(KVD/2)+kvh*(HD/2)+c_*16); } \
        for(int i2=threadIdx.x;i2<TK;i2+=(int)blockDim.x){ \
            if((kt_)+i2<=maxpos){ \
                q36_cpa16(&rKs[i2][0],Ks+(size_t)((kt_)+i2)*NB+kvh*8); \
                asm volatile("cp.async.ca.shared.global [%0],[%1],8;\n" \
                    ::"r"((unsigned)__cvta_generic_to_shared(&rVs[i2][0])), \
                      "l"(Vs+(size_t)((kt_)+i2)*NB+kvh*8)); } } \
        }while(0)
    /* smem->smem dequant of the raw tile (rows beyond maxpos zeroed) */
    #define Q36_FA3_KDQ(kt_) do{ \
        for(int i2=threadIdx.x;i2<TK*HD;i2+=(int)blockDim.x){ \
            int p_=i2>>8, d_=i2&(HD-1); \
            float x_=((kt_)+p_<=maxpos)?__half2float(rKs[p_][d_>>5])*(float)rK[p_][d_]:0.f; \
            sK[p_][d_]=__float2half(x_); \
        }}while(0)
    #define Q36_FA3_VDQ(kt_) do{ \
        for(int i2=threadIdx.x;i2<TK*(HD/2);i2+=(int)blockDim.x){ \
            int p_=i2>>7, b_=i2&(HD/2-1); \
            float sc_=((kt_)+p_<=maxpos)?q36_e8m0_half(rVs[p_][b_>>4]):0.f; \
            uint8_t vb_=rV[p_][b_]; \
            sV[p_][b_*2  ]=__float2half(sc_*mxfp4_val(vb_&0xFu)); \
            sV[p_][b_*2+1]=__float2half(sc_*mxfp4_val((uint32_t)vb_>>4)); \
        }}while(0)
    if(!kvq){ Q36_FA3_CPA(sK,Kh,0); asm volatile("cp.async.commit_group;\n"); }
    else    { Q36_FA3_CPARAW(0);    asm volatile("cp.async.commit_group;\n"); }
    for(int kt=0;kt<=maxpos;kt+=TK){
        int tlen=min(TK,maxpos+1-kt);
        int have_next=(kt+TK<=maxpos);
        int active=(qw0<T && kt<=ctx0+qw0+15);
        /* V(kt) streams during the S phase; K(kt) forced complete here */
        if(!kvq){
            Q36_FA3_CPA(sV,Vh,kt);
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n");
        } else {
            /* raw(kt) arrived (issued a full tile ago); barrier makes every
             * thread's async chunks visible before the cross-thread dequant */
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            Q36_FA3_KDQ(kt); Q36_FA3_VDQ(kt);
        }
        __syncthreads();               /* sK(kt) visible to all warps */
        if(active){
            /* ---- S = Q.K^T : 4 n-tiles of 8 positions.  (Dead end: K
             * B-frags via non-trans ldmatrix.x4 measured 2% SLOWER -- the
             * phase is tensor-bound, u32 loads dual-issue for free, and
             * ldmatrix only added latency to the mma dependency chain.) ---- */
            float sf[4][4];
            #pragma unroll
            for(int n=0;n<4;n++){sf[n][0]=0.f;sf[n][1]=0.f;sf[n][2]=0.f;sf[n][3]=0.f;}
            #pragma unroll
            for(int ks=0;ks<16;ks++){
                #pragma unroll
                for(int n=0;n<4;n++){
                    const __half*kb=&sK[n*8+g][ks*16+t4*2];
                    unsigned b0=*(const unsigned*)kb, b1=*(const unsigned*)(kb+8);
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        :"+f"(sf[n][0]),"+f"(sf[n][1]),"+f"(sf[n][2]),"+f"(sf[n][3])
                        :"r"(qr[ks][0]),"r"(qr[ks][1]),"r"(qr[ks][2]),"r"(qr[ks][3]),
                         "r"(b0),"r"(b1));
                }
            }
            /* causal mask; also masks the zero-padded tail (pos>maxpos>=ctx0+q) */
            if(kt+TK-1>ctx0+q0){
                #pragma unroll
                for(int n=0;n<4;n++){
                    int p=kt+n*8+t4*2;
                    if(p  >ctx0+q0)sf[n][0]=-1e30f;
                    if(p+1>ctx0+q0)sf[n][1]=-1e30f;
                    if(p  >ctx0+q1)sf[n][2]=-1e30f;
                    if(p+1>ctx0+q1)sf[n][3]=-1e30f;
                }
            }
            /* ---- online softmax on the fragments (2 shuffles per stat) ---- */
            float lm0=fmaxf(fmaxf(fmaxf(sf[0][0],sf[0][1]),fmaxf(sf[1][0],sf[1][1])),
                            fmaxf(fmaxf(sf[2][0],sf[2][1]),fmaxf(sf[3][0],sf[3][1])));
            float lm1=fmaxf(fmaxf(fmaxf(sf[0][2],sf[0][3]),fmaxf(sf[1][2],sf[1][3])),
                            fmaxf(fmaxf(sf[2][2],sf[2][3]),fmaxf(sf[3][2],sf[3][3])));
            #pragma unroll
            for(int o2=1;o2<4;o2<<=1){
                lm0=fmaxf(lm0,__shfl_xor_sync(0xffffffff,lm0,o2));
                lm1=fmaxf(lm1,__shfl_xor_sync(0xffffffff,lm1,o2));
            }
            float mn0=fmaxf(m0,lm0), mn1=fmaxf(m1,lm1);
            float f0=__expf(m0-mn0), f1=__expf(m1-mn1);
            /* P as PV A-fragments: k-step j <- S n-tiles {2j, 2j+1} */
            float ps0=0.f,ps1=0.f;
            #pragma unroll
            for(int n=0;n<4;n++){
                float p00=__expf(sf[n][0]-mn0), p01=__expf(sf[n][1]-mn0);
                float p10=__expf(sf[n][2]-mn1), p11=__expf(sf[n][3]-mn1);
                ps0+=p00+p01; ps1+=p10+p11;
                __half2 h0=__floats2half2_rn(p00,p01), h1=__floats2half2_rn(p10,p11);
                pA[n>>1][(n&1)*2  ]=*(unsigned*)&h0;
                pA[n>>1][(n&1)*2+1]=*(unsigned*)&h1;
            }
            #pragma unroll
            for(int o2=1;o2<4;o2<<=1){
                ps0+=__shfl_xor_sync(0xffffffff,ps0,o2);
                ps1+=__shfl_xor_sync(0xffffffff,ps1,o2);
            }
            s0=s0*f0+ps0; s1=s1*f1+ps1;
            if(__any_sync(0xffffffff,(mn0>m0)|(mn1>m1))){   /* FA2 rescale skip */
                #pragma unroll
                for(int n=0;n<32;n++){oa[n][0]*=f0;oa[n][1]*=f0;oa[n][2]*=f1;oa[n][3]*=f1;}
            }
            m0=mn0; m1=mn1;
        }
        __syncthreads();               /* sK(kt) consumed by all warps */
        /* K(kt+TK) streams during the PV phase; V(kt) forced complete */
        if(!kvq){
            if(have_next){
                Q36_FA3_CPA(sK,Kh,kt+TK);
                asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n");
            } else asm volatile("cp.async.wait_group 0;\n");
            /* zero cp.async-skipped tail rows: stale V would turn P=0 rows
             * into 0*garbage NaNs in the PV mma (K needs no zeroing: stale
             * S entries are overwritten by the causal mask assignment) */
            if(tlen<TK)
                for(int i2=threadIdx.x;i2<(TK-tlen)*HD;i2+=(int)blockDim.x)
                    sV[tlen+(i2>>8)][i2&(HD-1)]=__float2half(0.f);
        } else if(have_next){          /* raw(kt+TK) streams during PV */
            Q36_FA3_CPARAW(kt+TK);
            asm volatile("cp.async.commit_group;\n");
        }
        __syncthreads();               /* sV(kt) visible to all warps */
        if(active){
            /* ---- O += P.V (B-frags off row-major sV via ldmatrix.trans) ---- */
            #pragma unroll
            for(int j=0;j<2;j++){
                #pragma unroll
                for(int np=0;np<16;np++){   /* n-tile pair {2np, 2np+1} */
                    unsigned b00,b01,b10,b11;
                    unsigned sp=(unsigned)__cvta_generic_to_shared(&sV[j*16+vrow][np*16+vcol]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                        "{%0,%1,%2,%3},[%4];\n"
                        :"=r"(b00),"=r"(b01),"=r"(b10),"=r"(b11):"r"(sp));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        :"+f"(oa[2*np][0]),"+f"(oa[2*np][1]),"+f"(oa[2*np][2]),"+f"(oa[2*np][3])
                        :"r"(pA[j][0]),"r"(pA[j][1]),"r"(pA[j][2]),"r"(pA[j][3]),
                         "r"(b00),"r"(b01));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        :"+f"(oa[2*np+1][0]),"+f"(oa[2*np+1][1]),"+f"(oa[2*np+1][2]),"+f"(oa[2*np+1][3])
                        :"r"(pA[j][0]),"r"(pA[j][1]),"r"(pA[j][2]),"r"(pA[j][3]),
                         "r"(b10),"r"(b11));
                }
            }
        }
        __syncthreads();               /* sV(kt) consumed (next V overwrite) */
    }
    #undef Q36_FA3_CPA
    #undef Q36_FA3_CPARAW
    #undef Q36_FA3_KDQ
    #undef Q36_FA3_VDQ
    float inv0=1.f/s0, inv1=1.f/s1;
    #pragma unroll
    for(int n=0;n<32;n++){
        int d=n*8+t4*2;
        if(q0<T){ float2 w=make_float2(oa[n][0]*inv0,oa[n][1]*inv0);
            *(float2*)(Ob+(size_t)q0*(NH*HD)+(size_t)h*HD+d)=w; }
        if(q1<T){ float2 w=make_float2(oa[n][2]*inv1,oa[n][3]*inv1);
            *(float2*)(Ob+(size_t)q1*(NH*HD)+(size_t)h*HD+d)=w; }
    }
}

/* Batched-Q causal attention for prefill: one block per (query token, head),
 * per-warp online softmax over the prefix, block combine.  Replaces the
 * per-token flash-decode loop (T x 2 launches -> 1 launch per layer). */
__global__ void k_attn_pf(const float*Qb,const int8_t*Kc,const __half*Ks,
                          const uint8_t*Vc,const uint8_t*Vs,float*Ob,
                          int c0,int n_head,int n_head_kv,int head_dim,float scale,int kvq){Q36_GDS();
    int t=blockIdx.x, h=blockIdx.y;
    int seqlen=c0+t+1;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31, nw=blockDim.x>>5;
    /* byte -> (lo,hi) e2m1 value pair LUT in smem: 1 read replaces ~12 ALU */
    __shared__ float2 vlut[256];
    for(int b2=threadIdx.x;b2<256;b2+=blockDim.x)
        vlut[b2]=make_float2(mxfp4_val(b2&0xF),mxfp4_val(b2>>4));
    __syncthreads();
    int kvh=h/(n_head/n_head_kv), kv_dim=n_head_kv*head_dim;
    const float*q=Qb+(size_t)t*(n_head*head_dim)+(size_t)h*head_dim;
    float m=-1e30f,s=0.f,racc[8],ql[8];
    #pragma unroll
    for(int e2=0;e2<8;e2++){ racc[e2]=0.f; ql[e2]=q[lane*8+e2]; }
    int nb=kv_dim/32, blk=kvh*(head_dim/32)+(lane>>2);
    for(int p=warp;p<seqlen;p+=nw){
        float dot=0, mn,f,pw;
        if(kvq){
            uint2 kq=*(const uint2*)(Kc+(size_t)p*kv_dim+kvh*head_dim+lane*8);
            const int8_t*k8=(const int8_t*)&kq;
            float kd=__half2float(Ks[(size_t)p*nb+blk]);
            #pragma unroll
            for(int e2=0;e2<8;e2++) dot+=ql[e2]*(float)k8[e2];
            dot=warpsum_xor(dot*kd)*scale;
            mn=fmaxf(m,dot); f=__expf(m-mn); pw=__expf(dot-mn);
            unsigned vq=*(const unsigned*)(Vc+(size_t)p*(kv_dim/2)+kvh*(head_dim/2)+lane*4);
            const uint8_t*v4=(const uint8_t*)&vq;
            float vd=q36_e8m0_half(Vs[(size_t)p*nb+blk]);
            #pragma unroll
            for(int e2=0;e2<4;e2++){
                float2 lv=vlut[v4[e2]];
                racc[e2*2]  =racc[e2*2]*f  +pw*vd*lv.x;
                racc[e2*2+1]=racc[e2*2+1]*f+pw*vd*lv.y;
            }
        } else {
            const __half*krow=(const __half*)Kc+(size_t)p*kv_dim+kvh*head_dim;
            uint4 kq=*(const uint4*)(krow+lane*8);
            const __half*kh=(const __half*)&kq;
            #pragma unroll
            for(int e2=0;e2<8;e2++) dot+=ql[e2]*__half2float(kh[e2]);
            dot=warpsum_xor(dot)*scale;
            mn=fmaxf(m,dot); f=__expf(m-mn); pw=__expf(dot-mn);
            const __half*vrow=(const __half*)Vc+(size_t)p*kv_dim+kvh*head_dim;
            uint4 vq=*(const uint4*)(vrow+lane*8);
            const __half*vh=(const __half*)&vq;
            #pragma unroll
            for(int e2=0;e2<8;e2++) racc[e2]=racc[e2]*f+pw*__half2float(vh[e2]);
        }
        s=s*f+pw; m=mn;
    }
    __shared__ float sm[8],ss[8],sa[8][256];
    if(lane==0){ sm[warp]=m; ss[warp]=s; }
    #pragma unroll
    for(int e2=0;e2<8;e2++) sa[warp][lane*8+e2]=racc[e2];
    __syncthreads();
    if(threadIdx.x<head_dim){
        int d=threadIdx.x;
        float bm=-1e30f; for(int w=0;w<nw;w++) bm=fmaxf(bm,sm[w]);
        float bs=0,ba=0;
        for(int w=0;w<nw;w++){ float f=__expf(sm[w]-bm); bs+=ss[w]*f; ba+=sa[w][d]*f; }
        Ob[(size_t)t*(n_head*head_dim)+(size_t)h*head_dim+d]=ba/bs;
    }
}

/* Y[t][r] = X[t][:] . W[r][:]  -- tiled 16x16, K tiles of 64 staged in smem */
__global__ void k_gemm(int wtype,const void*W,const float*X,float*Y,int T,int M,int K){Q36_GDS();
    __shared__ float sw[16][64], sx[16][65];
    int rb=blockIdx.x*16, tb=blockIdx.y*16, tid=threadIdx.x;
    float acc=0;
    for(int k0=0;k0<K;k0+=64){
        for(int i=tid;i<16*64;i+=256){ int rr=i>>6,kk=i&63;
            sw[rr][kk]=(rb+rr<M)?q36_w_at(wtype,W,M,K,rb+rr,k0+kk):0.f; }
        for(int i=tid;i<16*64;i+=256){ int tt=i>>6,kk=i&63; sx[tt][kk]=(tb+tt<T)?X[(size_t)(tb+tt)*K+k0+kk]:0.f; }
        __syncthreads();
        int tr=tid>>4, tt=tid&15;
        #pragma unroll
        for(int kk=0;kk<64;kk++) acc+=sw[tr][kk]*sx[tt][kk];
        __syncthreads();
    }
    int r=rb+(tid>>4), t=tb+(tid&15);
    if(r<M&&t<T) Y[(size_t)t*M+r]=acc;
}
/* expert GEMM over that expert's gathered tokens: grid.z = expert id.
 * elist entries encode (token,slot) as t*8+s.  x_by_entry=0 reads X by token
 * (gate/up: the shared post-norm hidden), 1 reads X by entry (down: the
 * per-(token,slot) swiglu output).  Y is always written by entry. */
__global__ void k_gemm_expert(int wtype,const void*W,uint64_t estride,
                              const int*elist,const int*ecount,int cap,
                              const float*X,uint64_t x_stride,int x_by_entry,
                              float*Y,uint64_t y_stride,int M,int K){Q36_GDS();
    int ex=blockIdx.z, nt=ecount[ex];
    int tb=blockIdx.y*16;
    if(tb>=nt) return;
    const uint8_t*Wb=(const uint8_t*)W+(uint64_t)ex*estride;
    __shared__ float sw[16][64], sx[16][65];
    __shared__ int stok[16];
    int rb=blockIdx.x*16, tid=threadIdx.x;
    if(tid<16) stok[tid]=(tb+tid<nt)?elist[(size_t)ex*cap+tb+tid]:-1;
    __syncthreads();
    float acc=0;
    for(int k0=0;k0<K;k0+=64){
        for(int i=tid;i<16*64;i+=256){ int rr=i>>6,kk=i&63; sw[rr][kk]=q36_w_at(wtype,Wb,M,K,rb+rr,k0+kk); }
        for(int i=tid;i<16*64;i+=256){ int tt=i>>6,kk=i&63; int ts=stok[tt];
            size_t row=(ts<0)?0:(x_by_entry?(size_t)ts:(size_t)(ts>>3));
            sx[tt][kk]=(ts>=0)?X[row*x_stride+k0+kk]:0.f; }
        __syncthreads();
        int tr=tid>>4, tt=tid&15;
        #pragma unroll
        for(int kk=0;kk<64;kk++) acc+=sw[tr][kk]*sx[tt][kk];
        __syncthreads();
    }
    int r=rb+(tid>>4), ti=tb+(tid&15);
    if(r<M&&ti<nt){ int ts=elist[(size_t)ex*cap+ti];
        Y[(size_t)ts*y_stride+r]=acc; }
}

/* ---- batched elementwise / small kernels for the chunk ------------------ */
__global__ void k_embed_b(const block_q8_0*emb,const int*toks,float*H,int T,int D){Q36_GDS();
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=T*D)return;
    int t=i/D, d=i%D;
    const block_q8_0*b=emb+(size_t)toks[t]*(D/32)+d/32;
    H[i]=q36_fp16(b->d)*b->qs[d%32];
}
__global__ void k_rmsnorm_b(const float*x,const float*w,float*y,int n,float eps){Q36_GDS();
    int t=blockIdx.x; x+=(size_t)t*n; y+=(size_t)t*n;
    __shared__ float part[32]; __shared__ float ss;
    float loc=0; for(int i=threadIdx.x;i<n;i+=blockDim.x) loc+=x[i]*x[i];
    loc=warpsum(loc);
    int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    if(lane==0)part[wid]=loc; __syncthreads();
    if(threadIdx.x==0){float s=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)s+=part[i];ss=rsqrtf(s/n+eps);}
    __syncthreads();
    for(int i=threadIdx.x;i<n;i+=blockDim.x) y[i]=x[i]*ss*w[i];
}
/* per-head norms with a batch dimension on grid.y and a row stride */
__global__ void k_head_rmsnorm_b(float*x,const float*w,int head_dim,uint64_t row_stride,float eps){Q36_GDS();
    float*xh=x+(size_t)blockIdx.y*row_stride+(size_t)blockIdx.x*head_dim;
    __shared__ float ss; if(threadIdx.x==0)ss=0; __syncthreads();
    float loc=0; for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) loc+=xh[i]*xh[i];
    loc=warpsum(loc); if((threadIdx.x&31)==0) atomicAdd(&ss,loc);
    __syncthreads(); float sc=rsqrtf(ss/head_dim+eps);
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) xh[i]=xh[i]*sc*w[i];
}
__global__ void k_head_l2norm_b(float*x,int head_dim,uint64_t row_stride,float eps){Q36_GDS();
    float*xh=x+(size_t)blockIdx.y*row_stride+(size_t)blockIdx.x*head_dim;
    __shared__ float ss; if(threadIdx.x==0)ss=0; __syncthreads();
    float loc=0; for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) loc+=xh[i]*xh[i];
    loc=warpsum(loc); if((threadIdx.x&31)==0) atomicAdd(&ss,loc);
    __syncthreads(); float sc=rsqrtf(ss+eps);
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) xh[i]=xh[i]*sc;
}
__global__ void k_split_qgate_b(const float*Qfull,float*q,float*gate,int T,int n_head,int head_dim){Q36_GDS();
    int i=blockIdx.x*blockDim.x+threadIdx.x, per=n_head*head_dim;
    if(i>=T*per)return;
    int t=i/per, d=i%per, h=d/head_dim, dd=d%head_dim;
    q[i]   =Qfull[(size_t)t*per*2+h*head_dim*2+dd];
    gate[i]=Qfull[(size_t)t*per*2+h*head_dim*2+head_dim+dd];
}
__global__ void k_rope_b(float*x,int head_dim,int rot_dim,int pos0,uint64_t row_stride,float base){Q36_GDS();
    int t=blockIdx.y, h=blockIdx.x, i=threadIdx.x;
    if(i>=rot_dim/2)return;
    float*xh=x+(size_t)t*row_stride+(size_t)h*head_dim;
    float inv=__powf(base,-2.0f*i/rot_dim), ang=(float)(pos0+t)*inv;
    float c=__cosf(ang),s=__sinf(ang);
    float a=xh[i],b=xh[i+rot_dim/2];
    xh[i]=a*c-b*s; xh[i+rot_dim/2]=a*s+b*c;
}
__global__ void k_kv_append_b(const float*k,const float*v,int8_t*Kc,__half*Ks,
                              uint8_t*Vc,uint8_t*Vs,int pos0,int T,int kv_dim,int kvq){Q36_GDS();
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    int nb=kv_dim/32;
    if(warp>=T*2*nb)return;
    int t=warp/(2*nb), w=warp%(2*nb), pos=pos0+t;
    if(!kvq){
        if(w<nb) ((__half*)Kc)[(size_t)pos*kv_dim+w*32+lane]=__float2half(k[(size_t)t*kv_dim+w*32+lane]);
        else { int b=w-nb; ((__half*)Vc)[(size_t)pos*kv_dim+b*32+lane]=__float2half(v[(size_t)t*kv_dim+b*32+lane]); }
        return;
    }
    if(w<nb){
        float x=k[(size_t)t*kv_dim+w*32+lane], a=fabsf(x);
        #pragma unroll
        for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
        float id=(a>0.f)?127.f/a:0.f;
        int q=(int)lrintf(x*id);
        Kc[(size_t)pos*kv_dim+w*32+lane]=(int8_t)(q>127?127:(q<-127?-127:q));
        if(lane==0)Ks[(size_t)pos*nb+w]=__float2half(a/127.f);
    } else {
        int b=w-nb;
        float x=v[(size_t)t*kv_dim+b*32+lane], a=fabsf(x);
        #pragma unroll
        for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
        int e2=(a>0.f)?(int)ceilf(log2f(a/6.f)):-127;
        if(e2<-127)e2=-127;
        float inv=__int_as_float((127-e2)<<23);
        uint8_t code=e2m1_encode(x*inv);
        /* byte k holds dims (2k,2k+1) -- matches the consecutive-pair reads */
        uint8_t hi=__shfl_down_sync(0xffffffff,(unsigned)code,1)&0xF;
        if((lane&1)==0) Vc[(size_t)pos*(kv_dim/2)+b*16+(lane>>1)]=(uint8_t)(code|(hi<<4));
        if(lane==0) Vs[(size_t)pos*nb+b]=(uint8_t)(127+e2);
    }
}
/* causal depthwise conv, fully parallel over (token, channel): the k=4
 * window is chunk-local for t>=3, so only the first three tokens touch the
 * persistent history.  (The serial-per-channel version looped T tokens in
 * one thread: 0.38ms/layer at T=2048; this is ~30x more parallel.) */
__global__ void k_dn_conv_pf(const float*in,const float*w,float*hist,float*out,
                             int T,int convd,int K){Q36_GDS();
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(size_t)T*convd)return;
    int t=(int)(i/convd), c=(int)(i%convd);
    float x3=in[(size_t)t*convd+c];
    float x2=(t>=1)?in[(size_t)(t-1)*convd+c]:hist[c*3+2];
    float x1=(t>=2)?in[(size_t)(t-2)*convd+c]:hist[c*3+t+1];
    float x0=(t>=3)?in[(size_t)(t-3)*convd+c]:hist[c*3+t];
    out[i]=silu(w[c*4]*x0+w[c*4+1]*x1+w[c*4+2]*x2+w[c*4+3]*x3);
}
/* update the 3-deep history from the chunk tail (reads before writes for T<3) */
__global__ void k_dn_conv_hist(const float*in,float*hist,int T,int convd){Q36_GDS();
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=convd)return;
    float h[3];
    #pragma unroll
    for(int j=0;j<3;j++){ int gt=T-3+j;
        h[j]=(gt>=0)?in[(size_t)gt*convd+c]:hist[c*3+T+j]; }
    #pragma unroll
    for(int j=0;j<3;j++) hist[c*3+j]=h[j];
}
/* chunk scan: state column stays in registers across all T tokens */
__global__ void k_dn_scan_pf(const float*CB,const float*G,const float*B,float*S,float*O,
                             int T,int nvh,int nkh,int hkd,int hvd,int convd,int voff){Q36_GDS();
    /* sync-free: warps read q/k straight through L1 (all warps of a head hit
     * the same cache lines), eliminating 2 block syncs per token.
     * Dead end (measured): 4 j-columns per warp to amortize q/k loads --
     * the token loop is SERIAL, so warps (h x j) are the only parallelism;
     * 4x fewer warps (24 -> 6/SM) ran 1.61 -> 2.28ms despite 4x fewer loads. */
    int h=blockIdx.y;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    int j=blockIdx.x*(blockDim.x>>5)+warp;
    int kh=h%nkh;
    float*Sj=S+((size_t)h*hvd+j)*hkd;
    float sd[4];
    #pragma unroll
    for(int c=0;c<4;c++) sd[c]=Sj[c*32+lane];
    float qsc=rsqrtf((float)hkd);
    const float*qbase=CB+(size_t)kh*hkd, *kbase=CB+(size_t)(nkh+kh)*hkd;
    for(int t=0;t<T;t++){
        size_t roff=(size_t)t*convd;
        float qv[4],kv[4];
        #pragma unroll
        for(int c=0;c<4;c++){ qv[c]=qbase[roff+c*32+lane]; kv[c]=kbase[roff+c*32+lane]; }
        float expg=__expf(G[(size_t)t*nvh+h]), b=B[(size_t)t*nvh+h];
        float vj=CB[roff+voff+(size_t)h*hvd+j];
        float kvold=0;
        #pragma unroll
        for(int c=0;c<4;c++){ sd[c]*=expg; kvold+=kv[c]*sd[c]; }
        kvold=warpsum_xor(kvold);
        float delta=vj-kvold, o=0;
        #pragma unroll
        for(int c=0;c<4;c++){ sd[c]+=b*kv[c]*delta; o+=qv[c]*sd[c]; }
        o=warpsum(o);
        if(lane==0) O[(size_t)t*(nvh*hvd)+(size_t)h*hvd+j]=o*qsc;
    }
    #pragma unroll
    for(int c=0;c<4;c++) Sj[c*32+lane]=sd[c];
}
/* fixed-order partials, not atomicAdd: 4-warp float atomics made the prefill
 * nondeterministic at the ulp level (greedy streams flipped at near-ties
 * between runs); same fix as the decode k_dn_gnorm. */
__global__ void k_dn_gnorm_b(float*out,const float*gain,const float*z,int hvd,uint64_t row_stride,float eps){Q36_GDS();
    int t=blockIdx.y, h=blockIdx.x;
    float*oh=out+(size_t)t*row_stride+(size_t)h*hvd;
    const float*zh=z+(size_t)t*row_stride+(size_t)h*hvd;
    __shared__ float part[32]; __shared__ float ss;
    float loc=0; for(int i=threadIdx.x;i<hvd;i+=blockDim.x) loc+=oh[i]*oh[i];
    loc=warpsum(loc);
    if((threadIdx.x&31)==0) part[threadIdx.x>>5]=loc;
    __syncthreads();
    if(threadIdx.x==0){ float s=0; int nw=(blockDim.x+31)/32; for(int w=0;w<nw;w++)s+=part[w]; ss=s; }
    __syncthreads(); float sc=rsqrtf(ss/hvd+eps);
    for(int i=threadIdx.x;i<hvd;i+=blockDim.x) oh[i]=oh[i]*sc*gain[i]*silu(zh[i]);
}
__global__ void k_moe_dispatch(const int*topk,int*ecount,int*elist,int T,int nslots,int cap){Q36_GDS();
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=T*nslots)return;
    int e2=topk[i];
    int p=atomicAdd(&ecount[e2],1);
    elist[(size_t)e2*cap+p]=i;   /* entry = t*8+s */
}
__global__ void k_moe_combine_pf(float*moe,const float*DN,const float*w,
                                 const float*SH,const float*shg,int D,int nslots,int inc_sh){Q36_GDS();
    int t=blockIdx.y, i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=D)return;
    float acc=0;
    for(int s=0;s<nslots;s++) acc+=w[(size_t)t*nslots+s]*DN[((size_t)t*nslots+s)*D+i];
    /* multi-GPU: only one device contributes the shared expert, or the
     * all-reduce would sum it N times */
    moe[(size_t)t*D+i]=acc+(inc_sh?sigm(shg[t])*SH[(size_t)t*D+i]:0.f);
}

static void pf_gemm(q36_engine*e,int wtype,const void*W,const float*X,float*Y,int T,int M,int K){
    if(wtype==0){   /* split-Q8 weights -> int8 tensor cores, per-32 rescale */
        k_quant_q8b<<<gr(T*(K/32),8),256>>>(X,e->pfXq8,e->pfXsf,T,K);
        k_gemm_i8<<<dim3(gr(M,128),gr(T,64)),256>>>(W,e->pfXq8,e->pfXsf,Y,T,M,K);
    } else if(M<=32){
        /* skinny f32 rows (alpha/beta M=32, shared-gate M=1): the tc kernel's
         * 128-token tiles leave <=32 blocks on 170 SMs; the plain 16x16 tiler
         * splits T 8x finer and fills the GPU */
        k_gemm<<<dim3(gr(M,16),gr(T,16)),256>>>(wtype,W,X,Y,T,M,K);
    } else {
        k_gemm_tc<<<dim3(gr(M,16),gr(T,128)),256>>>(wtype,W,X,Y,T,M,K);
    }
}

/* one prefill layer through the MoE PARTIAL (leaves the summand in e->pfMOE);
 * the caller adds the (possibly all-reduced) result to pfH.  Extracted so the
 * multi-GPU driver can insert the cross-device reduce between them. */
/* MoE block shared by the batched paths (prefill + batch-tiled decode):
 * reads the residual-updated hidden in e->pfH, writes the summand to
 * e->pfMOE.  Lifted verbatim from pf_layer_moe -- one tiled implementation. */
static void pf_moe_block(q36_engine*e,dblock*db,int T){
    enum{T0=Q36_PF_CHUNK,D=Q36_D_MODEL,NU=Q36_N_EXPERT_USED,FF=Q36_EXPERT_FF};
    /* MoE: route all tokens, bucket by expert, one GEMM per expert */
    k_rmsnorm_b<<<T,256>>>(e->pfH,db->post_norm,e->pfX,D,Q36_RMS_EPS);
    pf_gemm(e,2,db->router,e->pfX,e->pfRL,T,Q36_N_EXPERT,D);
    k_router_topk<<<T,32>>>(e->pfRL,Q36_N_EXPERT,NU,e->pf_topk,e->pf_topw,e->expert_scale,1);
    CK(cudaMemsetAsync(e->pf_ecount,0,Q36_N_EXPERT*sizeof(int)));
    k_moe_dispatch<<<gr(T*NU,256),256>>>(e->pf_topk,e->pf_ecount,e->pf_elist,T,NU,T0);
    dim3 gge(gr(FF,128),gr(T,16),Q36_N_EXPERT);
    if(db->gate_exps.elayout==2 && db->up_exps.elayout==2){
        /* W4A8 block-scaled MMA: quantize the chunk activations once,
         * then feed MXFP4 weights to the tensor cores natively.
         * 64-entry tiles (grid.y matches the widened kernel). */
        dim3 gge2(gr(FF,128),gr(T,64),Q36_N_EXPERT);
        k_quant_e4m3<<<gr(T*(D/32),8),256>>>(e->pfX,e->pfXq,e->pfXs,T,D);
        k_gemm_expert_mma<<<gge2,256>>>(db->gate_exps.d,db->gate_exps.expert_stride,
            e->se0,e->se1,e->pf_elist,e->pf_ecount,T0,e->pfXq,e->pfXs,D,e->pfGU_g,FF,FF,D);
        k_gemm_expert_mma<<<gge2,256>>>(db->up_exps.d,db->up_exps.expert_stride,
            e->se0,e->se1,e->pf_elist,e->pf_ecount,T0,e->pfXq,e->pfXs,D,e->pfGU_u,FF,FF,D);
    } else {
        k_gemm_expert_tc<<<gge,256>>>(db->gate_exps.elayout==2?1:0,db->gate_exps.d,
            db->gate_exps.expert_stride,e->se0,e->se1,e->pf_elist,e->pf_ecount,T0,
            e->pfX,D,0,e->pfGU_g,FF,FF,D);
        k_gemm_expert_tc<<<gge,256>>>(db->up_exps.elayout==2?1:0,db->up_exps.d,
            db->up_exps.expert_stride,e->se0,e->se1,e->pf_elist,e->pf_ecount,T0,
            e->pfX,D,0,e->pfGU_u,FF,FF,D);
    }
    k_silu_mul<<<gr(T*NU*FF,256),256>>>(e->pfGU_g,e->pfGU_u,T*NU*FF);
    dim3 gde(gr(D,128),gr(T,16),Q36_N_EXPERT);
    if(db->down_exps.elayout==1){
        /* int8 MMA: quantize the per-entry swiglu outputs, s8 weights as
         * stored; 64-entry tiles (the tc fallback below stays at 16) */
        dim3 gde64(gr(D,128),gr(T,64),Q36_N_EXPERT);
        k_quant_q8b<<<gr(T*NU*(FF/32),8),256>>>(e->pfGU_g,e->pfXq8,e->pfXsf,T*NU,FF);
        k_gemm_expert_i8<<<gde64,256>>>(db->down_exps.d,db->down_exps.expert_stride,
            e->se0,e->se1,e->pf_elist,e->pf_ecount,T0,e->pfXq8,e->pfXsf,FF,e->pfDN,D,D,FF);
    } else {
        k_gemm_expert_tc<<<gde,256>>>(db->down_exps.elayout==2?1:0,db->down_exps.d,
            db->down_exps.expert_stride,e->se0,e->se1,e->pf_elist,e->pf_ecount,T0,
            e->pfGU_g,FF,1,e->pfDN,D,D,FF);
    }
    /* shared expert (pfGU_u is free after the silu, stream-ordered) */
    pf_gemm(e,0,db->sh_gate.d,e->pfX,e->pfGU_u,T,FF,D);
    pf_gemm(e,0,db->sh_up.d,e->pfX,e->pfSU,T,FF,D);
    k_silu_mul<<<gr(T*FF,256),256>>>(e->pfGU_u,e->pfSU,T*FF);
    pf_gemm(e,0,db->sh_down.d,e->pfGU_u,e->pfSH,T,D,FF);
    pf_gemm(e,2,db->sh_gate_inp,e->pfX,e->pfshg,T,1,D);
    k_moe_combine_pf<<<dim3(gr(D,256),T),256>>>(e->pfMOE,e->pfDN,e->pf_topw,
        e->pfSH,e->pfshg,D,NU,e->inc_shared);
}

static void pf_layer_moe(q36_engine*e,int L,int T,int c0){
    enum{T0=Q36_PF_CHUNK,D=Q36_D_MODEL,NU=Q36_N_EXPERT_USED,FF=Q36_EXPERT_FF};
    /* sharded: entries routed to non-resident experts are never written; zero
     * pfDN so the combine sums exact partials */
    if(e->se1-e->se0<Q36_N_EXPERT)
        CK(cudaMemsetAsync(e->pfDN,0,(size_t)T*NU*D*sizeof(float)));
    dblock*db=&e->blk[L];
    k_rmsnorm_b<<<T,256>>>(e->pfH,db->attn_norm,e->pfX,D,Q36_RMS_EPS);
    if(db->is_attn){
        pf_gemm(e,0,db->q.d,e->pfX,e->pfQKV,T,db->q.M,D);
        k_split_qgate_b<<<gr(T*Q36_Q_DIM,256),256>>>(e->pfQKV,e->pfQ,e->pfGATE,T,Q36_N_HEAD,Q36_HEAD_DIM);
        k_head_rmsnorm_b<<<dim3(Q36_N_HEAD,T),64>>>(e->pfQ,db->q_norm,Q36_HEAD_DIM,Q36_Q_DIM,Q36_RMS_EPS);
        pf_gemm(e,0,db->k.d,e->pfX,e->pfK,T,Q36_KV_DIM,D);
        pf_gemm(e,0,db->v.d,e->pfX,e->pfV,T,Q36_KV_DIM,D);
        k_head_rmsnorm_b<<<dim3(Q36_N_HEAD_KV,T),64>>>(e->pfK,db->k_norm,Q36_HEAD_DIM,Q36_KV_DIM,Q36_RMS_EPS);
        k_rope_b<<<dim3(Q36_N_HEAD,T),Q36_ROT_DIM/2>>>(e->pfQ,Q36_HEAD_DIM,Q36_ROT_DIM,c0,Q36_Q_DIM,Q36_ROPE_FREQ_BASE);
        k_rope_b<<<dim3(Q36_N_HEAD_KV,T),Q36_ROT_DIM/2>>>(e->pfK,Q36_HEAD_DIM,Q36_ROT_DIM,c0,Q36_KV_DIM,Q36_ROPE_FREQ_BASE);
        k_kv_append_b<<<gr(T*2*(Q36_KV_DIM/32)*32,256),256>>>(e->pfK,e->pfV,
            KCS(e,L,e->pf_slot),KSS(e,L,e->pf_slot),VCS(e,L,e->pf_slot),VSS(e,L,e->pf_slot),
            c0,T,Q36_KV_DIM,e->kvq);
        float scale=1.f/sqrtf((float)Q36_HEAD_DIM);
        static int use_pf2=-1;
        if(use_pf2<0) use_pf2=getenv("Q36_PF2")?1:0;
        if(use_pf2)   /* pre-tensor-core path kept for A/B */
            k_attn_pf2<<<dim3(gr(T,Q36_FA_TQ),Q36_N_HEAD),256>>>(e->pfQ,
                KCS(e,L,e->pf_slot),KSS(e,L,e->pf_slot),VCS(e,L,e->pf_slot),VSS(e,L,e->pf_slot),
                e->pfO,c0,T,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,scale,e->kvq);
        else
            k_attn_pf3<<<dim3(gr(T,Q36_FA3_TQ),Q36_N_HEAD),128>>>(
                e->pfQ,KCS(e,L,e->pf_slot),KSS(e,L,e->pf_slot),VCS(e,L,e->pf_slot),VSS(e,L,e->pf_slot),
                e->pfO,c0,T,scale,e->kvq);
        k_sigmoid_mul<<<gr(T*Q36_Q_DIM,256),256>>>(e->pfO,e->pfGATE,T*Q36_Q_DIM);
        pf_gemm(e,0,db->o.d,e->pfO,e->pfOUT,T,D,Q36_Q_DIM);
    } else {
        pf_gemm(e,0,db->qkv.d,e->pfX,e->pfQKV,T,CONVD,D);
        pf_gemm(e,0,db->gate.d,e->pfX,e->pfZ,T,VALD,D);
        pf_gemm(e,2,db->alpha,e->pfX,e->pfA,T,NVH,D);
        pf_gemm(e,2,db->beta,e->pfX,e->pfB,T,NVH,D);
        k_dn_gates<<<dim3(1,T),64>>>(e->pfA,db->dt_bias,db->ssm_a,e->pfB,e->pfG,e->pfBt,NVH);
        k_dn_conv_pf<<<gr(T*CONVD,256),256>>>(e->pfQKV,db->conv1d,
            e->convhist[L]+(size_t)e->pf_slot*CONVD*3,e->pfCB,T,CONVD,Q36_SSM_CONV_K);
        k_dn_conv_hist<<<gr(CONVD,256),256>>>(e->pfQKV,
            e->convhist[L]+(size_t)e->pf_slot*CONVD*3,T,CONVD);
        k_head_l2norm_b<<<dim3(NKH,T),64>>>(e->pfCB,HKD,CONVD,Q36_RMS_EPS);
        k_head_l2norm_b<<<dim3(NKH,T),64>>>(e->pfCB+KEYD,HKD,CONVD,Q36_RMS_EPS);
        k_dn_scan_pf<<<dim3(HVD/8,NVH),256>>>(e->pfCB,e->pfG,e->pfBt,
            e->Sstate[L]+(size_t)e->pf_slot*NVH*HKD*HVD,e->pfO,
            T,NVH,NKH,HKD,HVD,CONVD,2*KEYD);
        k_dn_gnorm_b<<<dim3(NVH,T),128>>>(e->pfO,db->ssm_norm,e->pfZ,HVD,VALD,Q36_RMS_EPS);
        pf_gemm(e,0,db->ssm_out.d,e->pfO,e->pfOUT,T,D,VALD);
    }
    k_add<<<gr(T*D,256),256>>>(e->pfH,e->pfOUT,T*D);
    pf_moe_block(e,db,T);
}

/* Batched prefill: chunks of Q36_PF_CHUNK tokens.  Weights are read once per
 * chunk instead of once per token.  Returns the argmax token after the final
 * prompt token (same contract as sequential prefill).
 * pos0 > 0 EXTENDS the current sequence (server prefix cache): the chunk
 * offset c0 was already threaded through rope/kv-append/attention for
 * chunks 2+, so starting it at pos0 is the same mechanism.  The SSM
 * conv-history/Sstate carry over as running state -- extension is only
 * valid if positions [0,pos0) are exactly what was previously processed
 * (hybrid state cannot rewind; the caller enforces strict extension). */
/* ================= BATCH-TILED multi-tenant decode =====================
 * Distinct from BOTH the single-stream path (q36_decode_body, untouched)
 * and the GEMV multi-tenant path (q36_decode_body_mt).  Runs all dense
 * projections and the entire MoE through the TILED tensor-core kernels
 * (pf_gemm / pf_moe_block -- the same ones prefill uses at 13k t/s), with
 * T = B tenant rows.  Only the per-sequence work (rope/KV-append/attention
 * over each slot's own cache; SSM scan over each slot's own state) stays
 * per-slot.  Reuses the prefill scratch (pfH/pfX/pfQKV/...), so no new big
 * buffers.  Hidden state for tenant i lives at pf*[i*stride]. */
static void bt_layer(q36_engine*e,dblock*db,int L,int B){
    enum{D=Q36_D_MODEL,NU=Q36_N_EXPERT_USED,FF=Q36_EXPERT_FF};
    float scale=1.f/sqrtf((float)Q36_HEAD_DIM);
    static int nomma=-1; if(nomma<0) nomma=getenv("Q36_NOMMA")?1:0;
    int maxch=e->gchunks_mt;                 /* need-bucket set by step */
    k_rmsnorm_b<<<B,256>>>(e->pfH,db->attn_norm,e->pfX,D,Q36_RMS_EPS);
    if(db->is_attn){
        pf_gemm(e,0,db->q.d,e->pfX,e->pfQKV,B,db->q.M,D);
        k_split_qgate_b<<<gr(B*Q36_Q_DIM,256),256>>>(e->pfQKV,e->pfQ,e->pfGATE,B,Q36_N_HEAD,Q36_HEAD_DIM);
        k_head_rmsnorm_b<<<dim3(Q36_N_HEAD,B),64>>>(e->pfQ,db->q_norm,Q36_HEAD_DIM,Q36_Q_DIM,Q36_RMS_EPS);
        pf_gemm(e,0,db->k.d,e->pfX,e->pfK,B,Q36_KV_DIM,D);
        pf_gemm(e,0,db->v.d,e->pfX,e->pfV,B,Q36_KV_DIM,D);
        k_head_rmsnorm_b<<<dim3(Q36_N_HEAD_KV,B),64>>>(e->pfK,db->k_norm,Q36_HEAD_DIM,Q36_KV_DIM,Q36_RMS_EPS);
        for(int i=0;i<B;i++){                /* per-slot: own position + KV cache */
            float*q=e->pfQ+(size_t)i*Q36_Q_DIM, *kx=e->pfK+(size_t)i*Q36_KV_DIM, *vx=e->pfV+(size_t)i*Q36_KV_DIM;
            k_rope_p<<<Q36_N_HEAD,Q36_ROT_DIM/2>>>(q,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_pos_b+i,Q36_ROPE_FREQ_BASE);
            k_rope_p<<<Q36_N_HEAD_KV,Q36_ROT_DIM/2>>>(kx,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_pos_b+i,Q36_ROPE_FREQ_BASE);
            k_kv_append<<<gr(2*(Q36_KV_DIM/32)*32,256),256>>>(kx,vx,
                KCS(e,L,i),KSS(e,L,i),VCS(e,L,i),VSS(e,L,i),e->d_pos_b+i,Q36_KV_DIM,e->kvq);
            float*aout=e->pfO+(size_t)i*Q36_Q_DIM;
            if(!e->kvq && !nomma && maxch>1){
                k_attn_dec_mma<2><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(q,
                    (const __half*)KCS(e,L,i),(const __half*)VCS(e,L,i),
                    e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos_b+i,scale);
                k_attn_merge2<<<dim3(Q36_N_HEAD,Q36_HEAD_DIM/32),256>>>(e->pacc,e->pms,aout,maxch*4,e->d_pos_b+i);
            } else {
                k_attn_partial<<<dim3(maxch,Q36_N_HEAD),256>>>(q,KCS(e,L,i),KSS(e,L,i),VCS(e,L,i),VSS(e,L,i),
                    e->pacc,e->pms,(maxch==1)?aout:NULL,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_pos_b+i,scale,e->kvq);
                if(maxch>1) k_attn_merge<<<Q36_N_HEAD,Q36_HEAD_DIM>>>(e->pacc,e->pms,aout,Q36_HEAD_DIM,maxch,e->d_pos_b+i,Q36_ATTN_CHUNK);
            }
        }
        k_sigmoid_mul<<<gr(B*Q36_Q_DIM,256),256>>>(e->pfO,e->pfGATE,B*Q36_Q_DIM);
        pf_gemm(e,0,db->o.d,e->pfO,e->pfOUT,B,D,Q36_Q_DIM);
    } else {
        pf_gemm(e,0,db->qkv.d,e->pfX,e->pfQKV,B,CONVD,D);
        pf_gemm(e,0,db->gate.d,e->pfX,e->pfZ,B,VALD,D);
        pf_gemm(e,2,db->alpha,e->pfX,e->pfA,B,NVH,D);
        pf_gemm(e,2,db->beta,e->pfX,e->pfB,B,NVH,D);
        k_dn_gates<<<dim3(1,B),64>>>(e->pfA,db->dt_bias,db->ssm_a,e->pfB,e->pfG,e->pfBt,NVH);
        for(int i=0;i<B;i++)                 /* conv: each slot's own history */
            k_dn_conv<<<gr(CONVD,256),256>>>(e->pfQKV+(size_t)i*CONVD,db->conv1d,
                e->convhist[L]+(size_t)i*CONVD*3,e->pfCB+(size_t)i*CONVD,CONVD,Q36_SSM_CONV_K);
        k_head_l2norm_b<<<dim3(NKH,B),64>>>(e->pfCB,HKD,CONVD,Q36_RMS_EPS);
        k_head_l2norm_b<<<dim3(NKH,B),64>>>(e->pfCB+KEYD,HKD,CONVD,Q36_RMS_EPS);
        k_dn_scan_b<<<dim3((HVD+7)/8,NVH,B),256>>>(e->pfCB,e->pfCB+KEYD,e->pfCB+2*KEYD,
            e->pfG,e->pfBt,e->Sstate[L],e->pfO,NVH,NKH,HKD,HVD,
            CONVD,NVH,(uint64_t)NVH*HKD*HVD,VALD);
        k_dn_gnorm_b<<<dim3(NVH,B),128>>>(e->pfO,db->ssm_norm,e->pfZ,HVD,VALD,Q36_RMS_EPS);
        pf_gemm(e,0,db->ssm_out.d,e->pfO,e->pfOUT,B,D,VALD);
    }
    k_add<<<gr(B*D,256),256>>>(e->pfH,e->pfOUT,B*D);
    pf_moe_block(e,db,B);
}

/* one batch-tiled decode step: all nslots tenants advance one token */
static void q36_decode_body_bt(q36_engine*e,int B){
    enum{D=Q36_D_MODEL};
    for(int i=0;i<B;i++)
        k_embed_dt<<<gr(D,256),256>>>(e->tok_embd,e->d_tok_b+i,e->pfH+(size_t)i*D,D);
    for(int L=0;L<Q36_N_LAYER;L++){
        bt_layer(e,&e->blk[L],L,B);
        k_add<<<gr(B*D,256),256>>>(e->pfH,e->pfMOE,B*D);
    }
    /* final norm -> per-tenant hidden in xb for the fused head */
    for(int i=0;i<B;i++)
        k_rmsnorm<<<1,1024>>>(e->pfH+(size_t)i*D,e->out_norm,e->xb+(size_t)i*D,D,Q36_RMS_EPS);
}

/* batched residual-add + rmsnorm: one block per token; identical reduction
 * structure to k_add_rmsnorm so per-token results are bit-identical. */
__global__ void k_add_rmsnorm_b(float*h,const float*a,const float*w,float*y,int n,float eps){Q36_GDS();
    int t=blockIdx.x;
    h+=(size_t)t*n; a+=(size_t)t*n; y+=(size_t)t*n;
    __shared__ float part[32]; __shared__ float ss;
    float loc=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){ float v=h[i]+a[i]; h[i]=v; loc+=v*v; }
    loc=warpsum(loc);
    int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    if(lane==0)part[wid]=loc; __syncthreads();
    if(threadIdx.x==0){float s=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)s+=part[i];ss=rsqrtf(s/n+eps);}
    __syncthreads();
    for(int i=threadIdx.x;i<n;i+=blockDim.x) y[i]=h[i]*ss*w[i];
}
/* rope with a per-token DEVICE position (grid.y = token); same per-element
 * math as k_rope_p. */
__global__ void k_rope_pb(float*x,uint64_t row_stride,int head_dim,int rot_dim,
                          const int*posp,float base){Q36_GDS();
    int t=blockIdx.y, h=blockIdx.x, i=threadIdx.x; if(i>=rot_dim/2)return;
    float*xh=x+(size_t)t*row_stride+(size_t)h*head_dim;
    float inv=__powf(base,-2.0f*i/rot_dim), ang=(float)posp[t]*inv;
    float c=__cosf(ang),s=__sinf(ang);
    float a=xh[i],b=xh[i+rot_dim/2];
    xh[i]=a*c-b*s; xh[i+rot_dim/2]=a*s+b*c;
}
/* kv append for the 2-token verify: grid.y = token, per-token device pos;
 * body identical to k_kv_append. */
__global__ void k_kv_append2(const float*k,uint64_t kstride,const float*v,uint64_t vstride,
                             int8_t*Kc,__half*Ks,uint8_t*Vc,uint8_t*Vs,
                             const int*posp,int kv_dim,int kvq){Q36_GDS();
    int t=blockIdx.y;
    k+=(size_t)t*kstride; v+=(size_t)t*vstride;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    int nb=kv_dim/32;
    if(warp>=2*nb)return;
    int pos=posp[t];
    if(!kvq){
        if(warp<nb) ((__half*)Kc)[(size_t)pos*kv_dim+warp*32+lane]=__float2half(k[warp*32+lane]);
        else { int b=warp-nb; ((__half*)Vc)[(size_t)pos*kv_dim+b*32+lane]=__float2half(v[b*32+lane]); }
        return;
    }
    if(warp<nb){
        float x=k[warp*32+lane], a=fabsf(x);
        #pragma unroll
        for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
        float d=a/127.f, id=(a>0.f)?127.f/a:0.f;
        int q=(int)lrintf(x*id);
        Kc[(size_t)pos*kv_dim+warp*32+lane]=(int8_t)(q>127?127:(q<-127?-127:q));
        if(lane==0)Ks[(size_t)pos*nb+warp]=__float2half(d);
    } else {
        int b=warp-nb;
        float x=v[b*32+lane], a=fabsf(x);
        #pragma unroll
        for(int o=16;o>0;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
        int e2=(a>0.f)?(int)ceilf(log2f(a/6.f)):-127;
        if(e2<-127)e2=-127;
        float inv=__int_as_float((127-e2)<<23);
        uint8_t code=e2m1_encode(x*inv);
        uint8_t hi=__shfl_down_sync(0xffffffff,(unsigned)code,1)&0xF;
        if((lane&1)==0) Vc[(size_t)pos*(kv_dim/2)+b*16+(lane>>1)]=(uint8_t)(code|(hi<<4));
        if(lane==0) Vs[(size_t)pos*nb+b]=(uint8_t)(127+e2);
    }
}

/* ================= MTP (nextn) self-speculative decode ==================
 * The nextn module predicts token t+2 from the main model's residual hidden
 * h_t and the sampled token x_{t+1}:
 *   z      = eh_proj @ [enorm(embed(x_{t+1})) | hnorm(h_t)]   (vLLM order)
 *   z      = TransformerBlock40(z)          own KV, true rope positions
 *   draft  = argmax(output_head(rmsnorm(z, shared_head_norm)))
 * The loop verifies [x_{t+1}, draft] in ONE B=2 forward (weights read once),
 * accepts when the model agrees, and rolls the SSM state back one step when
 * it doesn't -- output is bit-identical to plain greedy decode. */

extern "C" void q36_engine_reset(q36_engine*e);

/* One nextn forward.  Token comes from e->d_vtok[ktok] (device), the
 * previous hidden from `hid`; appends KV at slot d_mm[2*kmm+1] with rope
 * position d_mm[2*kmm]; with_head writes the draft argmax to
 * e->d_vtok[ktok+1] -- so K chained draft steps (hid = e->mtp_z of the
 * previous step, DeepSeek/vLLM style) fill d_vtok[1..K] in place.
 * Reuses the solo decode scratch (never runs concurrently with a step). */
static void q36_mtp_body(q36_engine*e,const float*hid,int ktok,int kmm,
                         int maxch,int with_head){
    enum{D=Q36_D_MODEL};
    dblock*db=&e->mtp_db;
    const int*mm=e->d_mm+2*kmm;
    static int swap=-1; if(swap<0) swap=getenv("Q36_MTP_SWAP")?1:0;
    static int msync=-1; if(msync<0) msync=getenv("Q36_MTP_SYNC")?1:0;
    #define MSY(tag) do{ if(msync){ cudaError_t se_=cudaDeviceSynchronize(); \
        fprintf(stderr,"mtp[%s]: %s\n",tag,se_?cudaGetErrorString(se_):"ok"); \
        if(se_)exit(1);} }while(0)
    k_embed_dt<<<gr(D,256),256>>>(e->tok_embd,e->d_vtok+ktok,e->mtp_emb,D);
    /* upstream (vLLM qwen3_next_mtp) fuses cat([enorm(embed),hnorm(hidden)]);
     * Q36_MTP_SWAP=1 flips the halves for A/B (wrong order is still lossless,
     * it just kills the accept rate) */
    k_rmsnorm<<<1,1024>>>(e->mtp_emb,e->mtp_enorm,e->mtp_cat+(swap?D:0),D,Q36_RMS_EPS);
    k_rmsnorm<<<1,1024>>>(hid,e->mtp_hnorm,e->mtp_cat+(swap?0:D),D,Q36_RMS_EPS);
    MSY("embed+norms");
    mv(&e->mtp_eh,e->mtp_cat,e->mtp_z);
    MSY("eh_proj");
    /* standard attention block on the fused stream, own KV cache */
    k_rmsnorm<<<1,1024>>>(e->mtp_z,db->attn_norm,e->x,D,Q36_RMS_EPS);
    k_matvec_multi3<<<gr(Q36_Q_DIM*2+2*Q36_KV_DIM,8),256>>>(
        db->q.d,Q36_Q_DIM*2,e->tmp, db->k.d,Q36_KV_DIM,e->kbuf,
        db->v.d,Q36_KV_DIM,e->vbuf, e->x,D);
    k_split_qgate<<<gr(Q36_Q_DIM,256),256>>>(e->tmp,e->q,e->gate,Q36_N_HEAD,Q36_HEAD_DIM);
    k_head_rmsnorm<<<Q36_N_HEAD,64>>>(e->q,db->q_norm,Q36_HEAD_DIM,Q36_RMS_EPS);
    k_head_rmsnorm<<<Q36_N_HEAD_KV,64>>>(e->kbuf,db->k_norm,Q36_HEAD_DIM,Q36_RMS_EPS);
    k_rope_p<<<Q36_N_HEAD,Q36_ROT_DIM/2>>>(e->q,Q36_HEAD_DIM,Q36_ROT_DIM,mm,Q36_ROPE_FREQ_BASE);
    k_rope_p<<<Q36_N_HEAD_KV,Q36_ROT_DIM/2>>>(e->kbuf,Q36_HEAD_DIM,Q36_ROT_DIM,mm,Q36_ROPE_FREQ_BASE);
    k_kv_append<<<gr(2*(Q36_KV_DIM/32)*32,256),256>>>(e->kbuf,e->vbuf,
        (int8_t*)e->mtpKc,e->mtpKs,e->mtpVc,e->mtpVs,mm+1,Q36_KV_DIM,0);
    float scale=1.f/sqrtf((float)Q36_HEAD_DIM);
    static int nomma=-1; if(nomma<0) nomma=getenv("Q36_NOMMA")?1:0;
    int mode=(maxch<=1||attn_chunk()!=Q36_ATTN_CHUNK2||nomma)?1:0;
    if(mode==0){
        if(maxch*Q36_N_HEAD_KV>=512)
            k_attn_dec_mma<3><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(e->q,
                (const __half*)e->mtpKc,(const __half*)e->mtpVc,
                e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,mm+1,scale);
        else
            k_attn_dec_mma<2><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(e->q,
                (const __half*)e->mtpKc,(const __half*)e->mtpVc,
                e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,mm+1,scale);
        k_attn_merge2<<<dim3(Q36_N_HEAD,Q36_HEAD_DIM/32),256>>>(
            e->pacc,e->pms,e->attn,maxch*4,mm+1);
    } else {
        k_attn_partial<<<dim3(maxch,Q36_N_HEAD),256>>>(e->q,(const int8_t*)e->mtpKc,e->mtpKs,
            e->mtpVc,e->mtpVs,e->pacc,e->pms,(maxch==1)?e->attn:NULL,
            Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,mm+1,scale,0);
        if(maxch>1)
            k_attn_merge<<<Q36_N_HEAD,Q36_HEAD_DIM>>>(e->pacc,e->pms,e->attn,Q36_HEAD_DIM,
                maxch,mm+1,Q36_ATTN_CHUNK);
    }
    k_sigmoid_mul<<<gr(Q36_Q_DIM,256),256>>>(e->attn,e->gate,Q36_Q_DIM);
    MSY("attn");
    mv(&db->o,e->attn,e->x);
    k_add_rmsnorm<<<1,1024>>>(e->mtp_z,e->x,db->post_norm,e->x,D,Q36_RMS_EPS);
    MSY("o+addnorm");
    moe_ffn(e,db,e->x,Q36_N_LAYER);
    MSY("moe");
    k_add_rmsnorm<<<1,1024>>>(e->mtp_z,e->moe,e->mtp_shn,e->x,D,Q36_RMS_EPS);
    if(with_head){
        mv(&e->output,e->x,e->logits);
        q36_cuda_argmax_async(e->logits,Q36_N_VOCAB,e->d_vtok+ktok+1);
    }
    MSY("head");
    #undef MSY
}

/* Verify forward: tokens d_vtok[0..B-1] at positions d_vpos[0..B-1] of the
 * SOLO sequence, sharing every weight read across the group (batched GEMVs)
 * while the sequence-dependent parts (attention, conv, scan) run per token
 * IN ORDER.  After each token j < B-1 the SSM state + conv history are
 * checkpointed to snapshot j, so a partial accept rolls back exactly to any
 * prefix.  Token i's attention dispatch mirrors the solo decode body at the
 * same depth (caps[i] = solo chunk bucket at seqlen pos+1+i), so every
 * token's compute is kernel-for-kernel identical to plain decode -- MTP
 * output is bit-identical greedy.  Argmaxes land in d_vout[0..B-1]. */
static void q36_verify_body(q36_engine*e,int B,const int*caps){
    enum{D=Q36_D_MODEL,NU=Q36_N_EXPERT_USED,FF=Q36_EXPERT_FF};
    float scale=1.f/sqrtf((float)Q36_HEAD_DIM);
    static int nomma=-1;
    if(nomma<0) nomma=getenv("Q36_NOMMA")?1:(getenv("Q36_ATTN2")?2:0);
    size_t SSZ=(size_t)NVH*HKD*HVD, CHZ=(size_t)CONVD*3;
    int nssm=(Q36_N_LAYER/Q36_ATTN_INTERVAL)*(Q36_ATTN_INTERVAL-1);
    for(int i=0;i<B;i++)
        k_embed_dt<<<gr(D,256),256>>>(e->tok_embd,e->d_vtok+i,e->vh+(size_t)i*D,D);
    for(int i=0;i<B;i++)
        k_rmsnorm<<<1,1024>>>(e->vh+(size_t)i*D,e->blk[0].attn_norm,e->vx+(size_t)i*D,D,Q36_RMS_EPS);
    for(int L=0;L<Q36_N_LAYER;L++){
        dblock*db=&e->blk[L];
        if(db->is_attn){
            k_matvec_multi3_b<<<gr(Q36_Q_DIM*2+2*Q36_KV_DIM,8),256>>>(
                db->q.d,Q36_Q_DIM*2,e->vtmp,8192, db->k.d,Q36_KV_DIM,e->vkbuf,8192,
                db->v.d,Q36_KV_DIM,e->vvbuf,4096, e->vx,D,B,D);
            /* per-token small ops batched across the group (bit-exact: the
             * _b variants share reduction structure with the solo kernels) */
            k_split_qgate_b<<<gr(B*Q36_Q_DIM,256),256>>>(e->vtmp,e->vq,e->vgate,B,Q36_N_HEAD,Q36_HEAD_DIM);
            k_head_rmsnorm_b<<<dim3(Q36_N_HEAD,B),64>>>(e->vq,db->q_norm,Q36_HEAD_DIM,4096,Q36_RMS_EPS);
            k_head_rmsnorm_b<<<dim3(Q36_N_HEAD_KV,B),64>>>(e->vkbuf,db->k_norm,Q36_HEAD_DIM,8192,Q36_RMS_EPS);
            k_rope_pb<<<dim3(Q36_N_HEAD,B),Q36_ROT_DIM/2>>>(e->vq,4096,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_vpos,Q36_ROPE_FREQ_BASE);
            k_rope_pb<<<dim3(Q36_N_HEAD_KV,B),Q36_ROT_DIM/2>>>(e->vkbuf,8192,Q36_HEAD_DIM,Q36_ROT_DIM,e->d_vpos,Q36_ROPE_FREQ_BASE);
            k_kv_append2<<<dim3(gr(2*(Q36_KV_DIM/32)*32,256),B),256>>>(e->vkbuf,8192,e->vvbuf,4096,
                e->Kc[L],e->Ks[L],e->Vc[L],e->Vs[L],e->d_vpos,Q36_KV_DIM,e->kvq);
            for(int i=0;i<B;i++){
                float*q=e->vq+(size_t)i*4096;
                float*aout=e->vattn+(size_t)i*4096;
                int maxch=caps[i];
                int mode;   /* mirror solo attn_layer dispatch at this depth */
                if(maxch<=1||attn_chunk()!=Q36_ATTN_CHUNK2) mode=1;
                else if(e->kvq||nomma==1) mode=1;
                else mode=(nomma==2)?2:0;
                if(mode==0){
                    if(maxch*Q36_N_HEAD_KV>=512)
                        k_attn_dec_mma<3><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(q,
                            (const __half*)e->Kc[L],(const __half*)e->Vc[L],
                            e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_vpos+i,scale);
                    else
                        k_attn_dec_mma<2><<<dim3(maxch,Q36_N_HEAD_KV),128>>>(q,
                            (const __half*)e->Kc[L],(const __half*)e->Vc[L],
                            e->pacc,e->pms,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_vpos+i,scale);
                    k_attn_merge2<<<dim3(Q36_N_HEAD,Q36_HEAD_DIM/32),256>>>(
                        e->pacc,e->pms,aout,maxch*4,e->d_vpos+i);
                } else if(mode==2){
                    k_attn_partial2<<<dim3(maxch,Q36_N_HEAD_KV),256>>>(q,e->Kc[L],e->Ks[L],e->Vc[L],e->Vs[L],
                        e->pacc,e->pms,NULL,Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_vpos+i,scale,e->kvq);
                    k_attn_merge<<<Q36_N_HEAD,Q36_HEAD_DIM>>>(e->pacc,e->pms,aout,Q36_HEAD_DIM,
                        maxch,e->d_vpos+i,0);
                } else {
                    k_attn_partial<<<dim3(maxch,Q36_N_HEAD),256>>>(q,e->Kc[L],e->Ks[L],e->Vc[L],e->Vs[L],
                        e->pacc,e->pms,(maxch==1)?aout:NULL,
                        Q36_N_HEAD,Q36_N_HEAD_KV,Q36_HEAD_DIM,e->d_vpos+i,scale,e->kvq);
                    if(maxch>1)
                        k_attn_merge<<<Q36_N_HEAD,Q36_HEAD_DIM>>>(e->pacc,e->pms,aout,Q36_HEAD_DIM,
                            maxch,e->d_vpos+i,Q36_ATTN_CHUNK);
                }
            }
            k_sigmoid_mul<<<gr(B*Q36_Q_DIM,256),256>>>(e->vattn,e->vgate,B*Q36_Q_DIM);
            k_matvec_gb<<<gr(D,8),256>>>(1,db->o.d,e->vattn,e->vx,D,Q36_Q_DIM,B,4096,D);
        } else {
            int si=e->ssm_sidx[L];
            k_matvec_multi3_b<<<gr(CONVD+VALD,8),256>>>(
                db->qkv.d,CONVD,e->vtmp,8192, db->gate.d,VALD,e->vvbuf,4096,
                NULL,0,NULL,0, e->vx,D,B,D);
            k_matvec_f32_b<<<gr(2*NVH,8),256>>>(db->alpha,e->vx,e->vg512,2*NVH,D,B,D,64);
            for(int i=0;i<B;i++)
                k_dn_gates<<<gr(NVH,64),64>>>(e->vg512+(size_t)i*64,db->dt_bias,db->ssm_a,
                    e->vg512+(size_t)i*64+NVH,e->vq+(size_t)i*4096,e->vq+(size_t)i*4096+NVH,NVH);
            /* conv + scan are serial across the group; checkpoint j after
             * token j (j < B-1) for prefix-exact rollback */
            for(int i=0;i<B;i++){
                k_dn_conv<<<gr(CONVD,256),256>>>(e->vtmp+(size_t)i*8192,db->conv1d,
                    e->convhist[L],e->vkbuf+(size_t)i*8192,CONVD,Q36_SSM_CONV_K);
                if(i<B-1)
                    CK(cudaMemcpyAsync(e->convsnap+((size_t)i*nssm+si)*CHZ,e->convhist[L],
                        CHZ*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            }
            k_head_l2norm_b<<<dim3(NKH,B),64>>>(e->vkbuf,HKD,8192,Q36_RMS_EPS);
            k_head_l2norm_b<<<dim3(NKH,B),64>>>(e->vkbuf+KEYD,HKD,8192,Q36_RMS_EPS);
            for(int i=0;i<B;i++){
                k_dn_scan<<<dim3((HVD+7)/8,NVH),256>>>(e->vkbuf+(size_t)i*8192,
                    e->vkbuf+(size_t)i*8192+KEYD,e->vkbuf+(size_t)i*8192+2*KEYD,
                    e->vq+(size_t)i*4096,e->vq+(size_t)i*4096+NVH,
                    e->Sstate[L],e->vattn+(size_t)i*4096,NVH,NKH,HKD,HVD);
                if(i<B-1)
                    CK(cudaMemcpyAsync(e->Ssnap+((size_t)i*nssm+si)*SSZ,e->Sstate[L],
                        SSZ*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            }
            k_dn_gnorm_b<<<dim3(NVH,B),128>>>(e->vattn,db->ssm_norm,e->vvbuf,HVD,4096,Q36_RMS_EPS);
            k_matvec_gb<<<gr(D,8),256>>>(1,db->ssm_out.d,e->vattn,e->vx,D,VALD,B,4096,D);
        }
        k_add_rmsnorm_b<<<B,1024>>>(e->vh,e->vx,db->post_norm,e->vx,D,Q36_RMS_EPS);
        k_matvec_f32_b<<<gr(Q36_N_EXPERT+1,8),256>>>(db->router,e->vx,e->vrl,
            Q36_N_EXPERT+1,D,B,D,Q36_N_EXPERT+1);
        for(int i=0;i<B;i++)
            k_router_topk<<<1,32>>>(e->vrl+(size_t)i*(Q36_N_EXPERT+1),Q36_N_EXPERT,NU,
                e->vtopk+(size_t)i*NU,e->vtopw+(size_t)i*NU,e->expert_scale,1);
        dim3 gg(gr(FF,8),B*NU,2);
        k_expert_gemv2_bt<<<gg,256>>>(db->gate_exps.elayout,db->gate_exps.d,db->gate_exps.expert_stride,
            db->up_exps.elayout,db->up_exps.d,db->up_exps.expert_stride,
            e->se0,e->se1,e->vtopk,e->vx,e->vg512,e->vu512,FF,D,NU,D);
        k_silu_mul_slots<<<gr(B*NU*FF,256),256>>>(e->vg512,e->vu512,B*NU*FF);
        dim3 gd(gr(D,8),B*NU);
        k_expert_gemv_bt<<<gd,256>>>(db->down_exps.elayout,db->down_exps.d,db->down_exps.expert_stride,
            e->se0,e->se1,e->vtopk,e->vg512,FF,e->vd2048,D,FF);
        k_matvec_multi3_b<<<gr(2*FF,8),256>>>(db->sh_gate.d,FF,e->vsh_a,FF,
            db->sh_up.d,FF,e->vsh_b,FF, NULL,0,NULL,0, e->vx,D,B,D);
        k_silu_mul<<<gr(B*FF,256),256>>>(e->vsh_a,e->vsh_b,B*FF);
        k_matvec_gb<<<gr(D,8),256>>>(1,db->sh_down.d,e->vsh_a,e->vshexp,D,FF,B,FF,D);
        for(int i=0;i<B;i++)
            k_moe_combine<<<gr(D,256),256>>>(e->vmoe+(size_t)i*D,e->vd2048+(size_t)i*NU*D,
                e->vtopw+(size_t)i*NU,e->vshexp+(size_t)i*D,
                e->vrl+(size_t)i*(Q36_N_EXPERT+1)+Q36_N_EXPERT,D,NU);
        const float*nw=(L+1<Q36_N_LAYER)?e->blk[L+1].attn_norm:e->out_norm;
        k_add_rmsnorm_b<<<B,1024>>>(e->vh,e->vmoe,nw,e->vx,D,Q36_RMS_EPS);
    }
    /* fused head: one 0.5GB weight read serves all B positions.  Argmax
     * tie-breaking must match plain decode EXACTLY (exact logit ties between
     * duplicate vocab rows otherwise pick different ids and the greedy
     * stream silently diverges) -- the solo greedy head uses the SAME fused
     * kernel at B=1; its tie order is a pure function of the partial values,
     * independent of B. */
    { const dw*W=&e->output;
      int Rmix=(W->elayout==5)?(int)W->row_bytes:0;
      k_head_argmax_p1<<<gr(Q36_N_VOCAB,8),256>>>(W->elayout,W->d,e->vx,Rmix,
          W->M,W->K,B,D,e->v_pv,e->v_pi,e->v_nblk);
      k_head_argmax_p2<<<B,256>>>(e->v_pv,e->v_pi,e->v_nblk,B,e->d_vout);
    }
}

/* partial accept of m tokens: restore SSM state + conv history to the
 * checkpoint taken after verify token m-1 */
static void q36_mtp_rollback(q36_engine*e,int m){
    size_t SSZ=(size_t)NVH*HKD*HVD, CHZ=(size_t)CONVD*3;
    int nssm=(Q36_N_LAYER/Q36_ATTN_INTERVAL)*(Q36_ATTN_INTERVAL-1);
    size_t js=(size_t)(m-1)*nssm;
    for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn){
        int si=e->ssm_sidx[L];
        CK(cudaMemcpyAsync(e->Sstate[L],e->Ssnap+(js+si)*SSZ,SSZ*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        CK(cudaMemcpyAsync(e->convhist[L],e->convsnap+(js+si)*CHZ,CHZ*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    }
}

extern "C" void q36_engine_set_mtp_k(q36_engine*e,int k){
    if(!e->has_mtp) return;
    if(k<1)k=1; if(k>Q36_MTP_MAXB-1)k=Q36_MTP_MAXB-1;
    if(k!=e->mtp_k){
        e->mtp_k=k;
        if(e->gx_v){cudaGraphExecDestroy(e->gx_v);e->gx_v=NULL;}
        if(e->gx_mn){cudaGraphExecDestroy(e->gx_mn);e->gx_mn=NULL;}
    }
}

/* Speculative decode step at draft depth K: processes `token` at `pos`,
 * drafts K tokens with the chained nextn module, verifies all K+1 in one
 * B=K+1 forward, and returns the accepted prefix in out[] (1..K+1 tokens).
 * out[j] = y_j; the caller advances pos by the return count and feeds
 * out[count-1] next.  Falls back to a plain step (and stays there until the
 * next prefill) when MTP is unavailable, unprimed, sampled, or at the
 * context edge. */
extern "C" int q36_engine_step_mtp(q36_engine*e,int token,int pos,int out[Q36_MTP_MAXB]){
    enum{D=Q36_D_MODEL};
    int K=e->mtp_k, B=K+1;
    if(!e->has_mtp||!e->mtp_primed||e->s_temp>0.f||pos+B>=e->ctx){
        e->mtp_primed=0;
        out[0]=q36_engine_step(e,token,pos);
        return 1;
    }
    int csz=attn_chunk(), P=pos;
    int maxcap=(e->ctx+csz-1)/csz;
    if(e->mtp_base<0) e->mtp_base=P-1;   /* first nextn entry this conversation */
    int sP=P-1-e->mtp_base;              /* nextn kv slot of entry index P-1 */
    /* per-token verify buckets = solo's bucket at the same depth (bit-parity) */
    int caps[Q36_MTP_MAXB];
    for(int i=0;i<B;i++){
        int nd=(P+i+1+csz-1)/csz;
        caps[i]=(nd<=1)?1:((nd+7)&~7); if(caps[i]>maxcap)caps[i]=maxcap;
    }
    /* nextn attention bucket: deepest slot this cycle is sP+K (the fill) */
    int ndm=(sP+K+1+csz-1)/csz;
    int capm=(ndm<=1)?1:((ndm+7)&~7); if(capm>maxcap)capm=maxcap;
    /* one H2D: token + K draft (pos,slot) pairs + fill pair + verify pos */
    int hb[20]={0};
    hb[0]=token;
    for(int k=0;k<K;k++){ hb[8+2*k]=P+k; hb[9+2*k]=sP+k; }
    hb[14]=P+K; hb[15]=sP+K;             /* fill pair (used only on full accept) */
    for(int i=0;i<B;i++) hb[16+i]=P+i;
    CK(cudaMemcpy(e->d_vtok,hb,sizeof hb,cudaMemcpyHostToDevice));
    int nograph=(getenv("Q36_DEBUG")||getenv("Q36_NOGRAPH"))?1:0;
    if(nograph){
        for(int k=0;k<K;k++)
            q36_mtp_body(e,k?e->mtp_z:e->mtp_h,k,k,capm,1);
        q36_verify_body(e,B,caps);
    } else {
        int regrow=(!e->gx_v||capm>e->gmch);
        for(int i=0;i<B;i++) if(caps[i]>e->gvc[i]) regrow=1;
        if(regrow){
            if(e->gx_v){cudaGraphExecDestroy(e->gx_v);e->gx_v=NULL;}
            if(e->gx_mn){cudaGraphExecDestroy(e->gx_mn);e->gx_mn=NULL;}
            e->gmch=capm; for(int i=0;i<B;i++) e->gvc[i]=caps[i]; cudaGraph_t g;
            /* whole cycle (K chained drafts + verify) in ONE graph */
            CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
            for(int k=0;k<K;k++)
                q36_mtp_body(e,k?e->mtp_z:e->mtp_h,k,k,capm,1);
            q36_verify_body(e,B,caps);
            CK(cudaStreamEndCapture(cudaStreamPerThread,&g));
            if(!getenv("Q36_NOPDL")) pdl_edges(g);
            CK(cudaGraphInstantiate(&e->gx_v,g,0)); cudaGraphDestroy(g);
            /* fill-only graph: true-h entry, no head (mm pair 3) */
            CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
            q36_mtp_body(e,e->mtp_h,0,3,capm,0);
            CK(cudaStreamEndCapture(cudaStreamPerThread,&g));
            if(!getenv("Q36_NOPDL")) pdl_edges(g);
            CK(cudaGraphInstantiate(&e->gx_mn,g,0)); cudaGraphDestroy(g);
        }
        static int prof=-1; if(prof<0) prof=getenv("Q36_MTP_PROF")?1:0;
        if(prof){
            static cudaEvent_t t0,t1; static double sv=0,sc=0; static long n=0;
            static struct timespec w0; struct timespec w1;
            if(!t0){CK(cudaEventCreate(&t0));CK(cudaEventCreate(&t1));
                    clock_gettime(CLOCK_MONOTONIC,&w0);}
            CK(cudaEventRecord(t0,cudaStreamPerThread));
            CK(cudaGraphLaunch(e->gx_v,cudaStreamPerThread));
            CK(cudaEventRecord(t1,cudaStreamPerThread));
            CK(cudaEventSynchronize(t1));
            float a; cudaEventElapsedTime(&a,t0,t1);
            clock_gettime(CLOCK_MONOTONIC,&w1);
            double wall=(w1.tv_sec-w0.tv_sec)*1e3+(w1.tv_nsec-w0.tv_nsec)*1e-6;
            w0=w1;
            sv+=a; sc+=wall; n++;
            if(n%50==0) fprintf(stderr,"[prof n=%ld: cycle-graph %.2fms cycle-wall %.2fms]\n",n,sv/n,sc/n);
        } else
            CK(cudaGraphLaunch(e->gx_v,cudaStreamPerThread));
    }
    int rb[8]; CK(cudaMemcpy(rb,e->d_vtok,sizeof rb,cudaMemcpyDeviceToHost));
    /* rb[1..K] = drafts, rb[4..4+K] = y_0..y_K; accept while y_i == d_{i+1} */
    int m=1;
    while(m<B && rb[4+m-1]==rb[m]) m++;
    e->mtp_cycles++; e->mtp_accepts+=m-1;
    if(getenv("Q36_MTP_TRACE"))
        fprintf(stderr,"TRACE %d %d -> m=%d y={%d %d %d %d} d={%d %d %d}\n",
                P,token,m,rb[4],rb[5],rb[6],rb[7],rb[1],rb[2],rb[3]);
    for(int j=0;j<m;j++) out[j]=rb[4+j];
    if(m==B){
        /* full accept: the drafting chain covered nextn entries P-1..P+K-2;
         * fill entry P+K-1 with the TRUE hidden so the cache stays dense,
         * then hand h_{P+K} to the next cycle */
        CK(cudaMemcpyAsync(e->mtp_h,e->vh+(size_t)(B-2)*D,(size_t)D*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        int fb[1]={rb[4+B-2]};   /* token_{P+K} = y_{K-1} */
        CK(cudaMemcpy(e->d_vtok,fb,sizeof fb,cudaMemcpyHostToDevice));
        if(nograph) q36_mtp_body(e,e->mtp_h,0,3,capm,0);
        else CK(cudaGraphLaunch(e->gx_mn,cudaStreamPerThread));
        CK(cudaMemcpyAsync(e->mtp_h,e->vh+(size_t)(B-1)*D,(size_t)D*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    } else {
        q36_mtp_rollback(e,m);
        CK(cudaMemcpyAsync(e->mtp_h,e->vh+(size_t)(m-1)*D,(size_t)D*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    }
    return m;
}
extern "C" int q36_engine_has_mtp(q36_engine*e){ return e->has_mtp; }
extern "C" void q36_engine_mtp_stats(q36_engine*e,long*cycles,long*accepts){
    if(cycles)*cycles=e->mtp_cycles; if(accepts)*accepts=e->mtp_accepts;
}

/* Isolation gate for the checkpoint/rollback (build this FIRST, per plan):
 * a B-token verify pass followed by rollback-to-prefix-m must reproduce m
 * solo steps' SSM state + conv history BIT-FOR-BIT, and the verify argmaxes
 * must equal the solo steps'.  Exercises every checkpoint slot.  0 = pass. */
extern "C" int q36_engine_mtp_selftest(q36_engine*e){
    enum{D=Q36_D_MODEL};
    if(!e->has_mtp){ fprintf(stderr,"mtp-selftest: no MTP module\n"); return -1; }
    size_t SSZ=(size_t)NVH*HKD*HVD, CHZ=(size_t)CONVD*3;
    int nssm=0; for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn) nssm++;
    size_t per=(SSZ+CHZ)*4;
    uint8_t*h1=(uint8_t*)malloc((size_t)nssm*per), *h2=(uint8_t*)malloc((size_t)nssm*per);
    const int toks[4]={9707,17,264,1988};
    int bad=0;
    for(int m=1;m<Q36_MTP_MAXB;m++){       /* rollback target = after token m-1 */
        /* reference: m solo steps from fresh state */
        q36_engine_reset(e);
        int ref=-1;
        for(int t=0;t<m;t++) ref=q36_engine_step(e,toks[t],t);
        CK(cudaDeviceSynchronize());
        for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn){
            uint8_t*d=h1+(size_t)e->ssm_sidx[L]*per;
            CK(cudaMemcpy(d,e->Sstate[L],SSZ*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d+SSZ*4,e->convhist[L],CHZ*4,cudaMemcpyDeviceToHost));
        }
        /* B=4 verify from fresh state, then roll back to prefix m */
        q36_engine_reset(e);
        int hb[20]={0};
        for(int t=0;t<4;t++){ hb[t]=toks[t]; hb[16+t]=t; }
        CK(cudaMemcpy(e->d_vtok,hb,sizeof hb,cudaMemcpyHostToDevice));
        int caps[4]={1,1,1,1};
        q36_verify_body(e,4,caps);
        q36_mtp_rollback(e,m);
        CK(cudaDeviceSynchronize());
        for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn){
            uint8_t*d=h2+(size_t)e->ssm_sidx[L]*per;
            CK(cudaMemcpy(d,e->Sstate[L],SSZ*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d+SSZ*4,e->convhist[L],CHZ*4,cudaMemcpyDeviceToHost));
        }
        int y; CK(cudaMemcpy(&y,e->d_vout+m-1,4,cudaMemcpyDeviceToHost));
        int mb=0;
        for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn){
            size_t off=(size_t)e->ssm_sidx[L]*per;
            if(memcmp(h1+off,h2+off,per)){ mb++; fprintf(stderr,"mtp-selftest: m=%d L%d state MISMATCH\n",m,L); }
        }
        if(y!=ref){ mb++; fprintf(stderr,"mtp-selftest: m=%d verify argmax %d != solo %d\n",m,y,ref); }
        fprintf(stderr,"mtp-selftest: prefix m=%d %s\n",m,mb?"FAIL":"PASS (state bit-exact, argmax match)");
        bad+=mb;
    }
    free(h1); free(h2);
    q36_engine_reset(e);
    return bad?1:0;
}

extern "C" int q36_engine_prefill_from(q36_engine*e,const int*toks,int n,int pos0){
    enum{T0=Q36_PF_CHUNK,D=Q36_D_MODEL,NU=Q36_N_EXPERT_USED,FF=Q36_EXPERT_FF};
    /* KV caches are sized exactly ctx positions; writing past them is
     * silent device-memory corruption (measured: a 739-token prompt at
     * --ctx 256 scribbled ~500KB over neighboring buffers without
     * faulting).  Refuse instead -- fitting the prompt is the caller's
     * policy (CLI auto-truncates, server auto-purges). */
    if(n<1||pos0<0||pos0+n>e->ctx){
        fprintf(stderr,"q36: prefill of %d tokens at pos %d exceeds ctx %d -- refused\n",
                n,pos0,e->ctx);
        return -1;
    }
    int last=-1;
    for(int off=0;off<n;off+=T0){
        int T=(n-off<T0)?(n-off):T0;
        CK(cudaMemcpy(e->pf_toks,toks+off,T*sizeof(int),cudaMemcpyHostToDevice));
        k_embed_b<<<gr(T*D,256),256>>>(e->tok_embd,e->pf_toks,e->pfH,T,D);
        for(int L=0;L<Q36_N_LAYER;L++){
            pf_layer_moe(e,L,T,pos0+off);
            k_add<<<gr(T*D,256),256>>>(e->pfH,e->pfMOE,T*D);
        }
        if(off+T>=n){
            if(e->has_mtp){   /* prime speculative decode: h at the last
                                 prompt position (residual, pre-out_norm) */
                CK(cudaMemcpyAsync(e->mtp_h,e->pfH+(size_t)(T-1)*D,(size_t)D*4,
                                   cudaMemcpyDeviceToDevice,cudaStreamPerThread));
                if(pos0==0) e->mtp_base=-1;
                e->mtp_primed=1;
            }
            k_rmsnorm<<<1,256>>>(e->pfH+(size_t)(T-1)*D,e->out_norm,e->x,D,Q36_RMS_EPS);
            mv(&e->output,e->x,e->logits);
            last=q36_cuda_argmax(e->logits,Q36_N_VOCAB);
        }
    }
    return last;
}
extern "C" int q36_engine_prefill(q36_engine*e,const int*toks,int n){
    return q36_engine_prefill_from(e,toks,n,0);
}

/* Multi-tenant de-risk probe: time one full 40-layer forward step at batch
 * width B through the (already-batched) prefill machinery.  This measures
 * the weight-amortization curve -- dense GEMMs, MoE dispatch/expert GEMMs,
 * norms -- which is the core uncertainty for continuous batching.  It is
 * NOT a correct batched decode (all B rows belong to one sequence; real
 * batching adds per-seq KV attention, per-seq SSM state traffic, and the
 * head, which the caller composes analytically).  Returns seconds/step. */
extern "C" double q36_engine_fake_batch(q36_engine*e,int B,int c0,int reps){
    enum{D=Q36_D_MODEL};
    if(B<1||B>Q36_PF_CHUNK) return -1.0;
    struct timespec a,b;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&a);
    for(int r=0;r<reps;r++){
        k_embed_b<<<gr(B*D,256),256>>>(e->tok_embd,e->pf_toks,e->pfH,B,D);
        for(int L=0;L<Q36_N_LAYER;L++){
            pf_layer_moe(e,L,B,c0);
            k_add<<<gr(B*D,256),256>>>(e->pfH,e->pfMOE,B*D);
        }
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&b);
    return ((b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9)/reps;
}

/* streaming logsumexp over one logits row + target pick -> NLL (one block) */
__global__ void k_nll_one(const float*logits,int n,int target,float*out){Q36_GDS();
    int tid=threadIdx.x;
    float m=-1e30f,s=0.f;
    for(int i=tid;i<n;i+=blockDim.x){
        float v=logits[i], mn=fmaxf(m,v);
        s=s*__expf(m-mn)+__expf(v-mn); m=mn;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1){
        float mo=__shfl_xor_sync(0xffffffff,m,o), so=__shfl_xor_sync(0xffffffff,s,o);
        float mn=fmaxf(m,mo);
        s=s*__expf(m-mn)+so*__expf(mo-mn); m=mn;
    }
    __shared__ float sm[32],ss[32];
    int lane=tid&31,w=tid>>5;
    if(lane==0){sm[w]=m;ss[w]=s;}
    __syncthreads();
    if(tid==0){
        float gm=sm[0],gs=ss[0]; int nw=(int)blockDim.x>>5;
        for(int i=1;i<nw;i++){
            float mn=fmaxf(gm,sm[i]);
            gs=gs*__expf(gm-mn)+ss[i]*__expf(sm[i]-mn); gm=mn;
        }
        out[0]=gm+__logf(gs)-logits[target];
    }
}

/* Perplexity scoring: run ONE window (2 <= n <= Q36_PF_CHUNK) from a fresh
 * state; return the summed NLL of targets toks[first..n-1] (logits row t
 * scores toks[t+1]).  Windowing matches llama-perplexity (independent
 * windows, second half scored), so the same GGUF through llama.cpp is a
 * direct correctness reference for our kernels.  The output head is our
 * FP8-e4m3 requant -- any PPL delta includes that choice by design. */
extern "C" void q36_engine_reset(q36_engine*e);
extern "C" double q36_engine_nll(q36_engine*e,const int*toks,int n,int first,int*count){
    enum{T0=Q36_PF_CHUNK,D=Q36_D_MODEL};
    if(n<2||n>T0||first<1||first>=n){ if(count)*count=0; return 0.0; }
    q36_engine_reset(e);
    CK(cudaMemcpy(e->pf_toks,toks,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));
    k_embed_b<<<gr(n*D,256),256>>>(e->tok_embd,e->pf_toks,e->pfH,n,D);
    for(int L=0;L<Q36_N_LAYER;L++){
        pf_layer_moe(e,L,n,0);
        k_add<<<gr(n*D,256),256>>>(e->pfH,e->pfMOE,n*D);
    }
    k_rmsnorm_b<<<n,256>>>(e->pfH,e->out_norm,e->pfX,D,Q36_RMS_EPS);
    if(!e->d_nll) CK(cudaMalloc(&e->d_nll,T0*sizeof(float)));
    for(int t=first-1;t<=n-2;t++){
        mv(&e->output,e->pfX+(size_t)t*D,e->logits);
        k_nll_one<<<1,1024>>>(e->logits,Q36_N_VOCAB,toks[t+1],e->d_nll+t);
    }
    int cnt=n-first;
    float*h=(float*)malloc((size_t)cnt*sizeof(float));
    CK(cudaMemcpy(h,e->d_nll+first-1,(size_t)cnt*sizeof(float),cudaMemcpyDeviceToHost));
    double sum=0; for(int i=0;i<cnt;i++) sum+=h[i];
    free(h);
    if(count)*count=cnt;
    return sum;
}

/* ============================ multi-GPU driver ===========================
 * Expert-parallel prefill over N GPUs (dynamic).  Dense/attention/SSM + KV
 * replicated per device (identical compute -> KV needs no sharding); routed
 * experts sharded; per-layer partial-MoE all-reduce (gather+sum+bcast over
 * PCIe P2P, cross-device event ordering).  Device 0 holds the FULL expert
 * set so decode runs on it unchanged. */
extern "C" void q36_engine_reset(q36_engine*e);
typedef struct q36_multi {
    int n;
    q36_engine*eng[16];
    cudaEvent_t evA[16], evR;
    float*gath[16];
} q36_multi;

extern "C" q36_multi* q36_multi_create(q36_model*m,int ctx,int ngpu){
    int nd=0; cudaGetDeviceCount(&nd);
    if(ngpu>nd)ngpu=nd; if(ngpu>16)ngpu=16;
    q36_multi*mg=(q36_multi*)calloc(1,sizeof(q36_multi));
    mg->n=ngpu;
    g_no_autoselect=1;
    for(int d=0;d<ngpu;d++){
        CK(cudaSetDevice(d));
        for(int p=0;p<ngpu;p++) if(p!=d) cudaDeviceEnablePeerAccess(p,0); /* ok if already on */
        cudaGetLastError();                            /* swallow "already enabled" */
        if(d==0){ g_shard_e0=0; g_shard_e1=Q36_N_EXPERT; }      /* full: decode host */
        else { g_shard_e0=d*Q36_N_EXPERT/ngpu; g_shard_e1=(d+1)*Q36_N_EXPERT/ngpu; }
        fprintf(stderr,"gpu %d: experts [%d,%d)%s\n",d,g_shard_e0,g_shard_e1,d==0?" (full resident)":"");
        mg->eng[d]=q36_engine_create(m,ctx);
        mg->eng[d]->inc_shared=(d==0);
        CK(cudaEventCreateWithFlags(&mg->evA[d],cudaEventDisableTiming));
    }
    CK(cudaSetDevice(0));
    CK(cudaEventCreateWithFlags(&mg->evR,cudaEventDisableTiming));
    for(int d=1;d<ngpu;d++)
        CK(cudaMalloc(&mg->gath[d],(size_t)Q36_PF_CHUNK*Q36_D_MODEL*sizeof(float)));
    g_shard_e0=0; g_shard_e1=Q36_N_EXPERT;
    return mg;
}
extern "C" q36_engine* q36_multi_engine0(q36_multi*mg){ return mg->eng[0]; }
extern "C" void q36_multi_reset(q36_multi*mg){ for(int d=0;d<mg->n;d++){ cudaSetDevice(d); q36_engine_reset(mg->eng[d]); } cudaSetDevice(0); }

extern "C" int q36_multi_prefill(q36_multi*mg,const int*toks,int n){
    enum{T0=Q36_PF_CHUNK,D=Q36_D_MODEL};
    int N=mg->n, last=-1;
    if(n<1||n>mg->eng[0]->ctx){   /* same OOB guard as q36_engine_prefill_from */
        fprintf(stderr,"q36: prefill of %d tokens exceeds ctx %d -- refused\n",n,mg->eng[0]->ctx);
        return -1;
    }
    /* prefill shard for device 0 (restored for decode at the end) */
    mg->eng[0]->se0=0; mg->eng[0]->se1=Q36_N_EXPERT/N;
    for(int c0=0;c0<n;c0+=T0){
        int T=(n-c0<T0)?(n-c0):T0;
        for(int d=0;d<N;d++){ CK(cudaSetDevice(d)); q36_engine*e=mg->eng[d];
            CK(cudaMemcpyAsync(e->pf_toks,toks+c0,(size_t)T*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
            k_embed_b<<<gr(T*D,256),256>>>(e->tok_embd,e->pf_toks,e->pfH,T,D);
        }
        for(int L=0;L<Q36_N_LAYER;L++){
            for(int d=0;d<N;d++){ CK(cudaSetDevice(d));
                pf_layer_moe(mg->eng[d],L,T,c0);
                CK(cudaEventRecord(mg->evA[d],cudaStreamPerThread));
            }
            CK(cudaSetDevice(0));
            for(int d=1;d<N;d++){
                CK(cudaStreamWaitEvent(cudaStreamPerThread,mg->evA[d],0));
                CK(cudaMemcpyPeerAsync(mg->gath[d],0,mg->eng[d]->pfMOE,d,(size_t)T*D*4,cudaStreamPerThread));
            }
            for(int d=1;d<N;d++)
                k_add<<<gr(T*D,256),256>>>(mg->eng[0]->pfMOE,mg->gath[d],T*D);
            for(int d=1;d<N;d++)
                CK(cudaMemcpyPeerAsync(mg->eng[d]->pfMOE,d,mg->eng[0]->pfMOE,0,(size_t)T*D*4,cudaStreamPerThread));
            CK(cudaEventRecord(mg->evR,cudaStreamPerThread));
            for(int d=0;d<N;d++){ CK(cudaSetDevice(d));
                if(d>0) CK(cudaStreamWaitEvent(cudaStreamPerThread,mg->evR,0));
                k_add<<<gr(T*D,256),256>>>(mg->eng[d]->pfH,mg->eng[d]->pfMOE,T*D);
            }
        }
        if(c0+T>=n){
            CK(cudaSetDevice(0)); q36_engine*e=mg->eng[0];
            k_rmsnorm<<<1,256>>>(e->pfH+(size_t)(T-1)*D,e->out_norm,e->x,D,Q36_RMS_EPS);
            mv(&e->output,e->x,e->logits);
            last=q36_cuda_argmax(e->logits,Q36_N_VOCAB);
        }
    }
    mg->eng[0]->se0=0; mg->eng[0]->se1=Q36_N_EXPERT;
    CK(cudaSetDevice(0));
    return last;
}

/* Copy current logits to host (for validation diffs against a reference). */
extern "C" void q36_engine_copy_logits(q36_engine*e,float*out,int n){
    CK(cudaMemcpy(out,e->logits,n*sizeof(float),cudaMemcpyDeviceToHost));
}

/* Reset per-conversation state: zero SSM recurrent state + conv history so a
 * new prompt starts clean.  KV cache is overwritten from pos 0, no clear needed. */
extern "C" void q36_engine_reset(q36_engine*e){
    for(int L=0;L<Q36_N_LAYER;L++){
        if(!e->blk[L].is_attn){
            CK(cudaMemset(e->Sstate[L],0,(size_t)e->nslots*NVH*HKD*HVD*sizeof(float)));
            CK(cudaMemset(e->convhist[L],0,(size_t)e->nslots*CONVD*3*sizeof(float)));
        }
    }
    e->mtp_base=-1; e->mtp_primed=0;  /* nextn KV restarts with the conversation */
}
extern "C" void q36_engine_slot_reset(q36_engine*e,int slot){
    for(int L=0;L<Q36_N_LAYER;L++){
        if(!e->blk[L].is_attn){
            CK(cudaMemset(e->Sstate[L]+(size_t)slot*NVH*HKD*HVD,0,(size_t)NVH*HKD*HVD*sizeof(float)));
            CK(cudaMemset(e->convhist[L]+(size_t)slot*CONVD*3,0,(size_t)CONVD*3*sizeof(float)));
        }
    }
}

/* ========== hybrid state checkpoint export/import ==========================
 * (the core state save/load primitive of the checkpoint cache)
 *
 * The engine's reusable per-conversation state at a token boundary N is
 *   - attention KV: positions [0,N) of the 10 attention layers' caches.
 *     Storage is position-major, so a position range is one contiguous slab
 *     per component and can be exported/imported piecewise; and
 *   - SSM/DeltaNet: all 30 recurrent layers' state + conv history AS OF
 *     token N -- one fixed-size blob (~63MB) that is a function of the
 *     entire ordered prefix.  It cannot be sliced per token; a checkpoint
 *     restores it wholesale.
 * Blob bytes are opaque to callers and their layout depends on e->kvq (the
 * KV-quant mode changes component strides), so any persisted checkpoint key
 * MUST carry the kv mode + model identity (see the server's disk header).
 * Slot 0 / solo only, like MTP. */

typedef struct { void*base; size_t stride; } ckcomp;   /* stride = bytes/position */
static int ck_kv_comps(q36_engine*e,int L,ckcomp c[4]){
    if(!e->blk[L].is_attn) return 0;
    if(!e->kvq){   /* fp16 mode: K,V as __half[pos][kv_dim]; no scale planes */
        c[0].base=e->Kc[L]; c[0].stride=(size_t)Q36_KV_DIM*sizeof(__half);
        c[1].base=e->Vc[L]; c[1].stride=(size_t)Q36_KV_DIM*sizeof(__half);
        return 2;
    }
    c[0].base=e->Kc[L]; c[0].stride=Q36_KV_DIM;                      /* int8 codes  */
    c[1].base=e->Ks[L]; c[1].stride=(Q36_KV_DIM/32)*sizeof(__half);  /* fp16 scales */
    c[2].base=e->Vc[L]; c[2].stride=Q36_KV_DIM/2;                    /* e2m1 pairs  */
    c[3].base=e->Vs[L]; c[3].stride=Q36_KV_DIM/32;                   /* ue8m0 scales*/
    return 4;
}

extern "C" size_t q36_engine_state_ssm_bytes(q36_engine*e){
    size_t per=((size_t)NVH*HKD*HVD+(size_t)CONVD*3)*sizeof(float), b=0;
    for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn) b+=per;
    return b;
}
extern "C" size_t q36_engine_state_kv_bytes(q36_engine*e,int npos){
    size_t per=0; ckcomp c[4];
    for(int L=0;L<Q36_N_LAYER;L++){
        int k=ck_kv_comps(e,L,c);
        for(int i=0;i<k;i++) per+=c[i].stride;
    }
    return per*(size_t)npos;
}
extern "C" size_t q36_engine_state_size(q36_engine*e,int n){
    return q36_engine_state_ssm_bytes(e)+q36_engine_state_kv_bytes(e,n);
}

/* SSM state + conv history <-> host, all recurrent layers in layer order.
 * Loading rewinds the conversation to an arbitrary boundary, so the nextn
 * (MTP) cache is invalidated -- the next prefill re-primes it. */
extern "C" int q36_engine_ssm_save(q36_engine*e,void*dst){
    uint8_t*p=(uint8_t*)dst;
    size_t SB=(size_t)NVH*HKD*HVD*sizeof(float), CB=(size_t)CONVD*3*sizeof(float);
    for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn){
        CK(cudaMemcpy(p,e->Sstate[L],SB,cudaMemcpyDeviceToHost));   p+=SB;
        CK(cudaMemcpy(p,e->convhist[L],CB,cudaMemcpyDeviceToHost)); p+=CB;
    }
    return 0;
}
extern "C" int q36_engine_ssm_load(q36_engine*e,const void*src){
    const uint8_t*p=(const uint8_t*)src;
    size_t SB=(size_t)NVH*HKD*HVD*sizeof(float), CB=(size_t)CONVD*3*sizeof(float);
    for(int L=0;L<Q36_N_LAYER;L++) if(!e->blk[L].is_attn){
        CK(cudaMemcpy(e->Sstate[L],p,SB,cudaMemcpyHostToDevice));   p+=SB;
        CK(cudaMemcpy(e->convhist[L],p,CB,cudaMemcpyHostToDevice)); p+=CB;
    }
    e->mtp_base=-1; e->mtp_primed=0;
    return 0;
}

/* attention KV <-> host blob.  buf holds positions [p0,p1) as per-layer
 * component slabs; transfer only the sub-range [a,b) (p0<=a<b<=p1) in
 * either direction.  The sub-range matters on restore: KV is positional
 * and append-only, so pages the GPU still holds for the same prefix are
 * valid as-is and only the stale range needs the H2D copy. */
extern "C" int q36_engine_kv_range(q36_engine*e,int p0,int p1,void*buf,int a,int b,int save){
    if(p0<0||p1<p0||p1>e->ctx||a<p0||b>p1||a>=b) return -1;
    uint8_t*bp=(uint8_t*)buf; ckcomp c[4];
    for(int L=0;L<Q36_N_LAYER;L++){
        int k=ck_kv_comps(e,L,c);
        for(int i=0;i<k;i++){
            uint8_t*dev =(uint8_t*)c[i].base+(size_t)a*c[i].stride;
            uint8_t*host=bp+(size_t)(a-p0)*c[i].stride;
            size_t sz=(size_t)(b-a)*c[i].stride;
            if(save) CK(cudaMemcpy(host,dev,sz,cudaMemcpyDeviceToHost));
            else     CK(cudaMemcpy(dev,host,sz,cudaMemcpyHostToDevice));
            bp+=(size_t)(p1-p0)*c[i].stride;
        }
    }
    return 0;
}

/* design-doc C ABI: one flat blob = [ssm section][kv slabs for [0,n)].
 * save = snapshot after exactly n prefill positions; load = restore, after
 * which the caller prefills from position n. */
extern "C" int q36_engine_state_save(q36_engine*e,int n,void*dst){
    if(n<1||n>e->ctx) return -1;
    if(q36_engine_ssm_save(e,dst)) return -1;
    return q36_engine_kv_range(e,0,n,(uint8_t*)dst+q36_engine_state_ssm_bytes(e),0,n,1);
}
extern "C" int q36_engine_state_load(q36_engine*e,int n,const void*src){
    if(n<1||n>e->ctx) return -1;
    if(q36_engine_ssm_load(e,src)) return -1;
    return q36_engine_kv_range(e,0,n,(uint8_t*)src+q36_engine_state_ssm_bytes(e),0,n,0);
}

/* Isolation gate for checkpoint export/import (same contract as the MTP
 * gate): a snapshot at the end-of-prompt boundary, restored after arbitrary
 * intervening damage, must reproduce the greedy continuation
 * TOKEN-FOR-TOKEN and the SSM state BIT-FOR-BIT.  Exercises both restore
 * shapes the server uses: full reload, and SSM + partial-KV reload (only
 * the pages an interloper conversation overwrote).  0 = pass. */
extern "C" int q36_engine_state_selftest(q36_engine*e){
    enum{G=24,M=64};                     /* continuation length, damage size */
    int N=e->ctx/2<2500?e->ctx/2:2500;   /* >Q36_PF_CHUNK when ctx allows: cross a chunk boundary */
    if(N<=M){ fprintf(stderr,"state-selftest: ctx %d too small\n",e->ctx); return -1; }
    int*toks=(int*)malloc((size_t)N*sizeof(int));
    unsigned long long rs=0x243F6A8885A308D3ULL;
    for(int i=0;i<N;i++){ rs=rs*6364136223846793005ULL+1442695040888963407ULL;
        toks[i]=(int)((rs>>33)%200000); }
    float st=e->s_temp; e->s_temp=0.f;   /* greedy: bit-exact compare */
    size_t ssmb=q36_engine_state_ssm_bytes(e);
    uint8_t*blob=(uint8_t*)malloc(q36_engine_state_size(e,N));
    uint8_t*ssm2=(uint8_t*)malloc(ssmb);
    int ref[G],out[G],bad=0;
    /* reference: prefill, snapshot at the boundary, greedy-continue */
    q36_engine_reset(e);
    ref[0]=q36_engine_prefill(e,toks,N);
    if(q36_engine_state_save(e,N,blob)){ fprintf(stderr,"state-selftest: save refused\n"); bad++; }
    for(int i=1;i<G;i++) ref[i]=q36_engine_step(e,ref[i-1],N+i-1);
    /* A: full restore after decode advanced SSM state + appended KV */
    if(q36_engine_state_load(e,N,blob)){ fprintf(stderr,"state-selftest: load refused\n"); bad++; }
    if(q36_engine_ssm_save(e,ssm2)==0 && memcmp(blob,ssm2,ssmb)){
        fprintf(stderr,"state-selftest: A ssm NOT bit-exact after load\n"); bad++; }
    out[0]=ref[0];
    for(int i=1;i<G;i++) out[i]=q36_engine_step(e,out[i-1],N+i-1);
    { int mb=0; for(int i=0;i<G;i++) if(out[i]!=ref[i]) mb++;
      if(mb){ fprintf(stderr,"state-selftest: A full restore: %d/%d continuation tokens diverge\n",mb,G); bad++; }
      else fprintf(stderr,"state-selftest: A full restore PASS (ssm bit-exact, %d tokens match)\n",G); }
    /* B: the server's partial path -- another conversation overwrote KV
     * [0,M) and the SSM state; reload the SSM blob + only the stale pages */
    q36_engine_reset(e);
    { int junk[M]; for(int i=0;i<M;i++) junk[i]=toks[N-1-i];
      q36_engine_prefill(e,junk,M); }
    if(q36_engine_ssm_load(e,blob)){ fprintf(stderr,"state-selftest: ssm load refused\n"); bad++; }
    if(q36_engine_kv_range(e,0,N,blob+ssmb,0,M,0)){ fprintf(stderr,"state-selftest: kv range refused\n"); bad++; }
    out[0]=ref[0];
    for(int i=1;i<G;i++) out[i]=q36_engine_step(e,out[i-1],N+i-1);
    { int mb=0; for(int i=0;i<G;i++) if(out[i]!=ref[i]) mb++;
      if(mb){ fprintf(stderr,"state-selftest: B partial-KV restore: %d/%d continuation tokens diverge\n",mb,G); bad++; }
      else fprintf(stderr,"state-selftest: B partial-KV restore PASS (%d tokens match)\n",G); }
    free(toks); free(blob); free(ssm2);
    e->s_temp=st; q36_engine_reset(e);
    fprintf(stderr,"state-selftest: %s\n",bad?"FAIL":"PASS");
    return bad?1:0;
}

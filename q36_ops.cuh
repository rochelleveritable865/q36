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

#ifndef QWEN36_OPS_CUH
#define QWEN36_OPS_CUH

/* Clean-room CUDA ops for the q36 decode forward pass.  Implemented from the
 * public model architecture (gated GQA attention, softmax-routed MoE with a
 * sigmoid-gated shared expert, gated-DeltaNet linear attention), NOT ported
 * from any existing engine.  Correctness first; these are the reference kernels
 * that the optimized versions must match. */

#include "q36_dequant.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define WARP 32
__device__ __forceinline__ float warpsum(float v){
    #pragma unroll
    for(int o=WARP/2;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
    return v;
}
/* butterfly variant: the sum lands in every lane */
__device__ __forceinline__ float warpsum_xor(float v){
    #pragma unroll
    for(int o=WARP/2;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffff,v,o);
    return v;
}
__device__ __forceinline__ float silu(float x){ return x/(1.f+__expf(-x)); }
__device__ __forceinline__ float sigm(float x){ return 1.f/(1.f+__expf(-x)); }

/* ---- elementwise helpers ---------------------------------------------- */
__global__ void k_add(float*a,const float*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=b[i];}
__global__ void k_mul(float*a,const float*b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]*=b[i];}
__global__ void k_axpy(float*a,const float*b,float s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=s*b[i];}
__global__ void k_silu_mul(float*g,const float*u,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)g[i]=silu(g[i])*u[i];}
__global__ void k_sigmoid_mul(float*a,const float*g,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]*=sigm(g[i]);}

/* RMSNorm over n (single block). */
__global__ void k_rmsnorm(const float*x,const float*w,float*y,int n,float eps){
    __shared__ float ss;
    float loc=0; for(int i=threadIdx.x;i<n;i+=blockDim.x) loc+=x[i]*x[i];
    loc=warpsum(loc); __shared__ float part[32];
    int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    if(lane==0)part[wid]=loc; __syncthreads();
    if(threadIdx.x==0){float s=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)s+=part[i];ss=rsqrtf(s/n+eps);}
    __syncthreads();
    for(int i=threadIdx.x;i<n;i+=blockDim.x) y[i]=x[i]*ss*w[i];
}

/* Per-head RMSNorm over head_dim with a shared [head_dim] gain (q/k norm). */
__global__ void k_head_rmsnorm(float*x,const float*w,int head_dim,float eps){
    int h=blockIdx.x; float*xh=x+h*head_dim;
    __shared__ float ss; if(threadIdx.x==0) ss=0; __syncthreads();
    float loc=0;
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) loc+=xh[i]*xh[i];
    loc=warpsum(loc); if((threadIdx.x&31)==0) atomicAdd(&ss,loc);
    __syncthreads(); float sc=rsqrtf(ss/head_dim+eps);
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) xh[i]=xh[i]*sc*w[i];
}
/* Per-head true L2 norm over head_dim (delta-net q/k): x / sqrt(sum(x^2)+eps).
 * NOTE: this is L2 (no division by n), unlike RMSNorm -- getting this wrong
 * scales q,k by sqrt(head_dim) and blows up the delta-rule dot products. */
__global__ void k_head_l2norm(float*x,int head_dim,float eps){
    int h=blockIdx.x; float*xh=x+h*head_dim;
    __shared__ float ss; if(threadIdx.x==0) ss=0; __syncthreads();
    float loc=0;
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) loc+=xh[i]*xh[i];
    loc=warpsum(loc); if((threadIdx.x&31)==0) atomicAdd(&ss,loc);
    __syncthreads(); float sc=rsqrtf(ss+eps);
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) xh[i]=xh[i]*sc;
}

/* Partial NeoX RoPE over the first rot_dim dims of each head. Position `pos`. */
__global__ void k_rope(float*x,int head_dim,int rot_dim,int pos,float base){
    int h=blockIdx.x, i=threadIdx.x; if(i>=rot_dim/2) return;
    float*xh=x+h*head_dim;
    float inv=__powf(base,-2.0f*i/rot_dim), ang=pos*inv, c=__cosf(ang),s=__sinf(ang);
    float a=xh[i], b=xh[i+rot_dim/2];
    xh[i]=a*c-b*s; xh[i+rot_dim/2]=a*s+b*c;
}

/* Split-Q8_0 row dot: quants and scales live in separate aligned regions
 * (repacked at upload), so lanes issue coalesced 128-bit loads.  Lane l
 * handles half-blocks of 16 elems: a warp covers 512 contiguous bytes. */
__device__ __forceinline__ float dot_q8_0_split(const int8_t*qs,const __half*ds,
                                                const float*x,int K,int lane){
    float acc=0; int nhb=K/16;
    for(int hb=lane; hb<nhb; hb+=WARP){
        int4 q4=*(const int4*)(qs+(size_t)hb*16);
        float d=__half2float(ds[hb>>1]);
        const float4*x4=(const float4*)(x+(size_t)hb*16);
        float4 xa=x4[0], xb=x4[1], xc=x4[2], xd=x4[3];
        char4 c0=*(char4*)&q4.x, c1=*(char4*)&q4.y, c2=*(char4*)&q4.z, c3=*(char4*)&q4.w;
        float s= c0.x*xa.x + c0.y*xa.y + c0.z*xa.z + c0.w*xa.w
               + c1.x*xb.x + c1.y*xb.y + c1.z*xb.z + c1.w*xb.w
               + c2.x*xc.x + c2.y*xc.y + c2.z*xc.z + c2.w*xc.w
               + c3.x*xd.x + c3.y*xd.y + c3.z*xd.z + c3.w*xd.w;
        acc += d*s;
    }
    return warpsum(acc);
}

/* Branchless MXFP4 (E2M1) decode: the constant-memory table serializes under
 * divergent per-lane indices; pure ALU is faster.  Codes {0..7} map to
 * {0,1,2,3,4,6,8,12} = ((2|(c&1))<<(c>>1))>>1 with a zero mask; bit 3 = sign. */
__device__ __forceinline__ float mxfp4_val(uint32_t c){
    uint32_t a=c&7u;
    float mag=(float)(((2u|(a&1u))<<(a>>1))>>1);
    mag=a?mag:0.f;
    return (c&8u)?-mag:mag;
}

/* Split-MXFP4 row dot: nibbles and E8M0 scales in separate regions; one
 * 128-bit load fetches a full 32-elem block per lane. */
/* one 32-elem MXFP4 block dot with explicit float4 activation loads */
__device__ __forceinline__ float mxfp4_block_dot(int4 q,const float*xb){
    const uint8_t*qb=(const uint8_t*)&q;
    const float4*x4=(const float4*)xb;
    float xr[32];
    #pragma unroll
    for(int t=0;t<8;t++){ float4 v=x4[t]; xr[t*4]=v.x; xr[t*4+1]=v.y; xr[t*4+2]=v.z; xr[t*4+3]=v.w; }
    float s=0;
    #pragma unroll
    for(int j=0;j<16;j++){ s+=mxfp4_val(qb[j]&0xF)*xr[j]; s+=mxfp4_val(qb[j]>>4)*xr[j+16]; }
    return s;
}
__device__ __forceinline__ float dot_mxfp4_split(const uint8_t*qs,const uint8_t*es,
                                                 const float*x,int K,int lane){
    /* GEMV shapes here run ~one wave of warps, so DRAM latency must hide
     * behind ILP, not occupancy: issue BOTH blocks' weight loads up front. */
    float acc=0; int nb=K/32;
    for(int b0=lane;b0<nb;b0+=2*WARP){
        int b1=b0+WARP;
        int4 qA=*(const int4*)(qs+(size_t)b0*16);
        float dA=q36_e8m0_half(es[b0]);
        int4 qB={0,0,0,0}; float dB=0.f;
        if(b1<nb){ qB=*(const int4*)(qs+(size_t)b1*16); dB=q36_e8m0_half(es[b1]); }
        acc+=dA*mxfp4_block_dot(qA,x+(size_t)b0*32);
        if(b1<nb) acc+=dB*mxfp4_block_dot(qB,x+(size_t)b1*32);
    }
    return warpsum(acc);
}

/* Split-e4m3 row dot (FP8 weights + per-32 ue8m0 scales): the output-head
 * format.  Same half-block-per-lane geometry as the q8 dot. */
__device__ __forceinline__ float dot_e4m3_split(const uint8_t*qs,const uint8_t*es,
                                                const float*x,int K,int lane){
    float acc=0; int nhb=K/16;
    for(int hb=lane; hb<nhb; hb+=WARP){
        uint4 q4=*(const uint4*)(qs+(size_t)hb*16);
        float d=q36_e8m0_half(es[hb>>1])*2.f;   /* true 2^(e-127) */
        const float4*x4=(const float4*)(x+(size_t)hb*16);
        float4 xa=x4[0], xb=x4[1], xc=x4[2], xd=x4[3];
        const __nv_fp8_e4m3*f8=(const __nv_fp8_e4m3*)&q4;
        float s= (float)f8[0]*xa.x + (float)f8[1]*xa.y + (float)f8[2]*xa.z + (float)f8[3]*xa.w
               + (float)f8[4]*xb.x + (float)f8[5]*xb.y + (float)f8[6]*xb.z + (float)f8[7]*xb.w
               + (float)f8[8]*xc.x + (float)f8[9]*xc.y + (float)f8[10]*xc.z + (float)f8[11]*xc.w
               + (float)f8[12]*xd.x + (float)f8[13]*xd.y + (float)f8[14]*xd.z + (float)f8[15]*xd.w;
        acc += d*s;
    }
    return warpsum(acc);
}

/* Split-NVFP4 row dot (E2M1 nibbles + per-16 e4m3 scales): the 4.5-bit
 * output-head format.  Nibbles hold true-E2M1 bit patterns whose halved-
 * table values are {0,1,2,3,4,6,8,12} = 2x E2M1, so the stored amax/12
 * scale is doubled here and decode uses the sm_120a HARDWARE fp4 convert
 * cvt.rn.f16x2.e2m1x2 (the mxfp4_val ALU version was ~12 ops/pair and ate
 * the byte savings: measured 0% decode gain). */
__device__ __forceinline__ float2 q36_e2m1x2(uint32_t byte){
    unsigned r;
    asm("{.reg .b8 lo,hi; mov.b16 {lo,hi}, %1; cvt.rn.f16x2.e2m1x2 %0, lo;}\n"
        :"=r"(r):"h"((unsigned short)byte));
    return __half22float2(*(__half2*)&r);
}
__device__ __forceinline__ float dot_nvfp4_split(const uint8_t*qs,const uint8_t*es,
                                                 const float*x,int K,int lane){
    float acc=0; int nb=K/16;
    for(int b=lane; b<nb; b+=WARP){
        uint2 q2=*(const uint2*)(qs+(size_t)b*8);
        const uint8_t*n8=(const uint8_t*)&q2;
        float d=(float)*(const __nv_fp8_e4m3*)&es[b]*2.f;
        const float4*x4=(const float4*)(x+(size_t)b*16);
        float4 xa=x4[0], xb=x4[1], xc=x4[2], xd=x4[3];
        float2 v0=q36_e2m1x2(n8[0]), v1=q36_e2m1x2(n8[1]);
        float2 v2=q36_e2m1x2(n8[2]), v3=q36_e2m1x2(n8[3]);
        float2 v4=q36_e2m1x2(n8[4]), v5=q36_e2m1x2(n8[5]);
        float2 v6=q36_e2m1x2(n8[6]), v7=q36_e2m1x2(n8[7]);
        float s= v0.x*xa.x + v0.y*xa.y + v1.x*xa.z + v1.y*xa.w
               + v2.x*xb.x + v2.y*xb.y + v3.x*xb.z + v3.y*xb.w
               + v4.x*xc.x + v4.y*xc.y + v5.x*xc.z + v5.y*xc.w
               + v6.x*xd.x + v6.y*xd.y + v7.x*xd.z + v7.y*xd.w;
        acc += d*s;
    }
    return warpsum(acc);
}

/* Per-warp quantized row-dot helpers (duplicated per-TU; no -rdc needed). */
__device__ __forceinline__ float dot_q8_0_row(const block_q8_0*row,const float*x,int K,int lane){
    float acc=0; int nb=K/32;
    for(int b=lane;b<nb;b+=WARP){ const block_q8_0*blk=row+b; float d=q36_fp16(blk->d);
        const float*xb=x+b*32;
        #pragma unroll
        for(int j=0;j<32;j++) acc+=d*blk->qs[j]*xb[j]; }
    return warpsum(acc);
}
__device__ __forceinline__ float dot_mxfp4_row(const block_mxfp4*row,const float*x,int K,int lane){
    float acc=0; int nb=K/32;
    for(int b=lane;b<nb;b+=WARP){ const block_mxfp4*blk=row+b; float d=q36_e8m0_half(blk->e);
        const float*xb=x+b*32;
        #pragma unroll
        for(int j=0;j<16;j++){ acc+=d*mxfp4_val(blk->qs[j]&0xF)*xb[j]; acc+=d*mxfp4_val(blk->qs[j]>>4)*xb[j+16]; } }
    return warpsum(acc);
}
__device__ __forceinline__ float dot_q5_K_row(const block_q5_K*row,const float*x,int K,int lane){
    float acc=0; int nb=K/QK_K;
    for(int sb=lane;sb<nb;sb+=WARP){ float tmp[QK_K]; q36_deq_q5_K(row+sb,tmp);
        const float*xb=x+sb*QK_K;
        #pragma unroll
        for(int j=0;j<QK_K;j++) acc+=tmp[j]*xb[j]; }
    return warpsum(acc);
}
__device__ __forceinline__ float dot_q6_K_row(const block_q6_K*row,const float*x,int K,int lane){
    float acc=0; int nb=K/QK_K;
    for(int sb=lane;sb<nb;sb+=WARP){ float tmp[QK_K]; q36_deq_q6_K(row+sb,tmp);
        const float*xb=x+sb*QK_K;
        #pragma unroll
        for(int j=0;j<QK_K;j++) acc+=tmp[j]*xb[j]; }
    return warpsum(acc);
}

/* F32 dense matvec y[M]=W[M,K]@x[K], W row-major f32 (router, small). */
__global__ void k_matvec_f32(const float*W,const float*x,float*y,int M,int K){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP, lane=threadIdx.x&31;
    if(warp>=M)return; const float*row=W+(size_t)warp*K; float acc=0;
    for(int k=lane;k<K;k+=WARP) acc+=row[k]*x[k];
    acc=warpsum(acc); if(lane==0)y[warp]=acc;
}

/* Embedding lookup: dequant one Q8_0 row of tok_embd into h[D]. */
__global__ void k_embed_q8(const block_q8_0*emb,int token,float*h,int D){
    const block_q8_0*row=emb+(size_t)token*(D/32);
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=D)return;
    const block_q8_0*b=row+i/32; float d=q36_fp16(b->d); h[i]=d*b->qs[i%32];
}

/* ---- gated GQA attention (decode: one query position `pos`) ------------
 * Q split per head is [q(head_dim), gate(head_dim)] interleaved.  K/V cache is
 * f32 [layer][pos][kv_head*head_dim].  One block per query head. */
__global__ void k_attn_decode(const float*Q,        /* [n_head, head_dim] (post-norm+rope) */
                              const float*Kc,const float*Vc, /* caches, contiguous [pos+1][kv_dim] */
                              float*out,             /* [n_head*head_dim] */
                              int n_head,int n_head_kv,int head_dim,int seqlen,float scale){
    int h=blockIdx.x; if(h>=n_head)return;
    int kvh=h/(n_head/n_head_kv);
    int kv_dim=n_head_kv*head_dim;
    const float*q=Q+h*head_dim;
    extern __shared__ float sh[];        /* scores[seqlen] */
    /* scores */
    for(int t=threadIdx.x;t<seqlen;t+=blockDim.x){
        const float*k=Kc+(size_t)t*kv_dim+kvh*head_dim;
        float s=0; for(int d=0;d<head_dim;d++) s+=q[d]*k[d];
        sh[t]=s*scale;
    }
    __syncthreads();
    /* softmax (thread 0; seqlen small at bring-up, optimize later) */
    __shared__ float sm;
    if(threadIdx.x==0){ float m=-1e30f; for(int t=0;t<seqlen;t++) if(sh[t]>m)m=sh[t];
        float s=0; for(int t=0;t<seqlen;t++){ sh[t]=__expf(sh[t]-m); s+=sh[t]; } sm=s; }
    __syncthreads();
    float inv=1.f/sm;
    for(int d=threadIdx.x;d<head_dim;d+=blockDim.x){
        float acc=0;
        for(int t=0;t<seqlen;t++){ const float*v=Vc+(size_t)t*kv_dim+kvh*head_dim; acc+=sh[t]*v[d]; }
        out[h*head_dim+d]=acc*inv;
    }
}

/* Split Qfull[8192] -> q[4096] (drop gate) and gate[4096], per head layout
 * [q256,gate256]*16. */
__global__ void k_split_qgate(const float*Qfull,float*q,float*gate,int n_head,int head_dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x, tot=n_head*head_dim; if(i>=tot)return;
    int h=i/head_dim, d=i%head_dim;
    q[i]=Qfull[h*head_dim*2+d];
    gate[i]=Qfull[h*head_dim*2+head_dim+d];
}

/* ---- MoE routing: softmax over experts, pick top-k, renormalize --------
 * Single WARP per batch row: each lane holds 8 of the 256 logits in
 * registers; top-k = k rounds of shuffle argmax with in-register masking.
 * No shared memory, no block syncs -- the 17 __syncthreads of the block
 * version made this 8us/launch; this runs in ~1us. */
__global__ void k_router_topk(const float*logits,int n_expert,int k,
                              int*idx,float*wt,float scale,int renorm){
    logits+=(size_t)blockIdx.x*n_expert; idx+=(size_t)blockIdx.x*k; wt+=(size_t)blockIdx.x*k;
    int lane=threadIdx.x&31;
    float v[8];                       /* logit (lane + j*32) */
    #pragma unroll
    for(int j=0;j<8;j++) v[j]=logits[lane+j*32];
    float m=-1e30f;
    #pragma unroll
    for(int j=0;j<8;j++) m=fmaxf(m,v[j]);
    #pragma unroll
    for(int o=16;o>0;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,o));
    float s=0;
    #pragma unroll
    for(int j=0;j<8;j++) s+=__expf(v[j]-m);
    #pragma unroll
    for(int o=16;o>0;o>>=1) s+=__shfl_xor_sync(0xffffffff,s,o);
    for(int r=0;r<k;r++){
        float bm=-1e30f; int bj=0;
        #pragma unroll
        for(int j=0;j<8;j++) if(v[j]>bm){bm=v[j];bj=j;}
        float wv=bm; int wi=lane+bj*32;
        #pragma unroll
        for(int o=16;o>0;o>>=1){
            float ov=__shfl_xor_sync(0xffffffff,wv,o);
            int   oi=__shfl_xor_sync(0xffffffff,wi,o);
            if(ov>wv||(ov==wv&&oi<wi)){wv=ov;wi=oi;}
        }
        if(lane==0){ idx[r]=wi; wt[r]=__expf(wv-m)/s; }
        if(lane==(wi&31)) v[wi>>5]=-1e30f;    /* mask the winner */
    }
    if(lane==0&&renorm){ float ws=0; for(int r=0;r<k;r++)ws+=wt[r];
        for(int r=0;r<k;r++) wt[r]=wt[r]/ws*scale; }
    else if(lane==0){ for(int r=0;r<k;r++) wt[r]*=scale; }
}

#endif

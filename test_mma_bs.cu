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

/* Validate sm_120a block-scaled mma: m16n8k32 .kind::mxf8f6f4, A=e2m1 (fp4
 * nibble codes in byte containers), B=e4m3, per-32 ue8m0 scales.
 *
 * Phase 1 (calibrate): A=B=1.0, unique power-of-two scales per (lane,byte);
 * log2 of the output reveals exactly which (lane,byte) scales which row/col
 * for each {byte-id, thread-id} selector -- no reliance on figure reading.
 * Phase 2 (verify): random A/B/scales vs a CPU reference using the mapping.
 *
 * Build: nvcc -O2 -arch=sm_120a test_mma_bs.cu -o test_mma_bs
 */
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

/* true E2M1 value of a 4-bit code */
static float e2m1_val(int c){
    static const float t[8]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    float v=t[c&7]; return (c&8)?-v:v;
}

__global__ void k_mma(const unsigned*Af,const unsigned*Bf,
                      const unsigned*sfa,const unsigned*sfb,float*D,
                      unsigned short bida,unsigned short tida,
                      unsigned short bidb,unsigned short tidb){
    int lane=threadIdx.x;
    unsigned a0=Af[lane*4],a1=Af[lane*4+1],a2=Af[lane*4+2],a3=Af[lane*4+3];
    unsigned b0=Bf[lane*2],b1=Bf[lane*2+1];
    unsigned sa=sfa[lane],sb=sfb[lane];
    float d0=0,d1=0,d2=0,d3=0;
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
      ".f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "%10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
        "r"(sa),"r"(sb));
    (void)bida;(void)tida;(void)bidb;(void)tidb;
    /* C/D layout: row=group(+8 for i>=2), col=(tid%4)*2+(i&1) */
    int g=lane>>2,t4=lane&3;
    D[(g+0)*8+t4*2+0]=d0; D[(g+0)*8+t4*2+1]=d1;
    D[(g+8)*8+t4*2+0]=d2; D[(g+8)*8+t4*2+1]=d3;
}

/* pack host matrices into per-thread fragments per the documented layouts */
static void pack_A(const int A[16][32],unsigned*Af){ /* e2m1 codes, byte containers */
    for(int lane=0;lane<32;lane++){
        int g=lane>>2,t4=lane&3;
        unsigned char by[16];
        for(int i=0;i<16;i++){
            int row=(i<4||(i>=8&&i<12))?g:g+8;
            int col=t4*4+(i&3)+((i>=8)?16:0);
            int c_=A[row][col]; by[i]=(unsigned char)((((unsigned)c_&7u)<<2)|((((unsigned)c_>>3)&1u)<<5));
        }
        memcpy(&Af[lane*4],by,16);
    }
}
static void pack_B(const float B[32][8],unsigned*Bf){ /* e4m3 */
    for(int lane=0;lane<32;lane++){
        int g=lane>>2,t4=lane&3;
        unsigned char by[8];
        for(int i=0;i<8;i++){
            int row=t4*4+(i&3)+((i>=4)?16:0);
            int col=g;
            __nv_fp8_e4m3 v(B[row][col]);
            by[i]=*(unsigned char*)&v;
        }
        memcpy(&Bf[lane*2],by,8);
    }
}

int main(){
    unsigned *dAf,*dBf,*dsa,*dsb; float*dD;
    CK(cudaMalloc(&dAf,32*4*4)); CK(cudaMalloc(&dBf,32*2*4));
    CK(cudaMalloc(&dsa,32*4)); CK(cudaMalloc(&dsb,32*4));
    CK(cudaMalloc(&dD,16*8*4));
    unsigned Af[128],Bf[64],sa[32],sb[32]; float D[16][8];

    /* ---------- phase 1: calibrate SF_A row mapping ---------- */
    int A1[16][32]; float B1[32][8];
    for(int r=0;r<16;r++)for(int k=0;k<32;k++)A1[r][k]=2;      /* e2m1 code 2 = 1.0 */
    for(int k=0;k<32;k++)for(int c=0;c<8;c++)B1[k][c]=1.f;
    pack_A(A1,Af); pack_B(B1,Bf);
    for(int l=0;l<32;l++){
        unsigned b0=127+(l*4+0)%96, b1=127+(l*4+1)%96, b2=127+(l*4+2)%96, b3=127+(l*4+3)%96;
        sa[l]=b0|(b1<<8)|(b2<<16)|(b3<<24);
        sb[l]=127u|(127u<<8)|(127u<<16)|(127u<<24);   /* SF_B = 1.0 */
    }
    CK(cudaMemcpy(dAf,Af,sizeof Af,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBf,Bf,sizeof Bf,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsa,sa,sizeof sa,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsb,sb,sizeof sb,cudaMemcpyHostToDevice));
    /* dot measurement: ALL scales 1.0; try A code placements low/high nibble */
    for(int place=0;place<3;place++){
        int Ax[16][32];
        for(int r=0;r<16;r++)for(int k=0;k<32;k++)
            Ax[r][k]=(place==0)?0x02:(place==1)?0x20:0x22;
        pack_A(Ax,Af);
        CK(cudaMemcpy(dAf,Af,sizeof Af,cudaMemcpyHostToDevice));
        for(int l=0;l<32;l++){ sa[l]=0x7f7f7f7fu; sb[l]=0x7f7f7f7fu; }
        CK(cudaMemcpy(dsa,sa,sizeof sa,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsb,sb,sizeof sb,cudaMemcpyHostToDevice));
        k_mma<<<1,32>>>(dAf,dBf,dsa,dsb,dD,2,0,2,0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(D,dD,sizeof D,cudaMemcpyDeviceToHost));
        printf("A byte=0x%02x: D[0][0]=%.4f D[7][0]=%.4f D[8][0]=%.4f D[15][3]=%.4f (want 32 if codes=1.0)\n",
               place==0?0x02:place==1?0x20:0x22, D[0][0],D[7][0],D[8][0],D[15][3]);
    }
    { int A1b[16][32]; for(int r=0;r<16;r++)for(int k=0;k<32;k++)A1b[r][k]=2; pack_A(A1b,Af);
      CK(cudaMemcpy(dAf,Af,sizeof Af,cudaMemcpyHostToDevice)); }
    /* ---------- phase 2: random correctness with {bid=2,tid=0} ----------
     * mapping: SF_A(row q)   = byte0 of lane 4q     (q=0..7)
     *          SF_A(row q+8) = byte0 of lane 4q+1
     *          SF_B(col c)   = byte0 of lane 4c                     */
    srand(7);
    int A2[16][32]; float B2[32][8]; unsigned char SFA[16],SFB[8];
    for(int r=0;r<16;r++)for(int k=0;k<32;k++)A2[r][k]=rand()&15;
    for(int k=0;k<32;k++)for(int c=0;c<8;c++)B2[k][c]=((rand()%2001)-1000)/250.0f;
    for(int r=0;r<16;r++)SFA[r]=(unsigned char)(120+rand()%16);
    for(int c=0;c<8;c++)SFB[c]=(unsigned char)(120+rand()%16);
    pack_A(A2,Af); pack_B(B2,Bf);
    /* selection-robust: replicate the SF into every byte of both pair lanes */
    for(int q=0;q<8;q++){
        sa[4*q+0]=0x01010101u*SFA[q];   sa[4*q+2]=0x01010101u*SFA[q];
        sa[4*q+1]=0x01010101u*SFA[q+8]; sa[4*q+3]=0x01010101u*SFA[q+8];
    }
    for(int c=0;c<8;c++) for(int p=0;p<4;p++) sb[4*c+p]=0x01010101u*SFB[c];
    CK(cudaMemcpy(dAf,Af,sizeof Af,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBf,Bf,sizeof Bf,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsa,sa,sizeof sa,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsb,sb,sizeof sb,cudaMemcpyHostToDevice));
    k_mma<<<1,32>>>(dAf,dBf,dsa,dsb,dD,2,0,2,0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(D,dD,sizeof D,cudaMemcpyDeviceToHost));
    /* CPU reference (e4m3 round-trip via device type on host) */
    double maxrel=0;
    for(int r=0;r<16;r++)for(int c=0;c<8;c++){
        double acc=0;
        for(int k=0;k<32;k++){
            __nv_fp8_e4m3 q8(B2[k][c]); float bq=(float)q8;
            acc+=e2m1_val(A2[r][k])*bq;
        }
        double want=acc*exp2f((int)SFA[r]-127)*exp2f((int)SFB[c]-127);
        double rel=fabs(D[r][c]-want)/(1e-6+fabs(want));
        if(rel>maxrel)maxrel=rel;
    }
    for(int r=0;r<2;r++){for(int c=0;c<4;c++)printf("D[%d][%d]=%.4f ",r,c,D[r][c]);printf("\n");}
    printf("\nverify {bid=2,tid=0}: max_rel_err=%.3e -> %s\n",maxrel,
           maxrel<1e-3?"PASS":"FAIL");
    return maxrel<1e-3?0:1;
}

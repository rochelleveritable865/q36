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

/* q36 driver.
 *   ./q36                       -> interactive chat REPL
 *   ./q36 --tokens 760 ...      -> raw-token bring-up/validation mode
 *
 * Tokenization is fully native (byte-level BPE from the GGUF vocab+merges,
 * parity-verified token-for-token against the reference tokenizer). */
#include "q36_model.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct q36_engine q36_engine;
q36_engine* q36_engine_create(q36_model*m,int ctx);
int  q36_engine_step(q36_engine*e,int token,int pos);
int  q36_engine_step_ex(q36_engine*e,int token,int pos,int want_logits);
int  q36_engine_prefill(q36_engine*e,const int*toks,int n);
void q36_engine_reset(q36_engine*e);
void q36_engine_set_kvq(q36_engine*e,int on);
typedef struct q36_multi q36_multi;
q36_multi* q36_multi_create(q36_model*m,int ctx,int ngpu);
q36_engine* q36_multi_engine0(q36_multi*mg);
void q36_multi_reset(q36_multi*mg);
int  q36_multi_prefill(q36_multi*mg,const int*toks,int n);
void q36_engine_set_sampler(q36_engine*e,float temp,int topk,float topp,unsigned long long seed);
int  q36_engine_step_mtp(q36_engine*e,int token,int pos,int out[4]);
int  q36_engine_has_mtp(q36_engine*e);
void q36_engine_set_mtp_k(q36_engine*e,int k);
void q36_engine_mtp_stats(q36_engine*e,long*cycles,long*accepts);
int  q36_engine_mtp_selftest(q36_engine*e);
int  q36_engine_state_selftest(q36_engine*e);

static const char *g_model="/mnt/Qwen3.6-35B-A3B-MXFP4_MOE.gguf";
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* parity harness: --enctest "text" prints native ids for diffing against
 * the reference tokenizer */

int main(int argc,char**argv){
    /* n_predict<0: generate until the model ends naturally (EOS) or the
     * context fills; -n N caps it explicitly */
    int n_predict=-1, ctx=8192, raw[4096], nraw=0, chat=1;
    float temp=0.f, topp=0.95f; int topk=20, kvq=0, ngpu=1, mtp=0, autotrunc=1; unsigned long long seed=42;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-m")&&i+1<argc) g_model=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc) n_predict=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ctx")&&i+1<argc) ctx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--tokens")){ chat=0; while(i+1<argc&&argv[i+1][0]!='-') raw[nraw++]=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--temp")&&i+1<argc) temp=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--top-k")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--top-p")&&i+1<argc) topp=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=(unsigned long long)atoll(argv[++i]);
        else if(!strcmp(argv[i],"--kv-quant")) kvq=1;
        else if(!strcmp(argv[i],"--no-auto-trunc")) autotrunc=0;
        else if(!strcmp(argv[i],"--mtp")){ mtp=1;   /* optional draft depth 1-3 */
            if(i+1<argc&&argv[i+1][0]>='1'&&argv[i+1][0]<='3'&&!argv[i+1][1]) mtp=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--gpus")&&i+1<argc) ngpu=atoi(argv[++i]);
    }
    q36_model m; char err[256];
    if(q36_model_open(&m,g_model,err,sizeof err)){ fprintf(stderr,"open: %s\n",err); return 1; }
    q36_vocab vocab; q36_vocab_init(&vocab,&m.gguf);
    q36_encode_init(&vocab,&m.gguf);
    fprintf(stderr,"loaded (%d attn, %d ssm). uploading weights...\n",m.n_attn,m.n_ssm);
    double t0=now();
    q36_multi*mg=NULL; q36_engine*e;
    if(ngpu>1){ mg=q36_multi_create(&m,ctx,ngpu); e=q36_multi_engine0(mg); }
    else e=q36_engine_create(&m,ctx);
    if(kvq){ q36_engine_set_kvq(e,1); fprintf(stderr,"KV cache: Q8-K / MXFP4-V (2.5x smaller; best for very long contexts)\n"); }
    fprintf(stderr,"ready in %.1fs.\n",now()-t0);
    if(temp>0.f){ q36_engine_set_sampler(e,temp,topk,topp,seed);
        fprintf(stderr,"sampling: temp=%.2f top-k=%d top-p=%.2f seed=%llu\n",temp,topk,topp,seed); }
    if(mtp&&!q36_engine_has_mtp(e)){ fprintf(stderr,"--mtp: model has no nextn module (need the MTP GGUF); disabled\n"); mtp=0; }
    if(mtp&&temp>0.f){ fprintf(stderr,"--mtp: sampled decode not supported yet; disabled\n"); mtp=0; }
    if(mtp&&ngpu>1){ fprintf(stderr,"--mtp: single-GPU only; disabled\n"); mtp=0; }
    if(mtp){ q36_engine_set_mtp_k(e,mtp);
             fprintf(stderr,"MTP speculative decode: ON (draft depth %d)\n",mtp); }
    if(getenv("Q36_MTP_TEST")) return q36_engine_mtp_selftest(e);
    if(getenv("Q36_STATE_TEST")) return q36_engine_state_selftest(e);

    if(getenv("Q36_ENCTEST")){
        char line[8192];
        while(fgets(line,sizeof line,stdin)){
            size_t L2=strlen(line); if(L2&&line[L2-1]=='\n')line[L2-1]=0;
            int ids[4096],k2=q36_encode(&vocab,line,ids,4096);
            for(int i2=0;i2<k2;i2++)printf("%d ",ids[i2]);
            printf("\n"); fflush(stdout);
        }
        return 0;
    }
    if(!chat){ /* raw-token validation mode */
        if(nraw>=ctx){ fprintf(stderr,"[--tokens: %d > ctx %d, clamped to %d]\n",nraw,ctx,ctx-1); nraw=ctx-1; }
        int pos=0,last=-1; for(int i=0;i<nraw;i++){ last=q36_engine_step(e,raw[i],pos); pos++; }
        char tb[64]; q36_detok_one(&vocab,last,tb,sizeof tb); printf("first pred %d [%s]\ngen:",last,tb);
        for(int i=0;(n_predict<0||i<n_predict)&&pos<ctx;i++){ int tk=q36_engine_step(e,last,pos++); last=tk;
            q36_detok_one(&vocab,tk,tb,sizeof tb); printf("%s",tb); fflush(stdout);
            if(tk==(int)m.eos_id) break; }
        printf("\n"); q36_vocab_free(&vocab); q36_model_close(&m); return 0;
    }

    /* interactive chat */
    printf("\nq36 chat. type a question (Ctrl-D to quit).\n");
    char *linebuf=NULL; size_t lcap=0; static int toks[140000];
    for(;;){
        printf("\n> "); fflush(stdout);
        ssize_t nr=getline(&linebuf,&lcap,stdin);
        if(nr<=0){ printf("\nbye\n"); break; }
        if(linebuf[nr-1]=='\n') linebuf[nr-1]=0;
        if(!linebuf[0]) continue;

        size_t plen=strlen(linebuf)+64;
        char *prompt=(char*)malloc(plen);
        snprintf(prompt,plen,
            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",linebuf);
        int nt=q36_encode(&vocab,prompt,toks,140000);
        free(prompt);
        if(nt<=0){ fprintf(stderr,"tokenize failed\n"); continue; }

        /* Auto-fit (DEFAULT; --no-auto-trunc disables): the KV cache holds
         * exactly --ctx positions and the engine refuses oversized prefills,
         * so shrink here -- the single-turn CLI analogue of the server's
         * --auto-purge.  Keep the chat-template header and the TAIL of the
         * content (the question usually sits at the end), and reserve room
         * to actually generate. */
        {
            int reserve=(n_predict>0&&n_predict<ctx/2)?n_predict:ctx/4;
            int budget=ctx-reserve; if(budget<16)budget=16;
            if(nt>budget){
                if(!autotrunc){
                    fprintf(stderr,"[prompt %d tok exceeds budget %d (ctx %d - %d reserved); "
                            "skipped -- drop --no-auto-trunc or raise --ctx]\n",nt,budget,ctx,reserve);
                    continue;
                }
                static int np=-1;   /* template-header token count (estimate) */
                if(np<0){ int hb[16]; np=q36_encode(&vocab,"<|im_start|>user\n",hb,16); if(np<0)np=0; }
                int keep=budget-np; if(keep<8){ np=0; keep=budget; }
                memmove(toks+np,toks+nt-keep,(size_t)keep*sizeof(int));
                fprintf(stderr,"[prompt %d tok > budget %d (ctx %d - %d reserved for output): "
                        "kept header + last %d tok]\n",nt,budget,ctx,reserve,keep);
                nt=np+keep;
            }
        }

        if(mg) q36_multi_reset(mg); else q36_engine_reset(e);
        double tp=now();
        int last=mg? q36_multi_prefill(mg,toks,nt) : q36_engine_prefill(e,toks,nt);
        if(last<0){ fprintf(stderr,"[prefill refused]\n"); continue; }
        int pos=nt;
        double prefill_dt=now()-tp;
        fprintf(stderr,"[prefill %d tok in %.2fs = %.1f tok/s]\n",nt,prefill_dt,nt/prefill_dt);
        double td=now(); int gen=0; char tb[64];
        long c0=0,a0=0; if(mtp) q36_engine_mtp_stats(e,&c0,&a0);
        int pend[3],npend=0,ip=0;   /* extra tokens from an accepted MTP run */
        for(int i=0;(n_predict<0||i<n_predict)&&pos<ctx;i++){
            q36_detok_one(&vocab,last,tb,sizeof tb); printf("%s",tb); fflush(stdout);
            int tk;
            if(ip<npend) tk=pend[ip++];
            else if(mtp){
                int o4[4],k=q36_engine_step_mtp(e,last,pos,o4);
                pos+=k; tk=o4[0];
                npend=k-1; ip=0; for(int j=0;j<npend;j++) pend[j]=o4[j+1];
            } else { tk=q36_engine_step(e,last,pos++);
                if(getenv("Q36_TRACE")) fprintf(stderr,"TRACE %d %d -> %d\n",pos-1,last,tk); }
            last=tk; gen++;
            if(tk==(int)m.eos_id) break;
        }
        double dt=now()-td;
        fprintf(stderr,"\n[%d tok, %.1f tok/s]\n",gen,gen/dt);
        if(mtp){
            long c1,a1; q36_engine_mtp_stats(e,&c1,&a1);
            if(c1>c0) fprintf(stderr,"[mtp K=%d: accept %.1f%% (%ld of %ld drafts), %.2f tok/verify]\n",
                mtp,100.0*(a1-a0)/((c1-c0)*mtp),a1-a0,(c1-c0)*(long)mtp,
                1.0+(double)(a1-a0)/(c1-c0));
        }
    }
    free(linebuf); q36_vocab_free(&vocab); q36_model_close(&m);
    return 0;
}

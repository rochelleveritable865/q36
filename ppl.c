/* q36_ppl: perplexity over a text corpus, llama-perplexity-compatible.
 *
 *   ./q36_ppl -f wiki.test.raw [-m gguf] [--window 512] [--max-windows N]
 *             [--kv-quant]
 *
 * Methodology (matches llama.cpp's llama-perplexity defaults): the token
 * stream is cut into INDEPENDENT windows of --window tokens (no carryover;
 * the tail remainder is discarded); within each window only the second half
 * is scored (the first half is context warm-up).  ppl = exp(sum_nll/count).
 * Because our tokenizer is parity-verified against the reference, running
 * llama-perplexity on the SAME GGUF and corpus gives a direct
 * kernel-correctness reference: any delta isolates engine numerics
 * (fp16 FA/QK^T, FP8-e4m3 output head, requantized experts, KV mode).
 *
 * --- Attribution & License ---
 * This file is a mixed-license source file:
 *
 * 1. The custom engine evaluation harness, corpus tokenization wrappers, and
 *    file execution logic are licensed under the GNU Affero General Public
 *    License version 3 (AGPL-3.0).
 *
 *    Copyright (C) 2026 Ambud Sharma
 *
 * 2. The perplexity measurement windowing methodology is aligned with llama.cpp
 *    (licensed under the MIT License):
 *
 *    Copyright (c) 2023-2026 Georgi Gerganov and llama.cpp contributors
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all
 *    copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */
#include "q36_model.h"
#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct q36_engine q36_engine;
q36_engine* q36_engine_create(q36_model*m,int ctx);
void q36_engine_set_kvq(q36_engine*e,int on);
double q36_engine_nll(q36_engine*e,const int*toks,int n,int first,int*count);

static const char *g_model="/mnt/Qwen3.6-35B-A3B-MXFP4_MOE.gguf";
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc,char**argv){
    const char*file=NULL;
    int window=512, maxw=0, kvq=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-m")&&i+1<argc) g_model=argv[++i];
        else if(!strcmp(argv[i],"-f")&&i+1<argc) file=argv[++i];
        else if(!strcmp(argv[i],"--window")&&i+1<argc) window=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-windows")&&i+1<argc) maxw=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--kv-quant")) kvq=1;
        else { fprintf(stderr,"usage: %s -f corpus.txt [-m gguf] [--window n] "
                       "[--max-windows n] [--kv-quant]\n",argv[0]); return 1; }
    }
    if(!file){ fprintf(stderr,"need -f corpus\n"); return 1; }
    if(window<2||window>2048){ fprintf(stderr,"--window must be 2..2048\n"); return 1; }

    FILE*fp=fopen(file,"rb");
    if(!fp){ perror(file); return 1; }
    fseek(fp,0,SEEK_END); long fsz=ftell(fp); fseek(fp,0,SEEK_SET);
    char*text=(char*)malloc((size_t)fsz+1);
    if(fread(text,1,(size_t)fsz,fp)!=(size_t)fsz){ fprintf(stderr,"read failed\n"); return 1; }
    fclose(fp); text[fsz]=0;

    q36_model m; char err[256];
    if(q36_model_open(&m,g_model,err,sizeof err)){ fprintf(stderr,"open: %s\n",err); return 1; }
    q36_vocab vocab; q36_vocab_init(&vocab,&m.gguf);
    q36_encode_init(&vocab,&m.gguf);

    fprintf(stderr,"tokenizing %ld bytes...\n",fsz);
    int cap=(int)fsz+16;
    int*toks=(int*)malloc((size_t)cap*sizeof(int));
    int nt=q36_encode(&vocab,text,toks,cap);
    free(text);
    if(nt<=0){ fprintf(stderr,"tokenization failed\n"); return 1; }
    int nw=nt/window;
    if(maxw&&nw>maxw) nw=maxw;
    fprintf(stderr,"%d tokens -> %d windows of %d (scoring second halves)\n",nt,nw,window);
    if(!nw){ fprintf(stderr,"corpus smaller than one window\n"); return 1; }

    double t0=now();
    q36_engine*e=q36_engine_create(&m,window+64);
    if(kvq) q36_engine_set_kvq(e,1);
    fprintf(stderr,"engine ready in %.1fs | kv %s\n",now()-t0,kvq?"q8/mxfp4":"fp16");

    double nll=0; long count=0;
    t0=now();
    for(int w=0;w<nw;w++){
        int c=0;
        nll+=q36_engine_nll(e,toks+(size_t)w*window,window,window/2,&c);
        count+=c;
        double ppl=exp(nll/(double)count);
        printf("[%d/%d] ppl %.4f\r",w+1,nw,ppl);
        if((w+1)%16==0||w+1==nw) printf("\n");
        fflush(stdout);
    }
    double dt=now()-t0;
    printf("\nfinal: ppl %.4f over %ld tokens (%d windows of %d, second halves)\n",
           exp(nll/(double)count),count,nw,window);
    printf("%.1fs total, %.2fs/window\n",dt,dt/nw);
    q36_vocab_free(&vocab); q36_model_close(&m);
    return 0;
}

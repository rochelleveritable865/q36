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

/* q36_server: OpenAI-compatible HTTP server, zero dependencies (POSIX sockets,
 * hand-rolled HTTP/1.1 + SSE + minimal JSON).
 *
 *   ./q36_server [--port 8080] [--ctx 32768] [--kv-quant] [-m gguf]
 *
 * Endpoints:
 *   POST /v1/chat/completions   OpenAI chat (messages, tools); stream|json
 *   POST /v1/completions        raw prompt, no template
 *   POST /v1/messages           Anthropic Messages (system, tools); +count_tokens
 *   GET  /v1/models, /health
 *
 * Tool calling follows the GGUF's embedded chat template (Qwen3-Coder XML
 * style): tools render as a "# Tools" system section, the model replies
 *   <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
 * which is parsed back into OpenAI tool_calls / Anthropic tool_use blocks;
 * argument types are coerced via the declared JSON schema (fallback: values
 * that parse as JSON stay raw, everything else becomes a string).  Tool
 * results (role:"tool" / tool_result blocks) render as <tool_response>
 * wraps grouped into user turns, exactly as the template does.
 *
 * Performance model: the engine is single-stream; requests are served
 * sequentially (FIFO via the listen backlog).  The win for chat is the
 * PREFIX CACHE, two tiers:
 *   1. live state: when the new prompt strictly extends the token ids
 *      currently in the KV/SSM state, prefill only the delta;
 *   2. state checkpoints: end-of-prompt
 *      snapshots of the hybrid KV+SSM state, keyed by prompt tokens, held
 *      in DRAM (LRU) with an optional disk tier.  Restores the longest
 *      stored prefix when the live state diverges -- the normal agent-loop
 *      case, where clients resend a reconstructed history (think stripped,
 *      tool calls re-rendered) that never matches generated text.
 * The hybrid SSM state cannot rewind, so anything short of a strict
 * extension or a checkpoint boundary resets and prefills from scratch.
 *
 * Sampling: prefill returns the ARGMAX token, so for temp>0 the last
 * prompt token goes through q36_engine_step instead -- the first generated
 * token is then properly sampled.  Defaults follow the GGUF-recommended
 * temp 1.0 / top-k 20 / top-p 0.95; temperature:0 selects greedy. */
#define _GNU_SOURCE
#include "q36_model.h"
#include "tokenizer.h"
#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct q36_engine q36_engine;
q36_engine* q36_engine_create(q36_model*m,int ctx);
int  q36_engine_step(q36_engine*e,int token,int pos);
int  q36_engine_prefill(q36_engine*e,const int*toks,int n);
int  q36_engine_prefill_from(q36_engine*e,const int*toks,int n,int pos0);
void q36_engine_reset(q36_engine*e);
void q36_engine_set_kvq(q36_engine*e,int on);
void q36_engine_set_sampler(q36_engine*e,float temp,int topk,float topp,unsigned long long seed);
/* MTP self-speculative decode (q36_engine.cu); step_mtp falls back to a
 * plain step near the ctx edge or when the sampler is non-greedy */
int  q36_engine_step_mtp(q36_engine*e,int token,int pos,int out[4]);
int  q36_engine_has_mtp(q36_engine*e);
void q36_engine_set_mtp_k(q36_engine*e,int k);
void q36_engine_mtp_stats(q36_engine*e,long*cycles,long*accepts);
/* hybrid state checkpoint primitives (q36_engine.cu) */
size_t q36_engine_state_ssm_bytes(q36_engine*e);
size_t q36_engine_state_kv_bytes(q36_engine*e,int npos);
int  q36_engine_ssm_save(q36_engine*e,void*dst);
int  q36_engine_ssm_load(q36_engine*e,const void*src);
int  q36_engine_kv_range(q36_engine*e,int p0,int p1,void*buf,int a,int b,int save);

#define MODEL_ID "qwen3.6-35b-a3b"
#define MAX_STOP 4

static const char *g_model_path="/mnt/Qwen3.6-35B-A3B-MXFP4_MOE.gguf";
static q36_model g_m; static q36_vocab g_vocab; static q36_engine*g_e;
static int g_ctx=32768, g_eos=-1, g_imend=-1;
static int *g_cache_ids, g_cache_n=0;   /* token ids currently in KV/SSM state */
static int *g_toks;
static int g_autopurge=1;               /* DEFAULT ON: drop oldest turns / left-
                                           truncate to fit ctx (--no-auto-purge
                                           restores hard 400s on overflow) */
static int g_hide_think=0;              /* --hide-think: never emit the leading
                                           <think> region (tokens still generate;
                                           Qwen3.6 reasons before answering) */
static int g_mtp=0;                     /* --mtp [K]: self-speculative decode for
                                           greedy requests (needs the MTP GGUF's
                                           nextn module; sampled requests fall
                                           back to the plain step path) */
static float g_temp=1.0f;               /* --temp: default when the client omits
                                           temperature.  Agent harnesses (Claude
                                           Code) rarely send one; the GGUF's 1.0
                                           is tuned for chat -- lower (or 0 =
                                           greedy, deterministic) keeps long tool
                                           loops on rails */
static unsigned long long g_reqid=0;
/* lifetime token counters (per-request log line, reported in millions):
 * pf/gen = tokens the engine actually computed, hit = skipped via cache */
static unsigned long long g_tok_pf=0, g_tok_gen=0, g_tok_hit=0;
/* tool-call format reliability: <tool_call> blocks the model emitted vs
 * blocks that failed to parse (harness-friction metric for agent evals) */
static unsigned long long g_tc_ok=0, g_tc_bad=0;
static volatile sig_atomic_t g_stop=0;   /* SIGINT/SIGTERM: finish request, spill, exit */
static void on_term(int sig){ (void)sig; g_stop=1; }

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---------------- growable buffer ---------------- */
typedef struct { char*p; size_t n,cap; } sbuf;
static void sb_put(sbuf*b,const char*s,size_t n){
    if(b->n+n+1>b->cap){ size_t c=b->cap?b->cap:4096; while(c<b->n+n+1)c*=2;
        b->p=(char*)realloc(b->p,c); b->cap=c; }
    memcpy(b->p+b->n,s,n); b->n+=n; b->p[b->n]=0;
}
static void sb_str(sbuf*b,const char*s){ sb_put(b,s,strlen(s)); }
static void sb_fmt(sbuf*b,const char*fmt,...){
    char tmp[512]; va_list ap; va_start(ap,fmt);
    int k=vsnprintf(tmp,sizeof tmp,fmt,ap); va_end(ap);
    if(k>0) sb_put(b,tmp,(size_t)k<sizeof tmp?(size_t)k:sizeof tmp-1);
}
/* JSON string-escape s into b (UTF-8 passes through raw) */
static void sb_jesc(sbuf*b,const char*s,size_t n){
    for(size_t i=0;i<n;i++){ unsigned char c=(unsigned char)s[i];
        switch(c){
        case '"':  sb_put(b,"\\\"",2); break;
        case '\\': sb_put(b,"\\\\",2); break;
        case '\n': sb_put(b,"\\n",2);  break;
        case '\r': sb_put(b,"\\r",2);  break;
        case '\t': sb_put(b,"\\t",2);  break;
        default:
            if(c<0x20){ char u[8]; snprintf(u,sizeof u,"\\u%04x",c); sb_put(b,u,6); }
            else sb_put(b,(const char*)&s[i],1);
        }
    }
}

/* ---------------- minimal JSON parser ---------------- */
typedef struct { const char*p,*end; } jp;
static int jch(jp*j){ while(j->p<j->end){ char c=*j->p;
    if(c==' '||c=='\t'||c=='\n'||c=='\r'){j->p++;continue;} return (unsigned char)c; } return -1; }
static void utf8_put(sbuf*b,unsigned cp){
    char u[4];
    if(cp<0x80){ u[0]=(char)cp; sb_put(b,u,1); }
    else if(cp<0x800){ u[0]=(char)(0xC0|cp>>6); u[1]=(char)(0x80|(cp&63)); sb_put(b,u,2); }
    else if(cp<0x10000){ u[0]=(char)(0xE0|cp>>12); u[1]=(char)(0x80|((cp>>6)&63)); u[2]=(char)(0x80|(cp&63)); sb_put(b,u,3); }
    else { u[0]=(char)(0xF0|cp>>18); u[1]=(char)(0x80|((cp>>12)&63)); u[2]=(char)(0x80|((cp>>6)&63)); u[3]=(char)(0x80|(cp&63)); sb_put(b,u,4); }
}
static int jhex4(jp*j,unsigned*out){
    if(j->end-j->p<4) return -1; unsigned v=0;
    for(int i=0;i<4;i++){ char c=j->p[i]; v<<=4;
        if(c>='0'&&c<='9')v|=(unsigned)(c-'0');
        else if(c>='a'&&c<='f')v|=(unsigned)(c-'a'+10);
        else if(c>='A'&&c<='F')v|=(unsigned)(c-'A'+10);
        else return -1; }
    j->p+=4; *out=v; return 0;
}
/* parse a JSON string -> malloc'd UTF-8 (NULL on error) */
static char*jstring(jp*j){
    if(jch(j)!='"') return NULL;
    j->p++;
    sbuf b={0,0,0}; sb_put(&b,"",0);
    while(j->p<j->end){
        unsigned char c=(unsigned char)*j->p++;
        if(c=='"') return b.p;
        if(c=='\\'){
            if(j->p>=j->end) break;
            char e=*j->p++;
            switch(e){
            case '"': sb_put(&b,"\"",1); break;
            case '\\':sb_put(&b,"\\",1); break;
            case '/': sb_put(&b,"/",1);  break;
            case 'b': sb_put(&b,"\b",1); break;
            case 'f': sb_put(&b,"\f",1); break;
            case 'n': sb_put(&b,"\n",1); break;
            case 'r': sb_put(&b,"\r",1); break;
            case 't': sb_put(&b,"\t",1); break;
            case 'u': { unsigned cp;
                if(jhex4(j,&cp)) goto bad;
                if(cp>=0xD800&&cp<0xDC00 && j->end-j->p>=6 && j->p[0]=='\\'&&j->p[1]=='u'){
                    jp save=*j; j->p+=2; unsigned lo;
                    if(jhex4(j,&lo)==0 && lo>=0xDC00 && lo<0xE000)
                        cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00);
                    else *j=save;
                }
                utf8_put(&b,cp); break; }
            default: goto bad;
            }
        } else sb_put(&b,(const char*)&c,1);
    }
bad:
    free(b.p); return NULL;
}
static int jskip(jp*j);
static int jskip_scalar(jp*j){
    while(j->p<j->end){ char c=*j->p;
        if(c==','||c=='}'||c==']'||c==' '||c=='\t'||c=='\n'||c=='\r') return 0;
        j->p++; }
    return 0;
}
static int jskip(jp*j){
    int c=jch(j); if(c<0) return -1;
    if(c=='"'){ char*s=jstring(j); if(!s)return -1; free(s); return 0; }
    if(c=='{'||c=='['){ char close=(c=='{')?'}':']';
        j->p++;
        if(jch(j)==close){ j->p++; return 0; }
        for(;;){
            if(c=='{'){ char*k=jstring(j); if(!k)return -1; free(k);
                if(jch(j)!=':')return -1; j->p++; }
            if(jskip(j)) return -1;
            int d=jch(j);
            if(d==','){ j->p++; continue; }
            if(d==close){ j->p++; return 0; }
            return -1;
        }
    }
    return jskip_scalar(j);
}
static double jnumber(jp*j,int*ok){
    jch(j); char*endp; double v=strtod(j->p,&endp);
    if(endp==j->p){ *ok=0; return 0; }
    j->p=endp; *ok=1; return v;
}

/* ---------------- request ---------------- */
/* protocol: 0 = /v1/completions (raw), 1 = OpenAI chat, 2 = Anthropic
 * Messages (/v1/messages).  2 parses like 1 plus top-level "system" and
 * "stop_sequences", and answers in the Anthropic response/SSE shapes. */
enum { PROTO_CMPL=0, PROTO_CHAT=1, PROTO_ANTH=2 };

#define MAX_MSG 512
#define MAX_TOOLS 64
typedef struct {
    sbuf prompt;                        /* raw prompt (/v1/completions)     */
    sbuf sys;                           /* top-level system (anthropic)     */
    char*mrole[MAX_MSG];                /* parsed messages (/v1/chat/...)   */
    char*mtext[MAX_MSG];
    int nmsg;
    /* tool declarations: tjson = the {"type":"function",...} line for the
     * <tools> system section (raw request spans re-emitted verbatim);
     * tool_name + parameter-schema span drive output type coercion */
    sbuf tjson[MAX_TOOLS]; char*tool_name[MAX_TOOLS];
    const char*tsch0[MAX_TOOLS],*tsch1[MAX_TOOLS];
    int ntools;
    int stream, max_tokens, topk, has_temp;
    float temp, topp;
    long long seed; int has_seed;
    char*stop[MAX_STOP]; int nstop;
} req_t;
static void req_free(req_t*r){
    free(r->prompt.p); free(r->sys.p);
    for(int i=0;i<r->nmsg;i++){ free(r->mrole[i]); free(r->mtext[i]); }
    for(int i=0;i<r->ntools;i++){ free(r->tjson[i].p); free(r->tool_name[i]); }
    for(int i=0;i<r->nstop;i++) free(r->stop[i]);
}

/* capture the RAW JSON span of the next value (whitespace-trimmed) */
static int jspan(jp*j,const char**s0,const char**s1){
    jch(j); *s0=j->p;
    if(jskip(j)) return -1;
    *s1=j->p;
    return 0;
}

/* ============ tool calling (format from the GGUF's embedded template:
 * Qwen3-Coder XML style -- <function=NAME><parameter=K>V</parameter>...) === */

/* render one tool call as the template's XML from an arguments value
 * (JSON object; string params emitted raw, non-strings as JSON) */
static int xml_from_args(sbuf*out,const char*name,jp*j,int first_and_empty){
    sb_str(out,first_and_empty?"<tool_call>\n<function=":"\n<tool_call>\n<function=");
    sb_str(out,name); sb_str(out,">\n");
    int c=jch(j);
    if(c=='{'){
        j->p++;
        if(jch(j)=='}') j->p++;
        else for(;;){
            char*k=jstring(j); if(!k)return -1;
            if(jch(j)!=':'){free(k);return -1;} j->p++;
            sb_str(out,"<parameter="); sb_str(out,k); sb_str(out,">\n"); free(k);
            int d=jch(j);
            if(d=='"'){ char*s=jstring(j); if(!s)return -1; sb_str(out,s); free(s); }
            else { const char*a,*b; if(jspan(j,&a,&b))return -1; sb_put(out,a,(size_t)(b-a)); }
            sb_str(out,"\n</parameter>\n");
            d=jch(j);
            if(d==','){j->p++;continue;}
            if(d=='}'){j->p++;break;}
            return -1;
        }
    } else if(jskip(j)) return -1;   /* non-object arguments: no parameters */
    sb_str(out,"</function>\n</tool_call>");
    return 0;
}

/* one element of a request "tools" array: OpenAI {"type","function":{...}}
 * (re-emitted verbatim) or Anthropic {name, description, input_schema}
 * (rewrapped).  Records name + parameter-schema span for output coercion. */
static int parse_tool_el(const char*s,const char*e,req_t*r){
    jp j={s,e};
    const char*fn0=NULL,*fn1=NULL,*sc0=NULL,*sc1=NULL,*ds0=NULL,*ds1=NULL;
    char*name=NULL;
    if(jch(&j)!='{')return -1;
    j.p++;
    if(jch(&j)!='}') for(;;){
        char*k=jstring(&j); if(!k){free(name);return -1;}
        if(jch(&j)!=':'){free(k);free(name);return -1;} j.p++;
        int bad=0;
        if(!strcmp(k,"function")) bad=jspan(&j,&fn0,&fn1);
        else if(!strcmp(k,"name")){ free(name); name=jstring(&j); bad=!name; }
        else if(!strcmp(k,"input_schema")) bad=jspan(&j,&sc0,&sc1);
        else if(!strcmp(k,"description")) bad=jspan(&j,&ds0,&ds1);
        else bad=jskip(&j);
        free(k);
        if(bad){free(name);return -1;}
        int d=jch(&j);
        if(d==','){j.p++;continue;}
        if(d=='}'){j.p++;break;}
        free(name);return -1;
    }
    if(fn0){                          /* openai shape: dig into "function" */
        jp f={fn0,fn1};
        if(jch(&f)=='{'){
            f.p++;
            if(jch(&f)!='}') for(;;){
                char*k=jstring(&f); if(!k){free(name);return -1;}
                if(jch(&f)!=':'){free(k);free(name);return -1;} f.p++;
                int bad=0;
                if(!strcmp(k,"name")){ free(name); name=jstring(&f); bad=!name; }
                else if(!strcmp(k,"parameters")) bad=jspan(&f,&sc0,&sc1);
                else bad=jskip(&f);
                free(k);
                if(bad){free(name);return -1;}
                int d=jch(&f);
                if(d==','){f.p++;continue;}
                if(d=='}'){f.p++;break;}
                free(name);return -1;
            }
        }
    }
    if(!name) return -1;
    if(r->ntools>=MAX_TOOLS){ free(name); return 0; }
    int t=r->ntools;
    sbuf*tj=&r->tjson[t]; memset(tj,0,sizeof*tj); sb_put(tj,"",0);
    if(fn0) sb_put(tj,s,(size_t)(e-s));           /* verbatim */
    else {
        sb_str(tj,"{\"type\": \"function\", \"function\": {\"name\": \"");
        sb_jesc(tj,name,strlen(name));
        sb_str(tj,"\"");
        if(ds0){ sb_str(tj,", \"description\": "); sb_put(tj,ds0,(size_t)(ds1-ds0)); }
        sb_str(tj,", \"parameters\": ");
        if(sc0) sb_put(tj,sc0,(size_t)(sc1-sc0)); else sb_str(tj,"{}");
        sb_str(tj,"}}");
    }
    r->tool_name[t]=name; r->tsch0[t]=sc0; r->tsch1[t]=sc1;
    r->ntools++;
    return 0;
}

/* declared JSON-schema type of tool FNAME's parameter PNAME: one of
 * 's'tring, 'j'son-typed (number/bool/object/array), 0 unknown */
static char schema_ptype(req_t*r,const char*fname,size_t fn,const char*pname,size_t pn){
    for(int t=0;t<r->ntools;t++){
        if(strlen(r->tool_name[t])!=fn||strncmp(r->tool_name[t],fname,fn)) continue;
        const char*s=r->tsch0[t],*e=r->tsch1[t];
        if(!s) return 0;
        char key[128];
        if(pn+2>=sizeof key) return 0;
        snprintf(key,sizeof key,"\"%.*s\"",(int)pn,pname);
        const char*p=s;
        while((p=(const char*)memmem(p,(size_t)(e-p),key,pn+2))){
            const char*q=p+pn+2;
            while(q<e&&(*q==' '||*q=='\t'||*q=='\n'||*q=='\r'))q++;
            if(q<e&&*q==':'){                     /* it's a property key */
                const char*ty=(const char*)memmem(q,(size_t)(e-q),"\"type\"",6);
                if(!ty) return 0;
                jp j={ty+6,e};
                if(jch(&j)==':'){ j.p++;
                    char*v=jstring(&j);
                    if(v){ char c=strcmp(v,"string")?'j':'s'; free(v); return c; }
                }
                return 0;
            }
            p+=pn+2;
        }
        return 0;
    }
    return 0;
}

/* does [v,v+n) parse as one complete JSON value? */
static int val_is_json(const char*v,size_t n){
    jp j={v,v+n};
    if(jskip(&j)) return 0;
    return jch(&j)<0;
}

/* parsed model output tool call: name span (into gen text) + built args */
typedef struct { const char*name; size_t nlen; sbuf args; } otc_t;

/* parse "<tool_call>\n<function=NAME>\n<parameter=P>\nV\n</parameter>...
 * </function>\n</tool_call>" blocks out of the generated text; arguments
 * are typed via the declared schema (fallback: JSON-parseable -> raw). */
static int parse_out_tools(req_t*r,const char*text,size_t from,otc_t*tc,int max){
    int n=0;
    const char*p=text+from;
    while(n<max&&(p=strstr(p,"<tool_call>"))){
        const char*f=strstr(p,"<function=");
        const char*blk_end=strstr(p+11,"</tool_call>");
        if(!f||!blk_end||f>blk_end){ p+=11; continue; }
        const char*nm=f+10, *ne=nm;
        while(*ne&&*ne!='>'&&*ne!='\n')ne++;
        if(*ne!='>'){ p=blk_end+12; continue; }
        otc_t*c=&tc[n];
        c->name=nm; c->nlen=(size_t)(ne-nm);
        memset(&c->args,0,sizeof c->args); sb_put(&c->args,"",0);
        sb_str(&c->args,"{");
        int np=0, ok=1;
        const char*q=ne+1;
        while(ok){
            const char*pa=strstr(q,"<parameter=");
            const char*fe=strstr(q,"</function>");
            if(!fe||fe>blk_end){ ok=0; break; }
            if(!pa||pa>fe) break;                /* no more parameters */
            const char*pn=pa+11,*pe=pn;
            while(*pe&&*pe!='>'&&*pe!='\n')pe++;
            if(*pe!='>'){ ok=0; break; }
            const char*v=pe+1; if(*v=='\n')v++;
            const char*ve=strstr(v,"</parameter>");
            if(!ve||ve>fe){ ok=0; break; }
            size_t vn=(size_t)(ve-v);
            while(vn&&v[vn-1]=='\n')vn--;        /* template: value\n</parameter> */
            if(np++) sb_str(&c->args,", ");
            sb_str(&c->args,"\"");
            sb_jesc(&c->args,pn,(size_t)(pe-pn));
            sb_str(&c->args,"\": ");
            char ty=schema_ptype(r,c->name,c->nlen,pn,(size_t)(pe-pn));
            int raw=(ty=='j'||(ty==0&&vn&&(strchr("{[-0123456789",v[0])||
                     (vn==4&&!strncmp(v,"true",4))||(vn==5&&!strncmp(v,"false",5))||
                     (vn==4&&!strncmp(v,"null",4)))))&&val_is_json(v,vn);
            if(raw) sb_put(&c->args,v,vn);
            else { sb_str(&c->args,"\""); sb_jesc(&c->args,v,vn); sb_str(&c->args,"\""); }
            q=ve+12;
        }
        if(ok){ sb_str(&c->args,"}"); n++; }
        else free(c->args.p);
        p=blk_end+12;
    }
    return n;
}
/* Qwen chat template from messages[first..], always keeping a leading
 * system message; `first` rises when --auto-purge drops old turns.
 * Tool text (the # Tools section, <function=...> call syntax and
 * <tool_response> grouping) reproduces the GGUF's embedded jinja template
 * verbatim -- the model is trained on exactly these strings. */
static void build_prompt(req_t*r,int first,sbuf*out){
    out->n=0; if(out->p) out->p[0]=0;
    int has_sys=r->nmsg>0&&(!strcmp(r->mrole[0],"system")||!strcmp(r->mrole[0],"developer"));
    if(r->ntools){
        sb_str(out,"<|im_start|>system\n"
            "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for(int t=0;t<r->ntools;t++){
            sb_str(out,"\n");
            sb_put(out,r->tjson[t].p,r->tjson[t].n);
        }
        sb_str(out,"\n</tools>"
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
            "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block "
            "must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the "
            "function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current "
            "knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if(has_sys){ sb_str(out,"\n\n"); sb_str(out,r->mtext[0]); }
        sb_str(out,"<|im_end|>\n");
    } else if(has_sys){
        sb_str(out,"<|im_start|>system\n");
        sb_str(out,r->mtext[0]);
        sb_str(out,"<|im_end|>\n");
    }
    const char*prev=NULL;
    for(int i=has_sys?1:0;i<r->nmsg;i++){
        if(i<first) continue;
        const char*role=r->mrole[i];
        if(!strcmp(role,"tool")){        /* grouped into one user turn */
            if(!prev||strcmp(prev,"tool")) sb_str(out,"<|im_start|>user");
            sb_str(out,"\n<tool_response>\n");
            sb_str(out,r->mtext[i]);
            sb_str(out,"\n</tool_response>");
            if(i+1>=r->nmsg||strcmp(r->mrole[i+1],"tool")) sb_str(out,"<|im_end|>\n");
        } else {
            sb_fmt(out,"<|im_start|>%s\n",role);
            sb_str(out,r->mtext[i]);
            sb_str(out,"<|im_end|>\n");
        }
        prev=role;
    }
    sb_str(out,"<|im_start|>assistant\n");
}

/* content: string, or array of typed parts.  anth=1 additionally renders
 * Anthropic tool_use blocks as template XML tool calls and tool_result
 * blocks as <tool_response> wraps (results arrive inside user messages). */
static int parse_content_ex(jp*j,sbuf*out,int anth){
    int c=jch(j);
    if(c=='"'){ char*s=jstring(j); if(!s)return -1; sb_str(out,s); free(s); return 0; }
    if(c=='['){
        j->p++;
        if(jch(j)==']'){ j->p++; return 0; }
        for(;;){
            if(jch(j)!='{') return -1;
            j->p++;
            char*type=NULL,*name=NULL;
            sbuf text={0,0,0},res={0,0,0};
            const char*in0=NULL,*in1=NULL;
            int bad=0;
            for(;;){
                char*k=jstring(j); if(!k){bad=1;break;}
                if(jch(j)!=':'){free(k);bad=1;break;} j->p++;
                if(!strcmp(k,"text")){ char*s=jstring(j);
                    if(!s)bad=1; else { sb_str(&text,s); free(s); } }
                else if(anth&&!strcmp(k,"type")){ free(type); type=jstring(j); if(!type)bad=1; }
                else if(anth&&!strcmp(k,"name")){ free(name); name=jstring(j); if(!name)bad=1; }
                else if(anth&&!strcmp(k,"input")){ if(jspan(j,&in0,&in1))bad=1; }
                else if(anth&&!strcmp(k,"content")){ if(parse_content_ex(j,&res,0))bad=1; }
                else if(jskip(j))bad=1;
                free(k);
                if(bad)break;
                int d=jch(j);
                if(d==','){j->p++;continue;}
                if(d=='}'){j->p++;break;}
                bad=1;break;
            }
            if(!bad){
                if(type&&!strcmp(type,"tool_use")&&name){
                    jp a={in0?in0:"{}",in0?in1:NULL};
                    if(!in0){ a.p="{}"; a.end=a.p+2; }
                    if(xml_from_args(out,name,&a,out->n==0))bad=1;
                } else if(type&&!strcmp(type,"tool_result")){
                    if(out->n) sb_str(out,"\n");
                    sb_str(out,"<tool_response>\n");
                    if(res.p) sb_put(out,res.p,res.n);
                    sb_str(out,"\n</tool_response>");
                } else if(text.p)
                    sb_put(out,text.p,text.n);
            }
            free(type);free(name);free(text.p);free(res.p);
            if(bad)return -1;
            int d=jch(j);
            if(d==','){j->p++;continue;}
            if(d==']'){j->p++;return 0;}
            return -1;
        }
    }
    if(c=='n'){ return jskip(j); }   /* null content (tool turns) */
    return -1;
}
static int parse_content(jp*j,sbuf*out){ return parse_content_ex(j,out,0); }

/* OpenAI assistant "tool_calls": [{id?,type?,function:{name,arguments}}]
 * (arguments = JSON-encoded string, or an inline object from lax clients);
 * renders the template XML into tcxml. */
static int parse_tool_calls(jp*j,sbuf*tcxml){
    if(jch(j)!='[')return -1;
    j->p++;
    if(jch(j)==']'){j->p++;return 0;}
    for(;;){
        if(jch(j)!='{')return -1;
        j->p++;
        char*name=NULL,*argstr=NULL;
        const char*a0=NULL,*a1=NULL,*f0=NULL,*f1=NULL;
        int bad=0;
        if(jch(j)!='}') for(;;){
            char*k=jstring(j); if(!k){bad=1;break;}
            if(jch(j)!=':'){free(k);bad=1;break;} j->p++;
            if(!strcmp(k,"function")) bad=jspan(j,&f0,&f1);
            else if(!strcmp(k,"name")){ free(name); name=jstring(j); bad=!name; }
            else if(!strcmp(k,"arguments")){
                if(jch(j)=='"'){ free(argstr); argstr=jstring(j); bad=!argstr; }
                else bad=jspan(j,&a0,&a1);
            }
            else bad=jskip(j);
            free(k);
            if(bad)break;
            int d=jch(j);
            if(d==','){j->p++;continue;}
            if(d=='}'){j->p++;break;}
            bad=1;break;
        } else j->p++;
        if(!bad&&f0){
            jp f={f0,f1};
            if(jch(&f)=='{'){
                f.p++;
                if(jch(&f)!='}') for(;;){
                    char*k=jstring(&f); if(!k){bad=1;break;}
                    if(jch(&f)!=':'){free(k);bad=1;break;} f.p++;
                    if(!strcmp(k,"name")){ free(name); name=jstring(&f); bad=!name; }
                    else if(!strcmp(k,"arguments")){
                        if(jch(&f)=='"'){ free(argstr); argstr=jstring(&f); bad=!argstr; }
                        else bad=jspan(&f,&a0,&a1);
                    }
                    else bad=jskip(&f);
                    free(k);
                    if(bad)break;
                    int d=jch(&f);
                    if(d==','){f.p++;continue;}
                    if(d=='}'){f.p++;break;}
                    bad=1;break;
                }
            }
        }
        if(!bad&&name){
            jp a;
            if(argstr){ a.p=argstr; a.end=argstr+strlen(argstr); }
            else if(a0){ a.p=a0; a.end=a1; }
            else { a.p="{}"; a.end=a.p+2; }
            bad=xml_from_args(tcxml,name,&a,tcxml->n==0)!=0;
        }
        free(name); free(argstr);
        if(bad)return -1;
        int d=jch(j);
        if(d==','){j->p++;continue;}
        if(d==']'){j->p++;return 0;}
        return -1;
    }
}
static int parse_stop(jp*j,req_t*r){
    int c=jch(j);
    if(c=='"'){ char*s=jstring(j); if(!s)return -1; r->stop[r->nstop++]=s; return 0; }
    if(c=='['){ j->p++;
        if(jch(j)==']'){j->p++;return 0;}
        for(;;){ char*s=jstring(j); if(!s)return -1;
            if(r->nstop<MAX_STOP) r->stop[r->nstop++]=s; else free(s);
            int d=jch(j);
            if(d==','){j->p++;continue;}
            if(d==']'){j->p++;return 0;}
            return -1; } }
    return jskip(j);
}
/* proto>=1: build the Qwen template from messages[]; proto=0: raw "prompt" */
static int parse_request(const char*body,size_t blen,req_t*r,int proto){
    jp j={body,body+blen};
    int ok, saw_input=0;
    if(jch(&j)!='{') return -1;
    j.p++;
    if(jch(&j)=='}') return -1;
    for(;;){
        char*key=jstring(&j); if(!key)return -1;
        if(jch(&j)!=':'){free(key);return -1;} j.p++;
        if(proto!=PROTO_CMPL && !strcmp(key,"messages")){
            free(key);
            if(jch(&j)!='[')return -1;
            j.p++; saw_input=1;
            if(jch(&j)==']'){j.p++;}
            else for(;;){
                if(jch(&j)!='{')return -1;
                j.p++;
                char*role=NULL; sbuf content={0,0,0}; sb_put(&content,"",0);
                sbuf tcxml={0,0,0};
                int have_content=0;
                for(;;){
                    char*k2=jstring(&j); if(!k2){free(role);free(content.p);free(tcxml.p);return -1;}
                    if(jch(&j)!=':'){free(k2);free(role);free(content.p);free(tcxml.p);return -1;} j.p++;
                    if(!strcmp(k2,"role")){ free(role); role=jstring(&j);
                        if(!role){free(k2);free(content.p);free(tcxml.p);return -1;} }
                    else if(!strcmp(k2,"content")){
                        if(parse_content_ex(&j,&content,proto==PROTO_ANTH)){free(k2);free(role);free(content.p);free(tcxml.p);return -1;}
                        have_content=1; }
                    else if(!strcmp(k2,"tool_calls")){
                        if(parse_tool_calls(&j,&tcxml)){free(k2);free(role);free(content.p);free(tcxml.p);return -1;}
                        have_content=1; }
                    else if(jskip(&j)){free(k2);free(role);free(content.p);free(tcxml.p);return -1;}
                    free(k2);
                    int d=jch(&j);
                    if(d==','){j.p++;continue;}
                    if(d=='}'){j.p++;break;}
                    free(role);free(content.p);free(tcxml.p);return -1;
                }
                if(tcxml.n){          /* template: content\n\n<tool_call>... */
                    if(content.n) sb_str(&content,"\n\n");
                    sb_put(&content,tcxml.p,tcxml.n);
                }
                free(tcxml.p);
                if(role&&have_content&&r->nmsg<MAX_MSG){
                    r->mrole[r->nmsg]=role; r->mtext[r->nmsg]=content.p; r->nmsg++;
                } else { free(role); free(content.p); }
                int d=jch(&j);
                if(d==','){j.p++;continue;}
                if(d==']'){j.p++;break;}
                return -1;
            }
        }
        else if(proto==PROTO_CMPL && !strcmp(key,"prompt")){
            free(key);
            char*s=jstring(&j); if(!s)return -1;
            sb_str(&r->prompt,s); free(s); saw_input=1;
        }
        else if(proto==PROTO_ANTH && !strcmp(key,"system")){
            free(key);                       /* string or [{type:text,...}] */
            if(parse_content(&j,&r->sys))return -1;
        }
        else if(proto==PROTO_ANTH && !strcmp(key,"stop_sequences")){ free(key);
            if(parse_stop(&j,r))return -1; }
        else if(proto!=PROTO_CMPL && !strcmp(key,"tools")){
            free(key);
            int c=jch(&j);
            if(c=='n'){ if(jskip(&j))return -1; }
            else{
                if(c!='[')return -1;
                j.p++;
                if(jch(&j)==']') j.p++;
                else for(;;){
                    const char*t0,*t1;
                    if(jspan(&j,&t0,&t1))return -1;
                    if(parse_tool_el(t0,t1,r))return -1;
                    int d=jch(&j);
                    if(d==','){j.p++;continue;}
                    if(d==']'){j.p++;break;}
                    return -1;
                }
            }
        }
        else if(!strcmp(key,"stream")){ free(key);
            r->stream=(jch(&j)=='t'); if(jskip(&j))return -1; }
        else if(!strcmp(key,"temperature")){ free(key);
            r->temp=(float)jnumber(&j,&ok); if(!ok)return -1; r->has_temp=1; }
        else if(!strcmp(key,"top_p")){ free(key);
            r->topp=(float)jnumber(&j,&ok); if(!ok)return -1; }
        else if(!strcmp(key,"top_k")){ free(key);
            r->topk=(int)jnumber(&j,&ok); if(!ok)return -1; }
        else if(!strcmp(key,"max_tokens")||!strcmp(key,"max_completion_tokens")){ free(key);
            r->max_tokens=(int)jnumber(&j,&ok); if(!ok)return -1; }
        else if(!strcmp(key,"seed")){ free(key);
            r->seed=(long long)jnumber(&j,&ok); if(!ok)return -1; r->has_seed=1; }
        else if(!strcmp(key,"stop")){ free(key);
            if(parse_stop(&j,r))return -1; }
        else { free(key); if(jskip(&j))return -1; }
        int d=jch(&j);
        if(d==','){j.p++;continue;}
        if(d=='}'){j.p++;break;}
        return -1;
    }
    return saw_input?0:-1;
}

/* ---------------- HTTP plumbing ---------------- */
static int send_all(int fd,const char*p,size_t n){
    while(n){ ssize_t k=send(fd,p,n,MSG_NOSIGNAL);
        if(k<=0){ if(k<0&&errno==EINTR)continue; return -1; }
        p+=k; n-=(size_t)k; }
    return 0;
}
static void send_json(int fd,int code,const char*status,const char*body){
    char hdr[256];
    int hn=snprintf(hdr,sizeof hdr,
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        code,status,strlen(body));
    if(send_all(fd,hdr,(size_t)hn)==0) send_all(fd,body,strlen(body));
}
static void send_err(int fd,int code,const char*msg){
    sbuf b={0,0,0};
    sb_str(&b,"{\"error\":{\"message\":\"");
    sb_jesc(&b,msg,strlen(msg));
    sb_str(&b,"\",\"type\":\"invalid_request_error\"}}");
    send_json(fd,code,code==400?"Bad Request":"Internal Server Error",b.p);
    free(b.p);
}

/* trailing incomplete UTF-8 sequence length at the end of [p,p+n) */
static size_t utf8_tail(const char*p,size_t n){
    size_t i=n;
    while(i>0){
        unsigned char c=(unsigned char)p[i-1];
        if(c<0x80) return 0;                          /* ascii: complete */
        if(c>=0xC0){                                  /* lead byte at i-1 */
            size_t need=(c>=0xF0)?4:(c>=0xE0)?3:2;
            size_t have=n-(i-1);
            return have<need? have:0;
        }
        i--;                                          /* continuation byte */
        if(n-i>=4) return 0;                          /* malformed; emit as-is */
    }
    return 0;
}

/* ---------------- hybrid state checkpoint cache ----------------
 * (local checkpoint cache + DRAM/disk tiering)
 *
 * The strict-extension live cache misses every agent-loop turn: clients
 * resend a RECONSTRUCTED history (think stripped, tool calls re-rendered)
 * that diverges at the first assistant turn, and the SSM state cannot
 * rewind.  So snapshot the engine state at END OF PROMPT -- before
 * generation, when the state covers exactly the tokens the client sent --
 * keyed by those tokens.  A later request restores the longest stored
 * prefix and prefills only the delta.  Correctness is unconditional: state
 * := checkpoint(N), then [N,nt) prefills from the real tokens, so output is
 * bit-identical to a full prefill; checkpoints only buy speed.
 *
 * Storage: one DRAM entry per conversation lineage.  The SSM blob (~63MB,
 * all 30 recurrent layers) is overwritten whole each snapshot -- it cannot
 * be sliced; attention KV grows by append-only per-turn chunks, so a turn
 * costs ~63MB + ~20KB/new-token of D2H, not a full re-export.  Snapshots
 * are gated on prompt length (the SSM blob costs as much as the KV of ~3k
 * tokens; short chats don't pay) and evicted LRU past a byte budget,
 * spilling to --cache-dir when set.  Restores skip the KV H2D for pages
 * the GPU still holds for the same prefix (tracked in g_kv_ids: KV is
 * positional and append-only, so pages survive both generation and
 * strict-extension turns; only a conflicting prefill overwrites them). */

typedef struct { int p0,p1; size_t off; } kvchunk_t;
typedef struct ckpt {
    int *ids,n;                        /* prompt-token prefix this state reproduces */
    uint8_t *ssm;                      /* opaque SSM+conv blob (all recurrent layers) */
    uint8_t *kv; size_t kv_sz,kv_cap;  /* attn KV, append-only chunk slabs */
    kvchunk_t *ch; int nch,chcap;
    unsigned long long tick;           /* LRU stamp */
    struct ckpt*next;
} ckpt_t;
typedef struct dent {                  /* disk-tier index entry (ids stay resident) */
    char*path; int*ids; int n; size_t bytes; unsigned long long tick;
    struct dent*next;
} dent_t;
/* on-disk blob: header + ids + chunk table + ssm + kv.  `key` folds model
 * identity, --kv-quant, --ctx and the blob-layout version: state bytes are
 * engine-layout-specific, so a mismatched restore is a crash, not a miss. */
typedef struct { unsigned long long magic,key,tick; int n,nch;
                 unsigned long long ssm_sz,kv_sz; } ckhdr_t;
#define CK_MAGIC 0x513336434B505431ULL   /* "Q36CKPT1" */
#define CK_LAYOUT_VER 1

static ckpt_t*g_ck; static dent_t*g_dk;
static int g_ck_on=1, g_ck_min=2048;         /* --no-state-cache, --cache-min */
static int g_ck_log=0;                       /* --cache-log: per-op [ckpt] lines
                                                (warnings always print; the [msg]
                                                line carries the cached count) */
#define CKLOG(...) do{ if(g_ck_log) fprintf(stderr,__VA_ARGS__); }while(0)
static size_t g_ck_ram=0;                    /* --cache-ram (MB); 0 = auto at startup:
                                                ~4 full-ctx checkpoints, capped at a
                                                quarter of system RAM, floor 4GB */
static size_t g_ck_bytes;
static const char*g_ck_dir;                  /* --cache-dir: disk tier (opt-in) */
static size_t g_ck_disk_max=(size_t)32768<<20, g_ck_disk_bytes;  /* --cache-disk */
static unsigned long long g_ck_tick, g_ck_key;
static size_t g_ssm_sz, g_kv_psz;            /* SSM blob bytes; KV bytes/position */
static int *g_kv_ids, g_kv_n=0;              /* token whose K/V holds each GPU position */

static unsigned long long fnv64(const void*p,size_t n,unsigned long long h){
    const unsigned char*s=(const unsigned char*)p;
    if(!h) h=1469598103934665603ULL;
    while(n--){ h^=*s++; h*=1099511628211ULL; }
    return h;
}
static size_t ck_mem(ckpt_t*c){
    return sizeof*c+(size_t)c->n*sizeof(int)+(c->ssm?g_ssm_sz:0)
          +c->kv_cap+(size_t)c->chcap*sizeof(kvchunk_t);
}
static void ck_drop(ckpt_t*c){
    ckpt_t**pp=&g_ck;
    while(*pp&&*pp!=c) pp=&(*pp)->next;
    if(*pp) *pp=c->next;
    g_ck_bytes-=ck_mem(c);
    free(c->ids); free(c->ssm); free(c->kv); free(c->ch); free(c);
}
static void dk_evict(size_t incoming){
    while(g_dk&&g_ck_disk_bytes+incoming>g_ck_disk_max){
        dent_t**op=&g_dk;
        for(dent_t**pp=&g_dk;*pp;pp=&(*pp)->next) if((*pp)->tick<(*op)->tick) op=pp;
        dent_t*old=*op; *op=old->next;
        unlink(old->path); g_ck_disk_bytes-=old->bytes;
        CKLOG("[ckpt] disk evicted %d tok (%s)\n",old->n,old->path);
        free(old->path); free(old->ids); free(old);
    }
}
static void ck_spill(ckpt_t*c){
    ckhdr_t h; memset(&h,0,sizeof h);
    h.magic=CK_MAGIC; h.key=g_ck_key; h.tick=c->tick;
    h.n=c->n; h.nch=c->nch; h.ssm_sz=g_ssm_sz; h.kv_sz=c->kv_sz;
    size_t bytes=sizeof h+(size_t)c->n*sizeof(int)
                +(size_t)c->nch*sizeof(kvchunk_t)+g_ssm_sz+c->kv_sz;
    dk_evict(bytes);
    char path[512];
    snprintf(path,sizeof path,"%s/ck-%016llx.q36ck",g_ck_dir,
             fnv64(c->ids,(size_t)c->n*sizeof(int),g_ck_key));
    FILE*f=fopen(path,"wb");
    int ok=f&&fwrite(&h,sizeof h,1,f)==1
        &&fwrite(c->ids,sizeof(int),(size_t)c->n,f)==(size_t)c->n
        &&fwrite(c->ch,sizeof(kvchunk_t),(size_t)c->nch,f)==(size_t)c->nch
        &&fwrite(c->ssm,1,g_ssm_sz,f)==g_ssm_sz
        &&fwrite(c->kv,1,c->kv_sz,f)==c->kv_sz;
    if(f&&fclose(f)) ok=0;
    if(!ok){
        fprintf(stderr,"[ckpt] disk spill failed (%s); dropping %d tok\n",path,c->n);
        if(f) unlink(path);
        ck_drop(c); return;
    }
    for(dent_t**pp=&g_dk;*pp;pp=&(*pp)->next)   /* re-spilled lineage: same path */
        if(!strcmp((*pp)->path,path)){ dent_t*s=*pp; *pp=s->next;
            g_ck_disk_bytes-=s->bytes; free(s->path); free(s->ids); free(s); break; }
    dent_t*d=(dent_t*)calloc(1,sizeof*d);
    char*dp=d?strdup(path):NULL;
    if(!dp){ free(d); unlink(path); ck_drop(c); return; }
    d->path=dp; d->ids=c->ids; c->ids=NULL; d->n=c->n;
    d->bytes=bytes; d->tick=c->tick; d->next=g_dk; g_dk=d;
    g_ck_disk_bytes+=bytes;
    CKLOG("[ckpt] spilled %d tok to disk (%.0fMB, disk %zu/%zuMB)\n",
        d->n,bytes/1048576.0,g_ck_disk_bytes>>20,g_ck_disk_max>>20);
    ck_drop(c);
}
static void ck_evict(ckpt_t*keep){
    while(g_ck_bytes>g_ck_ram){
        ckpt_t*old=NULL;
        for(ckpt_t*i=g_ck;i;i=i->next) if(i!=keep&&(!old||i->tick<old->tick)) old=i;
        if(!old) break;                 /* a single lineage may exceed the cap */
        if(g_ck_dir) ck_spill(old);
        else { CKLOG("[ckpt] evicted %d tok (LRU, dram %zuMB over cap)\n",
                       old->n,g_ck_bytes>>20); ck_drop(old); }
    }
}
/* restore engine state to checkpoint boundary c->n; 0 = ok.  SSM loads
 * whole; KV pulls only the range whose GPU pages no longer match. */
static int ck_restore(ckpt_t*c){
    double t0=now();
    if(q36_engine_ssm_load(g_e,c->ssm)) return -1;
    int live=0;
    while(live<g_kv_n&&live<c->n&&g_kv_ids[live]==c->ids[live]) live++;
    size_t kvb=0;
    for(int i=0;live<c->n&&i<c->nch;i++){
        kvchunk_t*h=&c->ch[i];
        int a=h->p0>live?h->p0:live, b=h->p1<c->n?h->p1:c->n;
        if(a>=b) continue;
        if(q36_engine_kv_range(g_e,h->p0,h->p1,c->kv+h->off,a,b,0)){
            g_kv_n=0;               /* partial load: position map no longer trusted */
            return -1;
        }
        kvb+=(size_t)(b-a)*g_kv_psz;
    }
    if(live<c->n) memcpy(g_kv_ids+live,c->ids+live,(size_t)(c->n-live)*sizeof(int));
    if(g_kv_n<c->n) g_kv_n=c->n;
    c->tick=++g_ck_tick;
    CKLOG("[ckpt] restored %d tok in %.0fms (ssm %.0fMB h2d, kv %.0fMB h2d, %d pos gpu-valid)\n",
        c->n,(now()-t0)*1e3,g_ssm_sz/1048576.0,kvb/1048576.0,live<c->n?live:c->n);
    return 0;
}
/* reinflate a spilled checkpoint into DRAM (move semantics: file removed) */
static ckpt_t* dk_load(dent_t*d){
    double t0=now();
    FILE*f=fopen(d->path,"rb");
    ckhdr_t h; ckpt_t*c=NULL;
    int ok=f&&fread(&h,sizeof h,1,f)==1&&h.magic==CK_MAGIC&&h.key==g_ck_key
         &&h.n==d->n&&h.n>0&&h.n<=g_ctx&&h.nch>0&&h.ssm_sz==g_ssm_sz
         &&fseek(f,(long)((size_t)h.n*sizeof(int)),SEEK_CUR)==0;
    if(ok){
        c=(ckpt_t*)calloc(1,sizeof*c);
        if(c){
            c->ch=(kvchunk_t*)malloc((size_t)h.nch*sizeof(kvchunk_t));
            c->ssm=(uint8_t*)malloc(g_ssm_sz);
            c->kv=(uint8_t*)malloc(h.kv_sz?h.kv_sz:1);
        }
        ok=c&&c->ch&&c->ssm&&c->kv
         &&fread(c->ch,sizeof(kvchunk_t),(size_t)h.nch,f)==(size_t)h.nch;
        for(int i=0;ok&&i<h.nch;i++){    /* chunk table sanity before use */
            kvchunk_t*k=&c->ch[i];
            if(k->p0<0||k->p1<=k->p0||k->p1>h.n
             ||k->off+(size_t)(k->p1-k->p0)*g_kv_psz>h.kv_sz) ok=0;
        }
        ok=ok&&fread(c->ssm,1,g_ssm_sz,f)==g_ssm_sz
             &&fread(c->kv,1,h.kv_sz,f)==h.kv_sz;
    }
    if(f) fclose(f);
    for(dent_t**pp=&g_dk;*pp;pp=&(*pp)->next) if(*pp==d){ *pp=d->next; break; }
    unlink(d->path); g_ck_disk_bytes-=d->bytes;
    if(!ok){
        fprintf(stderr,"[ckpt] disk blob unusable, removed (%s)\n",d->path);
        if(c){ free(c->ch); free(c->ssm); free(c->kv); free(c); }
        free(d->path); free(d->ids); free(d);
        return NULL;
    }
    c->n=h.n; c->nch=h.nch; c->chcap=h.nch;
    c->kv_sz=h.kv_sz; c->kv_cap=h.kv_sz?h.kv_sz:1;
    c->ids=d->ids; d->ids=NULL;          /* index kept them resident: move back */
    c->tick=++g_ck_tick;
    c->next=g_ck; g_ck=c; g_ck_bytes+=ck_mem(c);
    CKLOG("[ckpt] reloaded %d tok from disk (%.0fMB) in %.0fms\n",
        c->n,d->bytes/1048576.0,(now()-t0)*1e3);
    free(d->path); free(d);
    ck_evict(c);
    return c;
}
/* longest stored strict-prefix of toks[0..nt) across DRAM + disk tiers.
 * Strict (c->n < nt) because the boundary state has no logits: at least
 * the last prompt token must prefill after a restore. */
static ckpt_t* ck_lookup(const int*toks,int nt){
    ckpt_t*c=NULL;
    for(ckpt_t*i=g_ck;i;i=i->next)
        if(i->n<nt&&(!c||i->n>c->n)&&!memcmp(i->ids,toks,(size_t)i->n*sizeof(int))) c=i;
    dent_t*d=NULL;
    for(dent_t*i=g_dk;i;i=i->next)
        if(i->n<nt&&i->n>(c?c->n:0)&&(!d||i->n>d->n)
           &&!memcmp(i->ids,toks,(size_t)i->n*sizeof(int))) d=i;
    if(d){ ckpt_t*r=dk_load(d); if(r) c=r; }
    return c;
}
/* end-of-prompt snapshot.  Extends the conversation's lineage in place
 * (append the KV delta, overwrite the SSM blob) rather than storing a
 * second overlapping copy; the trade-off is that a lineage keeps only its
 * NEWEST boundary -- fine for linear agent loops, a fork from an older
 * turn re-prefills.  A failed allocation only skips the extension: the
 * entry still holds the previous boundary's valid state. */
static void ck_note(const int*toks,int nt){
    if(!g_ck_on||nt<g_ck_min) return;
    double t0=now();
    ckpt_t*c=NULL;
    for(ckpt_t*i=g_ck;i;i=i->next)
        if(i->n<=nt&&(!c||i->n>c->n)&&!memcmp(i->ids,toks,(size_t)i->n*sizeof(int))) c=i;
    if(c&&c->n==nt){ c->tick=++g_ck_tick; return; }   /* same prompt: same state */
    int p0=c?c->n:0, isnew=!c;
    size_t need=(size_t)(nt-p0)*g_kv_psz;
    if(isnew){
        c=(ckpt_t*)calloc(1,sizeof*c);
        if(c) c->ssm=(uint8_t*)malloc(g_ssm_sz);
        if(!c||!c->ssm){ free(c); fprintf(stderr,"[ckpt] snapshot skipped (alloc)\n"); return; }
        c->next=g_ck; g_ck=c;
        g_ck_bytes+=ck_mem(c);
    }
    g_ck_bytes-=ck_mem(c);
    int ok=1;
    { int*p=(int*)realloc(c->ids,(size_t)nt*sizeof(int)); if(p)c->ids=p; else ok=0; }
    if(ok&&c->nch==c->chcap){
        int cap=c->chcap?2*c->chcap:8;
        kvchunk_t*p=(kvchunk_t*)realloc(c->ch,(size_t)cap*sizeof(kvchunk_t));
        if(p){ c->ch=p; c->chcap=cap; } else ok=0;
    }
    if(ok&&c->kv_sz+need>c->kv_cap){
        size_t cap=c->kv_cap?c->kv_cap:4096;
        while(cap<c->kv_sz+need) cap*=2;
        uint8_t*p=(uint8_t*)realloc(c->kv,cap);
        if(p){ c->kv=p; c->kv_cap=cap; } else ok=0;
    }
    g_ck_bytes+=ck_mem(c);
    if(!ok){
        fprintf(stderr,"[ckpt] snapshot skipped (alloc %.0fMB)\n",need/1048576.0);
        if(isnew) ck_drop(c);
        return;
    }
    memcpy(c->ids,toks,(size_t)nt*sizeof(int));
    if(q36_engine_kv_range(g_e,p0,nt,c->kv+c->kv_sz,p0,nt,1)
     ||q36_engine_ssm_save(g_e,c->ssm)){
        if(isnew) ck_drop(c);           /* extension untouched: entry stays at p0 */
        return;
    }
    c->ch[c->nch].p0=p0; c->ch[c->nch].p1=nt; c->ch[c->nch].off=c->kv_sz; c->nch++;
    c->kv_sz+=need; c->n=nt; c->tick=++g_ck_tick;
    CKLOG("[ckpt] %s %d tok (+%d, ssm %.0fMB + kv %.0fMB) in %.0fms | dram %zu/%zuMB\n",
        isnew?"saved":"extended",nt,nt-p0,g_ssm_sz/1048576.0,c->kv_sz/1048576.0,
        (now()-t0)*1e3,g_ck_bytes>>20,g_ck_ram>>20);
    ck_evict(c);
}
/* --cache-dir startup scan: index usable blobs (header + ids only) */
static void dk_scan(void){
    mkdir(g_ck_dir,0755);   /* ok if it exists */
    DIR*dr=opendir(g_ck_dir);
    if(!dr){ fprintf(stderr,"[ckpt] --cache-dir %s: %s; disk tier off\n",
                     g_ck_dir,strerror(errno)); g_ck_dir=NULL; return; }
    struct dirent*de; int kept=0,skip=0;
    while((de=readdir(dr))){
        size_t L=strlen(de->d_name);
        if(L<7||strcmp(de->d_name+L-6,".q36ck")) continue;
        char path[512]; snprintf(path,sizeof path,"%s/%s",g_ck_dir,de->d_name);
        FILE*f=fopen(path,"rb"); ckhdr_t h; int*ids=NULL;
        int ok=f&&fread(&h,sizeof h,1,f)==1&&h.magic==CK_MAGIC&&h.key==g_ck_key
             &&h.n>0&&h.n<=g_ctx&&h.nch>0&&h.ssm_sz==g_ssm_sz;
        if(ok){
            ids=(int*)malloc((size_t)h.n*sizeof(int));
            ok=ids&&fread(ids,sizeof(int),(size_t)h.n,f)==(size_t)h.n;
        }
        if(f) fclose(f);
        if(!ok){ free(ids); skip++; continue; }   /* stale key/other config: leave it */
        dent_t*d=(dent_t*)calloc(1,sizeof*d);
        char*dp=d?strdup(path):NULL;
        if(!dp){ free(d); free(ids); skip++; continue; }
        d->path=dp; d->ids=ids; d->n=h.n;
        d->bytes=sizeof h+(size_t)h.n*sizeof(int)
                +(size_t)h.nch*sizeof(kvchunk_t)+h.ssm_sz+h.kv_sz;
        d->tick=h.tick; d->next=g_dk; g_dk=d;
        g_ck_disk_bytes+=d->bytes;
        if(h.tick>g_ck_tick) g_ck_tick=h.tick;
        kept++;
    }
    closedir(dr);
    if(kept||skip)
        fprintf(stderr,"[ckpt] disk index: %d usable, %d skipped, %zuMB\n",
                kept,skip,g_ck_disk_bytes>>20);
}

/* ---------------- generation ---------------- */
typedef struct {
    int fd, stream, chat, proto;
    unsigned long long id; long created;
    sbuf text;            /* full generated text */
    size_t emitted;       /* text-block bytes already sent (stream) */
    size_t holdback;      /* stop-string straddle guard */
    size_t text_limit;    /* freeze streamed text here (tool call started) */
    int send_fail, first;
    int in_toks, out_toks;   /* anthropic usage fields */
    otc_t*tc; int ntc;       /* parsed output tool calls */
    /* leading <think> region, mirrored from the generation loop's tracker:
     * th_state -1 unknown / 1 inside / 0 resolved; [7..th1) = think content
     * (when th1>0), body = first byte after "</think>".  Anthropic gets it
     * as a real thinking content block (or nothing under --hide-think);
     * OpenAI keeps the raw text unless --hide-think. */
    int th_state; size_t th1, body;
    int tk_open, tk_closed, tx_open;  /* anthropic stream block state */
    size_t temitted;                  /* thinking content bytes sent */
    int blk;                          /* next anthropic content block index */
} emit_t;

/* visible-text start under --hide-think: past the think region (all of it
 * when the block never closed), past blank lines that follow the tag */
static size_t vis_start(emit_t*em){
    size_t b=em->th_state==1?em->text.n:em->body;
    if(b>0) while(b<em->text.n&&em->text.p[b]=='\n') b++;
    return b;
}

/* internal finish markers ("stop"/"length"/"stop_sequence"/"tool") ->
 * per-protocol wire values */
static const char*fin_openai(const char*f){
    if(!strcmp(f,"tool"))return "tool_calls";
    return strcmp(f,"stop_sequence")?f:"stop";
}
static const char*fin_anthropic(const char*f){
    if(!strcmp(f,"stop"))return "end_turn";
    if(!strcmp(f,"length"))return "max_tokens";
    if(!strcmp(f,"tool"))return "tool_use";
    return "stop_sequence";
}

static int emit_chunk(emit_t*em,const char*finish){
    /* stream one SSE delta of any newly-safe bytes (plus finish marker) */
    size_t safe=em->text.n;
    if(em->text_limit!=(size_t)-1&&safe>em->text_limit) safe=em->text_limit;
    if(!finish){
        if(safe>em->holdback) safe-=em->holdback; else safe=0;
        safe-=utf8_tail(em->text.p,safe);
        if(safe<=em->emitted) return 0;
    }
    if(finish && safe>em->emitted) safe-=utf8_tail(em->text.p,safe);
    if(em->proto==PROTO_ANTH){
        /* Anthropic SSE: named events.  The model's leading <think> region
         * streams as a real thinking content block (Claude Code renders it
         * as thinking, not text; suppressed by --hide-think), the remainder
         * as a text block, tool_use blocks after. */
        sbuf b={0,0,0};
        if(em->first){
            sb_fmt(&b,"event: message_start\ndata: {\"type\":\"message_start\",\"message\":"
                "{\"id\":\"msg_%llu\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"" MODEL_ID "\","
                "\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,"
                "\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}\n\n",em->id,em->in_toks);
            em->first=0;
        }
        int resolved=em->th_state>=0||finish;
        int has_think=em->th_state==1||em->th1>0;
        if(resolved&&has_think&&!g_hide_think&&!em->tk_closed){
            size_t c0=7;                     /* content starts after "<think>" */
            while(c0<em->text.n&&em->text.p[c0]=='\n') c0++;
            int closing=em->th_state!=1||finish;
            size_t cend;
            if(em->th_state!=1)      cend=em->th1;            /* closed at the tag */
            else if(finish)          cend=safe;               /* unterminated think */
            else                     cend=safe>c0+8?safe-8:c0;/* "</think>" straddle */
            if(cend<c0) cend=c0;
            cend-=utf8_tail(em->text.p,cend);
            if(!em->tk_open&&cend>c0){       /* open on first content byte:
                                                whitespace-only think (common at
                                                low temp) emits no block, like
                                                the real API omits empty blocks */
                sb_fmt(&b,"event: content_block_start\ndata: {\"type\":\"content_block_start\","
                    "\"index\":%d,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\","
                    "\"signature\":\"\"}}\n\n",em->blk);
                em->tk_open=1; em->temitted=c0;
            }
            if(em->tk_open&&cend>em->temitted){
                sb_fmt(&b,"event: content_block_delta\ndata: {\"type\":\"content_block_delta\","
                    "\"index\":%d,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"",em->blk);
                sb_jesc(&b,em->text.p+em->temitted,cend-em->temitted);
                sb_str(&b,"\"}}\n\n");
                em->temitted=cend;
            }
            if(closing){
                if(em->tk_open){
                    sb_fmt(&b,"event: content_block_delta\ndata: {\"type\":\"content_block_delta\","
                        "\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"\"}}\n\n",em->blk);
                    sb_fmt(&b,"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":%d}\n\n",em->blk);
                    em->blk++;
                }
                em->tk_closed=1;
            }
        }
        /* text block: starts past the think region (at 0 when none) */
        if(resolved&&em->th_state!=1){
            if(!em->tx_open){
                if(em->emitted<em->body) em->emitted=em->body;
                if(em->body>0)           /* trim blank lines after </think> */
                    while(em->emitted<safe&&em->text.p[em->emitted]=='\n') em->emitted++;
                /* open on first text byte; at finish only when it would be
                 * the sole block (the real API omits empty text blocks) */
                if(em->emitted<safe||(finish&&em->blk==0&&!em->ntc)){
                    sb_fmt(&b,"event: content_block_start\ndata: {\"type\":\"content_block_start\","
                        "\"index\":%d,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",em->blk);
                    em->tx_open=1;
                }
            }
            if(em->tx_open&&safe>em->emitted){
                sb_fmt(&b,"event: content_block_delta\ndata: {\"type\":\"content_block_delta\","
                    "\"index\":%d,\"delta\":{\"type\":\"text_delta\",\"text\":\"",em->blk);
                sb_jesc(&b,em->text.p+em->emitted,safe-em->emitted);
                sb_str(&b,"\"}}\n\n");
                em->emitted=safe;
            }
        }
        if(finish){
            if(em->tx_open){
                sb_fmt(&b,"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":%d}\n\n",em->blk);
                em->blk++;
            }
            for(int k=0;k<em->ntc;k++){       /* tool_use blocks after the text */
                sb_fmt(&b,"event: content_block_start\ndata: {\"type\":\"content_block_start\","
                    "\"index\":%d,\"content_block\":{\"type\":\"tool_use\","
                    "\"id\":\"toolu_%llu_%d\",\"name\":\"",em->blk+k,em->id,k);
                sb_jesc(&b,em->tc[k].name,em->tc[k].nlen);
                sb_str(&b,"\",\"input\":{}}}\n\n");
                sb_fmt(&b,"event: content_block_delta\ndata: {\"type\":\"content_block_delta\","
                    "\"index\":%d,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"",em->blk+k);
                sb_jesc(&b,em->tc[k].args.p,em->tc[k].args.n);
                sb_str(&b,"\"}}\n\n");
                sb_fmt(&b,"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":%d}\n\n",em->blk+k);
            }
            sb_fmt(&b,"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":"
                "{\"stop_reason\":\"%s\",\"stop_sequence\":null},"
                "\"usage\":{\"output_tokens\":%d}}\n\n",fin_anthropic(finish),em->out_toks);
            sb_str(&b,"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        }
        int rc=b.n?send_all(em->fd,b.p,b.n):0;
        free(b.p);
        if(rc) em->send_fail=1;
        return rc?-1:0;
    }
    if(g_hide_think){   /* openai wire: suppress the think region entirely */
        if(!finish&&em->th_state!=0) return 0;   /* unknown or inside: hold */
        size_t v=em->th_state==1?safe:em->body;  /* unterminated: all hidden */
        if(v>0) while(v<safe&&em->text.p[v]=='\n') v++;
        if(em->emitted<v) em->emitted=v;
    }
    sbuf b={0,0,0};
    sb_fmt(&b,"data: {\"id\":\"%s-%llu\",\"object\":\"%s\",\"created\":%ld,"
              "\"model\":\"" MODEL_ID "\",\"choices\":[{\"index\":0,",
        em->chat?"chatcmpl":"cmpl",em->id,
        em->chat?"chat.completion.chunk":"text_completion",em->created);
    if(em->chat){
        sb_str(&b,"\"delta\":{");
        if(em->first){ sb_str(&b,"\"role\":\"assistant\",\"content\":\""); em->first=0; }
        else sb_str(&b,"\"content\":\"");
    } else sb_str(&b,"\"text\":\"");
    if(safe>em->emitted) sb_jesc(&b,em->text.p+em->emitted,safe-em->emitted);
    em->emitted=safe>em->emitted?safe:em->emitted;
    if(em->chat){
        sb_str(&b,"\"");
        if(finish&&em->ntc){              /* tool_calls inside the delta */
            sb_str(&b,",\"tool_calls\":[");
            for(int k=0;k<em->ntc;k++){
                sb_fmt(&b,"%s{\"index\":%d,\"id\":\"call_%llu_%d\",\"type\":\"function\","
                    "\"function\":{\"name\":\"",k?",":"",k,em->id,k);
                sb_jesc(&b,em->tc[k].name,em->tc[k].nlen);
                sb_str(&b,"\",\"arguments\":\"");
                sb_jesc(&b,em->tc[k].args.p,em->tc[k].args.n);
                sb_str(&b,"\"}}");
            }
            sb_str(&b,"]");
        }
        sb_str(&b,"}");
    } else sb_str(&b,"\"");
    if(finish) sb_fmt(&b,",\"finish_reason\":\"%s\"}]}\n\n",fin_openai(finish));
    else sb_str(&b,",\"finish_reason\":null}]}\n\n");
    int rc=send_all(em->fd,b.p,b.n);
    free(b.p);
    if(rc){ em->send_fail=1; return -1; }
    if(finish){ if(send_all(em->fd,"data: [DONE]\n\n",14)) em->send_fail=1; }
    return 0;
}

/* anthropic top-level system -> leading system message (purge keeps it) */
static void req_sysfold(req_t*r){
    if(!r->sys.n||r->nmsg>=MAX_MSG) return;
    memmove(r->mrole+1,r->mrole,(size_t)r->nmsg*sizeof(char*));
    memmove(r->mtext+1,r->mtext,(size_t)r->nmsg*sizeof(char*));
    r->mrole[0]=strdup("system"); r->mtext[0]=r->sys.p;
    r->sys.p=NULL; r->sys.n=r->sys.cap=0;
    r->nmsg++;
}

/* POST /v1/messages/count_tokens: tokenize without generating */
static void handle_count(int fd,const char*body,size_t blen){
    req_t r; memset(&r,0,sizeof r);
    if(parse_request(body,blen,&r,PROTO_ANTH)||!r.nmsg){
        req_free(&r); send_err(fd,400,"malformed request body"); return; }
    req_sysfold(&r);
    build_prompt(&r,0,&r.prompt);
    int nt=q36_encode(&g_vocab,r.prompt.p,g_toks,4*g_ctx);
    sbuf b={0,0,0};
    sb_fmt(&b,"{\"input_tokens\":%d}",nt>0?nt:0);
    send_json(fd,200,"OK",b.p);
    free(b.p); req_free(&r);
}

static void handle_generate(int fd,const char*body,size_t blen,int proto){
    int chat=(proto!=PROTO_CMPL);
    req_t r; memset(&r,0,sizeof r);
    r.temp=1.0f; r.topk=20; r.topp=0.95f; r.max_tokens=-1; r.seed=42;
    if(parse_request(body,blen,&r,proto)||(chat&&!r.nmsg)){
        req_free(&r); send_err(fd,400,"malformed request body"); return; }
    if(!r.has_temp) r.temp=g_temp;
    if(proto==PROTO_ANTH) req_sysfold(&r);

    /* tokenize; if over budget and --auto-purge: drop oldest non-system
     * turns (chat) or left-truncate tokens (completions).  The hybrid SSM
     * state cannot shift/rewind, so a purge is a rebuild-and-reprefill. */
    int reserve=r.max_tokens>0?r.max_tokens:(g_ctx/8<256?256:g_ctx/8);
    if(reserve>g_ctx/2) reserve=g_ctx/2;
    int nt, purged=0;
    if(chat){
        int first=1;
        build_prompt(&r,first-1,&r.prompt);       /* first-1==0: all messages */
        nt=q36_encode(&g_vocab,r.prompt.p,g_toks,4*g_ctx);
        while(g_autopurge && nt>0 && nt>g_ctx-1-reserve && first<r.nmsg){
            build_prompt(&r,first++,&r.prompt);
            nt=q36_encode(&g_vocab,r.prompt.p,g_toks,4*g_ctx);
            purged=first-1;
        }
    } else {
        nt=q36_encode(&g_vocab,r.prompt.p,g_toks,4*g_ctx);
        if(g_autopurge && nt>g_ctx-1-reserve){
            int keep=g_ctx-1-reserve;
            memmove(g_toks,g_toks+(nt-keep),(size_t)keep*sizeof(int));
            purged=nt-keep; nt=keep;
        }
    }
    if(purged) fprintf(stderr,"[purge] dropped %d %s to fit ctx %d\n",
                       purged,chat?"message(s)":"token(s)",g_ctx);
    if(nt<=0){ req_free(&r); send_err(fd,400,"tokenization failed"); return; }
    if(nt>=g_ctx-1){ req_free(&r);
        send_err(fd,400,g_autopurge?"prompt exceeds context even after purge"
                                   :"prompt exceeds context (auto-purge disabled)"); return; }
    int maxn=r.max_tokens>0?r.max_tokens:g_ctx-nt-1;
    if(maxn>g_ctx-nt-1) maxn=g_ctx-nt-1;

    /* prefix cache: strict extension of the live state; else restore the
     * longest stored checkpoint and prefill just the delta; else full
     * reset.  (The SSM state cannot rewind, so anything short of strict
     * extension needs a checkpoint or a rebuild.) */
    int common=0;
    while(common<g_cache_n&&common<nt&&g_cache_ids[common]==g_toks[common]) common++;
    int pos0=(g_cache_n>0&&common==g_cache_n&&nt>g_cache_n)?g_cache_n:0;
    if(pos0==0){
        ckpt_t*c=g_ck_on?ck_lookup(g_toks,nt):NULL;
        if(c&&ck_restore(c)==0) pos0=c->n;
        else q36_engine_reset(g_e);
        g_cache_n=0;
    }

    int sampled=(r.temp>0.f);
    unsigned long long seed=r.has_seed?(unsigned long long)r.seed
                                      :(unsigned long long)(now()*1e6)+g_reqid;
    q36_engine_set_sampler(g_e,sampled?r.temp:0.f,r.topk,r.topp,seed);

    double t0=now();
    int ext=nt-pos0, cur;
    if(sampled){
        if(ext>1) q36_engine_prefill_from(g_e,g_toks+pos0,ext-1,pos0);
        cur=q36_engine_step(g_e,g_toks[nt-1],nt-1);
    } else {
        cur=q36_engine_prefill_from(g_e,g_toks+pos0,ext,pos0);
    }
    double pf_ms=(now()-t0)*1e3;
    memcpy(g_cache_ids,g_toks,(size_t)nt*sizeof(int)); g_cache_n=nt;
    memcpy(g_kv_ids+pos0,g_toks+pos0,(size_t)(nt-pos0)*sizeof(int));
    if(g_kv_n<nt) g_kv_n=nt;
    ck_note(g_toks,nt);   /* end-of-prompt snapshot: before generation, so the
                             key never depends on generated (think/tool-call)
                             text that clients won't resend verbatim */

    emit_t em; memset(&em,0,sizeof em);
    em.fd=fd; em.stream=r.stream; em.chat=chat; em.proto=proto; em.first=1;
    em.id=++g_reqid; em.created=time(NULL); em.in_toks=nt;
    em.text_limit=(size_t)-1; em.th_state=-1;
    sb_put(&em.text,"",0);
    for(int i=0;i<r.nstop;i++){ size_t L=strlen(r.stop[i]); if(L>em.holdback)em.holdback=L; }
    if(r.ntools&&em.holdback<16) em.holdback=16;   /* don't stream a partial <tool_call> */
    if(r.stream){
        const char*h="HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if(send_all(fd,h,strlen(h))) em.send_fail=1;
    }

    double t1=now();
    int pos=nt, gen=0;
    const char*finish="length";
    char tb[64];
    /* MTP self-speculative decode: greedy requests only (verify is argmax
     * equality; the engine itself also refuses when the sampler is hot).
     * step_mtp returns 1..K+1 tokens per call; all but the last are
     * already fed into the engine, so they are recorded as in-state the
     * moment they are returned -- even if a stop/EOS lands mid-batch and
     * they are never emitted.  g_cache_ids must mirror the REAL state or
     * the next request's prefix-extension check would trust stale KV. */
    int use_mtp=(g_mtp&&!sampled);
    int pend[3],npend=0,ip=0;
    long mc0=0,ma0=0; if(use_mtp) q36_engine_mtp_stats(g_e,&mc0,&ma0);
    /* reasoning-model stop handling: while inside an unclosed LEADING <think>
     * block, user stop strings are suppressed -- thinking routinely quotes
     * the prompt (e.g. gsm8k sends stop "Question:" and the model writes
     * 'User says: "Question: ..."', truncating every answer at ~23 tokens).
     * EOS and <|im_end|> still end generation.  Gate opens at </think>. */
    int think_gate=-1;                       /* -1 undecided, 1 gated, 0 open */
    size_t body0=0;                          /* scan offset past the think block */
    while(!em.send_fail){
        if(cur==g_eos||cur==g_imend){ finish="stop"; break; }
        q36_detok_one(&g_vocab,cur,tb,sizeof tb);
        size_t tn=strlen(tb);
        if(tn>0) sb_put(&em.text,tb,tn);
        gen++;
        if(think_gate==-1&&em.text.n>=7)
            think_gate=strncmp(em.text.p,"<think>",7)==0;
        if(think_gate==1){
            char*te=strstr(em.text.p,"</think>");
            if(te){ think_gate=0; body0=(size_t)(te+8-em.text.p);
                    em.th1=(size_t)(te-em.text.p); em.body=body0; }
        }
        em.th_state=think_gate;
        if(r.ntools&&em.text_limit==(size_t)-1&&think_gate<1){
            char*tcp=strstr(em.text.p+body0,"<tool_call>");
            if(tcp) em.text_limit=(size_t)(tcp-em.text.p);  /* freeze text stream */
        }
        int stopped=0;
        if(think_gate<1)
        for(int i=0;i<r.nstop;i++){          /* stop-string scan (with slack for straddle) */
            size_t L=strlen(r.stop[i]);
            size_t from=em.text.n>tn+2*L?em.text.n-tn-2*L:0;
            char*hit=strstr(em.text.p+from,r.stop[i]);
            if(hit){ em.text.n=(size_t)(hit-em.text.p); em.text.p[em.text.n]=0;
                     finish="stop_sequence"; stopped=1; break; }
        }
        if(stopped) break;
        if(gen>=maxn) break;
        if(r.stream) emit_chunk(&em,NULL);
        if(ip<npend){ cur=pend[ip++]; continue; } /* fed + recorded by step_mtp below */
        if(use_mtp){
            g_cache_ids[g_cache_n++]=cur;    /* about to be fed to the engine */
            g_kv_ids[pos]=cur; if(g_kv_n<=pos) g_kv_n=pos+1;
            int o4[4],k=q36_engine_step_mtp(g_e,cur,pos,o4);
            for(int j=0;j+1<k;j++){          /* accepted drafts: fed at pos+1+j */
                g_cache_ids[g_cache_n++]=o4[j];
                g_kv_ids[pos+1+j]=o4[j];
            }
            pos+=k; if(g_kv_n<pos) g_kv_n=pos;
            cur=o4[0];
            npend=k-1; ip=0; for(int j=0;j<npend;j++) pend[j]=o4[j+1];
        } else {
            g_cache_ids[g_cache_n++]=cur;    /* about to be fed to the engine */
            g_kv_ids[pos]=cur; if(g_kv_n<=pos) g_kv_n=pos+1;
            cur=q36_engine_step(g_e,cur,pos++);
        }
    }
    double dt=now()-t1;

    /* parse tool calls out of the generated text (post-think region) */
    otc_t otc[MAX_TOOLS]; int ntc=0;
    if(r.ntools&&em.text.n>body0){
        ntc=parse_out_tools(&r,em.text.p,body0,otc,MAX_TOOLS);
        int seen=0;
        for(const char*p=em.text.p+body0;(p=strstr(p,"<tool_call>"));p+=11) seen++;
        g_tc_ok+=(unsigned long long)ntc;
        if(seen>ntc) g_tc_bad+=(unsigned long long)(seen-ntc);
    }
    if(ntc){
        finish="tool";
        em.tc=otc; em.ntc=ntc;
        if(em.text_limit==(size_t)-1){
            char*tcp=strstr(em.text.p+body0,"<tool_call>");
            if(tcp) em.text_limit=(size_t)(tcp-em.text.p);
        }
        if(em.text_limit!=(size_t)-1){   /* content = text before the calls */
            size_t cl=em.text_limit;
            while(cl>em.emitted&&(em.text.p[cl-1]=='\n'||em.text.p[cl-1]==' ')) cl--;
            em.text_limit=cl;
            if(em.text.n>cl) em.text.n=cl;
        }
    }

    em.out_toks=gen;
    if(em.th_state<0) em.th_state=0;   /* output too short to ever resolve */
    if(r.stream){
        if(!em.send_fail) emit_chunk(&em,finish);
    } else if(proto==PROTO_ANTH){
        sbuf b={0,0,0};
        sb_fmt(&b,"{\"id\":\"msg_%llu\",\"type\":\"message\",\"role\":\"assistant\","
                  "\"model\":\"" MODEL_ID "\",\"content\":[",em.id);
        int blocks=0;
        if((em.th_state==1||em.th1>0)&&!g_hide_think){
            size_t c0=7, ce=em.th_state==1?em.text.n:em.th1;
            while(c0<ce&&em.text.p[c0]=='\n') c0++;
            if(ce>c0){          /* whitespace-only think (common at low temp)
                                   emits no block, like the real API omits
                                   empty content blocks */
                sb_str(&b,"{\"type\":\"thinking\",\"thinking\":\"");
                sb_jesc(&b,em.text.p+c0,ce-c0);
                sb_str(&b,"\",\"signature\":\"\"}"); blocks++;
            }
        }
        size_t v=vis_start(&em);
        if(em.text.n>v){
            sb_fmt(&b,"%s{\"type\":\"text\",\"text\":\"",blocks?",":"");
            sb_jesc(&b,em.text.p+v,em.text.n-v);
            sb_str(&b,"\"}"); blocks++;
        }
        for(int k=0;k<ntc;k++){
            sb_fmt(&b,"%s{\"type\":\"tool_use\",\"id\":\"toolu_%llu_%d\",\"name\":\"",
                   blocks?",":"",em.id,k);
            sb_jesc(&b,otc[k].name,otc[k].nlen);
            sb_str(&b,"\",\"input\":");
            sb_put(&b,otc[k].args.p,otc[k].args.n);
            sb_str(&b,"}"); blocks++;
        }
        sb_fmt(&b,"],\"stop_reason\":\"%s\",\"stop_sequence\":null,"
                  "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}",
            fin_anthropic(finish),nt,gen);
        send_json(fd,200,"OK",b.p);
        free(b.p);
    } else {
        sbuf b={0,0,0};
        sb_fmt(&b,"{\"id\":\"%s-%llu\",\"object\":\"%s\",\"created\":%ld,"
                  "\"model\":\"" MODEL_ID "\",\"choices\":[{\"index\":0,",
            chat?"chatcmpl":"cmpl",em.id,
            chat?"chat.completion":"text_completion",em.created);
        if(chat) sb_str(&b,"\"message\":{\"role\":\"assistant\",\"content\":\"");
        else     sb_str(&b,"\"text\":\"");
        { size_t v=g_hide_think?vis_start(&em):0;
          sb_jesc(&b,em.text.p+v,em.text.n>v?em.text.n-v:0); }
        if(chat){
            sb_str(&b,"\"");
            if(ntc){
                sb_str(&b,",\"tool_calls\":[");
                for(int k=0;k<ntc;k++){
                    sb_fmt(&b,"%s{\"id\":\"call_%llu_%d\",\"type\":\"function\","
                        "\"function\":{\"name\":\"",k?",":"",em.id,k);
                    sb_jesc(&b,otc[k].name,otc[k].nlen);
                    sb_str(&b,"\",\"arguments\":\"");
                    sb_jesc(&b,otc[k].args.p,otc[k].args.n);
                    sb_str(&b,"\"}}");
                }
                sb_str(&b,"]");
            }
            sb_str(&b,"}");
        } else sb_str(&b,"\"");
        sb_fmt(&b,",\"finish_reason\":\"%s\"}],"
                  "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
                  "\"timings\":{\"prompt_n\":%d,\"prompt_ms\":%.1f,\"prompt_per_second\":%.1f,"
                  "\"predicted_n\":%d,\"predicted_ms\":%.1f,\"predicted_per_second\":%.1f,"
                  "\"cached_tokens\":%d}}",
            fin_openai(finish),nt,gen,nt+gen,
            ext,pf_ms,ext>0?ext*1000.0/pf_ms:0.0,
            gen,dt*1e3,gen>0?gen/dt:0.0,pos0);
        send_json(fd,200,"OK",b.p);
        free(b.p);
    }
    g_tok_pf+=(unsigned long long)ext; g_tok_gen+=(unsigned long long)gen;
    g_tok_hit+=(unsigned long long)pos0;
    char mtps[48]="";
    if(use_mtp){
        long mc1,ma1; q36_engine_mtp_stats(g_e,&mc1,&ma1);
        if(mc1>mc0) snprintf(mtps,sizeof mtps," | mtp accept %.0f%% (%.2f tok/verify)",
            100.0*(ma1-ma0)/((mc1-mc0)*g_mtp),1.0+(double)(ma1-ma0)/(mc1-mc0));
    }
    fprintf(stderr,"[%s] prompt %d tok (%d cached) prefill %.0f t/s | gen %d tok %.1f t/s | %s"
        " | lifetime %.2fM (pf %.2fM + gen %.2fM), cache-hit %.2fM | tc ok %llu bad %llu%s\n",
        proto==PROTO_ANTH?"msg":chat?"chat":"cmpl",
        nt,pos0,ext>0?ext*1000.0/pf_ms:0.0,gen,gen>0?gen/dt:0.0,finish,
        (g_tok_pf+g_tok_gen)/1e6,g_tok_pf/1e6,g_tok_gen/1e6,g_tok_hit/1e6,
        g_tc_ok,g_tc_bad,mtps);
    for(int k=0;k<ntc;k++) free(otc[k].args.p);
    free(em.text.p);
    req_free(&r);
}

/* ---------------- connection handling ---------------- */
static void handle_conn(int fd){
    sbuf req={0,0,0};
    char buf[65536];
    size_t hdr_end=0; long clen=-1;
    for(;;){
        ssize_t k=recv(fd,buf,sizeof buf,0);
        if(k<=0){ free(req.p); return; }
        sb_put(&req,buf,(size_t)k);
        if(!hdr_end){ char*e=strstr(req.p,"\r\n\r\n"); if(e) hdr_end=(size_t)(e-req.p)+4; }
        if(hdr_end){
            if(clen<0){
                char*h=strcasestr(req.p,"content-length:");
                clen=(h&&h<req.p+hdr_end)?atol(h+15):0;
                if(clen>64*1024*1024){ send_err(fd,400,"body too large"); free(req.p); return; }
            }
            if(req.n>=hdr_end+(size_t)clen) break;
        }
        if(req.n>80*1024*1024){ free(req.p); return; }
    }
    char method[8]={0}, path[256]={0};
    sscanf(req.p,"%7s %255s",method,path);
    const char*body=req.p+hdr_end; size_t blen=(size_t)clen;

    if(!strcmp(method,"OPTIONS")){
        const char*h="HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n";
        send_all(fd,h,strlen(h));
    }
    else {
        /* tolerant routing: strip the query string and trailing slashes,
         * collapse repeated /v1 prefixes -- SDKs build paths from base_url,
         * so base_url without /v1 (or ending in /v1 while the SDK appends
         * /v1/messages, as the Anthropic SDK does) must still route. */
        char norm[128];
        snprintf(norm,sizeof norm,"%s",path);
        char*q=strchr(norm,'?'); if(q)*q=0;
        size_t pl=strlen(norm);
        while(pl>1&&norm[pl-1]=='/') norm[--pl]=0;
        const char*pp=norm;
        while(!strncmp(pp,"/v1/",4)) pp+=3;
        if(!strcmp(method,"GET")&&(!strcmp(norm,"/health")||!strcmp(norm,"/")))
            send_json(fd,200,"OK","{\"status\":\"ok\"}");
        else if(!strcmp(method,"GET")&&!strcmp(pp,"/models"))
            send_json(fd,200,"OK","{\"object\":\"list\",\"data\":[{\"id\":\"" MODEL_ID "\","
                      "\"object\":\"model\",\"owned_by\":\"q36\"}]}");
        else if(!strcmp(method,"POST")&&!strcmp(pp,"/chat/completions"))
            handle_generate(fd,body,blen,PROTO_CHAT);
        else if(!strcmp(method,"POST")&&!strcmp(pp,"/completions"))
            handle_generate(fd,body,blen,PROTO_CMPL);
        else if(!strcmp(method,"POST")&&!strcmp(pp,"/messages"))
            handle_generate(fd,body,blen,PROTO_ANTH);
        else if(!strcmp(method,"POST")&&!strcmp(pp,"/messages/count_tokens"))
            handle_count(fd,body,blen);
        else {
            char em[256];
            snprintf(em,sizeof em,"unknown endpoint: %s %s (serving POST /v1/chat/completions, "
                     "POST /v1/completions, POST /v1/messages, GET /v1/models, GET /health)",method,path);
            fprintf(stderr,"[400] %s %s\n",method,path);
            send_err(fd,400,em);
        }
    }
    free(req.p);
}

int main(int argc,char**argv){
    int port=8080, kvq=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-m")&&i+1<argc) g_model_path=argv[++i];
        else if(!strcmp(argv[i],"--port")&&i+1<argc) port=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ctx")&&i+1<argc) g_ctx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--kv-quant")) kvq=1;
        else if(!strcmp(argv[i],"--auto-purge")) g_autopurge=1;   /* back-compat no-op */
        else if(!strcmp(argv[i],"--no-auto-purge")) g_autopurge=0;
        else if(!strcmp(argv[i],"--temp")&&i+1<argc) g_temp=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"--mtp")){ g_mtp=1;   /* optional draft depth 1-3 */
            if(i+1<argc&&argv[i+1][0]>='1'&&argv[i+1][0]<='3'&&!argv[i+1][1]) g_mtp=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"--hide-think")) g_hide_think=1;
        else if(!strcmp(argv[i],"--cache-log")) g_ck_log=1;
        else if(!strcmp(argv[i],"--no-state-cache")) g_ck_on=0;
        else if(!strcmp(argv[i],"--cache-ram")&&i+1<argc) g_ck_ram=(size_t)atoll(argv[++i])<<20;
        else if(!strcmp(argv[i],"--cache-min")&&i+1<argc) g_ck_min=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cache-dir")&&i+1<argc) g_ck_dir=argv[++i];
        else if(!strcmp(argv[i],"--cache-disk")&&i+1<argc) g_ck_disk_max=(size_t)atoll(argv[++i])<<20;
        else { fprintf(stderr,"usage: %s [-m gguf] [--port n] [--ctx n] [--kv-quant] [--no-auto-purge]\n"
                      "  [--temp t]      default temperature when the request omits one (default 1.0;\n"
                      "                  0 = greedy/deterministic -- recommended for agent harnesses)\n"
                      "  [--mtp [K]]     self-speculative decode for greedy requests (needs the MTP\n"
                      "                  GGUF's nextn module; draft depth K=1..3, default 1; sampled\n"
                      "                  requests use the plain path; output unchanged, just faster)\n"
                      "  [--hide-think]  suppress <think> output entirely (default: /v1/messages\n"
                      "                  emits it as a thinking content block; OpenAI wire keeps raw text)\n"
                      "  state checkpoint cache:\n"
                      "    [--no-state-cache] [--cache-ram MB (default auto: ~4 full-ctx ckpts, <=RAM/4)]\n"
                      "    [--cache-min tokens=2048] [--cache-log (per-op [ckpt] lines, default quiet)]\n"
                      "    [--cache-dir path (disk tier, opt-in)] [--cache-disk MB=32768]\n",argv[0]); return 1; }
    }
    signal(SIGPIPE,SIG_IGN);
    {   /* no SA_RESTART: accept() returns EINTR so the loop can wind down */
        struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_handler=on_term;
        sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL);
    }
    char err[256];
    if(q36_model_open(&g_m,g_model_path,err,sizeof err)){ fprintf(stderr,"open: %s\n",err); return 1; }
    q36_vocab_init(&g_vocab,&g_m.gguf);
    q36_encode_init(&g_vocab,&g_m.gguf);
    double t0=now();
    g_e=q36_engine_create(&g_m,g_ctx);
    if(kvq) q36_engine_set_kvq(g_e,1);
    g_eos=(int)g_m.eos_id;
    { int ids[4]; int k=q36_encode(&g_vocab,"<|im_end|>",ids,4); g_imend=(k==1)?ids[0]:-1; }
    /* encode buffer is 4x ctx: q36_encode clamps at cap, so an exactly-ctx
     * buffer made the auto-purge left-truncation keep the FIRST ctx tokens'
     * tail instead of the prompt's true tail.  4x gives purge a real window;
     * prompts beyond even that keep the tail of the first 4*ctx tokens. */
    g_toks=(int*)malloc((size_t)4*g_ctx*sizeof(int));
    g_cache_ids=(int*)malloc((size_t)g_ctx*sizeof(int));
    g_kv_ids=(int*)malloc((size_t)g_ctx*sizeof(int));
    g_ssm_sz=q36_engine_state_ssm_bytes(g_e);
    g_kv_psz=q36_engine_state_kv_bytes(g_e,1);
    if(!g_ck_ram){
        /* auto cap: a fixed default can't serve both short-chat and
         * long-agent hosts -- one full-context fp16 checkpoint alone is
         * ~5GB at 256k.  Budget ~4 of them so concurrent agent sessions
         * don't thrash, but never more than a quarter of system RAM. */
        size_t one=g_ssm_sz+(size_t)g_ctx*g_kv_psz, ram=0;
        FILE*f=fopen("/proc/meminfo","r");
        if(f){ char ln[128];
            while(fgets(ln,sizeof ln,f))
                if(!strncmp(ln,"MemTotal:",9)){ ram=(size_t)atoll(ln+9)<<10; break; }
            fclose(f); }
        g_ck_ram=4*one;
        if(ram&&g_ck_ram>ram/4) g_ck_ram=ram/4;
        if(g_ck_ram<(size_t)4096<<20) g_ck_ram=(size_t)4096<<20;
    }
    if(g_ck_on){
        /* build-invalidation guard for persisted blobs: state bytes are
         * layout-specific, so the disk key folds model identity (basename +
         * file size -- hashing the 20GB GGUF at startup is not worth it),
         * the KV mode, ctx, and the blob-layout version */
        const char*mb=strrchr(g_model_path,'/'); mb=mb?mb+1:g_model_path;
        struct stat stt; long long msz=stat(g_model_path,&stt)?0:(long long)stt.st_size;
        int cfg[3]={kvq,g_ctx,CK_LAYOUT_VER};
        g_ck_key=fnv64(cfg,sizeof cfg,fnv64(&msz,sizeof msz,fnv64(mb,strlen(mb),0)));
        if(g_ck_dir) dk_scan();
        fprintf(stderr,"state cache: ssm blob %.0fMB + kv %.1fKB/tok | dram %zuMB, min %d tok%s%s\n",
            g_ssm_sz/1048576.0,g_kv_psz/1024.0,g_ck_ram>>20,g_ck_min,
            g_ck_dir?" | disk ":"",g_ck_dir?g_ck_dir:"");
    }
    if(g_mtp&&!q36_engine_has_mtp(g_e)){
        fprintf(stderr,"--mtp: model has no nextn module (need the MTP GGUF); disabled\n");
        g_mtp=0;
    }
    if(g_mtp) q36_engine_set_mtp_k(g_e,g_mtp);
    fprintf(stderr,"engine ready in %.1fs | ctx %d | kv %s | default temp %.2f%s%s\n",
            now()-t0,g_ctx,kvq?"q8/mxfp4":"fp16",g_temp,g_temp<=0.f?" (greedy)":"",
            g_mtp==1?" | mtp K=1":g_mtp==2?" | mtp K=2":g_mtp==3?" | mtp K=3":"");

    int s=socket(AF_INET,SOCK_STREAM,0);
    int one=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    struct sockaddr_in a; memset(&a,0,sizeof a);
    a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons((uint16_t)port);
    if(bind(s,(struct sockaddr*)&a,sizeof a)||listen(s,128)){ perror("bind/listen"); return 1; }
    fprintf(stderr,"listening on 0.0.0.0:%d  (POST /v1/chat/completions)\n",port);
    while(!g_stop){
        int fd=accept(s,NULL,NULL);
        if(fd<0){ if(errno==EINTR)continue; perror("accept"); break; }
        setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one);
        handle_conn(fd);
        close(fd);
    }
    if(g_ck_dir&&g_ck){   /* graceful shutdown: persist warm checkpoints so a
                             restart resumes from the disk tier (SIGKILL still
                             loses whatever was DRAM-only) */
        int n=0;
        while(g_ck){ ck_spill(g_ck); n++; }
        fprintf(stderr,"[ckpt] shutdown: spilled %d checkpoint(s) to %s\n",n,g_ck_dir);
    }
    return 0;
}

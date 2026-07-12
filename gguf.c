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

#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---- cursor over the mapped bytes -------------------------------------- */

typedef struct { const uint8_t *p; const uint8_t *end; } cur;

static bool take(cur *c, void *dst, size_t n) {
    if ((size_t)(c->end - c->p) < n) return false;
    if (dst) memcpy(dst, c->p, n);
    c->p += n;
    return true;
}
static bool take_u32(cur *c, uint32_t *v){ return take(c, v, 4); }
static bool take_u64(cur *c, uint64_t *v){ return take(c, v, 8); }

/* A GGUF string is u64 length + raw bytes; we keep a pointer, no copy. */
static bool take_str(cur *c, gguf_str *s) {
    uint64_t n;
    if (!take_u64(c, &n)) return false;
    if ((uint64_t)(c->end - c->p) < n) return false;
    s->ptr = (const char *)c->p;
    s->len = n;
    c->p += n;
    return true;
}

uint64_t gguf_type_block_size(uint32_t t) {
    switch (t) {
        case 0:  return 4;    /* F32  */
        case 1:  return 2;    /* F16  */
        case 8:  return 34;   /* Q8_0 */
        case 13: return 176;  /* Q5_K */
        case 14: return 210;  /* Q6_K */
        case 39: return 17;   /* MXFP4 */
        default: return 0;
    }
}
uint64_t gguf_type_block_elems(uint32_t t) {
    switch (t) {
        case 0: case 1: return 1;
        case 8:  return 32;
        case 13: case 14: return 256;
        case 39: return 32;
        default: return 0;
    }
}
const char *gguf_type_name(uint32_t t) {
    switch (t) {
        case 0:  return "F32";  case 1:  return "F16"; case 8: return "Q8_0";
        case 13: return "Q5_K"; case 14: return "Q6_K"; case 39: return "MXFP4";
        default: return "?";
    }
}

/* Size in bytes of a single scalar of the given gguf value type (not ARR). */
static size_t gguf_scalar_size(gguf_vtype t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0; /* STR/ARR handled separately */
    }
}

/* Skip/record one metadata value payload starting at the cursor. */
static bool read_value(cur *c, gguf_kv *kv) {
    uint32_t t;
    if (!take_u32(c, &t)) return false;
    kv->type = (gguf_vtype)t;
    kv->data = c->p;
    if (t == GGUF_STR) {
        gguf_str s;
        if (!take_str(c, &s)) return false;
        kv->data_len = (const uint8_t *)c->p - (const uint8_t *)kv->data;
        return true;
    }
    if (t == GGUF_ARR) {
        uint32_t et; uint64_t n;
        if (!take_u32(c, &et) || !take_u64(c, &n)) return false;
        kv->arr_type = (gguf_vtype)et;
        kv->arr_len  = n;
        kv->data     = c->p; /* elements begin here */
        if (et == GGUF_STR) {
            for (uint64_t i = 0; i < n; i++) { gguf_str s; if (!take_str(c, &s)) return false; }
        } else {
            size_t es = gguf_scalar_size((gguf_vtype)et);
            if (es == 0) return false;
            if (!take(c, NULL, es * n)) return false;
        }
        kv->data_len = (const uint8_t *)c->p - (const uint8_t *)kv->data;
        return true;
    }
    size_t es = gguf_scalar_size((gguf_vtype)t);
    if (es == 0) return false;
    kv->data_len = es;
    return take(c, NULL, es);
}

int gguf_open(gguf_file *g, const char *path, char *err, size_t errlen) {
    memset(g, 0, sizeof(*g));
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { snprintf(err, errlen, "open %s: %s", path, strerror(errno)); return -1; }

    struct stat st;
    if (fstat(g->fd, &st) != 0) { snprintf(err, errlen, "fstat: %s", strerror(errno)); goto fail; }
    g->map_size = st.st_size;
    g->map = mmap(NULL, g->map_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->map == MAP_FAILED) { snprintf(err, errlen, "mmap: %s", strerror(errno)); goto fail; }
    /* Weights are uploaded to the GPU in one resident pass, so we stream the
     * whole file linearly.  MADV_SEQUENTIAL enables aggressive readahead (the
     * opposite of MADV_RANDOM, which disables it and turns a 20 GB load into
     * seek-bound 4 KB reads); WILLNEED kicks the readahead off immediately. */
    madvise(g->map, g->map_size, MADV_SEQUENTIAL);
    madvise(g->map, g->map_size, MADV_WILLNEED);

    cur c = { (const uint8_t *)g->map, (const uint8_t *)g->map + g->map_size };
    uint8_t magic[4];
    if (!take(&c, magic, 4) || memcmp(magic, "GGUF", 4) != 0) {
        snprintf(err, errlen, "not a GGUF file"); goto fail;
    }
    if (!take_u32(&c, &g->version) || g->version != 3) {
        snprintf(err, errlen, "unsupported GGUF version %u", g->version); goto fail;
    }
    if (!take_u64(&c, &g->n_tensors) || !take_u64(&c, &g->n_kv)) {
        snprintf(err, errlen, "truncated header"); goto fail;
    }

    g->kv = calloc(g->n_kv, sizeof(*g->kv));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (!take_str(&c, &g->kv[i].key) || !read_value(&c, &g->kv[i])) {
            snprintf(err, errlen, "bad metadata kv #%llu", (unsigned long long)i); goto fail;
        }
    }

    g->tensors = calloc(g->n_tensors, sizeof(*g->tensors));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (!take_str(&c, &t->name) || !take_u32(&c, &t->n_dims)) {
            snprintf(err, errlen, "bad tensor info #%llu", (unsigned long long)i); goto fail;
        }
        if (t->n_dims > 4) { snprintf(err, errlen, "tensor %llu n_dims %u", (unsigned long long)i, t->n_dims); goto fail; }
        for (uint32_t d = 0; d < t->n_dims; d++)
            if (!take_u64(&c, &t->dims[d])) { snprintf(err, errlen, "tensor dims"); goto fail; }
        if (!take_u32(&c, &t->ggml_type) || !take_u64(&c, &t->offset)) {
            snprintf(err, errlen, "tensor type/offset"); goto fail;
        }
    }

    /* Tensor data section is aligned. Default alignment 32 unless overridden. */
    g->alignment = 32;
    uint32_t align;
    if (gguf_u32(g, "general.alignment", &align) && align) g->alignment = align;
    size_t here = (const uint8_t *)c.p - (const uint8_t *)g->map;
    size_t pad  = (g->alignment - (here % g->alignment)) % g->alignment;
    g->data_base = (const uint8_t *)g->map + here + pad;

    /* Resolve each tensor's absolute data pointer and on-disk byte size. */
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) nelem *= t->dims[d];
        uint64_t bs = gguf_type_block_size(t->ggml_type);
        uint64_t be = gguf_type_block_elems(t->ggml_type);
        t->nbytes = (bs && be) ? (nelem / be) * bs : 0;
        t->data   = g->data_base + t->offset;
        if ((const uint8_t *)t->data + t->nbytes > (const uint8_t *)g->map + g->map_size) {
            snprintf(err, errlen, "tensor %.*s runs past EOF",
                     (int)t->name.len, t->name.ptr); goto fail;
        }
    }
    return 0;

fail:
    gguf_close(g);
    return -1;
}

void gguf_close(gguf_file *g) {
    if (g->map && g->map != MAP_FAILED) munmap(g->map, g->map_size);
    if (g->fd >= 0) close(g->fd);
    free(g->kv);
    free(g->tensors);
    memset(g, 0, sizeof(*g));
    g->fd = -1;
}

static bool key_eq(const gguf_str *k, const char *s) {
    size_t n = strlen(s);
    return k->len == n && memcmp(k->ptr, s, n) == 0;
}

const gguf_kv *gguf_find(const gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (key_eq(&g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

bool gguf_u32(const gguf_file *g, const char *key, uint32_t *out) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv) return false;
    if (kv->type == GGUF_U32 || kv->type == GGUF_I32) { memcpy(out, kv->data, 4); return true; }
    if (kv->type == GGUF_U64 || kv->type == GGUF_I64) { uint64_t v; memcpy(&v, kv->data, 8); *out=(uint32_t)v; return true; }
    return false;
}
bool gguf_u64(const gguf_file *g, const char *key, uint64_t *out) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv) return false;
    if (kv->type == GGUF_U64 || kv->type == GGUF_I64) { memcpy(out, kv->data, 8); return true; }
    if (kv->type == GGUF_U32 || kv->type == GGUF_I32) { uint32_t v; memcpy(&v, kv->data, 4); *out=v; return true; }
    return false;
}
bool gguf_f32(const gguf_file *g, const char *key, float *out) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type != GGUF_F32) return false;
    memcpy(out, kv->data, 4); return true;
}
bool gguf_str_dup(const gguf_file *g, const char *key, char *out, size_t cap) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type != GGUF_STR) return false;
    gguf_str s; const uint8_t *p = kv->data; uint64_t n; memcpy(&n, p, 8);
    s.ptr = (const char *)p + 8; s.len = n;
    size_t m = s.len < cap - 1 ? s.len : cap - 1;
    memcpy(out, s.ptr, m); out[m] = 0;
    return true;
}

const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (key_eq(&g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

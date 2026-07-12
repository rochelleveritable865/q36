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

#ifndef Q36_GGUF_H
#define Q36_GGUF_H

/* Minimal GGUF v3 reader.  mmap-backed: metadata and the tensor directory are
 * parsed eagerly, but tensor *data* is never copied here -- callers get a
 * pointer into the mapping and upload to the GPU themselves.  This keeps model
 * open O(header) instead of O(21 GB). */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GGUF_U8=0, GGUF_I8=1, GGUF_U16=2, GGUF_I16=3, GGUF_U32=4, GGUF_I32=5,
    GGUF_F32=6, GGUF_BOOL=7, GGUF_STR=8, GGUF_ARR=9, GGUF_U64=10, GGUF_I64=11,
    GGUF_F64=12,
} gguf_vtype;

typedef struct { const char *ptr; uint64_t len; } gguf_str; /* not NUL-terminated */

typedef struct {
    gguf_str key;
    gguf_vtype type;
    gguf_vtype arr_type;    /* valid when type==GGUF_ARR */
    uint64_t   arr_len;     /* valid when type==GGUF_ARR */
    const void *data;       /* pointer to the value bytes inside the mapping */
    uint64_t    data_len;   /* size of the value payload in bytes */
} gguf_kv;

typedef struct {
    gguf_str name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t offset;        /* offset within the tensor data section */
    const void *data;       /* absolute pointer into the mapping (filled at open) */
    uint64_t nbytes;        /* size of the tensor data on disk */
} gguf_tensor;

typedef struct {
    int fd;
    void   *map;
    size_t  map_size;

    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    gguf_kv     *kv;
    gguf_tensor *tensors;

    const uint8_t *data_base;   /* start of the aligned tensor data section */
    uint32_t alignment;
} gguf_file;

/* Open and parse. Returns 0 on success, negative on error (msg filled). */
int  gguf_open(gguf_file *g, const char *path, char *err, size_t errlen);
void gguf_close(gguf_file *g);

/* Metadata accessors. Return false if key missing or wrong type. */
const gguf_kv *gguf_find(const gguf_file *g, const char *key);
bool gguf_u32(const gguf_file *g, const char *key, uint32_t *out);
bool gguf_u64(const gguf_file *g, const char *key, uint64_t *out);
bool gguf_f32(const gguf_file *g, const char *key, float *out);
bool gguf_str_dup(const gguf_file *g, const char *key, char *out, size_t cap);

/* Tensor lookup by exact name (e.g. "blk.3.attn_q.weight"). */
const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name);

/* Bytes-per-block / block element count for the quant types we support. */
uint64_t gguf_type_block_size(uint32_t ggml_type);   /* bytes per block */
uint64_t gguf_type_block_elems(uint32_t ggml_type);  /* elems per block */
const char *gguf_type_name(uint32_t ggml_type);

#ifdef __cplusplus
}
#endif

#endif

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

#ifndef Q36_TOKENIZER_H
#define Q36_TOKENIZER_H
#include "gguf.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Byte-level BPE vocab view over the GGUF.  Encode is delegated to an external
 * reference tokenizer for exact correctness during bring-up; decode is native
 * (id -> token string -> byte-level unmap -> UTF-8). */
typedef struct {
    int n;
    const char **tok;   /* pointer into the gguf mapping (not NUL-terminated) */
    int *len;
} q36_vocab;

int  q36_vocab_init(q36_vocab *v, const gguf_file *g);
void q36_vocab_free(q36_vocab *v);
/* Append the decoded UTF-8 text for one token id to buf (bounded). Returns
 * number of bytes written. */
int  q36_detok_one(const q36_vocab *v, int id, char *buf, int cap);
/* native byte-level BPE encode (init once after vocab_init) */
int  q36_encode_init(q36_vocab *v, const gguf_file *g);
int  q36_encode(const q36_vocab *v, const char *text, int *out, int cap);

#ifdef __cplusplus
}
#endif
#endif

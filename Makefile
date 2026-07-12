# This file is part of q36, a dedicated CUDA inference engine for
# Qwen3.6-35B-A3B.
#
# Copyright (C) 2026 Ambud Sharma
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this program.  If not, see
# <https://www.gnu.org/licenses/>.

# Qwen3.6-35B-A3B dedicated inference engine.
#
# Two build tiers:
#   make tools   -> CPU-only loader/binder/dequant validation (no GPU needed)
#   make all     -> every binary: CPU tools + CUDA engine, benchmark, server,
#                   perplexity harness, MMA harness (needs nvcc + a GPU)
#
# CUDA arch: target is RTX 5090 (Blackwell, sm_120) -> tested on CUDA 13.1.
# Override ARCH=... to build for another GPU.

# parallel by default: shared objects below compile concurrently
MAKEFLAGS += -j$(shell nproc)

CC      ?= cc
NVCC    ?= nvcc
CFLAGS  ?= -O3 -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation
ARCH    ?= -gencode arch=compute_120a,code=sm_120a
# per-thread default stream makes <<<>>> launches capturable by CUDA graphs
NVFLAGS ?= -O3 -std=c++17 $(ARCH) --use_fast_math -lineinfo -Xcompiler -fopenmp \
           --default-stream per-thread

# engine translation units compile once to objects shared by every binary
CORE_OBJ = q36_cuda.o q36_engine.o gguf.o q36_model.o

# ---- CPU validation tools (build + run on this box) ----------------------
.PHONY: tools
tools: gguf_dump q36_info test_dequant

.PHONY: all
all: tools q36 q36_bench q36_server q36_ppl test_mma_bs

gguf_dump: gguf_dump.c gguf.c
	$(CC) $(CFLAGS) $^ -o $@
q36_info: q36_info.c q36_model.c gguf.c
	$(CC) $(CFLAGS) $^ -o $@
test_dequant: test_dequant.c gguf.c
	$(CC) $(CFLAGS) -Wno-unused-function $^ -lm -o $@

.PHONY: test
test: test_dequant q36_info
	./test_dequant $(MODEL)
	./q36_info $(MODEL)

# ---- full CUDA engine (build on the GPU server) --------------------------
# nvcc compiles the .c files as C and the .cu as C++/CUDA; each unit builds
# to an object once (in parallel) and the binaries just link.
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o: %.c
	$(NVCC) $(NVFLAGS) -c $< -o $@

# any header change rebuilds every object (coarse but always correct)
$(CORE_OBJ) tokenizer.o main.o bench.o server.o ppl.o: $(wildcard *.h *.cuh)

q36: main.o tokenizer.o $(CORE_OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@

# llama-bench-style throughput benchmark (raw synthetic context, no tokenizer)
q36_bench: bench.o $(CORE_OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@

# OpenAI-compatible HTTP server (zero-dependency; prefix cache, SSE streaming)
q36_server: server.o tokenizer.o $(CORE_OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@

# perplexity harness (llama-perplexity-compatible windowing)
q36_ppl: ppl.o tokenizer.o $(CORE_OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@

# hardware validation of the sm_120a block-scaled MMA fragment/scale mappings
# (source of the calibrated layouts the engine's MMA kernels rely on)
test_mma_bs: test_mma_bs.cu
	$(NVCC) $(NVFLAGS) test_mma_bs.cu -o $@

.PHONY: clean
clean:
	rm -f gguf_dump q36_info test_dequant test_mma_bs q36 q36_bench q36_server q36_ppl *.o

MODEL ?= /path/to/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

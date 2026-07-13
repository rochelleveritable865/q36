# q36 — a dedicated CUDA inference engine for Qwen3.6-35B-A3B (MXFP4)

A single-model, zero-dependency C/CUDA inference engine targeting
**Qwen3.6-35B-A3B** in its MXFP4 GGUF form, purpose-built for one GPU. The
goal is not generality — it is to test one hypothesis:

> **Can a model-specific engine beat generic engines (llama.cpp/vLLM/SGLang)
> by specializing to this exact architecture, quant format, and one GPU?**

**Answer so far: yes — every end-to-end workload and every prefill/decode
point except decode at 90k depth** (see benchmarks below).

Target hardware: **RTX 5090 (32 GB) / RTX 6000 Pro (96 GB), sm_120a (Blackwell)**, CUDA 13.1 (tested; earlier 12.x with sm_120a support is untested). (Note: While architecturally compatible with the RTX 6000 Pro Blackwell, the engine has not yet been physically tested on it.)
No cuBLAS, no cuDNN, no frameworks: the quant formats live *inside* the
kernels; weights are never materialized to fp16/fp32. For the systems
reasoning behind these choices (roofline math, the Blackwell MMA path,
hybrid-state caching), see [ARCHITECTURE.md](ARCHITECTURE.md).

## Benchmarks (q36_bench, greedy, 1 GPU, fp16 KV)

| test | q36 | llama.cpp `-fa 1` (b9954, best `-b/-ub`) |
|---|---:|---:|
| pp2048 | **13,665 t/s** | 10,836 |
| pp8192 | **12,615** | 10,596 |
| pp32768 | **10,665** | 9,400 |
| pp90112 | **7,843** | 7,112 |
| tg128 | **294.4** | 280.5 |
| tg128 @ d32768 | **251.5** | 246.2 |
| tg128 @ d90112 | 199.5 | **208.3** |

```
prefill t/s                                       q36 ▓   llama.cpp ░
pp2048    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  13,665
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  10,836
pp8192    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  12,615
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  10,596
pp32768   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  10,665
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░  9,400
pp90112   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  7,843
          ░░░░░░░░░░░░░░░░░░░░  7,112

decode t/s at depth
tg128     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  294.4
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  280.5
@ d32768  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  251.5
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  246.2
@ d90112  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  199.5
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  208.3
```

Prefill leads at every point (+10% to +26%); decode leads flat (+5.0%)
and at 32k depth (+2.2%), and trails by 4% at 90k. The opt-in
mixed-precision head (`Q36_MIX_HEAD=32768`) adds a further ~+1.7% decode
for +0.24% ppl.
End-to-end (prompt + generation together) q36 wins every measured
workload: short chat (2k+200tok) +9%, 8k+500tok +9%, 32k+300tok +10%,
90k+2000tok +4% -- before counting the server prefix cache on
multi-turn. Both columns measured back-to-back on the same idle 5090
(clocked at a 400W power limit), single GPU, identical GGUF.

`pp` = prompt processing (prefill), `tg` = token generation at a context
depth. Exact commands (2026-07-13):

```sh
./q36_bench -m <model.gguf> -p 2048,8192,32768,90112 -n 128 -d 0,32768,90112 -r 3

llama-bench -m <model.gguf> -fa 1 -p 2048,8192,32768,90112 -n 0 -b 2048 -ub 512,1024,2048 -r 3
llama-bench -m <model.gguf> -fa 1 -p 0 -n 128 -d 0,32768,90112 -r 3
```

The llama.cpp prefill column takes its best result per row across the
`-ub` sweep (`-ub 2048` at 8k/32k, `-ub 1024` at 2k/90k; `-b 1024 -ub
1024` was also measured and never best; decode is batch-1 and
insensitive to `-b/-ub`). Model loads in ~5.5s warm; ~26 GB VRAM
resident; 90k-context retrieval quality verified in both KV modes.

## What this model actually is (verified from the GGUF)

`arch = qwen35moe` — a **hybrid linear-attention (gated DeltaNet / SSM) +
full-attention MoE**, not a plain transformer:

| Property | Value |
|---|---|
| Blocks | 40 (**10 full-attention + 30 SSM**, `full_attention_interval=4`) |
| d_model / vocab / ctx | 2048 / 248320 / 262144 |
| Attention | GQA 16 Q / 2 KV heads, head_dim 256, q/k RMSNorm, fused output gate, partial mRoPE (rot 64, base 1e7) |
| SSM block | fused in-proj → causal conv1d (k=4) → gated-delta scan (state 128, 16 groups, dt-rank 32, inner 4096) → group RMSNorm → gate → out-proj |
| MoE (every block) | F32 router → top-8 of 256 experts, expert FFN width 512 SwiGLU, + 1 shared expert (sigmoid-gated) |
| Quant mix | attn/shared/embed/output = **Q8_0**, routed gate/up = **MXFP4**, routed down = **Q5_K** (Q8-requantized at load), norms/router/ssm = F32 |

The hybrid SSM design is why this is a good hypothesis test: generic engines
are least mature on linear-attention/SSM scheduling, so the headroom is real.
Batch=1 decode is memory-bound at **2.544 GiB active bytes/token** → the
5090's 1.79 TB/s puts the roofline at ~656 t/s (realistic ~426); prefill is
where specialization pays hardest.

## Build

```sh
# On any box (no GPU): loader/binder/dequant validation
make tools
make test MODEL=/path/to/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

# On the 5090 box (CUDA 13.1, nvcc on PATH):
make all       # everything below plus the CPU tools
make q36       # chat CLI / REPL
make q36_bench    # llama-bench-style benchmark
make q36_server   # OpenAI-compatible HTTP server
make q36_ppl      # perplexity harness (llama-perplexity-compatible)
make test_mma_bs  # hardware validation of the block-scaled MMA mappings
```

> [!IMPORTANT]
> The CUDA build requires targeting `compute_120a` / `sm_120a` (e.g., `-gencode arch=compute_120a,code=sm_120a`). The `a` suffix is strictly required because plain `sm_120` does not expose the Blackwell block-scaled MMA instructions, causing compilation to fail. Additionally, `--default-stream per-thread` is mandatory to support CUDA-graph capture.

## Usage

### Chat CLI

```sh
./q36 [--ctx 8192] [-n 256] [--temp 1.0 --top-k 20 --top-p 0.95 --seed 42]
         [--kv-quant] [--gpus N] [-m model.gguf]
```

Interactive REPL; reads a question per line, streams the answer, prints
prefill/decode tok/s. Defaults to greedy (temp 0) so regressions are
deterministic; `--temp 1.0 --top-k 20 --top-p 0.95` are the
GGUF-recommended sampling settings. `--kv-quant` stores K as Q8 and V as
MXFP4 (2.5× smaller KV; decode wins beyond ~30k context, costs ~5% below).

### Benchmark

```sh
./q36_bench -p 2048,16384,90112 -n 128 -d 0,32768 -r 3 [--kv-quant]
```

Raw synthetic token context (no tokenizer, content-independent), markdown
table out, mean ± std over `-r` runs.

### OpenAI-compatible server

```sh
./q36_server [--port 8080] [--ctx 32768] [--kv-quant] [--auto-purge]
             [--cache-ram MB] [--cache-min tokens] [--cache-dir path] [--no-state-cache]
```

- `POST /v1/chat/completions` — messages → Qwen chat template; SSE
  streaming (`"stream": true`) and non-streaming; `temperature`, `top_p`,
  `top_k`, `max_tokens`, `seed`, `stop` honored; responses carry `usage`
  and `timings` (incl. `cached_tokens`).
- `POST /v1/completions` — raw prompt, no template.
- `POST /v1/messages` (+ `count_tokens`) — Anthropic Messages API; the
  model's leading `<think>` region is emitted as a real `thinking` content
  block (whitespace-only regions are omitted, like the real API), tool
  calls as `tool_use` blocks. `--hide-think` suppresses the think region
  on both protocols.
- `--temp t` — default temperature when the request omits one (default
  1.0; `0` = greedy — recommended under agent harnesses, which rarely
  send a temperature).
- `GET /v1/models`, `GET /health`; CORS enabled.
- `scripts/server_smoke.sh` — end-to-end wire-format regression suite
  (both protocols, streaming grammar, tool calls, think handling).
- **Prefix cache**: the server tracks the exact token ids in the KV/SSM
  state; a prompt that strictly extends them (the normal multi-turn case)
  prefills only the delta. The hybrid SSM state cannot rewind, so anything
  short of strict extension needs a checkpoint (below) or a re-prefill.
- **State checkpoint cache**: the engine
  state (attention KV pages + the ~63MB SSM/DeltaNet blob) is snapshotted
  at each end-of-prompt boundary, keyed by the prompt tokens, and the
  longest stored prefix is restored when the live state diverges — the
  normal agentic-loop case, where clients resend a *reconstructed* history
  (think stripped, tool calls re-rendered) that never matches generated
  text. Measured on a 78.7k-token turn: 13.6s re-prefill → 0.4s (restore +
  delta). Checkpoints live in DRAM (LRU, `--cache-ram MB`; default auto:
  ~4 full-context checkpoints, capped at a quarter of system RAM, floor
  4GB), are gated on prompt length (`--cache-min tokens`, default 2048), and
  optionally tier to disk (`--cache-dir path`, capped by `--cache-disk MB`)
  including across graceful restarts. `--no-state-cache` disables;
  `--cache-log` enables per-operation `[ckpt]` log lines (default quiet —
  the per-request line already reports the cached token count).
- `--auto-purge`: when a prompt would overflow `--ctx`, oldest non-system
  messages are dropped (chat) or tokens left-truncated (completions);
  without it, oversized prompts get a 400.
- Requests are served sequentially (the engine is single-stream by design).

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Project layout

| File | Role |
|---|---|
| `gguf.[ch]` | mmap GGUF v3 loader (validated against the real 20.2 GiB file) |
| `q36.h` | fixed model shape + quant block layouts, compile-time constants |
| `q36_model.[ch]` | weight binding + hybrid-schedule validation (733 tensors) |
| `q36_dequant.cuh` | ggml-exact Q8_0/MXFP4/Q5_K/Q6_K dequant (CPU-testable) |
| `q36_ops.cuh` | reference/decode kernels: split-quant GEMV cores, warp reductions |
| `q36_cuda.cu` | dequant-matvec host API, argmax, small kernels |
| `q36_engine.cu` | the engine: upload/repack, CUDA-graph decode, tensor-core FA prefill, W4A8 + int8 MMA GEMMs, gated-DeltaNet scan, MoE dispatch, multi-GPU driver, on-device sampler |
| `tokenizer.[ch]` | native byte-level BPE, both directions, parity-verified |
| `main.c` / `bench.c` / `server.c` | chat CLI / benchmark / OpenAI server |
| `test_dequant.c`, `q36_info.c`, `gguf_dump.c` | CPU-only validation tools (`make tools`) |
| `ppl.c` | perplexity harness (`q36_ppl`), on-device NLL scoring |
| `test_mma_bs.cu` | sm_120a block-scaled MMA calibration harness (bit-exact vs CPU) |

### Accuracy validation

```sh
./q36_ppl -f wiki.test.raw [--window 512] [--kv-quant] [--max-windows N]
```

Replicates llama-perplexity's methodology (independent windows, second
halves scored) so the same GGUF through llama.cpp is a direct reference.
Measured on WikiText-2-raw test (580 windows, same GGUF, pinned GPU):

| engine | PPL |
|---|---:|
| llama.cpp `-fa 1` | 6.8086 ± 0.044 |
| q36 (fp16 KV) | **6.7541** |
| q36 `--kv-quant` | 6.7908 |

Per-window trajectories match llama.cpp to ~0.01-0.05 nats: the fp16
tensor-core FA, FP8-e4m3 output head, and Q5K→Q8 expert requant cost
nothing measurable; kv-quant costs +0.54%.  Full run ~110s = release gate.

## Engineering notes

- **Correctness gate**: every kernel change is validated by answer checks
  (greedy answers must stay exact) before speed is even measured; wrong-but-
  fast kernels look plausible. The gate commands are in
  [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Decode** runs as a single captured CUDA graph with device-resident
  token/position/argmax (zero host syncs per token).
- **Prefill** is chunked (2048 tokens), all GEMMs on tensor cores: W4A8
  block-scaled MMA (`kind::mxf8f6f4`, MXFP4 weights fed natively), int8 MMA
  for Q8 paths, and a custom FlashAttention-2 kernel (`k_attn_pf3`) with
  register-resident Q fragments, online softmax on the mma C-fragments, and
  a single-buffer intra-tile cp.async pipeline. The fragment/scale mappings
  are hardware-calibrated via the `test_mma_bs` harness.
- Multi-GPU expert parallelism works but is a net loss on GeForce (no P2P);
  the `--gpus N` path is kept for P2P-capable hardware.
- Env toggles: `Q36_DEBUG`, `Q36_NOGRAPH`, `Q36_NOPDL` (disable
  programmatic dependent launch in the decode graph), `Q36_MEGA`, `Q36_NOFORK`,
  `Q36_RSTAT`, `Q36_ENCTEST`, `Q36_PF2` (pre-tensor-core prefill attention
  fallback for A/B).

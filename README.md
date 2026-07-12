# q36

A hyper-optimized, zero-dependency C/CUDA inference engine for Qwen 3.6 35B on RTX 5090 / Blackwell.

A zero-dependency C/CUDA codebase specialized for a single model + GPU pairing — Qwen3.6-35B-A3B (MXFP4 GGUF) on RTX 5090 (sm_120a) — on the bet that a dedicated engine beats generic runtimes (llama.cpp / vLLM / SGLang) on their long tail. **Status: faster than llama.cpp at every measured point.**

### Key Features

* 🚀 **Extreme Prefill Throughput**: Tensor-core FlashAttention prefill reaching **13.4k tokens/sec** (at context depth 2,048, measured on a GPU clocked at 400W) and **7.6k tokens/sec** (at context depth 90k).
* ⚡ **Fast Decode**: Native token generation speeds of **270+ tokens/sec** using captured CUDA graphs with zero host syncs.
* 🧠 **Hybrid Architecture Support**: Native CUDA implementations for both 10 full-attention layers (using W4A8 block-scaled MMA MoE) and 30 recurrent SSM layers (gated-DeltaNet scans).
* 💾 **Dual-Tier State Management**: Zero-overhead VRAM saving (**2.5× smaller KV cache** via `--kv-quant`) and DRAM/Disk state checkpoint caching (restoring context states in **3ms**).
* 🌐 **OpenAI-Compatible Server**: A zero-dependency, prefix-cached HTTP server (`q36_server`) supporting SSE streaming and tool calling.
* 🛠️ **Developer Tooling**: Built-in benchmark tools (`q36_bench`), perplexity evaluation harnesses (`q36_ppl`), and CPU-only validation helpers.

## Quick Start

### 1. Prerequisites
* **GPU**: An NVIDIA RTX 5090 or RTX 6000 Pro Blackwell GPU (Blackwell architecture, `sm_120a` / compute capability 12.0+ is required). *Note: The engine is architecturally compatible with the RTX 6000 Pro Blackwell (and benefits from its 96 GB VRAM), but has not yet been physically tested on it.*
* **Software**: Linux with [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-downloads) installed — the version the engine is developed and tested on (12.x releases that expose `sm_120a` may work but are untested). Ensure the CUDA compiler (`nvcc`) is on your system `PATH`.
* **Model**: Download the model weights in MXFP4 GGUF format from [Hugging Face: unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF).

### 2. Build & Validate
Run the following commands from the repository root:

```bash
# Compile and run validation tests on CPU (no GPU required):
make tools
./q36_info /path/to/model.gguf

# Compile the full GPU-accelerated engine, benchmark, and OpenAI server:
make q36 q36_bench q36_server
```

### 3. Usage
* **Interactive CLI**: Run the interactive chat shell:
  ```bash
  ./q36 -m /path/to/model.gguf
  ```
* **OpenAI Server**: Start the HTTP API server with state caching enabled:
  ```bash
  ./q36_server -m /path/to/model.gguf --port 8080 --ctx 32768
  ```
* **Benchmark**: Run throughput tests:
  ```bash
  ./q36_bench -m /path/to/model.gguf
  ```

### 4. KV Cache & State Management (DRAM / VRAM)

Due to the hybrid model architecture (10 attention layers + 30 recurrent SSM layers), `q36` optimizes KV cache and state memory at two levels:

#### VRAM KV Cache Quantization
By default, attention keys and values are stored in FP16. Toggle quantization with the `--kv-quant` flag:
* **Mechanism**: Quantizes keys to **Q8_0** (8-bit) and values to **MXFP4** (4-bit block-scaled FP4).
* **VRAM savings**: Reduces active VRAM KV cache size by **2.5×**, freeing up space for long contexts and larger batches.
* **Performance trade-offs**:
  * *Short contexts*: Adds small kernel overhead (~5% slower decoding).
  * *Long contexts (30k+ tokens)*: Speeds up generation. Since the quantized cache transfers 40% of the bytes of FP16, it overcomes memory-bandwidth bottlenecks.
  * *Accuracy cost*: Negligible perplexity increase (+0.54%).

#### DRAM & Disk State Caching (Offloading)
Standard prefix caching fails in multi-turn agent/tool loops when prompts are reconstructed and resent. `q36_server` provides a **State Checkpoint Cache** to offload recurrent states and KV pages:
* **Mechanism**: At each prompt end, the server snapshots the live engine state (attention KV + the ~63MB recurrent SSM state) to system DRAM. If the next prompt shares a prefix, the state is restored from DRAM in **3ms** instead of triggering a multi-second re-prefill.
* **Configuration Flags**:
  * `--cache-ram MB`: Memory allocated for DRAM cache (defaults to auto: ~4 full-context checkpoints, capped at 25% of system RAM, floor 4GB).
  * `--cache-min tokens`: Minimum prefix token length to trigger checkpointing (default: `2048`).
  * `--cache-dir path`: Local directory path to write evicted checkpoints to disk (LRU tiering). Checkpoints persisted here survive server restarts.
  * `--no-state-cache`: Disables checkpoint caching completely.

### 5. Multi-Slot Capabilities & Continuous Batching

The underlying `q36` engine supports high-throughput, multi-tenant execution using **continuous batching** and **slot management** primitives. While the OpenAI HTTP server (`q36_server`) currently processes requests sequentially (on Slot 0), the core library exposes a full multi-tenant scheduler API:

#### Multi-Slot Engine API
* `q36_engine_create_mt(model, max_ctx, n_slots)`: Instantiates an engine with `n_slots` concurrent sequence states. Attention KV cache pages and recurrent SSM state blocks are separate per slot.
* `q36_engine_prefill_slot(engine, slot, tokens, len, pos0)`: Processes prompts into a specific slot.
* `q36_engine_step_active(engine, n_active, active_tokens, positions, out_tokens)`: Performs a single batched decode step for all `n_active` slots. To avoid graph launch overheads, active slots are padded/bucketed to `{8, 16, 32, 48, 64}` to reuse captured CUDA graphs.
* `q36_engine_slot_move(engine, dst_slot, src_slot)`: Instantly relocates a slot's entire history (attention KV, SSM state, convolution buffers) in memory (~40us overhead). This allows **slot compaction** upon request eviction to maintain a contiguous active slot block.

#### Dynamic Dual Decode Paths
The engine dynamically switches decoding strategies based on the active batch size:
1. **GEMV Mode (Batch < 16)**: Lowest latency; serial state updates.
2. **Batch-Tiled Mode (Batch >= 16)**: Uses tensor cores and tiled GEMM kernels to load and share weight matrices across all active sequences, achieving aggregate throughput of up to **1,653 tokens/sec** at Batch=64. The engine automatically transitions between these modes, which can also be forced via `Q36_MT_GEMV` / `Q36_MT_TILED`.

#### Running Continuous Batching Benchmarks
You can simulate staggered request arrival, execution, and eviction (utilizing slot compaction under churn) via the benchmark tool:
```bash
# Run a continuous batching simulation with up to 48 concurrent slots:
Q36_CB=48 Q36_CB_NREQ=256 ./q36_bench -m /path/to/model.gguf
```

---

## Code Navigation & Documentation Links

### Core Implementation
* [q36_engine.cu](q36_engine.cu): Core engine loop, CUDA graph setup, Blackwell W4A8 MMA kernels, and FlashAttention.
* [q36_dequant.cuh](q36_dequant.cuh): Quantization formats, layouts, and CPU-parallelized dequantization.
* [tokenizer.c](tokenizer.c): Native BPE tokenizer implementation.
* [server.c](server.c): Zero-dependency OpenAI-compatible HTTP server.

### Architecture Guides
* [ARCHITECTURE.md](docs/ARCHITECTURE.md): Systems deep dive — decode roofline analysis, Blackwell block-scaled MMA, the SSM/attention caching asymmetry, multi-GPU trade-offs.
* [ENGINE.md](docs/ENGINE.md): Detailed engine reference — benchmarks, CLI/server flags, project layout.

---

## License and Third-Party Attribution

This project is licensed under the [AGPL-3.0 License](LICENSE). If you run a modified version of these engines as a network service, you must make your modified source available to its users.

Copyright is retained by [Ambud Sharma](https://github.com/ambud). The AGPL applies to everyone else's use of this code; Ambud Sharma reserves the right to offer this software under other license terms (dual licensing).

Contributions require the [Contributor License Agreement](CLA.md) (see [CONTRIBUTING.md](CONTRIBUTING.md)), preserving the project's ability to dual-license.

### Third-Party Software

This codebase contains third-party components adapted from **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (licensed under the **MIT License**):
* Dequantization constants, block layout structures, and scaling logic in [q36_dequant.cuh](q36_dequant.cuh).
* Perplexity calculation windowing strategies in [ppl.c](ppl.c).

The full MIT License text and copyright notices for these components are preserved in the respective source files.

---

## Author

Built by [Ambud Sharma](https://github.com/ambud) — connect with me on [LinkedIn](https://www.linkedin.com/in/ambud/).

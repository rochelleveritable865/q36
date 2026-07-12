# q36 Architecture

This document explains the systems engineering behind q36: why the
performance numbers are what they are, and the three or four design
problems that make a dedicated engine for this model genuinely different
from a generic runtime. For building and running the engine, see
[ENGINE.md](ENGINE.md).

Qwen3.6-35B-A3B is a hybrid model: 40 blocks, of which 10 are
full-attention and 30 are recurrent (gated DeltaNet, an SSM variant),
every block carrying a 256-expert MoE FFN. Each of the sections below is
a direct consequence of that architecture meeting one specific GPU — the
RTX 5090 (Blackwell, `sm_120a`, 32 GB, 1.79 TB/s).

---

## 1. Decode roofline: where the ceiling is and how close the engine gets

Single-stream token generation is memory-bound: to produce one token, the
GPU must read every weight byte that participates in the forward pass
exactly once, and at batch 1 there is no reuse to hide behind. So the
first question for any decode optimization is simply: *how many bytes is
that, and how fast can this card move them?*

For this model in its shipped quantization mix (reported by `q36_info`,
computed from the actual GGUF tensors):

| component | bytes/token | share |
|---|---:|---:|
| Q8_0 dense (attention + SSM + shared expert) | 1492.9 MB | 54.7% |
| Q8_0 output head (248320-token vocab) | 540.3 MB | 19.8% |
| MXFP4 routed experts (top-8 of 256) | 347.6 MB | 12.7% |
| Q5_K/Q6_K routed experts | 246.7 MB | 9.0% |
| F32 (router / conv / SSM gates) | 103.5 MB | 3.8% |
| **total active bytes per token** | **2.544 GiB** | |

At the RTX 5090's 1.79 TB/s of memory bandwidth, 2.544 GiB per token puts
the absolute physical ceiling at **≈656 tokens/s**. No engine can pass
this without changing the bytes (deeper quantization, speculation);
well-optimized kernels on real workloads typically sustain ~65% of peak
bandwidth, i.e. a realistic ceiling around 426 t/s.

q36 measures **272 t/s** — 41% of the absolute roofline, 64% of the
realistic one. The decode path that achieves this:

- **One captured CUDA graph per token.** The entire 40-block forward
  pass, including sampling, is captured once and replayed; per-token
  launch overhead is a single graph launch.
- **Zero host synchronization.** Token ids, positions, and the
  sampled/argmax result live in device memory; the host never blocks on
  the GPU inside the generation loop.
- **On-device sampling.** Temperature/top-k/top-p run as kernels, so
  sampled generation keeps the same zero-sync property as greedy.

The roofline also explains why this repository's larger wins are in
*prefill* (see §2): decode can only close the remaining gap to the
ceiling, while prefill throughput is compute-bound and rewards kernel
specialization much more.

### Batch scaling: amortizing the weight read

The moment more than one sequence decodes concurrently, the economics
change: the same 2.544 GiB weight read can serve B tokens if the kernels
share it. The multi-slot engine switches strategies on batch size:

- **GEMV mode (batch < 16):** each weight row is dotted against the
  active batch's vectors while it sits in L1. Lowest latency; serial
  state updates.
- **Batch-tiled mode (batch ≥ 16):** tiled tensor-core GEMMs load weight
  tiles once and apply them across all active sequences. Aggregate
  throughput reaches ~1,650 t/s at batch 64.

Active slots are padded to bucket sizes {8, 16, 32, 48, 64} so each
bucket reuses a pre-captured CUDA graph instead of re-capturing per batch
size, and a ~40 µs slot-move primitive compacts live slots when requests
finish, keeping the active set contiguous.

---

## 2. Blackwell `sm_120a`: feeding MXFP4 to the tensor cores natively

The model's routed experts ship as MXFP4: E2M1 4-bit codes with one
shared UE8M0 exponent per 32 weights. Most engines dequantize such
weights to fp16 (in registers or shared memory) before every matrix
multiply. Blackwell removes that step: consumer `sm_120a` exposes
*block-scaled* MMA instructions that consume fp4 codes and their scale
factors directly.

The instruction q36 builds its expert prefill GEMM on:

```
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
    .f32.e2m1.e4m3.f32.ue8m0
```

Mixed operand types are the key feature: the A operand is E2M1 (the MXFP4
weights, fed with **zero conversion** — the GGUF's nibbles-plus-per-32-
scale layout is exactly the hardware's format), while B is E4M3 (fp8), so
only the activations need on-the-fly quantization, and to fp8 rather than
the quality-riskier fp4. This is the standard W4A8 recipe, executed
entirely inside the tensor core. (A pure-fp4 W4A4 form, `kind::mxf4` at
m16n8k64, exists and is faster still, but quantizing *activations* to
fp4 is a measurable quality risk; q36 ships the W4A8 path.)

Getting the instruction to produce correct numbers required establishing
a few facts about the hardware that the documentation leaves easy to get
wrong, all validated bit-exactly against a CPU reference by the
`test_mma_bs.cu` harness in this repository:

- **The E2M1 code sits at bits [5:2] of its byte container** (an
  "E2M3-aligned" position): each 4-bit code `c` must be expanded to the
  byte `((c&7)<<2) | ((c>>3)<<5)` before staging. Feed raw low nibbles
  and the results are silently wrong.
- **Scale-selector operands must be immediates.** The `{byte-id,
  thread-id}` selectors that pick which lane supplies scale factors do
  not accept registers.
- **Per-lane scale mapping is fixed and non-obvious:** the scale for A
  row *q* comes from lane 4*q* (row *q*+8 from lane 4*q*+1), and for B
  column *c* from lane 4*c*, byte 0.

Because the engine owns weight loading, all of this is resolved at load
time: expert weights are pre-swizzled into the exact per-thread fragment
layout the MMA expects, so the kernel's inner loop does plain 128-bit
vector loads — no runtime permutes, no dequantization stage, and half
the shared-memory traffic of a dequantize-to-fp16 pipeline. This is the
core of the engine's prefill lead (13.4k t/s at pp2048 vs 9.4k for
llama.cpp on the same GPU and GGUF).

### Hardware FP4→FP16 conversion on the decode path

Decode-side GEMV cannot use the MMA form efficiently (a batch-1 vector
wastes 7/8 of an n=8 tile), so fp4 weights on the decode path are
expanded to fp16 — and Blackwell helps here too: `cvt.rn.f16x2.e2m1x2`
converts two E2M1 codes held in a `.b8` register to two fp16 values in a
single instruction. In a memory-bound kernel the ALU is not free —
a multi-instruction shift/mask/select decode per nibble eats the
instruction-issue headroom needed to keep loads in flight. The hardware
converter is what makes the opt-in mixed-precision fp4 output head
(`Q36_MIX_HEAD`) a net win: +1.7% end-to-end decode for +0.24%
perplexity.

---

## 3. The caching asymmetry: attention KV vs recurrent state

Every mainstream KV-caching system — vLLM's paged prefix cache, SGLang's
RadixAttention, LMCache's tiered sharing — is architected around one
premise: cached state is a **per-token tensor** you can slice into
blocks, hash, evict, and recombine. For this model, that premise holds
for 10 of 40 layers and fails for the other 30.

**Attention layers are the easy case.** Their KV cache is positional and
cheap: 10 layers × 2 KV heads × 256 head-dim × K-and-V × 2 bytes =
**~20 KB per token**. Any contiguous position range can be saved,
restored, or discarded independently.

**Recurrent layers break the premise.** A DeltaNet layer's state is a
fixed-size matrix that is a function of the *entire ordered prefix* — the
recurrence is order-dependent and not invertible. Three hard consequences:

1. **No rewind.** State at token N cannot be turned into state at token
   N−k. A cache can only ever extend forward from a stored point.
2. **No slicing.** "The state for tokens 100–116" does not exist as an
   object; there is only "the state after token 116, given everything
   before it". Block-granular caching and out-of-order chunk reuse
   (e.g. RAG-style CacheBlend) are impossible for these layers.
3. **Checkpoints are whole-state or nothing:** all 30 layers' recurrent
   matrices plus convolution history — an indivisible **~63 MB blob** at
   one specific token boundary.

So a hybrid cache must be **dense per block for attention, sparse per
checkpoint for SSM** — storing the 63 MB blob at block granularity the
way attention pages are stored would cost ~246 KB/token of pure overhead.
That asymmetry, more than any single kernel, is why hybrid-model caching
is its own engineering problem.

The cost model falls out directly: one SSM checkpoint costs as much
storage as ~3,000 tokens of attention KV, so checkpointing only pays for
prefixes long enough that re-prefilling them costs more than the blob
moves. The server gates snapshots on prompt length (default 2,048 tokens)
accordingly.

### What the server builds on top

The state checkpoint cache in `q36_server` snapshots engine state (KV
pages + the SSM blob) at **end of prompt, before generation**, keyed by
the exact prompt tokens. That timing is load-bearing: agent clients
resend a *reconstructed* history each turn — reasoning stripped, tool
calls re-rendered — which never byte-matches the tokens the model
actually generated, so any cache keyed on generated text scores zero.
Keyed on the client's own prompt, reconstruction cannot break matching.

A new request restores the longest stored prefix and prefills only the
delta. Restores are cheap in the common case — KV pages still resident on
the GPU for the same prefix are detected by position→token map and not
re-uploaded, so a typical agent turn moves the 63 MB SSM blob and nothing
else (~3 ms). Correctness is unconditional: state is restored to a token
boundary and the remainder prefilled from real tokens, so outputs are
bit-identical to a full prefill; the cache affects speed only. Measured
on a 78.7k-token agent turn: 13.6 s re-prefill → 0.4 s.

Checkpoints tier from a DRAM LRU (default 4 GB) to optional disk, and
every persisted blob carries an invalidation key of model identity +
KV-quantization mode + context size + layout version — restoring a
mismatched state layout would be a crash, not a cache miss.

---

## 4. Multi-GPU: why expert sharding loses on GeForce

A 256-expert MoE looks like an obvious sharding target: put 128 experts
on each of two GPUs and halve the weight read. q36 implements this
(`--gpus N`), it is numerically correct — and on GeForce hardware it is a
measured net **loss**: −10% at N=2, −33% at N=4 (prefill).

The reason is the interconnect. Consumer GeForce cards have no NVLink and
no peer-to-peer DMA, so every layer's partial results must be staged
through host memory over PCIe — 8–16 MB of reductions per layer, 40
layers deep, on the latency-critical path. The communication cost grows
faster than the sharding saves, and at batch 1 the per-expert compute is
far too small to hide any of it.

The path is kept in the tree because the loss is an artifact of the
interconnect, not the design: on P2P-capable hardware (NVLink datacenter
parts), or under batch serving where compute per transferred byte rises,
the same sharding turns profitable.

For the target hardware, the conclusion is simpler: at 20.2 GiB, the
entire quantized model fits resident in a single 32 GB card with room for
long-context KV — so single-GPU, zero-copy execution is not a limitation
but the design point everything above is optimized for.

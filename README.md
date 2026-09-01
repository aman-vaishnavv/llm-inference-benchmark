# llm-inference-benchmark

Hands-on benchmarking of LLM inference with vLLM on a T4 GPU. Covers memory layout, batch scaling, KV cache behavior, and quantization impact — measured on a real model, not simulated.

---

## Setup

- **Model**: Qwen2.5-7B-Instruct-AWQ (4-bit AWQ quantization)
- **GPU**: NVIDIA T4 (15GB VRAM)
- **Framework**: vLLM 0.28.0
- **Environment**: Kaggle (2x T4 GPUs available)

---

## Memory breakdown at load time

```
Weights (AWQ int4):   5.29 GB   ← fp16 would be ~14 GB, doesn't fit on T4
KV cache:             5.47 GB
Peak activations:     1.06 GB
CUDAGraph memory:     0.51 GB
────────────────────────────────
Total:               ~13.1 GB / 15 GB

Max concurrent requests (2048 tokens each): ~50
```

AWQ int4 quantization is what makes a 7B model viable on a 15GB T4. Without it, fp16 weights alone consume the entire GPU leaving no room for KV cache or activations.

---

## Batch scaling

Same prompt, increasing batch size, fixed 100 output tokens:

```
Batch    Throughput    Latency/req    Throughput gain
──────────────────────────────────────────────────────
1        43.7 tok/s    2.287s         1.0x
2        86.0 tok/s    1.163s         2.0x
4        166.3 tok/s   0.601s         3.8x
8        313.1 tok/s   0.319s         7.2x
16       514.7 tok/s   0.194s         11.8x
32       769.3 tok/s   0.130s         17.6x
```

Throughput scales near-linearly up to batch 8 — the GPU has spare capacity and batching fills it. Beyond batch 8 gains compress as the T4 approaches memory bandwidth saturation.

Latency per request keeps improving even as throughput gains flatten — at batch 32 each user waits 0.130s vs 2.287s at batch 1, a 17.6x improvement. This is the continuous batching win: the GPU is never idle waiting for slow requests to finish.

---

## KV cache — context length vs latency

Fixed 50 output tokens, increasing prompt length:

```
Context                Prompt tokens    Time      tok/sec
──────────────────────────────────────────────────────────
short  (~10 tokens)    5                1.176s    42.5
medium (~50 tokens)    44               1.183s    42.3
long   (~120 tokens)   123              1.199s    41.7
```

Going from 5 to 123 prompt tokens adds only 0.023s — essentially nothing. The KV cache eliminates redundant recomputation: keys and values for prompt tokens are computed once during prefill and reused across all decode steps. Without the KV cache, generation time would scale linearly with prompt length.

This is what makes long-context RAG practical — appending retrieved documents to the prompt has negligible impact on generation latency.

---

## Key takeaways

**Quantization unlocks the hardware** — AWQ int4 reduces weight memory 2.6x, fitting a 7B model on a T4 that would otherwise require an A100.

**Batching is the primary throughput lever** — 17.6x throughput improvement from batch 1 to batch 32 with only a 2x increase in per-request latency.

**KV cache eliminates the prompt length bottleneck** — prefill is fast (parallel), decode dominates latency, and prompt length barely matters.

**The latency vs throughput tradeoff is real** — small batches feel fast for individual users, large batches are efficient for the system. Production serving picks the right operating point based on SLA requirements.

---

## What wasn't covered

**Speculative decoding** — uses a small draft model to predict multiple tokens ahead, verified by the large model in parallel. Attempted with Qwen2.5 but blocked by vocab size mismatch between 7B (152064) and smaller models (151936) in this model family.

**Tensor parallelism** — splitting the model across multiple GPUs for larger models that don't fit on one device. Relevant for 70B+ models in production.

**Prefix caching** — reusing KV cache across requests that share a common prefix (system prompt, retrieved context). Significant latency win for RAG workloads.

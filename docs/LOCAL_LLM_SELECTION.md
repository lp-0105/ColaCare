# Local LLM Runtime Selection

## Decision

Use **Ollama 0.32.0 with `qwen3:4b` Q4_K_M** for the RTX 4060 Laptop smoke test.

The choice is operational, not clinical: Ollama supplies a stable local process, OpenAI-compatible `/v1/chat/completions`, JSON-Schema constrained output, and a small enough footprint for one 4096-token sequence. Its official model entry reports 4.02B parameters, Q4_K_M, and a 2.5 GB file. Qwen reports strong instruction-following, multilingual support, switchable thinking, and a native 32,768-token context.

Sources:

- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama context length](https://docs.ollama.com/context-length)
- [Ollama qwen3:4b](https://ollama.com/library/qwen3:4b)
- [Qwen3-4B official model card](https://huggingface.co/Qwen/Qwen3-4B)

## Runtime comparison

| Option | Windows/4060 deployment | OpenAI endpoint | JSON control | Verdict |
| --- | --- | --- | --- | --- |
| Ollama | Already installed; native Windows GPU; simple lifecycle | Built in on 11434 | `response_format` JSON/schema | Selected |
| llama.cpp | Lightweight CUDA/Vulkan and GGUF control | `llama-server` built in | JSON grammar/schema | Strong fallback; more binary/model-file handling |
| Transformers + bitsandbytes 4-bit | Maximum Python control | Needs a maintained server layer | Extra constraint and validation work | Too many dependencies for first smoke |
| vLLM | Excellent Linux throughput | Built in | Guided/structured output | Preferred for L20, not native Windows, excessive for one local sequence |

References:

- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

## Qwen3-4B Q4_K_M memory estimate

The architecture has 36 layers, 8 key/value heads, and head dimension 128. Ollama's default FP16 KV cache is approximately:

```text
36 * 8 * 128 * 2 (K,V) * 2 bytes = 147,456 bytes/token
```

| Component | 4096 context | 8192 context |
| --- | ---: | ---: |
| Quantized weights on disk | 2.5 GB | 2.5 GB |
| FP16 KV cache | 0.56 GiB | 1.13 GiB |
| CUDA graphs, buffers, allocator/runtime | 0.7-1.5 GiB | 0.8-1.6 GiB |
| Estimated model-service total | about 3.8-4.8 GiB | about 4.4-5.4 GiB |

The 256 output tokens share the context budget. Display applications also use VRAM, so acceptance measures actual system usage. Ollama documents FP16 as default; `q8_0` can roughly halve KV memory with Flash Attention, but the first test keeps defaults to reduce variables.

## Fixed first-version parameters

```text
batch size:             1
concurrent sequences:   1
context length:         4096
max output tokens:      256
temperature:            0
DoctorAgent count:      1
discussion rounds:      0
RAG:                    disabled
thinking:               disabled with /no_think
```

Multiple agents run sequentially. No long-context requests run concurrently. Models at 14B are excluded, and no second candidate is downloaded. The smoke checks format and instruction adherence only, not clinical quality.

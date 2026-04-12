# ASUS GX10 / DGX Spark — Qwen3.5 Speed Hack 🚀

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![Hardware](https://img.shields.io/badge/Hardware-ASUS_GX10_|_DGX_Spark-green)

> Unlock a **4-5x speedup** for Qwen3.5 on your ASUS GX10 / DGX Spark.
> Hybrid INT4+FP8 + MTP speculative decoding, automated in one shell script.
>
> ⚡ **35B:** `30 t/s → 112+ t/s`
> ⚡ **122B:** `10 t/s → 51 t/s`
>
> *Zero deep-learning expertise required. Just run the script.*

---

## The Story

When I first got my ASUS GX10 on March 26th, I ran Qwen3.5-35B with Ollama and got ~30 t/s. Painfully slow for a 128GB GPU machine.

I spent over a week fighting every framework I could find — sglang, vLLM, various Docker builds — and couldn't break 50 t/s. Eventually I gave up on pre-built binaries and manually compiled llama.cpp from source, applied my own optimizations, and finally hit **72 t/s on 35B** and **30 t/s on 122B**. Better, but still not what this hardware deserves.

Then I found **[albond's repo](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** — a technique that merges Intel's INT4 AutoRound weights with Qwen's FP8 weights into a hybrid checkpoint, then runs it on a patched vLLM with FlashInfer. It was designed for the 122B model only, but the results were jaw-dropping.

I adapted the approach for 35B, extended it to work with any INT4 AutoRound + FP8 model pair, added the MTP speculative decoding weights, and wrapped the entire pipeline in this interactive shell script.

The results:

- **35B: 72 t/s → 112+ t/s** (and up to 158 tok/s total on 4 concurrent requests)
- **122B: 30 t/s → 51 t/s**
- **27B: 10 t/s → 24 t/s**

I gave albond 10 stars on GitHub (by starring and unstarring repeatedly — he truly deserves it). This script is my way of paying it forward to everyone else who just bought this machine and is wondering why it feels slow.

---

## What This Does

- Builds a Hybrid INT4+FP8 checkpoint (albond's technique, extended for 35B + any AutoRound INT4 model)
- Adds MTP (Multi-Token Prediction) speculative decoding weights
- Runs everything inside Docker with patched vLLM (FlashInfer 0.6.7, FP8 dispatch fix)
- Interactive menu: install → start → benchmark — no Docker expertise needed

## Quick Start

```bash
wget https://raw.githubusercontent.com/phuongncn/asus-gx10-qwen35-speed-hack/main/vllm.sh
chmod +x vllm.sh
./vllm.sh
```

Then select **1** (Install) → choose your model → select **2** (Start server).

## Benchmark Results

### Qwen3.5-35B-A3B Hybrid INT4+FP8 (sequential, single request)

| Task     | Tokens | Time   | Speed     |
|----------|--------|--------|-----------|
| Q&A      | 256    | 2.14s  | 119 tok/s |
| Code     | 453    | 3.47s  | 130 tok/s |
| JSON     | 1024   | 8.05s  | 127 tok/s |
| Math     | 32     | 0.29s  | 108 tok/s |
| LongCode | 2048   | 15.54s | 131 tok/s |

**Concurrent (4 parallel requests): 158.7 tok/s total throughput**

### Speed comparison

| Model        | Ollama | llama.cpp (manual build) | This hack        | Method                |
|--------------|--------|--------------------------|------------------|-----------------------|
| Qwen3.5-35B  | ~30 t/s | ~72 t/s                 | **112+ t/s** ⚡  | Hybrid INT4+FP8 + MTP |
| Qwen3.5-122B | ~10 t/s | ~30 t/s                 | **51 t/s** ⚡    | Hybrid INT4+FP8 + MTP |
| Qwen3.5-27B  | ~10 t/s | —                        | **24 t/s** ⚡    | Hybrid INT4+FP8       |

## Hardware

- **ASUS GX10** (NVIDIA DGX Spark) — 128GB unified GPU memory
- Works on any NVIDIA DGX Spark / similar high-VRAM single-GPU machine

## Prerequisites

- Docker installed and running
- `nvidia-smi` accessible
- Free disk space: **35B ~20GB**, **122B ~70GB**
- HuggingFace account (free) for model downloads
- *(Note: 122B uses nearly all 128GB unified memory — close other heavy apps before loading)*

## Menu Options

| Option | Description |
|--------|-------------|
| 1 | First-time setup: clone repo, build Docker, download + build hybrid model |
| 2 | Select model and start vLLM server |
| 3 | Stop server |
| 4 | View logs |
| 5 | Run benchmark |
| 6 | Rebuild Docker image (no-cache) |

## Supported Models

| Model | INT4 Source | FP8 Source | Speed |
|-------|-------------|------------|-------|
| Qwen3.5-35B-A3B | `Intel/Qwen3.5-35B-A3B-int4-AutoRound` | `Qwen/Qwen3.5-35B-A3B-FP8` | ~112 tok/s |
| Qwen3.5-122B-A10B | `Intel/Qwen3.5-122B-A10B-int4-AutoRound` | *(via install.sh)* | ~51 tok/s |
| Any AutoRound INT4 | Custom HF repo | Custom FP8 HF repo | varies |

## How It Works (Technical)

1. **Hybrid checkpoint**: Takes Intel's INT4 AutoRound weights + Qwen's FP8 weights → merges expert layers (INT4) with attention layers (FP8) using albond's `build-hybrid-checkpoint.py`
2. **MTP weights**: Injects Multi-Token Prediction speculative decoding tensors → +15-25% speed on top of hybrid
3. **Patched vLLM**: Custom Docker image with FlashInfer 0.6.7 + FP8 dispatch fix required for hybrid to work correctly
4. **Memory flush**: Drops OS page cache before loading for maximum available VRAM

## Credits

- **[albond](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** — original hybrid INT4+FP8 technique and patched vLLM Docker image for 122B. This repo extends his work to 35B and any AutoRound INT4 model.
- **Intel** — AutoRound INT4 quantized models
- **Qwen team** — FP8 model releases

## License

MIT — see [LICENSE](LICENSE)
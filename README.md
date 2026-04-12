# ASUS GX10 / DGX Spark — Qwen3.5 Speed Hack 🚀

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![Hardware](https://img.shields.io/badge/Hardware-ASUS_GX10_|_DGX_Spark-green)

> Unzip a **4-5x speedup** for Qwen3.5 on your ASUS GX10 unified memory architecture. 
> Runs hybrid INT4+FP8 and MTP speculative decoding in a single, automated shell script.
>
> ⚡ **35B:** `70 t/s ➔ 112+ t/s`
> ⚡ **122B:** `30 t/s ➔ 51 t/s`
>
> *Zero deep-learning expertise required. Just clone, click, and fly.*

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
| Task     | Tokens | Time  | Speed      |
|----------|--------|-------|------------|
| Q&A      | 256    | 2.14s | 119 tok/s  |
| Code     | 453    | 3.47s | 130 tok/s  |
| JSON     | 1024   | 8.05s | 127 tok/s  |
| Math     | 32     | 0.29s | 108 tok/s  |
| LongCode | 2048   | 15.54s| 131 tok/s  |

**Concurrent (4 parallel requests): 158.7 tok/s total throughput**

### Speed comparison
| Model | Before | After | Method |
|-------|--------|-------|--------|
| Qwen3.5-35B  | ~70 t/s | **112+ t/s** | Hybrid INT4+FP8 + MTP |
| Qwen3.5-122B | ~30 t/s | **51 t/s**   | Hybrid INT4+FP8 + MTP |
| Qwen3.5-27B  | ~10 t/s | **24 t/s** | Hybrid INT4+FP8 |

## Hardware

- **ASUS GX10** (NVIDIA DGX Spark) — 128GB unified GPU memory
- Works on any NVIDIA DGX Spark / similar high-VRAM single-GPU machine

## Prerequisites

- Docker installed and running
- `nvidia-smi` accessible
- ~100GB free disk (for 35B: ~20GB, for 122B: ~150GB)
- HuggingFace account (free) for model downloads
- *(Note: This uses almost all of the 128GB unified memory for the 122B model, ensure no other heavy applications are running)*

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

## The Story

I bought an ASUS GX10 (NVIDIA DGX Spark) with 128GB unified GPU memory, hoping to run Qwen3.5-122B locally. Initial benchmarks gave ~30 tok/s, but I wanted more.

Then I discovered albond's hybrid INT4+FP8 technique, which merges Intel's quantized weights with Qwen's FP8 releases. Combined with MTP (Multi-Token Prediction) speculative decoding, I achieved **51 tok/s for 122B** and **112+ tok/s for 35B** — a 4-5x speedup over baseline.

This repo automates the entire pipeline: hybrid checkpoint building, MTP weight injection, Docker containerization with patched vLLM, and interactive benchmarking — all in one shell script.

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
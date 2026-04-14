# ASUS GX10 / DGX Spark — Qwen3.5 Speed Hack 🚀

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![Hardware](https://img.shields.io/badge/Hardware-ASUS_GX10_|_DGX_Spark-green)

> Unlock a **4-5x speedup** for Qwen3.5 on your ASUS GX10 / DGX Spark.
> Two modes: **Hybrid INT4+FP8 + MTP** for max single-request speed, or **Native FP8 + MTP** for full quality with better concurrent throughput.
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

In April 2026, after community feedback about INT4 quality tradeoffs, I added a **Native FP8 + MTP** mode — slower per request, but full quality and higher concurrent throughput.

I gave albond 10 stars on GitHub (by starring and unstarring repeatedly — he truly deserves it). This script is my way of paying it forward to everyone else who just bought this machine and is wondering why it feels slow.

---

## What This Does

- Builds a Hybrid INT4+FP8 checkpoint (albond's technique, extended for 35B + any AutoRound INT4 model)
- Adds MTP (Multi-Token Prediction) speculative decoding weights
- Runs everything inside Docker with patched vLLM (FlashInfer 0.6.7, FP8 dispatch fix)
- **Native FP8 + MTP** mode: runs `Qwen/Qwen3.5-35B-A3B-FP8` directly with MTP weights — no INT4 merging, full FP8 quality
- Interactive menu: install → start → benchmark — no Docker expertise needed

## Quick Start

```bash
git clone https://github.com/phuongncn/asus-gx10-qwen35-speed-hack
cd asus-gx10-qwen35-speed-hack
chmod +x vllm.sh scripts/*.sh
./vllm.sh
```

Then select **1** (Install) → choose your model → select **2** (Start server).

## Benchmark Results

### Qwen3.5-35B-A3B Native FP8 + MTP (sequential, single request)

| Task     | Tokens | Time   | Speed     |
|----------|--------|--------|-----------|
| Q&A      | 256    | 3.85s  | 66 tok/s  |
| Code     | 512    | 7.05s  | 73 tok/s  |
| JSON     | 1024   | 14.62s | 70 tok/s  |
| Math     | 32     | 0.52s  | 61 tok/s  |
| LongCode | 2048   | 27.51s | 74 tok/s  |

**Concurrent (4 parallel requests): 185.4 tok/s total throughput**

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
| Qwen3.5-35B (native FP8) | N/A ¹ | ~49 t/s | **~70 t/s** ⚡ / **185 tok/s** concurrent | Native FP8 + MTP |
| Qwen3.5-122B | ~10 t/s | ~30 t/s                 | **51 t/s** ⚡    | Hybrid INT4+FP8 + MTP |
| Qwen3.5-27B  | ~4-12 t/s | —                        | **24 t/s** ⚡    | Hybrid INT4+FP8       |

> ¹ Ollama does not support FP8 models — no GGUF available for Qwen3.5-35B-A3B-FP8.
> llama.cpp ~49 t/s = best community result with native SM121 kernel build ([source](https://forums.developer.nvidia.com/t/dgx-spark-13-49-tok-s-with-qwen3-5-35b-native-sm121-kernel-build-guide/365083)).

## Which Mode Should You Use?

| | Hybrid INT4+FP8 + MTP | Native FP8 + MTP |
|---|---|---|
| Single-request speed | **~112 tok/s** | ~70 tok/s |
| Concurrent throughput (4×) | 158 tok/s total | **185 tok/s total** |
| Output quality | INT4 quantization artifacts possible | Full FP8 quality |
| Disk usage | ~20GB | ~35GB |

**Choose Hybrid** if you use the model solo and want the absolute fastest responses.
**Choose Native FP8** if you care about output quality, run multiple concurrent users, or noticed quality issues with INT4.

> This mode was added after a question from **stefan132** on the NVIDIA forums:
> *"Would it make sense to avoid Int4 and use Qwen/Qwen3.5-35B-A3B-FP8 directly with MTP? AutoRound INT4 is great, but more and more I realize serious quality problems."*
> Short answer: yes, it does make sense — and now it's a first-class option in this script.

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
| 1 | First-time setup: clone repo, build Docker, download + build model (submenu: 122B / 35B Hybrid / 35B FP8+MTP / Custom / Both) |
| 2 | Select model and start vLLM server |
| 3 | Stop server |
| 4 | View logs |
| 5 | Run benchmark |
| 6 | Rebuild Docker image (--no-cache) |
| 7 | Build checkpoint (submenu: Hybrid INT4+FP8 / Native FP8+MTP) — no Docker rebuild, skips download if already present |

## Project Structure

```
vllm.sh                  ← entry point / menu router
scripts/
  common.sh              ← shared vars, color helpers (sourced by others)
  env-check.sh           ← docker + GPU validation
  install.sh             ← option 1: download, build hybrid/FP8 checkpoint
  build-docker.sh        ← option 6: build vllm-sm121 + vllm-qwen35-v2 images (no model download)
  build-hybrid.sh        ← option 7: build hybrid INT4+FP8 or native FP8+MTP checkpoint
  start-server.sh        ← option 2: model selection + docker run + health poll
  benchmark.sh           ← option 5: sequential + concurrent benchmark
  add-mtp-from-fp8.py   ← extracts MTP weights from FP8 source when INT4 has none
```

Each script can also be run standalone:
```bash
bash scripts/benchmark.sh 8000        # benchmark against port 8000
bash scripts/install.sh               # run install without menu wrapper
```

## Supported Models

| Model | INT4 Source | FP8 Source | Speed |
|-------|-------------|------------|-------|
| Qwen3.5-35B-A3B | `Intel/Qwen3.5-35B-A3B-int4-AutoRound` | `Qwen/Qwen3.5-35B-A3B-FP8` | ~112 tok/s |
| Qwen3.5-35B-A3B FP8+MTP | *(none — native FP8)* | `Qwen/Qwen3.5-35B-A3B-FP8` | ~70 tok/s |
| Qwen3.5-122B-A10B | `Intel/Qwen3.5-122B-A10B-int4-AutoRound` | *(via install.sh)* | ~51 tok/s |
| Any AutoRound INT4 | Custom HF repo | Custom FP8 HF repo | varies |

## How It Works (Technical)

1. **Hybrid checkpoint**: Takes Intel's INT4 AutoRound weights + Qwen's FP8 weights → merges expert layers (INT4) with attention layers (FP8) using albond's `build-hybrid-checkpoint.py`
2. **MTP weights**: Injects Multi-Token Prediction speculative decoding tensors → +15-25% speed on top of hybrid. Source is INT4 model's `model_extra_tensors.safetensors` when available; otherwise extracted directly from the FP8 source via `add-mtp-from-fp8.py`
3. **Patched vLLM**: Custom Docker image with FlashInfer 0.6.7 + FP8 dispatch fix required for hybrid to work correctly
4. **Memory flush**: Drops OS page cache before loading for maximum available VRAM

## Credits

- **[albond](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)** — original hybrid INT4+FP8 technique and patched vLLM Docker image for 122B. This repo extends his work to 35B and any AutoRound INT4 model.
- **Intel** — AutoRound INT4 quantized models
- **Qwen team** — FP8 model releases

## License

MIT — see [LICENSE](LICENSE)
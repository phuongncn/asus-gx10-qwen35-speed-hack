# Reddit Post — r/LocalLLaMA

## Title
Spent over a week hitting a wall at 72 t/s on GX10 — found a technique that pushed Qwen3.5-35B to 131 t/s

---

## Body

TL;DR: Hybrid INT4+FP8 checkpoint + MTP speculative decoding = massive speed jump. Script at the bottom to automate everything.

---

When I got my GX10 in late March, Ollama gave me ~30 t/s on Qwen3.5-35B. Fine, expected. Spent the next week trying sglang, vLLM, every Docker build I could find — couldn't break 50 t/s. Eventually gave up on pre-built binaries, compiled llama.cpp from source with manual optimizations and hit **72 t/s**. Better, but still felt like the 128GB GPU was laughing at me.

Then I found **albond's repo** (search: DGX_Spark_Qwen3.5-122B-A10B-AR-INT4 on GitHub).

His technique merges Intel's INT4 AutoRound weights with Qwen's FP8 weights into a hybrid checkpoint, then runs it on a patched vLLM with FlashInfer. He built it for 122B only — I adapted it for 35B, added MTP speculative decoding weights on top, and got this:

| Task | Speed |
|---|---|
| Q&A | 119 tok/s |
| Code | 130 tok/s |
| JSON | 127 tok/s |
| Math | 108 tok/s |
| Long code (2048 tok) | 131 tok/s |

**4 concurrent requests: 158 tok/s total throughput.**

For 122B: 30 t/s → **51 t/s**.

---

I gave albond's repo 10 stars (by starring and unstarring repeatedly — he deserves it). Couldn't find a way to give more.

To make it easier for everyone else who just bought this machine and is wondering why it feels slow, I wrapped the entire pipeline into one shell script — hybrid build, MTP weights, Docker launch, benchmark — all behind an interactive menu. No Docker expertise needed.

*(links in first comment)*

Happy to answer questions. Also works for Qwen3.5-27B (~10 t/s → 24 t/s) and should work for any INT4 AutoRound + FP8 model pair.

---

## First Comment (post immediately after submitting)

Original technique by albond: https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4

My script to automate the full pipeline (hybrid build, MTP weights, Docker launch, benchmark): https://github.com/phuongncn/asus-gx10-qwen35-speed-hack

---

## Flair
Resources

#!/usr/bin/env python3
"""Extract MTP tensors from FP8 source and add to hybrid checkpoint.

Use when Intel AutoRound INT4 source does NOT have model_extra_tensors.safetensors
(e.g. Qwen3.5-27B). Reads MTP tensors from FP8 model — either local path or HF repo.

Usage (local):
    python add-mtp-from-fp8.py \
        --fp8-source /path/to/fp8-snapshot \
        --target ~/models/Qwen3.5-27B-hybrid-int4fp8

Usage (HF repo, downloads needed shards only):
    python add-mtp-from-fp8.py \
        --fp8-repo Qwen/Qwen3.5-27B-FP8 \
        --target ~/models/Qwen3.5-27B-hybrid-int4fp8
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fp8-source",
                       help="Path to local FP8 model snapshot dir")
    group.add_argument("--fp8-repo",
                       help="HuggingFace repo ID (downloads needed shards only)")
    parser.add_argument("--target", required=True,
                        help="Path to hybrid checkpoint to update")
    args = parser.parse_args()

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    target_dir = Path(args.target)
    if not target_dir.exists():
        print(f"ERROR: Target not found: {target_dir}")
        sys.exit(1)

    target_mtp = target_dir / "model_extra_tensors.safetensors"
    if target_mtp.exists():
        print("MTP weights already present — skipping")
        return

    # ── Lấy index từ local hoặc HF ───────────────────���────────────
    if args.fp8_source:
        fp8_dir = Path(args.fp8_source)
        if not fp8_dir.exists():
            print(f"ERROR: FP8 source not found: {fp8_dir}")
            sys.exit(1)
        fp8_index_path = fp8_dir / "model.safetensors.index.json"
        with open(fp8_index_path) as f:
            fp8_idx = json.load(f)
        def get_shard_path(shard_name):
            p = fp8_dir / shard_name
            if not p.exists():
                print(f"ERROR: Shard not found locally: {p}")
                print("  Hint: use --fp8-repo instead to download from HF")
                sys.exit(1)
            return p
    else:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
            sys.exit(1)
        print(f"Downloading index from {args.fp8_repo}...")
        import os, tempfile
        idx_path = hf_hub_download(
            repo_id=args.fp8_repo,
            filename="model.safetensors.index.json",
        )
        with open(idx_path) as f:
            fp8_idx = json.load(f)
        _shard_cache: dict[str, Path] = {}
        def get_shard_path(shard_name):
            if shard_name not in _shard_cache:
                print(f"  Downloading shard {shard_name}...")
                local = hf_hub_download(
                    repo_id=args.fp8_repo,
                    filename=shard_name,
                )
                _shard_cache[shard_name] = Path(local)
            return _shard_cache[shard_name]

    # ── Tìm MTP tensors ────────────────────────────────────────────
    mtp_map = {k: v for k, v in fp8_idx["weight_map"].items()
               if "mtp" in k.lower()}
    if not mtp_map:
        print("ERROR: No MTP tensors found in FP8 source")
        sys.exit(1)

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in mtp_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    print(f"Found {len(mtp_map)} MTP tensors across {len(shard_to_keys)} shard(s)")

    # ── Extract tensors ────────────────────────────────────────────
    tensors = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = get_shard_path(shard_name)
        print(f"  Reading {shard_name} ({len(keys)} MTP tensors)...")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors total")

    # ── Save model_extra_tensors.safetensors ───────────────────────
    print(f"Saving -> {target_mtp}")
    save_file(tensors, str(target_mtp))
    size_mb = target_mtp.stat().st_size / 1e6
    print(f"  Size: {size_mb:.1f} MB")

    # ── Update target index ────────────────────────────────────────
    target_index_path = target_dir / "model.safetensors.index.json"
    with open(target_index_path) as f:
        tgt_idx = json.load(f)

    before = len(tgt_idx["weight_map"])
    for key in mtp_map:
        tgt_idx["weight_map"][key] = "model_extra_tensors.safetensors"
    with open(target_index_path, "w") as f:
        json.dump(tgt_idx, f, indent=2)

    print(f"Added {len(tgt_idx['weight_map']) - before} MTP tensor mappings")
    print(f"Total tensors: {len(tgt_idx['weight_map'])}")
    print("Done. Use --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'")
    print("Note: Qwen3.5-27B supports MTP-1 only (not 2).")


if __name__ == "__main__":
    main()

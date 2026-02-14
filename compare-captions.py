#!/usr/bin/env python3
"""
Compare captions from base (bf16) vs 4-bit quantized Florence-2.

Runs the same images through both model variants and saves results
side-by-side to .tmp/caption_comparison.jsonl for manual inspection.

Usage:
    python compare-captions.py
    python compare-captions.py --num-samples 50 --model florence-community/Florence-2-large
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from importlib.util import spec_from_file_location, module_from_spec

# Import from the fast script
_spec = spec_from_file_location(
    "fast", Path(__file__).parent / "datacomp-caption-fast.py"
)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_model = _mod.load_model
make_tar_dataset = _mod.make_tar_dataset
collate_fn = _mod.collate_fn
generate_captions = _mod.generate_captions
MODELS = _mod.MODELS
DETAIL_LEVELS = _mod.DETAIL_LEVELS
DEFAULT_MODEL = _mod.DEFAULT_MODEL
DEFAULT_DETAIL = _mod.DEFAULT_DETAIL


def collect_captions(
    model,
    processor,
    tar_path: Path,
    num_samples: int,
    batch_size: int,
    task_prompt: str,
    device: str,
) -> list[dict]:
    """Run model on up to num_samples images from a tar, return list of {key, caption}."""
    dataset = make_tar_dataset(tar_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    results = []
    for keys, images, orig_captions, metas in loader:
        if not images:
            continue

        captions = generate_captions(model, processor, images, task_prompt, device)

        for key, caption, orig in zip(keys, captions, orig_captions):
            results.append({"key": key, "caption": caption, "original_caption": orig})
            if len(results) >= num_samples:
                break

        for img in images:
            img.close()

        if len(results) >= num_samples:
            break

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs 4-bit quantized captions"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=MODELS.keys(), metavar="MODEL"
    )
    parser.add_argument(
        "--detail",
        default=DEFAULT_DETAIL,
        choices=DETAIL_LEVELS.keys(),
        metavar="LEVEL",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path.home() / "Data/datacomp-proper-wds"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-samples", type=int, default=20, help="Number of images to compare"
    )
    args = parser.parse_args()

    task_prompt = DETAIL_LEVELS[args.detail]
    tmp_dir = Path(__file__).parent / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    tar_files = sorted(args.data_dir.glob("*.tar"))
    if not tar_files:
        print("No tar files found")
        return

    tar_path = tar_files[0]
    device = "cuda"

    # --- Run base model (bf16) ---
    print("Loading base model (bf16)...")
    model_base, processor = load_model(args.model, device, quantize=None)

    print(f"Generating captions with base model from {tar_path.name}...")
    t0 = time.time()
    base_results = collect_captions(
        model_base,
        processor,
        tar_path,
        args.num_samples,
        args.batch_size,
        task_prompt,
        device,
    )
    base_time = time.time() - t0
    print(f"  {len(base_results)} captions in {base_time:.1f}s")

    # Free base model
    del model_base
    torch.cuda.empty_cache()

    # --- Run 4-bit model ---
    print("Loading 4-bit quantized model...")
    model_4bit, processor = load_model(args.model, device, quantize="4bit")

    print(f"Generating captions with 4-bit model from {tar_path.name}...")
    t0 = time.time()
    quant_results = collect_captions(
        model_4bit,
        processor,
        tar_path,
        args.num_samples,
        args.batch_size,
        task_prompt,
        device,
    )
    quant_time = time.time() - t0
    print(f"  {len(quant_results)} captions in {quant_time:.1f}s")

    del model_4bit
    torch.cuda.empty_cache()

    # --- Build comparison ---
    quant_by_key = {r["key"]: r["caption"] for r in quant_results}

    comparisons = []
    for r in base_results:
        key = r["key"]
        if key not in quant_by_key:
            continue
        comparisons.append(
            {
                "key": key,
                "original_caption": r["original_caption"],
                "base_caption": r["caption"],
                "quant_4bit_caption": quant_by_key[key],
                "match": r["caption"].strip() == quant_by_key[key].strip(),
            }
        )

    # Save JSONL
    output_path = tmp_dir / "caption_comparison.jsonl"
    with open(output_path, "w") as f:
        for c in comparisons:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Also save a readable text version
    txt_path = tmp_dir / "caption_comparison.txt"
    with open(txt_path, "w") as f:
        exact_matches = sum(1 for c in comparisons if c["match"])
        f.write(f"Model: {args.model} | Detail: {args.detail}\n")
        f.write(
            f"Samples: {len(comparisons)} | Exact matches: {exact_matches}/{len(comparisons)}\n"
        )
        f.write(f"Base time: {base_time:.1f}s | 4-bit time: {quant_time:.1f}s\n")
        f.write("=" * 80 + "\n\n")

        for i, c in enumerate(comparisons, 1):
            f.write(f"--- Sample {i}: {c['key']} ---\n")
            f.write(f"  Original: {c['original_caption']}\n")
            f.write(f"  Base:     {c['base_caption']}\n")
            f.write(f"  4-bit:    {c['quant_4bit_caption']}\n")
            f.write(f"  Match:    {'YES' if c['match'] else 'NO'}\n\n")

    print("\nResults saved to:")
    print(f"  {output_path}")
    print(f"  {txt_path}")

    exact_matches = sum(1 for c in comparisons if c["match"])
    print(
        f"\nExact matches: {exact_matches}/{len(comparisons)} ({100 * exact_matches / max(len(comparisons), 1):.0f}%)"
    )


if __name__ == "__main__":
    main()

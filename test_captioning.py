#!/usr/bin/env python3
"""
Test and benchmark Florence-2-large captioning on DataComp WebDataset samples.

Usage:
    python test_captioning.py [--batch-sizes 4,8,16,32] [--num-images 100] [--tar-file PATH]
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration


DEFAULT_TAR = Path.home() / "Data/datacomp-proper-wds/0000000.tar"
MODEL_ID = "florence-community/Florence-2-large"


def load_images_from_tar(tar_path: str, max_images: int = 200) -> list[dict[str, Any]]:
    """Load images and metadata from a WebDataset tar file."""
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key = member.name.rsplit(".", 1)[0]
            ext = member.name.rsplit(".", 1)[1] if "." in member.name else ""

            if ext in ("jpg", "jpeg", "png", "webp"):
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data = fobj.read()
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    current["key"] = key
                    current["image"] = img
                except Exception:
                    continue
            elif ext == "txt":
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                current["original_caption"] = (
                    fobj.read().decode("utf-8", errors="replace").strip()
                )
            elif ext == "json":
                try:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    current["metadata"] = json.load(fobj)
                except Exception:
                    pass

            # Emit sample when we have image + caption
            if "image" in current and "original_caption" in current:
                samples.append(current)
                current = {}
                if len(samples) >= max_images:
                    break

    print(f"Loaded {len(samples)} images from {tar_path}")
    return samples


def load_model(model_id: str, device: str = "cuda") -> tuple[Any, Any]:
    """Load Florence-2 model and processor."""
    print(f"Loading {model_id}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    loaded_model: Any = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model: Any = loaded_model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    return model, processor


def caption_batch(
    model, processor, images: list[Image.Image], device: str = "cuda"
) -> list[str]:
    """Generate captions for a batch of images."""
    task_prompt = "<MORE_DETAILED_CAPTION>"
    prompts = [task_prompt] * len(images)

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device, torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            early_stopping=True,
        )

    results = processor.batch_decode(generated_ids, skip_special_tokens=False)
    captions = []
    for r in results:
        parsed = processor.post_process_generation(r, task=task_prompt)
        captions.append(parsed.get(task_prompt, r))
    return captions


def benchmark_batch_size(
    model, processor, images: list[Image.Image], batch_size: int, device: str = "cuda"
) -> dict:
    """Benchmark a specific batch size."""
    n = min(
        len(images), max(batch_size * 4, 32)
    )  # Use enough images for stable measurement
    test_images = images[:n]

    # Warmup
    caption_batch(model, processor, test_images[:batch_size], device)
    torch.cuda.synchronize()

    t0 = time.time()
    total = 0
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i : i + batch_size]
        if not batch:
            break
        caption_batch(model, processor, batch, device)
        total += len(batch)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    throughput = total / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "batch_size": batch_size,
        "images_processed": total,
        "elapsed_sec": round(elapsed, 2),
        "images_per_sec": round(throughput, 2),
        "peak_gpu_gb": round(mem, 2),
        "est_10m_hours": round(10_000_000 / throughput / 3600, 1),
        "est_10m_days": round(10_000_000 / throughput / 3600 / 24, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Florence-2 captioning")
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16,32",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="Number of images to load for testing",
    )
    parser.add_argument("--tar-file", default=str(DEFAULT_TAR), help="Path to tar file")
    parser.add_argument(
        "--show-captions",
        type=int,
        default=5,
        help="Number of sample captions to display",
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Load images
    samples = load_images_from_tar(args.tar_file, args.num_images)
    images = [s["image"] for s in samples]

    # Load model
    model, processor = load_model(MODEL_ID)

    # Show sample captions
    if args.show_captions > 0:
        print(f"\n{'=' * 80}")
        print(f"Sample captions (first {args.show_captions} images)")
        print(f"{'=' * 80}")
        sample_imgs = images[: args.show_captions]
        captions = caption_batch(model, processor, sample_imgs)
        for i, (s, cap) in enumerate(zip(samples[: args.show_captions], captions)):
            print(f"\n[{i}] Key: {s['key']}")
            print(f"    Original:  {s['original_caption']}")
            print(f"    Generated: {cap}")
            print(f"    Size: {s['image'].size}")

    # Benchmark each batch size
    print(f"\n{'=' * 80}")
    print("Benchmarking batch sizes...")
    print(f"{'=' * 80}")

    results = []
    for bs in batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats()
            r = benchmark_batch_size(model, processor, images, bs)
            results.append(r)
            print(
                f"  batch_size={r['batch_size']:3d}  |  {r['images_per_sec']:6.2f} img/s  |  "
                f"peak GPU: {r['peak_gpu_gb']:.2f} GB  |  est 10M: {r['est_10m_days']:.1f} days"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs:3d}  |  OOM!")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  batch_size={bs:3d}  |  Error: {e}")

    # Summary
    if results:
        best = max(results, key=lambda x: x["images_per_sec"])
        print(
            f"\nBest: batch_size={best['batch_size']} at {best['images_per_sec']} img/s "
            f"(~{best['est_10m_days']} days for 10M images)"
        )

    print(f"\nFinal GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark script for the fast captioning pipeline.

Processes a small number of tars to a .tmp/ directory, measuring throughput
and GPU utilization without affecting production data.

Usage:
    python benchmark.py
    python benchmark.py --num-tars 3 --batch-size 32
    python benchmark.py --model florence-community/Florence-2-large
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
generate_captions_safe = _mod.generate_captions_safe
MODELS = _mod.MODELS
DETAIL_LEVELS = _mod.DETAIL_LEVELS
DEFAULT_MODEL = _mod.DEFAULT_MODEL
DEFAULT_DETAIL = _mod.DEFAULT_DETAIL


def benchmark_tar(
    tar_path: Path,
    model,
    processor,
    batch_size: int,
    task_prompt: str,
    device: str,
) -> dict:
    """Benchmark a single tar. Returns timing stats."""
    from torch.utils.data import DataLoader

    dataset = make_tar_dataset(tar_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    total_images = 0
    total_inference_time = 0.0
    total_io_time = 0.0
    batch_count = 0

    t_start = time.time()
    t_io_start = time.time()

    for keys, images, orig_captions, metas in loader:
        io_time = time.time() - t_io_start
        total_io_time += io_time

        if not images:
            t_io_start = time.time()
            continue

        stats = {"total_images": 0, "errors": 0}

        t_infer = time.time()
        generate_captions_safe(model, processor, images, task_prompt, device, stats)
        inference_time = time.time() - t_infer
        total_inference_time += inference_time

        n = len(images)
        total_images += n
        batch_count += 1

        for img in images:
            img.close()

        t_io_start = time.time()

    wall_time = time.time() - t_start

    return {
        "tar": tar_path.name,
        "images": total_images,
        "batches": batch_count,
        "wall_time_s": round(wall_time, 2),
        "inference_time_s": round(total_inference_time, 2),
        "io_time_s": round(total_io_time, 2),
        "img_per_sec": round(total_images / wall_time, 2) if wall_time > 0 else 0,
        "gpu_busy_pct": round(100 * total_inference_time / wall_time, 1)
        if wall_time > 0
        else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark captioning throughput")
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
        "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantize model weights (4bit NF4 or 8bit via bitsandbytes)",
    )
    parser.add_argument(
        "--num-tars", type=int, default=2, help="Number of tars to benchmark"
    )
    parser.add_argument(
        "--warmup-tars",
        type=int,
        default=1,
        help="Warmup tars (excluded from results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose torch.compile/dynamo/inductor logging",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging

        import torch._dynamo
        import torch._inductor

        torch._dynamo.config.verbose = True
        torch._inductor.config.trace.enabled = True

        for name in [
            "torch._dynamo",
            "torch._inductor",
            "torch.jit",
            "torch._functorch",
        ]:
            logging.getLogger(name).setLevel(logging.DEBUG)

    task_prompt = DETAIL_LEVELS[args.detail]

    # Use .tmp/ for output
    tmp_dir = Path(__file__).parent / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    tar_files = sorted(args.data_dir.glob("*.tar"))
    total_needed = args.warmup_tars + args.num_tars
    if len(tar_files) < total_needed:
        print(f"Need {total_needed} tars but only found {len(tar_files)}")
        return

    tars_to_run = tar_files[:total_needed]

    device = "cuda"
    model, processor = load_model(args.model, device, quantize=args.quantize)

    quant_label = args.quantize or "none"
    print(
        f"\nBenchmark: {args.model} | batch={args.batch_size} | quantize={quant_label}"
    )
    print(f"Warmup: {args.warmup_tars} tar(s) | Measure: {args.num_tars} tar(s)\n")

    # Warmup
    for i, tar_path in enumerate(tars_to_run[: args.warmup_tars]):
        print(f"Warmup {i + 1}/{args.warmup_tars}: {tar_path.name}...")
        benchmark_tar(
            tar_path,
            model,
            processor,
            args.batch_size,
            task_prompt,
            device,
        )

    # Benchmark
    results = []
    for i, tar_path in enumerate(tars_to_run[args.warmup_tars :]):
        print(f"Benchmark {i + 1}/{args.num_tars}: {tar_path.name}...")
        result = benchmark_tar(
            tar_path,
            model,
            processor,
            args.batch_size,
            task_prompt,
            device,
        )
        results.append(result)
        print(
            f"  {result['images']} images | {result['img_per_sec']} img/s | "
            f"GPU busy: {result['gpu_busy_pct']}% | "
            f"inference: {result['inference_time_s']}s | io: {result['io_time_s']}s"
        )

    # Summary
    total_images = sum(r["images"] for r in results)
    total_wall = sum(r["wall_time_s"] for r in results)
    total_infer = sum(r["inference_time_s"] for r in results)
    avg_rate = total_images / total_wall if total_wall > 0 else 0
    avg_gpu = 100 * total_infer / total_wall if total_wall > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({args.num_tars} tars, {total_images} images)")
    print(f"  Throughput:    {avg_rate:.1f} img/s")
    print(f"  GPU busy:      {avg_gpu:.1f}%")
    print(f"  Wall time:     {total_wall:.1f}s")
    print(f"  Inference:     {total_infer:.1f}s")
    print(f"  I/O overhead:  {total_wall - total_infer:.1f}s")
    print(f"{'=' * 60}")

    # Save results
    results_path = tmp_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "batch_size": args.batch_size,
                "quantize": args.quantize or "none",
                "summary": {
                    "total_images": total_images,
                    "avg_img_per_sec": round(avg_rate, 2),
                    "avg_gpu_busy_pct": round(avg_gpu, 1),
                    "total_wall_time_s": round(total_wall, 2),
                },
                "per_tar": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

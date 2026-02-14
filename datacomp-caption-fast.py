#!/usr/bin/env python3
"""
Fast captioning pipeline for DataComp-10M using Florence-2.

Uses webdataset + PyTorch DataLoader for async I/O prefetching,
torch.compile for optimized inference, and direct JSONL append.
Single-GPU only.

ML optimizations:
  - bitsandbytes 4-bit quantization (--quantize 4bit) for lower VRAM + bigger batches
  - SDPA fused attention kernels (no flash_attn compile needed)
  - torch.inference_mode() for faster inference
  - Static KV cache for torch.compile-friendly generation
  - TF32 matmul and cuDNN benchmark for faster kernels

Usage:
    python datacomp-caption-fast.py --batch-size 16
    python datacomp-caption-fast.py --quantize 4bit --batch-size 48
    python datacomp-caption-fast.py --resume
    python datacomp-caption-fast.py --list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Florence2ForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DETAIL_LEVELS = {
    "brief": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "more_detailed": "<MORE_DETAILED_CAPTION>",
}

MODELS = {
    "florence-community/Florence-2-base": "0.23B params, ~24 img/s, ~5 days for 10M",
    "florence-community/Florence-2-large": "0.77B params, ~4 img/s, ~29 days for 10M",
}

DEFAULT_MODEL = "florence-community/Florence-2-base"
DEFAULT_DETAIL = "detailed"
MAX_NEW_TOKENS = 256
NUM_BEAMS = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA performance flags
# ---------------------------------------------------------------------------

# Allow TF32 for matmuls — ~2x faster on Ampere+ with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Auto-tune convolution algorithms for the hardware
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        log.warning("Forced exit.")
        sys.exit(1)
    log.info("Shutdown requested — finishing current tar then saving checkpoint...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(
    model_id: str,
    device: str = "cuda",
    quantize: str | None = None,
) -> tuple[Any, Any]:
    log.info(f"Loading {model_id} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)

    # Build quantization config
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    if quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
        log.info("Using 4-bit NF4 quantization via bitsandbytes")
    elif quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs["device_map"] = "auto"
        log.info("Using 8-bit quantization via bitsandbytes")

    model: Any = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        **load_kwargs,
    )

    # Only move to device if not using device_map (quantized models handle this)
    if "device_map" not in load_kwargs:
        model = model.to(device)

    model.eval()

    # torch.compile — skip for quantized models (not compatible with bnb layers)
    if quantize is None:
        log.info("torch.compile: starting compilation with mode='max-autotune'...")
        t_compile = time.time()
        try:
            model = torch.compile(model, mode="max-autotune")
            log.info(
                f"torch.compile: graph captured in {time.time() - t_compile:.1f}s "
                f"(kernels will be compiled on first inference)"
            )
        except Exception as e:
            log.warning(
                f"torch.compile failed after {time.time() - t_compile:.1f}s, "
                f"falling back to eager mode: {e}"
            )
    else:
        log.info("torch.compile: skipped (not compatible with quantized models)")

    mem = torch.cuda.memory_allocated(device) / 1e9
    log.info(f"Model loaded in {time.time() - t0:.1f}s | GPU mem: {mem:.2f} GB")
    return model, processor


# ---------------------------------------------------------------------------
# WebDataset pipeline
# ---------------------------------------------------------------------------


def make_tar_dataset(tar_path: Path) -> wds.WebDataset:
    """Create a webdataset pipeline for a single tar file.

    Decodes images to PIL and extracts key, image, caption text, and JSON metadata.
    """

    def extract_fields(sample):
        key = sample["__key__"]
        image = (
            sample.get("jpg")
            or sample.get("jpeg")
            or sample.get("png")
            or sample.get("webp")
        )
        if image is None:
            return None
        caption = sample.get("txt", "")
        meta = sample.get("json", {})
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8", errors="replace")
        return key, image, caption, meta

    dataset = (
        wds.WebDataset(str(tar_path), empty_check=False, shardshuffle=False)
        .decode("pil")
        .map(extract_fields)
        .select(lambda x: x is not None)
    )
    return dataset


def collate_fn(samples):
    """Collate webdataset samples into batched format for inference."""
    keys, images, captions, metas = [], [], [], []
    for key, image, caption, meta in samples:
        if not isinstance(image, Image.Image):
            continue
        try:
            image = image.convert("RGB")
        except Exception:
            continue
        keys.append(key)
        images.append(image)
        captions.append(caption)
        metas.append(meta)
    return keys, images, captions, metas


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------


_first_inference = True


def generate_captions(
    model,
    processor,
    images: list[Image.Image],
    task_prompt: str,
    device: str,
) -> list[str]:
    global _first_inference

    prompts = [task_prompt] * len(images)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device, torch.bfloat16)

    if _first_inference:
        log.info(
            f"torch.compile: first inference (batch={len(images)}) — "
            f"kernel compilation happening now, this may take a few minutes..."
        )
        t_first = time.time()

    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            cache_implementation="static",
        )

    if _first_inference:
        log.info(
            f"torch.compile: first inference completed in {time.time() - t_first:.1f}s — "
            f"subsequent batches will be fast"
        )
        _first_inference = False

    decoded = processor.batch_decode(ids, skip_special_tokens=False)
    captions = []
    for text in decoded:
        parsed = processor.post_process_generation(text, task=task_prompt)
        caption = parsed.get(task_prompt, text.strip())
        caption = caption.replace("<pad>", "").strip()
        captions.append(caption)
    return captions


def generate_captions_safe(
    model,
    processor,
    images: list[Image.Image],
    task_prompt: str,
    device: str,
    stats: dict,
) -> list[str]:
    """Generate captions with OOM recovery (halve batch and retry)."""
    try:
        return generate_captions(model, processor, images, task_prompt, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(images) // 2
        if mid == 0:
            log.error("OOM on single image, skipping")
            stats["errors"] += 1
            return [""]
        log.warning(f"OOM on batch of {len(images)} — splitting in half")
        left = generate_captions_safe(
            model, processor, images[:mid], task_prompt, device, stats
        )
        right = generate_captions_safe(
            model, processor, images[mid:], task_prompt, device, stats
        )
        return left + right
    except Exception as e:
        log.error(f"Error generating captions: {e}")
        stats["errors"] += len(images)
        return [""] * len(images)


# ---------------------------------------------------------------------------
# Process one tar
# ---------------------------------------------------------------------------


def process_tar(
    tar_path: Path,
    model,
    processor,
    output_file,
    batch_size: int,
    task_prompt: str,
    device: str,
    stats: dict,
    image_bar: tqdm | None = None,
) -> int:
    """Process a single tar file via DataLoader. Returns images captioned."""
    dataset = make_tar_dataset(tar_path)
    # num_workers=0 because webdataset can't split a single shard across workers.
    # I/O is still fast because webdataset streams the tar sequentially and the
    # GPU inference time dominates anyway.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    tar_name = tar_path.name
    count = 0

    for keys, images, orig_captions, metas in loader:
        if _shutdown_requested or not images:
            break

        captions = generate_captions_safe(
            model,
            processor,
            images,
            task_prompt,
            device,
            stats,
        )

        for key, caption, orig, meta in zip(keys, captions, orig_captions, metas):
            record = {
                "key": key,
                "caption": caption,
                "original_caption": orig,
                "tar_file": tar_name,
                "uid": meta.get("uid", "") if isinstance(meta, dict) else "",
                "url": meta.get("url", "") if isinstance(meta, dict) else "",
                "width": meta.get("width") if isinstance(meta, dict) else None,
                "height": meta.get("height") if isinstance(meta, dict) else None,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        n = len(images)
        count += n
        stats["total_images"] += n

        if image_bar is not None:
            image_bar.update(n)
            image_bar.set_postfix_str(f"{tar_name}")

        # Close images to free memory
        for img in images:
            img.close()

    # Flush writes for this tar
    output_file.flush()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    models_help = "Available models:\n" + "\n".join(
        f"  {name}  ({desc})" for name, desc in MODELS.items()
    )
    detail_help = "Available detail levels:\n" + "\n".join(
        f"  {name:15s} -> {prompt}" for name, prompt in DETAIL_LEVELS.items()
    )

    parser = argparse.ArgumentParser(
        description="Fast DataComp-10M captioning with Florence-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"{models_help}\n\n{detail_help}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODELS.keys(),
        metavar="MODEL",
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--detail",
        default=DEFAULT_DETAIL,
        choices=DETAIL_LEVELS.keys(),
        metavar="LEVEL",
        help=f"Caption detail level (default: {DEFAULT_DETAIL})",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path.home() / "Data/datacomp-proper-wds"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Defaults to data-dir/captions/"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit"],
        default=None,
        help="Quantize model weights (4bit NF4 or 8bit via bitsandbytes). Saves VRAM, allows bigger batches.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose torch.compile/dynamo/inductor logging",
    )
    args = parser.parse_args()

    if args.list_models:
        for name, desc in MODELS.items():
            print(f"{name}  ({desc})")
        return

    if args.verbose:
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

        log.setLevel(logging.DEBUG)
        log.info("Verbose mode: torch dynamo/inductor logging enabled")

    task_prompt = DETAIL_LEVELS[args.detail]

    # Paths
    output_dir = args.output_dir or (args.data_dir / "captions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "datacomp-10m-captions.jsonl"
    checkpoint_path = output_dir / "progress.json"

    # Get tar files
    tar_files = sorted(args.data_dir.glob("*.tar"))
    log.info(f"Found {len(tar_files)} tar files in {args.data_dir}")

    if not tar_files:
        log.error("No tar files found")
        return

    # Guard against accidental overwrites
    if not args.resume and (output_path.exists() or checkpoint_path.exists()):
        log.warning(f"Existing output found: {output_path}")
        print("\nAn existing captioning run was found. What would you like to do?")
        print("  [r] Resume from checkpoint")
        print("  [q] Quit (do nothing)")
        choice = input("\nChoice [r/q]: ").strip().lower()
        if choice == "r":
            args.resume = True
        else:
            log.info("Aborted.")
            return

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    completed_tars = set(checkpoint.get("completed_tars", []))

    tars_remaining = [t for t in tar_files if t.name not in completed_tars]
    log.info(
        f"Resuming: {len(completed_tars)} tars completed"
        if completed_tars
        else "Starting fresh"
    )
    log.info(f"Processing {len(tars_remaining)} remaining tar files...")

    if not tars_remaining:
        log.info("All tars already completed!")
        return

    # Load model
    device = "cuda"
    model, processor = load_model(args.model, device, quantize=args.quantize)

    log.info(
        f"Model: {args.model} | Detail: {args.detail} ({task_prompt}) | "
        f"Batch: {args.batch_size} | "
        f"Quantize: {args.quantize or 'none'}"
    )

    # Process tars
    stats: dict[str, int] = {"total_images": 0, "errors": 0}
    start_time = time.time()
    session_images = 0

    image_bar = tqdm(
        total=len(tars_remaining) * 10_000,
        desc="Images",
        unit="img",
        dynamic_ncols=True,
        position=0,
    )
    tar_bar = tqdm(
        total=len(tars_remaining),
        desc="Tars",
        unit="tar",
        dynamic_ncols=True,
        position=1,
    )

    output_file = open(output_path, "a")
    try:
        for tar_path in tars_remaining:
            if _shutdown_requested:
                break

            count = process_tar(
                tar_path,
                model,
                processor,
                output_file,
                args.batch_size,
                task_prompt,
                device,
                stats,
                image_bar,
            )

            if _shutdown_requested:
                # Don't checkpoint partial tar
                break

            session_images += count
            completed_tars.add(tar_path.name)

            save_checkpoint(
                checkpoint_path,
                {
                    "completed_tars": sorted(completed_tars),
                    "total_images": stats["total_images"],
                    "errors": stats["errors"],
                    "last_update": datetime.now().isoformat(),
                },
            )

            elapsed = time.time() - start_time
            rate = session_images / elapsed if elapsed > 0 else 0
            tar_bar.update(1)
            tar_bar.set_postfix_str(f"{rate:.1f} img/s | {stats['errors']} errors")
    finally:
        output_file.close()

    image_bar.close()
    tar_bar.close()

    total_elapsed = time.time() - start_time
    rate = round(session_images / total_elapsed, 2) if total_elapsed > 0 else 0
    log.info(
        f"Done! {session_images} images in {timedelta(seconds=int(total_elapsed))} ({rate} img/s)"
    )
    if _shutdown_requested:
        log.info("Interrupted — run with --resume to continue.")


if __name__ == "__main__":
    main()

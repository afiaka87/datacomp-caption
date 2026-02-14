#!/usr/bin/env python3
"""
Production captioning pipeline for DataComp-10M using Florence-2.

Streams WebDataset tar files, generates captions, and saves to JSONL
with robust checkpointing and resume support. Supports multi-GPU.

Usage:
    python caption_dataset.py --batch-size 16
    python caption_dataset.py --num-gpus 4
    python caption_dataset.py --gpus 0,2,3
    python caption_dataset.py --resume  # Resume from last checkpoint
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import multiprocessing as mp
import multiprocessing.synchronize
import signal
import sys
import tarfile
import threading
import time
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Florence2ForConditionalGeneration

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
# Graceful shutdown
# ---------------------------------------------------------------------------

# For single-GPU mode, we use a global flag.
# For multi-GPU mode, we use an mp.Event shared across processes.
_shutdown_requested = False
_shutdown_event: multiprocessing.synchronize.Event | None = None


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        log.warning("Forced exit.")
        sys.exit(1)
    log.info("Shutdown requested — finishing current batch then saving checkpoint...")
    _shutdown_requested = True
    if _shutdown_event is not None:
        _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown(
    shutdown_event: multiprocessing.synchronize.Event | None = None,
) -> bool:
    """Check whether shutdown has been requested (works in both modes)."""
    if shutdown_event is not None:
        return shutdown_event.is_set()
    return _shutdown_requested


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def get_gpu_devices(
    num_gpus: int | None = None, gpus: list[int] | None = None
) -> list[str]:
    """Return list of CUDA device strings to use.

    Auto-detects all available GPUs if neither argument is provided.
    --gpus takes precedence over --num-gpus if both are given.
    """
    available = torch.cuda.device_count()
    if available == 0:
        raise ValueError("No CUDA GPUs available")

    if gpus is not None:
        for g in gpus:
            if g >= available:
                raise ValueError(
                    f"GPU {g} not available (only {available} GPUs detected)"
                )
        return [f"cuda:{g}" for g in gpus]

    if num_gpus is not None:
        if num_gpus > available:
            raise ValueError(
                f"Requested {num_gpus} GPUs but only {available} available"
            )
        return [f"cuda:{i}" for i in range(num_gpus)]

    return [f"cuda:{i}" for i in range(available)]


def distribute_tars(tar_files: list[Path], num_gpus: int) -> list[list[Path]]:
    """Split tar files into contiguous chunks, one per GPU.

    Uses ceil division so earlier GPUs may get one extra tar.
    """
    if not tar_files:
        return [[] for _ in range(num_gpus)]

    chunks = []
    n = len(tar_files)
    start = 0
    for i in range(num_gpus):
        # Remaining GPUs get remaining tars split evenly
        remaining_gpus = num_gpus - i
        chunk_size = math.ceil((n - start) / remaining_gpus)
        chunks.append(tar_files[start : start + chunk_size])
        start += chunk_size
    return chunks


def merge_completed_tars(
    tar_order: list[str],
    temp_dir: Path,
    output_path: Path,
    already_merged: set[str],
) -> tuple[set[str], int]:
    """Merge per-tar temp JSONL files into the final output in tar order.

    Scans tar_order sequentially, appending each tar's temp file to output_path
    as long as the next expected temp file exists. Stops at the first gap.

    Returns (set of newly merged tar names, total records written).
    """
    newly_merged = set()
    total_records = 0

    for tar_name in tar_order:
        if tar_name in already_merged:
            continue

        temp_file = temp_dir / f"{tar_name}.jsonl"
        if not temp_file.exists():
            break  # Gap — stop merging

        # Append to output
        with open(output_path, "a") as out_f:
            with open(temp_file) as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_records += 1

        # Clean up temp file
        temp_file.unlink()
        newly_merged.add(tar_name)

    return newly_merged, total_records


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, state: dict):
    tmp = checkpoint_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(checkpoint_path)


# ---------------------------------------------------------------------------
# Tar file reading
# ---------------------------------------------------------------------------


def iter_tar_samples(
    tar_path: str,
) -> Generator[tuple[str, Image.Image | None, str, dict[str, Any]], None, None]:
    """Yield (key, image, original_caption, metadata_dict) from a tar file.

    Groups consecutive files by their stem (key) and emits once complete.
    Handles missing images gracefully (skips sample).
    """
    current_key: str | None = None
    current: dict[str, Any] = {}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            dot = name.rfind(".")
            if dot == -1:
                continue
            key = name[:dot]
            ext = name[dot + 1 :]

            # New sample group
            if key != current_key:
                if current_key is not None and "image" in current:
                    yield (
                        current_key,
                        current.get("image"),
                        current.get("caption", ""),
                        current.get("meta", {}),
                    )
                current_key = key
                current = {}

            if ext in ("jpg", "jpeg", "png", "webp"):
                try:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    current["image"] = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    pass
            elif ext == "txt":
                try:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    current["caption"] = (
                        fobj.read().decode("utf-8", errors="replace").strip()
                    )
                except Exception:
                    pass
            elif ext == "json":
                try:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    current["meta"] = json.load(fobj)
                except Exception:
                    pass

        # Last sample
        if current_key is not None and "image" in current:
            yield (
                current_key,
                current.get("image"),
                current.get("caption", ""),
                current.get("meta", {}),
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(model_id: str, device: str = "cuda") -> tuple[Any, Any]:
    log.info(f"Loading {model_id} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    loaded_model: Any = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model: Any = loaded_model.to(device)
    model.eval()

    # Attempt torch.compile for extra speed (non-fatal if it fails)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log.info("torch.compile succeeded")
    except Exception as e:
        log.info(f"torch.compile skipped: {e}")

    log.info(
        f"Model loaded in {time.time() - t0:.1f}s  |  GPU mem: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
    )
    return model, processor


def generate_captions(
    model,
    processor,
    images: list[Image.Image],
    task_prompt: str,
    device: str = "cuda",
) -> list[str]:
    """Generate captions for a batch of images."""
    prompts = [task_prompt] * len(images)
    inputs = processor(
        text=prompts, images=images, return_tensors="pt", padding=True
    ).to(device, torch.bfloat16)

    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS
        )

    decoded = processor.batch_decode(ids, skip_special_tokens=False)
    captions = []
    for text in decoded:
        parsed = processor.post_process_generation(text, task=task_prompt)
        caption = parsed.get(task_prompt, text.strip())
        caption = caption.replace("<pad>", "").strip()
        captions.append(caption)
    return captions


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_tar_files(data_dir: Path) -> list[Path]:
    """Get sorted list of tar files."""
    tars = sorted(data_dir.glob("*.tar"))
    return tars


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
    shutdown_event: multiprocessing.synchronize.Event | None = None,
) -> int:
    """Process a single tar file. Returns number of images captioned."""
    keys_batch = []
    images_batch = []
    origcaps_batch = []
    metas_batch = []
    count = 0
    tar_name = tar_path.name

    for key, image, orig_caption, meta in iter_tar_samples(str(tar_path)):
        if is_shutdown(shutdown_event):
            break

        keys_batch.append(key)
        images_batch.append(image)
        origcaps_batch.append(orig_caption)
        metas_batch.append(meta)

        if len(images_batch) >= batch_size:
            n = _flush_batch(
                model,
                processor,
                output_file,
                keys_batch,
                images_batch,
                origcaps_batch,
                metas_batch,
                tar_name,
                task_prompt,
                device,
                stats,
            )
            count += n
            if image_bar is not None:
                image_bar.update(n)
                image_bar.set_postfix_str(f"{tar_name} | {keys_batch[0][:12]}...")
            keys_batch, images_batch, origcaps_batch, metas_batch = [], [], [], []

    # Flush remaining
    if images_batch and not is_shutdown(shutdown_event):
        n = _flush_batch(
            model,
            processor,
            output_file,
            keys_batch,
            images_batch,
            origcaps_batch,
            metas_batch,
            tar_name,
            task_prompt,
            device,
            stats,
        )
        count += n
        if image_bar is not None:
            image_bar.update(n)

    return count


def _flush_batch(
    model,
    processor,
    output_file,
    keys,
    images,
    orig_captions,
    metas,
    tar_name,
    task_prompt,
    device,
    stats,
) -> int:
    """Generate captions for a batch and write to output."""
    try:
        captions = generate_captions(model, processor, images, task_prompt, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning(f"OOM on batch of {len(images)} — splitting in half")
        mid = len(images) // 2
        if mid == 0:
            log.error("OOM on single image, skipping")
            stats["errors"] += 1
            return 0
        n = _flush_batch(
            model,
            processor,
            output_file,
            keys[:mid],
            images[:mid],
            orig_captions[:mid],
            metas[:mid],
            tar_name,
            task_prompt,
            device,
            stats,
        )
        n += _flush_batch(
            model,
            processor,
            output_file,
            keys[mid:],
            images[mid:],
            orig_captions[mid:],
            metas[mid:],
            tar_name,
            task_prompt,
            device,
            stats,
        )
        return n
    except Exception as e:
        log.error(f"Error generating captions: {e}")
        stats["errors"] += len(images)
        return 0

    for key, caption, orig, meta in zip(keys, captions, orig_captions, metas):
        record = {
            "key": key,
            "caption": caption,
            "original_caption": orig,
            "tar_file": tar_name,
            "uid": meta.get("uid", ""),
            "url": meta.get("url", ""),
            "width": meta.get("width"),
            "height": meta.get("height"),
        }
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["total_images"] += len(images)
    # Close images to free memory
    for img in images:
        img.close()
    return len(images)


def write_tar_temp(temp_dir: Path, tar_name: str, records: list[dict]):
    """Atomically write a per-tar temp JSONL file."""
    path = temp_dir / f"{tar_name}.jsonl"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------


def worker_process(
    gpu_idx: int,
    device: str,
    tar_files: list[Path],
    model_id: str,
    task_prompt: str,
    batch_size: int,
    temp_dir: Path,
    shutdown_event: multiprocessing.synchronize.Event,
    num_gpus: int = 1,
):
    """Worker process: loads model on assigned GPU, processes assigned tars."""
    worker_log = logging.getLogger(f"worker-{gpu_idx}")
    worker_log.info(f"Starting on {device} with {len(tar_files)} tars")

    model, processor = load_model(model_id, device)

    stats = {"total_images": 0, "errors": 0}

    # Each worker gets its own tqdm bars, offset by rank to avoid overlap
    image_bar = tqdm(
        total=len(tar_files) * 10_000,
        desc=f"GPU {gpu_idx} imgs",
        unit="img",
        dynamic_ncols=True,
        position=gpu_idx * 2,
    )
    tar_bar = tqdm(
        total=len(tar_files),
        desc=f"GPU {gpu_idx} tars",
        unit="tar",
        dynamic_ncols=True,
        position=gpu_idx * 2 + 1,
    )

    start_time = time.time()
    session_images = 0

    for tar_path in tar_files:
        if is_shutdown(shutdown_event):
            break

        tar_name = tar_path.name
        records: list[dict[str, Any]] = []

        count = process_tar_to_records(
            tar_path,
            model,
            processor,
            records,
            batch_size,
            task_prompt,
            device,
            stats,
            shutdown_event,
            image_bar=image_bar,
        )

        if is_shutdown(shutdown_event):
            worker_log.info(f"Shutdown during {tar_name}, discarding partial tar")
            break

        write_tar_temp(temp_dir, tar_name, records)
        session_images += count

        elapsed = time.time() - start_time
        rate = session_images / elapsed if elapsed > 0 else 0
        tar_bar.update(1)
        tar_bar.set_postfix_str(f"{rate:.1f} img/s | {stats['errors']} errors")

    image_bar.close()
    tar_bar.close()
    worker_log.info(
        f"Worker done. {stats['total_images']} images, {stats['errors']} errors"
    )


def process_tar_to_records(
    tar_path: Path,
    model,
    processor,
    records: list[dict],
    batch_size: int,
    task_prompt: str,
    device: str,
    stats: dict,
    shutdown_event: multiprocessing.synchronize.Event | None = None,
    image_bar: tqdm | None = None,
) -> int:
    """Process a tar file, appending records to the list. Returns image count."""
    keys_batch = []
    images_batch = []
    origcaps_batch = []
    metas_batch = []
    count = 0
    tar_name = tar_path.name

    for key, image, orig_caption, meta in iter_tar_samples(str(tar_path)):
        if is_shutdown(shutdown_event):
            break

        keys_batch.append(key)
        images_batch.append(image)
        origcaps_batch.append(orig_caption)
        metas_batch.append(meta)

        if len(images_batch) >= batch_size:
            n = _flush_batch_to_records(
                model,
                processor,
                records,
                keys_batch,
                images_batch,
                origcaps_batch,
                metas_batch,
                tar_name,
                task_prompt,
                device,
                stats,
            )
            count += n
            if image_bar is not None:
                image_bar.update(n)
                image_bar.set_postfix_str(f"{tar_name} | {keys_batch[0][:12]}...")
            keys_batch, images_batch, origcaps_batch, metas_batch = [], [], [], []

    if images_batch and not is_shutdown(shutdown_event):
        n = _flush_batch_to_records(
            model,
            processor,
            records,
            keys_batch,
            images_batch,
            origcaps_batch,
            metas_batch,
            tar_name,
            task_prompt,
            device,
            stats,
        )
        count += n
        if image_bar is not None:
            image_bar.update(n)

    return count


def _flush_batch_to_records(
    model,
    processor,
    records,
    keys,
    images,
    orig_captions,
    metas,
    tar_name,
    task_prompt,
    device,
    stats,
) -> int:
    """Generate captions for a batch and append to records list."""
    try:
        captions = generate_captions(model, processor, images, task_prompt, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning(f"OOM on batch of {len(images)} — splitting in half")
        mid = len(images) // 2
        if mid == 0:
            log.error("OOM on single image, skipping")
            stats["errors"] += 1
            return 0
        n = _flush_batch_to_records(
            model,
            processor,
            records,
            keys[:mid],
            images[:mid],
            orig_captions[:mid],
            metas[:mid],
            tar_name,
            task_prompt,
            device,
            stats,
        )
        n += _flush_batch_to_records(
            model,
            processor,
            records,
            keys[mid:],
            images[mid:],
            orig_captions[mid:],
            metas[mid:],
            tar_name,
            task_prompt,
            device,
            stats,
        )
        return n
    except Exception as e:
        log.error(f"Error generating captions: {e}")
        stats["errors"] += len(images)
        return 0

    for key, caption, orig, meta in zip(keys, captions, orig_captions, metas):
        records.append(
            {
                "key": key,
                "caption": caption,
                "original_caption": orig,
                "tar_file": tar_name,
                "uid": meta.get("uid", ""),
                "url": meta.get("url", ""),
                "width": meta.get("width"),
                "height": meta.get("height"),
            }
        )

    stats["total_images"] += len(images)
    for img in images:
        img.close()
    return len(images)


# ---------------------------------------------------------------------------
# Merger loop (runs in a background thread in the main process)
# ---------------------------------------------------------------------------


def merger_loop(
    tar_order: list[str],
    temp_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    already_merged: set[str],
    shutdown_event: multiprocessing.synchronize.Event,
    poll_interval: float = 2.0,
):
    """Background thread that merges per-tar temp files into the final JSONL in order.

    Runs until shutdown_event is set AND all available files are merged.
    """
    merged = set(already_merged)
    total_images = 0

    while True:
        newly_merged, count = merge_completed_tars(
            tar_order,
            temp_dir,
            output_path,
            merged,
        )
        merged.update(newly_merged)
        total_images += count

        if newly_merged:
            # Update checkpoint
            save_checkpoint(
                checkpoint_path,
                {
                    "completed_tars": sorted(merged),
                    "merged_up_to": _get_merged_cursor(tar_order, merged),
                    "total_images": total_images,
                    "last_update": datetime.now().isoformat(),
                },
            )

        if is_shutdown(shutdown_event):
            # Do one final merge pass then exit
            newly_merged, count = merge_completed_tars(
                tar_order,
                temp_dir,
                output_path,
                merged,
            )
            merged.update(newly_merged)
            total_images += count
            if newly_merged:
                save_checkpoint(
                    checkpoint_path,
                    {
                        "completed_tars": sorted(merged),
                        "merged_up_to": _get_merged_cursor(tar_order, merged),
                        "total_images": total_images,
                        "last_update": datetime.now().isoformat(),
                    },
                )
            break

        # Check if all tars are merged
        if len(merged) == len(tar_order):
            break

        time.sleep(poll_interval)

    log.info(f"Merger done. {len(merged)} tars merged, {total_images} records written.")


def _get_merged_cursor(tar_order: list[str], merged: set[str]) -> str | None:
    """Return the last tar name that has been merged in sequential order."""
    cursor = None
    for name in tar_order:
        if name in merged:
            cursor = name
        else:
            break
    return cursor


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
        description="Caption DataComp-10M with Florence-2",
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
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect all)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use, e.g. '0,2,3' (overrides --num-gpus)",
    )
    args = parser.parse_args()

    if args.list_models:
        for name, desc in MODELS.items():
            print(f"{name}  ({desc})")
        return

    task_prompt = DETAIL_LEVELS[args.detail]

    # Resolve GPU devices
    gpu_list = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    devices = get_gpu_devices(num_gpus=args.num_gpus, gpus=gpu_list)
    num_gpus = len(devices)
    log.info(f"Using {num_gpus} GPU(s): {devices}")

    # Paths
    output_dir = args.output_dir or (args.data_dir / "captions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "datacomp-10m-captions.jsonl"
    checkpoint_path = output_dir / "progress.json"
    stats_path = output_dir / "stats.json"
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Get tar files
    tar_files = get_tar_files(args.data_dir)
    tar_order = [t.name for t in tar_files]
    log.info(f"Found {len(tar_files)} tar files in {args.data_dir}")

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
    already_merged = set()
    if args.resume:
        merged_cursor = checkpoint.get("merged_up_to")
        if merged_cursor:
            for name in tar_order:
                if name in completed_tars:
                    already_merged.add(name)
                if name == merged_cursor:
                    break
        if completed_tars:
            log.info(
                f"Resuming: {len(completed_tars)} tars completed, {len(already_merged)} merged"
            )

    # Remaining tars to process
    tars_remaining = [t for t in tar_files if t.name not in completed_tars]
    log.info(f"Processing {len(tars_remaining)} remaining tar files...")

    if not tars_remaining:
        log.info("All tars already completed!")
        return

    # -----------------------------------------------------------------------
    # Single-GPU fast path (no multiprocessing overhead)
    # -----------------------------------------------------------------------
    if num_gpus == 1:
        log.info(f"Single-GPU mode on {devices[0]}")
        _run_single_gpu(
            args,
            devices[0],
            tar_files,
            tars_remaining,
            tar_order,
            completed_tars,
            already_merged,
            task_prompt,
            output_path,
            checkpoint_path,
            stats_path,
            temp_dir,
        )
        return

    # -----------------------------------------------------------------------
    # Multi-GPU mode
    # -----------------------------------------------------------------------
    global _shutdown_event
    _shutdown_event = mp.Event()

    # Merge any completed-but-not-merged temp files from previous run
    if args.resume:
        newly_merged, count = merge_completed_tars(
            tar_order, temp_dir, output_path, already_merged
        )
        already_merged.update(newly_merged)
        if newly_merged:
            log.info(f"Catch-up merge: {len(newly_merged)} tars, {count} records")

    # Distribute tars
    chunks = distribute_tars(tars_remaining, num_gpus)
    for i, chunk in enumerate(chunks):
        log.info(f"  GPU {i} ({devices[i]}): {len(chunk)} tars")

    log.info(
        f"Model: {args.model} | Detail: {args.detail} ({task_prompt}) | Batch size: {args.batch_size}"
    )

    # Start merger thread
    merger = threading.Thread(
        target=merger_loop,
        args=(
            tar_order,
            temp_dir,
            output_path,
            checkpoint_path,
            already_merged,
            _shutdown_event,
        ),
        daemon=True,
    )
    merger.start()

    # Start worker processes
    ctx = mp.get_context("spawn")
    processes = []
    for i, (device, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue
        p = ctx.Process(
            target=worker_process,
            args=(
                i,
                device,
                chunk,
                args.model,
                task_prompt,
                args.batch_size,
                temp_dir,
                _shutdown_event,
                num_gpus,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    # Signal merger to finish and wait
    _shutdown_event.set()
    merger.join(timeout=30)

    log.info("All workers finished.")

    # Final stats
    cp = load_checkpoint(checkpoint_path)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "completed_tars": len(cp.get("completed_tars", [])),
                "total_tars": len(tar_files),
                "total_images": cp.get("total_images", 0),
                "num_gpus": num_gpus,
                "model": args.model,
                "detail": args.detail,
                "finished": len(cp.get("completed_tars", [])) == len(tar_files),
                "last_update": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


def _run_single_gpu(
    args,
    device,
    tar_files,
    tars_remaining,
    tar_order,
    completed_tars,
    already_merged,
    task_prompt,
    output_path,
    checkpoint_path,
    stats_path,
    temp_dir,
):
    """Single-GPU code path — writes per-tar temp files and merges inline."""
    stats = {"total_images": 0, "errors": 0}

    model, processor = load_model(args.model, device)

    log.info(
        f"Model: {args.model} | Detail: {args.detail} ({task_prompt}) | Batch size: {args.batch_size}"
    )

    start_time = time.time()
    session_images = 0

    image_bar = tqdm(
        total=len(tars_remaining) * 10_000,
        desc="GPU 0 imgs",
        unit="img",
        dynamic_ncols=True,
        position=0,
    )
    tar_bar = tqdm(
        total=len(tars_remaining),
        desc="GPU 0 tars",
        unit="tar",
        dynamic_ncols=True,
        position=1,
    )

    for tar_path in tars_remaining:
        if _shutdown_requested:
            break

        records: list[dict[str, Any]] = []
        count = process_tar_to_records(
            tar_path,
            model,
            processor,
            records,
            args.batch_size,
            task_prompt,
            device,
            stats,
            image_bar=image_bar,
        )

        if _shutdown_requested:
            break

        # Write temp file atomically
        write_tar_temp(temp_dir, tar_path.name, records)
        session_images += count
        image_bar.update(count)

        # Merge immediately (single GPU, so always in order)
        completed_tars.add(tar_path.name)
        newly_merged, _ = merge_completed_tars(
            tar_order, temp_dir, output_path, already_merged
        )
        already_merged.update(newly_merged)

        save_checkpoint(
            checkpoint_path,
            {
                "completed_tars": sorted(completed_tars),
                "merged_up_to": _get_merged_cursor(tar_order, already_merged),
                "total_images": stats["total_images"],
                "errors": stats["errors"],
                "last_update": datetime.now().isoformat(),
            },
        )

        elapsed = time.time() - start_time
        rate = session_images / elapsed if elapsed > 0 else 0
        tar_bar.update(1)
        tar_bar.set_postfix_str(f"{rate:.1f} img/s | {stats['errors']} errors")

    image_bar.close()
    tar_bar.close()

    total_elapsed = time.time() - start_time
    rate = round(session_images / total_elapsed, 2) if total_elapsed > 0 else 0
    with open(stats_path, "w") as f:
        json.dump(
            {
                "session_images": session_images,
                "session_elapsed_sec": round(total_elapsed, 1),
                "session_rate": rate,
                "completed_tars": len(completed_tars),
                "total_tars": len(tar_files),
                "total_images": stats["total_images"],
                "errors": stats["errors"],
                "model": args.model,
                "detail": args.detail,
                "finished": len(completed_tars) == len(tar_files),
                "last_update": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    log.info(
        f"Done! {session_images} images in {timedelta(seconds=int(total_elapsed))} ({rate} img/s)"
    )
    if _shutdown_requested:
        log.info("Interrupted — run with --resume to continue.")


if __name__ == "__main__":
    main()

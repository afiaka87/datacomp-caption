# datacomp-caption

Generate synthetic captions for [DataComp](https://www.datacomp.ai/) image datasets stored in WebDataset tar format using [Florence-2](https://huggingface.co/florence-community/Florence-2-base).

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# Single GPU (auto-detected)
./run-default.sh

# Multiple GPUs (auto-detects all available)
./run-multi-gpu.sh

# Specify GPU count or specific GPUs
python caption_dataset.py --num-gpus 4
python caption_dataset.py --gpus 0,2,3

# Resume an interrupted run
python caption_dataset.py --resume

# All options
python caption_dataset.py \
    --model florence-community/Florence-2-base \
    --detail detailed \
    --batch-size 16 \
    --data-dir ~/Data/datacomp-proper-wds \
    --output-dir ~/Data/datacomp-proper-wds/captions \
    --resume
```

## Features

- Reads WebDataset tar files containing `.jpg`/`.webp` images with `.txt` captions and `.json` metadata
- Generates captions with Florence-2 (base or large) at three detail levels: `brief`, `detailed`, `more_detailed`
- Multi-GPU support via multiprocessing with contiguous tar file sharding across GPUs
- Outputs a single ordered JSONL file with incremental merging from per-tar temp files
- Checkpoint/resume support -- can resume with a different GPU count than the original run
- Graceful shutdown on SIGINT/SIGTERM, saving progress before exit
- OOM recovery by splitting batches in half

## Models

| Model | Params | Approximate speed |
|-------|--------|-------------------|
| `florence-community/Florence-2-base` | 0.23B | ~24 img/s per GPU |
| `florence-community/Florence-2-large` | 0.77B | ~4 img/s per GPU |

## Output format

Each line in the output JSONL contains:

```json
{
  "key": "00000001",
  "caption": "The generated caption text",
  "original_caption": "Original caption from the dataset",
  "tar_file": "0000000.tar",
  "uid": "abc123",
  "url": "https://...",
  "width": 512,
  "height": 384
}
```

## Tests

```bash
uv run pytest
```

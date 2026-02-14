#!/usr/bin/env bash
set -euo pipefail

exec uv run python caption_dataset.py \
    --model florence-community/Florence-2-large \
    --detail detailed \
    --batch-size 64 \
    --num-gpus 2 \
    "$@"

#!/usr/bin/env bash
# Execute from the repository root.
set -euo pipefail
E=experiments/sunset_cloudy_v3_evaluation_2026-11-17
CHECKPOINT=experiments/day_cloudy_cnn_close_v3_2026-11-17/finetune_v3/best.pth
export OMP_NUM_THREADS=2
for archive in carla_following combined_800; do
  .venv/bin/python -B -u evaluate_distance_cnn.py --checkpoint "$CHECKPOINT" \
    --data "dataset/$archive/sunset_cloudy" --all-episodes --batch-size 128 \
    --output "$E/$archive" > "$E/$archive.log" 2>&1
done

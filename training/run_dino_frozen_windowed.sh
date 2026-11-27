#!/usr/bin/env bash
# Frozen DINOv2 Small comparison with W&B and the two-camera viewer.
# Extra CLI flags override these defaults, e.g. --steps 100000 --eval-interval 5000 --eval-cases 50.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
run_name="hj-dino-frozen-day_sunny-$(date +%Y%m%d_%H%M%S)"

exec .venv/bin/python -u -m training.train_dino_hj \
    --outdir "training/_local/$run_name" \
    --road-environment straight --straight-vehicle vehicle.audi.tt \
    --condition day_sunny --performance-policy safety \
    --safety-objective following --min-gap 1 --max-gap 9 \
    --termination-gap-range 0.2 11 \
    --gap-range 2 8 --challenge-gap-range 1.5 3 \
    --history-frames 4 --speed-range 10 20 \
    --random-steps 1000 --learning-starts 1000 \
    --exploration-episode-fraction 0 --exploration-hold-steps 5 15 \
    --gamma-start 0.85 --gamma-final 0.99 --gamma-anneal-steps 100000 \
    --alpha 0.05 --alpha-final 0.05 \
    --dino-unfreeze-blocks 0 \
    --steps 100000 --eval-interval 5000 --eval-cases 50 \
    --action-diagnostics-interval 100 \
    --port 2990 --start-server --server .local/carla-0.9.15/CarlaUE4.sh \
    --require-cuda --windowed \
    --wandb --wandb-project carla-sage \
    --wandb-entity a-h-hamzeh-washington-university-in-st-louis --wandb-name "$run_name" \
    "$@"

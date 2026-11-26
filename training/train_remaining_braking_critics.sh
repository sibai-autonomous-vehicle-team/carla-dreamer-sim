#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

conditions=(
  day_cloudy
  day_rainy
  sunset_sunny
  sunset_cloudy
  sunset_rainy
  night_sunny
  night_cloudy
  night_rainy
)

for condition in "${conditions[@]}"; do
  output="training/_local/${condition}_critic_500_s42"
  if [[ -f "$output/checkpoint.pt" ]]; then
    printf 'Skipping completed model %s\n' "$condition"
    continue
  fi

  printf '\nTraining %s\n' "$condition"
  .venv/bin/python -u -m training.train_braking_critic \
    --source "dataset/braking_lane_tracked_500_per_condition/$condition" \
    --output "$output" \
    --episodes 500 \
    --train-episodes 400 \
    --epochs 5 \
    --batch-size 8 \
    --history-frames 5 \
    --lr 0.0001 \
    --seed 42
done

printf '\nAll eight remaining critic models are complete.\n'

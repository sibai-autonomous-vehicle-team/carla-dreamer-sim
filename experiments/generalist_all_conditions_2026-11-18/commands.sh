#!/usr/bin/env bash
# Run from the repository root.
set -euo pipefail
E=experiments/generalist_all_conditions_2026-11-18
DATA=dataset/combined_800
CONDITIONS=(day_sunny day_cloudy day_rainy sunset_sunny sunset_cloudy sunset_rainy
            night_sunny night_cloudy night_rainy)
export OMP_NUM_THREADS=2
export TORCH_HOME="$PWD/.local/torch"

train() {
  .venv/bin/python -B -u train_distance_cnn.py --mixed-data "$DATA" --epochs 10 \
    --batch-size 128 --seed 42 --amp --output "$E/training" > "$E/training.log" 2>&1
}

evaluate() {
  mkdir -p "$E/evaluation"
  for condition in "${CONDITIONS[@]}"; do
    mkdir -p "$E/evaluation/$condition"
    .venv/bin/python -B -u evaluate_distance_cnn.py --data "$DATA/$condition" \
      --checkpoint "$E/training/best.pth" --split "$E/training/splits/$condition.json" \
      --batch-size 128 --output "$E/evaluation/$condition" \
      > "$E/evaluation/$condition/evaluation.log" 2>&1
  done
  .venv/bin/python -B -m analysis.summarize_generalist --experiment "$E"
  .venv/bin/python -B -m analysis.plot_distance_cnn_curves --epochs "$E/training/epochs.csv" \
    --output "$E/training_curves.png"
}

publish() {
  .venv/bin/python -B "$E/publish_generalist.py"
}

case "${1:-}" in
  train) train ;;
  evaluate) evaluate ;;
  publish) publish ;;
  all) train; evaluate; publish ;;
  *) printf '%s\n' 'Usage: bash commands.sh {train|evaluate|publish|all}' >&2; exit 2 ;;
esac

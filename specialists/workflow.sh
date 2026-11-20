#!/usr/bin/env bash
# Run from the repository root. Use one CARLA client at a time.
set -euo pipefail
if [[ $# != 3 ]]; then
  printf '%s\n' 'Usage: bash specialists/workflow.sh CONDITION STAGE EXPERIMENT_DIR' >&2
  exit 2
fi
CONDITION="$1"
STAGE="$2"
E="$3"
if [[ ! "$CONDITION" =~ ^(day|sunset|night)_(sunny|cloudy|rainy)$ ]]; then
  printf 'Unknown specialist condition: %s\n' "$CONDITION" >&2
  exit 2
fi
PROFILE="specialists/$CONDITION"
DATA="dataset/combined_800/$CONDITION"
ROUTES=specialists/routes.json
DAY=specialists/day_cloudy/cnn.pth
ARCHIVE="$E/archive/best.pth"
SYNC="$E/sync_model/best.pth"
export OMP_NUM_THREADS=2
export TORCH_HOME="$PWD/.local/torch"
mkdir -p "$E"
if [[ -f "$E/condition.txt" ]]; then
  if [[ $(<"$E/condition.txt") != "$CONDITION" ]]; then
    printf 'Experiment directory belongs to a different condition: %s\n' "$E" >&2
    exit 2
  fi
elif [[ -e "$E/archive" || -e "$E/standard_captures" || -e "$E/close_captures" || -e "$E/sync_model" ]]; then
  printf 'Existing experiment data lack a condition record: %s\n' "$E" >&2
  exit 2
else
  printf '%s\n' "$CONDITION" > "$E/condition.txt"
fi

case "$STAGE" in
  server)
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      .local/carla-0.9.15/CarlaUE4.sh -RenderOffScreen -quality-level=Epic -nosound \
      -carla-rpc-port=2000 > "$E/carla-server.log" 2>&1 ;;
  train_archive)
    precision=()
    if [[ "${DISTANCE_CNN_AMP:-0}" == 1 ]]; then precision=(--amp); fi
    .venv/bin/python -B -u train_distance_cnn.py --data "$DATA" --epochs 10 \
      --batch-size 128 --seed 42 "${precision[@]}" --output "$E/archive" > "$E/training.log" 2>&1 ;;
  collect)
    .venv/bin/python -B -u -m analysis.collect_distance_cnn_sync --port 2000 \
      --condition "$CONDITION" --routes "$ROUTES" --output "$E/standard_captures" \
      --steps 145 --target-gaps 4.5 7.5 10.5 --capture-every 2 > "$E/standard_collection.log" 2>&1
    .venv/bin/python -B -u -m analysis.collect_distance_cnn_sync --port 2000 \
      --condition "$CONDITION" --routes "$ROUTES" --output "$E/close_captures" \
      --close-gaps --target-gaps 0.65 1.5 2.5 --steps 90 --capture-every 2 --seed 6200 \
      > "$E/close_collection.log" 2>&1 ;;
  train_sync)
    .venv/bin/python -B -m analysis.merge_distance_cnn_splits \
      --standard "$E/standard_captures/split.json" --close "$E/close_captures/split.json" \
      --condition "$CONDITION" --output "$E/merged_split.json"
    .venv/bin/python -B -u -m analysis.finetune_distance_cnn_sync --manifest "$E/merged_split.json" \
      --checkpoint "$ARCHIVE" --legacy-data "$DATA" --output "$E/sync_model" \
      --close-aware --epochs 20 --batch-size 32 --replay-frames 0 > "$E/finetuning.log" 2>&1 ;;
  publish)
    CHECKPOINT="$ARCHIVE"
    if [[ -f "$SYNC" ]]; then CHECKPOINT="$SYNC"; fi
    .venv/bin/python -B specialists/publish.py "$PROFILE" "$CHECKPOINT" ;;
  evaluate)
    .venv/bin/python -B -u evaluate_distance_cnn.py --checkpoint "$DAY" --data "$DATA" \
      --split "$E/archive/split.json" --batch-size 128 --output "$E/day_transfer" \
      > "$E/day_transfer.log" 2>&1
    .venv/bin/python -B -u evaluate_distance_cnn.py --checkpoint "$ARCHIVE" --data "$DATA" \
      --batch-size 128 --output "$E/archive_evaluation" > "$E/archive_evaluation.log" 2>&1
    .venv/bin/python -B -u evaluate_distance_cnn.py --checkpoint "$PROFILE/cnn.pth" --data "$DATA" \
      --split "$E/archive/split.json" --batch-size 128 --output "$E/specialist_evaluation" \
      > "$E/specialist_evaluation.log" 2>&1 ;;
  plots)
    .venv/bin/python -B -m analysis.plot_distance_cnn_curves --epochs "$E/archive/epochs.csv" \
      --output "$E/archive/training_curves.png"
    .venv/bin/python -B -m analysis.plot_distance_cnn_curves --epochs "$E/sync_model/epochs.csv" \
      --output "$E/sync_model/training_curves.png" ;;
  live)
    mkdir -p "$E/live"
    base=(--port 2000 --pid-profile "$PROFILE/pid.json")
    .venv/bin/python -B -u run_gap_pid.py "${base[@]}" --output "$E/live/ground_truth" \
      > "$E/live/ground_truth.log" 2>&1
    .venv/bin/python -B -u run_gap_pid.py "${base[@]}" --checkpoint "$DAY" \
      --output "$E/live/day_transfer" > "$E/live/day_transfer.log" 2>&1
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --specialist "$PROFILE" \
      --output "$E/live/specialist" > "$E/live/specialist.log" 2>&1 ;;
  *) printf 'Unknown workflow stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac

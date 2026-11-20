#!/usr/bin/env bash
# Run from the repository root. Use one CARLA client at a time.
set -euo pipefail
E=experiments/day_cloudy_cnn_close_v3_2026-11-17
PREVIOUS=experiments/day_cloudy_cnn_v2_2026-11-17
V1=experiments/day_cloudy_resnet18_pid_2026-11-17/best.pth
ROUTES="$PREVIOUS/routes.json"
V2="$PREVIOUS/finetune/best.pth"
V3="$E/finetune_v3/best.pth"
export OMP_NUM_THREADS=2

rollout() {
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --shared-route \
        --episodes 5 --steps 360 --seed 42 --warmup-steps 5 \
        --kp 0.9 --ki 0.05 --kd 0.15 "$@"
}

case "${1:-}" in
  server)
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      .local/carla-0.9.15/CarlaUE4.sh -RenderOffScreen -quality-level=Epic -nosound \
      -carla-rpc-port=2000 > "$E/carla-server.log" 2>&1 ;;
  collect)
    .venv/bin/python -B -u -m analysis.collect_distance_cnn_sync --port 2000 \
      --routes "$ROUTES" --output "$E/close_captures" --close-gaps \
      --target-gaps 0.65 1.5 2.5 --steps 90 --capture-every 2 --seed 6200 \
      > "$E/collection.log" 2>&1 ;;
  train)
    .venv/bin/python -B "$E/merge_splits.py"
    .venv/bin/python -B -u -m analysis.finetune_distance_cnn_sync \
      --manifest "$E/merged_split.json" --checkpoint "$V2" --legacy-reference "$V1" \
      --output "$E/finetune_v3" --close-aware --epochs 20 --batch-size 32 \
      --legacy-data dataset/combined_800/day_cloudy --replay-frames 0 > "$E/training.log" 2>&1 ;;
  normal_ground_truth|normal_v2|normal_v3|tight_ground_truth|tight_v2|tight_v3)
    mode="${1#*_}"
    scenario="${1%%_*}"
    conditions=(--target-gap 6)
    if [[ "$scenario" == tight ]]; then conditions=(--target-gap 2.5 --lead-start-index 6); fi
    checkpoint=()
    if [[ "$mode" == v2 ]]; then checkpoint=(--checkpoint "$V2"); fi
    if [[ "$mode" == v3 ]]; then checkpoint=(--checkpoint "$V3"); fi
    mkdir -p "$E/$scenario"
    rollout --route-manifest "$ROUTES" --route-split test "${conditions[@]}" \
      "${checkpoint[@]}" --output "$E/$scenario/$mode" > "$E/$scenario/$mode.log" 2>&1 ;;
  development)
    rollout --target-gap 6 --checkpoint "$V3" \
      --output "$E/development_v3" > "$E/development_v3.log" 2>&1 ;;
  plot_close)
    .venv/bin/python -B "$E/plot_close.py" ;;
  summary)
    .venv/bin/python -B "$E/summarize_v3.py" ;;
  *) exit 2 ;;
esac

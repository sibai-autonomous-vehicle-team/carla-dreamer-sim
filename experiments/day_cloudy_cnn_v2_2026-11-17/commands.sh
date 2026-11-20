#!/usr/bin/env bash
# Execute from the repository root with one client at a time on port 2000.
set -euo pipefail
E=experiments/day_cloudy_cnn_v2_2026-11-17
V1=experiments/day_cloudy_resnet18_pid_2026-11-17/best.pth
V2="$E/finetune/best.pth"
export OMP_NUM_THREADS=2

rollout() {
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --shared-route --episodes 5 \
        --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 \
        --kp 0.9 --ki 0.05 --kd 0.15 "$@"
}
case "${1:-}" in
  server)
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      .local/carla-0.9.15/CarlaUE4.sh -RenderOffScreen -quality-level=Epic -nosound \
      -carla-rpc-port=2000 > "$E/carla-server.log" 2>&1 ;;
  collect)
    .venv/bin/python -B -u -m analysis.collect_distance_cnn_sync --port 2000 \
      --routes "$E/routes.json" --output "$E/captures" --steps 145 \
      --target-gaps 4.5 7.5 10.5 --capture-every 2 > "$E/collection.log" 2>&1 ;;
  train)
    .venv/bin/python -B -u -m analysis.finetune_distance_cnn_sync \
      --manifest "$E/captures/split.json" --checkpoint "$V1" --output "$E/finetune" \
      --epochs 16 --batch-size 32 --legacy-data dataset/combined_800/day_cloudy \
      --replay-frames 0 > "$E/training.log" 2>&1 ;;
  ground_truth)
    mkdir -p "$E/live"
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --shared-route --route-manifest "$E/routes.json" \
      --route-split test --episodes 5 --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 \
      --kp 0.9 --ki 0.05 --kd 0.15 --output "$E/live/ground_truth" > "$E/live/ground_truth.log" 2>&1 ;;
  old_model)
    rollout --route-manifest "$E/routes.json" --route-split test \
      --checkpoint "$V1" --output "$E/live/cnn_v1" > "$E/live/cnn_v1.log" 2>&1 ;;
  new_model)
    rollout --route-manifest "$E/routes.json" --route-split test \
      --checkpoint "$V2" --output "$E/live/cnn_v2" > "$E/live/cnn_v2.log" 2>&1 ;;
  development)
    rollout --checkpoint "$V2" --output "$E/development_v2" > "$E/development_v2.log" 2>&1 ;;
  plot_predictions)
    .venv/bin/python -B "$E/plot_predictions.py" ;;
  summary)
    .venv/bin/python -B "$E/summarize_live.py" ;;
  *) exit 2 ;;
esac

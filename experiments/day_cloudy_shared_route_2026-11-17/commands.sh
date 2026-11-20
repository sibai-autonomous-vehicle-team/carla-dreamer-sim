#!/usr/bin/env bash
# Execute from the repo root. Start the server in its own terminal.
# Use one simulator client at a time. Logs and output go into this experiment.
set -euo pipefail
E=experiments/day_cloudy_shared_route_2026-11-17
CHECKPOINT=experiments/day_cloudy_resnet18_pid_2026-11-17/best.pth
export OMP_NUM_THREADS=2
case "${1:-}" in
  server)
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      .local/carla-0.9.15/CarlaUE4.sh -RenderOffScreen -quality-level=Epic -nosound \
      -carla-rpc-port=2000 > "$E/carla-server.log" 2>&1 ;;
  smoke)
    .venv/bin/python -B -u run_gap_pid.py --shared-route --port 2000 --episodes 1 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 \
      --output "$E/straight_smoke" > "$E/straight_smoke.log" 2>&1 ;;
  ground_truth)
    .venv/bin/python -B -u run_gap_pid.py --shared-route --port 2000 --episodes 5 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 \
      --output "$E/ground_truth" > "$E/ground_truth.log" 2>&1 ;;
  cnn)
    .venv/bin/python -B -u run_gap_pid.py --shared-route --port 2000 --episodes 5 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 --checkpoint "$CHECKPOINT" \
      --output "$E/cnn" > "$E/cnn.log" 2>&1 ;;
  audit)
    .venv/bin/python -B -u -m analysis.audit_distance_cnn_sync --port 2000 --shared-route \
      --checkpoint "$CHECKPOINT" --output "$E/sync_audit" > "$E/sync_audit.log" 2>&1 ;;
  gain_variant)
    VARIANT="$E/gain_variant"
    mkdir -p "$VARIANT"
    .venv/bin/python -B -u run_gap_pid.py --shared-route --port 2000 --episodes 5 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 --kp 0.9 --ki 0.05 --kd 0.15 \
      --output "$VARIANT/ground_truth" > "$VARIANT/ground_truth.log" 2>&1
    .venv/bin/python -B -u run_gap_pid.py --shared-route --port 2000 --episodes 5 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 --kp 0.9 --ki 0.05 --kd 0.15 \
      --checkpoint "$CHECKPOINT" --output "$VARIANT/cnn" > "$VARIANT/cnn.log" 2>&1 ;;
  repeat_42)
    REPEAT="$E/gain_variant/repeat_42"
    mkdir -p "$REPEAT"
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --shared-route --episodes 1 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 --kp 0.9 --ki 0.05 --kd 0.15 \
      --output "$REPEAT/ground_truth" > "$REPEAT/ground_truth.log" 2>&1
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --shared-route --episodes 1 \
      --steps 360 --seed 42 --target-gap 6 --warmup-steps 5 --kp 0.9 --ki 0.05 --kd 0.15 \
      --checkpoint "$CHECKPOINT" --output "$REPEAT/cnn" > "$REPEAT/cnn.log" 2>&1 ;;
  summary)
    .venv/bin/python -B "$E/summarize_routes.py" ;;
  summary_gain_variant)
    .venv/bin/python -B "$E/summarize_routes.py" --root "$E/gain_variant" ;;
  *)
    printf '%s\n' 'Usage: bash commands.sh {server|smoke|ground_truth|cnn|audit|gain_variant|repeat_42|summary|summary_gain_variant}' >&2
    exit 2 ;;
esac

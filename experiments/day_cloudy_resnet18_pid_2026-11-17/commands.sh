#!/usr/bin/env bash
# Executed from the repository root. Run the server command in its own shell.
set -euo pipefail
E=experiments/day_cloudy_resnet18_pid_2026-11-17
export OMP_NUM_THREADS=2
export TORCH_HOME="$PWD/.local/torch"
case "${1:-}" in
  setup)
    UV_CACHE_DIR="$PWD/.local/uv-cache" .local/bin/uv pip install --python .venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cu124 torchvision==0.21.0+cu124 > "$E/setup.log" 2>&1 ;;
  train)
    .venv/bin/python -B -u train_distance_cnn.py --data dataset/combined_800/day_cloudy --epochs 10 --batch-size 128 --seed 42 --output "$E" > "$E/training.log" 2>&1 ;;
  verify)
    .venv/bin/python -B "$E/verify_dataset.py" > "$E/dataset_verification.log" 2>&1 ;;
  server)
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json .local/carla-0.9.15/CarlaUE4.sh -RenderOffScreen -quality-level=Epic -nosound -carla-rpc-port=2000 > "$E/carla-server.log" 2>&1 ;;
  smoke)
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --episodes 1 --steps 5 --seed 42 --target-gap 6 --output "$E/smoke" > "$E/carla-smoke.log" 2>&1 ;;
  evaluate)
    .venv/bin/python -B -u evaluate_distance_cnn.py --checkpoint "$E/best.pth" --data dataset/combined_800/day_cloudy --batch-size 128 --output "$E" > "$E/evaluation.log" 2>&1 ;;
  live)
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --episodes 5 --steps 200 --warmup-steps 5 --seed 42 --target-gap 6 --kp 0.5 --ki 0 --kd 0.5 --diagnostic-checkpoint "$E/best.pth" --output "$E/ground_truth" > "$E/ground_truth.log" 2>&1
    .venv/bin/python -B -u run_gap_pid.py --port 2000 --episodes 5 --steps 200 --warmup-steps 5 --seed 42 --target-gap 6 --kp 0.5 --ki 0 --kd 0.5 --checkpoint "$E/best.pth" --output "$E/cnn" > "$E/cnn.log" 2>&1 ;;
  plot)
    .venv/bin/python -B "$E/plot_experiment.py" ;;
  *) exit 2 ;;
esac

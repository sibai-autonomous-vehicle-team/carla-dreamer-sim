#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Usage: ./collect_both.sh [new-output-root] [trajectories-per-config=500]
outdir="${1:-dataset/both_collectors_$(date +%Y%m%d_%H%M%S)}"
n_traj="${2:-500}"
python="${PYTHON:-.venv/bin/python}"
base_port="${BASE_PORT:-2000}"
if [[ ! "$n_traj" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Trajectory count must be a positive integer.' >&2
  exit 1
fi
if [[ ! "$base_port" =~ ^[1-9][0-9]*$ ]] || ((base_port > 59362)); then
  echo 'BASE_PORT must be an integer between 1 and 59362.' >&2
  exit 1
fi
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
mkdir -p -- "$outdir"
if [[ -n "$(ls -A -- "$outdir")" ]]; then
  echo "Output directory must be empty: $outdir" >&2
  exit 1
fi

for daytime in day sunset night; do
  for weather in sunny cloudy rainy; do
    echo "Collecting $n_traj trajectories per collector: $daytime / $weather"
    "$python" -B -u collect_parallel.py \
      --outdir "$outdir/${daytime}_${weather}" --n-traj "$n_traj" \
      --base-port "$base_port" \
      --first-collector following --second-collector lane-offsets --lane-frames 100 \
      --first-time-of-day "$daytime" --second-time-of-day "$daytime" \
      --first-weather "$weather" --second-weather "$weather"
    # Avoid recently closed TCP sockets from the preceding configuration.
    base_port=$((base_port + 20))
  done
done
echo "Both collectors completed all nine configurations: $outdir"

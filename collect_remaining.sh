#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# The four completed configurations are in dataset/both_collectors_run01.
outdir="${1:-dataset/remaining_following_$(date +%Y%m%d_%H%M%S)}"
n_traj="${2:-500}"
python="${PYTHON:-.venv/bin/python}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
if [[ ! "$n_traj" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Trajectory count must be a positive integer.' >&2
  exit 1
fi
mkdir -p -- "$outdir"
if [[ -n "$(ls -A -- "$outdir")" ]]; then
  echo "Output directory must be empty: $outdir" >&2
  exit 1
fi

conditions=(sunset:cloudy sunset:rainy night:sunny night:cloudy night:rainy)
for ((i=0; i<${#conditions[@]}; i+=2)); do
  IFS=: read -r first_time first_weather <<< "${conditions[i]}"
  args=(--n-traj "$n_traj" --outdir "$outdir/batch_$((i / 2 + 1))"
        --base-port "$((2200 + i * 10))"
        --first-collector following --second-collector following
        --first-time-of-day "$first_time" --first-weather "$first_weather")
  if ((i + 1 < ${#conditions[@]})); then
    IFS=: read -r second_time second_weather <<< "${conditions[i+1]}"
    args+=(--second-time-of-day "$second_time" --second-weather "$second_weather")
  else
    args+=(--single)
  fi
  "$python" -B -u collect_parallel.py "${args[@]}"
done
echo "All five remaining following datasets completed: $outdir"

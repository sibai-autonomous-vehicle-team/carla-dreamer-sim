#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

outdir="${1:-dataset/full_range_$(date +%Y%m%d_%H%M%S)}"
n_traj="${2:-500}"
python="${PYTHON:-.venv/bin/python}"
workers="${WORKERS:-1}"
if [[ ! "$n_traj" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Trajectory count must be a positive integer.' >&2
  exit 1
fi
if [[ "$workers" != 1 && "$workers" != 2 ]]; then
  echo 'WORKERS must be 1 or 2.' >&2
  exit 1
fi
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
mkdir -p -- "$outdir"
if [[ -n "$(ls -A -- "$outdir")" ]]; then
  echo "Output directory must be empty: $outdir" >&2
  exit 1
fi
conditions=(day:sunny day:cloudy day:rainy sunset:sunny sunset:cloudy sunset:rainy night:sunny night:cloudy night:rainy)
for ((i=0; i<${#conditions[@]}; i+=workers)); do
  IFS=: read -r first_time first_weather <<< "${conditions[i]}"
  args=(--n-traj "$n_traj" --controller full-range
        --base-port "$((2400 + i * 10))" --outdir "$outdir/batch_$((i / workers + 1))"
        --first-collector following --second-collector following
        --first-time-of-day "$first_time" --first-weather "$first_weather")
  if ((workers == 2 && i + 1 < ${#conditions[@]})); then
    IFS=: read -r second_time second_weather <<< "${conditions[i+1]}"
    args+=(--second-time-of-day "$second_time" --second-weather "$second_weather")
  else
    args+=(--single)
  fi
  "$python" -B -u collect_parallel.py "${args[@]}"
done
echo "All nine datasets completed with the full-range controller: $outdir"

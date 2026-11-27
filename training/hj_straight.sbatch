#!/bin/bash
#SBATCH --job-name=hj-straight
#SBATCH --partition=general-gpu
#SBATCH --gpus=1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

workspace=/storage3/fs1/netanel.raviv/Active/carla_workspace
cd "$workspace/carla-dreamer-sim"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export OMP_NUM_THREADS=8

condition="${CONDITION:-day_sunny}"
run_name="hj-straight-${condition}-${SLURM_JOB_ID}"
# Controlled straight-road visual HJ training; offscreen RGB rendering remains enabled.
# STEPS and CONDITION can be supplied with sbatch --export.
timeout --signal=INT --kill-after=120s 42300s \
    "$workspace/sage-env/bin/python" -u -m training.train_online_hj \
    --outdir "training/_local/$run_name" \
    --road-environment straight --straight-vehicle vehicle.audi.tt \
    --condition "$condition" --performance-policy safety \
    --safety-objective following --min-gap 1 --max-gap 9 \
    --termination-gap-range 0.2 11 \
    --gap-range 2 8 --challenge-gap-range 1.5 3 \
    --history-frames 4 --speed-range 10 20 \
    --gamma-start 0.85 --gamma-final 0.99 \
    --alpha 0.05 --alpha-final 0 --anneal-updates 10000 \
    --random-steps 1000 --learning-starts 1000 \
    --exploration-episode-fraction 0 --exploration-hold-steps 5 15 \
    --port 2970 --start-server \
    --server "$workspace/carla-0.9.15/CarlaUE4.sh" \
    --require-cuda --steps "${STEPS:-20000}" \
    --action-diagnostics-interval 100 \
    --eval-interval 5000 --eval-cases 50 \
    --wandb --wandb-project carla-sage --wandb-name "$run_name" \
    "$@" \
    || { status=$?; [[ "$status" -eq 124 ]] || exit "$status"; }

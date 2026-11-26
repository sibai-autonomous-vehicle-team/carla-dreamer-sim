#!/bin/bash
#SBATCH --job-name=sage-cloudy
#SBATCH --partition=general-gpu
#SBATCH --gpus=1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

workspace=/storage3/fs1/netanel.raviv/Active/carla_workspace
cd "$workspace/carla-dreamer-sim"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export OMP_NUM_THREADS=8

run_name="sage-cloudy-overnight-${SLURM_JOB_ID}"
# Reserve time for an interrupted checkpoint and CARLA cleanup before Slurm ends.
# Checkpoints do not currently support resuming training.
timeout --signal=INT --kill-after=120s 42300s \
    "$workspace/sage-env/bin/python" -u -m training.train_online_sage \
    --outdir "training/_local/$run_name" \
    --condition day_cloudy \
    --port 2810 --start-server \
    --server "$workspace/carla-0.9.15/CarlaUE4.sh" \
    --require-cuda --steps 100000 \
    --eval-interval 5000 --eval-cases 50 \
    --wandb --wandb-project carla-sage --wandb-name "$run_name" \
    || { status=$?; [[ "$status" -eq 124 ]] || exit "$status"; }

#!/bin/bash
#SBATCH --job-name=sage-diagnostic
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

run_name="sage-cloudy-no-safety-entropy-g099-${SLURM_JOB_ID}"
# Fresh diagnostic run: fixed safety discount, zero safety entropy.
# Goal SAC keeps entropy 0.2. Later --resume restores training with fresh replay.
timeout --signal=INT --kill-after=120s 42300s \
    "$workspace/sage-env/bin/python" -u -m training.train_online_sage \
    --outdir "training/_local/$run_name" \
    --condition day_cloudy \
    --port 2810 --start-server \
    --server "$workspace/carla-0.9.15/CarlaUE4.sh" \
    --require-cuda --steps 20000 \
    --alpha 0 --alpha-final 0 \
    --gamma-start 0.99 --gamma-final 0.99 \
    --performance-alpha 0.2 \
    --eval-interval 5000 --eval-cases 50 \
    --wandb --wandb-project carla-sage --wandb-name "$run_name" \
    || { status=$?; [[ "$status" -eq 124 ]] || exit "$status"; }

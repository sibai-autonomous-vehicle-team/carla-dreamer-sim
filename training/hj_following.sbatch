#!/bin/bash
#SBATCH --job-name=hj-following
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

run_name="hj-following-cloudy-${SLURM_JOB_ID}"
# One learned visual HJ safety policy; frozen contextual PID proposes actions.
# Gap/lane/contact failures, fixed discount .99, safety entropy .05 -> 0.
# This restores the two-sided objective, not every historical hyperparameter.
timeout --signal=INT --kill-after=120s 42300s \
    "$workspace/sage-env/bin/python" -u -m training.train_online_hj \
    --outdir "training/_local/$run_name" \
    --condition day_cloudy --performance-policy pid --pid-root specialists \
    --safety-objective following --min-gap 1 --max-gap 9 \
    --gap-range 2 8 --challenge-gap-range 1.5 3 \
    --history-frames 4 --speed-range 10 20 \
    --gamma-start 0.99 --gamma-final 0.99 \
    --alpha 0.05 --alpha-final 0 --anneal-updates 10000 \
    --random-steps 1000 --learning-starts 1000 --gate-start-updates 500 \
    --exploration-episode-fraction 0 --exploration-hold-steps 5 15 \
    --port 2810 --start-server \
    --server "$workspace/carla-0.9.15/CarlaUE4.sh" \
    --require-cuda --steps 20000 \
    --eval-interval 5000 --eval-cases 6 \
    --wandb --wandb-project carla-sage --wandb-name "$run_name" \
    || { status=$?; [[ "$status" -eq 124 ]] || exit "$status"; }

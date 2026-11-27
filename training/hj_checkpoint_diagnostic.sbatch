#!/bin/bash
#SBATCH --job-name=hj-check
#SBATCH --partition=general-gpu
#SBATCH --gpus=1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
workspace=/storage3/fs1/netanel.raviv/Active/carla_workspace
cd "$workspace/carla-dreamer-sim"
checkpoint="${1:?Pass the saved checkpoint path after the sbatch filename}"
shift
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export OMP_NUM_THREADS=8
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export WANDB_ENTITY="${WANDB_ENTITY:-a-h-hamzeh-washington-university-in-st-louis}"
exec "$workspace/sage-env/bin/python" -u -m training.diagnose_hj_checkpoint \
    --checkpoint "$checkpoint" \
    --outdir "training/_local/hj-check-${SLURM_JOB_ID}" \
    --server "$workspace/carla-0.9.15/CarlaUE4.sh" \
    --port 2995 --start-server --require-cuda \
    --eval-cases 50 --grid-points 21 --wandb "$@"

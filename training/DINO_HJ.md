Overnight DINOv2 HJ experiment
==============================

This is the single visual HJ safety actor-critic, with pretrained DINOv2 Small
(ViT-S/14) replacing ResNet-18. It uses four ordered native 224x224 RGB frames.
Each produces a 384-dimensional normalized CLS token; concatenation gives
1536 features before the heads. Measured ego speed is optional and off by
default. No privileged state or behavior cloning is used.

The actor and each of the two critics have independent backbone copies.
Transformer blocks 0..9, patch embedding, CLS token and positional embeddings
remain frozen. Blocks 10 and 11 and the final LayerNorm learn at 1e-5; heads
learn at 1e-4. Target critics update by Polyak averaging. Encoders remain in
evaluation mode to suppress stochastic layers while selected parameters retain
gradients. Final CLS features are recomputed; there is no replay feature cache.

The Slurm job keeps gamma=0.99 and entropy coefficient alpha=0.05 fixed. It
uses the existing HJ backup, replay, synchronized frames, longitudinal action
mapping, fixed route steering, held-out evaluation and W&B logger. This changes
both the encoder and schedules relative to the earlier ResNet experiment, so
any improvement would not isolate the effect of DINO alone.

On RIS, after syncing the updated repository, cache the official source and
pretrained weights once from the repository directory on the login node:

```bash
/storage3/fs1/netanel.raviv/Active/carla_workspace/sage-env/bin/python \
    -m training.dino_models
sbatch --account=compute2-di2cap training/hj_dino.sbatch
```

Source is pinned in `training/dino_models.py`. PyTorch 2.6 from the existing RIS
requirements supplies native attention; xformers is optional. The first cache
step needs internet access; the compute job can reuse the shared home cache.
DINO adds no required packages to the existing training environment.

Defaults: one H100, 32 GB host RAM, 12-hour allocation, 100k training steps,
1k random warmup, day sunny, straight road, Audi TT, no display/real-time pacing.
Evaluation uses 50 fixed cases every 5k environment steps and at the end. W&B
project is carla-sage, with run name hj-dino-last2-day_sunny-JOBID.

An interrupt is sent at 11h45m to allow the existing interrupted-checkpoint save
before allocation expiry. The target budget is 100k; actual completion within
the allocation depends on simulator/training throughput. Final and interrupted
checkpoints include the DINO weights, target critics, optimizers and counters.
Resume uses a new run directory and total step target; replay is rebuilt:

```bash
sbatch --account=compute2-di2cap training/hj_dino.sbatch \
    --resume training/_local/OLD_RUN/checkpoint_interrupted.pt --steps 100000
```

The checkpoint is specific to `training.train_dino_hj`, not interchangeable with
ResNet/state checkpoints. For a frozen-backbone comparison, add
`--dino-unfreeze-blocks 0` on a fresh run. Existing ResNet scripts are unchanged.

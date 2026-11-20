# Image-based controller learning

The actor remains the requested ResNet-18: 224x224 RGB -> ResNet-18 -> Linear(512,2)
-> tanh -> longitudinal [-3,3] and requested steering [-1,1]. ImageNet
initialization and RGB mean/std normalization match `train_distance_cnn.py`.
The backbone and BatchNorm affine weights train; running statistics stay frozen
in every training, actor-update, target, validation, and inference path.

New RGB runs default to **`--algorithm hj-td3-bc`**. The supervised baseline
(`bc`) and single-critic ablation (`ddpg`) are explicit options. Use fresh run
folders; old format-3 runs remain readable for inference/evaluation but cannot
resume into these format-4 training recipes.

## 1. Establish a supervised controller baseline

```bash
python -m training.train_ddpg --algorithm bc \
  --data training/_local/day_sunny.pt --image-root dataset/combined_800 \
  --batch-size 32 --updates 5000 --checkpoint-interval 1000 \
  --run-dir training/_local/day_sunny_bc_s42
```

This trains only the actor, without any critic or bootstrapped RL target. It
checks whether the image/action mapping can learn a sensible controller.

The default `--bc-target teacher` regenerates deterministic gap-PD and steering
labels from decision-time poses and current/past gaps. The defaults match the
nominal controller in `run_gap_pid.py`: gap target 5 m, kp=0.5, ki=0, kd=0.5,
heading steering gain 0.75. Teacher labels are clipped to the same action bounds.
`--teacher-gap`, `--teacher-kp`, `--teacher-kd`, and `--teacher-steer-gain` expose
these settings. The first aligned row has zero gap derivative, as after a PID
reset; later rows use the previous measured gap. No future pose or gap enters
the teacher action. The neural actor itself receives only the current image.

These are **controller-relabeled examples**, not executed expert demonstrations
or certified safe actions. Imitation uses only rows whose current and successor
margins are positive and whose corresponding collision flags are zero. This
selection uses future outcomes as an offline data filter, never as model input.
It deliberately avoids cloning collision actions or unlabelled exploration.
All eligible original transitions still train the RL critics.

For an explicit logged-action baseline use `--bc-target recorded`. Only interior,
collision-free rows within `--recorded-tolerance 0.15` of the teacher in both
normalized action dimensions are cloned. This is a consistency filter, not a
recovered exploration flag. Labels and eligibility counts are recorded in run
settings/metrics. Longitudinal and steering MSE divide by [3,1] respectively,
so their different command units do not arbitrarily dominate the loss.

Validate and inspect the actor before starting the RL stage:

```bash
python -m training.evaluate_ddpg --data training/_local/day_sunny.pt \
  --image-root dataset/combined_800 \
  --checkpoint training/_local/day_sunny_bc_s42/checkpoint_05000.pt \
  --split val --output training/_local/day_sunny_bc_s42/validation_05000 --no-plots
python -m training.plot_progress --run-dir training/_local/day_sunny_bc_s42
```

Watch it with `--modes actor_only`:

```bash
python -m training.evaluate_closed_loop --start-server --graphics windowed \
  --modes actor_only --checkpoints training/_local/day_sunny_bc_s42/checkpoint_05000.pt \
  --episodes 1 --steps 200 --output training/_local/day_sunny_bc_visible
```

BC has no safety critic. The harness rejects using it as a `safety_filter` and
plots label critic panels as unavailable rather than inventing safety scores.

## 2. Add HJ TD3+BC

After the supervised baseline behaves sensibly, initialize the RL actor from it:

```bash
python -m training.train_ddpg --algorithm hj-td3-bc \
  --data training/_local/day_sunny.pt --image-root dataset/combined_800 \
  --actor-init training/_local/day_sunny_bc_s42/checkpoint_05000.pt \
  --actor-warmup 0 --bc-alpha 0.1 --policy-delay 2 \
  --batch-size 32 --updates 10000 --checkpoint-interval 1000 \
  --run-dir training/_local/day_sunny_hj_td3bc_s42
```

Without `--actor-init`, the actor starts from ImageNet initialization with
`--actor-warmup 1000`: critic updates continue, while the actor optimizes only
imitation during this initial period. With BC initialization, the example
explicitly sets warmup to zero. The two critics start independently; the BC
checkpoint supplies no critic. Dataset/split, preprocessing, and action bounds
must match when importing the BC actor.

**TD3+BC** combines twin delayed deterministic policy gradients with behavior
cloning. This implementation uses:

- Two independent ResNet image critics; their minimum supplies the bootstrap
  continuation, reducing the influence of a single optimistic critic.
- Actor and target-network updates every two critic updates by default.
- Clipped target-action noise, default normalized std=0.2 and clip=0.5. Noise
  scales by each action's half-range and the result is clipped to valid bounds.
- Actor loss `-lambda * mean(Q1(image, actor(image))) + normalized_imitation_MSE`,
  with detached `lambda = bc_alpha / max(mean(abs(Q1)), 0.001)`.
- Imitation and actor-Q optimization on eligible imitation rows. If a sampled
  batch has none, actor/target updates are skipped while critics still train.
- Critic learning rate 1e-4 for new algorithms (`--critic-lr`), instead of the
  previous image critic's 1e-3. Actor lr remains 1e-4.

The conservative default alpha=0.1 emphasizes the baseline initially. It is an
experiment setting, not an established optimum. The original paper uses logged
actions for BC; teacher relabeling here is a declared extension, and recorded
cloning is available explicitly.

This is an **HJ adaptation**, not the paper's reward-return algorithm: it keeps
`h(d)=min(d-1,9-d)` and this project's discounted first-hit backup. It is not a
conformal safety guarantee. The target still does not encode lane keeping or
collision events; collision flags currently filter imitation examples only.
Single-frame images still lack explicit motion history.

References: [TD3+BC paper](https://arxiv.org/abs/2106.06860),
[authors' implementation](https://github.com/sfujim/TD3_BC).

## Evaluation and monitoring

The existing evaluator detects BC, single-critic and twin-critic checkpoints.
Twin safety scores use the minimum critic. Use validation for model selection
and keep the test split untouched until final evaluation.

```bash
python -m training.evaluate_ddpg --data training/_local/day_sunny.pt \
  --image-root dataset/combined_800 \
  --checkpoint training/_local/day_sunny_hj_td3bc_s42/checkpoint_10000.pt \
  --split val --output training/_local/day_sunny_hj_td3bc_s42/validation_10000 --no-plots
python -m training.plot_progress --run-dir training/_local/day_sunny_hj_td3bc_s42
```

Plots include training history, checkpoint action comparisons, imitation error,
and twin-critic disagreement. Metrics also monitor sampled steering sensitivity.
Frozen BatchNorm removes the train/eval statistic switch; it does not establish
that the critic ranks actions correctly. Check actual closed-loop behavior.

## Data, dependencies and checkpoints

Use the same frozen numerical cache/splits and combined image dataset. Source
and episode metadata are verified before image loading. Image i aligns to action
i, margin from distance i-1, successor image i+1 and successor margin distance i.
Images are memory-mapped with a bounded cache. Existing data is not relabelled
on disk and collection code is not modified by these algorithm changes.

If needed, rebuild the numerical cache:

```bash
python -m training.prepare_data dataset/carla_following \
  --new-source dataset/full_range_300 --conditions day_sunny \
  --output training/_local/day_sunny.pt
```

CUDA is automatic with CPU fallback; `--require-cuda` refuses fallback. The
matching torchvision dependency is in `requirements-training-gpu.txt`. Use
`--no-pretrained` only for an explicit random-initialization experiment.

Format-4 checkpoints save options, normalization policy, models, targets,
optimizers, split identity, sampler/RNG state, and partial logging accumulators.
Resume with `--resume CHECKPOINT`, the same algorithm/options/batch size, a larger
`--updates`, and omit `--actor-init`. Checkpoint interval can change. BC full
checkpoints are roughly 130 MiB; twin-critic full checkpoints are roughly 515 MiB.
One hundred twin-critic checkpoints need about 50 GiB; account for available disk.

Legacy numerical training requires `--architecture state-mlp --algorithm ddpg`.
Historical documentation/provenance remains in `LEGACY_MLP_README.md` and
`UPSTREAM_IMPORT.json`. The numerical baseline and old inference weights remain
available for comparison.

## Verification

```bash
python -m unittest discover -s tests -p 'test_visual*.py' -v
python -m unittest discover -s tests -p 'test_hj*.py' -v
```

Checks cover teacher alignment/filtering, train/eval BatchNorm equivalence,
normalized losses/noise, terminal and nonterminal twin HJ targets, delayed policy
updates, imitation warmup, missing-label batches, and legacy inference contracts.

### Temporal vision observations

New RGB CLI training uses `--history-frames 5` by default: five consecutive
10 Hz RGB images (0.4 seconds between oldest and newest). Each frame passes
through the same shared ResNet-18 backbone. Ordered 512-dimensional features
are concatenated into a 256-unit ReLU action head and two bounded outputs.
Critics also receive ordered history, with their existing action-conditioned
MLP head. There are no speed or other numerical state inputs to these models.
At episode start, missing past images repeat the first image. Replay successor
histories end at the successor frame, and history never crosses episodes.
Closed-loop evaluation reconstructs the same history from consecutive camera
frames and resets it each episode. The existing first-step PID warm-up remains.

Start a fresh temporal BC experiment (the numerical data cache can be reused):

```bash
python -m training.train_ddpg \
  --data training/_local/day_sunny.pt \
  --image-root dataset/combined_800 \
  --algorithm bc --history-frames 5 --batch-size 8 \
  --updates 5000 --checkpoint-interval 1000 --require-cuda \
  --run-dir training/_local/day_sunny_bc_temporal5_s42
```

Temporal CLI runs default to batch size 8 because backbone activation memory
scales with the number of input frames. Reduce it if CUDA memory is exhausted.
Evaluation loads history length from the checkpoint automatically. Old
single-frame checkpoints remain supported; `--history-frames 1` selects that
architecture explicitly, including when resuming an old run. Single-frame
actor initialization is rejected for temporal runs because the action head
has changed. This change does not alter the existing BC eligibility mask,
teacher labels, or first-failure transition selection.

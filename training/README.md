# Image-based controller learning

## True-gap braking collection

```bash
.venv/bin/python -u -m training.collect_braking --start-server \
  --trajectories-per-condition 500 --speed-range 12 20 --gap-range 2 5 --target-gap 3 \
  --brake-delay-range 1.5 2.5 --brake-duration 2 \
  --outdir dataset/braking_lane_tracked_500_per_condition
```

The existing CarDreamer browser viewer runs at `http://localhost:9700` with
the default CARLA port 2700. Add `--real-time` to pace collection to simulation
time, or `--headless` to disable the viewer. `--conditions day_sunny` restricts
weather. The default collects 500 accepted episodes per condition, totaling 4,500
across all nine conditions. For a short visual check, use
`--trajectories-per-condition 2 --conditions day_sunny` and a fresh output directory.

Each episode uses a random Town03 road spawn, equal sampled starting speeds,
and a true bumper-gap PID with gains 0.9, 0.05, and 1.0. The leader applies full
braking after a uniformly sampled 1.5–2.5 seconds for two seconds. Collection
ends at collision or the end of braking. PID actions have zero exploration noise.
Both cars use speed-dependent pure-pursuit steering along the saved lane route,
including during speed initialization. Gap placement follows the lane centerline.
Lane departures by either car and contacts with other objects cause a logged retry.
These checks apply before collision termination and can reduce coverage of tight
turns that the controller cannot negotiate at the sampled speed.
Actual handoff gaps and both measured speeds must lie in the requested ranges.
Their recorded values can differ from the sampled requests during setup.
The collector reuses the map and creates fresh vehicles, sensors, and PID state
for each episode. Failed attempts are logged. Invalid lane, route, or starting-state
scenarios immediately receive another recorded random spawn, preserving the sampled
speed, gap, brake delay, and condition. Scenario rejections can continue beyond 30
attempts. Other errors receive up to 30 consecutive retries, with spawn replacement
every three failures. Measured starting speeds must satisfy the requested range.
Resume with the same command and `--resume` to retain completed episodes.
Use a fresh directory for this lane-tracking recipe. Earlier collections used
heading steering and a shorter brake delay. The revised controller has unit
coverage but still needs a visual CARLA check on turns.

`plan.json` records all sampled settings and routes. Each condition directory
uses the original dataset layout with `obses/episode_000.pth`, `actions.pth`,
`states.pth`, `costs.pth`, `distances.pth`, and `seq_lengths.pth`. Episode indices
start at zero within each condition. RGB observations are uint8 tensors shaped
[T, H, W, C]. Actions, states, costs, and distances are lists of float32 tensors,
and sequence lengths are an int64 tensor. Each row uses the original timing
convention from `generate_dataset.py`, with pre-action RGB/state and post-action
distance/collision readings. The existing preparation pipeline reads these files.

Additional frame IDs, timestamps, velocities, applied controls, terminal RGB/state,
initial gap, episode plan, and summary are saved in
`condition/metadata/episode_000.pth`. Collision episodes end at the first collision
snapshot. Sequence lengths commit last, and resume recovers interrupted writes.
`summary.json` and `complete.json` describe the final dataset. Collections made
with the earlier bundled `.pt` layout require a fresh output directory for this
writer.

## Recorded-policy critic pilot

```bash
.venv/bin/python -u -m training.train_braking_critic \
  --output training/_local/day_sunny_critic_pilot_s42
```

This snapshots the first 256 committed day-sunny trajectories, groups matching
starting positions into the same split, and uses approximately 200 trajectories
for training and 56 for held-out testing. Numerical arrays are copied and immutable
RGB/metadata files are hard-linked, allowing the collector to keep appending.
`--prepare-only` writes the snapshot and audit. The same command then trains on
that snapshot. Each new training run requires its own output folder.

The ResNet-18 critic receives five past/current RGB frames and the recorded
requested action. Five fixed training epochs use the existing HJ backup with
gamma 0.99 and 1–9 m buffer bounds, applied backwards along the recorded PID
continuation. Targets absorb the first buffer hit and use the last measured
margin at recording cutoff. These labels describe the recorded policy over the
observed horizon. Action coverage follows that behavior policy.

`audit.json` records outcome counts and the frozen split. `report.json` contains
held-out error, false-safe and false-alarm rates, and collision-warning timing,
including a constant training-mean baseline. `diagnostics.png`,
`test_episodes.json`, `test_predictions.pt`, and `checkpoint.pt` provide the
plot, episode diagnostics, numerical predictions, and critic weights.

Train the remaining eight condition-specific critics sequentially with the
500-trajectory recipe using 400 training trajectories, five epochs, and seed 42.

```bash
bash training/train_remaining_braking_critics.sh
```

Each output is `training/_local/<condition>_critic_500_s42`. Completed checkpoints
are skipped when rerunning the script. An interrupted model restarts training
using its saved data snapshot and split. A training failure stops the batch.

Test full-brake interventions on the held-out routes with an isolated CARLA port.

```bash
.venv/bin/python -u -m training.evaluate_braking_critic \
  --pilot training/_local/day_sunny_critic_pilot_s42 \
  --outdir training/_local/day_sunny_critic_pilot_s42/closed_loop_full_brake \
  --port 2800 --start-server
```

Both arms use the recorded true-gap PID and route tracking. When the filtered
arm's nominal-action score is at or below zero, longitudinal control requests
full braking and retains the route-following steering. Positive scores release
the override. Each tick scores five current/past frames. Paired starting poses
and velocities must match within 0.001. Initialization failures retry the same
held-out route up to three times and appear explicitly in the summary. Runtime
lane departures are recorded as outcomes. Episode traces and summaries are JSON.
`--resume` preserves completed pairs and retries failed pairs.

View 20 filtered trajectories in a real-time chase-camera window with the
decision-time HJ score in the top-right corner.

```bash
.venv/bin/python -u -m training.evaluate_braking_critic \
  --pilot training/_local/day_sunny_critic_pilot_s42 \
  --outdir training/_local/day_sunny_critic_pilot_s42/visible_20_realtime \
  --port 2800 --start-server --n-scenarios 20 --modes filtered \
  --graphics windowed --real-time
```

The display shows PID or full-brake status, and the window title shows trajectory
progress. Each terminal frame stays visible for one second. The controller
releases the brake override when the nominal-action value becomes positive.
Esc or closing the window ends the evaluation.

## Actor-based training recipes

The actor uses ResNet-18: 224x224 RGB -> ResNet-18 -> Linear(512,2)
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

For a frozen CNN–PID nominal controller, pass its distance checkpoint to the
closed-loop evaluator. This development comparison uses the day-sunny
specialist's target and PID gains, with the HJ checkpoint's failure bounds.
Both modes receive synchronized RGB frames. Traces record the predicted gap,
camera frame, proposed PID action, HJ score, and executed action.

```bash
python -m training.evaluate_closed_loop --start-server --graphics vulkan \
  --checkpoints training/_local/day_sunny_hj_sac_margin_warmup/checkpoint_10000.pt \
  --nominal-cnn specialists/day_sunny/cnn.pth \
  --target-gap 6 --kp 0.9 --ki 0.05 --kd 0.15 \
  --modes pid safety_filter --episodes 3 --steps 150 \
  --output training/_local/day_sunny_hj_sac_cnn_pid_pilot
```

The filter uses the checkpoint's raw HJ switching threshold. Conformal
calibration requires a separately fitted and evaluated correction.

For a high-speed close-following brake test, add `--initial-gap 3
--initial-speed 15 --leader-brake-test --leader-brake-after 0.5
--leader-brake-duration 2` and use `--target-gap 3`. Both vehicles receive
velocity along their headings while wheel dynamics settle before the first
camera tick. Measured handoff speed must be within 0.2 m/s of the request.
The leader tracks 15 m/s, applies full braking from 0.5 to 2.5 seconds after
handoff, then resumes speed tracking. Saved traces include both measured
starting velocities. `--graphics windowed` shows the HJ score overlay.

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

### Conformal braking critic bounds

Generate condition-specific offsets from each checkpoint's saved held-out predictions:

```bash
.venv/bin/python -m training.conformal_critics calibrate --alpha 0.1
```

This writes `training/_local/conformal_braking.json`. At each simulation step,
provide the latest five RGB frames, oldest first, and the nine controller actions
in `CONDITIONS` order. Repeat the first frame to fill the history at episode start.
Actions are the same requested longitudinal and steering values used in the
braking dataset. The day-sunny checkpoint uses the most recent four frames;
the other eight use all five.

```python
from training.conformal_critics import ConformalCritics, CONDITIONS

predictor = ConformalCritics('training/_local',
    'training/_local/conformal_braking.json', device='cuda')  # construct once
# rgb_history: uint8 [5, 224, 224, 3]; actions: float [9, 2]
raw, lower_bounds = predictor.predict(rgb_history, actions)
# Both tensors have shape [9] on the GPU, aligned with CONDITIONS.
```

The eight matching networks are vectorized together on the GPU; the four-frame
network is evaluated separately. This is an inference interface, not a controller
selector or simulator integration. Keep tensors on the GPU until results are
needed on the CPU. GPU memory and latency depend on hardware; benchmark in the
actual simulator.

For each condition, calibration takes the maximum prediction-minus-observed-return
residual across all decision rows in each held-out starting-position group, then
uses the finite-sample one-sided conformal quantile. Under exchangeability of new
and calibration groups, and with the model fixed independently of calibration,
the nominal coverage is per condition for all observed rows of a new group; it is
not a simultaneous 90% guarantee across all nine critics. These held-out splits
were already used for model reports, so they are no longer independent test sets;
if those reports influenced model or method selection, use fresh untouched groups
for formal calibration and separately evaluate on additional data. The recorded
labels follow the collected PID action distribution and observed continuation.
Different controller actions, weather distributions, deployment policies, longer
horizons, and repeated online selection can invalidate the exchangeability
assumption. The bounds do not certify safety or counterfactual controller returns.

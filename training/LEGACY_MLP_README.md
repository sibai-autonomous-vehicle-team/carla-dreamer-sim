# Training integration

Selectively imported from https://github.com/smartonion/contextual-safety-filters
using the uploaded `contextual-safety-filters-main.zip`. `UPSTREAM_IMPORT.json`
records the archive hash and original source-file hashes. `UPSTREAM_README.md`
is the original documentation, including historical results; those results do
not describe this adaptation. Git history was not present in the ZIP.

The training package is independent of the existing image-based `CarlaDataset`.
It reads only states, actions, distances, costs, and episode lengths. Collection
code, simulator code, and existing loaders have not been replaced. No upstream
weights are imported: those weights used a 30 m upper boundary.

## Local settings

- New runs use `h(d) = min(d - 1, 9 - d)`, with failure at `h <= 0`.
- All nine time/weather conditions are supported. A cache with one condition
  trains one expert; omitting `--conditions` pools all nine into one model.
- `states[i]` precedes action `actions[i]`; `distances[i]` follows it. Thus the
  gap feature for pose i comes from distance row i-1 (`gap_offset=-1`).
- Pose-derived velocities use the configured 0.1 s simulation step.
- Episodes with nearby initial vehicle positions are grouped before a fixed
  70/15/15 train/validation/test split to reduce scenario leakage.
- Only training examples determine normalization. Checkpoints include settings,
  normalization, split identity, optimizer state, and resume information.

Run from the repository root. The existing `.venv` provides NumPy and PyTorch;
no extra dependencies are needed for preparation, training, or JSON diagnostics.
Plot generation additionally requires Matplotlib; use `--no-plots` without it.

```bash
# One condition-specific cache combining the two collection controllers
.venv/bin/python -m training.prepare_data dataset/carla_following \
  --new-source dataset/full_range_300 --conditions day_sunny \
  --output training/_local/day_sunny.pt

# Train the day-sunny expert; use --dataset new for full-range-only training
.venv/bin/python -m training.train_ddpg \
  --data training/_local/day_sunny.pt --dataset combined --updates 10000 \
  --run-dir training/_local/day_sunny_s42

# Offline diagnostics, not a closed-loop safety certification
.venv/bin/python -m training.evaluate_ddpg \
  --data training/_local/day_sunny.pt \
  --checkpoint training/_local/day_sunny_s42/checkpoint_10000.pt \
  --output training/_local/day_sunny_s42/test --no-plots
```

Repeat with the other eight condition names for nine experts. Keep separate
cache/output paths. Do not rebuild frozen splits during an experiment. Use a
new run directory for a fresh run, or `--resume PATH` to continue a checkpoint.
### GPU and CPU execution

The trainer automatically uses CUDA when available and falls back to CPU.
Models and training batches already move to the selected device; the offline
trajectory cache stays on CPU. The same data and training command work on both.

The project's `.venv` uses PyTorch 2.6.0 with CUDA 12.4 for the RTX 4070. To
reproduce this optional GPU overlay after installing collection requirements:

```bash
.local/bin/uv --cache-dir .local/uv-cache pip install \
  --python .venv/bin/python -r requirements-training-gpu.txt
```

Install the overlay last: `requirements-collection.txt` pins the CPU-only build.
The CUDA build also runs on CPU when CUDA is unavailable. Installation follows
[PyTorch's official version-specific instructions](https://pytorch.org/get-started/previous-versions/).
No separate system CUDA toolkit installation is needed for these wheels.

Add `--require-cuda` to the training command to fail instead of falling back.
To explicitly run on CPU with the same environment:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m training.train_ddpg \
  --data training/_local/day_sunny.pt --updates 10000 \
  --run-dir training/_local/day_sunny_cpu
```

The startup message and each run's `metrics.json` report the actual device.

## Interpretation and remaining research work

This is an offline numerical-state HJ-DDPG actor/critic baseline, not Dreamer
training and not conformal calibration. `train_risk.py` is an upstream legacy
one-sided baseline; `train_ddpg.py` is the two-sided training entry point.

The imported HJ backup uses discounting (gamma=0.99). Its critic is not exactly
the finite-horizon minimum safety margin: discounting can give positive targets
even before a small negative successor margin. Do not interpret Q>0 as a
calibrated guarantee or plug it into conformal calibration without specifying
the intended prediction target and policy-rollout labels.

The HJ training failure target uses the distance bounds; collision flags are
reported but are not incorporated into that target. Collision-partner labels
are absent from existing data. The stored distance is still the original
unsigned face-edge approximation, not a corrected signed following gap.
Those limitations require a deliberate target/geometry change before making
collision-safety claims. No changes to existing distance labels were made.

The held-out validation and test sets are not a separately reserved conformal
calibration set. Freeze the trained policies and collect independent matching
policy rollouts for that later stage. The existing `SafetyFilter` uses an
uncalibrated Q threshold. Closed-loop helpers require integration with a
running CARLA environment; offline diagnostics alone do not validate it.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_hj*.py' -v
```

Tests cover episode alignment, boundary truncation, grouped splits, HJ backups,
optimizer/target updates, checkpoint resume, inference reload, and mocked live
control mapping. `training/_local/` holds ignored caches and smoke-run outputs.

## Plot a saved run

```bash
.venv/bin/python -m training.plot_progress \
  --run-dir training/_local/day_sunny_s42
```

Writes `plots/training_progress.png`, plus `plots/checkpoint_comparison.png` and
`plots/checkpoint_comparison.csv` when evaluation reports are available under the
run directory. Uses validation reports by default; select `--split test` for
final test comparisons. No retraining or simulator is needed. Install optional
plotting dependencies with `uv pip install --python .venv/bin/python -r
requirements-plotting.txt`. Percentile shading describes variability across
states, not statistical confidence or closed-loop safety.

## Checkpoint frequency

Add `--checkpoint-interval 1000` to save every 1,000 total updates, in addition
to the existing milestones (including 200) and the final update. Omitting it
preserves the original saving schedule. The interval can change when resuming
and uses the total update count, not the number of updates since resume.

Continue a 40,000-update run to 100,000 total updates:

```bash
.venv/bin/python -m training.train_ddpg \
  --data training/_local/day_sunny.pt \
  --resume training/_local/day_sunny_s42/checkpoint_40000.pt \
  --updates 100000 --checkpoint-interval 1000
```

This resumes into the original run directory by default. Earlier intermediate
checkpoints cannot be recreated by resuming at 40,000; use a new run directory
without `--resume` for a fresh run with the finer saving schedule throughout.

## Matched closed-loop CARLA evaluation

The runnable harness starts a dedicated CARLA server when `--start-server` is
provided. It loads Town03 and reloads the world between rollouts for repeatable
physics. Without that flag it uses an existing **dedicated** localhost server;
that server's current world will be replaced. Do not share its port with a
collector or another ticking client.

Start with a smoke test:

```bash
.venv/bin/python -m training.evaluate_closed_loop --start-server \
  --checkpoints training/_local/day_sunny_s42/checkpoint_43000.pt \
  --episodes 1 --steps 20 --output training/_local/closed_loop_check
```

Then compare candidates on 20 matched development scenarios:

```bash
.venv/bin/python -m training.evaluate_closed_loop --start-server \
  --checkpoints \
    training/_local/day_sunny_s42/checkpoint_05000.pt \
    training/_local/day_sunny_s42/checkpoint_30000.pt \
    training/_local/day_sunny_s42/checkpoint_43000.pt \
    training/_local/day_sunny_s42/checkpoint_100000.pt \
  --condition day_sunny --episodes 20 --steps 200 \
  --output training/_local/closed_loop_day_sunny
```

Each checkpoint runs three frozen modes: the nominal gap PID, actor alone, and
PID with the learned filter. PID memory resets for every rollout. The first
command uses the PID while pose history is unavailable. Models use CPU inference
by default, leaving GPU memory for CARLA; this does not change the trained
weights. No online weight updates occur.

Outputs include saved scenario poses/routes, checkpoint hashes and evaluation
settings (`manifest.json`), frame-level per-scenario traces, `summary.csv`, and
`aggregate.json`. `complete.json` is written only after every rollout succeeds.
An existing output directory is rejected rather than overwritten. The harness
checks matched initial poses, velocities, gap and blueprints within each
checkpoint, plus physical starts across checkpoints; reset drift aborts the run.

The nominal baseline reuses `run_gap_pid.GapPID` and its heading-to-leader
steering, with a 5 m target and configurable gains. It uses the CARLA geometric
gap directly, not a CNN estimate. The leader follows a saved deterministic lane
route with a 2 m/s throttle feedback and heading P controller. This is a declared
evaluation leader, not an exact replay of the collection leader's route planner.

Gap violations include the initial frame and use the original stored-data
unsigned face-edge geometry, now sampled from synchronized transforms. Separate
initial-violation flags are recorded. Collision events retain frame and other
actor ID, so leader contacts are distinguishable from other collisions. Lane
departures count onsets of the ego center leaving all driving lanes; they do not
measure wrong-lane travel or body overlap with markings. Travel is path length,
not signed route progress. Rollouts continue for the fixed horizon after a
violation, so intervention and recovery behavior remain visible.

A per-frame IMU callback supplies a sensor-frame barrier; event-only collision
callbacks get a final 0.2 s grace period. Extremely delayed event delivery remains
a measurement limitation. No cameras are used and rendering is disabled; weather
changes do not test perception robustness. Physics follows CARLA's synchronous
fixed-delta/substep requirements. See the official [synchrony guide](https://carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep/)
and [collision sensor reference](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#collision-detector).

### Watch one rollout

Use a visible Unreal window with the spectator following the ego vehicle:

```bash
python -m training.evaluate_closed_loop --start-server \
  --graphics windowed --modes safety_filter \
  --checkpoints training/_local/day_sunny_s42/checkpoint_43000.pt \
  --episodes 1 --steps 200 --output training/_local/visible_demo
```

Windowed mode enables rendering and automatically paces actions to real time.
The window appears on the CARLA host's graphical desktop. Choose `--modes
actor_only` or `--modes pid` to watch those alone, or omit `--modes` to run all
three sequentially. No modes run in parallel. Use a fresh output folder each run.

Numerical evaluation defaults to `--graphics null` (Unreal `-nullrhi`), avoiding
Vulkan graphics-buffer allocation during repeated world reloads. `--graphics
vulkan` retains off-screen Vulkan rendering. The earlier off-screen run crashed
inside Vulkan's staging-buffer allocation after nine completed scenarios;
windowed mode uses Vulkan too and is not a proven fix for that engine crash.

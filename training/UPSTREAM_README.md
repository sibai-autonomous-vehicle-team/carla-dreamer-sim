# CARLA safety filter

This project learns when to override a PID. A critic scores the PID's action; when its score is `<= 0`, an actor supplies a replacement acceleration and steering command.

## What is trained

- **Input:** eight numerical features describing the two vehicles.
- **Output:** requested longitudinal acceleration/braking and steering.
- **Failure margin:** `min(gap - 1, 30 - gap)` in meters. The 1 m minimum is provisional; 30 m is a following-distance limit.
- **Data:** old collision-heavy and new braking-enabled trajectories across six shared conditions.

The actor and critic each have two hidden layers of 128 units. Both completed offline runs used 10,000 updates and seed 42.

## Loading and running the filter

The `weights/` directory contains the final actor, critic, and feature normalization for both 10,000-update runs:

- [Combined-data filter](weights/hj_ddpg_combined_s42_10000.pt) — the model used in the example below.
- [New-data-only filter](weights/hj_ddpg_new_only_s42_10000.pt) — the comparison model.

Below, `get_current_state_features()` and `get_pid_requested_action()` are placeholders for the CARLA integration code, not existing library functions.

```python
from training.models import SafetyFilter

policy = SafetyFilter.from_checkpoint(
    "weights/hj_ddpg_combined_s42_10000.pt"
)

# Both inputs must use the same conventions as training.
state = get_current_state_features()
nominal_action = get_pid_requested_action()

decision = policy.filter_action(state, nominal_action)
action_to_apply = decision.action
```

The public methods are:

| Method | Returns |
| --- | --- |
| `predict_safe_action(state)` | Learned actor's requested action |
| `score_action(state, action)` | Critic's uncalibrated safety estimate |
| `filter_action(state, nominal_action)` | PID action or replacement actor action, via `decision.action` |

The eight state features, in order, are:

`gap_m`, `relative_forward_m`, `relative_lateral_m`, `sin_relative_yaw`, `cos_relative_yaw`, `ego_forward_speed_mps`, `relative_forward_velocity_mps`, `relative_lateral_velocity_mps`.

Actions are `[longitudinal_request, steering_request]`, not raw CARLA actuator controls. Pass `decision.action` through the existing wrapper. Do not invert steering twice.

Use synchronized current-frame information. The archive's `gap_offset=-1` fixes stored-file alignment, not live timing. Use PID until previous-pose velocity features become available after a reset.

For closed-loop evaluation, compare on matched scenarios:

1. Braking-enabled PID alone
2. Learned actor alone
3. PID with the uncalibrated safety filter

Measure collisions, both distance boundaries, lane departures, progress, and interventions on matched scenarios. Freeze the model during evaluation. Use the team's braking-enabled PID as the main baseline, not the old throttle-only version.

## Results so far

| Held-out diagnostic | Combined data | New data only |
| --- | ---: | ---: |
| Actor requests braking | 33.60% | 6.26% |
| Median actor steering | −1.0 | −1.0 |
| PID/recorded actions flagged by critic | 985 / 53,030 | 1,017 / 53,030 |
| Flagged immediately before a recorded 1 m crossing | 444 / 444 | 439 / 444 |

## Reproduce training

Run from the repository root after setting the dataset paths:

```shell
python -m training.prepare_data "PATH_TO_OLD_DATA" --new-source "PATH_TO_NEW_DATA" --output training/_local/hj_six.pt
python -m training.train_ddpg --data training/_local/hj_six.pt --dataset combined --seed 42 --updates 10000 --run-dir training/_local/my_combined_run
python -m training.evaluate_ddpg --data training/_local/hj_six.pt --checkpoint training/_local/my_combined_run/checkpoint_10000.pt --output training/_local/my_combined_run/test
```

Shared inference weights belong in `weights/`. Keep datasets, resumable training checkpoints, plots, logs, and temporary tests under Git-ignored `training/_local/`.

Code map: `training/models.py` defines the model and filter; `training/prepare_data.py` builds the offline dataset; `training/train_ddpg.py` trains; `training/evaluate_ddpg.py` reports offline diagnostics.

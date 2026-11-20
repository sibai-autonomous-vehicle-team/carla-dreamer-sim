# Shared-route CARLA gap PID and synchronized CNN audit

The runner's `--shared-route` option selects five fixed Town03 routes for seeds 42–46. The ego and lead cars both steer toward a point 3 m ahead on the same lane centerline. Ego arrival requires crossing the scenario's along-route finish with at most 1 m lateral error. Route geometry continues beyond the ego finish for the lead car. The existing 10 m true-gap cutoff and 5 m route-deviation guard apply. Reaching the 360-step runner cap is a timeout.

Both modes used cloudy daylight, the same recorded start, 0.1 s ticks, a 6 m target, five neutral warm-up ticks and the same lead speed controller. The lead applies throttle below 2 m/s. The raw-model baseline reused the epoch-6 checkpoint from the original cloudy-day experiment. The baseline used PID gains of 0.5, 0 and 0.5 in both modes.

| Seed | Route | Ground-truth PID | Raw CNN PID |
|---|---|---|---|
| 42 | Eastbound straight | Ego finish | Distance cutoff |
| 43 | Westbound straight | Ego finish | Ego finish |
| 44 | Bend | Ego finish | Ego finish |
| 45 | Left turn | Ego finish | Distance cutoff |
| 46 | Right turn | Ego finish | Distance cutoff |

The baseline had no collisions or lane terminations in either mode. All logged paired starts matched, including first PID gap and step. The CNN estimates on the final decision of the three cutoff runs were 8.53, 8.37 and 8.03 m. Result and trace files are in this directory. In each trace, route progress and deviation are measured after the action. The primary gap, prediction, and command columns refer to decision time. The final true gap is recorded in the post-step column.

## Camera timing and model bias

The five diagnostic collection runs used varied ground-truth targets and a 14 m cutoff to observe samples above 10 m. This cutoff applies only to the audit. The audit joins images and simulator snapshots by CARLA frame ID. For 358 unique camera frames the current checkpoint had 0.994 m MAE and −0.939 m mean signed error. The 79 frames from 8–10 m averaged −0.670 m signed error. All 70 true gaps above 10 m were estimated below or at 10 m, with signed error averaging −3.690 m.

The image at decision time was 0–3 simulator ticks old. Image age alone changed the measured gap by 0.056 m MAE across 465 decision frames with a camera sample. The larger prediction bias persisted against capture-time labels. The original archived training images lack capture IDs, so the audit cannot establish their exact alignment retrospectively. The audit samples are retained for diagnosis and do not constitute independent route-held-out training and test datasets.

Commands are in `commands.sh`. The simulator was CARLA 0.9.15 with Epic offscreen rendering, Python 3.10 in `.venv`, and CUDA on an RTX 4070.

## Exploratory gain variant

The second set of paired runs used proportional, integral, and derivative gains of 0.90, 0.05, and 0.15. The ground-truth PID again reached all five ego finishes. The CNN reached three, including the eastbound straight, and exceeded the cutoff on both turns. No run collided, exceeded the route-deviation threshold, or timed out. Mean per-run tracking MAE was 0.882 m for ground truth and 1.310 m for CNN. The gain change was chosen after inspecting the same routes, and the run durations differ.

The seed 42 variant pair had 2.3 cm of vertical reset displacement and approximately 0.50 m/s of vertical velocity difference during settling. Its first PID input gaps differed by 2.2 mm. The other four pairs matched at logged precision. The separate seed 42 repeat checks whether the successful outcome recurs.

## Dataset coverage

Only 2,198 of the original 48,229 aligned training pairs had labels above 10 m. The synchronized audit shows that the checkpoint's long-gap error persists under exact frame matching. A retraining experiment should collect each target gap across multiple distinct routes and hold out complete routes, with particular attention to the 8–10 m and over-10 m bins. The original epoch-6 checkpoint was used in every run here.

In the separate seed 42 gain-variant replay, both modes reached the ego finish in 148 steps. The logged initialization records were identical.

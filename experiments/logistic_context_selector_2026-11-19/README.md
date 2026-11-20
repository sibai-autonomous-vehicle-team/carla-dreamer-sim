# Adapted logistic context selector

This CARLA Town03 pilot selects among the nine existing CNN–PID specialists. Collection and live evaluation use a 6 m PID target, gains of 0.90, 0.05 and 0.15, five neutral warmup ticks, and a **10 m true-gap cutoff**. The cutoff triggers when the measured gap strictly exceeds 10 m. The route deviation limit is 5 m. A full-route rollout has up to 360 controlled ticks after warmup.

## Selector and controller

The ordered inputs are `[sunny, cloudy, rainy, day, sunset, night, predicted_gap]`. The simulator preset supplies one active weather indicator and one active time-of-day indicator. The frozen generalist ResNet-18 supplies one shared raw gap estimate in metres at each decision. The existing inference path converts the CARLA RGB frame to a channels-first float tensor, divides by 255, and applies ImageNet channel normalization.

The selector fits nine independent, L2-regularized binary logistic heads in an affine layer with 72 learned parameters and zero hidden layers. The saved model records the training-only gap scaler and folds it into the coefficients so inference directly evaluates `sigmoid(W @ x + b)` on the seven raw inputs. `W` has shape `(9, 7)` and `b` has shape `(9,)`. The model records the feature order, specialist order and validation-selected advisory threshold.

The live selector scores all nine specialists every two seconds and gives control to the highest-scoring specialist. That specialist's raw CNN estimate drives the common PID at every tick until the next decision. The PID is initialized once per episode. On an actual specialist switch it retains its integral and clears its previous error for one tick to avoid a derivative jump caused by the estimator change. Each score denotes predicted survival over the labeled window. The live comparison separately measures destination completion.

**Fallback integration is pending.** The existing experimental safety filter was built for a separate day-sunny, 1–9 m setup. Decisions below the validation threshold are logged. The highest-scoring specialist remains active.

## Route splits and trials

The frozen plan assigns six source routes to training, three to validation, and three to test, with disjoint seed pools. It includes straights, bends and turns in each split. All scenarios and windows from a source route stay together. The 12 sampled centerlines are at least 22.59 m apart and at least 20.07 m from the earlier explicit route manifests. All 61 waypoints on each new centerline are within 3 m of pooled archived CNN training poses. The held-out selector trials may therefore show scenery familiar to the frozen CNNs. [Route plan](plan.json)

The primary schedule crosses 12 source routes with all nine weather/time presets, yielding 108 starting cases. It rotates three lead-start indices, 7, 10 and 12, and three lead behaviors, steady, pull-away and brake/restart. This creates 12 cases per index and behavior pair. The indices are waypoint offsets. Actual bumper gaps depend on the seeded vehicles and geometry and are recorded in the traces. Each case is independently reset for all nine specialists, yielding 972 planned short rollouts. Up to two consecutive windows come from each rollout while its specialist stays in control. A frozen expansion schedule supplies alternate seeds and stressors on development routes if a controller lacks both training label classes.

Each window begins with its generalist context observation. The default horizon is two seconds. The collector derives a tick count from the running world's fixed timestep, rounding up when needed. At 0.1 s the count is 20. Known collision, true gap above 10 m, or route-centerline deviation above 5 m resolves the window as a failure even when the violation ends the rollout early. A complete violation-free window is a success. A clean finish or time limit before the full horizon, missing context, and other incomplete windows are censored. Invalid and unmatched cases remain auditable and receive no training label. A simulator exception is retried once after restarting the simulator.

Ground truth is recorded at each CARLA snapshot and joined to images by capture frame. The controller uses a valid image already available at the decision tick and logs its age. Perception error uses the true gap at image capture. Tracking and labels use decision and post-step true gaps. A pair shares its route, weather, seeded actor types and positions, lead behavior, and warmup. The start check compares reset layout and measures full pose, velocity and gap at the post-warmup PID handoff. Limits at handoff are 0.03 m in either vehicle's position and in true gap, 1 degree of rotation, and 0.1 m/s of velocity. Early vertical settling can differ at raw reset, so both states are retained for inspection. [Initial smoke diagnostic](smoke_initial_reset_check.json)

## Smoke gate

The corrected handoff check passed on 27 trials from three training cases. The 945 decision rows used capture ages of zero to two ticks. The smoke audit checked consecutive decision ticks, capture-frame ordering, and the recorded gap for each capture frame. Counts below include the available windows from the two-window rollout cap. [Smoke report](smoke.json)

| Specialist | Success | Failure | Censored |
| --- | ---: | ---: | ---: |
| Day sunny | 6 | 0 | 0 |
| Day cloudy | 5 | 1 | 0 |
| Day rainy | 5 | 1 | 0 |
| Sunset sunny | 6 | 0 | 0 |
| Sunset cloudy | 5 | 1 | 0 |
| Sunset rainy | 6 | 0 | 0 |
| Night sunny | 3 | 2 | 0 |
| Night cloudy | 4 | 2 | 0 |
| Night rainy | 3 | 2 | 0 |

## Evaluation

The training split fits the coefficients and gap scaler. Validation sets the advisory threshold. The held-out classifier report measures binary scores and checks initial top-choice outcomes where all nine specialists have usable labels. Later windows reflect differences in the trajectories created by the acting specialists. The live benchmark pairs selector, generalist and weather-matched specialist on each of the 27 test cases.

A clean completion requires the destination flag without a simultaneous failure flag. Tracking MAE is the within-run mean absolute post-step true-gap error from 6 m. The live report gives matched results and all attempts, collision and cutoff counts, route terminations, tracking MAE, switches, advisory threshold crossings and fallback use. Tracking averages reflect each run's observed duration, which can vary with termination.

## Collection and fitting results

The collector saved all 972 short rollouts. The initial-state audit matched 107 of 108 cases. One training case differed at the actual control handoff and its 17 windows were excluded. The remaining examples supplied 869 training, 445 validation and 455 test labels.

| Specialist | Train success | Train failure | Validation success | Validation failure | Test success | Test failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Day sunny | 82 | 17 | 43 | 8 | 44 | 7 |
| Day cloudy | 78 | 19 | 41 | 9 | 43 | 8 |
| Day rainy | 84 | 14 | 40 | 9 | 42 | 8 |
| Sunset sunny | 88 | 12 | 46 | 5 | 47 | 5 |
| Sunset cloudy | 78 | 20 | 44 | 8 | 44 | 8 |
| Sunset rainy | 85 | 13 | 38 | 10 | 46 | 5 |
| Night sunny | 47 | 43 | 22 | 23 | 31 | 17 |
| Night cloudy | 72 | 27 | 43 | 10 | 39 | 13 |
| Night rainy | 58 | 32 | 27 | 19 | 32 | 16 |

All nine logistic regressions fitted using CPU L-BFGS and L2 regularization. On 27 matched held-out starts, the selector and the weather-matched choice each succeeded in 24 two-second windows. At least one specialist succeeded in 26. Across the 455 held-out labels, the model scored 0.410 log loss and 0.128 Brier score. The validation-selected advisory threshold was 0.922. Eleven of 27 validation choices met it and none failed within two seconds. Fifteen of 27 test choices met it and two failed. [Model report](model_report.json)

## Paired live results

All 27 test starts matched across the three arms, for 81 total rollouts. The largest control-handoff deviations across an arm pair were 0.0011 m in position, 0.0086 m/s in velocity and 0.000002 m in gap.

| Controller | Completed | Collisions | Gap-limit stops | Lane or route stops | Mean tracking MAE | Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selector | 16/27 | 3 | 8 | 0 | 1.647 m | 110 |
| Generalist | 13/27 | 0 | 14 | 0 | 1.748 m | 0 |
| Weather-matched specialist | 19/27 | 1 | 7 | 0 | 1.476 m | 0 |

The selector finished five cases the generalist missed and missed two cases the generalist finished. It finished one case the weather-matched specialist missed and missed four cases the weather-matched specialist finished. It scored below the advisory threshold on 142 of 208 decisions. No fallback ran. No arm had a time-limit or missing-input stop. [Paired results](live_report.json) · [Per-run CSV](live_results.csv)

The sunset-sunny turn at seed 30043 finished after 281 ticks and used three specialists across ten switches. Its timeline shows both CNN estimates and the selected probability at each decision. [Timeline](selector_run.png) · [Trajectory](live/test_03_turn_sunset_sunny/selector.json)

Live verification recomputed every argmax decision and all 11,416 PID updates from the saved model and predictions. The estimates can vary between simulator resets with matched physical starts. The generalist's first-frame estimates differed by up to 1.32 m across the nine independently rendered trials of a matched test start. The initial-window offline comparison uses a single weather-matched trial's context to score all nine outputs for that case.

## Reproduction and artifacts

Run from the repository root with CARLA port 2000 available. The script generates the plan, verifies the ten checkpoint fingerprints, runs the smoke gate, then collects, fits and evaluates.

```bash
bash experiments/logistic_context_selector_2026-11-19/commands.sh
```

Rerun that command to resume. An experiment lock prevents concurrent writers. The plan and settings file pin the cases, model hashes, and rollout parameters. A complete trajectory is published atomically and revalidated on reuse. Interrupted work resumes at the first missing trial or live rollout. `selector.experiment --stage smoke`, `--stage fit` and `--stage evaluate` can also rerun individual stages with the same arguments from the command file.

- [Plan and split manifest](plan.json) and [settings with checkpoint hashes](settings.json)
- [Pipeline log](pipeline.log), [smoke counts](smoke.json) and [smoke window labels](smoke_examples.jsonl)
- Collection traces under `trials/<case_id>/<specialist>.json` and paired runs under `live/<case_id>/<mode>.json`
- `collection.json` for matched-start checks and split-level counts, and `examples.jsonl` for each context, source route, seed, specialist, label, disposition, frame reference and observed horizon
- `model.json` for fitted affine weights, order, preprocessing and threshold, and `model_report.json` for label counts, probability metrics and held-out choice outcomes
- `live_report.json` for aggregate and per-condition metrics, `live_results.csv` for run-level summaries and `selector_run.png` for one trajectory

Each per-step trace includes decision and capture frame IDs, image age, capture-time and decision-time ground truth, post-step true gap, generalist and active controller predictions, PID command, throttle, brake, steering, progress and terminal flags. Selector traces also include all nine probabilities at decision ticks and the selected controller, its last decision score, switch and fallback flags at each tick.

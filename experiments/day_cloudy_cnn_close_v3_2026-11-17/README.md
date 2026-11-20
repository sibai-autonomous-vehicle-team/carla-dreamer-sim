# Close-gap CNN fine-tuning and CARLA validation

This experiment extends the cloudy-day ResNet-18 with moving close-gap examples. The starting checkpoint is v2, which had completed the prior 6 m benchmark and showed weak distance accuracy below 3 m.

## Collection and model selection

The close collection used the same route partition as v2, with eight training routes, three validation routes, and five test routes. Each route had runs targeting 0.65, 1.5, and 2.5 m. The lead moved at approximately 1 m/s and the ego used measured-gap feedback. Both vehicles were moving in all retained captures. Image and distance labels share the exact simulator frame. The new collection spans 0.30–3.25 m and contains 1,543 frames from 48 episodes, with zero collision endings during collection.

| New capture split | 0.30–<1 m | 1–<2 m | 2–3 m | Above 3 m | Total |
|---|---:|---:|---:|---:|---:|
| Training | 153 | 223 | 313 | 72 | 761 |
| Validation | 34 | 93 | 121 | 23 | 271 |
| Test | 87 | 162 | 230 | 32 | 511 |

The trainer mixed the close and standard synchronized collections while keeping all episodes on each route in the same partition. The combined split contains 2,451 training, 910 validation, and 1,583 test frames. Fine-tuning updates the final residual block and regression head and keeps the v2 BatchNorm statistics. The legacy dataset is evaluated for reference only, with zero replay frames.

Eligible checkpoints need at least a 10% reduction in validation MAE below 3 m relative to v2, absolute close MAE at most 0.60 m, and zero validation predictions of at least 6 m for a true gap below 3 m. In the standard-source validation set, the 3–8 m and 8–12 m MAEs may each increase by at most 0.10 m. Checkpoint selection uses only validation data.

## Live protocol

Live rollouts use cloudy Town03, the held-out route manifest, seeds 42–46, the stock lead controller, the same lane-center steering, and gains of 0.90, 0.05, and 0.15. Both CNNs supply raw distance predictions. The 6 m target uses the usual spawn. The 2.5 m target uses a closer lead waypoint on the same centerline. Both profiles keep a 10 m true-gap cutoff, 5 m route-deviation limit, and the explicit ego destination. Collision detection is active in both profiles. The 6 m development rerun uses the five routes from the earlier shared-route experiment.

## Offline results

Epoch 20 was selected by its validation score. Close validation MAE fell from 1.864 to 0.165 m over 248 images. The protected validation bands both passed.

| Test metric | CNN v2 | CNN v3 |
|---|---:|---:|
| Combined synchronized MAE (1,583 frames) | 0.956 m | 0.428 m |
| Synchronized MAE below 3 m (479 frames) | 1.925 m | 0.227 m |
| Signed error below 3 m | −1.785 m | +0.139 m |
| Negative estimates below 3 m | 218 | 0 |
| Synchronized 8–12 m MAE | 0.649 m | 0.680 m |
| Above-10 m detections | 203/283 | 195/283 |
| Legacy MAE below 3 m (781 frames) | 2.090 m | 0.359 m |
| Legacy overall MAE (6,070 frames) | 0.616 m | 0.376 m |

The five test routes are held out from the synchronized fine-tuning datasets. Earlier ResNet weights had seen legacy trajectories near these roads. The normal ground-truth output was moved into its final profile directory after an output-path correction. Its run settings and measurements are those in the command file.

## Live results

All three modes completed 5/5 held-out routes at both targets, with zero collisions, true-gap cutoffs, or route-deviation terminations.

| Target | Metric | Ground truth | CNN v2 | CNN v3 |
|---|---|---:|---:|---:|
| 6 m | Mean per-run tracking MAE | 0.866 m | 1.038 m | 1.079 m |
| 6 m | Minimum true gap | 3.901 m | 3.887 m | 2.665 m |
| 2.5 m | Mean per-run tracking MAE | 0.902 m | 1.142 m | 1.012 m |
| 2.5 m | Minimum true gap | 0.530 m | 0.032 m | 0.406 m |

At 2.5 m, the prediction MAE at decisions with true gap below 3 m was 1.195 m for v2 (352 decisions) and 0.264 m for v3 (435 decisions). These live decision sets differ because the controllers change the trajectory. The synchronized offline test compares identical camera frames. V2 emitted 83 negative estimates in the tighter rolls and v3 emitted zero. Tight v3 still approached to 0.406 m, so completion should be considered alongside minimum gap.

The 6 m v3 rerun reached 5/5 previous development destinations. The interrupted tight v2 attempt is archived separately and the complete five-run replay is used for the table. The CARLA restart can change camera delivery timing and reset settling. Per-seed pose and initial-gap deltas are recorded in the live CSV.

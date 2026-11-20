# Cloudy-day ResNet-18 follow-up

This experiment targets the underestimation observed near the 10 m cutoff in the original cloudy-day checkpoint. It holds the controller gains fixed across the live comparisons and uses the same shared-route finish rule as the previous experiment.

## Data and model

The route manifest contains eight training routes, three validation routes, and five test routes. New split centerlines traverse disjoint road IDs and have at least 20 m sampled horizontal separation between partitions. The legacy v1 checkpoint had training trajectories near these roads. The five test routes are held out from the new supervised updates, and their synchronized images are newly collected.

Each route is attempted with a ground-truth controller at target gaps of 4.5, 7.5, and 10.5 m. The 15 m collection cutoff allows frames above 10 m. Images and gap labels are joined by CARLA frame ID, and the exact recorded ego and lead transforms are used for the gap. The collector retains roughly every second matched frame. Split membership is determined by route before any fine-tuning.

The v2 candidate starts from the existing ResNet-18 and fine-tunes its final residual block and regression head. It uses synchronized training data and zero replay frames from the legacy archive. Training samples are balanced across gap bands. The only checkpoint selection data are the new validation routes. The validation selection score is a weighted mean of MAE below 8 m, from 8–10 m, and above 10 m, with weights 1, 2, and 2.

## Live comparison

Ground-truth PID, CNN v1, and CNN v2 use Town03 cloudy daylight, the same scenario manifest and seed sequence, a 6 m target, gains of 0.90, 0.05, and 0.15, and a neutral five-tick warm-up. Both CNNs use their raw predictions, and the lead controller and steering are shared. The 10 m true-gap cutoff, 5 m route guard, and 360-step cap apply equally. A successful run requires the ego vehicle to reach its explicit route finish without any other terminal flag.

Use `commands.sh` for exact collection, training, evaluation, and plotting commands. The previous CARLA process stalled during the first map load. That process was stopped, and the fresh server was used for collection and the measured live runs.

## Offline results

The new collection contains 48 episodes and 3,401 matched images. Training used 1,690 images, validation used 639, and test used 1,072. Checkpoint selection chose epoch 6 of 16. The weighted validation score was 1.462 m for v1 and 0.409 m for the selected v2.

| Held-out synchronized metric | CNN v1 | CNN v2 |
|---|---:|---:|
| All-frame MAE | 1.525 m | 0.537 m |
| Mean signed error | −1.508 m | −0.011 m |
| MAE, true gaps from 8 to 12 m | 2.397 m | 0.649 m |
| MAE, true gaps below 6 m | 0.460 m | 0.401 m |
| Correctly predicted above 10 m | 0/283 | 203/283 |
| Predicted above 10 m with true gap at most 10 m | 0/789 | 12/789 |

The synchronized test contains no gaps below 3 m. On the original archived test, overall MAE improved from 0.702 to 0.616 m, while 95th-percentile absolute error rose from 1.795 to 2.997 m. This is a useful trade-off to track when broadening beyond the shared-route conditions. See the summary JSON for each gap band and route group.

## Live results

| Route | Ground truth | CNN v1 | CNN v2 |
|---|---|---|---|
| Left turn | Finished | Gap cutoff | Finished |
| Right turn | Finished | Gap cutoff | Finished |
| First bend | Finished | Gap cutoff | Finished |
| Straight | Finished | Finished | Finished |
| Second bend | Finished | Finished | Finished |

Across the 15 held-out live rollouts there were zero collisions, route-deviation terminations, or timeouts. CNN v2's largest true gap was 9.31 m and its smallest was 3.85 m on these routes. Both CNNs used matching vehicle types and poses across their paired rollouts. The ground-truth seed 42 reset showed 2.3 cm settling displacement compared with both CNN modes and less than 0.1 mm difference in initial measured gap. The remaining starts matched at logged precision.

The v2 regression run also completed all five previous development routes using the same gains. These routes were available during fine-tuning. The right-turn development run approached to 1.87 m. In the original test archive, v2's MAE for true gaps below 3 m rose from 0.396 to 2.090 m. The new synchronized test has no examples in this bin, so completion results here support the tested 6 m following scenarios. To improve coverage of close following, collect synchronized near-gap approaches and require acceptable per-bin close-gap validation before widening use.

# Generalist CNN in live CARLA weather conditions

This experiment passes the saved pooled `generalist/cnn.pth` to the existing gap PID in nine conditions. The condition profile provides the preset, routes, starting seed, target, gains, and terminal thresholds. The input checkpoint was selected by validation during the archived generalist training. The PID receives raw CNN distances for throttle and braking, and the shared route follower steers. CARLA measures the true gap for logging and the 10 m cutoff.

The benchmark uses the same five Town03 routes and seeds 42–46 in each condition as the specialist evaluations. The target is 6 m, the PID gains are 0.90, 0.05, and 0.15, and the route-deviation limit is 5 m. The script saves per-step traces and per-run JSON. Clean completion means the ego car reached the route destination without a failure flag. Tracking MAE measures the actual simulator gap's deviation from the 6 m target across the steps taken.

The comparison pairs generalist and historical specialist runs by route and seed. It checks settings and recorded start states. Separate CARLA sessions can differ in camera delivery and reset settling. The reported live prediction and tracking errors describe each controller's realized trajectories.

To reproduce from the repository root with port 2000 free:

```bash
bash experiments/generalist_live_2026-11-18/commands.sh
```

This launches a dedicated offscreen CARLA instance, runs conditions in sequence, and generates the comparison report. A repeat in the same directory skips completed conditions and preserves interrupted output directories.

## Results

The generalist completed 27/45 routes. The historical specialists completed 37/45 under the matched conditions and PID settings.

| Condition | Generalist finished | Specialist finished | Generalist mean per-run gap tracking MAE |
|---|---:|---:|---:|
| Day sunny | 3/5 | 4/5 | 1.485 m |
| Day cloudy | 2/5 | 5/5 | 1.620 m |
| Day rainy | 2/5 | 4/5 | 1.751 m |
| Sunset sunny | 4/5 | 5/5 | 1.562 m |
| Sunset cloudy | 4/5 | 5/5 | 1.426 m |
| Sunset rainy | 3/5 | 4/5 | 1.548 m |
| Night sunny | 3/5 | 3/5 | 1.584 m |
| Night cloudy | 4/5 | 4/5 | 1.510 m |
| Night rainy | 2/5 | 3/5 | 1.534 m |

All 18 generalist misses were true-gap cutoffs. The generalist recorded zero collisions, route deviations, timeouts, or empty-camera terminations. At the final control decision before each cutoff, the generalist underestimated the true gap in all 18 runs. The average true gap on those decisions was 9.936 m and the average estimate was 6.466 m. On night-rainy seed 44 the true gap was 9.866 m, the estimate was 4.919 m, and the PID commanded braking while the diagnostic ground-truth PID commanded full throttle on that same trajectory.

The generalist averaged 1.558 m live tracking MAE across its runs and the specialists averaged 1.224 m. Each run contributes the mean error over its observed duration, which varies with terminal outcome. The test used the archived-data generalist checkpoint and the deployed specialist checkpoints. Most deployed specialists had additional synchronized fine-tuning.

The complete comparison and paired start checks are in `summary.json`. The per-condition CSV and comparison plot summarize the headline metrics.

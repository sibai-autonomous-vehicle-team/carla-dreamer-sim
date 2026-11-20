# CARLA distance specialists

Each of the nine lighting and weather combinations has its own directory. A ready specialist has `cnn.pth` for its ResNet-18 weights and `pid.json` for its CARLA condition, PID gains, target, and rollout settings. A pending directory contains an initial PID profile. The CNN architecture, preprocessing, training code, PID implementation, and route manifest are shared.

| Lighting | Sunny | Cloudy | Rainy |
|---|---|---|---|
| Day | `day_sunny` | `day_cloudy` | `day_rainy` |
| Sunset | `sunset_sunny` | `sunset_cloudy` | `sunset_rainy` |
| Night | `night_sunny` | `night_cloudy` | `night_rainy` |

Run a saved specialist from the repository root with CARLA already listening on port 2000.

```bash
.venv/bin/python -B -u run_gap_pid.py --specialist specialists/day_cloudy --output experiments/day_cloudy_replay
.venv/bin/python -B -u run_gap_pid.py --specialist specialists/sunset_cloudy --output experiments/sunset_cloudy_replay
```

For a transfer test, use the target condition's PID profile and explicitly supply a checkpoint from a different condition.

```bash
.venv/bin/python -B -u run_gap_pid.py --pid-profile specialists/sunset_cloudy/pid.json \
  --checkpoint specialists/day_cloudy/cnn.pth --output experiments/day_to_sunset_replay
```

See `workflow.sh` for shared stages for archive training, synchronized collection, joint fine-tuning, publishing, held-out evaluation, and live replay. Use a new experiment directory per training run. The default PID gains are the validated daytime settings. The sunset experiment uses these same gains to compare the two CNNs on equal terms. You can tune each specialist's PID independently by editing its own profile.

Published weights are local files. The profile records their SHA-256 and source experiment. The checkpoint files are excluded from Git so the nine profiles and shared code can stay small.

## Completed batch

All nine condition folders have a checkpoint and a PID profile. The comparison of held-out archive predictions and live routes is in `experiments/all_specialists_2026-11-17/README.md` and `comparison.csv`. A profile marked `validated` finished all five live benchmark routes. A profile marked `tested` has its live outcomes recorded in the report named by `live_report`.

Choose the matching folder with the shared runner.

```bash
.venv/bin/python -B -u run_gap_pid.py --specialist specialists/night_cloudy \
  --output experiments/night_cloudy_replay
```

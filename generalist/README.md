# Pooled generalist

`cnn.pth` is the ResNet-18 trained jointly over all nine CARLA archives. Its metadata and checksum are in `cnn.json`.

Pass the checkpoint to the shared PID runner together with the profile for the condition to simulate.

```bash
.venv/bin/python -B -u run_gap_pid.py \
  --pid-profile specialists/day_cloudy/pid.json --checkpoint generalist/cnn.pth \
  --output experiments/generalist_day_cloudy_replay
```

The training and offline evaluation report is at `experiments/generalist_all_conditions_2026-11-18/README.md`.

# Generalist distance CNN

One ResNet-18 predicts the lead-car gap in all nine lighting and weather combinations. The generalist uses the same 224 × 224 RGB preprocessing, single regression head, ImageNet initialization, Huber loss, AdamW optimizer, and 10-epoch archive-training protocol as the specialist pipeline. Training uses bfloat16 automatic mixed precision and validation uses float32.

The nine Linux combined archives each have 800 trajectories. Together they contain 557,080 logged rows and 549,880 aligned camera-gap pairs. Earlier audits matched all 7,200 image files to their Linux sources. The archive loader pairs camera frame 1 with gap 0 because frames are logged before the action and gaps after it. The final logged image is retained and the final post-action gap is excluded.

| Split | Trajectories | Aligned frames |
|---|---:|---:|
| Training | 5,760 | 433,275 |
| Validation | 720 | 59,917 |
| Test | 720 | 56,688 |

The split is condition-stratified and episode-level. It reproduces the specialist's seed-42 640/80/80 split within each condition. The trainer shuffles the pooled condition and episode pairs each epoch. Holdout episode IDs and archived targets were checked against all nine historical specialist evaluations. An audit found no exact duplicate complete numeric trajectories between conditions. Some initial poses recur, so the split tests held-out trajectories in the same map with potentially familiar scenery.

One shared checkpoint is selected by frame-weighted pooled validation MAE. The frozen checkpoint is then evaluated separately on each condition. Comparisons include the archive-only specialists and the deployed specialists. Several deployed specialists received additional synchronized fine-tuning, so their results measure the existing deployed alternatives. This experiment evaluates archived predictions. Live PID route completion is a separate benchmark.

## Reproduce

Run from the repository root. Use a fresh output directory for a new training experiment.

```bash
bash experiments/generalist_all_conditions_2026-11-18/commands.sh all
```

The individual stages are `train`, `evaluate`, and `publish`. They save the pooled split and per-condition split IDs, epoch metrics, generalist and specialist comparisons, training curves, test plots, and the verified checkpoint under `generalist/`.

Use a condition's PID profile to select the simulator weather, route parameters, and gains when running the published generalist.

```bash
.venv/bin/python -B -u run_gap_pid.py \
  --pid-profile specialists/day_cloudy/pid.json --checkpoint generalist/cnn.pth \
  --output experiments/generalist_day_cloudy_replay
```

## Results

All ten epochs completed in 42.9 minutes. Epoch 9 was selected at 0.720 m pooled validation MAE. On the unchanged 720 test trajectories, the pooled generalist scored 0.691 m MAE, 1.244 m RMSE, −0.423 m signed error, and 2.493 m 95th-percentile absolute error. The nine archive-only specialists scored 0.683 m pooled MAE and the nine deployed specialists scored 0.508 m.

| Condition | Generalist MAE | Archive-only specialist MAE | Deployed specialist MAE |
|---|---:|---:|---:|
| Day sunny | 0.814 m | 0.619 m | 0.408 m |
| Day cloudy | 0.633 m | 0.702 m | 0.376 m |
| Day rainy | 0.544 m | 0.702 m | 0.441 m |
| Sunset sunny | 0.513 m | 0.581 m | 0.362 m |
| Sunset cloudy | 0.476 m | 0.576 m | 0.423 m |
| Sunset rainy | 0.661 m | 0.732 m | 0.631 m |
| Night sunny | 0.748 m | 0.800 m | 0.800 m |
| Night cloudy | 0.711 m | 0.679 m | 0.338 m |
| Night rainy | 1.057 m | 0.753 m | 0.753 m |

The generalist improved on six of the nine archive-only specialists. Its largest regression was in night rain. Error statistics and plots for each condition are under `evaluation/`. The exported weights in `generalist/cnn.pth` match the validation-selected checkpoint by SHA-256.

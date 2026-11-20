# Sunset-cloudy ResNet-18 and gap PID

## Data and packaging

The experiment used `dataset/combined_800/sunset_cloudy` on Linux. It contains 500 consolidated trajectories and 300 full-range trajectories, with 61,736 recorded rows and 60,936 aligned camera-gap pairs. All merged image tensors and numeric data were verified against the documented Linux sources. Direct equality to the Mac path remains unverified. See `dataset_verification.json` and `dataset.sha256` for the input audit.

The archive training used ImageNet-pretrained ResNet-18, the existing 224 × 224 RGB preprocessing, a single regression output, Huber loss, and AdamW. The 640/80/80 trajectory split uses seed 42 and is saved under `archive/split.json`. The epoch checkpoint is selected by validation MAE. The evaluator uses this same test split for the daytime checkpoint and both sunset models. Stored images precede the action, and distances follow it, so the evaluator uses the existing one-row alignment and drops the final post-action gap.

The second training stage uses new synchronized standard-gap and close-gap images in cloudy sunset, with the same close-aware validation method used in the daytime experiment. It starts from the selected sunset archive model and learns from the sunset captures. New images and gap labels match the simulator frame ID. The new training, validation, and test routes are assigned before fitting. Those route groups are separate in the synchronized data, and the archive initialization may have seen nearby Town03 geometry. The synchronized checkpoint is eligible for publication only if it meets the close and standard validation limits. The archive checkpoint supplies a fallback if no synchronized epoch qualifies.

Each saved specialist directory contains the checkpoint and its PID profile. These runs start with the daytime-tuned gains of 0.90, 0.05, and 0.15, stored separately in the two profiles. The shared code loads either condition and can run a different checkpoint with the target condition's profile for transfer testing.

## Reproduction

Run from the repository root. The server command runs in its own shell and the CARLA client stages run sequentially.

```bash
E=experiments/sunset_cloudy_specialist_2026-11-17
bash specialists/workflow.sh sunset_cloudy server "$E"
```

```bash
E=experiments/sunset_cloudy_specialist_2026-11-17
for phase in train_archive collect train_sync publish evaluate plots live; do
  bash specialists/workflow.sh sunset_cloudy "$phase" "$E"
done
.venv/bin/python -B "$E/summarize_live.py"
```

Archive training and synchronized collection ran concurrently on the RTX 4070. The collection commands used the same route manifest content as `specialists/routes.json`, from the daytime experiment's copy. Pipeline inputs and resolved train and test episode paths are recorded in the experiment.

All live modes use the sunset-cloudy preset, five explicit ego destinations, seeds 42–46, a 6 m target, PID gains 0.90/0.05/0.15, five neutral warm-up steps, and the same lead behavior and lane-center steering. The termination thresholds are a true gap above 10 m and route deviation above 5 m. Mode differences are the gap source and its associated checkpoint. CNN estimates remain raw.

## Results

The archive checkpoint was selected from epoch 5 of 10 with 0.556 m validation MAE. The synchronized collection contributed 4,979 frames. From the 20-epoch fine-tuning run, epoch 4 passed the close and standard validation limits and had the lowest eligible score. On 1,591 synchronized test frames, fine-tuning improved the sunset archive model from 0.821 m to 0.458 m MAE. Its MAE below 3 m fell from 0.441 m to 0.196 m and its MAE above 10 m fell from 1.693 m to 0.564 m.

The following archived results all use the same 80 reserved sunset trajectories and 6,312 aligned frames.

| Metric | Daytime CNN v3 | Sunset archive CNN | Sunset specialist |
|---|---:|---:|---:|
| Overall MAE | 0.414 m | 0.576 m | 0.423 m |
| True gap below 3 m MAE | 0.411 m | 0.365 m | 0.391 m |
| True gap above 10 m MAE | 1.482 m | 1.815 m | 0.820 m |

Ground truth, the daytime CNN, and the sunset specialist each reached 5/5 ego destinations in the live sunset condition. Each had zero collisions, distance cutoffs, or route violations. The mean per-run tracking MAE was 0.883 m for ground truth, 1.123 m for the daytime transfer, and 1.070 m for the sunset specialist. The true-gap range was 3.79–9.22 m for the daytime transfer and 2.84–9.88 m for the sunset specialist. The paired start deltas and each run's terminal status are in `live/summary.json`.

The packaged sunset checkpoint is in `specialists/sunset_cloudy/cnn.pth`. Its associated `pid.json` records the matched controller settings and the checkpoint hash. The existing daytime checkpoint is independently packaged in `specialists/day_cloudy`.

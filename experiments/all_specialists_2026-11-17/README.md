# Nine CARLA condition specialists

This batch trains and evaluates the seven conditions remaining after `day_cloudy` and `sunset_cloudy`. Each uses the condition-matched, 800-trajectory archive under `dataset/combined_800`. The per-condition verification reports record frame counts, alignment checks, and the comparison to the documented Linux source files. The Mac folders were unavailable for direct comparison.

The shared archive trainer uses a seed-42 split of 640 training, 80 validation, and 80 test trajectories. Its ResNet-18, regression head, RGB normalization, Huber loss, and AdamW optimizer are shared between conditions. The CUDA batch uses mixed precision for training on the RTX 4070 and computes validation MAE in FP32. Each archive model runs for 10 epochs and its best validation checkpoint initializes the synchronized fine-tune.

The simulator collects both standard- and close-gap images under each condition's lighting and weather, with distances matched by capture frame. A 20-epoch fine-tune uses the same grouped route split and close-aware checkpoint gates as the sunset-cloudy experiment. The published result comes from validation selection. The archive checkpoint is the fallback if the synchronized train produces no eligible epoch.

Archived evaluation compares the condition specialist, its archive checkpoint, and the daytime-cloudy CNN on identical reserved trajectories for this condition. Live tests compare the condition specialist, daytime-cloudy transfer, and simulator ground truth on the same five explicit ego destinations. Both CNNs supply raw predictions to the PID. The 6 m target, PID gains of 0.90/0.05/0.15, 10 m true-gap cutoff, and 5 m route-deviation limit are held fixed within each comparison.

## Reproduction

Run from the repository root with CARLA port 2000 available.

```bash
E=experiments/all_specialists_2026-11-17
.venv/bin/python -B -u specialists/run_batch.py --output "$E"
.venv/bin/python -B specialists/finish_batch.py --output "$E"
```

The batch starts a dedicated offscreen CARLA process. GPU training and simulator work have separate queues and share one ticking client. Completed stages are recorded under each condition's `.completed` directory. If the batch is interrupted, a rerun with the same output path resumes synchronized collection and skips completed stages. Partial model-training stages restart after their output directories are preserved.

Each package in `specialists/<condition>` contains its own `cnn.pth` and `pid.json`. All conditions use the same implementation and route manifest. Comparison metrics and full experiment logs are kept here so the deployed folders stay small.

## Batch results

All stages completed for the seven new conditions. The comparison includes the prior day-cloudy and sunset-cloudy packages. The table uses condition-matched archived test frames for prediction MAE and the 6 m target for live route completion.

| Condition | Test frames | Prediction MAE | Daytime-cloudy transfer MAE | Live completion | Native terminal failures |
|---|---:|---:|---:|---:|---|
| Day sunny | 6,449 | 0.408 m | 0.424 m | 4/5 | One gap cutoff |
| Day cloudy | 6,070 | 0.376 m | 0.376 m | 5/5 | Zero |
| Day rainy | 5,917 | 0.441 m | 0.533 m | 4/5 | One gap cutoff |
| Sunset sunny | 6,106 | 0.362 m | 0.651 m | 5/5 | Zero |
| Sunset cloudy | 6,312 | 0.423 m | 0.414 m | 5/5 | Zero |
| Sunset rainy | 6,654 | 0.631 m | 0.925 m | 4/5 | One gap cutoff |
| Night sunny | 5,852 | 0.800 m | 1.721 m | 3/5 | Two gap cutoffs |
| Night cloudy | 6,157 | 0.338 m | 1.395 m | 4/5 | One collision |
| Night rainy | 7,171 | 0.753 m | 2.367 m | 3/5 | Two gap cutoffs |

Ground-truth PID finished all 45 corresponding routes. The native CNNs finished 37/45. Six native failures occurred on the test bend on road 65. Five ended at the 10 m true-gap cutoff. The night-cloudy run on that bend ended in a collision. There were no native route-deviation terminations or timeouts.

The night-sunny and night-rainy synchronized runs completed all 20 fine-tuning epochs and produced comparison reports. Their candidates fell outside the validation eligibility limits, so those folders use their best archive-trained checkpoints. The remaining seven folders use synchronized fine-tuned weights.

CARLA timed out loading the map during the first sunset-rainy transfer attempt. The runner started a fresh simulator process and repeated that condition's live comparison. The table uses the complete replacement comparison. The per-condition summaries contain terminal reasons and start-state and reset deltas.

All nine published checkpoint hashes and PID profile identities were verified. Profiles marked `validated` completed all five routes with zero collisions, cutoffs, route violations, or timeouts. Profiles marked `tested` have the reported benchmark failures. The individual report paths appear in `pid.json`; `comparison.csv` contains the full aggregate.

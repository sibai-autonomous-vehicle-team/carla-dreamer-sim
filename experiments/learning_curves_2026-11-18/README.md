# Distance CNN learning curves

The blue lines show **validation mean absolute error (MAE)** in metres at each epoch. Orange diamonds show **measured held-out test MAE**. Red crosses show measured test MAE for diagnostic checkpoints that did not replace the selected model.

- `archive_learning_curves.png` covers the ten archive epochs for every specialist and the pooled generalist. The specialist archive test points use each condition's own held-out trajectories. The generalist point pools those same test trajectories across all nine conditions.
- `synchronized_learning_curves.png` covers the specialists' synchronized fine-tuning runs on their own validation and test splits. Epoch 0 is the starting checkpoint. Day cloudy V2 and V3 are shown separately because their synchronized splits differ. Night sunny and night rainy retained their archive models after their fine-tuning runs. The sunset sunny epoch-2 cross uses an extra retained diagnostic checkpoint scored on the exact original synchronized test split. Its result and provenance are in `sunset_sunny_diagnostic.json`.

Validation was recorded every epoch. The original runs retained selected checkpoints and a few diagnostic checkpoints, so there are gaps between the observed test scores. Archive and synchronized panels use different datasets. Each figure uses a common scale within that figure.

## Reproduce the plots

From the repository root:

```bash
.venv/bin/python -B experiments/learning_curves_2026-11-18/plot_learning_curves.py
```

The script reads the saved training and test artifacts and the supplemental diagnostic result.

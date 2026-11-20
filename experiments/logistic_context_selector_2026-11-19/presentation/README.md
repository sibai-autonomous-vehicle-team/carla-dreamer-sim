# CARLA paired controller comparison

These figures use 27 matched full-route cases, with three Town03 source routes in each of nine lighting and weather conditions. Each method has one rollout per case, for 81 total rollouts. Cases share the route, actor types, seed, lead script, five neutral warmup ticks, PID gains and terminal rules. All starting states passed the control-handoff match. The largest deviations between arms were 0.0011 m in position and 0.000002 m in true gap. The PID target is **6 m**. The true-gap cutoff triggers above **10 m** and the route-deviation limit triggers above 5 m. [Case manifest](../plan.json) · [Experiment settings](../settings.json) · [Paired start checks](../live_report.json)

## Figures

| Figure | Contents | Files and figure data |
| --- | --- | --- |
| 1. Overall outcomes | Percentage of runs with clean route completion and percentage with one or more specification violations. Labels show the numerator and denominator. | [SVG](figure_1_overall.svg) · [PNG](figure_1_overall.png) · [CSV](figure_1_overall.csv) |
| 2. Terminal causes | Collision, true gap above 10 m, lane/route departure and timeout or other task stop. A separate panel reports simulator and camera errors. | [SVG](figure_2_terminal_causes.svg) · [PNG](figure_2_terminal_causes.png) · [CSV](figure_2_terminal_causes.csv) |
| 3. Condition heatmap | Full-route completion percentages and counts, with three cases per condition and one 0–100% scale. | [SVG](figure_3_by_condition.svg) · [PNG](figure_3_by_condition.png) · [CSV](figure_3_by_condition.csv) |
| 4. Paired trajectory | True gap, selected specialist and predicted success probability for sunset sunny turn seed 30043. The generalist exceeded 10 m at 12.9 s. Direct matching and logistic selection finished at 27.6 and 28.1 s. The selector made ten switches. | [SVG](figure_4_paired_run.svg) · [PNG](figure_4_paired_run.png) · [CSV](figure_4_paired_trace.csv) |
| Fallback status | Actual metrics are marked **Not implemented**. A separate advisory section reports scores below the validation threshold. | [SVG](fallback_status.svg) · [PNG](fallback_status.png) · [CSV](fallback_status.csv) |
| CNN diagram | Editable architecture flow from normalized RGB input through ResNet-18 and global average pooling, 512 features, and one linear gap output. | [Editable SVG](cnn_gap_regressor.svg) · [PNG](cnn_gap_regressor.png) |

Figure 4 uses the maximum observed switch count, breaking ties by selector trace length. Gap curves use decision-time measurements and append the final post-step measurement. The selected specialist and its probability are held between recorded decisions. The diagram uses the configured 224 × 224 camera size. A saved RGB frame was also passed through the frozen generalist to verify the `B × 3 × 224 × 224` input, 512 pooled features and `B × 1` output. [Selector trace for Figure 4](../live/test_03_turn_sunset_sunny/selector.json) · [Camera configuration](../../../car_dreamer/configs/common.yaml) · [CNN implementation](../../../run_gap_pid.py)

## Counts and classification

| Method | Clean completions | Any specification violation | Collision | Gap above 10 m |
| --- | ---: | ---: | ---: | ---: |
| Generalist CNN + PID | 13/27 | 14/27 | 0 | 14 |
| Direct weather/time match | 19/27 | 8/27 | 1 | 7 |
| Learned logistic selector | 16/27 | 11/27 | 3 | 8 |

Completion requires the destination flag alone. Collision, gap-limit and lane/route flags count once per episode in the overall violation metric. In the cause chart, simultaneous flags count once in each relevant category. A task timeout with no specification violation is a separate ending, and infrastructure errors have their own category. These runs recorded no simultaneous flags, lane/route departures, timeouts, other stops or simulator/camera errors. [Derived episode summary](paired_episode_summary.csv) · [Saved run CSV](../live_results.csv)

### Separate two-second results

In the fixed-specialist matched-start test, logistic and direct matching each succeeded in **24/27** two-second windows. The generalist is **Pending** for that particular comparison. A separately derived comparison of the first 20 controlled ticks in the paired live traces gives **22/27** for the generalist and **24/27** for each specialist method. The publication figures use the full-route outcomes in the table above. [Two-second outcome CSV](two_second_outcomes.csv) · [Saved fixed-specialist report](../model_report.json)

### Checkpoint exposure

The generalist was trained on pooled data across the nine conditions and received no synchronized specialist fine-tuning. Direct matching and the learned selector use the same nine deployed specialist checkpoints. Seven received additional synchronized fine-tuning. Night sunny and night rainy use their checkpoints from before fine-tuning because the fine-tuned candidates missed the validation gates. The learned selector additionally fits its logistic heads on separate short controller trials. It scores the deployed specialists using the preset and the frozen generalist’s gap estimate. [Specialist training report](../../all_specialists_2026-11-17/README.md) · [Generalist training report](../../generalist_all_conditions_2026-11-18/README.md) · [Selector implementation](../../../selector/model.py) · [Live controller implementation](../../../selector/rollout.py)

### Fallback advisory

The intended future rule uses a default controller when every specialist scores below the acceptance threshold. On the recorded trajectories, the highest score fell below 0.9222694662 on 142 of 208 decisions across 23 of 27 runs. A below-threshold decision is an advisory request, and specification violations are recorded from the simulator. The actual activation count, percentage of runs needing fallback and percentage of driving time in fallback are **Not implemented**. [Fitted threshold](../model.json) · [Fallback summary](fallback_status.csv)

## Reproduce the figures

From the repository root, run the plotting script with the existing Python environment.

```bash
.venv/bin/python -B -m analysis.plot_context_selector_presentation
```

The script reads the experiment manifest, verified run CSV, and per-step JSON traces in `live/<case_id>/<mode>.json`. It checks the three-arm match and cross-checks the counts against the saved experiment report. All PNGs are exported at **360 dpi**. The code only reads experiment inputs and writes plotted outputs and derived CSVs. [Plotting script](../../../analysis/plot_context_selector_presentation.py) · [Diagram helper](../../../analysis/draw_gap_cnn_diagram.py)

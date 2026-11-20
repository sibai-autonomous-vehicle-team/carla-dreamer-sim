# Live CARLA comparison at a 9 m distance cutoff

Across 90 live rollouts per cutoff, failures increased from **26 at 10 m to 55 at 9 m**. Distance-limit stops increased from 25 to 54. The night-cloudy specialist collided on seed 44 at both cutoffs.

| CNN driving the PID | 10 m live completions | 9 m live completions | 9 m replay completions | Live distance-limit stops, 10 m → 9 m |
| --- | ---: | ---: | ---: | ---: |
| Generalist | 27/45 | 9/45 | 8/45 | 18 → 36 |
| Specialist | 37/45 | 26/45 | 28/45 | 7 → 18 |

## Live completions by condition

Each cell gives completed routes out of five. A completion requires reaching the destination without another terminal flag.

| Condition | Generalist, 10 m → 9 m | Specialist, 10 m → 9 m |
| --- | ---: | ---: |
| Day sunny | 3 → 1 | 4 → 4 |
| Day cloudy | 2 → 1 | 5 → 4 |
| Day rainy | 2 → 0 | 4 → 2 |
| Sunset sunny | 4 → 3 | 5 → 4 |
| Sunset cloudy | 4 → 2 | 5 → 4 |
| Sunset rainy | 3 → 0 | 4 → 3 |
| Night sunny | 3 → 1 | 3 → 0 |
| Night cloudy | 4 → 1 | 4 → 4 |
| Night rainy | 2 → 0 | 3 → 1 |

## Method

Both cutoffs were evaluated on the same five Town03 routes per condition and seeds 42–46. The CNN drove longitudinal PID commands with a 6 m target, gains of 0.9, 0.05 and 0.15, a 0.1 s timestep, five warmup steps and a 360-step limit. Shared route steering, lead-car behavior and the 5 m route deviation limit matched the benchmark. The simulator ends the episode after a step if its measured true gap strictly exceeds the configured cutoff.

The replay applies the 9 m cutoff to the archived 10 m traces and ends each trace at the first crossing. Both live benchmarks recorded matching initial actor types, poses, velocities and gap. Live predictions varied after initialization and 50 of 90 live episodes matched the replay's termination step and reason exactly. The archived night-cloudy collision trace crossed 9 m before impact. In the fresh run the maximum gap was 7.45 m, so the 9 m cutoff did not activate before that collision.

## Commands and outputs

The benchmark ran from the repository root with Python 3.10 and CARLA 0.9.15.

```bash
bash experiments/live_cutoff_9m_2026-11-18/commands.sh
```

The command starts CARLA and runs five routes per model and condition through `run_gap_pid.py`. Each runner log records the batch, and each seed has CSV and JSON files under `live/<mode>/<condition>/`. Copies of the original and tested profiles and route manifests are in `profiles_10m/` and `profiles_9m/`. The active specialist profiles use 9 m and reference the new report.

[Completion plot](comparison.png) · [Condition metrics](comparison.csv) · [Run-level report](summary.json) · [Benchmark log](benchmark.log)

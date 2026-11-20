#!/usr/bin/env bash
# Run from the repository root with CARLA port 2000 available.
set -euo pipefail
E=experiments/generalist_live_2026-11-18
export OMP_NUM_THREADS=2
.venv/bin/python -B -u -m generalist.run_live --output "$E" > "$E/benchmark.log" 2>&1
.venv/bin/python -B -m analysis.summarize_generalist_live --experiment "$E"

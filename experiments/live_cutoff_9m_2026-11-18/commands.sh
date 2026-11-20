#!/usr/bin/env bash
# Execute from the repository root. Port 2000 must be available.
set -euo pipefail
E=experiments/live_cutoff_9m_2026-11-18
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -B -u "$E/run_cutoff_comparison.py" > "$E/benchmark.log" 2>&1
.venv/bin/python -B -m analysis.summarize_distance_cutoff --experiment "$E"

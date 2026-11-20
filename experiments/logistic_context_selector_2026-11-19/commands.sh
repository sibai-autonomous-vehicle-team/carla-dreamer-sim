#!/usr/bin/env bash
# Execute from the repository root with CARLA port 2000 free.
set -euo pipefail
E=experiments/logistic_context_selector_2026-11-19
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -B -m selector.plan --output "$E/plan.json"
.venv/bin/python -B -u -m selector.experiment --experiment "$E" \
  --horizon 2 --decision-interval 2 --windows 2 2>&1 | tee -a "$E/pipeline.log"

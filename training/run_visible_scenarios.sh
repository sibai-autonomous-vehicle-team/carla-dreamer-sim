#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "Run this from a terminal in the CARLA machine's graphical desktop (or remote desktop)." >&2
  exit 1
fi

checkpoint="${1:-training/_local/day_sunny_bc_s42/checkpoint_05000.pt}"
output="${2:-training/_local/day_sunny_bc_visible_$(date +%Y%m%d_%H%M%S)}"

.venv/bin/python -m training.evaluate_closed_loop \
  --start-server --graphics windowed --real-time \
  --modes actor_only --condition day_sunny \
  --checkpoints "$checkpoint" \
  --episodes 20 --steps 200 --seed 12345 \
  --output "$output"

.venv/bin/python -m training.plot_closed_loop --results "$output"

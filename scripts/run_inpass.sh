#!/usr/bin/env bash
# Evaluate the released audio of the in-pass systems and rebuild the experiment table and the figures.
#
#   scripts/run_inpass.sh            # audio -> per-clip rows -> experiments (CPU)
#   scripts/run_inpass.sh --quick    # the same from the committed per-clip rows, no models
#
# The baseline and the reference are read from results/baseline/rows.jsonl (scripts/run_baseline.sh):
# they are not measured again. The full run measures only clips that results/inpass/rows.jsonl lacks.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/inpass.yaml
if [[ "${1:-}" == "--quick" ]]; then
  uv run python -m emoswitch.evaluate --config "$CONFIG" --from-rows
else
  uv run python -m emoswitch.evaluate --config "$CONFIG"
fi
uv run python -m emoswitch.experiments --config "$CONFIG"
uv run python scripts/sync_readme.py --file HW2.md --config "$CONFIG" --check

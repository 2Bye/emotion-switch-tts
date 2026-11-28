#!/usr/bin/env bash
# Error analysis, slices and robustness checks of the system chosen in homework 2 (homework 3).
#
#   scripts/run_diagnostics.sh            # add noise to the released clips, measure them, rebuild the tables (CPU)
#   scripts/run_diagnostics.sh --quick    # the same from the committed rows of the noisy clips, no models
#
# The clean clips are not measured again: their rows are those of results/inpass/ and results/baseline/.
# The full run measures only the noise levels whose file under results/diagnostics/noise/ is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/diagnostics.yaml
if [[ "${1:-}" != "--quick" ]]; then
  uv run python -m emoswitch.perturb --config "$CONFIG"
fi
uv run python -m emoswitch.diagnose --config "$CONFIG"
uv run python scripts/sync_readme.py --file HW3.md --config "$CONFIG" --check

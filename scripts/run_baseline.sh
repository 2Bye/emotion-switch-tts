#!/usr/bin/env bash
# Evaluate the released baseline audio and rebuild every table and figure.
#
#   scripts/run_baseline.sh            # audio -> per-clip rows -> metrics -> tables and figures (CPU)
#   scripts/run_baseline.sh --quick    # the same from the committed per-clip rows, no models
#
# The full run downloads the judge (0.7 GB), the aligner (1.3 GB) and the ASR model (3.1 GB) on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/baseline.yaml
if [[ "${1:-}" == "--quick" ]]; then
  uv run python -m emoswitch.evaluate --config "$CONFIG" --from-rows
else
  uv run python -m emoswitch.evaluate --config "$CONFIG"
fi
uv run python -m emoswitch.report --config "$CONFIG"
uv run python -m emoswitch.eda --config "$CONFIG"
uv run python scripts/compare_with_reference.py > /dev/null
uv run python scripts/sync_readme.py --check

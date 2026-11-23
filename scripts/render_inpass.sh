#!/usr/bin/env bash
# Render every run of the in-pass experiment with the fine-tuned AR model (GPU).
# Needs a CUDA build of torch in the environment (the lock file resolves the default PyPI wheel).
#
#   scripts/render_inpass.sh <merged checkpoint dir> [audio root]
#
# The checkpoint is published separately: https://huggingface.co/byebye9/qwen3-tts-emotion-instruct
# The steering directions (config `inpass.steer.directions`) are estimated first if the file is absent.
# Writes <audio root>/<run>/*.wav and a manifest.jsonl per run.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: scripts/render_inpass.sh <merged checkpoint dir> [audio root]}
OUT=${2:-data/audio_rendered}
CONFIG=configs/inpass.yaml

DIRECTIONS=$(uv run python -c "
from emoswitch.config import load_config
print(load_config('$CONFIG')['inpass']['steer']['directions'])")
if [[ ! -f "$DIRECTIONS" ]]; then
  uv run --extra gen python scripts/estimate_steer.py --config "$CONFIG" --checkpoint "$CKPT"
fi
uv run python -c "
from emoswitch.config import load_config
for kind, set_name, seed in load_config('$CONFIG')['runs']:
    print(kind, set_name, seed)
" | while read -r KIND SET SEED; do
  uv run --extra gen python -m emoswitch.generate --config "$CONFIG" --checkpoint "$CKPT" \
    --set "$SET" --kind "$KIND" --seed "$SEED" --out "$OUT" < /dev/null
done

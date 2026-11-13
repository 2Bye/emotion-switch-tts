#!/usr/bin/env bash
# Render every run of the baseline experiment with the fine-tuned AR model (GPU, a few minutes per run).
# Needs a CUDA build of torch in the environment (the lock file resolves the default PyPI wheel).
#
#   scripts/render_baseline.sh <merged checkpoint dir> [audio root]
#
# The checkpoint is published separately: https://huggingface.co/byebye9/qwen3-tts-emotion-instruct
# Writes <audio root>/<run>/*.wav and a manifest.jsonl per run; concatenate those into
# data/manifest.jsonl to evaluate the new audio.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: scripts/render_baseline.sh <merged checkpoint dir> [audio root]}
OUT=${2:-data/audio_rendered}
CONFIG=configs/baseline.yaml

# The list of runs lives in the config: [kind, set, seed] per line.
uv run python -c "
import yaml
for kind, set_name, seed in yaml.safe_load(open('$CONFIG'))['runs']:
    print(kind, set_name, seed)
" | while read -r KIND SET SEED; do
  uv run --extra gen python -m emoswitch.generate --config "$CONFIG" --checkpoint "$CKPT" \
    --set "$SET" --kind "$KIND" --seed "$SEED" --out "$OUT" < /dev/null
done

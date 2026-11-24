"""Estimate the steering directions of the in-pass systems (GPU step).

    python scripts/estimate_steer.py --config configs/inpass.yaml --checkpoint <merged R dir>

For every sentence of config `inpass.steer.texts` and every emotion the model renders one single pass
(the decoding of the experiment). Each render is then run through the decoder once more, teacher-forced,
and the output of every decoder layer is averaged over the codec positions. The direction of a pair is
the difference of these means between the two emotions, per layer, scaled to unit length:

    direction[e1>e2][layer] = unit(mean over texts of h(e2) - mean over texts of h(e1))

The sentences are in no evaluation set, so nothing about the evaluated phrases enters the directions.
Writes <directions>.npz ({"e1>e2": [layers, d] float32}) and <directions>.json (what it was made from
and the size of each difference relative to the hidden state, per layer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from emoswitch.config import ROOT, load_config
from emoswitch.inpass import InPassEngine, load_model
from emoswitch.stimuli import emotions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/inpass.yaml")
    ap.add_argument("--checkpoint", required=True, help="directory of the merged fine-tuned model")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    sc, adv = cfg["inpass"]["steer"], cfg["adverbs"]
    texts = json.loads((ROOT / sc["texts"]).read_text(encoding="utf-8"))["texts"]
    engine = InPassEngine(load_model(args.checkpoint, args.device), cfg["speaker"], cfg["generation"])
    per_text = {e: [] for e in emotions(cfg)}
    for i, text in enumerate(texts, 1):
        for e in per_text:
            instruct = cfg["inpass"]["instruct"].format(adv=adv[e])
            codes, _ = engine.single_pass(text, instruct, sc["seed"])
            per_text[e].append(engine.layer_means(text, instruct, codes))
        print(f"[steer] {i}/{len(texts)} sentences", flush=True)
    mean = {e: torch.stack(v).mean(0) for e, v in per_text.items()}  # [layers, d]
    h_norm = torch.stack([m.norm(dim=-1) for m in mean.values()]).mean(0)  # [layers]
    directions, relative = {}, {}
    for e1 in mean:
        for e2 in mean:
            if e1 != e2:
                diff = mean[e2] - mean[e1]
                directions[f"{e1}>{e2}"] = (diff / diff.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
                relative[f"{e1}>{e2}"] = [round(float(x), 4) for x in diff.norm(dim=-1) / h_norm]
    out = ROOT / sc["directions"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **directions)
    weights = Path(args.checkpoint) / "model.safetensors"
    meta = {
        "config": cfg["config_path"],
        "texts": sc["texts"],
        "n_texts": len(texts),
        "emotions": list(mean),
        "seed": sc["seed"],
        "layers": int(h_norm.shape[0]),
        "hidden_size": int(next(iter(mean.values())).shape[1]),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights.exists() else None,
        "difference_norm_over_hidden_norm": relative,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[steer] {len(directions)} directions x {meta['layers']} layers -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

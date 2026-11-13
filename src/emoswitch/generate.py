"""Render one run of the baseline experiment with the fine-tuned AR model (GPU step).

    python -m emoswitch.generate --config configs/baseline.yaml --checkpoint <merged R dir> \\
        --set heldout --kind inline --seed 42

`--kind inline` renders the baseline (one pass, instruction naming both emotions) with its no-switch
rows; `--kind stitch` renders the reference (each sentence on its own, then joined) with its
no-switch rows. `--dry-run` prints the rows and touches nothing.

The checkpoint is not part of this repository. Nothing is trained here: the baseline is a way of
asking an existing model, so the only learned object is the checkpoint named on the command line.
Needs the `gen` extra and a CUDA build of torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from emoswitch.config import ROOT, load_config, load_phrases
from emoswitch.stimuli import build_rows


def crossfade(a: np.ndarray, b: np.ndarray, sample_rate: int, ms: float) -> np.ndarray:
    """Equal-power crossfade of the last `ms` of `a` into the first `ms` of `b`."""
    m = int(sample_rate * ms / 1000)
    if m <= 0 or len(a) < m or len(b) < m:
        return np.concatenate([a, b])
    t = (np.arange(m, dtype=np.float32) + 0.5) / m
    mix = a[-m:] * np.cos(t * np.pi / 2) + b[:m] * np.sin(t * np.pi / 2)
    return np.concatenate([a[:-m], mix, b[m:]])


def run_name(kind: str, set_name: str, seed: int) -> str:
    return f"{kind}_{set_name}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--checkpoint", help="directory of the merged fine-tuned model")
    ap.add_argument("--set", required=True, dest="set_name")
    ap.add_argument("--kind", required=True, choices=["inline", "stitch"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/audio_rendered", help="audio root; the run gets its own subdirectory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dry-run", action="store_true", help="print the rows as JSON lines; no model, no files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.set_name not in cfg["sets"]:
        raise SystemExit(f"[generate] unknown set {args.set_name!r}; the config has {list(cfg['sets'])}")
    gen = cfg["generation"]
    run = run_name(args.kind, args.set_name, args.seed)
    rows = build_rows(cfg, args.set_name, args.kind, args.seed)
    if args.dry_run:
        for row in rows:
            print(json.dumps({"run": run, **row}, ensure_ascii=False))
        return
    if not args.checkpoint:
        raise SystemExit("[generate] --checkpoint is required unless --dry-run")
    out_dir = (ROOT / args.out / run).resolve()
    if out_dir.exists() and any(p.suffix in (".wav", ".flac") for p in out_dir.iterdir()):
        raise SystemExit(f"[generate] {out_dir} already holds audio; render into a fresh directory")

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from transformers import set_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[generate] no CUDA device: install a CUDA build of torch or pass --device cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = Qwen3TTSModel.from_pretrained(args.checkpoint, device_map=args.device, dtype=dtype)
    # The vendor wrapper refuses `instruct` for 0.6B custom-voice models (its check is a substring
    # test against "0b6"); the fine-tuned checkpoint was trained with the instruction channel.
    model.model.tts_model_size = "0b6+instruct-ft"
    # Greedy decoding of the first codebook; the residual codebooks are sampled, so the seed matters.
    keys = ("max_new_tokens", "do_sample", "repetition_penalty", "subtalker_dosample", "subtalker_temperature",
            "subtalker_top_k", "subtalker_top_p")  # fmt: skip
    gen_kwargs = {k: gen[k] for k in keys}

    def synth(text: str, instruct: str | None) -> tuple[np.ndarray, int]:
        set_seed(args.seed)
        wavs, sr = model.generate_custom_voice(
            text=text, language=gen["language"], speaker=cfg["speaker"], instruct=instruct, **gen_kwargs
        )
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().to(torch.float32).numpy()
        return np.asarray(wav).reshape(-1), int(sr)

    phrases = {p["id"]: p for p in load_phrases(cfg, args.set_name)}
    stitch = cfg["systems"]["stitch"]
    manifest = []
    for row in rows:
        switch_time = None
        if row["condition"] == "stitch":
            p, adv = phrases[row["phrase_id"]], cfg["adverbs"]
            first, sr = synth(p["s1"], stitch["instruct"].format(adv=adv[row["emotion"]]))
            second, _ = synth(p["s2"], stitch["instruct"].format(adv=adv[row["emotion2"]]))
            wav = crossfade(first, second, sr, stitch["crossfade_ms"])
            switch_time = round(len(first) / sr, 3)  # end of the first segment, as the detector expects
        else:
            wav, sr = synth(row["text"], row["instruct"])
        path = out_dir / f"{row['id']}.wav"
        sf.write(path, wav, sr)
        audio = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        manifest.append(
            {
                "run": run,
                **row,
                "switch_time_s": switch_time,
                "audio": audio,
                "sample_rate": sr,
                "sha256_wav_source": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(f"[generate] {run}/{row['id']} ({len(wav) / sr:.2f} s)", flush=True)
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8"
    )
    print(f"[generate] {len(manifest)} clips -> {out_dir}; its manifest.jsonl has the schema of data/manifest.jsonl")


if __name__ == "__main__":
    main()

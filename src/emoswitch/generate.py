"""Render one run of the baseline experiment with the fine-tuned AR model (GPU step).

    python -m emoswitch.generate --config configs/baseline.yaml --checkpoint <merged R dir> \\
        --set heldout --kind inline --seed 42

`--kind inline` renders the baseline (one pass, instruction naming both emotions) with its no-switch
rows; `--kind stitch` renders the reference (each sentence on its own, then joined) with its
no-switch rows; `--kind inpass` (config configs/inpass.yaml) renders the in-pass systems and their
null rows. `--dry-run` prints the rows and touches nothing.

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
from emoswitch.stimuli import build_rows, inpass_systems


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


def render_inpass(cfg: dict, rows: list[dict], checkpoint: str, device: str, seed: int, out_dir):
    """Yield (row, wav, sample rate, render info) for the rows of an `inpass` run.

    Per phrase: one single pass per emotion (kept as wav under `<out>/single_pass/`), the sentence
    boundary of each from forced alignment, then every row as a two-phase render. A row whose boundary
    cannot be placed is emitted as the plain single pass of its first emotion and marked `no_boundary`:
    it is released and counted like any other clip."""
    import soundfile as sf
    import torch

    from emoswitch.align import ForcedAligner
    from emoswitch.inpass import InPassEngine, anchor_frames, boundary_time, load_model
    from emoswitch.judge import load_audio

    ip, jc, dc, adv = cfg["inpass"], cfg["judge"], cfg["detector"], cfg["adverbs"]
    specs = inpass_systems(cfg)
    engine = InPassEngine(load_model(checkpoint, device), cfg["speaker"], cfg["generation"])
    aligner = ForcedAligner(jc["sample_rate"], "cpu")  # as in the evaluation; keeps the GPU for the TTS model
    directions = None
    if any(spec["steer"] for spec in specs.values()):
        directions = {k: torch.from_numpy(v) for k, v in np.load(ROOT / ip["steer"]["directions"]).items()}
    (out_dir / "single_pass").mkdir()
    passes = {}  # (phrase, emotion) -> codes of the single pass and its boundary frame

    def single_pass(row: dict, emotion: str) -> dict:
        key = (row["phrase_id"], emotion)
        if key not in passes:
            codes, _ = engine.single_pass(row["text"], ip["instruct"].format(adv=adv[emotion]), seed)
            wav, sr = engine.decode(codes)
            path = out_dir / "single_pass" / f"{emotion}-{row['phrase_id']}.wav"
            sf.write(path, wav, sr)
            y = load_audio(str(path), jc["sample_rate"], jc["lufs"])
            duration = len(y) / jc["sample_rate"]
            t = boundary_time(aligner(y, row["text"]), row["switch_word"], duration, dc["switch_search_from_frac"])
            n = None if t is None else int(round(t * engine.fps))
            if n is not None and not 3 <= n < codes.shape[0] - 2:
                n = None
            passes[key] = {"codes": codes, "kept_frames": n}
        return passes[key]

    for row in rows:
        e1, e2 = row["emotion"], row["emotion2"]
        first = single_pass(row, e1)
        n = first["kept_frames"]
        if n is None:
            wav, sr = engine.decode(first["codes"])
            yield row, wav, sr, {"no_boundary": True}
            continue
        kept = first["codes"][:n]
        info = {"kept_frames": n, "boundary_s": round(n / engine.fps, 3)}
        context = steer = None
        spec = specs.get(row["condition"], {"anchor": False, "steer": False})  # the null row uses neither
        if spec["anchor"]:
            source = single_pass(row, e2)["codes"]
            k = anchor_frames(n, int(source.shape[0]), ip["anchor"]["share"], ip["anchor"]["min_tail_frames"])
            info["anchor_frames"] = k
            if k:
                context = torch.cat([source[:k], kept[k:]], dim=0)
        if spec["steer"]:
            layer = ip["steer"]["layer"]
            steer = (layer, ip["steer"]["alpha"], directions[f"{e1}>{e2}"][layer])
        codes, ended = engine.continue_from(
            row["text"], kept, ip["instruct"].format(adv=adv[e2]), seed, context=context, steer=steer
        )
        info["frames"], info["ended"] = int(codes.shape[0]), ended
        wav, sr = engine.decode(codes)
        yield row, wav, sr, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--checkpoint", help="directory of the merged fine-tuned model")
    ap.add_argument("--set", required=True, dest="set_name")
    ap.add_argument("--kind", required=True, choices=["inline", "stitch", "stitchnull", "inpass"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/audio_rendered", help="audio root; the run gets its own subdirectory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dry-run", action="store_true", help="print the rows as JSON lines; no model, no files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    known = [*cfg["sets"], *cfg.get("planned_sets", {})]
    if args.set_name not in known:
        raise SystemExit(f"[generate] unknown set {args.set_name!r}; the config has {known}")
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

    def release(row: dict, wav: np.ndarray, sr: int, extra: dict) -> dict:
        path = out_dir / f"{row['id']}.wav"
        sf.write(path, wav, sr)
        print(f"[generate] {run}/{row['id']} ({len(wav) / sr:.2f} s)", flush=True)
        return {
            "run": run,
            **row,
            **extra,
            "audio": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sample_rate": sr,
            "sha256_wav_source": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def write_manifest(manifest: list[dict]) -> None:
        (out_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8"
        )
        print(f"[generate] {len(manifest)} clips -> {out_dir}; its manifest.jsonl has the schema of the data manifest")

    if args.kind == "inpass":
        manifest = []
        for row, wav, sr, info in render_inpass(cfg, rows, args.checkpoint, args.device, args.seed, out_dir):
            # one decode of the whole phrase: there is no boundary by construction
            manifest.append(release(row, wav, sr, {"switch_time_s": None, "render": info}))
        write_manifest(manifest)
        return
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
        if row["condition"] in ("stitch", "stitch_null"):
            p, adv = phrases[row["phrase_id"]], cfg["adverbs"]
            first, sr = synth(p["s1"], stitch["instruct"].format(adv=adv[row["emotion"]]))
            second, _ = synth(p["s2"], stitch["instruct"].format(adv=adv[row["emotion2"]]))
            wav = crossfade(first, second, sr, stitch["crossfade_ms"])
            switch_time = round(len(first) / sr, 3)  # end of the first segment, as the detector expects
        else:
            wav, sr = synth(row["text"], row["instruct"])
        manifest.append(release(row, wav, sr, {"switch_time_s": switch_time}))
    write_manifest(manifest)


if __name__ == "__main__":
    main()

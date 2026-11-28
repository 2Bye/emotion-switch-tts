"""Measure released clips again after white noise is added to them.

    python -m emoswitch.perturb --config configs/diagnostics.yaml             # every level of the config
    python -m emoswitch.perturb --config configs/diagnostics.yaml --snr 20    # one level

The clips of the conditions and the set named under `noise` in the config are read from the manifest,
noise is added at the given signal-to-noise ratio and the result is measured with the same judge,
aligner and ASR as the clean clip (evaluate.measure). The generator is not run. The noisy audio is
written under `noise.audio_root` and is not tracked: it is rebuilt from the released clip, the seed of
the config, the level and the clip id. Rows are written to results/<name>/noise/snr_<level>.jsonl;
a level whose file is complete is skipped.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import zlib

import numpy as np

from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import measure, measure_fingerprint, read_jsonl, write_jsonl


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """`y` plus white Gaussian noise whose power is the power of `y` divided by 10^(snr_db/10).
    The sum is scaled down as a whole if its peak would clip."""
    power = float(np.mean(np.square(y, dtype=np.float64)))
    noise = rng.standard_normal(len(y)) * np.sqrt(power / 10 ** (snr_db / 10))
    out = y.astype(np.float64) + noise
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)


def clip_rng(seed: int, snr_db: float, run: str, clip_id: str) -> np.random.Generator:
    """One stream per clip and level, independent of the order in which clips are processed."""
    return np.random.default_rng([seed, int(round(snr_db * 10)) + 10_000, zlib.crc32(f"{run}/{clip_id}".encode())])


def selected(cfg: dict) -> list[dict]:
    nc = cfg["noise"]
    manifest = read_jsonl(ROOT / cfg["manifest"])
    return [m for m in manifest if m["set"] == nc["set"] and m["condition"] in nc["conditions"]]


def level_path(cfg: dict, snr_db: float):
    return ROOT / "results" / cfg["name"] / "noise" / f"snr_{snr_db:g}.jsonl"


def run_level(cfg: dict, snr_db: float, device: str) -> None:
    import soundfile as sf

    nc = cfg["noise"]
    clips = selected(cfg)
    path = level_path(cfg, snr_db)
    if path.exists() and len(read_jsonl(path)) == len(clips):
        print(f"[perturb] snr {snr_db:g} dB: {path.name} is complete")
        return
    started = time.time()
    noisy = []
    for m in clips:
        y, sr = sf.read(ROOT / m["audio"], dtype="float32")
        out = ROOT / nc["audio_root"] / f"snr_{snr_db:g}" / m["run"] / f"{m['id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, add_noise(y, snr_db, clip_rng(nc["seed"], snr_db, m["run"], m["id"])), sr, subtype="PCM_16")
        noisy.append({**m, "audio": str(out.relative_to(ROOT))})
    rows = measure(cfg, noisy, device)
    for row, m in zip(rows, clips, strict=True):
        row.update(audio=m["audio"], snr_db=snr_db)  # the released clip the noise was added to
    write_jsonl(path, rows)
    meta = {
        "measure_fingerprint": measure_fingerprint(cfg),
        "snr_db": snr_db,
        "noise_seed": nc["seed"],
        "n_clips": len(rows),
        "seconds": round(time.time() - started),
        "device": device,
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[perturb] snr {snr_db:g} dB: {len(rows)} clips in {meta['seconds']} s -> {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/diagnostics.yaml")
    ap.add_argument("--snr", type=float, default=None, help="one level of `noise.snr_db` (default: all)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    levels = cfg["noise"]["snr_db"] if args.snr is None else [args.snr]
    for snr_db in levels:
        run_level(cfg, float(snr_db), args.device)


if __name__ == "__main__":
    main()

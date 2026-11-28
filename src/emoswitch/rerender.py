"""Render the chosen system again on this machine and measure the new clips.

    python -m emoswitch.rerender --config configs/diagnostics.yaml --checkpoint <merged R dir>   # render + measure
    python -m emoswitch.rerender --config configs/diagnostics.yaml                               # measure only

For every seed under `rerender` in the config the rows of the named conditions are built as in
homework 2 (stimuli.build_rows) and rendered by the same code (generate.render_inpass) on the device of
the config. The audio is stored as FLAC, the manifest as `rerender.manifest`; a seed whose run is
complete in the manifest is not rendered again. Then every clip of the manifest that
results/<name>/rerender/rows.jsonl lacks is measured with the judge, aligner and ASR of the config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time

import numpy as np

from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import measure, measure_fingerprint, read_jsonl, split_measured, write_jsonl
from emoswitch.stimuli import build_rows


def run_name(set_name: str, seed: int) -> str:
    return f"rerender_{set_name}_s{seed}"


def planned_rows(cfg: dict, seed: int) -> list[dict]:
    rc = cfg["rerender"]
    return [r for r in build_rows(cfg, rc["set"], "inpass", seed) if r["condition"] in rc["conditions"]]


def render_seed(cfg: dict, seed: int, checkpoint: str) -> list[dict]:
    import soundfile as sf

    from emoswitch.generate import render_inpass

    rc = cfg["rerender"]
    run = run_name(rc["set"], seed)
    wav_dir, flac_dir = ROOT / rc["wav_root"] / run, ROOT / rc["audio_root"] / run
    for d in (wav_dir, flac_dir):  # a run is rendered whole: an interrupted one starts again
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    manifest = []
    for row, wav, sr, info in render_inpass(cfg, planned_rows(cfg, seed), checkpoint, rc["device"], seed, wav_dir):
        wav_path, flac_path = wav_dir / f"{row['id']}.wav", flac_dir / f"{row['id']}.flac"
        sf.write(wav_path, wav, sr)
        samples, _ = sf.read(wav_path, dtype="int16")
        sf.write(flac_path, samples, sr, subtype="PCM_16")
        if not np.array_equal(sf.read(flac_path, dtype="int16")[0], samples):
            raise SystemExit(f"[rerender] {flac_path}: FLAC does not decode to the source samples")
        manifest.append(
            {
                "run": run,
                **row,
                "switch_time_s": None,
                "render": info,
                "audio": str(flac_path.relative_to(ROOT)),
                "sample_rate": sr,
                "sha256_wav_source": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            }
        )
        print(f"[rerender] {run}/{row['id']} ({len(wav) / sr:.2f} s)", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/diagnostics.yaml")
    ap.add_argument("--checkpoint", default=None, help="directory of the merged model; without it nothing is rendered")
    ap.add_argument("--device", default="cpu", help="device of the measurement")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rc = cfg["rerender"]
    manifest_path = ROOT / rc["manifest"]
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []
    for seed in rc["seeds"] if args.checkpoint else []:
        run = run_name(rc["set"], seed)
        if sum(m["run"] == run for m in manifest) == len(planned_rows(cfg, seed)):
            print(f"[rerender] {run} is complete in {rc['manifest']}")
            continue
        started = time.time()
        manifest = [m for m in manifest if m["run"] != run] + render_seed(cfg, seed, args.checkpoint)
        write_jsonl(manifest_path, manifest)
        print(f"[rerender] {run}: rendered in {round(time.time() - started)} s on {rc['device']}", flush=True)

    out = ROOT / "results" / cfg["name"] / "rerender"
    rows_path = out / "rows.jsonl"
    kept, todo = split_measured(manifest, read_jsonl(rows_path) if rows_path.exists() else [])
    print(f"[rerender] {len(kept)} clips already measured, {len(todo)} to measure")
    started = time.time()
    by_key = {(r["run"], r["id"]): r for r in kept + (measure(cfg, todo, args.device) if todo else [])}
    write_jsonl(rows_path, [by_key[(m["run"], m["id"])] for m in manifest])
    if todo:
        meta_path = out / "measure_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"passes": []}
        meta["measure_fingerprint"] = measure_fingerprint(cfg)
        meta["passes"].append({"n_measured": len(todo), "seconds": round(time.time() - started),
                               "device": args.device, "platform": f"{platform.system()} {platform.machine()}",
                               "python": platform.python_version()})  # fmt: skip
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

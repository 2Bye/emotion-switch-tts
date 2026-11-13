"""Turn render runs exported from the GPU node into the released data of this repository.

    python scripts/import_node_export.py --export <dir with runs/ and rendered/>

Runs come in two kinds. `runs/<node dir>` was rendered and measured by the probe scripts of the
source project; `rendered/<run>` was rendered by this repository's `emoswitch.generate`.
For every run listed in the config it
  * keeps the rows of the baseline experiment (baseline or reference, plus their no-switch rows);
  * checks that each kept row is exactly the row `emoswitch.stimuli` builds from the config
    (id, text, instruction, emotions, expected sign), so the released audio is what the config says;
  * stores the audio as FLAC and verifies that the decoded samples equal the source wav;
  * writes data/manifest.jsonl (what to evaluate) and, for probe runs, data/reference/<run>.jsonl
    (what the node measured on the same clips, kept for comparison only: evaluation never reads it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from emoswitch.config import ROOT, load_config
from emoswitch.stimuli import build_rows

# node run directory -> (run kind, set, seed)
RUNS = {
    "inline_R_p2a_v2": ("inline", "p2a", 42),
    "inline_R_heldout_v2": ("inline", "heldout", 42),
    "hw1_inline_R_p2a_s43": ("inline", "p2a", 43),
    "hw1_inline_R_heldout_s43": ("inline", "heldout", 43),
    "hw1_inline_R_p2a_s44": ("inline", "p2a", 44),
    "hw1_inline_R_heldout_s44": ("inline", "heldout", 44),
    "segcont_R": ("stitch", "p2a", 42),
    "segcont_R_seed43": ("stitch", "p2a", 43),
    "segcont_R_heldout": ("stitch", "heldout", 42),
}
NODE_CONDITION = {"naive": "stitch"}  # the node calls the stitched reference "naive"
CHECKED = ("id", "condition", "emotion", "emotion2", "text", "instruct", "expect_sign", "switch_word")
REFERENCE_FIELDS = ("duration_s", "wer_target", "prefix_leakage", "looped", "hyp", "switch_time_s")
PROFILE_FIELDS = ("profile_centres_s", "profile_arousal", "profile_sd", "words", "step", "align_backend")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_probe_run(src: Path, expected: dict, run: str, audio_dir: Path) -> tuple[list, list, dict]:
    """A run rendered and measured by the source project's probe scripts."""
    node_rows = {r["id"]: r for r in read_jsonl(src / "t0_report.jsonl")}
    profiles = {r["id"]: r for r in read_jsonl(src / "profile_report.jsonl")}
    missing = sorted(set(expected) - set(node_rows))
    if missing:
        raise SystemExit(f"[import] {src.name}: rows the config asks for are absent: {missing}")
    manifest, reference = [], []
    for row_id, exp in expected.items():
        node = dict(node_rows[row_id])
        node["condition"] = NODE_CONDITION.get(node["condition"], node["condition"])
        diff = {k: (node.get(k), exp[k]) for k in CHECKED if node.get(k) != exp[k]}
        if diff:
            raise SystemExit(f"[import] {src.name}/{row_id}: node row differs from the config: {diff}")
        switch_time = node.get("switch_time_s") if exp["condition"] == "stitch" else None
        manifest.append({"run": run, **exp, "switch_time_s": switch_time,
                         **to_flac(src / "audio" / f"{row_id}.wav", audio_dir / f"{row_id}.flac")})  # fmt: skip
        prof = profiles.get(row_id, {})
        reference.append({"run": run, "id": row_id, "condition": exp["condition"],
                          **{k: node.get(k) for k in REFERENCE_FIELDS},
                          **{k: prof.get(k) for k in PROFILE_FIELDS}})  # fmt: skip
    meta = json.loads((src / "probe_meta.json").read_text(encoding="utf-8"))
    summary = src / "profile_summary.json"
    provenance = {
        "rendered_by": "probe script of the source project",
        "node_run": f"runs/{src.name}",
        "n_rows_released": len(manifest),
        "n_rows_in_node_run": len(node_rows),
        "conditions_in_node_run": sorted({r["condition"] for r in node_rows.values()}),
        "checkpoint": Path(meta.get("model") or meta.get("cv_model") or "").name or None,
        "probe": {k: meta.get(k) for k in ("run_id", "model_sha256", "seed", "gen_preset", "argv", "git", "versions")},
        "profile": (
            {k: json.loads(summary.read_text()).get(k) for k in ("window_s", "hop_s", "align", "detect_threshold")}
            if summary.exists()
            else None
        ),
    }
    return manifest, reference, provenance


def import_rendered_run(src: Path, expected: dict, run: str, audio_dir: Path) -> tuple[list, dict]:
    """A run rendered by this repository's own `emoswitch.generate`; nothing was measured on the node."""
    rendered = {r["id"]: r for r in read_jsonl(src / "manifest.jsonl")}
    manifest = []
    for row_id, exp in expected.items():
        got = rendered[row_id]
        diff = {k: (got.get(k), exp[k]) for k in CHECKED if got.get(k) != exp[k]}
        if diff:
            raise SystemExit(f"[import] {run}/{row_id}: rendered row differs from the config: {diff}")
        flac = to_flac(src / f"{row_id}.wav", audio_dir / f"{row_id}.flac")
        if flac["sha256_wav_source"] != got["sha256_wav_source"]:
            raise SystemExit(f"[import] {run}/{row_id}: wav does not match the hash in its manifest")
        manifest.append({"run": run, **exp, "switch_time_s": got["switch_time_s"], **flac})
    return manifest, {"rendered_by": "emoswitch.generate of this repository", "n_rows_released": len(manifest)}


def to_flac(wav_path: Path, flac_path: Path) -> dict:
    """Store the wav losslessly as FLAC and check that it decodes to the same samples."""
    samples, sr = sf.read(wav_path, dtype="int16")
    sf.write(flac_path, samples, sr, subtype="PCM_16")
    if not np.array_equal(sf.read(flac_path, dtype="int16")[0], samples):
        raise SystemExit(f"[import] {flac_path}: FLAC does not decode to the source samples")
    return {
        "audio": str(flac_path.relative_to(ROOT)),
        "sample_rate": sr,
        "sha256_wav_source": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="directory with runs/ (probe runs) and rendered/ (generate runs)")
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    export = Path(args.export)
    node_dir = {v: k for k, v in RUNS.items()}
    manifest, provenance = [], {}
    for kind, set_name, seed in cfg["runs"]:
        run = f"{kind}_{set_name}_s{seed}"
        expected = {r["id"]: r for r in build_rows(cfg, set_name, kind, seed)}
        audio_dir = ROOT / "data" / "audio" / run
        audio_dir.mkdir(parents=True, exist_ok=True)
        probe_src = export / "runs" / node_dir.get((kind, set_name, seed), "-")
        rendered_src = export / "rendered" / run
        if probe_src.exists():
            rows, reference, provenance[run] = import_probe_run(probe_src, expected, run, audio_dir)
            ref_path = ROOT / "data" / "reference" / f"{run}.jsonl"
            ref_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reference), encoding="utf-8")
        elif rendered_src.exists():
            rows, provenance[run] = import_rendered_run(rendered_src, expected, run, audio_dir)
        else:
            raise SystemExit(f"[import] {run}: neither {probe_src} nor {rendered_src} exists")
        manifest += rows
        print(f"[import] {run}: {len(rows)} clips ({provenance[run]['rendered_by']})")
    manifest_path = ROOT / "data" / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")
    (ROOT / "data" / "reference" / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[import] {len(manifest)} clips in {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

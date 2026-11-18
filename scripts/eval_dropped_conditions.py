"""Measure the conditions that were dropped when the node runs were imported. Exploratory only.

    python scripts/eval_dropped_conditions.py              # node wavs -> FLAC, manifest -> rows -> report (CPU)
    python scripts/eval_dropped_conditions.py --from-rows  # re-score the stored rows, no models

The node runs in data/raw_export/runs/ also hold rows the baseline experiment does not use: `inline_tag`
(ids `inline-tag-*`) in the inline runs and `cont` (ids `segcont-cont-*`) in the segcont runs. They are
measured here with this repository's judge, aligner, ASR and detector. Writes data/audio_dropped/<run>/,
data/manifest_dropped.jsonl, results/<name>/dropped_rows.jsonl and results/<name>/dropped_conditions.md.
Nothing here is read by the main evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from import_node_export import RUNS, read_jsonl, to_flac

from emoswitch.config import ROOT, load_config, load_phrases
from emoswitch.evaluate import BOUNDARIES, add_derived, measure, write_jsonl
from emoswitch.report import table
from emoswitch.stimuli import phrase_text

DROPPED = {"inline_tag": "inline-tag-", "cont": "segcont-cont-"}


def node_rows(export: Path) -> list[tuple[str, str, int, dict]]:
    """(run, set, seed, node row) of every dropped row, in the order of RUNS."""
    out = []
    for node_dir, (_, set_name, seed) in RUNS.items():
        for r in read_jsonl(export / "runs" / node_dir / "t0_report.jsonl"):
            if r["condition"] in DROPPED:
                assert r["id"].startswith(DROPPED[r["condition"]]), r["id"]
                out.append((node_dir, set_name, seed, r))
    return out


def build_manifest(cfg: dict, export: Path) -> list[dict]:
    """Store the dropped clips as FLAC and describe them in the schema of data/manifest.jsonl."""
    manifest = []
    for node_dir, set_name, seed, r in node_rows(export):
        kind = RUNS[node_dir][0]
        run = f"{kind}_{set_name}_s{seed}"
        phrase_id = r["id"].rsplit("-", 1)[1]
        phrase = next(p for p in load_phrases(cfg, set_name) if p["id"] == phrase_id)
        if (r["text"], r["switch_word"]) != (phrase_text(phrase), phrase["switch_word"]):
            raise SystemExit(f"[dropped] {node_dir}/{r['id']}: text differs from the phrase file")
        audio_dir = ROOT / "data" / "audio_dropped" / run
        audio_dir.mkdir(parents=True, exist_ok=True)
        flac = to_flac(export / "runs" / node_dir / "audio" / f"{r['id']}.wav", audio_dir / f"{r['id']}.flac")
        manifest.append(
            {
                "run": run,
                "set": set_name,
                "seed": seed,
                "phrase_id": phrase_id,
                "text": r["text"],
                "switch_word": r["switch_word"],
                "id": r["id"],
                "condition": r["condition"],
                "emotion": r["emotion"],
                "emotion2": r["emotion2"],
                "instruct": r["instruct"],
                "node_full_text": r["full_text"],
                "expect_sign": r["expect_sign"],
                # `cont` is stitched: the node row carries the end of the first segment
                "switch_time_s": r.get("switch_time_s") if r["condition"] == "cont" else None,
                **flac,
            }
        )
    return manifest


def describe(rows: list[dict]) -> str:
    """What each condition is, from the fields of the stored rows only."""
    out = []
    for cond in DROPPED:
        sub = [r for r in rows if r["condition"] == cond]
        instructs = sorted({f"`{r['instruct']}`" if r["instruct"] else "null" for r in sub})
        example = sub[0]
        text = (
            f"**`{cond}`** ({len(sub)} rows). `instruct` takes the values {', '.join(instructs)}. "
            f'Example for `{example["id"]}`, the text the model was given: "{example["node_full_text"]}"; '
            f'the text the clip is scored against: "{example["text"]}".'
        )
        if cond == "cont":
            text += " The boundary by construction is the end of the first segment, as the node row records it."
        out.append(text)
    return "\n\n".join(out)


def counts_table(rows: list[dict], gc: dict) -> str:
    out = []
    for key in dict.fromkeys((r["condition"], r["set"], r["seed"]) for r in rows):
        sub = [r for r in rows if (r["condition"], r["set"], r["seed"]) == key]
        by_rule = [str(sum(r["decisions"][b] for r in sub)) for b in BOUNDARIES]
        gated = sum(r["decisions"][r["boundary"]] and r["valid"] for r in sub)
        speech = [sum(not r["valid"] for r in sub), sum(r["leak"] for r in sub),
                  sum(r["wer"] > gc["max_clip_wer"] for r in sub)]  # fmt: skip
        out.append([*map(str, key), str(len(sub)), *by_rule, str(gated), *map(str, speech)])
    header = ["condition", "set", "seed", "n", "detected, stitched boundary by construction", "by forced alignment",
              "with validity gate (construction)", "invalid", "instruction spoken",
              f"WER > {gc['max_clip_wer']}"]  # fmt: skip
    return table(header, out)


def outcome_table(rows: list[dict]) -> str:
    names = ("detected", "flat", "wrong_sign", "no_changepoint", "misplaced", "no_step")
    out = []
    for key in dict.fromkeys((r["condition"], r["set"]) for r in rows):
        c = Counter(r["outcome"] for r in rows if (r["condition"], r["set"]) == key)
        out.append([*key, str(sum(c.values())), *[str(c.get(n, 0)) for n in names]])
    return table(["condition", "set", "n", *names], out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--export", default="data/raw_export", help="directory with the node runs/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--from-rows", action="store_true", help="skip the models and score the stored rows")
    args = ap.parse_args()

    cfg = load_config(args.config)
    export = ROOT / args.export
    out = ROOT / "results" / cfg["name"]
    rows_path = out / "dropped_rows.jsonl"
    if args.from_rows:
        rows = read_jsonl(rows_path)
    else:
        manifest = build_manifest(cfg, export)
        write_jsonl(ROOT / "data" / "manifest_dropped.jsonl", manifest)
        rows = measure(cfg, manifest, args.device)
        for r, m in zip(rows, manifest, strict=True):
            r["node_full_text"] = m["node_full_text"]
    add_derived(rows, cfg)  # dev and val rule: boundary by construction, gate reported but not applied
    write_jsonl(rows_path, rows)
    text = (
        "# Conditions dropped at import: exploratory measurement\n\n"
        "Generated by `python scripts/eval_dropped_conditions.py`; do not edit by hand. These rows are not part "
        "of the baseline experiment and of no table in summary.md. Same judge, aligner, ASR, gates and frozen "
        "detector as the main evaluation; each clip is scored against `text`.\n\n## What the conditions are\n\n"
        + describe(rows)
        + "\n\n## Detection per condition, set and seed\n\n"
        + counts_table(rows, cfg["gates"])
        + "\n\nA single-pass clip (`inline_tag`) has one boundary, the forced-aligned switch word, in both columns.\n"
        + "\n## Outcome of every row (boundary by construction)\n\n"
        + outcome_table(rows)
        + "\n"
    )
    (out / "dropped_conditions.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

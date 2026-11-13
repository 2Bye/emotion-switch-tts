"""Exploratory tables over the released evaluation data.

    python -m emoswitch.eda --config configs/baseline.yaml

Reads data/manifest.jsonl, the audio and results/<name>/rows.jsonl; writes results/<name>/eda.md and
fig_control_levels.png. No models are loaded: loudness is measured here, everything else comes from
the per-clip rows. Some no-switch clips occur in two runs byte for byte; every table except the
first counts each distinct audio file once.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from emoswitch.config import ROOT, load_config, load_phrases  # noqa: E402
from emoswitch.evaluate import read_jsonl, run_kind  # noqa: E402
from emoswitch.report import AXIS, COLOR, GRID, INK, INK2, SURFACE, fmt, table  # noqa: E402

CONDITIONS = ("plain", "control", "nl_instruct", "stitch")
LEVELS = ("sad", "plain", "happy", "angry")  # low to high nominal arousal


def loudness(path: Path) -> float:
    """Integrated loudness of the clip as released (LUFS), before any normalisation."""
    import pyloudnorm as pyln
    import soundfile as sf

    y, sr = sf.read(path, dtype="float64")
    return float(pyln.Meter(sr).integrated_loudness(y))


def span(xs: list[float], digits: int = 2) -> str:
    """min / median / max; a half-integer median of integer data keeps its half."""
    med = statistics.median(xs)
    med_digits = 1 if digits == 0 and med != int(med) else digits
    return f"{min(xs):.{digits}f} / {med:.{med_digits}f} / {max(xs):.{digits}f}"


def clips_table(manifest: list[dict], rows: dict) -> str:
    out, runs = [], list(dict.fromkeys(m["run"] for m in manifest))
    for run in runs:
        sub = [m for m in manifest if m["run"] == run]
        counts = Counter(m["condition"] for m in sub)
        seconds = sum(rows[(m["run"], m["id"])]["duration_s"] for m in sub)
        out.append([run, sub[0]["set"], str(sub[0]["seed"]), *[str(counts.get(c, 0)) for c in CONDITIONS],
                    str(len(sub)), f"{seconds:.1f}"])  # fmt: skip
    total = sum(rows[(m["run"], m["id"])]["duration_s"] for m in manifest)
    out.append(["**all**", "", "", *[str(sum(m["condition"] == c for m in manifest)) for c in CONDITIONS],
                str(len(manifest)), f"{total:.1f}"])  # fmt: skip
    return table(["run", "set", "seed", *CONDITIONS, "clips", "audio, s"], out)


def repeats_note(manifest: list[dict]) -> str:
    first_run, repeats = {}, Counter()
    for m in manifest:
        if m["sha256_wav_source"] in first_run:
            repeats[(m["run"], first_run[m["sha256_wav_source"]])] += 1
        else:
            first_run[m["sha256_wav_source"]] = m["run"]
    if not repeats:
        return f"All {len(manifest)} clips are distinct audio."
    listed = "; ".join(f"{n} no-switch clips of {run} repeat {src}" for (run, src), n in repeats.items())
    return f"{len(first_run)} of the {len(manifest)} clips are distinct audio. Byte-identical repeats: {listed}."


def text_table(cfg: dict) -> str:
    out = []
    for set_name in cfg["sets"]:
        phrases = load_phrases(cfg, set_name)
        w1 = [len(p["s1"].split()) for p in phrases]
        w2 = [len(p["s2"].split()) for p in phrases]
        both = [a + b for a, b in zip(w1, w2, strict=True)]
        switch_words = ", ".join(p["switch_word"] for p in phrases)
        out.append([set_name, str(len(phrases)), span(w1, 0), span(w2, 0), span(both, 0), switch_words])
    header = ["set", "phrases", "words in sentence 1 (min / median / max)", "words in sentence 2",
              "words in the phrase", "first word of sentence 2"]  # fmt: skip
    return table(header, out)


def duration_table(rows: list[dict]) -> str:
    out = []
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        frac = [r["step"]["switch_time_s"] / r["duration_s"] for r in sub if r["step"]]
        windows = [len(r["profile_arousal"]) for r in sub]
        out.append([cond, str(len(sub)), span([r["duration_s"] for r in sub]), span(frac), span(windows, 0),
                    str(sum(r["step"] is None for r in sub))])  # fmt: skip
    header = ["condition", "distinct clips", "duration, s (min / median / max)",
              "boundary position, share of the clip", "judge windows per clip",
              "clips without a located boundary"]  # fmt: skip
    return table(header, out)


def level_rows(rows: list[dict], set_name: str, level: str) -> list[dict]:
    """Whole-phrase single-pass renders of one set at one level (an emotion, or `plain`)."""
    return [
        r
        for r in rows
        if r["set"] == set_name
        and run_kind(r) == "inline"
        and (r["condition"] == "plain" if level == "plain" else r["condition"] == "control" and r["emotion"] == level)
    ]


def level_table(rows: list[dict], cfg: dict) -> str:
    """Whole-utterance arousal of the single-emotion renders: the levels a switch has to move between."""
    out = []
    for set_name in cfg["sets"]:
        for level in LEVELS:
            sub = level_rows(rows, set_name, level)
            if sub:
                levels = [float(np.mean(r["profile_arousal"])) for r in sub]
                out.append([set_name, level, str(len(sub)), fmt(float(np.mean(levels))),
                            fmt(float(np.std(levels, ddof=1))) if len(levels) > 1 else "n/a",
                            fmt(float(np.mean([r["profile_sd"] for r in sub]))),
                            f"{np.mean([r['lufs'] for r in sub]):.1f}"])  # fmt: skip
    header = ["set", "rendered as", "clips", "mean arousal", "SD across clips", "SD within a clip (window noise)",
              "loudness as rendered, LUFS"]  # fmt: skip
    return table(header, out)


def loudness_note(rows: list[dict]) -> str:
    sub = [r for r in rows if r["condition"] in ("control", "plain") and run_kind(r) == "inline"]
    lufs = [r["lufs"] for r in sub]
    level = [float(np.mean(r["profile_arousal"])) for r in sub]
    r = float(np.corrcoef(lufs, level)[0, 1])
    return (
        f"Across the {len(sub)} single-emotion and plain renders of the inline runs, loudness as rendered spans "
        f"{min(lufs):.1f} to {max(lufs):.1f} LUFS and correlates with judged arousal at r = {r:.2f}. "
        "Both follow the rendered emotion: the model speaks an aroused emotion louder. "
        "It is not the judge reading level, because the judge rescales every window before scoring."
    )


def wer_table(rows: list[dict]) -> str:
    """Every stretch of words the ASR wrote differently from the target, with the number of clips it occurs in."""
    import jiwer

    from emoswitch.gates import norm_text

    kinds, clips = Counter(), 0
    for r in rows:
        if r["wer"] == 0:
            continue
        clips += 1
        out = jiwer.process_words(norm_text(r["text"]), norm_text(r["hyp"]))
        ref, hyp = out.references[0], out.hypotheses[0]
        changed = [c for c in out.alignments[0] if c.type != "equal"]
        lo, hi = min(c.ref_start_idx for c in changed), max(c.ref_end_idx for c in changed)
        hlo, hhi = min(c.hyp_start_idx for c in changed), max(c.hyp_end_idx for c in changed)
        kinds[(" ".join(ref[lo:hi]) or "(nothing)", " ".join(hyp[hlo:hhi]) or "(nothing)", r["set"])] += 1
    if not kinds:
        return "No clip has a non-zero WER."
    body = [[f"`{a}`", f"`{b}`", set_name, str(n)] for (a, b, set_name), n in kinds.most_common()]
    note = f"\n\n{clips} of the {len(rows)} distinct clips have a non-zero WER; the table lists all of them."
    return table(["target words", "ASR wrote", "set", "distinct clips"], body) + note


def fig_levels(rows: list[dict], cfg: dict, path: Path) -> None:
    """Arousal of every single-emotion render, by commanded emotion: are the levels separable at all?"""
    import matplotlib.pyplot as plt

    sets = list(cfg["sets"])
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.8), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, set_name in zip(axes[0], sets, strict=True):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for x, level in enumerate(LEVELS):
            ys = [float(np.mean(r["profile_arousal"])) for r in level_rows(rows, set_name, level)]
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(ys)), ys, s=44, facecolors=COLOR["null"],
                       edgecolors=SURFACE, linewidths=1.2)  # fmt: skip
            if ys:
                ax.hlines(float(np.mean(ys)), x - 0.28, x + 0.28, color=INK, linewidth=2)
        ax.set_xticks(range(len(LEVELS)), ["no instruction" if lv == "plain" else f"all {lv}" for lv in LEVELS])
        ax.set_title(set_name, color=INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel("mean judge arousal of the clip", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, "One dot per clip (phrase x seed), single-pass renders with one emotion. Bar: mean.",
             color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default=None, help="results directory to read rows from and write to")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / (args.out or f"results/{cfg['name']}")
    manifest = read_jsonl(ROOT / args.manifest)
    rows = read_jsonl(out / "rows.jsonl")
    by_key = {(r["run"], r["id"]): r for r in rows}
    seen, distinct = set(), []
    for m in manifest:
        if m["sha256_wav_source"] not in seen:
            seen.add(m["sha256_wav_source"])
            distinct.append(by_key[(m["run"], m["id"])])
    for r in distinct:
        r["lufs"] = loudness(ROOT / r["audio"])
    rates = Counter(m["sample_rate"] for m in manifest)
    rate_note = "Sample rate: " + ", ".join(f"{sr} Hz ({n} clips)" for sr, n in rates.items()) + "."
    sections = [
        ("Released clips", clips_table(manifest, by_key) + f"\n\n{rate_note} {repeats_note(manifest)}"),
        ("Texts", text_table(cfg)),
        ("Durations and boundary position", duration_table(distinct)),
        ("Arousal levels of the single-emotion renders", level_table(distinct, cfg) + "\n\n" + loudness_note(distinct)),
        ("Clips with a non-zero WER", wer_table(distinct)),
    ]
    text = "# Evaluation data: exploratory tables\n\nGenerated by `python -m emoswitch.eda`; do not edit by hand.\n"
    for title, body in sections:
        text += f"\n## {title}\n\n{body}\n"
    (out / "eda.md").write_text(text, encoding="utf-8")
    fig_levels(distinct, cfg, out / "fig_control_levels.png")
    print(f"[eda] wrote eda.md and fig_control_levels.png to {out}")


if __name__ == "__main__":
    main()

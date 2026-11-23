"""Tables, figures and error examples from the scored rows.

    python -m emoswitch.report --config configs/baseline.yaml

Reads results/<name>/{rows.jsonl,metrics.json}; writes summary.md, fig_steps.png and
fig_profiles.png next to them. No models are loaded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # same pixels on every platform, no display needed

import numpy as np  # noqa: E402

from emoswitch.config import ROOT, load_config  # noqa: E402
from emoswitch.evaluate import (  # noqa: E402
    SWITCH_CONDITIONS,
    analysis_group,
    primary_step,
    read_jsonl,
)
from emoswitch.stimuli import NULL_CONDITIONS, emotions  # noqa: E402

LABEL = {"nl_instruct": "baseline: one pass, instruction", "stitch": "reference: stitched segments"}
# conditions as counted once per clip (`evaluate.analysis_group`)
GROUP_LABEL = {"nl_instruct": "baseline", "stitch": "reference", "control": "control (inline runs)",
               "plain": "plain (inline runs)", "stitch_null": "stitch_null"}  # fmt: skip
COLOR = {"nl_instruct": "#2a78d6", "stitch": "#eb6834", "null": "#898781", "stitch_null": "#b8a46a"}
SURFACE, GRID, AXIS, INK, INK2 = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
OUTCOMES = ("detected", "invalid", "flat", "wrong_sign", "no_changepoint", "misplaced", "no_step")
RULES = (("construction", "stitched boundary by construction"), ("forced_alignment", "by forced alignment"))


def fmt(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def ci(pair) -> str:
    return f"[{pair[0]:.2f}, {pair[1]:.2f}]"


def frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def null_label(name: str) -> str:
    """`control/inline` -> `control, inline runs`; the stitch_null rows have a run kind of their own."""
    cond, kind = name.split("/")
    return cond if cond == "stitch_null" else f"{cond}, {kind} runs"


def rule_label(rule: dict) -> str:
    boundary = "construction" if rule["boundary"] == "construction" else "forced alignment"
    return boundary + (" + validity" if rule["validity_gate"] else "")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def set_label(metrics: dict, set_name: str) -> str:
    return f"{set_name} ({metrics['set_roles'][set_name]})"


def main_table(metrics: dict) -> str:
    rows = []
    for cond, system in metrics["systems"].items():
        for set_name, m in system["by_set"].items():
            rows.append(
                [
                    LABEL[cond],
                    set_label(metrics, set_name),
                    str(m["n_phrases"]),
                    str(m["n_seeds"]),
                    f"{m['detected']}/{m['n']}",
                    f"**{fmt(m['det_rate'])}**",
                    ci(m["det_ci_wilson"]),
                    ci(m["det_ci_phrase_bootstrap"]),
                ]
            )
    header = ["system", "set", "phrases", "seeds", "detected", "det", "95% CI (Wilson, rows)",
              "95% CI (bootstrap over phrases)"]  # fmt: skip
    return table(header, rows)


def step_table(metrics: dict) -> str:
    rows = []
    for cond, system in metrics["systems"].items():
        for set_name, m in system["by_set"].items():
            rows.append(
                [
                    LABEL[cond],
                    set_label(metrics, set_name),
                    fmt(m["abs_delta_mean"]),
                    fmt(m["abs_delta_max"]),
                    fmt(m["commanded_step_mean"]),
                    fmt(m["ideal_step_mean"]),
                    fmt(m["realised_share"], 2),
                    fmt(m["sign_ok_rate"], 2),
                    fmt(m["changepoint_within_tol_rate"], 2),
                ]
            )
    header = ["system", "set", "mean abs step", "max abs step", "commanded step", "ideal step", "realised share",
              "sign ok", "change-point at boundary"]  # fmt: skip
    return table(header, rows)


def seed_table(metrics: dict) -> str:
    rows = []
    for cond, system in metrics["systems"].items():
        for set_name, m in system["by_set"].items():
            per_seed = [
                f"s{key.rsplit('/s', 1)[1]}: {v['detected']}/{v['n']}"
                for key, v in system["by_set_seed"].items()
                if key.startswith(f"{set_name}/")
            ]
            sd = "n/a (one seed)" if m["det_rate_seed_sd"] is None else fmt(m["det_rate_seed_sd"])
            rows.append([LABEL[cond], set_name, ", ".join(per_seed), fmt(m["det_rate_seed_mean"]), sd])
    return table(["system", "set", "detected per seed", "mean of seed rates", "SD over seeds"], rows)


def rule_table(metrics: dict) -> str:
    """docs/prereg_test.md, rules 1 and 2: every boundary rule with and without the validity gate."""
    rows = []
    for cond, system in metrics["systems"].items():
        for set_name, m in system["by_set"].items():
            d = m["detected_by_rule"]
            cells = [frac(d[f"{b}{g}"], m["n"]) for b, _ in RULES for g in ("", "+valid")]
            rows.append([LABEL[cond], set_label(metrics, set_name), *cells, str(m["n_invalid"]),
                         rule_label(metrics["primary_rule"][set_name])])  # fmt: skip
    header = ["system", "set", "stitched boundary by construction", "+ validity gate", "by forced alignment",
              "+ validity gate", "invalid clips", "primary rule"]  # fmt: skip
    note = ("\n\nA single-pass clip has one boundary, the forced-aligned switch word, under both rules. "
            "With the validity gate an invalid clip counts as not detected.")  # fmt: skip
    return table(header, rows) + note


def boundary_gap_note(rows: list[dict], metrics: dict) -> str:
    """How far the two boundaries of a stitched clip are apart, per set."""
    parts = []
    for set_name in metrics["set_roles"]:
        gaps = [
            r["step_aligned"]["switch_time_s"] - r["step"]["switch_time_s"]
            for r in rows
            if r["set"] == set_name and r["switch_time_source"] == "construction" and r["step"] and r["step_aligned"]
        ]
        if gaps:
            parts.append(
                f"{set_label(metrics, set_name)}: {min(gaps):.2f} / {float(np.median(gaps)):.2f} / {max(gaps):.2f} s "
                f"over {len(gaps)} clips"
            )
    if not parts:
        return ""
    return (
        "\n\nForced-aligned boundary minus the boundary by construction in stitched clips (min / median / max): "
        + "; ".join(parts)
        + "."
    )


def agreement_table(metrics: dict) -> str:
    """Does the primary decision on a (phrase, commanded pair) repeat across seeds?"""
    rows = []
    agreement = metrics["seed_agreement"]
    for key in [f"{g}/{s}" for s in metrics["set_roles"] for g in GROUP_LABEL if f"{g}/{s}" in agreement]:
        a, (group, set_name) = agreement[key], key.split("/")
        pairs = ", ".join(f"{name}: {k}/{n}" for name, (k, n) in a["seed_pairs"].items()) or "n/a"
        rows.append([GROUP_LABEL[group], set_label(metrics, set_name), str(a["n_cells"]),
                     str(a["n_cells_multi_seed"]), frac(a["unanimous"], a["n_cells_multi_seed"]),
                     str(a["unanimous_positive"]), pairs])  # fmt: skip
    header = ["condition", "set", "cells (phrase x pair)", "cells with 2+ seeds", "same decision on every seed",
              "of which positive on every seed", "same decision, per pair of seeds"]  # fmt: skip
    note = ("\n\nThe decision is `detected` for switch rows and `would pass the detector` for no-switch rows, "
            "under the primary rule of the set.")  # fmt: skip
    return table(header, rows) + note


def gates_table(metrics: dict) -> str:
    rows = []
    for cond, system in metrics["systems"].items():
        for set_name, m in system["by_set"].items():
            g = m["gates"]
            failed = ", ".join(k for k, v in g.items() if k != "passed" and not v)
            rows.append(
                [
                    LABEL[cond],
                    set_label(metrics, set_name),
                    str(m["n"]),
                    fmt(m["wer_mean"]),
                    fmt(m["wer_median"]),
                    fmt(m["leak_rate"], 2),
                    str(m["looped"]),
                    "pass" if g["passed"] else f"FAIL: {failed}",
                ]
            )
    return table(["system", "set", "n", "WER mean", "WER median", "leak rate", "looped", "gate"], rows)


def null_table(metrics: dict) -> str:
    rows = []
    for name, null in metrics["null"].items():
        for set_name, m in null["by_set"].items():
            by_b = m["exceed_by_boundary"]
            rows.append(
                [
                    null_label(name),
                    set_label(metrics, set_name),
                    str(m["n"]),
                    fmt(m["abs_delta_mean"]),
                    fmt(m["abs_delta_p95"]),
                    fmt(m["abs_delta_max"]),
                    f"{m['exceed']}/{m['n']} = {fmt(m['exceed_rate'])}",
                    ci(m["exceed_ci_wilson"]),
                    frac(by_b["construction"], m["n"]),
                    frac(by_b["forced_alignment"], m["n"]),
                    fmt(m["profile_sd_mean"]),
                ]
            )
    header = ["no-switch rows", "set", "n", "mean abs step", "p95 abs step", "max abs step",
              "would pass the detector (either sign), primary rule", "95% CI (Wilson)",
              "pass, stitched boundary by construction", "pass, by forced alignment", "profile SD"]  # fmt: skip
    note = ("\n\nOnly `stitch_null` rows have two boundaries; for the single-pass rows both columns are the same "
            "count. The validity gate is not applied to no-switch rows: it could only lower their rate.")  # fmt: skip
    return table(header, rows) + note


def sensitivity_table(metrics: dict, cfg: dict) -> str:
    groups = [g for g in GROUP_LABEL if any(k.startswith(f"{g}/") for v in metrics["sensitivity"] for k in v["counts"])]
    dc = cfg["detector"]
    rows = []
    for set_name in metrics["set_roles"]:
        for v in metrics["sensitivity"]:
            cells = [v["counts"].get(f"{g}/{set_name}") for g in groups]
            if not any(cells):
                continue
            tol = v["max_changepoint_err_s"]
            frozen = v["min_abs_delta"] == dc["min_abs_delta"] and tol == dc["max_changepoint_err_s"]
            label = "weighted estimator" if v["weighted_split"] else ("frozen" if frozen else "")
            rows.append([set_label(metrics, set_name), f"{v['min_abs_delta']:.2f}", "none" if tol is None else f"{tol}",
                         label] + [frac(*c) if c else "n/a" for c in cells])  # fmt: skip
    header = ["set", "threshold", "change-point tolerance, s", "variant"] + [GROUP_LABEL[g] for g in groups]
    note = (
        "\n\nSwitch rows: detected; no-switch rows: would pass the detector. Primary rule of each set. "
        "`weighted estimator`: the change-point split scaled by sqrt(n1*n2/n), at the frozen constants."
    )
    return table(header, rows) + note


def validity_table(rows: list[dict], cfg: dict) -> str:
    """docs/prereg_test.md, rule 2, per condition and set; each clip counted once."""
    gc, out = cfg["gates"], []
    for group in GROUP_LABEL:
        for set_name in dict.fromkeys(r["set"] for r in rows):
            sub = [r for r in rows if analysis_group(r) == group and r["set"] == set_name]
            if sub:
                out.append([GROUP_LABEL[group], set_name, str(len(sub)), str(sum(not r["valid"] for r in sub)),
                            str(sum(r["wer"] > gc["max_clip_wer"] for r in sub)), str(sum(r["leak"] for r in sub)),
                            str(sum(r["looped"] for r in sub))])  # fmt: skip
    header = ["condition", "set", "clips", "invalid", f"WER > {gc['max_clip_wer']}", "instruction spoken",
              f"longer than {gc['loop_threshold_s']:g} s"]  # fmt: skip
    return table(header, out)


def side_means(row: dict, t: float) -> tuple[float | None, float | None]:
    """Mean arousal of the windows centred before and at or after `t`."""
    a, c = np.asarray(row["profile_arousal"], dtype=float), np.asarray(row["profile_centres_s"], dtype=float)
    before, after = a[c < t], a[c >= t]
    return (float(before.mean()) if len(before) else None, float(after.mean()) if len(after) else None)


def length_table(rows: list[dict], cfg: dict) -> str:
    """The baseline speaks a two-sentence text, each reference segment a one-sentence text. Levels of the
    same emotion in both: whole-phrase controls against the segments of the stitched clips."""
    out = []
    for set_name in dict.fromkeys(r["set"] for r in rows):
        for e in emotions(cfg):
            ctrl = [r for r in rows if r["set"] == set_name and analysis_group(r) == "control" and r["emotion"] == e]
            stitched = [r for r in rows if r["set"] == set_name and r["condition"] == "stitch" and r["step"]]
            seg1 = [side_means(r, r["step"]["switch_time_s"])[0] for r in stitched if r["emotion"] == e]
            seg2 = [side_means(r, r["step"]["switch_time_s"])[1] for r in stitched if r["emotion2"] == e]
            if not ctrl and not seg1 and not seg2:
                continue
            c_sides = [side_means(r, r["step"]["switch_time_s"]) for r in ctrl if r["step"]]

            def mean(xs: list) -> str:
                xs = [x for x in xs if x is not None]
                return fmt(float(np.mean(xs))) if xs else "n/a"

            out.append([set_name, e, str(len(ctrl)), mean([float(np.mean(r["profile_arousal"])) for r in ctrl]),
                        mean([s[0] for s in c_sides]), mean([s[1] for s in c_sides]),
                        str(len(seg1)), mean(seg1), str(len(seg2)), mean(seg2)])  # fmt: skip
    header = ["set", "emotion", "control clips", "control, whole clip", "control, sentence 1", "control, sentence 2",
              "stitched clips with it as segment 1", "segment 1", "with it as segment 2", "segment 2"]  # fmt: skip
    note = ("\n\nMean judge arousal. Control: whole phrase in one emotion, one pass (\"Say this ...\", inline runs); "
            "its sentences split at the forced-aligned switch word. Segments: the windows on each side of the "
            "construction boundary of the `stitch` clips.")  # fmt: skip
    return table(header, out) + note


def outcome_table(rows: list[dict]) -> str:
    out = []
    for cond in SWITCH_CONDITIONS:
        for set_name in dict.fromkeys(r["set"] for r in rows):
            sub = [r for r in rows if r["condition"] == cond and r["set"] == set_name]
            if sub:
                counts = Counter(r["outcome"] for r in sub)
                out.append([LABEL[cond], set_name, str(len(sub))] + [str(counts.get(o, 0)) for o in OUTCOMES])
    return table(["system", "set", "n", *OUTCOMES], out)


def pair_table(metrics: dict) -> str:
    rows = []
    for cond, system in metrics["systems"].items():
        for key, m in system["by_pair"].items():
            set_name, pair = key.split("/")
            nearest = ", ".join(f"{name}: {n}" for name, n in m["nearest_level_counts"].items())
            rows.append([LABEL[cond], set_label(metrics, set_name), pair.replace(">", " → "),
                         f"{m['detected']}/{m['n']}", fmt(m["commanded_step_mean"]), fmt(m["ideal_step_mean"]),
                         nearest])  # fmt: skip
    header = ["system", "set", "commanded switch", "detected", "commanded step", "ideal step",
              "whole-clip arousal is nearest to the render of"]  # fmt: skip
    return table(header, rows)


def paired_table(metrics: dict) -> str:
    paired = metrics.get("paired_reference_vs_baseline")
    if not paired:
        return "The two systems share no rows."
    rows = []
    for set_name in metrics["set_roles"]:
        p = paired.get(set_name)
        if p:
            rows.append([set_label(metrics, set_name), str(p["n_pairs"]), str(p["reference_only"]),
                         str(p["baseline_only"]), f"{p['mcnemar_p']:.3g}", str(p["n_phrases"]),
                         str(p["phrases_reference_ahead"]), str(p["phrases_baseline_ahead"]),
                         f"{p['phrase_sign_test_p']:.3g}"])  # fmt: skip
    header = ["rows of", "paired rows", "detected by reference only", "by baseline only", "McNemar p (rows)",
              "phrases", "phrases where reference is ahead", "where baseline is ahead",
              "sign test p (phrases)"]  # fmt: skip
    note = ("\n\nPrimary rule of each set. A phrase is ahead for the system with the higher detection rate over "
            "all its rows of that phrase (pairs x seeds). Sets are not pooled: the dev set was used to calibrate "
            "the detector.")  # fmt: skip
    return table(header, rows) + note


def phrase_table(metrics: dict) -> str:
    """Per-phrase detection rate of each system, the unit of the sign test."""
    systems = metrics["systems"]
    rows = []
    for set_name in metrics["set_roles"]:
        per = {c: s["by_set"][set_name]["per_phrase"] for c, s in systems.items() if set_name in s["by_set"]}
        for key in sorted({k for d in per.values() for k in d}):
            cells = [f"{per[c][key][0]}/{per[c][key][1]}" if key in per.get(c, {}) else "n/a" for c in systems]
            rows.append([set_label(metrics, set_name), key.split("/", 1)[1], *cells])
    boot = [f"{LABEL[c].split(':')[0]} {s}: {ci(m['det_ci_phrase_bootstrap'])}" for c, sy in systems.items()
            for s, m in sy["by_set"].items()]  # fmt: skip
    note = "\n\n95% bootstrap interval over phrases of the detection rate: " + "; ".join(boot) + "."
    return table(["set", "phrase", *[LABEL[c].split(":")[0] for c in systems]], rows) + (note if boot else "")


def pick_examples(rows: list[dict], metrics: dict) -> list[dict]:
    """Three baseline misses (largest shortfall, distinct switches) and one reference hit.
    Test rows first, then validation rows; the choice depends on the rows only, so it is reproducible."""
    preference = {"test": 0, "val": 1, "dev": 2}
    order = {s: preference[role] for s, role in metrics["set_roles"].items()}
    usable = [r for r in rows if primary_step(r) and r.get("ideal_step") is not None]
    misses = [r for r in usable if r["condition"] == "nl_instruct" and not r["detected"]]
    misses.sort(key=lambda r: (order[r["set"]], r["seed"], -(r["ideal_step"] - r["commanded_step"]), r["id"]))
    picked, seen = [], set()
    for r in misses:
        if (r["emotion"], r["emotion2"]) not in seen:
            picked.append(r)
            seen.add((r["emotion"], r["emotion2"]))
        if len(picked) == 3:
            break
    hits = [r for r in usable if r["condition"] == "stitch" and r["detected"]]
    hits.sort(key=lambda r: (order[r["set"]], r["seed"], r["id"]))
    return picked + hits[:1]


def example_table(examples: list[dict]) -> str:
    rows = []
    for r in examples:
        rows.append(
            [
                f"[{r['run']}/{r['id']}](../../{r['audio']})",
                LABEL[r["condition"]].split(":")[0],
                f"{r['emotion']} → {r['emotion2']}",
                r["outcome"],
                fmt(r["commanded_step"]),
                fmt(r["ideal_step"]),
                fmt(primary_step(r)["changepoint_err_s"], 2),
                r["nearest_level"] or "n/a",
                fmt(r["wer"], 2),
            ]
        )
    header = ["clip", "system", "commanded", "outcome", "commanded step", "ideal step", "change-point error, s",
              "whole-clip arousal nearest to", "WER"]  # fmt: skip
    return table(header, rows)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_steps(rows: list[dict], metrics: dict, cfg: dict, path: Path) -> None:
    """Every row as a dot: the step at the boundary, by system and set."""
    import matplotlib.pyplot as plt

    thr = cfg["detector"]["min_abs_delta"]
    sets = [s for s in metrics["set_roles"] if any(r["set"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    groups = [("null", "no switch\nasked"), ("nl_instruct", "baseline"), ("stitch", "reference")]
    if any(r["condition"] == "stitch_null" for r in rows):
        groups.insert(1, ("stitch_null", "stitched,\none emotion"))
    for ax, set_name in zip(axes[0], sets, strict=True):
        _style(ax)
        ax.axhspan(-thr, thr, color=GRID, alpha=0.6, linewidth=0)
        for x, (key, _) in enumerate(groups):
            in_set = [r for r in rows if r["set"] == set_name and primary_step(r)]
            if key in ("null", "stitch_null"):  # inline runs only: the stitch runs repeat the same clips
                sub = [r for r in in_set if r["condition"] in NULL_CONDITIONS and analysis_group(r)]
                sub = [r for r in sub if (r["condition"] == "stitch_null") == (key == "stitch_null")]
                ys, filled = [primary_step(r)["delta"] for r in sub], [False] * len(sub)
            else:
                sub = [r for r in in_set if r["condition"] == key]
                ys, filled = [r["commanded_step"] for r in sub], [r["detected"] for r in sub]
            xs = x + rng.uniform(-0.22, 0.22, size=len(ys))
            for xi, yi, f in zip(xs, ys, filled, strict=True):
                ax.scatter(xi, yi, s=42, facecolors=COLOR[key] if f else SURFACE, edgecolors=COLOR[key], linewidths=1.6)
        ax.set_xticks(range(len(groups)), [g[1] for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(set_label(metrics, set_name), color=INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel("arousal step at the sentence boundary", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, f"Filled dot: counted as a switch. Grey band: below the detector threshold {thr}.\n"
             "Switch rows are shown in the commanded direction, no-switch rows with their raw sign.",
             color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_profiles(rows: list[dict], examples: list[dict], path: Path) -> None:
    """Arousal over time for the example rows, against the two single-emotion renders of the same phrase."""
    import matplotlib.pyplot as plt

    controls = {(r["run"], r["phrase_id"], r["emotion"]): r for r in rows if r["condition"] == "control"}
    n_rows = max(1, (len(examples) + 1) // 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.5, 3.7 * n_rows), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(examples) :]:
        ax.set_visible(False)
    for ax, r in zip(axes.flat, examples, strict=False):
        _style(ax)
        # dashed: the emotion commanded for sentence 1; dotted: the emotion commanded for sentence 2
        for emotion, dash, role in ((r["emotion"], (4, 2), "1st"), (r["emotion2"], (1, 2), "2nd")):
            c = controls.get((r["run"], r["phrase_id"], emotion))
            if c:
                ax.plot(c["profile_centres_s"], c["profile_arousal"], color=COLOR["null"], linewidth=1.4, dashes=dash)
                ax.annotate(f"all {emotion} ({role})", (c["profile_centres_s"][-1], c["profile_arousal"][-1]),
                            xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")  # fmt: skip
        ax.plot(r["profile_centres_s"], r["profile_arousal"], color=COLOR[r["condition"]], linewidth=2.2)
        ax.axvline(primary_step(r)["switch_time_s"], color=INK2, linewidth=1, linestyle=":")
        ax.set_title(f"{LABEL[r['condition']].split(':')[0]}: {r['emotion']} → {r['emotion2']}\n"
                     f"{r['outcome']}, step {r['commanded_step']:+.2f} of {r['ideal_step']:+.2f}",
                     color=INK, fontsize=9.5, loc="left")  # fmt: skip
        ax.set_xlabel("time, s", color=INK2, fontsize=9)
        ax.margins(x=0.22)
    for row_axes in axes:
        row_axes[0].set_ylabel("judge arousal (1 s window)", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, "Coloured line: the clip. Grey lines: the same phrase rendered in one emotion throughout "
             "(dashed: the emotion commanded\nfor sentence 1, dotted: for sentence 2). "
             "Vertical line: the commanded boundary.", color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / (args.out or f"results/{cfg['name']}")
    rows = read_jsonl(out / "rows.jsonl")
    text = "# Baseline results\n\nGenerated by `python -m emoswitch.report`; do not edit by hand.\n"
    if not rows:
        (out / "summary.md").write_text(text + "\nNo measured rows.\n", encoding="utf-8")
        print(f"[report] {out / 'rows.jsonl'} is empty; wrote an empty summary.md")
        return
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    examples = pick_examples(rows, metrics)
    sections = [
        ("Main result", main_table(metrics)),
        (
            "Detection under each boundary rule and validity gate",
            rule_table(metrics) + boundary_gap_note(rows, metrics),
        ),
        ("Seeds", seed_table(metrics)),
        ("Agreement between seeds", agreement_table(metrics)),
        ("Step size", step_table(metrics)),
        ("No-switch rows: what the detector reads when nothing was asked", null_table(metrics)),
        ("Paired comparison", paired_table(metrics)),
        ("Detection per phrase", phrase_table(metrics)),
        ("Sensitivity to the detector constants", sensitivity_table(metrics, cfg)),
        ("Speech gates", gates_table(metrics)),
        ("Clip validity", validity_table(rows, cfg)),
        ("Outcome of every switch row", outcome_table(rows)),
        ("By commanded switch", pair_table(metrics)),
        ("Arousal level by text length", length_table(rows, cfg)),
        ("Examples", example_table(examples) if examples else "No example rows."),
    ]
    for title, body in sections:
        text += f"\n## {title}\n\n{body}\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    fig_steps(rows, metrics, cfg, out / "fig_steps.png")
    if examples:
        fig_profiles(rows, examples, out / "fig_profiles.png")
    print(f"[report] wrote summary.md, fig_steps.png, fig_profiles.png to {out}")


if __name__ == "__main__":
    main()

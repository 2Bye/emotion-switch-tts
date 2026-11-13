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
from emoswitch.evaluate import SWITCH_CONDITIONS, read_jsonl, run_kind  # noqa: E402
from emoswitch.stimuli import NULL_CONDITIONS  # noqa: E402

LABEL = {"nl_instruct": "baseline: one pass, instruction", "stitch": "reference: stitched segments"}
COLOR = {"nl_instruct": "#2a78d6", "stitch": "#eb6834", "null": "#898781"}
SURFACE, GRID, AXIS, INK, INK2 = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
OUTCOMES = ("detected", "flat", "wrong_sign", "misplaced", "no_step")


def fmt(x, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def ci(pair) -> str:
    return f"[{pair[0]:.2f}, {pair[1]:.2f}]"


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
    for kind, null in metrics["null"].items():
        for set_name, m in null["by_set"].items():
            rows.append(
                [
                    f"{kind} runs",
                    set_label(metrics, set_name),
                    str(m["n"]),
                    fmt(m["abs_delta_mean"]),
                    fmt(m["abs_delta_p95"]),
                    fmt(m["abs_delta_max"]),
                    f"{m['exceed']}/{m['n']} = {fmt(m['exceed_rate'])}",
                    ci(m["exceed_ci_wilson"]),
                    fmt(m["profile_sd_mean"]),
                ]
            )
    header = ["no-switch rows of", "set", "n", "mean abs step", "p95 abs step", "max abs step",
              "would pass the detector (either sign)", "95% CI (Wilson)", "profile SD"]  # fmt: skip
    return table(header, rows)


def sensitivity_table(metrics: dict) -> str:
    cells = [f"{group}/{s}" for group in ("nl_instruct", "stitch", "null") for s in metrics["set_roles"]]
    cells = [c for c in cells if any(c in v["counts"] for v in metrics["sensitivity"])]
    names = {"nl_instruct": "baseline", "stitch": "reference", "null": "no-switch rows"}
    header = ["detector variant"] + [f"{names[c.split('/')[0]]}, {c.split('/')[1]}" for c in cells]
    rows = []
    for v in metrics["sensitivity"]:
        rows.append([v["variant"]] + [f"{v['counts'][c][0]}/{v['counts'][c][1]}" if c in v["counts"] else "n/a"
                                      for c in cells])  # fmt: skip
    return table(header, rows)


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
        for pair, m in system["by_pair"].items():
            nearest = ", ".join(f"{name}: {n}" for name, n in m["nearest_level_counts"].items())
            rows.append([LABEL[cond], pair.replace(">", " → "), f"{m['detected']}/{m['n']}",
                         fmt(m["commanded_step_mean"]), fmt(m["ideal_step_mean"]), nearest])  # fmt: skip
    header = ["system", "commanded switch", "detected", "commanded step", "ideal step",
              "whole-clip arousal is nearest to the render of"]  # fmt: skip
    return table(header, rows)


def paired_table(metrics: dict) -> str:
    paired = metrics.get("paired_reference_vs_baseline")
    if not paired:
        return "The two systems share no rows."
    rows = []
    for scope, p in paired.items():
        label = "all sets" if scope == "all_sets" else set_label(metrics, scope)
        rows.append([label, str(p["n_pairs"]), str(p["reference_only"]), str(p["baseline_only"]),
                     f"{p['mcnemar_p']:.2g}", str(p["n_phrases"]), str(p["phrases_reference_ahead"]),
                     str(p["phrases_baseline_ahead"]), f"{p['phrase_sign_test_p']:.2g}"])  # fmt: skip
    header = ["rows of", "paired rows", "detected by reference only", "by baseline only", "McNemar p (rows)",
              "phrases", "phrases where reference is ahead", "where baseline is ahead",
              "sign test p (phrases)"]  # fmt: skip
    return table(header, rows)


def pick_examples(rows: list[dict], cfg: dict) -> list[dict]:
    """Three typical baseline misses (largest shortfall, distinct switches) and one reference hit.
    Validation rows first; the choice depends on the rows only, so it is reproducible."""
    order = {s: -i for i, s in enumerate(cfg["sets"])}  # the last set of the config is the validation set
    usable = [r for r in rows if r["step"] and r.get("ideal_step") is not None]
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
                fmt(r["step"]["changepoint_err_s"], 2),
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
    sets = [s for s in cfg["sets"] if any(r["set"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    groups = [("null", "no switch asked"), ("nl_instruct", "baseline"), ("stitch", "reference")]
    for ax, set_name in zip(axes[0], sets, strict=True):
        _style(ax)
        ax.axhspan(-thr, thr, color=GRID, alpha=0.6, linewidth=0)
        for x, (key, _) in enumerate(groups):
            in_set = [r for r in rows if r["set"] == set_name and r["step"]]
            if key == "null":  # inline runs only: the stitch runs repeat the same clips
                sub = [r for r in in_set if r["condition"] in NULL_CONDITIONS and run_kind(r) == "inline"]
                ys, filled = [r["step"]["delta"] for r in sub], [False] * len(sub)
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
        ax.axvline(r["step"]["switch_time_s"], color=INK2, linewidth=1, linestyle=":")
        ax.set_title(f"{LABEL[r['condition']].split(':')[0]}: {r['emotion']} → {r['emotion2']}\n"
                     f"{r['outcome']}, step {r['commanded_step']:+.2f} of {r['ideal_step']:+.2f}",
                     color=INK, fontsize=9.5, loc="left")  # fmt: skip
        ax.set_xlabel("time, s", color=INK2, fontsize=9)
        ax.margins(x=0.22)
    for row_axes in axes:
        row_axes[0].set_ylabel("judge arousal (1 s window)", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, "Coloured line: the clip. Grey lines: the same phrase rendered in one emotion throughout "
             "(dashed: the emotion commanded for sentence 1, dotted: for sentence 2). "
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
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    examples = pick_examples(rows, cfg)
    sections = [
        ("Main result", main_table(metrics)),
        ("Seeds", seed_table(metrics)),
        ("Step size", step_table(metrics)),
        ("No-switch rows: what the detector reads when nothing was asked", null_table(metrics)),
        ("Paired comparison", paired_table(metrics)),
        ("Sensitivity to the detector constants", sensitivity_table(metrics)),
        ("Speech gates", gates_table(metrics)),
        ("Outcome of every switch row", outcome_table(rows)),
        ("By commanded switch (all sets and seeds)", pair_table(metrics)),
        ("Examples", example_table(examples) if examples else "No example rows."),
    ]
    text = "# Baseline results\n\nGenerated by `python -m emoswitch.report`; do not edit by hand.\n"
    for title, body in sections:
        text += f"\n## {title}\n\n{body}\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    fig_steps(rows, metrics, cfg, out / "fig_steps.png")
    if examples:
        fig_profiles(rows, examples, out / "fig_profiles.png")
    print(f"[report] wrote summary.md, fig_steps.png, fig_profiles.png to {out}")


if __name__ == "__main__":
    main()

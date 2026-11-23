"""Experiment table, ablation contrasts and figures of the in-pass systems.

    python -m emoswitch.experiments --config configs/inpass.yaml

Reads the scored rows of this experiment (results/<name>/rows.jsonl) and of the baseline experiment
(results/baseline/rows.jsonl: the baseline, the reference and their no-switch rows). Both were measured
with the same judge, aligner, ASR and detector settings; the script refuses to run otherwise. Writes into
results/<name>/: experiments.csv (one line per system, set and seed), predictions.csv (one line per
clip), experiments.json, experiments.md (tables) and figures. No models are loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # same pixels on every platform, no display needed

import numpy as np  # noqa: E402

from emoswitch import stats  # noqa: E402
from emoswitch.config import ROOT, load_config, set_role  # noqa: E402
from emoswitch.evaluate import analysis_group, primary_step, read_jsonl  # noqa: E402
from emoswitch.report import AXIS, GRID, INK, INK2, OUTCOMES, SURFACE, ci, fmt, frac, table  # noqa: E402
from emoswitch.stimuli import NULL_CONDITIONS  # noqa: E402

COLOR = {"nl_instruct": "#2a78d6", "stitch": "#eb6834", "swap": "#898781", "swap_anchor": "#3f9b6c",
         "swap_steer": "#b8a46a", "swap_anchor_steer": "#7a4fb5"}  # fmt: skip
BASELINE_DIR = "results/baseline"


def pair_key(r: dict) -> tuple:
    """Rows of two systems are paired when they share the seed, the phrase and the commanded switch."""
    return r["seed"], r["phrase_id"], r["emotion"], r["emotion2"]


def load_rows(cfg: dict, out) -> list[dict]:
    """Scored rows of this experiment plus the baseline experiment's rows of the same sets."""
    fingerprints = {
        d: json.loads((ROOT / d / "measure_meta.json").read_text(encoding="utf-8"))["measure_fingerprint"]
        for d in (BASELINE_DIR, out.relative_to(ROOT))
    }
    if len(set(fingerprints.values())) != 1:
        raise SystemExit(f"[experiments] the two experiments were measured with different settings: {fingerprints}")
    own = read_jsonl(out / "rows.jsonl")
    sets = {r["set"] for r in own}
    return own + [r for r in read_jsonl(ROOT / BASELINE_DIR / "rows.jsonl") if r["set"] in sets]


def systems_of(cfg: dict, rows: list[dict]) -> list[str]:
    present = {r["condition"] for r in rows}
    return [c for c, spec in cfg["systems"].items() if "role" in spec and c in present]


def summary(rows: list[dict], cfg: dict) -> dict:
    """Counts of one group of switch rows of one system."""
    sc = cfg["stats"]
    n, k = len(rows), sum(r["detected"] for r in rows)
    steps = [r["commanded_step"] for r in rows if r["commanded_step"] is not None]
    return {
        "n": n,
        "detected": k,
        "det": k / n,
        "ci_wilson": stats.wilson_ci(k, n, sc["ci"]),
        "ci_phrases": stats.cluster_bootstrap_ci(
            [float(r["detected"]) for r in rows], [r["phrase_id"] for r in rows], sc["n_boot"], sc["seed"], sc["ci"]
        ),
        "invalid": sum(not r["valid"] for r in rows),
        "wer_mean": float(np.mean([r["wer"] for r in rows])),
        "commanded_step_mean": float(np.mean(steps)) if steps else None,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }


def experiments(rows: list[dict], cfg: dict) -> list[dict]:
    """One record per system, set and seed, and one per system and set over all seeds (seed None)."""
    out = []
    order = {s: i for i, s in enumerate(cfg["sets"])}
    for cond in systems_of(cfg, rows):
        spec = cfg["systems"][cond]
        sub = [r for r in rows if r["condition"] == cond]
        for s in sorted({r["set"] for r in sub}, key=order.get):
            in_set = [r for r in sub if r["set"] == s]
            for seed in [*sorted({r["seed"] for r in in_set}), None]:
                g = [r for r in in_set if seed is None or r["seed"] == seed]
                out.append(
                    {
                        "system": cond,
                        "role": spec["role"],
                        "change": spec.get("change", ""),
                        "hypothesis": spec.get("hypothesis", ""),
                        "set": s,
                        "set_role": set_role(cfg, s),
                        "seed": seed,
                        "n_seeds": len({r["seed"] for r in g}),
                        "n_phrases": len({r["phrase_id"] for r in g}),
                        **summary(g, cfg),
                    }
                )
    return out


def contrast(rows: list[dict], cfg: dict, a: str, b: str, set_name: str) -> dict | None:
    """System `a` against system `b` on the rows both have. Row level: exact McNemar test. Phrase level,
    the unit of inference: sign test on the per-phrase difference of detection rates and a bootstrap
    interval of the mean difference over phrases."""
    sc = cfg["stats"]
    rows_a = {pair_key(r): r for r in rows if r["condition"] == a and r["set"] == set_name}
    rows_b = {pair_key(r): r for r in rows if r["condition"] == b and r["set"] == set_name}
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        return None
    da, db = [rows_a[k]["detected"] for k in keys], [rows_b[k]["detected"] for k in keys]
    per_phrase = defaultdict(lambda: [0, 0, 0])
    for k, x, y in zip(keys, da, db, strict=True):
        cell = per_phrase[k[1]]
        cell[0] += x
        cell[1] += y
        cell[2] += 1
    diffs = np.array([(x - y) / n for x, y, n in per_phrase.values()])
    rng = np.random.default_rng(sc["seed"])
    boot = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(sc["n_boot"])]
    lo, hi = np.quantile(boot, [(1 - sc["ci"]) / 2, 1 - (1 - sc["ci"]) / 2])
    row_level = stats.mcnemar_exact(da, db)
    a_wins, b_wins = int((diffs > 0).sum()), int((diffs < 0).sum())
    return {
        "a": a,
        "b": b,
        "set": set_name,
        "n_pairs": len(keys),
        "n_seeds": len({k[0] for k in keys}),
        "det_a": sum(da) / len(keys),
        "det_b": sum(db) / len(keys),
        "difference": float(diffs.mean()),
        "difference_ci_phrases": [float(lo), float(hi)],
        "a_only": row_level["a_only"],
        "b_only": row_level["b_only"],
        "mcnemar_p": row_level["p_value"],
        "n_phrases": len(diffs),
        "phrases_a_ahead": a_wins,
        "phrases_b_ahead": b_wins,
        "sign_test_p": stats.sign_test(a_wins, b_wins),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment of a family of p-values, in the order given."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def interaction(rows: list[dict], cfg: dict, set_name: str) -> dict | None:
    """Is the anchor worth more with steering than without it? Per phrase:
    (both - steer only) - (anchor only - neither), then the mean over phrases with a bootstrap interval."""
    sc = cfg["stats"]
    cells = {}
    for name, spec in cfg["systems"].items():
        if spec.get("role") == "inpass":
            cells[(spec["anchor"], spec["steer"])] = name
    if len(cells) != 4:
        return None
    rate = {}
    for key, name in cells.items():
        per_phrase = defaultdict(list)
        for r in rows:
            if r["condition"] == name and r["set"] == set_name:
                per_phrase[r["phrase_id"]].append(r["detected"])
        rate[key] = {p: float(np.mean(v)) for p, v in per_phrase.items()}
    phrases = sorted(set.intersection(*(set(v) for v in rate.values())))
    if not phrases:
        return None
    d = np.array(
        [
            (rate[(True, True)][p] - rate[(False, True)][p]) - (rate[(True, False)][p] - rate[(False, False)][p])
            for p in phrases
        ]
    )
    rng = np.random.default_rng(sc["seed"])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(sc["n_boot"])]
    lo, hi = np.quantile(boot, [(1 - sc["ci"]) / 2, 1 - (1 - sc["ci"]) / 2])
    return {
        "set": set_name,
        "n_phrases": len(phrases),
        "det": {f"anchor={a},steer={s}": float(np.mean(list(rate[(a, s)].values()))) for a, s in cells},
        "interaction": float(d.mean()),
        "interaction_ci_phrases": [float(lo), float(hi)],
    }


def null_rows(rows: list[dict], cfg: dict) -> list[dict]:
    """False-step rate of every kind of no-switch row, each clip counted once."""
    out = []
    groups = defaultdict(list)
    for r in rows:
        if r["condition"] in NULL_CONDITIONS and analysis_group(r):
            groups[(r["condition"], r["set"])].append(r)
    for (cond, s), g in groups.items():
        k = sum(r["exceeds_null"] for r in g)
        steps = [abs(primary_step(r)["delta"]) for r in g if primary_step(r)]
        out.append(
            {
                "condition": cond,
                "set": s,
                "n": len(g),
                "exceed": k,
                "rate": k / len(g),
                "ci_wilson": stats.wilson_ci(k, len(g), cfg["stats"]["ci"]),
                "abs_step_mean": float(np.mean(steps)) if steps else None,
                "abs_step_max": float(np.max(steps)) if steps else None,
                "invalid": sum(not r["valid"] for r in g),
            }
        )
    return out


def changepoint_offsets(rows: list[dict], cfg: dict) -> list[dict]:
    """Where the best split of the profile sits against the boundary, over the valid switch rows whose step
    has the commanded sign and the required size. Negative = before the boundary."""
    dc = cfg["detector"]
    thr, tol = dc["min_abs_delta"], dc["max_changepoint_err_s"]
    out = []
    order = {s: i for i, s in enumerate(cfg["sets"])}
    for cond in systems_of(cfg, rows):
        sub = [r for r in rows if r["condition"] == cond]
        for s in sorted({r["set"] for r in sub}, key=order.get):
            valid = [primary_step(r) for r in sub if r["set"] == s and r["valid"]]
            steps = [st for st in valid if st and st["sign_ok"] and abs(st["delta"]) >= thr]
            steps = [st for st in steps if st["changepoint_err_s"] is not None]
            off = [st["changepoint_s"] - st["switch_time_s"] for st in steps]
            far = [st["changepoint_err_s"] > tol for st in steps]  # the detector's own comparison
            out.append(
                {
                    "system": cond,
                    "set": s,
                    "n_valid": len(valid),
                    "n_step": len(steps),
                    "median_s": float(np.median(off)) if off else None,
                    "early": sum(f and o < 0 for f, o in zip(far, off, strict=True)),
                    "within": sum(not f for f in far),
                    "late": sum(f and o > 0 for f, o in zip(far, off, strict=True)),
                }
            )
    return out


def metrics_of(cfg: dict, out, key: str) -> dict:
    """`key` of metrics.json of this experiment and of the baseline experiment, read side by side."""
    read = lambda d: json.loads((d / "metrics.json").read_text(encoding="utf-8"))[key]  # noqa: E731
    return {"own": read(out), "baseline": read(ROOT / BASELINE_DIR)}


def by_pair(rows: list[dict], cfg: dict) -> dict:
    """system -> set -> 'e1>e2' -> [detected, n]."""
    out = {name: defaultdict(lambda: defaultdict(lambda: [0, 0])) for name in systems_of(cfg, rows)}
    for r in rows:
        if r["condition"] in out:
            cell = out[r["condition"]][r["set"]][f"{r['emotion']}>{r['emotion2']}"]
            cell[0] += r["detected"]
            cell[1] += 1
    return json.loads(json.dumps(out))


# ── tables ──────────────────────────────────────────────────────────────────


def set_name(cfg: dict, s: str) -> str:
    return f"{s} ({set_role(cfg, s)})"


def experiment_table(records: list[dict], cfg: dict, pooled: bool) -> str:
    rows = []
    for e in records:
        if (e["seed"] is None) != pooled:
            continue
        row = [e["system"], set_name(cfg, e["set"])]
        row += [str(e["n_seeds"])] if pooled else [str(e["seed"])]
        row += [frac(e["detected"], e["n"]), f"**{e['det']:.3f}**" if pooled else fmt(e["det"])]
        if pooled:
            row += [ci(e["ci_wilson"]), ci(e["ci_phrases"])]
        row += [str(e["invalid"]), fmt(e["wer_mean"]), fmt(e["commanded_step_mean"])]
        rows.append(row)
    head = ["system", "set", "seeds" if pooled else "seed", "detected", "det"]
    if pooled:
        head += ["95% CI (Wilson, rows)", "95% CI (bootstrap over phrases)"]
    return table([*head, "invalid clips", "WER mean", "mean commanded step"], rows)


def system_table(cfg: dict, names: list[str]) -> str:
    rows = [[n, cfg["systems"][n]["role"], cfg["systems"][n].get("change", ""), cfg["systems"][n].get("hypothesis", "")]
            for n in names]  # fmt: skip
    return table(["system", "role", "what is changed against the baseline", "hypothesis"], rows)


def contrast_table(contrasts: list[dict], cfg: dict) -> str:
    rows = []
    for c in contrasts:
        rows.append(
            [
                c["question"],
                f"{c['a']} vs {c['b']}",
                set_name(cfg, c["set"]),
                str(c["n_pairs"]),
                f"{c['det_a']:.3f} vs {c['det_b']:.3f}",
                f"{c['difference']:+.3f} {ci(c['difference_ci_phrases'])}",
                f"{c['a_only']} / {c['b_only']}",
                f"{c['mcnemar_p']:.3g}",
                f"{c['phrases_a_ahead']} / {c['phrases_b_ahead']} of {c['n_phrases']}",
                f"{c['sign_test_p']:.3g}",
                f"{c['sign_test_p_holm']:.3g}" if c["confirmatory"] else "",
            ]
        )
    head = ["question", "A vs B", "set", "paired rows", "det A vs B", "difference, 95% CI over phrases",
            "detected by A only / B only", "McNemar p (rows)", "phrases where A / B is ahead",
            "sign test p (phrases)", "Holm-adjusted, confirmatory tests"]  # fmt: skip
    return table(head, rows)


def interaction_table(items: list[dict], cfg: dict) -> str:
    rows = []
    for i in items:
        d = i["det"]
        rows.append(
            [
                set_name(cfg, i["set"]),
                str(i["n_phrases"]),
                fmt(d["anchor=False,steer=False"]),
                fmt(d["anchor=True,steer=False"]),
                fmt(d["anchor=False,steer=True"]),
                fmt(d["anchor=True,steer=True"]),
                f"{i['interaction']:+.3f} {ci(i['interaction_ci_phrases'])}",
            ]
        )
    head = ["set", "phrases", "neither", "anchor only", "steer only", "both",
            "interaction (both - steer only) - (anchor only - neither), 95% CI over phrases"]  # fmt: skip
    return table(head, rows)


def null_table(items: list[dict], cfg: dict) -> str:
    rows = [
        [
            i["condition"],
            set_name(cfg, i["set"]),
            str(i["n"]),
            f"{frac(i['exceed'], i['n'])} = {i['rate']:.3f}",
            ci(i["ci_wilson"]),
            fmt(i["abs_step_mean"]),
            fmt(i["abs_step_max"]),
            str(i["invalid"]),
        ]
        for i in items
    ]
    head = ["no-switch rows", "set", "n", "would pass the detector (either sign)", "95% CI (Wilson)",
            "mean abs step", "max abs step", "invalid clips"]  # fmt: skip
    return table(head, rows)


def outcome_table(records: list[dict], cfg: dict) -> str:
    rows = [
        [e["system"], set_name(cfg, e["set"]), str(e["n"]), *[str(e["outcomes"].get(o, 0)) for o in OUTCOMES]]
        for e in records
        if e["seed"] is None
    ]
    return table(["system", "set", "n", *OUTCOMES], rows)


def pair_table(pairs: dict, cfg: dict) -> str:
    names = [f"{a}>{b}" for a, b in cfg["pairs"]]
    rows = [
        [system, set_name(cfg, s), *[frac(*cells[p]) if p in cells else "" for p in names]]
        for system, by_set in pairs.items()
        for s, cells in by_set.items()
    ]
    return table(["system", "set", *[n.replace(">", " → ") for n in names]], rows)


def changepoint_table(items: list[dict], cfg: dict) -> str:
    tol = cfg["detector"]["max_changepoint_err_s"]
    rows = [
        [i["system"], set_name(cfg, i["set"]), frac(i["n_step"], i["n_valid"]),
         "" if i["median_s"] is None else f"{i['median_s']:+.2f}", str(i["early"]), str(i["within"]), str(i["late"])]
        for i in items
    ]  # fmt: skip
    head = ["system", "set", "rows with a step of the commanded sign and size, of the valid rows",
            "median offset of the change-point from the boundary, s", f"earlier than -{tol} s",
            f"within {tol} s", f"later than +{tol} s"]  # fmt: skip
    return table(head, rows)


def sensitivity_table(grids: dict, cfg: dict, sets: list[str], names: list[str]) -> str:
    """Counts of every system under each pair of detector constants and, at the frozen pair, with the weighted
    change-point estimator (`evaluate.sensitivity`); the frozen detector is in bold."""
    dc = cfg["detector"]
    base = {(g["min_abs_delta"], g["max_changepoint_err_s"], g["weighted_split"]): g["counts"]
            for g in grids["baseline"]}  # fmt: skip
    groups = [*names, *[c for c in NULL_CONDITIONS if c.endswith("_null")]]
    rows = []
    for s in sets:
        for g in grids["own"]:
            key = (g["min_abs_delta"], g["max_changepoint_err_s"], g["weighted_split"])
            counts = {**base.get(key, {}), **g["counts"]}
            frozen = key == (dc["min_abs_delta"], dc["max_changepoint_err_s"], False)
            cells = [frac(*counts[f"{n}/{s}"]) if f"{n}/{s}" in counts else "" for n in groups]
            tol = "none" if key[1] is None else f"{key[1]:.2f}"
            line = [set_name(cfg, s), f"{key[0]:.2f}", tol + (", weighted split" if key[2] else ""), *cells]
            rows.append([f"**{c}**" if frozen and c else c for c in line])
    return table(["set", "min step", "max change-point error, s", *groups], rows)


def agreement_table(agreement: dict, cfg: dict, sets: list[str], names: list[str]) -> str:
    merged = {**agreement["baseline"], **agreement["own"]}
    rows = []
    for n in [*names, *[c for c in NULL_CONDITIONS if c.endswith("_null")]]:
        for s in sets:
            a = merged.get(f"{n}/{s}")
            if a:
                rows.append([n, set_name(cfg, s), str(a["n_cells_multi_seed"]), str(a["unanimous"]),
                             str(a["unanimous_positive"]),
                             ", ".join(f"{k} {frac(*v)}" for k, v in a["seed_pairs"].items())])  # fmt: skip
    head = ["system", "set", "cells (phrase x switch) with more than one seed", "same decision at every seed",
            "of them detected", "same decision, by pair of seeds"]  # fmt: skip
    return table(head, rows)


# ── figures ─────────────────────────────────────────────────────────────────


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)


def _label(name: str) -> str:
    return {"nl_instruct": "baseline", "stitch": "reference\n(stitched)"}.get(name, name.replace("_", "\n+ "))


def fig_systems(records: list[dict], cfg: dict, path) -> None:
    """Detection rate of every system on every set: one dot per seed, a bar with the phrase-bootstrap interval."""
    import matplotlib.pyplot as plt

    sets = list(dict.fromkeys(e["set"] for e in records))
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes[0], sets, strict=True):
        pooled = [e for e in records if e["set"] == s and e["seed"] is None]
        for x, e in enumerate(pooled):
            color = COLOR.get(e["system"], INK2)
            ax.bar(x, e["det"], width=0.62, color=color, alpha=0.35, edgecolor=color)
            lo, hi = e["ci_phrases"]
            ax.plot([x, x], [lo, hi], color=color, linewidth=2)
            seeds = [r for r in records if r["set"] == s and r["system"] == e["system"] and r["seed"] is not None]
            ax.scatter(np.linspace(x - 0.15, x + 0.15, len(seeds)), [r["det"] for r in seeds], s=22, color=color,
                       zorder=3, edgecolor=SURFACE, linewidth=0.6)  # fmt: skip
            ax.text(x, min(0.99, hi + 0.03), f"{e['det']:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(len(pooled)), [_label(e["system"]) for e in pooled], fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(set_name(cfg, s), loc="left", fontsize=11, color=INK)
        _axes(ax)
    axes[0][0].set_ylabel("detection rate", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, "Bar: all seeds. Line: 95% bootstrap interval over phrases. Dots: single seeds.",
             color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_steps(rows: list[dict], cfg: dict, path) -> None:
    """The arousal step of every switch row, in the commanded direction, by system."""
    import matplotlib.pyplot as plt

    names = systems_of(cfg, rows)
    sets = [s for s in cfg["sets"] if any(r["set"] == s for r in rows)]
    thr = cfg["detector"]["min_abs_delta"]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, s in zip(axes[0], sets, strict=True):
        ax.axhspan(-thr, thr, color="#eae9e3", zorder=0)
        for x, name in enumerate(names):
            g = [r for r in rows if r["condition"] == name and r["set"] == s and r["commanded_step"] is not None]
            xs = x + rng.uniform(-0.22, 0.22, len(g))
            ys = [r["commanded_step"] for r in g]
            faces = [COLOR.get(name, INK2) if r["detected"] else SURFACE for r in g]
            ax.scatter(xs, ys, s=20, c=faces, edgecolors=COLOR.get(name, INK2), linewidths=1.1, zorder=2)
        ax.set_xticks(range(len(names)), [_label(n) for n in names], fontsize=8.5)
        ax.set_title(set_name(cfg, s), loc="left", fontsize=11, color=INK)
        _axes(ax)
    axes[0][0].set_ylabel("arousal step in the commanded direction", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, f"Filled dot: counted as a switch. Grey band: below the detector threshold {thr}.",
             color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_profiles(rows: list[dict], cfg: dict, path) -> None:
    """Mean arousal profile of every system around the boundary: time is counted from the boundary, the level
    from the clip's mean before it, in the commanded direction."""
    import matplotlib.pyplot as plt

    names = systems_of(cfg, rows)
    sets = [s for s in cfg["sets"] if any(r["set"] == s for r in rows)]
    hop, tol = cfg["judge"]["hop_s"], cfg["detector"]["max_changepoint_err_s"]
    lo, hi = -3.0, 3.0
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes[0], sets, strict=True):
        ax.axvspan(-tol, tol, color="#eae9e3", zorder=0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        for name in names:
            cells = defaultdict(list)
            for r in rows:
                step = primary_step(r) if r["condition"] == name and r["set"] == s and r["valid"] else None
                if not step:
                    continue
                t = np.asarray(r["profile_centres_s"]) - step["switch_time_s"]
                a = r["expect_sign"] * (np.asarray(r["profile_arousal"]) - step["mean_before"])
                for i, v in zip(np.floor(t / hop + 0.5).astype(int), a, strict=True):
                    cells[int(i)].append(float(v))
            n_clips = max((len(v) for v in cells.values()), default=0)
            idx = [i for i in sorted(cells) if lo <= i * hop <= hi and len(cells[i]) >= max(10, n_clips / 2)]
            ax.plot(
                [i * hop for i in idx],
                [float(np.mean(cells[i])) for i in idx],
                color=COLOR.get(name, INK2),
                linewidth=1.8,
                label=_label(name).replace("\n", " "),
            )
        ax.set_xlabel("time from the boundary, s", color=INK2, fontsize=10)
        ax.set_title(set_name(cfg, s), loc="left", fontsize=11, color=INK)
        _axes(ax)
    axes[0][0].set_ylabel("arousal against the level before the boundary,\nin the commanded direction",
                          color=INK2, fontsize=10)  # fmt: skip
    axes[0][-1].legend(frameon=False, fontsize=8.5, loc="upper left")
    note = (f"Mean over the valid switch rows; drawn where at least half of the clips have a window. "
            f"Grey band: where the detector accepts the change-point (within {tol} s).")  # fmt: skip
    fig.text(0.01, 0.01, note, color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_pairs(pairs: dict, cfg: dict, path) -> None:
    """Detection rate by commanded switch: systems in rows, switches in columns, one panel per set."""
    import matplotlib.pyplot as plt

    names = [f"{a}>{b}" for a, b in cfg["pairs"]]
    sets = [s for s in cfg["sets"] if any(s in by_set for by_set in pairs.values())]
    systems = list(pairs)
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets) + 1.2, 0.5 * len(systems) + 1.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes[0], sets, strict=True):
        grid = np.full((len(systems), len(names)), np.nan)
        for i, system in enumerate(systems):
            for j, p in enumerate(names):
                if p in pairs[system].get(s, {}):
                    k, n = pairs[system][s][p]
                    grid[i, j] = k / n
                    ax.text(j, i, f"{k}/{n}", ha="center", va="center", fontsize=8.5,
                            color="white" if k / n > 0.55 else INK)  # fmt: skip
        ax.imshow(grid, vmin=0, vmax=1, cmap="Purples", aspect="auto")
        ax.set_xticks(range(len(names)), [n.replace(">", " → ") for n in names], fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(systems)), systems if ax is axes[0][0] else [], fontsize=8.5, color=INK2)
        ax.set_title(set_name(cfg, s), loc="left", fontsize=11, color=INK)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_outcomes(records: list[dict], cfg: dict, path) -> None:
    """What happened to every switch row: stacked shares of the outcomes, by system and set."""
    import matplotlib.pyplot as plt

    shades = {"detected": "#3f9b6c", "invalid": "#0b0b0b", "flat": "#c3c2b7", "wrong_sign": "#d1584e",
              "no_changepoint": "#8d8b84", "misplaced": "#e2b33c", "no_step": "#5d5b56"}  # fmt: skip
    pooled = [e for e in records if e["seed"] is None]
    sets = list(dict.fromkeys(e["set"] for e in pooled))
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.0), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes[0], sets, strict=True):
        group = [e for e in pooled if e["set"] == s]
        bottom = np.zeros(len(group))
        for o in OUTCOMES:
            share = np.array([e["outcomes"].get(o, 0) / e["n"] for e in group])
            if share.any():
                ax.bar(range(len(group)), share, bottom=bottom, width=0.62, color=shades[o], label=o)
                bottom += share
        ax.set_xticks(range(len(group)), [_label(e["system"]) for e in group], fontsize=8.5)
        ax.set_title(set_name(cfg, s), loc="left", fontsize=11, color=INK)
        _axes(ax)
    axes[0][0].set_ylabel("share of switch rows", color=INK2, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ── files ───────────────────────────────────────────────────────────────────


def write_csv(path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def predictions(rows: list[dict], cfg: dict) -> list[dict]:
    out = []
    for r in rows:
        if r["condition"] not in systems_of(cfg, rows) and r["condition"] not in NULL_CONDITIONS:
            continue
        step = primary_step(r) or {}
        out.append(
            {
                **{k: r[k] for k in ("run", "id", "condition", "set", "seed", "phrase_id", "emotion", "emotion2")},
                "expect_sign": r["expect_sign"],
                "delta": step.get("delta"),
                "changepoint_err_s": step.get("changepoint_err_s"),
                "decision": r["exceeds_null"] if r["condition"] in NULL_CONDITIONS else r["detected"],
                "outcome": r.get("outcome"),
                "valid": r["valid"],
                "wer": round(r["wer"], 4),
                "duration_s": r["duration_s"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/inpass.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / "results" / cfg["name"]
    rows = load_rows(cfg, out)
    records = experiments(rows, cfg)
    sets = [s for s in cfg["sets"] if any(e["set"] == s for e in records if e["role"] == "inpass")]
    records = [e for e in records if e["set"] in sets]
    contrasts = []
    for s in sets:
        in_set = []
        for spec in cfg["contrasts"]:
            c = contrast(rows, cfg, spec["a"], spec["b"], s)
            if c:
                in_set.append({**c, "question": spec["question"], "confirmatory": spec.get("confirmatory", False)})
        family = [c for c in in_set if c["confirmatory"]]
        for c, p in zip(family, holm([c["sign_test_p"] for c in family]), strict=True):
            c["sign_test_p_holm"] = p
        contrasts += in_set
    interactions = [i for i in (interaction(rows, cfg, s) for s in sets) if i]
    nulls = [n for n in null_rows(rows, cfg) if n["set"] in sets]
    pairs = by_pair([r for r in rows if r["set"] in sets], cfg)
    names = systems_of(cfg, rows)
    offsets = [o for o in changepoint_offsets(rows, cfg) if o["set"] in sets]

    flat = [{**e, "ci_wilson": ci(e["ci_wilson"]), "ci_phrases": ci(e["ci_phrases"]), "seed": e["seed"] or "all",
             "outcomes": json.dumps(e["outcomes"], sort_keys=True)} for e in records]  # fmt: skip
    write_csv(out / "experiments.csv", flat, ["system", "role", "change", "hypothesis", "set", "set_role", "seed",
              "n_phrases", "n", "detected", "det", "ci_wilson", "ci_phrases", "invalid", "wer_mean",
              "commanded_step_mean", "outcomes"])  # fmt: skip
    preds = predictions([r for r in rows if r["set"] in sets], cfg)
    write_csv(out / "predictions.csv", preds, list(preds[0]))
    record = {"experiments": records, "contrasts": contrasts, "interaction": interactions, "null": nulls,
              "by_pair": pairs, "changepoint": offsets}  # fmt: skip
    (out / "experiments.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fig_systems(records, cfg, out / "fig_systems.png")
    fig_steps([r for r in rows if r["set"] in sets], cfg, out / "fig_inpass_steps.png")
    fig_pairs(pairs, cfg, out / "fig_pairs.png")
    fig_profiles([r for r in rows if r["set"] in sets], cfg, out / "fig_inpass_profiles.png")
    fig_outcomes(records, cfg, out / "fig_outcomes.png")
    sections = [
        ("Systems", system_table(cfg, names)),
        ("Results, all seeds", experiment_table(records, cfg, pooled=True)),
        ("Experiments: one line per system, set and seed", experiment_table(records, cfg, pooled=False)),
        ("Contrasts and ablations", contrast_table(contrasts, cfg)),
        ("Anchor by steering", interaction_table(interactions, cfg)),
        ("No-switch rows", null_table(nulls, cfg)),
        ("Outcome of every switch row", outcome_table(records, cfg)),
        ("By commanded switch", pair_table(pairs, cfg)),
        ("Where the change-point sits", changepoint_table(offsets, cfg)),
        ("Detector constants", sensitivity_table(metrics_of(cfg, out, "sensitivity"), cfg, sets, names)),
        ("Agreement between seeds", agreement_table(metrics_of(cfg, out, "seed_agreement"), cfg, sets, names)),
    ]
    text = "# In-pass systems: experiments\n\nGenerated by `python -m emoswitch.experiments`; do not edit by hand.\n"
    text += "".join(f"\n## {title}\n\n{body}\n" for title, body in sections)
    (out / "experiments.md").write_text(text, encoding="utf-8")
    print(f"[experiments] {len(records)} records, {len(contrasts)} contrasts -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

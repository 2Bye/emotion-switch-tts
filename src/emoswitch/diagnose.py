"""Error analysis, slices and robustness of the system chosen in homework 2.

    python -m emoswitch.diagnose --config configs/diagnostics.yaml

Reads the scored rows of the source experiment (config `source`, with the baseline experiment's rows of
the same sets) and the rows of the noisy clips (results/<name>/noise/, written by emoswitch.perturb).
Writes into results/<name>/: clips.csv (one line per clip of the chosen system with the type of its
miss), slices.csv, noise.csv, resample.csv, diagnostics.json, diagnostics.md (tables) and figures.
No models are loaded and nothing is tuned: every decision is the frozen detector's.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations

import matplotlib

matplotlib.use("Agg")  # same pixels on every platform, no display needed

import numpy as np  # noqa: E402

from emoswitch import detect, stats  # noqa: E402
from emoswitch.config import ROOT, load_config, set_role  # noqa: E402
from emoswitch.evaluate import add_derived, primary_step, read_jsonl  # noqa: E402
from emoswitch.experiments import COLOR, _axes, load_rows, set_name, write_csv  # noqa: E402
from emoswitch.report import AXIS, INK, INK2, SURFACE, ci, fmt, frac, table  # noqa: E402

# A miss with a step of the right size and sign is one of the first four; the rest are the detector's outcomes.
TYPES = ("edge_split", "early_change", "late_change", "other", "flat", "wrong_sign", "invalid", "no_step",
         "no_changepoint")  # fmt: skip
STEP_PRESENT = ("detected", "edge_split", "early_change", "late_change", "other")
TYPE_COLOR = {"detected": "#3f9b6c", "edge_split": "#b8a46a", "early_change": "#2a78d6", "late_change": "#eb6834",
              "other": "#898781", "flat": "#c9c7c0", "wrong_sign": "#a23b3b", "invalid": "#3a3936",
              "no_step": "#6d6c68", "no_changepoint": "#6d6c68"}  # fmt: skip
DIRECTION = {1: "up", -1: "down"}


# ── one clip ────────────────────────────────────────────────────────────────


def offsets(row: dict, cfg: dict, step: dict | None = None) -> tuple[float, float] | None:
    """Signed distance from the boundary to the change-point, by the detector's estimator and by the weighted
    one (negative = before the boundary). None when the row has no step or no change-point. `step`: the
    row's step at another boundary than the primary one."""
    step = step or primary_step(row)
    if not step or step["changepoint_s"] is None:
        return None
    weighted = detect.measure_step(row["profile_centres_s"], row["profile_arousal"], step["switch_time_s"],
                                   row["expect_sign"], cfg["detector"]["min_seg_windows"], True)  # fmt: skip
    t = step["switch_time_s"]
    return round(step["changepoint_s"] - t, 3), round(weighted["changepoint_s"] - t, 3)


def miss_type(row: dict, cfg: dict) -> str:
    """`detected`, or why the frozen detector did not count the row (docs/plan_hw3.md)."""
    if row["detected"]:
        return "detected"
    if not row["valid"]:
        return "invalid"  # also on a set whose primary rule does not gate on validity
    if row["outcome"] != "misplaced":
        return row["outcome"]
    plain, weighted = offsets(row, cfg)
    if abs(weighted) <= cfg["detector"]["max_changepoint_err_s"]:
        return "edge_split"
    if plain < 0 and weighted < 0:
        return "early_change"
    if plain > 0 and weighted > 0:
        return "late_change"
    return "other"


def null_levels(rows: list[dict], null: str) -> dict:
    """Level after the boundary in the clips with no commanded switch: (set, seed, phrase, emotion) -> level."""
    steps = {
        (r["set"], r["seed"], r["phrase_id"], r["emotion"]): primary_step(r) for r in rows if r["condition"] == null
    }
    return {k: step["mean_after"] for k, step in steps.items() if step}


def describe(row: dict, cfg: dict, plain_level: dict | None = None) -> dict:
    """`plain_level` (from `null_levels`): the level the same model reaches in sentence 2 of the same phrase and
    seed when the second emotion is asked for the whole clip."""
    step, off = primary_step(row), offsets(row, cfg)
    t = step["switch_time_s"] if step else None
    target = (plain_level or {}).get((row["set"], row["seed"], row["phrase_id"], row["emotion2"]))
    short_side = None
    if off:  # windows on the shorter side of the detector's split
        left = int((np.asarray(row["profile_centres_s"]) < step["changepoint_s"]).sum())
        short_side = min(left, len(row["profile_centres_s"]) - left)
    return {
        "run": row["run"],
        "id": row["id"],
        "audio": row["audio"],
        "condition": row["condition"],
        "set": row["set"],
        "seed": row["seed"],
        "phrase_id": row["phrase_id"],
        "emotion": row["emotion"],
        "emotion2": row["emotion2"],
        "direction": DIRECTION[row["expect_sign"]],
        "detected": row["detected"],
        "type": miss_type(row, cfg),
        "commanded_step": row["commanded_step"],
        "offset_s": off[0] if off else None,
        "offset_weighted_s": off[1] if off else None,
        "crossing_s": crossing(row, cfg),
        "short_side_windows": short_side,
        "level_before": step["mean_before"] if step else None,
        "level_after": step["mean_after"] if step else None,
        "level_plain": target,
        "beyond_plain": None
        if not step or target is None
        else round(row["expect_sign"] * (step["mean_after"] - target), 4),
        "seconds_before": t,
        "seconds_after": None if t is None else round(row["duration_s"] - t, 3),
        "valid": row["valid"],
        "wer": round(row["wer"], 4),
        "duration_s": row["duration_s"],
    }


def crossing(row: dict, cfg: dict, step: dict | None = None) -> float | None:
    """Signed distance from the boundary to the first window from which the profile is on the far side of
    the midpoint between its mean levels before and after the boundary for `min_seg_windows` windows in a
    row. A third reading of where the level changes; it uses no split of the profile, but the same boundary
    and the same two mean levels as the step. None when the profile never does that."""
    step = step or primary_step(row)
    if not step:
        return None
    c, a = np.asarray(row["profile_centres_s"]), np.asarray(row["profile_arousal"])
    side = np.sign(step["delta"]) * (a - (step["mean_before"] + step["mean_after"]) / 2) > 0
    k = cfg["detector"]["min_seg_windows"]
    for i in range(len(a) - k + 1):
        if side[i : i + k].all():
            return round(float(c[i]) - step["switch_time_s"], 3)
    return None


def boundary_candidates(row: dict) -> dict | None:
    """Other places to read the sentence boundary from the same alignment: the end of the last word of
    sentence 1 and the middle of the pause. The detector uses the start of the first word of sentence 2."""
    step = primary_step(row)
    words = row.get("words") or row.get("words_aligned") or []
    before = [w for w in words if step and w["end"] <= step["switch_time_s"] + 1e-6]
    if not before:
        return None
    start, end = step["switch_time_s"], before[-1]["end"]
    return {"start of sentence 2": start, "middle of the pause": (start + end) / 2, "end of sentence 1": end}


def boundary_readings(rows: list[dict], cfg: dict) -> list[dict]:
    """How many switch rows the unchanged detector would count if the boundary were read elsewhere in the pause.
    A diagnostic of the misses, not a metric: nothing is selected by it."""
    dc = cfg["detector"]
    out = []
    for cond in dict.fromkeys(r["condition"] for r in rows):
        for sign, direction in DIRECTION.items():
            g = [r for r in rows if r["condition"] == cond and r["expect_sign"] == sign]
            counts = Counter()
            for r in g:
                gated = cfg["primary_rule"][set_role(cfg, r["set"])]["validity_gate"] and not r["valid"]
                for name, t in (boundary_candidates(r) or {}).items():
                    step = detect.measure_step(r["profile_centres_s"], r["profile_arousal"], t, sign,
                                               dc["min_seg_windows"])  # fmt: skip
                    hit = detect.is_detected(step, dc["min_abs_delta"], dc["max_changepoint_err_s"])
                    counts[name] += bool(hit and not gated)
            pauses = [c["start of sentence 2"] - c["end of sentence 1"] for c in map(boundary_candidates, g) if c]
            out.append({"condition": cond, "direction": direction, "n": len(g), "counted": dict(counts),
                        "pause_median_s": float(np.median(pauses))})  # fmt: skip
    return out


def placement(recs: list[dict], cfg: dict) -> list[dict]:
    """Of the clips with a step of the commanded sign and size: how many have the change within the tolerance
    of the boundary by each of the three readings."""
    tol = cfg["detector"]["max_changepoint_err_s"]
    out = []
    for direction in ("up", "down", None):
        g = [r for r in recs if r["type"] in STEP_PRESENT and direction in (None, r["direction"])]
        out.append(
            {
                "direction": direction or "all",
                "n": len(g),
                **{
                    name: sum(r[field] is not None and abs(r[field]) <= tol for r in g)
                    for name, field in (
                        ("detector", "offset_s"),
                        ("weighted", "offset_weighted_s"),
                        ("crossing", "crossing_s"),
                    )
                },  # fmt: skip
                "crossing_median_s": float(np.median([r["crossing_s"] for r in g if r["crossing_s"] is not None])),
            }
        )
    return out


def calibration(rows: list[dict], cfg: dict, null: str) -> list[dict]:
    """The three readings where the answer is known. On the stitched reference the change sits at the join
    by construction: how often each reading falls within the tolerance of it, over the valid clips with a
    step of the commanded sign and size. On the clips with no commanded switch there is no change: how often
    the two estimators fall within the tolerance of the boundary anyway."""
    dc = cfg["detector"]
    thr, tol = dc["min_abs_delta"], dc["max_changepoint_err_s"]
    s, ref = cfg["analysis_set"], cfg["reference"]

    def within(values: list) -> int:
        return sum(v is not None and abs(v) <= tol for v in values)

    out = []
    for sign, direction in DIRECTION.items():
        g = [r for r in rows if r["condition"] == ref and r["set"] == s and r["expect_sign"] == sign and r["valid"]
             and r["switch_time_source"] == "construction" and r["step"] and r["step"]["sign_ok"]
             and abs(r["step"]["delta"]) >= thr and r["step"]["changepoint_s"] is not None]  # fmt: skip
        offs = [offsets(r, cfg, r["step"]) for r in g]
        out.append({"group": f"{ref}, switch {direction}: join by construction", "n": len(g),
                    "detector": within([o[0] for o in offs]), "weighted": within([o[1] for o in offs]),
                    "crossing": within([crossing(r, cfg, r["step"]) for r in g])})  # fmt: skip
    g = [r for r in rows if r["condition"] == null and r["set"] == s and offsets(r, cfg)]
    offs = [offsets(r, cfg) for r in g]
    out.append({"group": f"{null}: no switch, boundary by alignment", "n": len(g),
                "detector": within([o[0] for o in offs]), "weighted": within([o[1] for o in offs]),
                "crossing": None})  # fmt: skip
    return out


def edge_sides(recs: list[dict], cfg: dict) -> list[dict]:
    """Clips of the type `edge_split` by direction: on which side of the boundary the detector's split sits
    and how many windows its shorter side has."""
    k = cfg["detector"]["min_seg_windows"]
    out = []
    for direction in ("up", "down", None):
        g = [r for r in recs if r["type"] == "edge_split" and direction in (None, r["direction"])]
        out.append({"direction": direction or "all", "n": len(g),
                    "before": sum(r["offset_s"] < 0 for r in g), "after": sum(r["offset_s"] > 0 for r in g),
                    "short_min": sum(r["short_side_windows"] == k for r in g),
                    "short_mid": sum(k < r["short_side_windows"] <= k + 2 for r in g),
                    "short_long": sum(r["short_side_windows"] > k + 2 for r in g)})  # fmt: skip
    return out


# ── groups of clips ─────────────────────────────────────────────────────────


def group_stats(recs: list[dict], cfg: dict) -> dict:
    n, k = len(recs), sum(r["detected"] for r in recs)
    return {
        "n": n,
        "detected": k,
        "det": k / n if n else None,
        "ci_wilson": stats.wilson_ci(k, n, cfg["stats"]["ci"]),
        "step_present": sum(r["type"] in STEP_PRESENT for r in recs),
        "types": dict(Counter(r["type"] for r in recs if r["type"] != "detected")),
    }


def bin_label(x: float | None, edges: list[float], unit: str) -> str:
    if x is None:
        return "n/a"
    i = int(np.searchsorted(edges, x, side="right"))
    if i == 0:
        return f"< {edges[0]:.2f}{unit}"
    if i == len(edges):
        return f"≥ {edges[-1]:.2f}{unit}"
    return f"{edges[i - 1]:.2f}–{edges[i]:.2f}{unit}"


def equal_count_edges(values: list[float], n_groups: int) -> list[float]:
    return [float(q) for q in np.quantile(values, [i / n_groups for i in range(1, n_groups)])]


def slice_keys(recs: list[dict], cfg: dict) -> dict:
    """Slice name -> function from a clip record to its group. Edges of the time slices are set on `recs`."""
    sc = cfg["slices"]
    after = equal_count_edges([r["seconds_after"] for r in recs if r["seconds_after"] is not None],
                              sc["seconds_after_boundary"])  # fmt: skip
    before = equal_count_edges([r["seconds_before"] for r in recs if r["seconds_before"] is not None],
                               sc["seconds_before_boundary"])  # fmt: skip
    return {
        "direction": lambda r: r["direction"],
        "switch": lambda r: f"{r['emotion']} → {r['emotion2']}",
        "seconds after the boundary": lambda r: bin_label(r["seconds_after"], after, " s"),
        "seconds before the boundary": lambda r: bin_label(r["seconds_before"], before, " s"),
        "seed": lambda r: str(r["seed"]),
        "commanded step": lambda r: bin_label(r["commanded_step"], sc["commanded_step"], ""),
        "phrase": lambda r: r["phrase_id"],
    }


def group_order(label: str) -> tuple:
    """Groups in reading order: `< a`, `a–b`, `≥ b`; phrases by number; the rest as they come."""
    if label[:1] == "<":
        return (0, 0.0)
    if label[:1] == "≥":
        return (2, 0.0)
    head = label.split("–")[0].replace(" s", "")
    if label[:1] == "p" and label[1:].isdigit():
        return (1, float(label[1:]))
    try:
        return (1, float(head))
    except ValueError:
        return (1, 0.0)


def slices(recs: list[dict], cfg: dict) -> list[dict]:
    out = []
    for name, key in slice_keys(recs, cfg).items():
        groups = {}
        for r in recs:
            groups.setdefault(key(r), []).append(r)
        labels = list(groups)
        if name not in ("direction", "switch", "seed"):
            labels.sort(key=group_order)
        out += [{"slice": name, "group": g, **group_stats(groups[g], cfg)} for g in labels]
    return out


def cross(recs: list[dict], cfg: dict, a: str, b: str) -> dict:
    keys = slice_keys(recs, cfg)
    cells = {}
    for r in recs:
        cells.setdefault((keys[a](r), keys[b](r)), []).append(r)
    cols = sorted({k[1] for k in cells}, key=group_order)
    return {
        "rows": list(dict.fromkeys(k[0] for k in cells)),
        "cols": cols,
        "cells": {f"{i}|{j}": group_stats(v, cfg) for (i, j), v in cells.items()},
    }


def drift(rows: list[dict]) -> list[dict]:
    """Shape of the arousal profile in clips with no commanded switch, by emotion: what the level does on its
    own inside a clip."""
    out = []
    for emotion in dict.fromkeys(r["emotion"] for r in rows):
        g = [r for r in rows if r["emotion"] == emotion]
        prof = [np.asarray(r["profile_arousal"]) for r in g]
        first, mid, last = ([float(p[sl].mean()) for p in prof] for sl in (slice(3), slice(3, -3), slice(-3, None)))
        slope = [float(np.polyfit(r["profile_centres_s"], p, 1)[0]) for r, p in zip(g, prof, strict=True)]
        steps = [primary_step(r)["delta"] for r in g if primary_step(r)]
        out.append(
            {
                "emotion": emotion,
                "n": len(g),
                "first": float(np.mean(first)),
                "middle": float(np.mean(mid)),
                "last": float(np.mean(last)),
                "first_above_middle": sum(f > m for f, m in zip(first, mid, strict=True)),
                "last_below_middle": sum(e < m for e, m in zip(last, mid, strict=True)),
                "slope_per_s": float(np.mean(slope)),
                "step_at_boundary": float(np.mean(steps)),
                "step_negative": sum(s < 0 for s in steps),
            }
        )
    return out


def beyond(recs_by_condition: dict, cfg: dict) -> list[dict]:
    """Level after the switch against the level of a clip of the second emotion without a switch, by system and
    second emotion, over the valid clips. `beyond` is counted in the commanded direction."""
    thr = cfg["detector"]["min_abs_delta"]
    out = []
    for cond, recs in recs_by_condition.items():
        for emotion in dict.fromkeys(r["emotion2"] for r in recs):
            g = [r for r in recs if r["emotion2"] == emotion and r["valid"] and r["beyond_plain"] is not None]
            out.append(
                {
                    "condition": cond,
                    "emotion2": emotion,
                    "n": len(g),
                    "level_before": float(np.mean([r["level_before"] for r in g])),
                    "level_after": float(np.mean([r["level_after"] for r in g])),
                    "level_plain": float(np.mean([r["level_plain"] for r in g])),
                    "beyond_mean": float(np.mean([r["beyond_plain"] for r in g])),
                    "beyond_over_threshold": sum(r["beyond_plain"] >= thr for r in g),
                    "short_over_threshold": sum(r["beyond_plain"] <= -thr for r in g),
                }
            )
    return out


def invalid_by_anchor(rows: list[dict], cfg: dict) -> dict:
    """Invalid clips among the switch rows of the in-pass systems with and without the anchor, per set and
    over the sets, with Fisher's exact test over the sets."""
    from scipy.stats import fisher_exact

    specs = {c: spec for c, spec in cfg["systems"].items() if spec.get("role") == "inpass"}
    sets = [x for x in cfg["sets"] if any(r["set"] == x and r["condition"] in specs for r in rows)]
    out = {"sets": {}}
    total = {True: [0, 0], False: [0, 0]}
    for x in sets:
        out["sets"][x] = {}
        for anchor in (True, False):
            g = [r for r in rows if r["set"] == x and specs.get(r["condition"], {}).get("anchor") is anchor]
            k = sum(not r["valid"] for r in g)
            out["sets"][x]["anchor" if anchor else "no_anchor"] = [k, len(g)]
            total[anchor][0] += k
            total[anchor][1] += len(g)
    (ka, na), (kn, nn) = total[True], total[False]
    out["all"] = {"anchor": [ka, na], "no_anchor": [kn, nn]}
    out["fisher_p"] = float(fisher_exact([[ka, na - ka], [kn, nn - kn]])[1])
    return out


def seed_cells(recs: list[dict]) -> dict:
    """A cell is a phrase with a commanded switch; it has one clip per seed."""
    cells = {}
    for r in recs:
        cells.setdefault((r["phrase_id"], r["emotion"], r["emotion2"]), []).append(r)
    counts = Counter(sum(r["detected"] for r in g) for g in cells.values())
    never = [{"phrase_id": p, "switch": f"{a} → {b}", "types": [r["type"] for r in g]}
             for (p, a, b), g in cells.items() if not any(r["detected"] for r in g)]  # fmt: skip
    return {"n_cells": len(cells), "by_seeds_detected": {str(k): counts[k] for k in sorted(counts)}, "never": never}


def pick_examples(recs: list[dict]) -> list[dict]:
    """One clip per kind of miss, picked by rule: the miss with the largest commanded step, i.e. the clearest
    step that still was not counted (an invalid clip: the highest WER). `edge_split` gets one early and one late."""
    kinds = [("edge_split", "early"), ("edge_split", "late"), ("early_change", None), ("late_change", None),
             ("other", None), ("flat", None), ("wrong_sign", None), ("invalid", None)]  # fmt: skip
    out = []
    for kind, side in kinds:
        g = [r for r in recs if r["type"] == kind]
        if side:
            g = [r for r in g if (r["offset_s"] < 0) == (side == "early")]
        if not g:
            continue
        key = (lambda r: r["wer"]) if kind == "invalid" else (lambda r: r["commanded_step"] or 0.0)
        out.append({**max(g, key=key), "example_of": kind + (f", {side}" if side else "")})
    counted = [r for r in recs if r["detected"] and r["beyond_plain"] is not None]
    if counted:  # a counted clip: the one whose level goes furthest beyond the plain level of the second emotion
        out.append({**max(counted, key=lambda r: r["beyond_plain"]), "example_of": "counted, level beyond"})
    return out


# ── robustness ──────────────────────────────────────────────────────────────


def noise_levels(cfg: dict, out, clean: list[dict]) -> list[dict]:
    """One record per noise level (None = the released clips): the chosen system and the no-switch rows."""
    nc, system = cfg["noise"], cfg["system"]
    null = next(c for c in nc["conditions"] if c != system)
    keep = {(r["run"], r["id"]): r for r in clean if r["set"] == nc["set"] and r["condition"] in nc["conditions"]}
    levels = [(None, list(keep.values()))]
    for snr_db in nc["snr_db"]:
        path = out / "noise" / f"snr_{snr_db:g}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            add_derived(rows, cfg)
            levels.append((snr_db, rows))
    records = []
    for snr_db, rows in levels:
        sw = [r for r in rows if r["condition"] == system]
        nl = [r for r in rows if r["condition"] == null]
        k = sum(r["detected"] for r in sw)
        steps = [r["commanded_step"] for r in sw if r["commanded_step"] is not None]
        base = [keep[(r["run"], r["id"])] for r in sw]
        moved = [abs(primary_step(r)["switch_time_s"] - primary_step(b)["switch_time_s"])
                 for r, b in zip(sw, base, strict=True) if primary_step(r) and primary_step(b)]  # fmt: skip
        records.append(
            {
                "snr_db": snr_db,
                "n": len(sw),
                "detected": k,
                "det": k / len(sw),
                "ci_wilson": stats.wilson_ci(k, len(sw), cfg["stats"]["ci"]),
                "ci_phrases": stats.cluster_bootstrap_ci(
                    [float(r["detected"]) for r in sw],
                    [r["phrase_id"] for r in sw],
                    cfg["stats"]["n_boot"],
                    cfg["stats"]["seed"],
                    cfg["stats"]["ci"],
                ),  # fmt: skip
                "invalid": sum(not r["valid"] for r in sw),
                "wer_mean": float(np.mean([r["wer"] for r in sw])),
                "commanded_step_mean": float(np.mean(steps)) if steps else None,
                "step_present": sum(miss_type(r, cfg) in STEP_PRESENT for r in sw),
                "types": dict(Counter(t for t in (miss_type(r, cfg) for r in sw) if t != "detected")),
                "lost": sum(b["detected"] and not r["detected"] for r, b in zip(sw, base, strict=True)),
                "gained": sum(r["detected"] and not b["detected"] for r, b in zip(sw, base, strict=True)),
                "boundary_moved_median_s": float(np.median(moved)) if moved else None,
                "profile_sd_mean": float(np.mean([r["profile_sd"] for r in sw])),
                "null_n": len(nl),
                "null_exceeds": sum(r["exceeds_null"] for r in nl),
                "null_changed": sum(r["exceeds_null"] != keep[(r["run"], r["id"])]["exceeds_null"] for r in nl),
                "null_invalid": sum(not r["valid"] for r in nl),
            }
        )
    return records


def resample(rows: list[dict], cfg: dict) -> dict:
    """det of the chosen system on other seeds and other subsets of the phrases of the analysis set, from the
    same renders, and whether it stays above each single-component system there."""
    system, s = cfg["system"], cfg["analysis_set"]
    if cfg["resample"]["half_splits"] != "all":
        raise SystemExit("[diagnose] resample.half_splits: only `all` (every half of the phrases) is implemented")
    others = [c for c, spec in cfg["systems"].items() if spec.get("role") == "inpass" and c != system
              and (spec["anchor"] or spec["steer"])]  # fmt: skip
    cell = {c: [r for r in rows if r["condition"] == c and r["set"] == s] for c in (system, *others)}
    phrases = sorted({r["phrase_id"] for r in cell[system]}, key=lambda p: int(p[1:]))

    def det(cond: str, keep) -> tuple[int, int]:
        g = [r for r in cell[cond] if keep(r)]
        return sum(r["detected"] for r in g), len(g)

    def record(label: str, keep) -> dict:
        k, n = det(system, keep)
        rest = {c: det(c, keep) for c in others}
        return {"subset": label, "detected": k, "n": n, "det": k / n,
                "above_each_component": all(k / n > kc / nc for kc, nc in rest.values())}  # fmt: skip

    seeds = [record(f"seed {seed}", lambda r, seed=seed: r["seed"] == seed)
             for seed in sorted({r["seed"] for r in cell[system]})]  # fmt: skip
    lopo = [record(f"without {p}", lambda r, p=p: r["phrase_id"] != p) for p in phrases]
    halves = [record("half", lambda r, half=frozenset(half): r["phrase_id"] in half)
              for half in combinations(phrases, len(phrases) // 2)]  # fmt: skip
    values = np.array([h["det"] for h in halves])
    return {
        "all": record("all seeds, all phrases", lambda r: True),
        "seeds": seeds,
        "leave_one_phrase_out": lopo,
        "halves": {
            "n_subsets": len(halves),
            "phrases_per_subset": len(phrases) // 2,
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "at_least_half": int((values >= 0.5).sum()),
            "above_each_component": sum(h["above_each_component"] for h in halves),
            "values": [round(float(v), 4) for v in values],
        },
    }


def run_record(source: str, seeds: str, sw: list[dict], nl: list[dict], cfg: dict, base: dict | None = None) -> dict:
    """Counts of one group of renders of the chosen system (`sw`) with its no-switch rows (`nl`).
    `base`: (seed, id) -> the row of the same clip in the render it is compared with."""
    sc, thr = cfg["stats"], cfg["detector"]["min_abs_delta"]
    k = sum(r["detected"] for r in sw)
    plain = {(r["seed"], r["phrase_id"], r["emotion"]): primary_step(r)["mean_after"] for r in nl if primary_step(r)}
    beyond_plain = []
    for r in sw:
        step, key = primary_step(r), (r["seed"], r["phrase_id"], r["emotion2"])
        if r["valid"] and step and key in plain:
            beyond_plain.append(r["expect_sign"] * (step["mean_after"] - plain[key]))
    steps = [r["commanded_step"] for r in sw if r["commanded_step"] is not None]
    return {
        "source": source,
        "seeds": seeds,
        "n": len(sw),
        "detected": k,
        "det": k / len(sw),
        "ci_wilson": stats.wilson_ci(k, len(sw), sc["ci"]),
        "ci_phrases": stats.cluster_bootstrap_ci(
            [float(r["detected"]) for r in sw], [r["phrase_id"] for r in sw], sc["n_boot"], sc["seed"], sc["ci"]
        ),
        "by_direction": {
            d: {"n": sum(r["expect_sign"] == s for r in sw),
                "detected": sum(r["detected"] for r in sw if r["expect_sign"] == s)}
            for s, d in DIRECTION.items()
        },
        "invalid": sum(not r["valid"] for r in sw),
        "commanded_step_mean": float(np.mean(steps)) if steps else None,
        "types": dict(Counter(t for t in (miss_type(r, cfg) for r in sw) if t != "detected")),
        "null_n": len(nl),
        "null_exceeds": sum(r["exceeds_null"] for r in nl),
        "beyond_n": len(beyond_plain),
        "beyond_over_threshold": sum(v >= thr for v in beyond_plain),
        "same_decision": None if base is None
        else sum(r["detected"] == base[(r["seed"], r["id"])]["detected"] for r in sw),
    }  # fmt: skip


def rerender_records(cfg: dict, out, rows: list[dict]) -> list[dict]:
    """The renders of homework 2 and the renders made again (emoswitch.rerender): one record per seed and one
    over the seeds of each. Empty when nothing was rendered again."""
    path = out / "rerender" / "rows.jsonl"
    if not path.exists():
        return []
    rc, system = cfg["rerender"], cfg["system"]
    null = next(c for c in rc["conditions"] if c != system)
    again = read_jsonl(path)
    add_derived(again, cfg)
    first = [r for r in rows if r["set"] == rc["set"] and r["condition"] in rc["conditions"]]
    base = {(r["seed"], r["id"]): r for r in first}
    records = []
    for source, group, compare in (("homework 2: CUDA, bfloat16", first, None),
                                   (f"again: {rc['device'].upper()}, float32", again, base)):  # fmt: skip
        seeds = sorted({r["seed"] for r in group})
        for seed in [*seeds, None]:
            g = [r for r in group if seed is None or r["seed"] == seed]
            shared = compare if compare and seed in {k[0] for k in compare} else None
            records.append(run_record(source, ", ".join(map(str, seeds)) if seed is None else str(seed),
                                      [r for r in g if r["condition"] == system],
                                      [r for r in g if r["condition"] == null], cfg, shared))  # fmt: skip
    return records


def rerender_pairs(cfg: dict, out, rows: list[dict]) -> list[dict]:
    """Every two runs of the chosen system among the renders of homework 2 and the renders made again: the clips
    of the same phrase and switch on which the detector gives the same decision."""
    path = out / "rerender" / "rows.jsonl"
    if not path.exists():
        return []
    rc, system = cfg["rerender"], cfg["system"]
    again = read_jsonl(path)
    add_derived(again, cfg)
    runs = {}
    for source, group in (("homework 2", rows), ("again", again)):
        for r in group:
            if r["set"] == rc["set"] and r["condition"] == system:
                runs.setdefault((source, r["seed"]), {})[r["id"]] = r["detected"]
    names = list(runs)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(runs[a]) & set(runs[b]))
            same = sum(runs[a][k] == runs[b][k] for k in shared)
            kind = ("same seed, other hardware" if a[1] == b[1]
                    else "other seed, same hardware" if a[0] == b[0] else "other seed, other hardware")  # fmt: skip
            pairs.append({"a": f"{a[0]}, seed {a[1]}", "b": f"{b[0]}, seed {b[1]}", "kind": kind,
                          "n": len(shared), "same": same})  # fmt: skip
    return pairs


# ── tables ──────────────────────────────────────────────────────────────────


def pair_table(pairs: list[dict]) -> str:
    order = ["same seed, other hardware", "other seed, same hardware", "other seed, other hardware"]
    body = [[p["kind"], p["a"], p["b"], frac(p["same"], p["n"])]
            for p in sorted(pairs, key=lambda p: order.index(p["kind"]))]  # fmt: skip
    return table(["pair of runs", "run", "run", "clips with the same decision"], body)


def rerender_table(records: list[dict], cfg: dict) -> str:
    thr = cfg["detector"]["min_abs_delta"]
    body = [[r["source"], r["seeds"], frac(r["detected"], r["n"]), fmt(r["det"]), ci(r["ci_wilson"]),
             ci(r["ci_phrases"]), *(frac(d["detected"], d["n"]) for d in r["by_direction"].values()),
             "—" if r["same_decision"] is None else frac(r["same_decision"], r["n"]),
             str(r["invalid"]), fmt(r["commanded_step_mean"]), frac(r["null_exceeds"], r["null_n"]),
             frac(r["beyond_over_threshold"], r["beyond_n"]), type_cells(r["types"])] for r in records]  # fmt: skip
    return table(["render", "seed", "detected", "det", "95% Wilson interval", "95% interval over phrases",
                  *(f"detected, switch {d}" for d in DIRECTION.values()),
                  "same decision as the homework 2 clip", "invalid clips", "mean commanded step",
                  "no-switch rows with a step", f"level beyond the second emotion by ≥ {thr}",
                  "misses by type"], body)  # fmt: skip


def type_cells(counts: dict) -> str:
    return ", ".join(f"{t} {counts[t]}" for t in TYPES if counts.get(t)) or "—"


def chosen_table(groups: dict, cfg: dict) -> str:
    body = [[set_name(cfg, s), cfg["system"], frac(g["detected"], g["n"]), fmt(g["det"]), ci(g["ci_wilson"]),
             frac(g["step_present"], g["n"]), type_cells(g["types"])] for s, g in groups.items()]  # fmt: skip
    return table(["set", "system", "detected", "det", "95% Wilson interval",
                  "valid, step of the commanded sign and size", "misses by type"], body)  # fmt: skip


def type_table(columns: dict) -> str:
    """`columns`: header -> list of clip records."""
    counts = {h: Counter(r["type"] for r in recs) for h, recs in columns.items()}
    body = [["clips", *(str(len(recs)) for recs in columns.values())],
            ["detected", *(str(c["detected"]) for c in counts.values())]]  # fmt: skip
    body += [[t, *(str(c[t]) if c[t] else "—" for c in counts.values())] for t in TYPES
             if any(c[t] for c in counts.values())]  # fmt: skip
    return table(["", *columns], body)


def example_table(examples: list[dict], cfg: dict) -> str:
    body = [[e["example_of"], f"[{e['run']}/{e['id']}](../../{e['audio']})", set_name(cfg, e["set"]),
             f"{e['emotion']} → {e['emotion2']}", str(e["seed"]), fmt(e["commanded_step"]),
             "n/a" if e["offset_s"] is None else f"{e['offset_s']:+.2f}",
             "n/a" if e["offset_weighted_s"] is None else f"{e['offset_weighted_s']:+.2f}",
             fmt(e["level_after"]), fmt(e["level_plain"]), fmt(e["wer"], 2), fmt(e["duration_s"], 1)]
            for e in examples]  # fmt: skip
    return table(["example of", "clip", "set", "switch", "seed", "commanded step", "change-point − boundary, s",
                  "the same by the weighted estimator, s", "level after the boundary",
                  "the same in a clip of the second emotion without a switch", "WER", "duration, s"], body)  # fmt: skip


def drift_table(items: list[dict]) -> str:
    body = [[d["emotion"], str(d["n"]), fmt(d["first"]), fmt(d["middle"]), fmt(d["last"]),
             frac(d["first_above_middle"], d["n"]), frac(d["last_below_middle"], d["n"]),
             f"{d['slope_per_s']:+.4f}", f"{d['step_at_boundary']:+.3f}", frac(d["step_negative"], d["n"])]
            for d in items]  # fmt: skip
    return table(["emotion (no switch)", "clips", "first 3 windows", "middle", "last 3 windows",
                  "first above middle", "last below middle", "slope per second",
                  "mean step at the boundary", "step below zero"], body)  # fmt: skip


def placement_table(items: list[dict], cfg: dict) -> str:
    tol = cfg["detector"]["max_changepoint_err_s"]
    body = [[p["direction"], str(p["n"]), frac(p["detector"], p["n"]), frac(p["weighted"], p["n"]),
             frac(p["crossing"], p["n"]), f"{p['crossing_median_s']:+.2f}"] for p in items]  # fmt: skip
    return table(["direction", "valid clips with a step of the commanded sign and size",
                  f"change-point of the detector within {tol} s", f"weighted estimate within {tol} s",
                  f"midpoint crossing within {tol} s", "median crossing − boundary, s"], body)  # fmt: skip


def calibration_table(items: list[dict], cfg: dict) -> str:
    tol = cfg["detector"]["max_changepoint_err_s"]
    body = [[i["group"], str(i["n"]), frac(i["detector"], i["n"]), frac(i["weighted"], i["n"]),
             "n/a" if i["crossing"] is None else frac(i["crossing"], i["n"])] for i in items]  # fmt: skip
    return table(["clips", "n", f"change-point of the detector within {tol} s",
                  f"weighted estimate within {tol} s", f"midpoint crossing within {tol} s"], body)  # fmt: skip


def edge_table(items: list[dict], cfg: dict) -> str:
    k = cfg["detector"]["min_seg_windows"]
    body = [[i["direction"], str(i["n"]), str(i["before"]), str(i["after"]), str(i["short_min"]),
             str(i["short_mid"]), str(i["short_long"])] for i in items]  # fmt: skip
    return table(["direction", "edge_split clips", "detector's split before the boundary", "after the boundary",
                  f"shorter side of the split: {k} windows (the minimum)", f"{k + 1}–{k + 2} windows",
                  f"{k + 3} windows or more"], body)  # fmt: skip


def boundary_table(items: list[dict]) -> str:
    names = list(items[0]["counted"])
    body = [[i["condition"], i["direction"], str(i["n"]), *(frac(i["counted"][n], i["n"]) for n in names),
             fmt(i["pause_median_s"], 2)] for i in items]  # fmt: skip
    return table(["system", "direction", "clips", *(f"boundary = {n}" for n in names), "median pause, s"], body)


def beyond_table(items: list[dict], cfg: dict) -> str:
    thr = cfg["detector"]["min_abs_delta"]
    body = [[i["condition"], i["emotion2"], str(i["n"]), fmt(i["level_before"]), fmt(i["level_after"]),
             fmt(i["level_plain"]), f"{i['beyond_mean']:+.3f}", frac(i["beyond_over_threshold"], i["n"]),
             frac(i["short_over_threshold"], i["n"])] for i in items]  # fmt: skip
    return table(["system", "second emotion", "valid clips", "level before the boundary", "level after",
                  "level after the boundary in a clip of the second emotion without a switch",
                  "after − without a switch, in the commanded direction", f"beyond by ≥ {thr}",
                  f"short by ≥ {thr}"], body)  # fmt: skip


def invalid_table(inv: dict, cfg: dict) -> str:
    body = [[set_name(cfg, x), frac(*v["anchor"]), frac(*v["no_anchor"])] for x, v in inv["sets"].items()]
    body.append(["all sets", frac(*inv["all"]["anchor"]), frac(*inv["all"]["no_anchor"])])
    return (table(["set", "invalid clips, systems with the anchor", "invalid clips, systems without it"], body)
            + f"\n\nFisher's exact test over the sets, two-sided: p = {inv['fisher_p']:.3f}.")  # fmt: skip


def cell_table(cells: dict) -> str:
    n = cells["n_cells"]
    body = [[f"{k} of 3", frac(v, n)] for k, v in cells["by_seeds_detected"].items()]
    text = table(["seeds on which the cell is counted", "cells (phrase × switch)"], body)
    never = "; ".join(f"{c['phrase_id']} {c['switch']} ({', '.join(c['types'])})" for c in cells["never"])
    return text + f"\n\nNever counted: {never}."


def slice_table(items: list[dict]) -> str:
    body = [[i["slice"], i["group"], str(i["n"]), frac(i["detected"], i["n"]), fmt(i["det"]), ci(i["ci_wilson"]),
             frac(i["step_present"], i["n"]), type_cells(i["types"])] for i in items]  # fmt: skip
    return table(["slice", "group", "clips", "detected", "det", "95% Wilson interval",
                  "valid, step of the commanded sign and size", "misses by type"], body)  # fmt: skip


def cross_table(c: dict, a: str) -> str:
    def cell(i: str, j: str) -> str:
        g = c["cells"].get(f"{i}|{j}")
        return f"{g['detected']}/{g['n']} ({g['det']:.2f})" if g else "—"

    return table([a, *c["cols"]], [[i, *(cell(i, j) for j in c["cols"])] for i in c["rows"]])


def noise_table(levels: list[dict]) -> str:
    body = [["released clip" if lv["snr_db"] is None else f"{lv['snr_db']:g} dB", frac(lv["detected"], lv["n"]),
             fmt(lv["det"]), ci(lv["ci_phrases"]), str(lv["lost"]), str(lv["gained"]),
             frac(lv["step_present"], lv["n"]), fmt(lv["commanded_step_mean"]), str(lv["invalid"]),
             fmt(lv["wer_mean"], 4), fmt(lv["profile_sd_mean"]), frac(lv["null_exceeds"], lv["null_n"]),
             str(lv["null_changed"]), fmt(lv["boundary_moved_median_s"]), type_cells(lv["types"])]
            for lv in levels]  # fmt: skip
    return table(["signal-to-noise ratio", "detected", "det", "95% interval over phrases", "decisions lost",
                  "decisions gained", "valid, step of the commanded sign and size", "mean commanded step",
                  "invalid clips", "mean WER", "mean spread of the profile inside a clip",
                  "no-switch rows with a step", "no-switch rows, decision changed",
                  "median shift of the boundary, s", "misses by type"], body)  # fmt: skip


def resample_table(res: dict) -> str:
    def line(label: str, recs: list[dict]) -> list[str]:
        dets = [r["det"] for r in recs]
        q05, q95 = (float(q) for q in np.quantile(dets, [0.05, 0.95]))
        return [label, str(len(recs)), fmt(min(dets)), fmt(q05), fmt(float(np.median(dets))), fmt(q95),
                fmt(max(dets)), frac(sum(d >= 0.5 for d in dets), len(recs)),
                frac(sum(r["above_each_component"] for r in recs), len(recs))]  # fmt: skip

    h = res["halves"]
    body = [
        line("all seeds, all phrases", [res["all"]]),
        line("one seed", res["seeds"]),
        line("all phrases but one", res["leave_one_phrase_out"]),
        [f"half of the phrases ({h['phrases_per_subset']})", str(h["n_subsets"]), fmt(h["min"]), fmt(h["q05"]),
         fmt(h["median"]), fmt(h["q95"]), fmt(h["max"]), frac(h["at_least_half"], h["n_subsets"]),
         frac(h["above_each_component"], h["n_subsets"])],
    ]  # fmt: skip
    return table(["subset", "subsets", "lowest det", "5%", "median det", "95%", "highest det", "det ≥ 0.50",
                  "above each single-component system"], body)  # fmt: skip


def subset_table(res: dict) -> str:
    body = [[r["subset"], frac(r["detected"], r["n"]), fmt(r["det"]), "yes" if r["above_each_component"] else "no"]
            for r in (*res["seeds"], *res["leave_one_phrase_out"])]  # fmt: skip
    return table(["subset", "detected", "det", "above each single-component system"], body)


# ── figures ─────────────────────────────────────────────────────────────────


def fig_examples(examples: list[dict], rows_by_key: dict, typical: dict, cfg: dict, path) -> None:
    """Arousal profile of a counted clip and of one clip per kind of miss, with the boundary and both
    change-point estimates."""
    import matplotlib.pyplot as plt

    tol = cfg["detector"]["max_changepoint_err_s"]
    panels = [{**typical, "example_of": "detected (median step)"}, *examples]
    n_cols = 4
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.1 * n_rows), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, e in zip(axes.flat, panels, strict=False):
        row = rows_by_key[(e["run"], e["id"])]
        step = primary_step(row)
        c, a = np.asarray(row["profile_centres_s"]), np.asarray(row["profile_arousal"])
        color = TYPE_COLOR[e["type"]]
        ax.plot(c, a, color=color, linewidth=1.8, marker="o", markersize=3)
        if step:
            t = step["switch_time_s"]
            ax.axvspan(t - tol, t + tol, color="#eae9e3", zorder=0)
            ax.axvline(t, color=INK, linewidth=1.2)
            ax.hlines([step["mean_before"], step["mean_after"]], [c[0], t], [t, c[-1]], color=INK2, linewidth=0.9,
                      linestyle=":")  # fmt: skip
            if e["level_plain"] is not None:
                ax.hlines(e["level_plain"], t, c[-1], color="#a23b3b", linewidth=1.1, linestyle="-.")
            if e["offset_s"] is not None:
                ax.axvline(t + e["offset_s"], color=color, linewidth=1.4, linestyle="--")
                ax.axvline(t + e["offset_weighted_s"], color=color, linewidth=1.4, linestyle=":")
        ax.set_title(f"{e['example_of']}\n{e['emotion']} → {e['emotion2']}, {e['phrase_id']}, seed {e['seed']}"
                     f" ({set_name(cfg, e['set'])})", loc="left", fontsize=9, color=INK)  # fmt: skip
        ax.set_xlabel("time, s", color=INK2, fontsize=9)
        ax.set_ylim(-0.2, 1)
        _axes(ax)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    for r in range(n_rows):
        axes[r][0].set_ylabel("arousal (judge)", color=INK2, fontsize=9)
    note = (f"Black line: sentence boundary; grey band: ±{tol} s, where the detector accepts the change-point. "
            "Dashed: change-point of the detector; dotted vertical: weighted estimate.\n"
            "Dotted horizontal: mean level before and after the boundary. Dash-dotted: level after the boundary "
            "in a clip of the second emotion without a switch (same phrase and seed).")  # fmt: skip
    fig.text(0.01, 0.01, note, color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_offsets(recs: list[dict], cfg: dict, path) -> None:
    """Where the two estimators put the change-point of every clip with a step of the right size and sign."""
    import matplotlib.pyplot as plt

    tol = cfg["detector"]["max_changepoint_err_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, direction in zip(axes, ("up", "down"), strict=True):
        ax.axvspan(-tol, tol, color="#eae9e3", zorder=0)
        ax.axhspan(-tol, tol, color="#eae9e3", zorder=0, alpha=0.6)
        g = [r for r in recs if r["direction"] == direction and r["type"] in STEP_PRESENT]
        for t in STEP_PRESENT:
            pts = [r for r in g if r["type"] == t]
            if pts:
                ax.scatter([r["offset_s"] for r in pts], [r["offset_weighted_s"] for r in pts], s=26,
                           color=TYPE_COLOR[t], edgecolor=SURFACE, linewidth=0.5, label=f"{t} ({len(pts)})",
                           zorder=3)  # fmt: skip
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.set_title(f"switch {direction} ({len(g)} clips with a step)", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("change-point of the detector − boundary, s", color=INK2, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        _axes(ax)
    axes[0].set_ylabel("weighted estimate − boundary, s", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, f"Grey bands: within {tol} s of the boundary. The detector counts a clip inside the "
             "vertical band.", color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_slices(items: list[dict], cfg: dict, path) -> None:
    """det of every group of every slice with its Wilson interval."""
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(i["slice"] for i in items))
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(2.9 * len(names), 4.4),
        sharey=True,
        gridspec_kw={"width_ratios": [max(2, sum(i["slice"] == n for i in items)) for n in names]},
    )
    fig.patch.set_facecolor(SURFACE)
    color = COLOR[cfg["system"]]
    for ax, name in zip(axes, names, strict=True):
        g = [i for i in items if i["slice"] == name]
        for x, i in enumerate(g):
            ax.bar(x, i["det"], width=0.62, color=color, alpha=0.35, edgecolor=color)
            ax.plot([x, x], i["ci_wilson"], color=color, linewidth=2)
            ax.text(x, 1.02, str(i["n"]), ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(g)), [i["group"].replace(" → ", "\n→ ") for i in g], fontsize=7.5, rotation=40,
                      ha="right")  # fmt: skip
        ax.set_ylim(0, 1.1)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK)
        _axes(ax)
    axes[0].set_ylabel("detection rate", color=INK2, fontsize=10)
    fig.text(0.01, 0.01, "Line: 95% Wilson interval. Number above a bar: clips in the group.", color=INK2,
             fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_noise(levels: list[dict], cfg: dict, path) -> None:
    """det, false steps, invalid clips and the size of the step against the level of the added noise."""
    import matplotlib.pyplot as plt

    labels = ["released\nclip" if lv["snr_db"] is None else f"{lv['snr_db']:g} dB" for lv in levels]
    x = np.arange(len(levels))
    color = COLOR[cfg["system"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    ax.plot(x, [lv["det"] for lv in levels], color=color, linewidth=2, marker="o", label="det, chosen system")
    for xi, lv in zip(x, levels, strict=True):
        ax.plot([xi, xi], lv["ci_phrases"], color=color, linewidth=2, alpha=0.5)
    ax.plot(x, [lv["null_exceeds"] / lv["null_n"] for lv in levels], color="#898781", linewidth=1.8, marker="s",
            label="no-switch rows with a step")  # fmt: skip
    ax.plot(x, [lv["invalid"] / lv["n"] for lv in levels], color=TYPE_COLOR["invalid"], linewidth=1.8, marker="^",
            linestyle="--", label="invalid clips, chosen system")  # fmt: skip
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of clips", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    ax = axes[1]
    ax.plot(x, [lv["commanded_step_mean"] for lv in levels], color=color, linewidth=2, marker="o",
            label="mean commanded step")  # fmt: skip
    ax.plot(x, [lv["profile_sd_mean"] for lv in levels], color=INK2, linewidth=1.8, marker="s",
            label="mean spread of the profile inside a clip")  # fmt: skip
    ax.axhline(cfg["detector"]["min_abs_delta"], color=AXIS, linewidth=1, linestyle=":")
    ax.set_ylim(0, None)
    ax.set_ylabel("arousal units", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_xlabel("signal-to-noise ratio of the added white noise", color=INK2, fontsize=10)
        _axes(ax)
    fig.text(0.01, 0.01, "Left: vertical line = 95% bootstrap interval over phrases. Right: dotted line = the "
             "detector's threshold on the step.", color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_resample(res: dict, cfg: dict, path) -> None:
    """det over halves of the phrases, with single seeds and leave-one-phrase-out subsets marked."""
    import matplotlib.pyplot as plt

    color = COLOR[cfg["system"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor(SURFACE)
    ax.hist(res["halves"]["values"], bins=np.arange(0.4, 1.0001, 0.025), color=color, alpha=0.35, edgecolor=color)
    top = ax.get_ylim()[1]
    ax.scatter([r["det"] for r in res["leave_one_phrase_out"]], [top * 0.9] * len(res["leave_one_phrase_out"]),
               marker="|", s=160, color=INK, label="all phrases but one (12)")  # fmt: skip
    ax.scatter([r["det"] for r in res["seeds"]], [top * 0.8] * len(res["seeds"]), marker="o", s=34, color="#eb6834",
               label="one seed (3)", zorder=3)  # fmt: skip
    ax.axvline(res["all"]["det"], color=color, linewidth=2, label="all seeds, all phrases")
    ax.axvline(0.5, color=AXIS, linewidth=1, linestyle=":")
    ax.set_xlabel("detection rate on the subset", color=INK2, fontsize=10)
    ax.set_ylabel(f"halves of the phrases ({res['halves']['n_subsets']})", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_rerender(records: list[dict], cfg: dict, path) -> None:
    """det of every render of the chosen system: homework 2 and the renders made again."""
    import matplotlib.pyplot as plt

    sources = list(dict.fromkeys(r["source"] for r in records))
    colors = dict(zip(sources, (COLOR[cfg["system"]], "#eb6834"), strict=False))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    fig.patch.set_facecolor(SURFACE)
    for x, r in enumerate(records):
        color = colors[r["source"]]
        pooled = "," in r["seeds"]
        ax.bar(x, r["det"], width=0.62, color=color, alpha=0.6 if pooled else 0.3, edgecolor=color)
        ax.plot([x, x], r["ci_phrases"] if pooled else r["ci_wilson"], color=color, linewidth=2)
        ax.text(x, 1.02, f"{r['detected']}/{r['n']}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle=":")
    ax.set_xticks(range(len(records)), [("all seeds" if "," in r["seeds"] else f"seed {r['seeds']}") for r in records],
                  fontsize=8.5)  # fmt: skip
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("detection rate", color=INK2, fontsize=10)
    for source in sources:
        ax.bar([0], [0], color=colors[source], alpha=0.5, label=source)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _axes(ax)
    fig.text(0.01, 0.01, "Line: 95% Wilson interval of a seed, bootstrap interval over phrases for all seeds.",
             color=INK2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

CLIP_FIELDS = ["run", "id", "audio", "set", "seed", "phrase_id", "emotion", "emotion2", "direction", "detected",
               "type", "commanded_step", "offset_s", "offset_weighted_s", "crossing_s", "short_side_windows",
               "level_before", "level_after", "level_plain", "beyond_plain", "seconds_before", "seconds_after",
               "valid", "wer", "duration_s"]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/diagnostics.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(cfg, ROOT / cfg["source"])
    system, s, ref = cfg["system"], cfg["analysis_set"], cfg["reference"]
    null = next(c for c in cfg["noise"]["conditions"] if c != system)
    plain = null_levels(rows, null)
    switch = {c: [describe(r, cfg, plain) for r in rows if r["condition"] == c]
              for c, spec in cfg["systems"].items() if c == ref or spec.get("role") == "inpass"}  # fmt: skip
    chosen = switch[system]
    recs = [r for r in chosen if r["set"] == s]
    other_sets = [x for x in dict.fromkeys(r["set"] for r in chosen) if x != s]
    rows_by_key = {(r["run"], r["id"]): r for r in rows if r["condition"] == system}

    groups = {x: group_stats([r for r in chosen if r["set"] == x], cfg) for x in (s, *other_sets)}
    columns = {f"{system}, {set_name(cfg, s)}": recs}
    columns.update({f"— switch {d}": [r for r in recs if r["direction"] == d] for d in ("up", "down")})
    columns.update({f"{system}, {set_name(cfg, x)}": [r for r in chosen if r["set"] == x] for x in other_sets})
    for c in (ref, *(c for c in switch if c not in (system, ref))):
        in_set = [r for r in switch[c] if r["set"] == s]
        columns[f"{c}, {set_name(cfg, s)}"] = in_set
        if c == ref:
            columns.update({f"— {ref}, switch {d}": [r for r in in_set if r["direction"] == d] for d in ("up", "down")})
    examples = pick_examples(recs)
    kinds = {e["type"] for e in examples}
    examples[-1:-1] = [e for e in pick_examples([r for r in chosen if r["set"] != s]) if e["type"] not in kinds]
    counted = sorted((r for r in recs if r["detected"]), key=lambda r: r["commanded_step"])
    typical = counted[len(counted) // 2]
    drifts = drift([r for r in rows if r["condition"] == null and r["set"] == s])
    places = placement(recs, cfg)
    known = calibration(rows, cfg, null)
    edges = edge_sides(recs, cfg)
    inv = invalid_by_anchor(rows, cfg)
    cells = seed_cells(recs)
    compared = (system, ref, *(c for c in switch if c not in (system, ref)))
    by_system = {c: [r for r in switch[c] if r["set"] == s] for c in compared}
    by_system.update({f"{system}, {set_name(cfg, x)}": [r for r in chosen if r["set"] == x] for x in other_sets})
    levels_after = beyond(by_system, cfg)
    readings = boundary_readings([r for r in rows if r["condition"] in (system, ref) and r["set"] == s], cfg)
    items = slices(recs, cfg)
    by_time = cross(recs, cfg, "direction", "seconds after the boundary")
    by_before = cross(recs, cfg, "direction", "seconds before the boundary")
    elsewhere = [i for x in other_sets for i in slices([r for r in chosen if r["set"] == x], cfg)
                 if i["slice"] in ("direction", "switch")]  # fmt: skip
    renders = rerender_records(cfg, out, rows)
    pairs = rerender_pairs(cfg, out, rows)
    levels = noise_levels(cfg, out, rows)
    res = resample(rows, cfg)

    write_csv(out / "clips.csv", chosen, CLIP_FIELDS)
    flat = [{**i, "ci_wilson": ci(i["ci_wilson"]), "types": json.dumps(i["types"], sort_keys=True)} for i in items]
    write_csv(out / "slices.csv", flat, ["slice", "group", "n", "detected", "det", "ci_wilson", "step_present",
                                         "types"])  # fmt: skip
    flat = [
        {
            **lv,
            "snr_db": "clean" if lv["snr_db"] is None else lv["snr_db"],
            "ci_wilson": ci(lv["ci_wilson"]),
            "ci_phrases": ci(lv["ci_phrases"]),
            "types": json.dumps(lv["types"], sort_keys=True),
        }
        for lv in levels
    ]
    write_csv(out / "noise.csv", flat, ["snr_db", "n", "detected", "det", "ci_wilson", "ci_phrases", "lost",
              "gained", "step_present", "commanded_step_mean", "invalid", "wer_mean", "null_n", "null_exceeds",
              "null_changed", "null_invalid", "boundary_moved_median_s", "profile_sd_mean", "types"])  # fmt: skip
    write_csv(out / "resample.csv", [res["all"], *res["seeds"], *res["leave_one_phrase_out"]],
              ["subset", "detected", "n", "det", "above_each_component"])  # fmt: skip
    record = {"system": system, "set": s, "groups": groups, "examples": examples, "drift": drifts, "placement": places,
              "calibration": known, "edge_sides": edges, "direction_by_seconds_before": by_before,
              "other_sets": elsewhere, "rerender": renders, "rerender_pairs": pairs,
              "boundary_readings": readings, "beyond": levels_after,
              "invalid_by_anchor": inv, "seed_cells": cells, "slices": items,
              "direction_by_seconds_after": by_time, "noise": levels, "resample": res}  # fmt: skip
    (out / "diagnostics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig_examples(examples, rows_by_key, typical, cfg, out / "fig_error_examples.png")
    fig_offsets(recs, cfg, out / "fig_offsets.png")
    fig_slices([i for i in items if i["slice"] != "phrase"], cfg, out / "fig_slices.png")
    fig_noise(levels, cfg, out / "fig_noise.png")
    fig_resample(res, cfg, out / "fig_resample.png")
    if renders:
        fig_rerender(renders, cfg, out / "fig_rerender.png")
    sections = [
        ("Chosen system", chosen_table(groups, cfg)),
        ("Misses by type", type_table(columns)),
        ("Examples", example_table(examples, cfg)),
        ("Where the level changes, three readings", placement_table(places, cfg)),
        ("The three readings where the answer is known", calibration_table(known, cfg)),
        ("Edge splits", edge_table(edges, cfg)),
        ("Boundary read elsewhere in the pause", boundary_table(readings)),
        ("Level inside a clip without a switch", drift_table(drifts)),
        ("Level after the switch against a clip of the second emotion", beyond_table(levels_after, cfg)),
        ("Invalid clips and the anchor", invalid_table(inv, cfg)),
        ("Slices", slice_table([i for i in items if i["slice"] != "phrase"])),
        ("By phrase", slice_table([i for i in items if i["slice"] == "phrase"])),
        ("Direction by time after the boundary", cross_table(by_time, "direction")),
        ("Direction by time before the boundary", cross_table(by_before, "direction")),
        ("Direction and switch on the other set", slice_table(elsewhere)),
        ("Noise added to the clip", noise_table(levels)),
        ("Seeds and subsets of phrases", resample_table(res)),
        ("Single seeds and phrases left out", subset_table(res)),
        ("Cells by seeds", cell_table(cells)),
        ("Rendered again", rerender_table(renders, cfg) if renders else "Not rendered."),
        ("Decisions of two runs", pair_table(pairs) if pairs else "Not rendered."),
    ]
    text = "# Chosen system: diagnostics\n\nGenerated by `python -m emoswitch.diagnose`; do not edit by hand.\n"
    text += "".join(f"\n## {title}\n\n{body}\n" for title, body in sections)
    (out / "diagnostics.md").write_text(text, encoding="utf-8")
    print(f"[diagnose] {len(recs)} clips of {system} on {s}, {len(levels) - 1} noise levels -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Evaluate released audio: measure every clip, then score the systems.

    python -m emoswitch.evaluate --config configs/baseline.yaml              # audio -> rows -> metrics
    python -m emoswitch.evaluate --config configs/baseline.yaml --from-rows  # rows -> metrics, no models

`measure` needs the judge, the aligner and the ASR weights (downloaded on first use, CPU is enough).
It only measures the clips of the manifest that rows.jsonl does not hold yet (same run, id and audio
hash, same measurement settings), so adding runs does not re-measure the released ones.
`score` is pure arithmetic over the per-clip rows and runs in seconds.
Relative paths are resolved against the repository root, not the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from emoswitch import detect, gates, stats
from emoswitch.config import ROOT, load_config, set_role
from emoswitch.stimuli import NULL_CONDITIONS

SWITCH_CONDITIONS = ("nl_instruct", "stitch")
# Where a stitched clip's boundary is read (config `primary_rule`). A single-pass clip has one boundary only,
# the forced-aligned switch word, under both rules.
BOUNDARIES = ("construction", "forced_alignment")
# Config sections a per-clip row depends on. Scoring-only constants (thresholds, gates, intervals) are not here.
MEASURE_KEYS = ("judge", "asr")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def measure_fingerprint(cfg: dict) -> str:
    """Hash of everything in the config that a measured row depends on."""
    parts = {k: cfg[k] for k in MEASURE_KEYS}
    parts["detector"] = {k: cfg["detector"][k] for k in ("min_seg_windows", "switch_search_from_frac")}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def row_key(row: dict) -> tuple:
    """A measured row belongs to one audio file of one run."""
    return row["run"], row["id"], row["sha256_wav_source"]


def split_measured(manifest: list[dict], rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(rows already measured for clips of the manifest, manifest entries still to measure)."""
    have = {row_key(r): r for r in rows}
    return [have[row_key(m)] for m in manifest if row_key(m) in have], [m for m in manifest if row_key(m) not in have]


# ── measure: one clip -> one row ────────────────────────────────────────────


def aligned_switch(aligner, y: np.ndarray, row: dict, duration: float, dc: dict) -> tuple[list[dict], float | None]:
    """Forced-aligned words of the known text and the start of the switch word among them."""
    words = aligner(y, row["text"])
    return words, detect.switch_time(words, row["switch_word"], duration, dc["switch_search_from_frac"])


def measure(cfg: dict, manifest: list[dict], device: str) -> list[dict]:
    from emoswitch.align import ForcedAligner
    from emoswitch.asr import load_asr, transcribe
    from emoswitch.judge import ArousalJudge, load_audio, window_profile

    jc, dc = cfg["judge"], cfg["detector"]
    judge = ArousalJudge(jc["model"], jc["revision"], jc["sample_rate"], device)
    aligner = ForcedAligner(jc["sample_rate"], device)
    asr = load_asr(cfg["asr"], device)
    keep = ("run", "set", "seed", "id", "condition", "emotion", "emotion2", "phrase_id", "text", "instruct")
    rows = []
    for i, m in enumerate(manifest, 1):
        path = str(ROOT / m["audio"])
        y = load_audio(path, jc["sample_rate"], jc["lufs"])
        duration = len(y) / jc["sample_rate"]
        centres, arousal = window_profile(y, judge, jc["sample_rate"], jc["window_s"], jc["hop_s"])
        words, t_aligned = aligned_switch(aligner, y, m, duration, dc)
        aligned_step = detect.measure_step(centres, arousal, t_aligned, m["expect_sign"], dc["min_seg_windows"])
        row = {
            **{k: m[k] for k in keep},
            "switch_word": m["switch_word"],
            "expect_sign": m["expect_sign"],
            "audio": m["audio"],
            "sha256_wav_source": m["sha256_wav_source"],
            "duration_s": round(duration, 3),
            "profile_centres_s": centres,
            "profile_arousal": arousal,
            "profile_sd": round(float(np.std(arousal)), 4) if arousal else None,
            "hyp": transcribe(asr, path, cfg["asr"]),
        }
        if m.get("switch_time_s") is None:  # single pass: the boundary is the forced-aligned switch word
            row.update(words=words, switch_time_source="forced_alignment", step=aligned_step)
        else:  # stitched: `step` at the boundary by construction, `step_aligned` at the aligned switch word
            step = detect.measure_step(centres, arousal, m["switch_time_s"], m["expect_sign"], dc["min_seg_windows"])
            row.update(words=[], switch_time_source="construction", step=step)
            row.update(words_aligned=words, step_aligned=aligned_step)
        rows.append(row)
        print(
            f"[measure] {i}/{len(manifest)} {m['run']}/{m['id']}: "
            f"delta={row['step'] and row['step']['delta']} cp_err={row['step'] and row['step']['changepoint_err_s']}",
            flush=True,
        )
    return rows


def align_stitched(cfg: dict, rows: list[dict], device: str) -> int:
    """Add the forced-alignment step to measured stitched rows that lack it. The stored profile is reused,
    so only the aligner runs. Returns the number of rows completed."""
    todo = [r for r in rows if r["switch_time_source"] == "construction" and "step_aligned" not in r]
    if not todo:
        return 0
    from emoswitch.align import ForcedAligner
    from emoswitch.judge import load_audio

    jc, dc = cfg["judge"], cfg["detector"]
    aligner = ForcedAligner(jc["sample_rate"], device)
    for i, r in enumerate(todo, 1):
        y = load_audio(str(ROOT / r["audio"]), jc["sample_rate"], jc["lufs"])
        words, t = aligned_switch(aligner, y, r, len(y) / jc["sample_rate"], dc)
        r["words_aligned"] = words
        r["step_aligned"] = detect.measure_step(
            r["profile_centres_s"], r["profile_arousal"], t, r["expect_sign"], dc["min_seg_windows"]
        )
        print(f"[align] {i}/{len(todo)} {r['run']}/{r['id']}: boundary {t}", flush=True)
    return len(todo)


# ── score: rows -> metrics ──────────────────────────────────────────────────


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def step_for(row: dict, boundary: str) -> dict | None:
    """The row's step with a stitched clip's boundary read by `boundary` (one of BOUNDARIES)."""
    if row["switch_time_source"] == "forced_alignment" or boundary == "construction":
        return row["step"]
    return row["step_aligned"]


def primary_step(row: dict) -> dict | None:
    """The step under the primary rule of the row's set (written into the row by `add_derived`)."""
    return step_for(row, row["boundary"])


def primary_rules(cfg: dict, rows: list[dict]) -> dict:
    """Set name -> its primary rule (config `primary_rule` of the set's role)."""
    return {s: cfg["primary_rule"][set_role(cfg, s)] for s in dict.fromkeys(r["set"] for r in rows)}


def clip_leak_stems(cfg: dict, row: dict) -> list[str]:
    """Instruction word stems of the row's emotions, as `stimuli.build_rows` lists them."""
    return [s for e in dict.fromkeys((row["emotion"], row["emotion2"])) if e for s in cfg["leak_stems"][e]]


def is_valid(row: dict, gc: dict) -> bool:
    """A clip is valid when its speech is the text: WER within bounds, no spoken instruction, not looped."""
    return row["wer"] <= gc["max_clip_wer"] and not row["leak"] and not row["looped"]


def analysis_group(row: dict) -> str | None:
    """The condition a row is counted under in tables that count each clip once. The no-switch rows of a
    stitch run repeat those of the inline run of the same set and seed (byte for byte on the dev and val
    sets), so they are left out there (None)."""
    if row["condition"] in ("control", "plain") and run_kind(row) != "inline":
        return None
    return row["condition"]


def decision(row: dict) -> bool:
    """The row's primary decision: detected for a switch row, exceeds the null test for a no-switch row."""
    return row["detected"] if row["condition"] in SWITCH_CONDITIONS else row["exceeds_null"]


def outcome(step: dict | None, detected: bool, min_abs_delta: float, gated_out: bool = False) -> str:
    """Why a switch row was or was not counted. The checks are ordered: the first failed one names the miss,
    so a row below the size threshold is `flat` whatever its sign."""
    if detected:
        return "detected"
    if gated_out:
        return "invalid"  # the clip fails the validity rule of its set
    if step is None:
        return "no_step"  # boundary word not located, or the clip is too short for a step
    if abs(step["delta"]) < min_abs_delta:
        return "flat"
    if not step["sign_ok"]:
        return "wrong_sign"
    if step["changepoint_err_s"] is None:
        return "no_changepoint"  # too few windows for a split with min_seg_windows on each side
    return "misplaced"  # a step of the right size and sign whose change-point is away from the boundary


def add_derived(rows: list[dict], cfg: dict) -> None:
    """Speech checks, per-row decisions and the control-derived reference levels, written into the rows."""
    dc, gc = cfg["detector"], cfg["gates"]
    thr, tol = dc["min_abs_delta"], dc["max_changepoint_err_s"]
    rules = primary_rules(cfg, rows)
    controls = {(r["run"], r["phrase_id"], r["emotion"]): r for r in rows if r["condition"] == "control"}
    plains = {(r["run"], r["phrase_id"]): r for r in rows if r["condition"] == "plain"}
    for r in rows:
        r["wer"] = gates.word_error_rate(r["text"], r["hyp"])
        r["leak"] = gates.leaks(r["hyp"], r["text"], clip_leak_stems(cfg, r))
        r["looped"] = r["duration_s"] > gc["loop_threshold_s"]
        r["valid"] = is_valid(r, gc)
        rule = rules[r["set"]]
        r["boundary"] = rule["boundary"]
        step = primary_step(r)
        test = detect.exceeds_null if r["condition"] in NULL_CONDITIONS else detect.is_detected
        r["decisions"] = {b: test(step_for(r, b), thr, tol) for b in BOUNDARIES}
        if r["condition"] in NULL_CONDITIONS:
            r["exceeds_null"] = r["decisions"][rule["boundary"]]
            continue
        gated_out = rule["validity_gate"] and not r["valid"]
        r["detected"] = r["decisions"][rule["boundary"]] and not gated_out
        r["outcome"] = outcome(step, r["detected"], thr, gated_out)
        r["commanded_step"] = None if step is None else round(r["expect_sign"] * step["delta"], 4)
        c1 = controls.get((r["run"], r["phrase_id"], r["emotion"]))
        c2 = controls.get((r["run"], r["phrase_id"], r["emotion2"]))
        r["ideal_step"] = r["nearest_level"] = None
        if c1 and c2 and c1["step"] and c2["step"]:
            # what a perfect switch would show: sentence 1 as in the e1 control, sentence 2 as in the e2 control
            ideal = c2["step"]["mean_after"] - c1["step"]["mean_before"]
            r["ideal_step"] = round(r["expect_sign"] * ideal, 4)
            # which whole-phrase render of the same phrase the clip's mean arousal is nearest to
            level = float(np.mean(r["profile_arousal"]))
            candidates = {r["emotion"]: c1, r["emotion2"]: c2}
            if (r["run"], r["phrase_id"]) in plains:
                candidates["plain"] = plains[(r["run"], r["phrase_id"])]
            r["nearest_level"] = min(
                candidates, key=lambda name: abs(level - float(np.mean(candidates[name]["profile_arousal"])))
            )


def gate_check(rows: list[dict], cfg: dict) -> dict:
    """System-level speech gate over a group of rows. Rows are not dropped: a failed gate means the
    group's detection rate should not be compared at all."""
    gc = cfg["gates"]
    wer = float(np.mean([r["wer"] for r in rows]))  # the system gate; clip validity is `is_valid`
    checks = {
        "wer_mean_le_max": wer <= gc["max_mean_wer"],
        "no_loops": not any(r["looped"] for r in rows),
        "no_leak": not any(r["leak"] for r in rows),
    }
    return {**checks, "passed": all(checks.values())}


def summarise_switch(rows: list[dict], cfg: dict) -> dict:
    """Aggregates of one group of switch rows (one system; any slice of sets, seeds, pairs)."""
    sc = cfg["stats"]
    n, k = len(rows), sum(r["detected"] for r in rows)
    steps = [s for s in map(primary_step, rows) if s]
    paired = [r for r in rows if r["commanded_step"] is not None and r["ideal_step"] is not None]
    lo, hi = stats.wilson_ci(k, n, sc["ci"])
    blo, bhi = stats.cluster_bootstrap_ci(
        [float(r["detected"]) for r in rows],
        [f"{r['set']}/{r['phrase_id']}" for r in rows],
        sc["n_boot"],
        sc["seed"],
        sc["ci"],
    )
    ideal_sum = sum(r["ideal_step"] for r in paired)
    tol = cfg["detector"]["max_changepoint_err_s"]
    per_phrase = defaultdict(lambda: [0, 0])
    for r in rows:
        per_phrase[f"{r['set']}/{r['phrase_id']}"][0] += r["detected"]
        per_phrase[f"{r['set']}/{r['phrase_id']}"][1] += 1
    return {
        "n": n,
        "n_with_step": len(steps),
        "detected": k,
        "det_rate": k / n if n else None,
        "det_ci_wilson": [lo, hi],
        "det_ci_phrase_bootstrap": [blo, bhi],
        "n_phrases": len({(r["set"], r["phrase_id"]) for r in rows}),
        "detected_by_rule": {
            f"{b}{'+valid' if gated else ''}": sum(r["decisions"][b] and (r["valid"] or not gated) for r in rows)
            for b in BOUNDARIES
            for gated in (False, True)
        },
        "n_invalid": sum(not r["valid"] for r in rows),
        "per_phrase": dict(per_phrase),
        "abs_delta_mean": _mean([abs(s["delta"]) for s in steps]),
        "abs_delta_max": max(abs(s["delta"]) for s in steps) if steps else None,
        "commanded_step_mean": _mean([r["commanded_step"] for r in rows]),
        "sign_ok_rate": _mean([float(s["sign_ok"]) for s in steps]),
        "changepoint_within_tol_rate": _mean(
            [float(s["changepoint_err_s"] <= tol) for s in steps if s["changepoint_err_s"] is not None]
        ),
        "ideal_step_mean": _mean([r["ideal_step"] for r in paired]),
        "realised_share": sum(r["commanded_step"] for r in paired) / ideal_sum if ideal_sum > 0 else None,
        "wer_mean": _mean([r["wer"] for r in rows]),
        "wer_median": float(np.median([r["wer"] for r in rows])) if rows else None,
        "leak_rate": _mean([float(r["leak"]) for r in rows]),
        "looped": int(sum(r["looped"] for r in rows)),
        "gates": gate_check(rows, cfg) if rows else None,
    }


def summarise_null(rows: list[dict], cfg: dict) -> dict:
    """What the detector reads on rows where no switch was asked for (the false-step rate)."""
    steps = [s for s in map(primary_step, rows) if s]
    k = sum(r["exceeds_null"] for r in rows)
    lo, hi = stats.wilson_ci(k, len(rows), cfg["stats"]["ci"])
    abs_d = [abs(s["delta"]) for s in steps]
    return {
        "n": len(rows),
        "n_with_step": len(steps),
        "abs_delta_mean": _mean(abs_d),
        "abs_delta_p95": float(np.percentile(abs_d, 95)) if abs_d else None,
        "abs_delta_max": float(np.max(abs_d)) if abs_d else None,
        "exceed": k,
        "exceed_rate": k / len(rows) if rows else None,
        "exceed_ci_wilson": [lo, hi],
        "exceed_by_boundary": {b: sum(r["decisions"][b] for r in rows) for b in BOUNDARIES},
        "n_invalid": sum(not r["valid"] for r in rows),
        "profile_sd_mean": _mean([r["profile_sd"] for r in rows]),
        "wer_mean": _mean([r["wer"] for r in rows]),
        "leak_rate": _mean([float(r["leak"]) for r in rows]),
        "looped": int(sum(r["looped"] for r in rows)),
    }


def _group(rows: list[dict], *keys: str) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return dict(out)


def run_kind(row: dict) -> str:
    return row["run"].split("_")[0]


def paired_comparison(rows: list[dict], systems: dict) -> dict:
    """Reference against baseline, per set. Row level: exact McNemar test on the rows both systems have
    (same seed, phrase and commanded switch). Phrase level, the unit of inference (docs/prereg_test.md):
    two-sided sign test over phrases on the difference of the per-phrase detection rates, each rate over
    all rows of that system and phrase (pairs x seeds); phrases with equal rates are dropped."""
    key = lambda r: (r["set"], r["seed"], r["phrase_id"], r["emotion"], r["emotion2"])  # noqa: E731
    base = {key(r): r["detected"] for r in rows if r["condition"] == "nl_instruct"}
    ref = {key(r): r["detected"] for r in rows if r["condition"] == "stitch"}
    out = {}
    for s in set(systems["nl_instruct"]["by_set"]) & set(systems["stitch"]["by_set"]):
        keys = sorted(k for k in set(base) & set(ref) if k[0] == s)
        row_level = stats.mcnemar_exact([ref[k] for k in keys], [base[k] for k in keys])
        b_phr, r_phr = systems["nl_instruct"]["by_set"][s]["per_phrase"], systems["stitch"]["by_set"][s]["per_phrase"]
        diffs = [r_phr[p][0] / r_phr[p][1] - b_phr[p][0] / b_phr[p][1] for p in sorted(set(b_phr) & set(r_phr))]
        ref_wins, base_wins = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        out[s] = {
            "n_pairs": row_level["n_pairs"],
            "reference_only": row_level["a_only"],
            "baseline_only": row_level["b_only"],
            "mcnemar_p": row_level["p_value"],
            "n_phrases": len(diffs),
            "phrases_reference_ahead": ref_wins,
            "phrases_baseline_ahead": base_wins,
            "phrase_sign_test_p": stats.sign_test(ref_wins, base_wins),
        }
    return out


def seed_agreement(rows: list[dict]) -> dict:
    """How often the primary decision repeats across seeds. A cell is one phrase and one commanded pair
    (for a no-switch row: its emotion); it counts when at least two seeds of it exist."""
    out = {}
    for (group, s), g in _group([r for r in rows if analysis_group(r)], "condition", "set").items():
        cells = defaultdict(dict)
        for r in g:
            cells[(r["phrase_id"], r["emotion"], r["emotion2"])][r["seed"]] = decision(r)
        multi = [c for c in cells.values() if len(c) >= 2]
        seed_pairs = {}
        for a, b in combinations(sorted({r["seed"] for r in g}), 2):
            both = [c for c in cells.values() if a in c and b in c]
            if both:
                seed_pairs[f"s{a}-s{b}"] = [sum(c[a] == c[b] for c in both), len(both)]
        out[f"{group}/{s}"] = {
            "n_cells": len(cells),
            "n_cells_multi_seed": len(multi),
            "unanimous": sum(len(set(c.values())) == 1 for c in multi),
            "unanimous_positive": sum(all(c.values()) for c in multi),
            "seed_pairs": seed_pairs,
        }
    return out


def sensitivity(rows: list[dict], cfg: dict) -> list[dict]:
    """Primary counts per set over the grid of detector constants in config `sensitivity`, plus the weighted
    change-point estimator at the frozen constants. Shows how much of the result depends on constants that
    were chosen by hand on the dev set. Each clip is counted once (`analysis_group`)."""
    dc, grid = cfg["detector"], cfg["sensitivity"]
    rules = primary_rules(cfg, rows)
    variants = [(thr, tol, False) for thr in grid["min_abs_delta"] for tol in grid["max_changepoint_err_s"]]
    variants.append((dc["min_abs_delta"], dc["max_changepoint_err_s"], True))
    out = []
    for thr, tol, weighted in variants:
        counts = {}
        for r in rows:
            group = analysis_group(r)
            if group is None:
                continue
            step = primary_step(r)
            if weighted and step:
                step = detect.measure_step(
                    r["profile_centres_s"],
                    r["profile_arousal"],
                    step["switch_time_s"],
                    r["expect_sign"],
                    dc["min_seg_windows"],
                    weighted_split=True,
                )
            if r["condition"] in SWITCH_CONDITIONS:
                gated_out = rules[r["set"]]["validity_gate"] and not r["valid"]
                hit = detect.is_detected(step, thr, tol) and not gated_out
            else:
                hit = detect.exceeds_null(step, thr, tol)
            cell = counts.setdefault(f"{group}/{r['set']}", [0, 0])
            cell[0] += hit
            cell[1] += 1
        out.append({"min_abs_delta": thr, "max_changepoint_err_s": tol, "weighted_split": weighted, "counts": counts})
    return out


def score(cfg: dict, rows: list[dict]) -> dict:
    set_order = {s: i for i, s in enumerate(cfg["sets"])}  # dev first, validation second, as in the config
    rows.sort(key=lambda r: (set_order[r["set"]], r["seed"], r["run"], r["id"]))
    add_derived(rows, cfg)
    metrics = {
        "config": cfg["config_path"],
        "measure_fingerprint": measure_fingerprint(cfg),
        "n_rows": len(rows),
        "set_roles": {s: set_role(cfg, s) for s in cfg["sets"]},
        "primary_rule": {s: cfg["primary_rule"][set_role(cfg, s)] for s in cfg["sets"]},
        "systems": {},
        "null": {},
    }
    for cond in SWITCH_CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        system = {
            "role": cfg["systems"][cond]["role"],
            "pooled": summarise_switch(sub, cfg),
            "by_set": {},
            "by_set_seed": {},
            "by_pair": {},
        }
        for (s,), g in _group(sub, "set").items():
            seed_rates = [sum(r["detected"] for r in gg) / len(gg) for gg in _group(g, "seed").values()]
            system["by_set"][s] = {
                **summarise_switch(g, cfg),
                "n_seeds": len(seed_rates),
                "det_rate_seed_mean": float(np.mean(seed_rates)),
                "det_rate_seed_sd": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else None,
            }
        for (s, seed), g in _group(sub, "set", "seed").items():
            system["by_set_seed"][f"{s}/s{seed}"] = summarise_switch(g, cfg)
        for (e1, e2), g in _group(sub, "emotion", "emotion2").items():
            nearest = [r["nearest_level"] for r in g if r["nearest_level"]]
            system["by_pair"][f"{e1}>{e2}"] = {
                **summarise_switch(g, cfg),
                "nearest_level_counts": {name: nearest.count(name) for name in sorted(set(nearest))},
            }
        metrics["systems"][cond] = system
    null_groups = defaultdict(list)  # condition/run kind: the no-switch rows of stitch runs are kept apart
    for r in rows:
        if r["condition"] in NULL_CONDITIONS:
            null_groups[f"{r['condition']}/{run_kind(r)}"].append(r)
    for name, g in null_groups.items():
        metrics["null"][name] = {
            "pooled": summarise_null(g, cfg),
            "by_set": {s: summarise_null(gg, cfg) for (s,), gg in _group(g, "set").items()},
        }
    if {"nl_instruct", "stitch"} <= set(metrics["systems"]):
        metrics["paired_reference_vs_baseline"] = paired_comparison(rows, metrics["systems"])
    metrics["seed_agreement"] = seed_agreement(rows)
    metrics["sensitivity"] = sensitivity(rows, cfg)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default=None, help="output directory (default: results/<config name>)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--from-rows", action="store_true", help="skip the models and score the existing rows.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / (args.out or f"results/{cfg['name']}")
    rows_path, meta_path = out / "rows.jsonl", out / "measure_meta.json"
    if args.from_rows:
        measured_with = json.loads(meta_path.read_text(encoding="utf-8"))["measure_fingerprint"]
        if measured_with != measure_fingerprint(cfg):
            raise SystemExit(
                f"[score] {rows_path} was measured with other judge/ASR/detector settings than {args.config}; "
                "re-run without --from-rows"
            )
        rows = read_jsonl(rows_path)
    else:
        started = time.time()
        manifest = read_jsonl(ROOT / args.manifest)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        previous = []
        if meta and rows_path.exists():
            if meta["measure_fingerprint"] == measure_fingerprint(cfg):
                previous = read_jsonl(rows_path)
            else:
                print(f"[measure] {rows_path} was measured with other settings; measuring every clip")
        kept, todo = split_measured(manifest, previous)
        print(f"[measure] {len(kept)} clips already measured, {len(todo)} to measure")
        new = measure(cfg, todo, args.device) if todo else []
        n_aligned = align_stitched(cfg, kept, args.device)
        by_key = {row_key(r): r for r in kept + new}
        rows = [by_key[row_key(m)] for m in manifest]
        write_jsonl(rows_path, rows)  # before scoring: a scoring error must not cost the measurement
        import ctranslate2
        import faster_whisper
        import torch
        import torchaudio
        import transformers

        this_pass = {
            "n_measured": len(todo),
            "n_aligned_only": n_aligned,
            "seconds": round(time.time() - started),
            "device": args.device,
            "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "versions": {
                m.__name__: m.__version__ for m in (torch, torchaudio, transformers, faster_whisper, ctranslate2)
            },
        }
        if previous:  # the first pass stays at the top level; later, incremental passes are listed
            meta["n_clips"] = len(rows)
            if todo or n_aligned:
                meta.setdefault("later_passes", []).append(this_pass)
        else:
            meta = {"measure_fingerprint": measure_fingerprint(cfg), "n_clips": len(rows), **this_pass}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics = score(cfg, rows)
    write_jsonl(rows_path, rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cond, system in metrics["systems"].items():
        for s, m in system["by_set"].items():
            print(
                f"[score] {cond:12s} {s:8s} gates {'pass' if m['gates']['passed'] else 'FAIL'}  "
                f"det {m['detected']}/{m['n']} = {m['det_rate']:.3f}  by rule {m['detected_by_rule']}  "
                f"WER {m['wer_mean']:.4f}"
            )
    print(f"[score] wrote rows.jsonl and metrics.json to {out}")


if __name__ == "__main__":
    main()

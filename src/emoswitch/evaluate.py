"""Evaluate released audio: measure every clip, then score the systems.

    python -m emoswitch.evaluate --config configs/baseline.yaml              # audio -> rows -> metrics
    python -m emoswitch.evaluate --config configs/baseline.yaml --from-rows  # rows -> metrics, no models

`measure` needs the judge, the aligner and the ASR weights (downloaded on first use, CPU is enough).
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
from pathlib import Path

import numpy as np

from emoswitch import detect, gates, stats
from emoswitch.config import ROOT, load_config, set_role
from emoswitch.stimuli import NULL_CONDITIONS

SWITCH_CONDITIONS = ("nl_instruct", "stitch")
# Config sections a per-clip row depends on. Scoring-only constants (thresholds, intervals) are not here.
MEASURE_KEYS = ("judge", "asr", "gates")


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


# ── measure: one clip -> one row ────────────────────────────────────────────


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
        # A stitched clip knows its boundary by construction. A single-pass clip does not, so its
        # boundary is where forced alignment puts the first word of sentence 2.
        t_switch, source, words = m.get("switch_time_s"), "construction", []
        if t_switch is None:
            words = aligner(y, m["text"])
            t_switch = detect.switch_time(words, m["switch_word"], duration, dc["switch_search_from_frac"])
            source = "forced_alignment"
        step = detect.measure_step(centres, arousal, t_switch, m["expect_sign"], dc["min_seg_windows"])
        hyp = transcribe(asr, path, cfg["asr"])
        rows.append(
            {
                **{k: m[k] for k in keep},
                "switch_word": m["switch_word"],
                "expect_sign": m["expect_sign"],
                "audio": m["audio"],
                "duration_s": round(duration, 3),
                "profile_centres_s": centres,
                "profile_arousal": arousal,
                "profile_sd": round(float(np.std(arousal)), 4) if arousal else None,
                "hyp": hyp,
                "words": words,
                "wer": round(gates.word_error_rate(m["text"], hyp), 4),
                "leak": gates.leaks(hyp, m["text"], m["leak_stems"]),
                "looped": duration > cfg["gates"]["loop_threshold_s"],
                "switch_time_source": source,
                "step": step,
            }
        )
        print(
            f"[measure] {i}/{len(manifest)} {m['run']}/{m['id']}: "
            f"delta={step and step['delta']} cp_err={step and step['changepoint_err_s']} wer={rows[-1]['wer']}",
            flush=True,
        )
    return rows


# ── score: rows -> metrics ──────────────────────────────────────────────────


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def outcome(step: dict | None, detected: bool, min_abs_delta: float) -> str:
    """Why a switch row was or was not counted. The checks are ordered: the first failed one names the miss,
    so a row below the size threshold is `flat` whatever its sign."""
    if detected:
        return "detected"
    if step is None:
        return "no_step"  # boundary word not located, or the clip is too short for a step
    if abs(step["delta"]) < min_abs_delta:
        return "flat"
    if not step["sign_ok"]:
        return "wrong_sign"
    return "misplaced"  # a step of the right size and sign whose change-point is away from the boundary


def add_derived(rows: list[dict], cfg: dict) -> None:
    """Per-row decisions and the control-derived reference levels, written into the rows."""
    dc = cfg["detector"]
    thr, tol = dc["min_abs_delta"], dc["max_changepoint_err_s"]
    controls = {(r["run"], r["phrase_id"], r["emotion"]): r for r in rows if r["condition"] == "control"}
    plains = {(r["run"], r["phrase_id"]): r for r in rows if r["condition"] == "plain"}
    for r in rows:
        step = r["step"]
        if r["condition"] in NULL_CONDITIONS:
            r["exceeds_null"] = detect.exceeds_null(step, thr, tol)
            continue
        r["detected"] = detect.is_detected(step, thr, tol)
        r["outcome"] = outcome(step, r["detected"], thr)
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
    wer = float(np.mean([r["wer"] for r in rows]))
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
    steps = [r["step"] for r in rows if r["step"]]
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
    return {
        "n": n,
        "n_with_step": len(steps),
        "detected": k,
        "det_rate": round(k / n, 4) if n else None,
        "det_ci_wilson": [round(lo, 4), round(hi, 4)],
        "det_ci_phrase_bootstrap": [round(blo, 4), round(bhi, 4)],
        "n_phrases": len({(r["set"], r["phrase_id"]) for r in rows}),
        "abs_delta_mean": _mean([abs(s["delta"]) for s in steps]),
        "abs_delta_max": round(max(abs(s["delta"]) for s in steps), 4) if steps else None,
        "commanded_step_mean": _mean([r["commanded_step"] for r in rows]),
        "sign_ok_rate": _mean([float(s["sign_ok"]) for s in steps]),
        "changepoint_within_tol_rate": _mean(
            [float(s["changepoint_err_s"] <= tol) for s in steps if s["changepoint_err_s"] is not None]
        ),
        "ideal_step_mean": _mean([r["ideal_step"] for r in paired]),
        "realised_share": (round(sum(r["commanded_step"] for r in paired) / ideal_sum, 4) if ideal_sum > 0 else None),
        "wer_mean": _mean([r["wer"] for r in rows]),
        "wer_median": round(float(np.median([r["wer"] for r in rows])), 4) if rows else None,
        "leak_rate": _mean([float(r["leak"]) for r in rows]),
        "looped": int(sum(r["looped"] for r in rows)),
        "gates": gate_check(rows, cfg) if rows else None,
    }


def summarise_null(rows: list[dict], cfg: dict) -> dict:
    """What the detector reads on rows where no switch was asked for."""
    steps = [r["step"] for r in rows if r["step"]]
    k = sum(r["exceeds_null"] for r in rows)
    lo, hi = stats.wilson_ci(k, len(rows), cfg["stats"]["ci"])
    abs_d = [abs(s["delta"]) for s in steps]
    return {
        "n": len(rows),
        "n_with_step": len(steps),
        "abs_delta_mean": _mean(abs_d),
        "abs_delta_p95": round(float(np.percentile(abs_d, 95)), 4) if abs_d else None,
        "abs_delta_max": round(float(np.max(abs_d)), 4) if abs_d else None,
        "exceed": k,
        "exceed_rate": round(k / len(rows), 4) if rows else None,
        "exceed_ci_wilson": [round(lo, 4), round(hi, 4)],
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


def paired_comparison(rows: list[dict]) -> dict:
    """Reference against baseline on the rows both have: same set, seed, phrase and commanded switch.

    Rows of one phrase are not independent, so next to the row-level McNemar test there is a sign
    test over phrases: a phrase counts for the system that is detected in more of its common rows.
    """
    key = lambda r: (r["set"], r["seed"], r["phrase_id"], r["emotion"], r["emotion2"])  # noqa: E731
    base = {key(r): r["detected"] for r in rows if r["condition"] == "nl_instruct"}
    ref = {key(r): r["detected"] for r in rows if r["condition"] == "stitch"}
    common = sorted(set(base) & set(ref))

    def compare(keys: list) -> dict:
        per_phrase = defaultdict(lambda: [0, 0])
        for k in keys:
            per_phrase[(k[0], k[2])][0] += ref[k]
            per_phrase[(k[0], k[2])][1] += base[k]
        ref_wins = sum(a > b for a, b in per_phrase.values())
        base_wins = sum(b > a for a, b in per_phrase.values())
        row_level = stats.mcnemar_exact([ref[k] for k in keys], [base[k] for k in keys])
        return {
            "n_pairs": row_level["n_pairs"],
            "reference_only": row_level["a_only"],
            "baseline_only": row_level["b_only"],
            "mcnemar_p": row_level["p_value"],
            "n_phrases": len(per_phrase),
            "phrases_reference_ahead": ref_wins,
            "phrases_baseline_ahead": base_wins,
            "phrase_sign_test_p": stats.sign_test(ref_wins, base_wins),
        }

    out = {s: compare([k for k in common if k[0] == s]) for s in dict.fromkeys(k[0] for k in common)}
    out["all_sets"] = compare(common)
    return out


def sensitivity(rows: list[dict], cfg: dict) -> list[dict]:
    """The headline counts under other detector constants. The first variant is the frozen detector;
    the rest show how much of the result depends on constants that were chosen by hand on the dev set."""
    dc = cfg["detector"]
    variants = [("frozen detector", dc["min_abs_delta"], dc["max_changepoint_err_s"], False)]
    variants += [(f"threshold {t}", t, dc["max_changepoint_err_s"], False) for t in (0.10, 0.20)]
    variants += [(f"change-point tolerance {t} s", dc["min_abs_delta"], t, False) for t in (0.5, 1.0)]
    variants += [("no change-point condition", dc["min_abs_delta"], None, False)]
    variants += [("weighted change-point estimator", dc["min_abs_delta"], dc["max_changepoint_err_s"], True)]
    out = []
    for name, thr, tol, weighted in variants:
        counts = {}
        for r in rows:
            step = r["step"]
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
                group, hit = r["condition"], detect.is_detected(step, thr, tol)
            elif run_kind(r) == "inline":  # the stitch runs repeat these clips byte for byte
                group, hit = "null", detect.exceeds_null(step, thr, tol)
            else:
                continue
            cell = counts.setdefault(f"{group}/{r['set']}", [0, 0])
            cell[0] += hit
            cell[1] += 1
        out.append({"variant": name, "min_abs_delta": thr, "max_changepoint_err_s": tol, "weighted_split": weighted,
                    "counts": counts})  # fmt: skip
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
                "det_rate_seed_mean": round(float(np.mean(seed_rates)), 4),
                "det_rate_seed_sd": (round(float(np.std(seed_rates, ddof=1)), 4) if len(seed_rates) > 1 else None),
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
    for (kind,), g in _group(
        [{**r, "kind": run_kind(r)} for r in rows if r["condition"] in NULL_CONDITIONS], "kind"
    ).items():
        metrics["null"][kind] = {
            "pooled": summarise_null(g, cfg),
            "by_set": {s: summarise_null(gg, cfg) for (s,), gg in _group(g, "set").items()},
        }
    if {"nl_instruct", "stitch"} <= set(metrics["systems"]):
        metrics["paired_reference_vs_baseline"] = paired_comparison(rows)
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
        rows = measure(cfg, read_jsonl(ROOT / args.manifest), args.device)
        write_jsonl(rows_path, rows)  # before scoring: a scoring error must not cost the measurement
        import ctranslate2
        import faster_whisper
        import torch
        import torchaudio
        import transformers

        meta = {
            "measure_fingerprint": measure_fingerprint(cfg),
            "n_clips": len(rows),
            "seconds": round(time.time() - started),
            "device": args.device,
            "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "versions": {
                m.__name__: m.__version__ for m in (torch, torchaudio, transformers, faster_whisper, ctranslate2)
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics = score(cfg, rows)
    write_jsonl(rows_path, rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cond, system in metrics["systems"].items():
        for s, m in system["by_set"].items():
            print(
                f"[score] {cond:12s} {s:8s} gates {'pass' if m['gates']['passed'] else 'FAIL'}  "
                f"det {m['detected']}/{m['n']} = {m['det_rate']} CI {m['det_ci_wilson']}  "
                f"|delta| {m['abs_delta_mean']}  WER {m['wer_mean']}"
            )
    print(f"[score] wrote rows.jsonl and metrics.json to {out}")


if __name__ == "__main__":
    main()

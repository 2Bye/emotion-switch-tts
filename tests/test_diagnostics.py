"""Error analysis and robustness checks of the chosen system: rules and the committed results."""

import csv
import json

import numpy as np
import pytest

from emoswitch import diagnose, perturb
from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import measure_fingerprint, read_jsonl
from emoswitch.experiments import load_rows

CFG = load_config("configs/diagnostics.yaml")
INPASS = load_config("configs/inpass.yaml")
RESULTS = ROOT / "results" / CFG["name"]


def test_the_diagnostics_measure_and_score_like_the_experiment_they_read():
    assert measure_fingerprint(CFG) == measure_fingerprint(INPASS)
    for key in ("systems", "inpass", "detector", "gates", "primary_rule", "sets", "manifest"):
        assert CFG[key] == INPASS[key], key
    assert CFG["systems"][CFG["system"]] == {**INPASS["systems"]["swap_anchor_steer"]}


def _row(centres, arousal, t, sign, detected=False, outcome="misplaced", valid=True):
    from emoswitch import detect

    step = detect.measure_step(centres, arousal, t, sign, CFG["detector"]["min_seg_windows"])
    return {"profile_centres_s": centres, "profile_arousal": arousal, "expect_sign": sign, "step": step,
            "boundary": "forced_alignment", "switch_time_source": "forced_alignment", "detected": detected,
            "outcome": outcome, "valid": valid}  # fmt: skip


def test_miss_types_follow_the_rules_of_the_plan():
    centres = [0.5 + 0.25 * i for i in range(24)]  # 0.5 .. 6.25 s
    # a clean step down at 3.5 s with a burst in the first three windows: the plain split goes to the burst
    burst = [1.0] * 3 + [0.6] * 9 + [0.2] * 12
    row = _row(centres, burst, 3.5, -1)
    plain, weighted = diagnose.offsets(row, CFG)
    assert plain < -CFG["detector"]["max_changepoint_err_s"] and abs(weighted) <= 0.25
    assert diagnose.miss_type(row, CFG) == "edge_split"
    # the level falls a second and a half before the boundary by either estimator
    early = [0.7] * 6 + [0.2] * 18
    assert diagnose.miss_type(_row(centres, early, 3.5, -1), CFG) == "early_change"
    late = [0.7] * 18 + [0.2] * 6
    assert diagnose.miss_type(_row(centres, late, 3.5, -1), CFG) == "late_change"
    assert diagnose.miss_type(_row(centres, late, 3.5, -1, detected=True), CFG) == "detected"
    assert diagnose.miss_type(_row(centres, late, 3.5, -1, outcome="flat"), CFG) == "flat"
    assert diagnose.miss_type(_row(centres, late, 3.5, -1, outcome="flat", valid=False), CFG) == "invalid"


def test_crossing_is_where_the_level_stays_beyond_the_midpoint():
    centres = [0.5 + 0.25 * i for i in range(24)]
    profile = [0.2] * 10 + [0.9, 0.2] + [0.8] * 12  # one window above the midpoint does not count
    row = _row(centres, profile, 3.5, 1)
    assert diagnose.crossing(row, CFG) == pytest.approx(centres[12] - 3.5)


def test_bins_and_their_order():
    edges = [2.78, 3.6]
    labels = [diagnose.bin_label(x, edges, " s") for x in (1.0, 2.78, 3.0, 3.6, 9.0, None)]
    assert labels == ["< 2.78 s", "2.78–3.60 s", "2.78–3.60 s", "≥ 3.60 s", "≥ 3.60 s", "n/a"]
    assert sorted(set(labels) - {"n/a"}, key=diagnose.group_order) == ["< 2.78 s", "2.78–3.60 s", "≥ 3.60 s"]
    assert sorted(["p10", "p2", "p0"], key=diagnose.group_order) == ["p0", "p2", "p10"]


def test_added_noise_has_the_requested_level_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = (0.1 * np.sin(np.linspace(0, 2000, 48000)) * rng.uniform(0.5, 1.0, 48000)).astype(np.float32)
    for snr_db in (30, 10, 0):
        noisy = perturb.add_noise(y, snr_db, perturb.clip_rng(0, snr_db, "run", "clip"))
        noise = noisy.astype(np.float64) - y
        assert 10 * np.log10(np.mean(y.astype(np.float64) ** 2) / np.mean(noise**2)) == pytest.approx(snr_db, abs=0.1)
        again = perturb.add_noise(y, snr_db, perturb.clip_rng(0, snr_db, "run", "clip"))
        assert np.array_equal(noisy, again)
    other = perturb.add_noise(y, 10, perturb.clip_rng(0, 10, "run", "another clip"))
    assert not np.array_equal(other, perturb.add_noise(y, 10, perturb.clip_rng(0, 10, "run", "clip")))
    assert float(np.abs(perturb.add_noise(y * 9, 0, perturb.clip_rng(0, 0, "run", "clip"))).max()) < 1.0


# ── the committed results ───────────────────────────────────────────────────

ROWS = load_rows(CFG, ROOT / CFG["source"])
COMMITTED = json.loads((RESULTS / "diagnostics.json").read_text(encoding="utf-8"))


def test_noisy_rows_cover_the_selected_clips_at_every_level():
    clips = perturb.selected(CFG)
    assert {m["condition"] for m in clips} == set(CFG["noise"]["conditions"])
    key = lambda r: (r["run"], r["id"], r["sha256_wav_source"])  # noqa: E731
    for snr_db in CFG["noise"]["snr_db"]:
        path = perturb.level_path(CFG, snr_db)
        rows = read_jsonl(path)
        assert list(map(key, rows)) == list(map(key, clips)), snr_db
        assert all(r["snr_db"] == snr_db and r["audio"].startswith("data/audio/") for r in rows)
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["measure_fingerprint"] == measure_fingerprint(CFG) and meta["noise_seed"] == CFG["noise"]["seed"]


def test_committed_diagnostics_are_what_the_rows_give():
    system, s, ref = CFG["system"], CFG["analysis_set"], CFG["reference"]
    null = "swap_null"
    plain = diagnose.null_levels(ROWS, null)
    described = {c: [diagnose.describe(r, CFG, plain) for r in ROWS if r["condition"] == c]
                 for c in ("swap", "swap_anchor", "swap_steer", system, ref)}  # fmt: skip
    chosen = described[system]
    recs = [r for r in chosen if r["set"] == s]
    dev = [r for r in chosen if r["set"] != s]
    by_system = {
        c: [r for r in described[c] if r["set"] == s] for c in (system, ref, "swap", "swap_anchor", "swap_steer")
    }
    by_system[f"{system}, p2a (dev)"] = dev
    again = {
        "groups": {s: diagnose.group_stats(recs, CFG), "p2a": diagnose.group_stats(dev, CFG)},
        "drift": diagnose.drift([r for r in ROWS if r["condition"] == null and r["set"] == s]),
        "placement": diagnose.placement(recs, CFG),
        "calibration": diagnose.calibration(ROWS, CFG, null),
        "edge_sides": diagnose.edge_sides(recs, CFG),
        "boundary_readings": diagnose.boundary_readings(
            [r for r in ROWS if r["condition"] in (system, ref) and r["set"] == s], CFG
        ),
        "beyond": diagnose.beyond(by_system, CFG),
        "invalid_by_anchor": diagnose.invalid_by_anchor(ROWS, CFG),
        "seed_cells": diagnose.seed_cells(recs),
        "slices": diagnose.slices(recs, CFG),
        "direction_by_seconds_after": diagnose.cross(recs, CFG, "direction", "seconds after the boundary"),
        "direction_by_seconds_before": diagnose.cross(recs, CFG, "direction", "seconds before the boundary"),
        "noise": diagnose.noise_levels(CFG, RESULTS, ROWS),
        "resample": diagnose.resample(ROWS, CFG),
        "rerender": diagnose.rerender_records(CFG, RESULTS, ROWS),
        "rerender_pairs": diagnose.rerender_pairs(CFG, RESULTS, ROWS),
    }
    for name, value in again.items():
        assert json.loads(json.dumps(value)) == COMMITTED[name], name
    assert set(COMMITTED) - set(again) == {"system", "set", "examples", "other_sets"}
    with (RESULTS / "clips.csv").open(encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    assert [(c["run"], c["id"], c["type"]) for c in clips] == [(r["run"], r["id"], r["type"]) for r in chosen]


def test_rendered_again_runs_are_what_the_config_builds_and_every_clip_is_measured():
    from emoswitch import rerender

    rc = CFG["rerender"]
    manifest = read_jsonl(ROOT / rc["manifest"])
    fields = ("id", "set", "seed", "phrase_id", "condition", "emotion", "emotion2", "text", "instruct",
              "switch_word", "leak_stems", "expect_sign")  # fmt: skip
    for seed in rc["seeds"]:
        run = rerender.run_name(rc["set"], seed)
        got = [{k: m[k] for k in fields} for m in manifest if m["run"] == run]
        assert got == rerender.planned_rows(CFG, seed), run
    assert all((ROOT / m["audio"]).exists() and m["switch_time_s"] is None for m in manifest)
    rows = read_jsonl(RESULTS / "rerender" / "rows.jsonl")
    key = lambda r: (r["run"], r["id"], r["sha256_wav_source"])  # noqa: E731
    assert list(map(key, rows)) == list(map(key, manifest))
    meta = json.loads((RESULTS / "rerender" / "measure_meta.json").read_text(encoding="utf-8"))
    assert meta["measure_fingerprint"] == measure_fingerprint(CFG)


def test_the_clean_level_of_the_noise_check_is_the_result_of_homework_2():
    experiments = json.loads((ROOT / CFG["source"] / "experiments.json").read_text(encoding="utf-8"))["experiments"]
    ref = next(e for e in experiments if e["system"] == CFG["system"] and e["set"] == CFG["analysis_set"]
               and e["seed"] is None)  # fmt: skip
    clean = COMMITTED["noise"][0]
    assert clean["snr_db"] is None and (clean["detected"], clean["n"]) == (ref["detected"], ref["n"])
    assert COMMITTED["groups"][CFG["analysis_set"]]["detected"] == ref["detected"]
    assert COMMITTED["resample"]["all"]["detected"] == ref["detected"]
    assert [lv["snr_db"] for lv in COMMITTED["noise"][1:]] == CFG["noise"]["snr_db"]


def test_every_miss_has_one_type_and_every_example_is_a_released_clip():
    for s, g in COMMITTED["groups"].items():
        assert g["detected"] + sum(g["types"].values()) == g["n"], s
        assert set(g["types"]) <= set(diagnose.TYPES)
    in_set = {e["type"] for e in COMMITTED["examples"] if e["set"] == CFG["analysis_set"]}
    assert in_set - {"detected"} == set(COMMITTED["groups"][CFG["analysis_set"]]["types"])  # every kind of miss
    assert all((ROOT / e["audio"]).exists() for e in COMMITTED["examples"])


def test_the_homework_3_report_tables_are_copies_and_its_headline_counts_are_the_committed_ones():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from sync_readme import BLOCK, render

    report = (ROOT / "HW3.md").read_text(encoding="utf-8")
    assert len(BLOCK.findall(report)) >= 8
    assert render(report, RESULTS) == report
    assert "PENDING" not in report
    prose = " ".join(report.split())
    g = COMMITTED["groups"][CFG["analysis_set"]]
    assert f"{g['detected']} из {g['n']} ({g['det']:.3f})" in prose
    for lv in COMMITTED["noise"][1:]:
        assert f"{lv['detected']} из {lv['n']} ({lv['det']:.3f})" in prose, lv["snr_db"]
    quiet = COMMITTED["noise"][1]
    changed = quiet["lost"] + quiet["gained"]
    assert f"{changed} клипов из {quiet['n']} ({quiet['lost']} потеряно, {quiet['gained']} получено)" in prose
    beyond = [b for b in COMMITTED["beyond"] if b["condition"] == CFG["system"]]
    assert f"{sum(b['beyond_over_threshold'] for b in beyond)} из {sum(b['n'] for b in beyond)} годных" in prose
    down = next(c for c in COMMITTED["calibration"] if c["group"].startswith("stitch, switch down"))
    assert f"({down['n'] - down['detector']} из {down['n']})" in prose
    halves = COMMITTED["resample"]["halves"]
    assert f"между {halves['q05']:.3f} и {halves['q95']:.3f}" in prose
    assert f"получено {', '.join(str(lv['gained']) for lv in COMMITTED['noise'][1:-1])} и" in prose
    null = [lv["null_exceeds"] for lv in COMMITTED["noise"]]
    assert f"у {', '.join(map(str, null[1:-1]))} и {null[-1]} из 108 при {null[0]} из 108" in prose
    seeds = CFG["rerender"]["seeds"]
    again = [r for r in COMMITTED["rerender"] if r["source"].startswith("again")]
    assert [r["seeds"] for r in again] == [*map(str, seeds), ", ".join(map(str, seeds))]
    runs, pooled = again[:-1], again[-1]
    for r in again:
        assert f"{r['detected']} из {r['n']} ({r['det']:.3f})" in prose, r["seeds"]
    assert f"дают {runs[0]['detected']}, {runs[1]['detected']} и {runs[2]['detected']} из 48" in prose
    beyond = [r["beyond_over_threshold"] for r in runs]
    assert f"в {beyond[0]}, {beyond[1]} и {beyond[2]} клипах из 48" in prose
    assert f"вместе {pooled['beyond_over_threshold']} из {pooled['beyond_n']}" in prose
    up, down = pooled["by_direction"]["up"], pooled["by_direction"]["down"]
    assert f"вверх {up['detected']} из {up['n']}, вниз {down['detected']} из {down['n']}" in prose
    assert all(r["invalid"] == 0 and r["null_exceeds"] == 0 for r in again)
    pairs = COMMITTED["rerender_pairs"]
    same = [p["same"] for p in pairs if p["kind"] == "same seed, other hardware"]
    here = [p["same"] for p in pairs if p["kind"] == "other seed, same hardware"]
    across = [p["same"] for p in pairs if p["kind"] == "other seed, other hardware"]
    assert len(same) == 1 and len(here) == 6 and len(across) == 8 and all(p["n"] == 48 for p in pairs)
    assert f"у {same[0]} клипов из 48. Два разных сида на одном железе согласуются в {min(here)}–{max(here)}" in prose
    assert f"разные сиды на разном железе — в {min(across)}–{max(across)}" in prose

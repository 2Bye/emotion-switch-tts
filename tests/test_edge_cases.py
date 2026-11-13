"""Outcomes, short clips, the generator's pure parts and the small statistics helpers."""

import json
import subprocess
import sys

import numpy as np
import pytest

from emoswitch import detect, stats
from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import outcome, score
from emoswitch.generate import crossfade
from emoswitch.stimuli import build_rows

CFG = load_config("configs/baseline.yaml")
THR, TOL = CFG["detector"]["min_abs_delta"], CFG["detector"]["max_changepoint_err_s"]
CENTRES = [round(0.5 + 0.25 * i, 3) for i in range(20)]


def step_of(arousal, t_switch, sign):
    return detect.measure_step(CENTRES[: len(arousal)], arousal, t_switch, sign, 3)


def test_each_outcome_is_reachable():
    flat = step_of([0.5] * 20, CENTRES[10], -1)
    wrong = step_of([0.3] * 10 + [0.6] * 10, CENTRES[10], -1)
    late = step_of([0.6] * 8 + [0.3] * 12, CENTRES[12], -1)
    hit = step_of([0.6] * 10 + [0.3] * 10, CENTRES[10], -1)
    cases = {"flat": flat, "wrong_sign": wrong, "misplaced": late, "detected": hit, "no_step": None}
    for name, step in cases.items():
        assert outcome(step, detect.is_detected(step, THR, TOL), THR) == name


def test_a_clip_too_short_for_a_change_point_is_scored_without_crashing():
    row = next(r for r in build_rows(CFG, "heldout", "inline", 42) if r["condition"] == "nl_instruct")
    arousal = [0.6, 0.6, 0.3, 0.3]  # four windows: no split has three on each side
    step = detect.measure_step(CENTRES[:4], arousal, CENTRES[2], row["expect_sign"], 3)
    assert step["changepoint_err_s"] is None and not detect.is_detected(step, THR, TOL)
    measured = {"run": "inline_heldout_s42", "audio": "x.flac", "duration_s": 1.5, "profile_centres_s": CENTRES[:4],
                "profile_arousal": arousal, "profile_sd": 0.15, "hyp": row["text"], "words": [], "wer": 0.0,
                "leak": False, "looped": False, "switch_time_source": "forced_alignment"}  # fmt: skip
    metrics = score(CFG, [{**row, **measured, "step": step}, {**row, **measured, "id": "other", "step": None}])
    pooled = metrics["systems"]["nl_instruct"]["pooled"]
    assert pooled["detected"] == 0 and pooled["n"] == 2 and pooled["n_with_step"] == 1
    json.dumps(metrics, allow_nan=False)  # metrics must stay valid JSON


def test_dropping_the_change_point_condition_only_adds_detections():
    late = step_of([0.6] * 8 + [0.3] * 12, CENTRES[12], -1)
    assert not detect.is_detected(late, THR, TOL) and detect.is_detected(late, THR, None)


def test_weighted_split_is_less_drawn_to_the_clip_edges():
    rng = np.random.default_rng(0)
    plain = weighted = 0
    for _ in range(300):
        a = rng.normal(0.5, 0.05, 20)  # no structure at all
        plain += detect.best_split(a, 3)[0] in (3, 17)
        weighted += detect.best_split(a, 3, weighted=True)[0] in (3, 17)
    assert weighted < plain


def test_crossfade_shortens_by_the_overlap_and_keeps_level():
    a, b = np.ones(2400, dtype=np.float32), np.ones(2400, dtype=np.float32)
    out = crossfade(a, b, 24_000, 20.0)
    assert len(out) == 4800 - 480
    assert out[:1920].tolist() == [1.0] * 1920 and float(np.max(out)) < 1.5
    assert len(crossfade(a[:100], b, 24_000, 20.0)) == 2500  # too short to fade: plain concatenation


def test_generator_dry_run_prints_the_rows_and_writes_nothing():
    before = sorted(p.name for p in (ROOT / "data").rglob("*"))
    cmd = [sys.executable, "-m", "emoswitch.generate", "--set", "heldout", "--kind", "stitch", "--seed", "42",
           "--dry-run"]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout
    rows = [json.loads(line) for line in out.splitlines()]
    assert [r["id"] for r in rows] == [r["id"] for r in build_rows(CFG, "heldout", "stitch", 42)]
    assert sorted(p.name for p in (ROOT / "data").rglob("*")) == before


def test_generator_rejects_an_unknown_set():
    cmd = [sys.executable, "-m", "emoswitch.generate", "--set", "nosuchset", "--kind", "inline", "--dry-run"]
    assert subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).returncode != 0


def test_sign_test_and_empty_interval():
    assert stats.sign_test(4, 0) == pytest.approx(0.125)
    assert stats.sign_test(7, 0) == pytest.approx(2 * 0.5**7)
    assert stats.sign_test(0, 0) == 1.0
    assert all(np.isnan(x) for x in stats.wilson_ci(0, 0))

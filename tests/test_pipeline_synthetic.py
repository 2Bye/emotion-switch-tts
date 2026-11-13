"""Scoring and reporting on fabricated profiles: no audio, no models.

The fake systems are built to have a known answer: the baseline keeps one level over the whole
utterance (never a switch), the reference steps exactly at the boundary (always a switch).
"""

import json

import pytest

from emoswitch import detect
from emoswitch.config import load_config
from emoswitch.evaluate import score
from emoswitch.stimuli import build_rows

LEVEL = {"angry": 0.64, "happy": 0.57, "sad": 0.29, None: 0.45}
CENTRES = [round(0.5 + 0.25 * i, 3) for i in range(20)]
T_SWITCH = CENTRES[10]


def fake_row(cfg: dict, row: dict, run: str) -> dict:
    if row["condition"] == "stitch":
        arousal = [LEVEL[row["emotion"]]] * 10 + [LEVEL[row["emotion2"]]] * 10
    elif row["condition"] == "nl_instruct":  # one global emotion: the more aroused of the two
        arousal = [max(LEVEL[row["emotion"]], LEVEL[row["emotion2"]])] * 20
    else:
        arousal = [LEVEL[row["emotion"]]] * 20
    step = detect.measure_step(CENTRES, arousal, T_SWITCH, row["expect_sign"], cfg["detector"]["min_seg_windows"])
    return {
        **row,
        "run": run,
        "audio": f"data/audio/{run}/{row['id']}.flac",
        "duration_s": 6.0,
        "profile_centres_s": CENTRES,
        "profile_arousal": arousal,
        "profile_sd": 0.0,
        "hyp": row["text"],
        "words": [],
        "wer": 0.0,
        "leak": False,
        "looped": False,
        "switch_time_source": "construction",
        "step": step,
    }


@pytest.fixture(scope="module")
def scored():
    cfg = load_config("configs/baseline.yaml")
    rows = []
    for set_name in cfg["sets"]:
        for kind in ("inline", "stitch"):
            rows += [fake_row(cfg, r, f"{kind}_{set_name}_s42") for r in build_rows(cfg, set_name, kind, 42)]
    return cfg, rows, score(cfg, rows)


def test_row_counts_follow_the_design(scored):
    _, rows, metrics = scored
    # 3 dev phrases + 4 validation phrases, 4 commanded switches each
    assert metrics["systems"]["nl_instruct"]["by_set"]["p2a"]["n"] == 12
    assert metrics["systems"]["nl_instruct"]["by_set"]["heldout"]["n"] == 16
    assert metrics["systems"]["stitch"]["pooled"]["n"] == 28
    # per phrase and run: 1 plain + 3 single-emotion controls
    assert metrics["null"]["inline"]["pooled"]["n"] == 7 * 4
    assert len(rows) == 2 * (28 + 28)


def test_flat_baseline_scores_zero_and_stepping_reference_scores_one(scored):
    _, rows, metrics = scored
    base, ref = metrics["systems"]["nl_instruct"]["pooled"], metrics["systems"]["stitch"]["pooled"]
    assert base["det_rate"] == 0.0 and ref["det_rate"] == 1.0
    assert base["realised_share"] == 0.0 and ref["realised_share"] == 1.0
    assert {r["outcome"] for r in rows if r["condition"] == "nl_instruct"} == {"flat"}
    assert metrics["paired_reference_vs_baseline"]["all_sets"]["reference_only"] == 28


def test_single_emotion_controls_never_pass_the_detector(scored):
    _, _, metrics = scored
    assert metrics["null"]["inline"]["pooled"]["exceed_rate"] == 0.0
    assert metrics["null"]["stitch"]["pooled"]["abs_delta_max"] == 0.0


def test_global_emotion_names_the_control_the_utterance_sits_on(scored):
    _, rows, _ = scored
    row = next(r for r in rows if r["id"] == "inline-nl-angry2sad-p0" and r["set"] == "p2a")
    assert row["nearest_level"] == "angry"
    assert row["ideal_step"] == pytest.approx(0.35)


def test_expected_sign_follows_the_arousal_order(scored):
    _, rows, _ = scored
    signs = {(r["emotion"], r["emotion2"]): r["expect_sign"] for r in rows if r["condition"] == "nl_instruct"}
    assert signs == {("happy", "sad"): -1, ("sad", "happy"): 1, ("sad", "angry"): 1, ("angry", "sad"): -1}


def test_report_renders_tables_and_figures(scored, tmp_path, monkeypatch):
    from emoswitch import report

    cfg, rows, metrics = scored
    out = tmp_path / "results"
    out.mkdir()
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["report", "--out", str(out)])
    report.main()
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "| baseline: one pass, instruction | heldout (val) | 4 | 1 | 0/16 | **0.000** |" in text
    assert (out / "fig_steps.png").stat().st_size > 10_000
    assert (out / "fig_profiles.png").stat().st_size > 10_000

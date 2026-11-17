"""Scoring and reporting on fabricated profiles: no audio, no models.

The fake systems are built to have a known answer: the baseline keeps one level over the whole
utterance (never a switch), the reference steps exactly at its boundary by construction (always a
switch there) while its forced-aligned boundary sits 1 s later (never a switch there). The planned
test set is included, with fabricated rows only, so that its primary rule (forced alignment) is used.
"""

import json

import pytest

from emoswitch import detect
from emoswitch.config import load_config
from emoswitch.evaluate import score
from emoswitch.stimuli import build_rows

LEVEL = {"angry": 0.64, "happy": 0.57, "sad": 0.29, None: 0.45}
CENTRES = [round(0.5 + 0.25 * i, 3) for i in range(20)]
T_SWITCH, T_ALIGNED = CENTRES[10], CENTRES[14]


def fake_row(cfg: dict, row: dict, run: str) -> dict:
    if row["condition"] in ("stitch", "stitch_null"):
        arousal = [LEVEL[row["emotion"]]] * 10 + [LEVEL[row["emotion2"]]] * 10
    elif row["condition"] == "nl_instruct":  # one global emotion: the more aroused of the two
        arousal = [max(LEVEL[row["emotion"]], LEVEL[row["emotion2"]])] * 20
    else:
        arousal = [LEVEL[row["emotion"]]] * 20
    common = {"run": run, "audio": f"data/audio/{run}/{row['id']}.flac", "sha256_wav_source": f"{run}/{row['id']}"}
    measured = {"duration_s": 6.0, "profile_centres_s": CENTRES, "profile_arousal": arousal, "profile_sd": 0.0,
                "hyp": row["text"]}  # fmt: skip
    min_seg = cfg["detector"]["min_seg_windows"]
    if row["condition"] in ("stitch", "stitch_null"):
        at = {t: detect.measure_step(CENTRES, arousal, t, row["expect_sign"], min_seg) for t in (T_SWITCH, T_ALIGNED)}
        return {**row, **common, **measured, "words": [], "switch_time_source": "construction", "step": at[T_SWITCH],
                "words_aligned": [], "step_aligned": at[T_ALIGNED]}  # fmt: skip
    step = detect.measure_step(CENTRES, arousal, T_SWITCH, row["expect_sign"], min_seg)
    return {**row, **common, **measured, "words": [], "switch_time_source": "forced_alignment", "step": step}


@pytest.fixture(scope="module")
def scored():
    cfg = load_config("configs/baseline.yaml")
    cfg["sets"] = {**cfg["sets"], **cfg["planned_sets"]}
    rows = []
    for set_name in cfg["sets"]:
        for kind in ("inline", "stitch", "stitchnull"):
            rows += [fake_row(cfg, r, f"{kind}_{set_name}_s42") for r in build_rows(cfg, set_name, kind, 42)]
    return cfg, rows, score(cfg, rows)


def test_row_counts_follow_the_design(scored):
    _, rows, metrics = scored
    # 3 dev, 4 validation and 12 test phrases, 4 commanded switches each
    assert metrics["systems"]["nl_instruct"]["by_set"]["p2a"]["n"] == 12
    assert metrics["systems"]["nl_instruct"]["by_set"]["heldout"]["n"] == 16
    assert metrics["systems"]["nl_instruct"]["by_set"]["test"]["n"] == 48
    assert metrics["systems"]["stitch"]["pooled"]["n"] == 76
    # per phrase and run: 1 plain + 3 single-emotion controls; 3 same-emotion stitched rows
    assert metrics["null"]["control/inline"]["pooled"]["n"] == 19 * 3
    assert metrics["null"]["plain/inline"]["pooled"]["n"] == 19
    assert metrics["null"]["stitch_null/stitchnull"]["pooled"]["n"] == 19 * 3
    assert len(rows) == 19 * (8 + 8 + 3)


def test_flat_baseline_scores_zero_and_stepping_reference_scores_one(scored):
    _, rows, metrics = scored
    base, ref = metrics["systems"]["nl_instruct"]["by_set"], metrics["systems"]["stitch"]["by_set"]
    assert base["p2a"]["det_rate"] == base["test"]["det_rate"] == 0.0
    assert ref["p2a"]["det_rate"] == ref["heldout"]["det_rate"] == 1.0
    assert ref["p2a"]["realised_share"] == 1.0
    assert {r["outcome"] for r in rows if r["condition"] == "nl_instruct"} == {"flat"}
    paired = metrics["paired_reference_vs_baseline"]
    assert "all_sets" not in paired  # dev and val are never pooled
    assert (paired["p2a"]["reference_only"], paired["heldout"]["reference_only"]) == (12, 16)
    assert paired["heldout"]["phrases_reference_ahead"] == 4 and paired["heldout"]["phrase_sign_test_p"] == 0.125


def test_test_set_reads_the_stitched_boundary_by_forced_alignment(scored):
    _, rows, metrics = scored
    ref = metrics["systems"]["stitch"]["by_set"]
    assert ref["test"]["detected"] == 0 and ref["test"]["detected_by_rule"]["construction"] == 48
    assert ref["heldout"]["detected_by_rule"] == {"construction": 16, "construction+valid": 16,
                                                   "forced_alignment": 0, "forced_alignment+valid": 0}  # fmt: skip
    assert metrics["primary_rule"]["test"] == {"boundary": "forced_alignment", "validity_gate": True}
    assert {r["outcome"] for r in rows if r["condition"] == "stitch" and r["set"] == "test"} == {"misplaced"}


def test_no_switch_rows_never_pass_the_detector(scored):
    _, _, metrics = scored
    for name in ("control/inline", "plain/inline", "control/stitch", "stitch_null/stitchnull"):
        assert metrics["null"][name]["pooled"]["exceed_rate"] == 0.0, name
        assert metrics["null"][name]["pooled"]["abs_delta_max"] == 0.0, name
    assert metrics["null"]["stitch_null/stitchnull"]["by_set"]["test"]["exceed_by_boundary"] == {
        "construction": 0, "forced_alignment": 0}  # fmt: skip


def test_sensitivity_grid_and_seed_agreement_cover_every_condition(scored):
    cfg, _, metrics = scored
    grid = cfg["sensitivity"]
    assert len(metrics["sensitivity"]) == len(grid["min_abs_delta"]) * len(grid["max_changepoint_err_s"]) + 1
    assert set(metrics["sensitivity"][0]["counts"]) == {
        f"{c}/{s}" for c in ("nl_instruct", "stitch", "control", "plain", "stitch_null") for s in cfg["sets"]
    }
    assert metrics["seed_agreement"]["stitch/test"]["n_cells_multi_seed"] == 0  # one seed per cell here


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
    rule_row = "| reference: stitched segments | test (test) | 48/48 | 48/48 | 0/48 | 0/48 | 0 |"
    assert f"{rule_row} forced alignment + validity |" in text
    assert (out / "fig_steps.png").stat().st_size > 10_000
    assert (out / "fig_profiles.png").stat().st_size > 10_000

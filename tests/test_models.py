"""Checks that need the judge, aligner and ASR weights (about 5 GB on first use).

Not part of the default run:  uv run pytest -m models
"""

import numpy as np
import pytest

from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import measure, read_jsonl

pytestmark = pytest.mark.models
CFG = load_config("configs/baseline.yaml")
MEASURED = ("duration_s", "profile_centres_s", "profile_arousal", "words", "hyp", "wer", "leak", "step")


def test_remeasuring_released_clips_reproduces_the_committed_rows():
    manifest = read_jsonl(ROOT / "data/manifest.jsonl")
    wanted = {("inline_heldout_s42", "inline-nl-angry2sad-p1"), ("stitch_heldout_s42", "segcont-naive-angry2sad-p0")}
    subset = [m for m in manifest if (m["run"], m["id"]) in wanted]
    committed = {(r["run"], r["id"]): r for r in read_jsonl(ROOT / "results" / CFG["name"] / "rows.jsonl")}
    for row in measure(CFG, subset, "cpu"):
        old = committed[(row["run"], row["id"])]
        assert np.allclose(row["profile_arousal"], old["profile_arousal"], atol=0.01)
        assert row["step"]["switch_time_s"] == pytest.approx(old["step"]["switch_time_s"], abs=0.05)
        assert row["step"]["delta"] == pytest.approx(old["step"]["delta"], abs=0.005)
        assert {k: row[k] for k in MEASURED if k not in ("profile_arousal", "step", "words", "hyp", "wer")} == {
            k: old[k] for k in MEASURED if k not in ("profile_arousal", "step", "words", "hyp", "wer")
        }


def test_the_judge_does_not_read_playback_level():
    """Every window is rescaled before scoring, so a gain change must not move the profile."""
    from emoswitch.judge import ArousalJudge, load_audio, window_profile

    jc = CFG["judge"]
    judge = ArousalJudge(jc["model"], jc["revision"], jc["sample_rate"])
    clip = next(m for m in read_jsonl(ROOT / "data/manifest.jsonl") if m["condition"] == "nl_instruct")
    y = load_audio(str(ROOT / clip["audio"]), jc["sample_rate"], None)
    _, raw = window_profile(y, judge, jc["sample_rate"], jc["window_s"], jc["hop_s"])
    _, quiet = window_profile(y * 0.1, judge, jc["sample_rate"], jc["window_s"], jc["hop_s"])
    _, normalised = window_profile(
        load_audio(str(ROOT / clip["audio"]), jc["sample_rate"], jc["lufs"]),
        judge,
        jc["sample_rate"],
        jc["window_s"],
        jc["hop_s"],
    )
    assert np.max(np.abs(np.array(raw) - np.array(quiet))) < 0.005
    assert np.max(np.abs(np.array(raw) - np.array(normalised))) < 0.005

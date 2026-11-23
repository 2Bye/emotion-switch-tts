"""The in-pass systems: what can be checked without the TTS model."""

import json
import re

import numpy as np
import pytest
import torch

from emoswitch import experiments
from emoswitch.config import ROOT, load_config, load_phrases
from emoswitch.evaluate import measure_fingerprint
from emoswitch.inpass import HiddenSteer, PrefixRepetitionPenalty, anchor_frames, boundary_time
from emoswitch.stimuli import NULL_CONDITIONS, build_rows, inpass_systems

BASE = load_config("configs/baseline.yaml")
CFG = load_config("configs/inpass.yaml")


def test_the_inpass_config_measures_and_scores_like_the_baseline():
    shared = ("sets", "pairs", "adverbs", "anchor_arousal", "leak_stems", "speaker", "generation", "judge", "asr",
              "detector", "gates", "primary_rule", "stats")  # fmt: skip
    assert all(CFG[k] == BASE[k] for k in shared)
    assert measure_fingerprint(CFG) == measure_fingerprint(BASE)
    assert {k: v for k, v in CFG["systems"]["nl_instruct"].items() if k in BASE["systems"]["nl_instruct"]} == (
        BASE["systems"]["nl_instruct"]
    )


def test_inpass_rows_cover_every_system_and_switch():
    rows = build_rows(CFG, "p2a", "inpass", 42)
    systems = list(inpass_systems(CFG))
    assert systems == ["swap", "swap_anchor", "swap_steer", "swap_anchor_steer"]
    n_phrases = len(load_phrases(CFG, "p2a"))
    assert len(rows) == n_phrases * (3 + len(CFG["pairs"]) * len(systems))
    assert len({r["id"] for r in rows}) == len(rows)
    null = [r for r in rows if r["condition"] == "swap_null"]
    assert len(null) == 3 * n_phrases and all(r["expect_sign"] == 0 and r["emotion"] == r["emotion2"] for r in null)
    assert "swap_null" in NULL_CONDITIONS
    switch = [r for r in rows if r["condition"] != "swap_null"]
    assert {r["condition"] for r in switch} == set(systems)
    assert all(r["expect_sign"] in (-1, 1) and r["emotion"] != r["emotion2"] for r in switch)
    # every system gets the same commanded switches on the same phrases
    per_system = [
        {(r["phrase_id"], r["emotion"], r["emotion2"]) for r in switch if r["condition"] == s} for s in systems
    ]
    assert all(cells == per_system[0] for cells in per_system)


def test_the_four_systems_are_the_two_by_two_of_the_components():
    cells = {(spec["anchor"], spec["steer"]) for spec in inpass_systems(CFG).values()}
    assert cells == {(False, False), (True, False), (False, True), (True, True)}


def test_steering_sentences_are_in_no_evaluation_set():
    norm = lambda t: re.sub(r"[^a-z ]", "", t.lower())  # noqa: E731
    texts = json.loads((ROOT / CFG["inpass"]["steer"]["texts"]).read_text(encoding="utf-8"))["texts"]
    evaluated = [norm(p[s]) for name in CFG["sets"] for p in load_phrases(CFG, name) for s in ("s1", "s2")]
    assert len(set(texts)) == len(texts) == 24
    assert not [t for t in texts if any(norm(t) in e or e in norm(t) for e in evaluated)]
    stems = [s for group in CFG["leak_stems"].values() for s in group]
    assert not [t for t in texts if any(re.search(rf"\b{s}", t.lower()) for s in stems)]


@pytest.mark.parametrize(
    ("n_kept", "n_source", "expected"),
    [(50, 80, 30), (15, 80, 7), (8, 80, 0), (5, 80, 0), (50, 20, 19), (50, 1, 0)],
)
def test_anchor_frames_leave_a_real_tail(n_kept, n_source, expected):
    assert anchor_frames(n_kept, n_source, share=0.6, min_tail=8) == expected


def test_boundary_is_the_middle_of_the_pause_before_the_switch_word():
    words = [
        {"w": "later", "start": 0.1, "end": 0.4},
        {"w": "city", "start": 2.0, "end": 2.5},
        {"w": "later", "start": 3.1, "end": 3.5},
        {"w": "we", "start": 3.6, "end": 3.7},
    ]
    assert boundary_time(words, "Later", 6.0, 0.25) == pytest.approx(2.8)  # the early "later" is skipped
    assert boundary_time(words, "tomorrow", 6.0, 0.25) is None
    assert boundary_time(words[2:], "later", 6.0, 0.25) is None  # no word before it


def test_steering_shifts_only_the_last_position():
    layer = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(4))
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[[0.0, 3.0, 4.0, 0.0], [0.0, 0.0, 2.0, 0.0]]])
    with HiddenSteer(layer, 0.5, direction) as hook:
        y = layer(x)
    assert hook.calls == 1 and hook.handle is None
    assert torch.equal(y[:, 0], x[:, 0])
    assert torch.allclose(y[0, 1], torch.tensor([1.0, 0.0, 2.0, 0.0]))  # 0.5 * ||h|| = 1 along the direction
    assert torch.equal(layer(x), x)  # the hook is gone
    with HiddenSteer(layer, 0.0, direction):
        assert torch.equal(layer(x), x)  # alpha 0 is the identity
    with pytest.raises(ValueError):
        HiddenSteer(layer, 0.5, torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_repetition_penalty_counts_the_context_frames():
    scores = torch.tensor([[2.0, -2.0, 2.0, -2.0]])
    out = PrefixRepetitionPenalty(2.0, torch.tensor([[0, 1]]))(torch.tensor([[2]]), scores.clone())
    assert out.tolist() == [[1.0, -4.0, 1.0, -2.0]]


def _row(condition, seed, phrase, pair, detected):
    return {"condition": condition, "set": "p2a", "seed": seed, "phrase_id": phrase, "emotion": pair[0],
            "emotion2": pair[1], "detected": detected}  # fmt: skip


def test_contrast_pairs_rows_and_tests_over_phrases():
    pairs = [("sad", "angry"), ("angry", "sad")]
    rows = []
    for phrase, hits_a in (("p0", (True, True)), ("p1", (True, False)), ("p2", (False, False))):
        for pair, hit in zip(pairs, hits_a, strict=True):
            rows.append(_row("swap_anchor", 42, phrase, pair, hit))
            rows.append(_row("swap", 42, phrase, pair, False))
    rows.append(_row("swap_anchor", 43, "p0", pairs[0], True))  # no partner: not counted
    c = experiments.contrast(rows, CFG, "swap_anchor", "swap", "p2a")
    assert (c["n_pairs"], c["a_only"], c["b_only"]) == (6, 3, 0)
    assert (c["det_a"], c["det_b"]) == (0.5, 0.0)
    assert (c["n_phrases"], c["phrases_a_ahead"], c["phrases_b_ahead"]) == (3, 2, 0)
    assert c["sign_test_p"] == 0.5 and c["difference"] == pytest.approx(0.5)
    assert experiments.contrast(rows, CFG, "swap_anchor", "swap", "test") is None


def test_interaction_is_the_difference_of_differences():
    rates = {"swap": 0.0, "swap_anchor": 0.25, "swap_steer": 0.25, "swap_anchor_steer": 1.0}
    rows = []
    for name, rate in rates.items():
        for phrase in ("p0", "p1"):
            for i in range(4):
                rows.append(_row(name, 42 + i, phrase, ("sad", "angry"), i < rate * 4))
    out = experiments.interaction(rows, CFG, "p2a")
    assert out["interaction"] == pytest.approx((1.0 - 0.25) - (0.25 - 0.0))
    assert out["n_phrases"] == 2 and np.isclose(out["det"]["anchor=True,steer=True"], 1.0)


# ── the released in-pass data ───────────────────────────────────────────────

MANIFEST = experiments.read_jsonl(ROOT / CFG["manifest"])
RESULTS = ROOT / "results" / CFG["name"]
BUILT_FIELDS = ("id", "set", "seed", "phrase_id", "condition", "emotion", "emotion2", "text", "instruct",
                "switch_word", "leak_stems", "expect_sign")  # fmt: skip


def test_released_inpass_runs_are_what_the_config_builds():
    by_run = {}
    for m in MANIFEST:
        by_run.setdefault(m["run"], []).append(m)
    assert list(by_run) == [f"{kind}_{set_name}_s{seed}" for kind, set_name, seed in CFG["runs"]]
    for run, rows in by_run.items():
        kind, set_name, seed = run.split("_")
        assert [{k: r[k] for k in BUILT_FIELDS} for r in rows] == build_rows(CFG, set_name, kind, int(seed[1:])), run


def test_inpass_clips_are_single_decodes_with_a_render_record():
    for m in MANIFEST:
        assert m["switch_time_s"] is None, m["id"]  # no boundary by construction: scored like the baseline
        assert (ROOT / m["audio"]).exists(), m["audio"]
        render = m["render"]
        assert render.get("no_boundary") or render["kept_frames"] >= 3
        spec = inpass_systems(CFG).get(m["condition"])
        assert ("anchor_frames" in render) == bool(spec and spec["anchor"] and not render.get("no_boundary"))


def test_committed_inpass_rows_cover_the_manifest_and_score_to_the_committed_table():
    rows = experiments.read_jsonl(RESULTS / "rows.jsonl")
    key = lambda r: (r["run"], r["id"], r["sha256_wav_source"])  # noqa: E731
    assert sorted(map(key, rows)) == sorted(map(key, MANIFEST))
    assert all(r["switch_time_source"] == "forced_alignment" for r in rows)
    committed = json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8"))
    all_rows = experiments.load_rows(CFG, RESULTS)
    sets = {r["set"] for r in rows}
    again = [e for e in experiments.experiments(all_rows, CFG) if e["set"] in sets]
    assert json.loads(json.dumps(again)) == committed["experiments"]


def test_holm_adjustment():
    assert experiments.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert experiments.holm([0.5, 0.0005]) == pytest.approx([0.5, 0.001])

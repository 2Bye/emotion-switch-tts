"""The released audio, the manifest, the committed results and the README must tell one story."""

import json
import re

import soundfile as sf

from emoswitch.config import ROOT, load_config
from emoswitch.evaluate import read_jsonl, score
from emoswitch.stimuli import build_rows

CFG = load_config("configs/baseline.yaml")
MANIFEST = read_jsonl(ROOT / "data/manifest.jsonl")
RESULTS = ROOT / "results" / CFG["name"]
BUILT_FIELDS = ("id", "set", "seed", "phrase_id", "condition", "emotion", "emotion2", "text", "instruct",
                "switch_word", "leak_stems", "expect_sign")  # fmt: skip


def runs() -> dict:
    out = {}
    for m in MANIFEST:
        out.setdefault(m["run"], []).append(m)
    return out


def test_released_runs_are_the_runs_of_the_config():
    assert list(runs()) == [f"{kind}_{set_name}_s{seed}" for kind, set_name, seed in CFG["runs"]]


def test_every_run_is_exactly_what_the_config_builds():
    for run, rows in runs().items():
        kind, set_name, seed = run.split("_")
        built = build_rows(CFG, set_name, kind, int(seed[1:]))
        assert [{k: r[k] for k in BUILT_FIELDS} for r in rows] == built, run


def test_audio_exists_is_mono_and_matches_the_manifest():
    assert len({m["audio"] for m in MANIFEST}) == len(MANIFEST)
    for m in MANIFEST:
        info = sf.info(ROOT / m["audio"])
        assert (info.channels, info.samplerate, info.format) == (1, m["sample_rate"], "FLAC"), m["audio"]


def test_only_stitched_clips_carry_a_boundary_by_construction():
    for m in MANIFEST:
        assert (m["switch_time_s"] is not None) == (m["condition"] == "stitch"), m["id"]


def test_evaluation_texts_contain_no_instruction_words_and_no_digits():
    stems = [s for group in CFG["leak_stems"].values() for s in group]
    for m in MANIFEST:
        assert not re.search(r"[\d\[\]]", m["text"])
        assert not any(re.search(rf"\b{s}", m["text"].lower()) for s in stems), m["text"]


def test_switch_word_is_unique_in_its_phrase():
    for m in MANIFEST:
        words = re.findall(r"[a-z']+", m["text"].lower())
        assert words.count(m["switch_word"]) == 1, m["text"]


def test_committed_rows_cover_the_manifest():
    rows = read_jsonl(RESULTS / "rows.jsonl")
    assert sorted((r["run"], r["id"]) for r in rows) == sorted((m["run"], m["id"]) for m in MANIFEST)


def test_committed_metrics_are_what_the_committed_rows_score_to():
    rows = read_jsonl(RESULTS / "rows.jsonl")
    committed = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    assert score(CFG, rows) == committed


def test_detector_decisions_agree_with_the_gpu_node():
    """Same clips, same method, different hardware: every decision must come out the same."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_with_reference import compare

    summary, mismatches, unreferenced = compare(CFG)
    assert not mismatches
    assert sum(s["n"] for s in summary) + sum(unreferenced.values()) == len(MANIFEST)
    for s in summary:
        assert s["same_grid"] == s["n"] and s["same_decision"] == s["n"], s["run"]
        assert s["arousal_max"] <= 0.01 and s["delta_max"] <= 0.005, s


def test_released_dev_clips_are_the_clips_the_detector_was_calibrated_on():
    """data/reference/calibration_run_sha256.tsv lists the wav hashes of the first dev run (2026-09-04)."""
    first = {}
    for line in (ROOT / "data/reference/calibration_run_sha256.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            name, sha, _ = line.split("\t")
            first[name] = sha
    dev = [m for m in MANIFEST if m["run"] == "inline_p2a_s42"]
    assert len(dev) == 24 and all(first[m["id"] + ".wav"] == m["sha256_wav_source"] for m in dev)


def test_readme_tables_are_copies_of_the_generated_ones():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from sync_readme import BLOCK, render

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(BLOCK.findall(readme)) >= 5
    assert render(readme, RESULTS) == readme

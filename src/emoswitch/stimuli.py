"""What is asked of the model: the rows of one render run, built from the config alone.

A row is one utterance to synthesise. Its `condition` is one of the four systems of the
config: `nl_instruct` (baseline), `stitch` (reference), `control` and `plain` (null rows).
"""

from __future__ import annotations

from emoswitch.config import load_phrases

RUN_KINDS = ("inline", "stitch")
NULL_CONDITIONS = ("control", "plain")


def phrase_text(phrase: dict) -> str:
    return f"{phrase['s1']} {phrase['s2']}"


def expected_sign(cfg: dict, e1: str, e2: str) -> int:
    """+1 if the commanded switch raises arousal, -1 if it lowers it."""
    anchor = cfg["anchor_arousal"]
    return 1 if anchor[e2] > anchor[e1] else -1


def emotions(cfg: dict) -> list[str]:
    return sorted({e for pair in cfg["pairs"] for e in pair})


def build_rows(cfg: dict, set_name: str, kind: str, seed: int) -> list[dict]:
    """Rows of one run: `inline` holds the baseline and its null rows, `stitch` the reference
    and its null rows. Order and ids follow the render scripts of the source project."""
    if kind not in RUN_KINDS:
        raise ValueError(f"unknown run kind {kind!r}")
    systems, adv, stems = cfg["systems"], cfg["adverbs"], cfg["leak_stems"]
    prefix = "inline" if kind == "inline" else "segcont"
    rows = []
    for phrase in load_phrases(cfg, set_name):
        pid, text = phrase["id"], phrase_text(phrase)
        common = {"set": set_name, "seed": seed, "phrase_id": pid, "text": text, "switch_word": phrase["switch_word"]}
        rows.append(
            {
                **common,
                "id": f"{prefix}-plain-none-{pid}",
                "condition": "plain",
                "emotion": None,
                "emotion2": None,
                "instruct": None,
                "leak_stems": [],
                "expect_sign": 0,
            }
        )
        for e in emotions(cfg):
            rows.append(
                {
                    **common,
                    "id": f"{prefix}-control-{e}-{pid}",
                    "condition": "control",
                    "emotion": e,
                    "emotion2": e,
                    "instruct": systems["control"]["instruct"].format(adv=adv[e]),
                    "leak_stems": list(stems[e]),
                    "expect_sign": 0,
                }
            )
        for e1, e2 in cfg["pairs"]:
            switch = {
                **common,
                "emotion": e1,
                "emotion2": e2,
                "leak_stems": list(stems[e1]) + list(stems[e2]),
                "expect_sign": expected_sign(cfg, e1, e2),
            }
            if kind == "inline":
                instruct = systems["nl_instruct"]["instruct"].format(adv1=adv[e1], adv2=adv[e2])
                rows.append(
                    {**switch, "id": f"inline-nl-{e1}2{e2}-{pid}", "condition": "nl_instruct", "instruct": instruct}
                )
            else:
                seg = systems["stitch"]["instruct"]
                rows.append(
                    {
                        **switch,
                        "id": f"segcont-naive-{e1}2{e2}-{pid}",
                        "condition": "stitch",
                        "instruct": f"seg1: {seg.format(adv=adv[e1])} | seg2: {seg.format(adv=adv[e2])}",
                    }
                )
    return rows

"""What is asked of the model: the rows of one render run, built from the config alone.

A row is one utterance to synthesise. Its `condition` is `nl_instruct` (baseline), `stitch`
(reference), an in-pass system (config systems with role `inpass`), or one of the no-switch conditions:
`control` and `plain` (single pass), `stitch_null` (the null row of the reference: two segments of
the same emotion, joined) and `swap_null` (the null row of the in-pass systems: the two-phase
mechanics with the same instruction in both phases).
"""

from __future__ import annotations

from emoswitch.config import load_phrases

RUN_KINDS = ("inline", "stitch", "stitchnull", "inpass")
NULL_CONDITIONS = ("control", "plain", "stitch_null", "swap_null")


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
        if kind == "inpass":
            rows += inpass_rows(cfg, common)
            continue
        if kind == "stitchnull":
            seg = systems["stitch"]["instruct"]
            for e in emotions(cfg):
                rows.append(
                    {
                        **common,
                        "id": f"segcont-null-{e}-{pid}",
                        "condition": "stitch_null",
                        "emotion": e,
                        "emotion2": e,
                        "instruct": f"seg1: {seg.format(adv=adv[e])} | seg2: {seg.format(adv=adv[e])}",
                        "leak_stems": list(stems[e]),
                        "expect_sign": 0,
                    }
                )
            continue
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


def inpass_systems(cfg: dict) -> dict:
    """Name -> spec of the in-pass systems: which components (`anchor`, `steer`) each switches on."""
    return {name: spec for name, spec in cfg["systems"].items() if spec.get("role") == "inpass"}


def inpass_rows(cfg: dict, common: dict) -> list[dict]:
    """Rows of one phrase in an `inpass` run: the null row of every emotion, then every commanded switch
    under every in-pass system of the config."""
    adv, stems, ip = cfg["adverbs"], cfg["leak_stems"], cfg["inpass"]
    pid = common["phrase_id"]
    rows = []
    for e in emotions(cfg):
        ins = ip["instruct"].format(adv=adv[e])
        rows.append(
            {
                **common,
                "id": f"inpass-null-{e}-{pid}",
                "condition": "swap_null",
                "emotion": e,
                "emotion2": e,
                "instruct": f"phase 1: {ins} | phase 2: {ins}",
                "leak_stems": list(stems[e]),
                "expect_sign": 0,
            }
        )
    for e1, e2 in cfg["pairs"]:
        for name in inpass_systems(cfg):
            rows.append(
                {
                    **common,
                    "id": f"inpass-{name}-{e1}2{e2}-{pid}",
                    "condition": name,
                    "emotion": e1,
                    "emotion2": e2,
                    "instruct": f"phase 1: {ip['instruct'].format(adv=adv[e1])} | "
                    f"phase 2: {ip['instruct'].format(adv=adv[e2])}",
                    "leak_stems": list(stems[e1]) + list(stems[e2]),
                    "expect_sign": expected_sign(cfg, e1, e2),
                }
            )
    return rows

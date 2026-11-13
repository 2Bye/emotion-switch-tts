"""The switch detector: does the arousal profile step at the commanded place, in the commanded direction?"""

from __future__ import annotations

import re

import numpy as np


def norm_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def switch_time(words: list[dict], switch_word: str, duration_s: float, search_from_frac: float) -> float | None:
    """Start time of the first sentence-2 word, read from the aligned words.

    The first quarter of the clip is skipped so that an earlier occurrence cannot be picked.
    """
    target = norm_word(switch_word)
    late = [w for w in words if w["start"] >= search_from_frac * duration_s]
    hits = [w for w in late if w["w"] == target] or [w for w in late if w["w"].startswith(target[:4])]
    return hits[0]["start"] if hits else None


def best_split(arousal: np.ndarray, min_side: int, weighted: bool = False) -> tuple[int | None, float]:
    """Index i maximising |mean(a[i:]) - mean(a[:i])| with at least `min_side` windows on each side.

    The plain gap is the frozen estimator. It favours splits near the clip edges, where one side is
    short and its mean is noisy. `weighted=True` scales the gap by sqrt(n1*n2/n), the usual two-sample
    weighting, and is used only in the sensitivity analysis.
    """
    best_i, best_gap, n = None, -1.0, len(arousal)
    for i in range(min_side, n - min_side + 1):
        gap = abs(arousal[i:].mean() - arousal[:i].mean())
        if weighted:
            gap *= np.sqrt(i * (n - i) / n)
        if gap > best_gap:
            best_gap, best_i = gap, i
    return best_i, best_gap


def measure_step(
    centres: list[float],
    arousal: list[float],
    t_switch: float | None,
    expect_sign: int,
    min_seg_windows: int,
    weighted_split: bool = False,
) -> dict | None:
    """Step of the profile at `t_switch` and the profile's own best change-point.

    delta = mean arousal of the windows centred at or after the switch minus the mean before it.
    Returns None when the switch time is unknown, the clip has fewer than three windows or one side
    of the switch has no window.
    """
    a, c = np.asarray(arousal, dtype=float), np.asarray(centres, dtype=float)
    if t_switch is None or len(a) < 3:
        return None
    before, after = a[c < t_switch], a[c >= t_switch]
    if not len(before) or not len(after):
        return None
    delta = float(after.mean() - before.mean())
    split, gap = best_split(a, max(1, min_seg_windows), weighted_split)
    t_cp = None if split is None else float(c[split])
    return {
        "switch_time_s": round(float(t_switch), 3),
        "mean_before": round(float(before.mean()), 4),
        "mean_after": round(float(after.mean()), 4),
        "delta": round(delta, 4),
        "sign_ok": None if expect_sign == 0 else bool(np.sign(delta) == np.sign(expect_sign)),
        "changepoint_s": None if t_cp is None else round(t_cp, 3),
        "changepoint_gap": None if split is None else round(float(gap), 4),
        "changepoint_err_s": None if t_cp is None else round(abs(t_cp - float(t_switch)), 3),
    }


def _at_boundary(step: dict, max_changepoint_err_s: float | None) -> bool:
    """None switches the change-point condition off (sensitivity analysis only)."""
    if max_changepoint_err_s is None:
        return True
    return step["changepoint_err_s"] is not None and step["changepoint_err_s"] <= max_changepoint_err_s


def is_detected(step: dict | None, min_abs_delta: float, max_changepoint_err_s: float | None) -> bool:
    """A switch counts when the step is large enough, has the commanded sign and sits at the boundary."""
    if not step:
        return False
    return bool(abs(step["delta"]) >= min_abs_delta and step["sign_ok"] and _at_boundary(step, max_changepoint_err_s))


def exceeds_null(step: dict | None, min_abs_delta: float, max_changepoint_err_s: float | None) -> bool:
    """The same test without the sign: what a no-switch row must NOT do. Its rate over the
    null rows is an upper bound on the detector's false-positive rate."""
    if not step:
        return False
    return bool(abs(step["delta"]) >= min_abs_delta and _at_boundary(step, max_changepoint_err_s))

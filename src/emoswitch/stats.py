"""Uncertainty for small evaluation sets: intervals for a rate, a clustered bootstrap, a paired test."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def wilson_ci(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = float(sps.norm.ppf(0.5 + ci / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(
    values: list[float], clusters: list[str], n_boot: int, seed: int, ci: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the mean, resampling whole clusters with replacement.

    Rows that share a phrase share its text and its sentence melody, so they are not
    independent; resampling phrases instead of rows keeps that dependence in the interval.
    """
    vals, cl = np.asarray(values, dtype=float), np.asarray(clusters)
    groups = [vals[cl == c] for c in sorted(set(clusters))]
    if not groups:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        means[b] = np.concatenate([groups[i] for i in picked]).mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(lo), float(hi))


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired outcomes (same phrase, pair and seed in both systems)."""
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = a_only + b_only
    p = 1.0 if n == 0 else float(sps.binomtest(a_only, n, 0.5).pvalue)
    return {"n_pairs": len(a), "a_only": a_only, "b_only": b_only, "p_value": p}


def sign_test(a_wins: int, b_wins: int) -> float:
    """Exact two-sided sign test on independent units (here: phrases), ties dropped."""
    n = a_wins + b_wins
    return 1.0 if n == 0 else float(sps.binomtest(a_wins, n, 0.5).pvalue)

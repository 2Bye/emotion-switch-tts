import numpy as np
import pytest

from emoswitch import gates, stats
from emoswitch.judge import window_profile

SR = 16_000


class CountingJudge:
    """Stands in for the model: returns the window length so the windowing can be checked."""

    def __call__(self, y):
        return {"arousal": len(y) / SR}


def test_window_profile_grid_on_a_five_second_clip():
    centres, arousal = window_profile(np.zeros(5 * SR, dtype=np.float32), CountingJudge(), SR, 1.0, 0.25)
    assert len(centres) == 17  # starts 0.00 … 4.00 s every 0.25 s
    assert centres[0] == 0.5 and centres[-1] == 4.5
    assert set(arousal) == {1.0}  # every window is exactly one second long


def test_window_profile_ignores_a_leftover_shorter_than_the_hop():
    centres, _ = window_profile(np.zeros(int(5.2 * SR), dtype=np.float32), CountingJudge(), SR, 1.0, 0.25)
    assert centres[-1] == 4.5  # the last 0.2 s is not long enough for a window of its own


def test_window_profile_on_a_clip_shorter_than_the_window():
    centres, arousal = window_profile(np.zeros(int(0.6 * SR), dtype=np.float32), CountingJudge(), SR, 1.0, 0.25)
    assert centres == [0.3] and arousal == [0.6]


def test_wer_ignores_case_and_punctuation():
    assert gates.word_error_rate("Tomorrow the office opens at nine.", " tomorrow the office opens at nine") == 0.0
    assert gates.word_error_rate("one two three four", "one two three") == 0.25
    assert gates.word_error_rate("one two", "") == 1.0


def test_leak_fires_on_a_spoken_instruction_word():
    stems = ["sad", "sadly", "happy", "happi", "cheerful"]
    assert gates.leaks("Sadly, we took the early train.", "We took the early train.", stems)
    assert gates.leaks("[happy] We took the early train.", "We took the early train.", stems)
    assert not gates.leaks("We took the early train.", "We took the early train.", stems)


def test_leak_does_not_fire_on_a_digit_the_target_explains():
    assert not gates.leaks("Tomorrow the office opens at 9.", "Tomorrow the office opens at nine.", [])
    assert gates.leaks("Tomorrow the office opens at 5.", "Tomorrow the office opens at nine.", [])


def test_wilson_interval_known_values():
    lo, hi = stats.wilson_ci(0, 28)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.1206, abs=1e-4)
    lo, hi = stats.wilson_ci(10, 16)
    assert lo == pytest.approx(0.3864, abs=1e-4) and hi == pytest.approx(0.8152, abs=1e-4)


def test_cluster_bootstrap_is_wider_than_the_row_bootstrap_when_clusters_agree_internally():
    values = [1.0] * 8 + [0.0] * 8
    by_phrase = stats.cluster_bootstrap_ci(values, ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4, 2000, 0)
    by_row = stats.cluster_bootstrap_ci(values, [str(i) for i in range(16)], 2000, 0)
    assert (by_phrase[1] - by_phrase[0]) > (by_row[1] - by_row[0])


def test_cluster_bootstrap_is_reproducible():
    values, clusters = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0], ["a", "a", "b", "b", "c", "c"]
    assert stats.cluster_bootstrap_ci(values, clusters, 500, 7) == stats.cluster_bootstrap_ci(values, clusters, 500, 7)


def test_mcnemar_counts_discordant_pairs():
    out = stats.mcnemar_exact([True] * 9 + [False] * 3, [False] * 12)
    assert (out["a_only"], out["b_only"]) == (9, 0)
    assert out["p_value"] == pytest.approx(2 * 0.5**9)
    assert stats.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0

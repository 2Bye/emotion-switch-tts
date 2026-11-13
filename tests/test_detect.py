import numpy as np

from emoswitch import detect

THR, TOL = 0.15, 0.75


def profile(levels_before: float, levels_after: float, n_before: int = 10, n_after: int = 10):
    """Synthetic profile: windows every 0.25 s starting at 0.5 s, a clean step between the halves."""
    centres = [round(0.5 + 0.25 * i, 3) for i in range(n_before + n_after)]
    arousal = [levels_before] * n_before + [levels_after] * n_after
    return centres, arousal, centres[n_before]


def test_clean_step_is_detected():
    centres, arousal, t = profile(0.6, 0.3)
    step = detect.measure_step(centres, arousal, t, expect_sign=-1, min_seg_windows=3)
    assert step["delta"] == -0.3
    assert step["sign_ok"] is True
    assert step["changepoint_err_s"] == 0.0
    assert detect.is_detected(step, THR, TOL)


def test_wrong_direction_is_not_detected():
    centres, arousal, t = profile(0.6, 0.3)
    step = detect.measure_step(centres, arousal, t, expect_sign=+1, min_seg_windows=3)
    assert step["sign_ok"] is False
    assert not detect.is_detected(step, THR, TOL)


def test_small_step_is_not_detected():
    centres, arousal, t = profile(0.50, 0.40)
    step = detect.measure_step(centres, arousal, t, expect_sign=-1, min_seg_windows=3)
    assert abs(step["delta"]) < THR
    assert not detect.is_detected(step, THR, TOL)


def test_step_far_from_the_boundary_is_not_detected():
    """The profile steps 1.5 s before the commanded word: large, right sign, wrong place."""
    centres, arousal, _ = profile(0.6, 0.3, n_before=6, n_after=14)
    t_commanded = centres[12]
    step = detect.measure_step(centres, arousal, t_commanded, expect_sign=-1, min_seg_windows=3)
    assert step["changepoint_err_s"] == 1.5
    assert not detect.is_detected(step, 0.05, TOL)


def test_zero_changepoint_error_counts_as_a_value():
    """A perfect change-point (error 0.0) must not be read as 'missing'."""
    centres, arousal, t = profile(0.3, 0.6)
    step = detect.measure_step(centres, arousal, t, expect_sign=+1, min_seg_windows=3)
    assert step["changepoint_err_s"] == 0.0
    assert detect.is_detected(step, THR, TOL)


def test_null_row_has_no_sign_and_is_never_detected_as_a_switch():
    centres, arousal, t = profile(0.6, 0.3)
    step = detect.measure_step(centres, arousal, t, expect_sign=0, min_seg_windows=3)
    assert step["sign_ok"] is None
    assert not detect.is_detected(step, THR, TOL)
    assert detect.exceeds_null(step, THR, TOL)


def test_best_split_respects_the_minimum_side():
    a = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert detect.best_split(a, 1)[0] == 1  # a single outlier window wins with min_side=1
    assert detect.best_split(a, 3)[0] == 3  # and cannot with three windows per side
    assert detect.best_split(a[:5], 3) == (None, -1.0)


def test_missing_switch_time_gives_no_step():
    centres, arousal, _ = profile(0.6, 0.3)
    assert detect.measure_step(centres, arousal, None, -1, 3) is None
    assert not detect.is_detected(None, THR, TOL)


def test_switch_time_skips_an_early_occurrence_and_falls_back_to_a_prefix():
    words = [
        {"w": "later", "start": 0.2, "end": 0.5},
        {"w": "we", "start": 0.6, "end": 0.7},
        {"w": "later", "start": 2.4, "end": 2.8},
    ]
    assert detect.switch_time(words, "Later", 6.0, 0.25) == 2.4
    words = [{"w": "afterward", "start": 3.1, "end": 3.6}]
    assert detect.switch_time(words, "afterwards", 6.0, 0.25) == 3.1
    assert detect.switch_time(words, "tomorrow", 6.0, 0.25) is None

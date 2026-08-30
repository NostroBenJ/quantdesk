"""Screening statistics, especially the overlap correction.

The correction is the load-bearing part: without it, overlapping windows
report a t-statistic several times too large, and the inflation grows with
the horizon — so the longer the horizon you care about, the more wrong the
naive answer gets. These tests pin it against a series whose true standard
error is known by construction.
"""

from __future__ import annotations

import math
import random
import statistics as st

import pytest

from quantdesk.data.research_stats import (
    Screen,
    inv_norm,
    overlap_corrected,
    stats,
    welch,
    welch_overlap,
)


def test_stats_against_hand_arithmetic():
    s = stats([10.0, 20.0, 30.0])
    assert s["mean"] == pytest.approx(20.0)
    assert s["sd"] == pytest.approx(10.0)
    assert s["t"] == pytest.approx(20.0 * math.sqrt(3) / 10.0)
    assert (s["lo"] + s["hi"]) / 2 == pytest.approx(s["mean"])


def test_stats_needs_two_observations():
    assert stats([]) is None
    assert stats([1.0]) is None


@pytest.mark.parametrize("p,want", [
    (0.975, 1.959963985), (0.995, 2.575829304), (0.5, 0.0),
])
def test_inv_norm_known_quantiles(p, want):
    assert inv_norm(p) == pytest.approx(want, abs=1e-6)


def test_inv_norm_is_symmetric():
    assert inv_norm(0.9) + inv_norm(0.1) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------- overlap correction


def block_series(horizon: int, blocks: int, seed: int = 0):
    """A series with a KNOWN number of independent observations.

    Each block is one draw repeated `horizon` times, which is exactly what
    overlapping windows do to a return series: many rows, few independent
    facts. The true standard error is computable from the blocks.
    """
    rng = random.Random(seed)
    truth = [rng.gauss(0.0, 100.0) for _ in range(blocks)]
    series = [v for v in truth for _ in range(horizon)]
    true_se = st.stdev(truth) / math.sqrt(blocks)
    return series, true_se


def test_correction_recovers_the_true_standard_error():
    series, true_se = block_series(horizon=21, blocks=300)
    corrected = overlap_corrected(series, 21)
    assert corrected["se"] == pytest.approx(true_se, rel=0.10)


def test_naive_standard_error_is_too_tight_by_sqrt_horizon():
    horizon = 21
    series, true_se = block_series(horizon=horizon, blocks=300)
    naive = stats(series)
    assert true_se / naive["se"] == pytest.approx(math.sqrt(horizon), rel=0.15)


@pytest.mark.parametrize("horizon", [5, 21, 63])
def test_correction_holds_across_horizons(horizon):
    series, true_se = block_series(horizon=horizon, blocks=200, seed=horizon)
    assert overlap_corrected(series, horizon)["se"] == pytest.approx(
        true_se, rel=0.15)


def test_the_inflation_grows_with_horizon():
    """The longer the horizon, the more wrong the uncorrected answer.

    This is why the trap bites hardest exactly where the interesting
    hypotheses live -- quarterly and annual return predictability.
    """
    ratios = []
    for horizon in (5, 21, 63):
        series, _ = block_series(horizon=horizon, blocks=200, seed=1)
        c = overlap_corrected(series, horizon)
        ratios.append(abs(c["naive_t"]) / max(abs(c["t"]), 1e-12))
    assert ratios[0] < ratios[1] < ratios[2]


def test_correction_leaves_the_mean_alone():
    """It widens the error bar; it must not move the estimate."""
    series, _ = block_series(horizon=21, blocks=100)
    assert overlap_corrected(series, 21)["mean"] == pytest.approx(
        stats(series)["mean"])


def test_horizon_one_is_the_uncorrected_case():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert overlap_corrected(xs, 1)["se"] == pytest.approx(stats(xs)["se"])


def test_welch_overlap_widens_relative_to_plain_welch():
    a, _ = block_series(horizon=21, blocks=100, seed=2)
    b, _ = block_series(horizon=21, blocks=100, seed=3)
    assert abs(welch_overlap(a, b, 21)["t"]) < abs(welch(a, b)["t"])


def test_welch_uses_unpooled_variance():
    """Two regimes are not two samples from one distribution."""
    a = [10.0] * 50 + [-10.0] * 50          # wide
    b = [0.1, -0.1] * 50                     # narrow
    w = welch(a, b)
    assert w["diff"] == pytest.approx(st.mean(a) - st.mean(b))
    # The wide sample must dominate the combined error.
    assert w["se"] > stats(b)["se"] * 10


# --------------------------------------------------------------- the screen


def test_threshold_gets_stricter_with_more_tests():
    s = Screen()
    assert s.threshold(1) == pytest.approx(1.959963985, abs=1e-6)
    assert s.threshold(20) == pytest.approx(3.023341439, abs=1e-5)
    assert s.threshold(20) > s.threshold(1)


def test_screen_counts_every_recorded_test(capsys):
    s = Screen()
    for k in range(4):
        s.record("t%d" % k, [1.0, 2.0, 3.0])
    assert s.n_tests == 4


def test_too_few_observations_still_counts_as_a_test(capsys):
    """A test that could not be run was still a look at the data."""
    s = Screen()
    s.record("empty", [])
    s.record("single", [1.0])
    assert s.n_tests == 2


def test_cost_floor_is_flagged(capsys):
    s = Screen(cost_bp=5.0)
    s.record("tiny", [0.5, 0.6, 0.4, 0.5])
    out = capsys.readouterr().out
    assert "below the 5.0bp cost floor" in out


def test_cost_floor_not_flagged_when_not_costed(capsys):
    s = Screen(cost_bp=5.0)
    s.record("vol", [0.5, 0.6, 0.4, 0.5], unit="vol pts", costed=False)
    assert "cost floor" not in capsys.readouterr().out

import numpy as np
import pytest

from bench.metrics import QUANTILES, coverage, crps, gmean, mase, seasonal_error
from bench.transforms import (
    cap_to_history,
    clip_nonneg,
    flip_average,
    hampel_clip,
    impute_linear,
    round_counts,
    scale_spread,
    sort_quantiles,
)


def test_seasonal_error_ignores_nans():
    history = np.array([1.0, 2.0, np.nan, 4.0, 6.0])
    assert seasonal_error(history, 1) == pytest.approx(1.5)  # diffs 1, 2 (pairs with NaN dropped)


def test_mase_matches_hand_computation():
    assert mase(np.array([1.0, 3.0, np.nan]), np.array([2.0, 2.0, 0.0]), scale=2.0) == pytest.approx(0.5)


def test_mase_zero_scale_is_nan():
    assert np.isnan(mase(np.ones(3), np.zeros(3), scale=0.0))


def test_crps_of_point_forecast_equals_relative_mae():
    y = np.array([[1.0, 2.0, 3.0]])
    point = np.array([[2.0, 2.0, 5.0]])
    q = np.repeat(point[..., None], len(QUANTILES), axis=-1)
    assert crps(y, q) == pytest.approx(np.abs(y - point).sum() / np.abs(y).sum())


def test_crps_perfect_forecast_is_zero_and_ignores_nan_targets():
    y = np.array([[1.0, np.nan, 3.0]])
    q = np.repeat(np.nan_to_num(y, nan=99.0)[..., None], 9, axis=-1)
    assert crps(y, q) == pytest.approx(0.0)


def test_coverage_counts_points_inside_outer_quantiles():
    q = np.tile(np.linspace(0, 8, 9), (1, 2, 1))
    assert coverage(np.array([[4.0, 9.0]]), q) == pytest.approx(0.5)


def test_gmean_skips_nan():
    assert gmean([1.0, 4.0, np.nan]) == pytest.approx(2.0)


def test_gmean_keeps_inf_so_blowups_are_visible():
    assert gmean([1.0, np.inf]) == np.inf


def test_cap_to_history_limits_blown_up_quantiles():
    q = np.array([[[1.0, 1e135, np.inf] + [0.0] * 6]])
    capped = cap_to_history(q, np.array([5.0]))
    assert capped.max() == 50.0 and capped[0, 0, 0] == 1.0


def test_impute_linear_fills_interior_and_edges():
    out = impute_linear(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
    np.testing.assert_allclose(out, [1.0, 1.0, 2.0, 3.0, 3.0])


def test_hampel_replaces_spike_but_keeps_intermittent_zeros():
    x = np.sin(np.arange(200) / 5.0)
    x[100] = 50.0
    assert abs(hampel_clip(x, season=1)[100]) < 2
    sparse = np.zeros(50)
    sparse[[10, 30]] = 3.0
    np.testing.assert_array_equal(hampel_clip(sparse, season=1), sparse)


def test_postprocessing_ops():
    q = np.array([[[-1.0, 0.4, 0.2, 1.6, 2.0, 2.5, 3.0, 3.0, 4.0]]])
    assert (np.diff(sort_quantiles(q), axis=-1) >= 0).all()
    assert clip_nonneg(q, np.array([True])).min() == 0.0
    assert clip_nonneg(q, np.array([False])).min() == -1.0
    np.testing.assert_array_equal(round_counts(q, np.array([True])), np.round(q))
    np.testing.assert_allclose(scale_spread(q, 0.0), np.full_like(q, 2.0))


def test_flip_average_is_identity_for_symmetric_model():
    q = np.sort(np.random.default_rng(0).normal(size=(2, 3, 9)), axis=-1)
    np.testing.assert_allclose(flip_average(q, -q[..., ::-1]), q)

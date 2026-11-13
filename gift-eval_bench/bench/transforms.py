"""Pre-processing (applied to the context) and post-processing (applied to quantile forecasts)."""

import warnings

import numpy as np
import pandas as pd


def impute_linear(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    return pd.Series(x).interpolate().ffill().bfill().fillna(0.0).to_numpy()


def hampel_clip(x: np.ndarray, season: int, n_sigmas: float = 5.0) -> np.ndarray:
    """Replace points more than n_sigmas robust deviations from a rolling median with that median."""
    half = max(season, 3)
    padded = np.pad(x.astype(float), half, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(windows, axis=1)
        mad = np.nanmedian(np.abs(windows - med[:, None]), axis=1) * 1.4826
        outlier = (np.abs(x - med) > n_sigmas * mad) & (mad > 0)
    return np.where(outlier, med, x)


def can_log(x: np.ndarray) -> bool:
    finite = x[np.isfinite(x)]
    return finite.size > 0 and finite.min() >= 0


def is_integer_valued(x: np.ndarray) -> bool:
    finite = x[np.isfinite(x)]
    return finite.size > 0 and np.allclose(finite, np.round(finite))


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    return np.sort(q, axis=-1)


def clip_nonneg(q: np.ndarray, nonneg: np.ndarray) -> np.ndarray:
    """nonneg: (n,) bool per task, True when the history never went below zero."""
    return np.where(nonneg[:, None, None], np.maximum(q, 0.0), q)


def round_counts(q: np.ndarray, integer: np.ndarray) -> np.ndarray:
    return np.where(integer[:, None, None], np.round(q), q)


def cap_to_history(q: np.ndarray, hist_max: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Clip quantiles above factor x the largest value seen in each task's history."""
    return np.minimum(q, factor * hist_max[:, None, None])


def scale_spread(q: np.ndarray, factor: float, median: int = 4) -> np.ndarray:
    mid = q[..., median : median + 1]
    return mid + factor * (q - mid)


def flip_average(q: np.ndarray, q_neg: np.ndarray) -> np.ndarray:
    """Average a forecast of x with the negated forecast of -x (quantile a of -Y is -quantile 1-a of Y)."""
    return 0.5 * (q - q_neg[..., ::-1])

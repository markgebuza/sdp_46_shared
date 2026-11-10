"""MASE and CRPS (mean weighted quantile loss), following the GIFT-Eval definitions."""

import numpy as np

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MEDIAN = 4


def seasonal_error(history: np.ndarray, season: int) -> float:
    diffs = np.abs(history[season:] - history[:-season])
    diffs = diffs[np.isfinite(diffs)]
    return float(diffs.mean()) if diffs.size else np.nan


def mase(target: np.ndarray, point: np.ndarray, scale: float) -> float:
    ok = np.isfinite(target)
    if not ok.any() or not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.abs(target[ok] - point[ok]).mean() / scale)


def quantile_loss_sums(targets: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, float]:
    """targets (n, H), quantiles (n, H, 9) -> per-level summed loss (9,), sum |y|. NaN targets ignored."""
    ok = np.isfinite(targets)
    y = np.where(ok, targets, 0.0)[..., None]
    q = np.where(ok[..., None], quantiles, 0.0)
    loss = 2 * np.abs((q - y) * ((y <= q) - QUANTILES))
    return loss.sum(axis=(0, 1)), float(np.abs(y).sum())


def crps(targets: np.ndarray, quantiles: np.ndarray) -> float:
    """Mean weighted quantile loss over levels 0.1..0.9 (GIFT-Eval's CRPS approximation)."""
    loss, denom = quantile_loss_sums(targets, quantiles)
    return float(np.mean(loss / denom)) if denom > 0 else np.nan


def coverage(targets: np.ndarray, quantiles: np.ndarray, lo: int = 0, hi: int = 8) -> float:
    ok = np.isfinite(targets)
    inside = (targets >= quantiles[..., lo]) & (targets <= quantiles[..., hi])
    return float(inside[ok].mean()) if ok.any() else np.nan


def gmean(values) -> float:
    """Geometric mean ignoring NaN (missing). inf is kept so a blown-up forecast is not hidden."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values) & (values > 0)]
    return float(np.exp(np.log(values).mean())) if values.size else np.nan

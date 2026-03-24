"""Diagnostics and performance metrics for FK forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from fractional_estimation import FKFitResult, fit_fractional_model
from fractional_model import FKParams, fractional_capacitance

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(diff**2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(np.abs(diff)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(100.0 * np.mean(np.abs((actual - predicted) / actual)))


def empirical_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    mask = np.isfinite(actual_arr) & np.isfinite(lower_arr) & np.isfinite(upper_arr)
    if not mask.any():
        return float("nan")
    inside = (actual_arr[mask] >= lower_arr[mask]) & (actual_arr[mask] <= upper_arr[mask])
    return float(np.mean(inside))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    mask = np.isfinite(lower_arr) & np.isfinite(upper_arr)
    if not mask.any():
        return float("nan")
    return float(np.mean(upper_arr[mask] - lower_arr[mask]))


def weighted_interval_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    actual_arr = np.asarray(actual, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    mask = np.isfinite(actual_arr) & np.isfinite(lower_arr) & np.isfinite(upper_arr)
    if not mask.any():
        return float("nan")
    actual_arr = actual_arr[mask]
    lower_arr = lower_arr[mask]
    upper_arr = upper_arr[mask]
    width = upper_arr - lower_arr
    below = np.maximum(lower_arr - actual_arr, 0.0)
    above = np.maximum(actual_arr - upper_arr, 0.0)
    score = width + (2.0 / alpha) * (below + above)
    return float(np.mean(score))


def coverage_gap(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, target: float) -> float:
    return empirical_coverage(actual, lower, upper) - float(target)


def information_criteria(
    residuals: np.ndarray,
    sigma: float,
    n_params: int,
) -> dict[str, float]:
    m = residuals.size
    ll = -0.5 * m * np.log(2 * np.pi * sigma**2) - 0.5 / sigma**2 * np.dot(residuals, residuals)
    aic = 2 * n_params - 2 * ll
    bic = np.log(m) * n_params - 2 * ll
    return {"loglik": float(ll), "AIC": float(aic), "BIC": float(bic)}


def information_criteria_from_fit(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_params: int,
    *,
    log_scale: bool = True,
    sigma: float | None = None,
) -> dict[str, float]:
    """Compute log-likelihood, AIC, and BIC from fitted observations.

    Parameters
    ----------
    observed, predicted
        Observed and fitted values on the same support.
    n_params
        Effective number of fitted parameters.
    log_scale
        When ``True``, compute residuals on log scale to match the FK
        log-normal observation model used in the manuscript.
    sigma
        Optional externally supplied residual scale. When omitted, the scale is
        estimated from the residual sum of squares with ``m - n_params``
        degrees of freedom.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted must share the same shape.")
    if obs.ndim != 1:
        obs = obs.reshape(-1)
        pred = pred.reshape(-1)

    if log_scale:
        mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0.0) & (pred > 0.0)
        if not mask.any():
            raise ValueError("Need at least one positive finite observation for log-scale criteria.")
        residuals = np.log(obs[mask]) - np.log(np.clip(pred[mask], 1e-15, np.inf))
    else:
        mask = np.isfinite(obs) & np.isfinite(pred)
        if not mask.any():
            raise ValueError("Need at least one finite observation for information criteria.")
        residuals = obs[mask] - pred[mask]

    if sigma is None:
        dof = max(1, residuals.size - int(n_params))
        sigma = float(np.sqrt(np.dot(residuals, residuals) / dof))
    info = information_criteria(residuals, float(sigma), n_params)
    info["sigma"] = float(sigma)
    return info


def waic(log_lik_samples: np.ndarray) -> float:
    """Compute WAIC from log-likelihood samples (shape: n_draws x n_obs)."""
    log_lik_samples = np.asarray(log_lik_samples)
    lppd = np.sum(np.log(np.mean(np.exp(log_lik_samples), axis=0)))
    p_waic = np.sum(np.var(log_lik_samples, axis=0))
    return float(-2.0 * (lppd - p_waic))


@dataclass
class ResidualDiagnostics:
    shapiro_p: float | None
    runs_p: float | None


def residual_diagnostics(residuals: np.ndarray) -> ResidualDiagnostics:
    if stats is None:  # pragma: no cover
        return ResidualDiagnostics(shapiro_p=None, runs_p=None)
    shapiro = stats.shapiro(residuals) if residuals.size >= 3 else (None, None)
    signs = np.sign(residuals - np.mean(residuals))
    non_zero = signs[signs != 0]
    if non_zero.size < 2:
        runs_p = None
    else:
        runs = 1 + np.sum(non_zero[1:] != non_zero[:-1])
        n_pos = np.sum(non_zero > 0)
        n_neg = np.sum(non_zero < 0)
        if n_pos == 0 or n_neg == 0:
            runs_p = None
        else:
            total = n_pos + n_neg
            mean_r = 1 + (2 * n_pos * n_neg) / total
            var_r = (
                2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)
                / (total**2 * (total - 1))
            )
            if var_r <= 0:
                runs_p = None
            else:
                z = (runs - mean_r) / math.sqrt(var_r)
                runs_p = float(math.erfc(abs(z) / math.sqrt(2)))
    return ResidualDiagnostics(
        shapiro_p=shapiro[1] if shapiro else None,
        runs_p=runs_p,
    )


@dataclass
class PrequentialResult:
    times: np.ndarray
    errors: np.ndarray


def prequential_forecast(
    times: Iterable[float],
    values: Iterable[float],
    *,
    min_window: int = 8,
    forecast_horizon: int = 1,
) -> PrequentialResult:
    """Forward-chaining forecast errors using FK fits."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    n = t.size
    errors = []
    times_out = []
    for end in range(min_window, n - forecast_horizon):
        try:
            fit_subset = fit_fractional_model(t[:end], y[:end])
            params = fit_subset.params
            future_time = t[end + forecast_horizon - 1]
            pred = fractional_capacitance(future_time, params)
        except Exception:
            continue
        errors.append(y[end + forecast_horizon - 1] - pred)
        times_out.append(future_time)
    if not errors:
        return PrequentialResult(times=np.array([]), errors=np.array([]))
    return PrequentialResult(times=np.array(times_out), errors=np.array(errors))


__all__ = [
    "rmse",
    "mae",
    "mape",
    "empirical_coverage",
    "mean_interval_width",
    "weighted_interval_score",
    "coverage_gap",
    "information_criteria",
    "information_criteria_from_fit",
    "waic",
    "ResidualDiagnostics",
    "residual_diagnostics",
    "PrequentialResult",
    "prequential_forecast",
]

"""Deterministic prediction utilities for the fractional-kinetics model."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from fractional_model import FKParams, fractional_capacitance, time_to_threshold

ArrayLike = Iterable[float] | np.ndarray


def predict_capacitance(
    t: ArrayLike,
    params: FKParams,
) -> np.ndarray:
    """Return plug-in forecasts on a time grid."""
    return fractional_capacitance(t, params)


def forecast_summary(
    t: ArrayLike,
    params: FKParams,
) -> dict[str, np.ndarray]:
    """Return dictionary with time grid and mean prediction."""
    t_arr = np.asarray(t, dtype=float)
    return {
        "time": t_arr,
        "prediction": predict_capacitance(t_arr, params),
    }


def failure_times(
    params: FKParams,
    thresholds: Sequence[float],
) -> dict[float, float]:
    """Compute threshold-crossing times for given ratios `q`."""
    results: dict[float, float] = {}
    for q in thresholds:
        results[q] = time_to_threshold(params, q)
    return results


def bootstrap_bias_correction(
    residuals: ArrayLike,
    *,
    n_bootstrap: int = 512,
    random_state: Optional[int] = None,
) -> dict[str, float]:
    """Estimate multiplicative bias factors from log-residual bootstrap samples.

    Residuals are assumed to be log-scale errors (log y - log mu). The returned
    bias factor can be applied as ``mean_curve / bias_factor`` to debias the
    plug-in forecast in small-sample regimes. The function also reports a
    percentile confidence interval for diagnostic use.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    res = np.asarray(residuals, dtype=float).ravel()
    res = res[np.isfinite(res)]
    if res.size == 0:
        raise ValueError("No finite residuals supplied for bootstrap bias correction.")

    rng = np.random.default_rng(random_state)
    draws = rng.choice(res, size=(n_bootstrap, res.size), replace=True)
    # Convert log-residuals to multiplicative factors and average per bootstrap.
    factors = np.exp(draws)
    mean_factors = np.mean(factors, axis=1)
    bias_factor = float(np.mean(mean_factors))
    ci_low, ci_high = np.quantile(mean_factors, [0.025, 0.975])
    return {
        "bias_factor": bias_factor,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


__all__ = ["predict_capacitance", "forecast_summary", "failure_times", "bootstrap_bias_correction"]

"""Time-respecting split conformal calibration for fractional-kinetics forecasts."""

from __future__ import annotations

import numpy as np


def _finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    m = sorted_values.size
    quantile_level = min(1.0, max(0.0, (1.0 - alpha) * (1.0 + 1.0 / (m + 1.0))))
    rank = int(np.ceil(quantile_level * m)) - 1
    rank = min(max(rank, 0), m - 1)
    return float(sorted_values[rank])


def conformal_intervals(
    calibration_obs: np.ndarray,
    calibration_samples: np.ndarray,
    test_samples: np.ndarray,
    *,
    alpha: float = 0.1,
) -> dict[str, np.ndarray]:
    """Calibrate predictive quantiles using a held-out sequential calibration block."""
    if calibration_samples.ndim != 2:
        raise ValueError("calibration_samples must be 2-D (n_samples, n_calibration).")
    if test_samples.ndim != 2:
        raise ValueError("test_samples must be 2-D (n_samples, n_test).")
    if calibration_samples.shape[0] != test_samples.shape[0]:
        raise ValueError("Calibration and test sample counts must match.")
    if calibration_samples.shape[1] != calibration_obs.size:
        raise ValueError("Mismatch between calibration observations and samples.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")

    lower_level = alpha / 2.0
    upper_level = 1.0 - lower_level

    base_lower_cal = np.quantile(calibration_samples, lower_level, axis=0)
    base_upper_cal = np.quantile(calibration_samples, upper_level, axis=0)
    scores = np.maximum.reduce(
        [
            base_lower_cal - calibration_obs,
            calibration_obs - base_upper_cal,
            np.zeros_like(calibration_obs, dtype=float),
        ]
    )
    q_hat = _finite_sample_quantile(scores, alpha)

    base_lower_test = np.quantile(test_samples, lower_level, axis=0)
    base_upper_test = np.quantile(test_samples, upper_level, axis=0)
    lower = base_lower_test - q_hat
    upper = base_upper_test + q_hat
    centers_test = np.median(test_samples, axis=0)
    scales_test = 0.5 * (base_upper_test - base_lower_test)

    return {
        "center": centers_test,
        "scale": scales_test,
        "base_lower": base_lower_test,
        "base_upper": base_upper_test,
        "lower": lower,
        "upper": upper,
        "q_hat": q_hat,
        "alpha": alpha,
    }


__all__ = ["conformal_intervals"]

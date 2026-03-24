"""Surrogate degradation models for the classical PICP pipeline.

This module provides light-weight reimplementations of the historical models
referenced by :mod:`core`.  They are intentionally simple but numerically
robust so that the modern fractional toolkit can coexist with the legacy
algorithms described in the documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from fractional_estimation import fit_fractional_model
from fractional_model import FKParams, fractional_capacitance


def _stable_logistic(x: float) -> float:
    if x >= 0:
        z = np.exp(-min(x, 700.0))
        return float(1.0 / (1.0 + z))
    z = np.exp(max(x, -700.0))
    return float(z / (1.0 + z))


@dataclass
class FitResult:
    params: Dict[str, float]
    fitted: np.ndarray
    residuals: np.ndarray
    aicc: float
    converged: bool
    diagnostics: Dict[str, float]
    success: bool = True


def _diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    denom = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2 = float(1 - np.sum(resid**2) / denom)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _ensure_1d(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError("time and observation arrays must share the same length")
    if t.size < 4:
        raise ValueError("at least four observations are required")
    order = np.argsort(t)
    return t[order], y[order]


class FKModel:
    """Wrapper around the modern fractional model for backwards compatibility."""

    def __init__(self) -> None:
        self._params: Optional[FKParams] = None

    def fit(self, t: np.ndarray, y: np.ndarray) -> FitResult:
        t_arr, y_arr = _ensure_1d(t, y)
        fit = fit_fractional_model(t_arr, y_arr)
        self._params = fit.params
        fitted = fractional_capacitance(t_arr, fit.params)
        diagnostics = _diagnostics(y_arr, fitted)
        return FitResult(
            params={
                "C0": float(fit.params.C0),
                "k": float(fit.params.k),
                "alpha": float(fit.params.alpha),
                "f_inf": float(fit.params.f_inf),
            },
            fitted=fitted,
            residuals=y_arr - fitted,
            aicc=np.nan,
            converged=bool(fit.success),
            diagnostics=diagnostics,
            success=bool(fit.success),
        )

    def predict(self, t: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("FKModel must be fitted before calling predict().")
        return fractional_capacitance(t, self._params)


class KWWModel:
    """Single stretched exponential (Kohlrausch) baseline."""

    def __init__(self) -> None:
        self._theta: Optional[np.ndarray] = None  # [c_inf, log_tau, logit_beta, c0]

    @staticmethod
    def _model(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        c_inf, log_tau, logit_beta, c0 = theta
        tau = np.exp(log_tau)
        beta = _stable_logistic(float(logit_beta))
        t_safe = np.maximum(t, 0.0)
        return c_inf + (c0 - c_inf) * np.exp(-np.power(t_safe / max(tau, 1e-6), beta))

    def fit(self, t: np.ndarray, y: np.ndarray) -> FitResult:
        t_arr, y_arr = _ensure_1d(t, y)
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scipy is required to fit the KWW model.") from exc

        c0_guess = float(y_arr[0])
        c_inf_guess = float(np.percentile(y_arr, 10))
        tau_guess = float(max(np.median(np.diff(t_arr)), 1.0))
        beta_guess = 0.7
        theta0 = np.array(
            [c_inf_guess, np.log(tau_guess), np.log(beta_guess / (1 - beta_guess)), c0_guess],
            dtype=float,
        )

        def residuals(theta: np.ndarray) -> np.ndarray:
            pred = self._model(t_arr, theta)
            return pred - y_arr

        result = least_squares(residuals, theta0, method="trf")
        self._theta = result.x.copy()
        fitted = self._model(t_arr, self._theta)
        diagnostics = _diagnostics(y_arr, fitted)
        params = {
            "C_inf": float(self._theta[0]),
            "tau": float(np.exp(self._theta[1])),
            "beta": _stable_logistic(float(self._theta[2])),
            "C0": float(self._theta[3]),
        }
        return FitResult(
            params=params,
            fitted=fitted,
            residuals=y_arr - fitted,
            aicc=np.nan,
            converged=bool(result.success),
            diagnostics=diagnostics,
            success=bool(result.success),
        )

    def predict(self, t: np.ndarray) -> np.ndarray:
        if self._theta is None:
            raise RuntimeError("KWWModel must be fitted before calling predict().")
        t_arr = np.asarray(t, dtype=float)
        return self._model(t_arr, self._theta)


class DonorProny2Model:
    """Two-term Prony series used for donor comparisons."""

    def __init__(self) -> None:
        self._theta: Optional[np.ndarray] = None  # [a, b1, b2, log_t1, log_r]

    @staticmethod
    def _model(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        a, b1, b2, log_t1, log_r = theta
        t1 = np.exp(np.clip(log_t1, -20, 20))
        r = 1.0 + np.exp(np.clip(log_r, -20, 20))
        t_safe = np.maximum(t, 0.0)
        denom1 = max(t1, 1e-6)
        denom2 = max(t1 * r, 1e-6)
        return a + b1 * np.exp(-t_safe / denom1) + b2 * np.exp(-t_safe / denom2)

    def fit(self, t: np.ndarray, y: np.ndarray) -> FitResult:
        t_arr, y_arr = _ensure_1d(t, y)
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scipy is required to fit the donor Prony-2 model.") from exc

        a_guess = float(y_arr[-1])
        amplitude = float(max(y_arr[0] - a_guess, 1e-3))
        b1_guess = amplitude * 0.6
        b2_guess = amplitude * 0.4
        t1_guess = float(max(np.median(np.diff(t_arr)), 1e-2))
        r_guess = 1.5
        theta0 = np.array([a_guess, b1_guess, b2_guess, np.log(t1_guess), np.log(r_guess - 1)], dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            pred = self._model(t_arr, theta)
            return pred - y_arr

        result = least_squares(residuals, theta0, method="trf")
        self._theta = result.x.copy()
        fitted = self._model(t_arr, self._theta)
        diagnostics = _diagnostics(y_arr, fitted)
        params = {
            "a": float(self._theta[0]),
            "b1": float(self._theta[1]),
            "b2": float(self._theta[2]),
            "t1": float(np.exp(self._theta[3])),
            "r": float(1.0 + np.exp(self._theta[4])),
        }
        return FitResult(
            params=params,
            fitted=fitted,
            residuals=y_arr - fitted,
            aicc=np.nan,
            converged=bool(result.success),
            diagnostics=diagnostics,
            success=bool(result.success),
        )

    def predict(self, t: np.ndarray) -> np.ndarray:
        if self._theta is None:
            raise RuntimeError("DonorProny2Model must be fitted before calling predict().")
        t_arr = np.asarray(t, dtype=float)
        return self._model(t_arr, self._theta)


__all__ = ["FitResult", "FKModel", "KWWModel", "DonorProny2Model"]

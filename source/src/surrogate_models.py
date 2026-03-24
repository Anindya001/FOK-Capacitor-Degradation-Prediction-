"""Surrogate degradation models for comparison with FK."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from fractional_diagnostics import information_criteria_from_fit

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for surrogate fitting. Install scipy>=1.7."
    ) from exc


ArrayLike = Iterable[float] | np.ndarray


def _split_series(times: np.ndarray, values: np.ndarray, train_ratio: float) -> dict[str, np.ndarray]:
    n_total = times.size
    train_idx = max(int(train_ratio * n_total), 6)
    train_idx = min(train_idx, n_total - 2)
    fit_t = times[:train_idx]
    fit_y = values[:train_idx]
    forecast_t = times[train_idx:]
    forecast_y = values[train_idx:]
    return {
        "fit_t": fit_t,
        "fit_y": fit_y,
        "forecast_t": forecast_t,
        "forecast_y": forecast_y,
        "train_idx": train_idx,
    }


def _prepare_series(times: ArrayLike, values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("times and values must share the same length.")
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if np.any(y <= 0.0):
        raise ValueError("values must be strictly positive for surrogate fitting.")
    return t, y


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff**2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.abs(diff)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def predict_classical(times: ArrayLike, params: dict[str, float]) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    c_inf = float(params["C_inf"])
    delta = float(params["Delta"])
    lam = max(float(params["lambda"]), 1e-6)
    return c_inf + delta * np.exp(-lam * t)


def predict_kww(times: ArrayLike, params: dict[str, float]) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    c_inf = float(params["C_inf"])
    tau = max(float(params["tau"]), 1e-6)
    beta = min(max(float(params["beta"]), 1e-6), 1.0 - 1e-6)
    c0 = float(params["C0"])
    return c_inf + (c0 - c_inf) * np.exp(-np.power(np.maximum(t, 0.0) / tau, beta))


def fit_classical_series(times: ArrayLike, values: ArrayLike) -> dict[str, object]:
    """Fit the classical exponential surrogate on the supplied series."""
    times_arr, values_arr = _prepare_series(times, values)

    def residuals(theta: np.ndarray) -> np.ndarray:
        c_inf = theta[0]
        delta = theta[1]
        lam = max(theta[2], 1e-6)
        pred = c_inf + delta * np.exp(-lam * times_arr)
        return pred - values_arr

    c0 = float(values_arr[0])
    c_inf_guess = float(np.percentile(values_arr, 20))
    delta_guess = c0 - c_inf_guess
    lam_guess = 1.0 / max(times_arr[-1] - times_arr[0], 1.0)
    x0 = np.array([c_inf_guess, delta_guess, lam_guess], dtype=float)

    result = least_squares(residuals, x0, method="trf")
    c_inf, delta, lam = result.x
    lam = max(lam, 1e-6)

    params = {
        "C_inf": float(c_inf),
        "Delta": float(delta),
        "lambda": float(lam),
    }
    fitted = predict_classical(times_arr, params)
    info = information_criteria_from_fit(values_arr, fitted, n_params=3, log_scale=True)
    return {
        "params": params,
        "time": times_arr,
        "prediction": fitted,
        "fitted": fitted,
        "residuals": values_arr - fitted,
        "log_residuals": np.log(values_arr) - np.log(np.clip(fitted, 1e-15, np.inf)),
        "sigma": float(info["sigma"]),
        "loglik": float(info["loglik"]),
        "AIC": float(info["AIC"]),
        "BIC": float(info["BIC"]),
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
    }


def fit_kww_series(times: ArrayLike, values: ArrayLike) -> dict[str, object]:
    """Fit the stretched-exponential KWW surrogate on the supplied series."""
    times_arr, values_arr = _prepare_series(times, values)

    def residuals(theta: np.ndarray) -> np.ndarray:
        c_inf = theta[0]
        tau = np.exp(theta[1])
        beta = 1.0 / (1.0 + np.exp(-theta[2]))
        c0 = theta[3]
        pred = c_inf + (c0 - c_inf) * np.exp(-np.power(np.maximum(times_arr, 0.0) / tau, beta))
        return pred - values_arr

    c0_guess = float(values_arr[0])
    c_inf_guess = float(np.percentile(values_arr, 10))
    tau_guess = max(np.median(np.diff(times_arr)), 1.0)
    beta_guess = 0.7
    x0 = np.array([c_inf_guess, math.log(tau_guess), math.log(beta_guess / (1 - beta_guess)), c0_guess])

    result = least_squares(residuals, x0, method="trf")
    c_inf = result.x[0]
    tau = math.exp(result.x[1])
    beta = 1.0 / (1.0 + math.exp(-result.x[2]))
    c0 = result.x[3]

    params = {
        "C_inf": float(c_inf),
        "tau": float(tau),
        "beta": float(beta),
        "C0": float(c0),
    }
    fitted = predict_kww(times_arr, params)
    info = information_criteria_from_fit(values_arr, fitted, n_params=4, log_scale=True)
    return {
        "params": params,
        "time": times_arr,
        "prediction": fitted,
        "fitted": fitted,
        "residuals": values_arr - fitted,
        "log_residuals": np.log(values_arr) - np.log(np.clip(fitted, 1e-15, np.inf)),
        "sigma": float(info["sigma"]),
        "loglik": float(info["loglik"]),
        "AIC": float(info["AIC"]),
        "BIC": float(info["BIC"]),
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
    }


def fit_classical(times: ArrayLike, values: ArrayLike, train_ratio: float = 0.7) -> dict[str, object]:
    """Fit the classical surrogate and evaluate it on a train/forecast split."""
    times_arr, values_arr = _prepare_series(times, values)
    splits = _split_series(times_arr, values_arr, train_ratio)
    fit = fit_classical_series(splits["fit_t"], splits["fit_y"])
    pred_all = predict_classical(times_arr, fit["params"])
    train_pred = pred_all[: splits["train_idx"]]
    forecast_pred = pred_all[splits["train_idx"] :]

    metrics = {
        "rmse_train": _rmse(values_arr[: splits["train_idx"]], train_pred),
        "mae_train": _mae(values_arr[: splits["train_idx"]], train_pred),
        "mape_train": _mape(values_arr[: splits["train_idx"]], train_pred),
        "AIC": float(fit["AIC"]),
        "BIC": float(fit["BIC"]),
        "sigma": float(fit["sigma"]),
    }
    if splits["forecast_t"].size:
        metrics["rmse_forecast"] = _rmse(values_arr[splits["train_idx"] :], forecast_pred)
        metrics["mae_forecast"] = _mae(values_arr[splits["train_idx"] :], forecast_pred)
        metrics["mape_forecast"] = _mape(values_arr[splits["train_idx"] :], forecast_pred)

    return {
        "params": fit["params"],
        "metrics": metrics,
        "time": times_arr,
        "prediction": pred_all,
        "train_idx": splits["train_idx"],
        "fit_result": fit,
    }


def fit_kww(times: ArrayLike, values: ArrayLike, train_ratio: float = 0.7) -> dict[str, object]:
    """Fit the KWW surrogate and evaluate it on a train/forecast split."""
    times_arr, values_arr = _prepare_series(times, values)
    splits = _split_series(times_arr, values_arr, train_ratio)
    fit = fit_kww_series(splits["fit_t"], splits["fit_y"])
    pred_all = predict_kww(times_arr, fit["params"])
    train_pred = pred_all[: splits["train_idx"]]
    forecast_pred = pred_all[splits["train_idx"] :]

    metrics = {
        "rmse_train": _rmse(values_arr[: splits["train_idx"]], train_pred),
        "mae_train": _mae(values_arr[: splits["train_idx"]], train_pred),
        "mape_train": _mape(values_arr[: splits["train_idx"]], train_pred),
        "AIC": float(fit["AIC"]),
        "BIC": float(fit["BIC"]),
        "sigma": float(fit["sigma"]),
    }
    if splits["forecast_t"].size:
        metrics["rmse_forecast"] = _rmse(values_arr[splits["train_idx"] :], forecast_pred)
        metrics["mae_forecast"] = _mae(values_arr[splits["train_idx"] :], forecast_pred)
        metrics["mape_forecast"] = _mape(values_arr[splits["train_idx"] :], forecast_pred)

    return {
        "params": fit["params"],
        "metrics": metrics,
        "time": times_arr,
        "prediction": pred_all,
        "train_idx": splits["train_idx"],
        "fit_result": fit,
    }


__all__ = [
    "fit_classical",
    "fit_kww",
    "fit_classical_series",
    "fit_kww_series",
    "predict_classical",
    "predict_kww",
]

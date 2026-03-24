"""Parameter estimation for the fractional-kinetics degradation model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from fractional_model import FKParams, ensure_monotonic, fractional_capacitance

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for fractional parameter estimation. "
        "Please install scipy>=1.7 to continue."
    ) from exc


ArrayLike = Iterable[float] | np.ndarray


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -700.0))
    return z / (1.0 + z)


def _logit(p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1.0 - p))


@dataclass
class FKFitResult:
    params: FKParams
    sigma: float
    covariance: np.ndarray
    transformed: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    cost: float
    grad_norm: float
    nfev: int
    njev: int
    sigma_log: float
    hessian_cond: Optional[float]
    monotonic: Optional[bool]


def _prepare_inputs(times: ArrayLike, values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape:
        raise ValueError("times and values must share the same shape.")
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    if np.any(y_sorted <= 0):
        raise ValueError("Capacitance values must be strictly positive.")
    return t_sorted, y_sorted


def _initial_guess(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    c0_guess = float(np.max(y))
    tail_count = max(3, min(10, y.size // 4))
    tail_mean = float(np.mean(y[-tail_count:]))
    f_inf_guess = min(max(tail_mean / c0_guess, 1e-3), 0.95)
    alpha_guess = 0.7
    mid_value = f_inf_guess + (1.0 - f_inf_guess) / math.e
    mid_cap = c0_guess * mid_value
    idx = np.searchsorted(y[::-1], mid_cap, side="left")
    if idx == 0:
        k_guess = 1e-3
    else:
        t_mid = float(t[-idx - 1])
        k_guess = max(1e-6, ((1.0 - mid_value) * math.gamma(1.0 + alpha_guess) / t_mid ** alpha_guess))
    return np.array(
        [
            math.log(k_guess),
            _logit(alpha_guess),
            _logit(f_inf_guess),
            math.log(c0_guess),
        ],
        dtype=float,
    )


def _unpack_params(xi: Sequence[float]) -> FKParams:
    kappa, a, b, c0 = xi
    k = math.exp(kappa)
    alpha = min(max(_logistic(a), 1e-6), 1.0 - 1e-6)
    f_inf = min(max(_logistic(b), 1e-8), 1.0 - 1e-6)
    C0 = math.exp(c0)
    return FKParams(C0=C0, k=k, alpha=alpha, f_inf=f_inf)


def fit_fractional_model(
    times: ArrayLike,
    values: ArrayLike,
    *,
    loss: str = "linear",
    huber_c: float = 1.345,
) -> FKFitResult:
    """Fit FK parameters to log-capacitance data via least squares."""
    t, y = _prepare_inputs(times, values)
    log_y = np.log(y)
    c0_lower = math.log(max(float(np.min(y)) * 0.5, 1e-6))
    c0_upper = math.log(max(float(np.max(y)) * 5.0, math.exp(c0_lower) * 1.1))
    lower_bounds = np.array(
        [
            math.log(1e-10),
            _logit(0.01),
            _logit(1e-6),
            c0_lower,
        ],
        dtype=float,
    )
    upper_bounds = np.array(
        [
            math.log(10.0),
            _logit(0.995),
            _logit(0.995),
            c0_upper,
        ],
        dtype=float,
    )

    def residuals(xi: np.ndarray) -> np.ndarray:
        try:
            params = _unpack_params(xi)
            mu = fractional_capacitance(t, params)
        except Exception:
            return np.full_like(log_y, 1e6, dtype=float)
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            return np.full_like(log_y, 1e6, dtype=float)
        log_mu = np.log(np.clip(mu, 1e-15, np.inf))
        return log_y - log_mu

    x0 = np.clip(_initial_guess(t, y), lower_bounds + 1e-6, upper_bounds - 1e-6)
    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss=loss,
        f_scale=huber_c,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=4000,
    )

    theta_hat = _unpack_params(result.x)
    res = result.fun
    m = res.size
    p = result.x.size
    dof = max(1, m - p)
    sigma_hat = math.sqrt(float(np.dot(res, res)) / dof)

    hessian_cond: Optional[float] = None
    if result.jac is not None and result.jac.size:
        jtj = result.jac.T @ result.jac
        try:
            cov = sigma_hat**2 * np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            cov = sigma_hat**2 * np.linalg.pinv(jtj)
        try:
            hessian_cond = float(np.linalg.cond(jtj))
        except np.linalg.LinAlgError:
            hessian_cond = None
    else:
        cov = np.full((p, p), np.nan, dtype=float)

    grad_norm = float(np.linalg.norm(result.grad)) if result.grad is not None else float("nan")
    try:
        monotonic_flag = bool(ensure_monotonic(t, theta_hat))
    except Exception:
        monotonic_flag = None

    return FKFitResult(
        params=theta_hat,
        sigma=sigma_hat,
        covariance=cov,
        transformed=result.x.copy(),
        residuals=res,
        success=bool(result.success),
        message=result.message,
        cost=float(result.cost),
        grad_norm=grad_norm,
        nfev=int(result.nfev),
        njev=int(getattr(result, "njev", 0)),
        sigma_log=float(math.log(max(sigma_hat, 1e-12))),
        hessian_cond=hessian_cond,
        monotonic=monotonic_flag,
    )


__all__ = ["FKFitResult", "fit_fractional_model"]

"""Fractional-kinetics model utilities for capacitor degradation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from math_utils import mittag_leffler, mittag_leffler_two_param

ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class FKParams:
    """Immutable container for fractional-kinetics parameters."""

    C0: float
    k: float
    alpha: float
    f_inf: float

    def validate(self) -> None:
        if not np.isfinite(self.C0) or self.C0 <= 0:
            raise ValueError("C0 must be positive and finite.")
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be positive and finite.")
        if not np.isfinite(self.alpha) or not (0 < self.alpha < 1):
            raise ValueError("alpha must lie in (0, 1).")
        if not np.isfinite(self.f_inf) or not (0 <= self.f_inf < 1):
            raise ValueError("f_inf must lie in [0, 1).")


def _as_array(t: ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(t, dtype=float)
    original_shape = arr.shape
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr, original_shape


def fractional_capacitance(t: ArrayLike, params: FKParams) -> np.ndarray:
    """Return capacitance values for times *t* under FK parameters."""
    params.validate()
    t_arr, original_shape = _as_array(t)
    ml = mittag_leffler(params.alpha, -params.k * np.power(t_arr, params.alpha))
    cap = params.C0 * (params.f_inf + (1.0 - params.f_inf) * ml)
    return cap.reshape(original_shape)


def normalized_deficit(t: ArrayLike, params: FKParams) -> np.ndarray:
    """Return the normalised deficit Δ_h = E_α(-k t^α)."""
    params.validate()
    t_arr, original_shape = _as_array(t)
    deficit = mittag_leffler(params.alpha, -params.k * np.power(t_arr, params.alpha))
    return deficit.reshape(original_shape)


def fractional_derivative(t: ArrayLike, params: FKParams) -> np.ndarray:
    """Compute dC/dt for the FK model."""
    params.validate()
    t_arr, original_shape = _as_array(t)
    safe_t = np.maximum(t_arr, 1e-12)
    ml_2p = mittag_leffler_two_param(
        params.alpha, params.alpha, -params.k * np.power(safe_t, params.alpha)
    )
    derivative = -params.C0 * (1.0 - params.f_inf) * params.k * np.power(
        safe_t, params.alpha - 1.0
    ) * ml_2p
    return derivative.reshape(original_shape)


def initial_time_guess(params: FKParams, ratio: float) -> float:
    """Heuristic initial guess for time given ratio r = E_alpha(-k t^alpha)."""
    gamma_term = math.gamma(1.0 + params.alpha)
    rough = ((1.0 - ratio) * gamma_term / params.k) ** (1.0 / params.alpha)
    return max(rough, 1e-12)


def time_to_threshold(
    params: FKParams,
    q: float,
    *,
    initial_guess: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> float:
    """Solve for T_q such that C(T_q) = q * C0."""
    params.validate()
    if not (params.f_inf < q < 1.0):
        raise ValueError("q must lie between f_inf and 1.")

    ratio = (q - params.f_inf) / (1.0 - params.f_inf)
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("Threshold ratio out of admissible range.")

    if initial_guess is None:
        initial_guess = initial_time_guess(params, ratio)

    def func(t: float) -> float:
        return float(mittag_leffler(params.alpha, -params.k * (t ** params.alpha)) - ratio)

    def dfunc(t: float) -> float:
        t_safe = max(t, 1e-12)
        ml_2p = float(
            mittag_leffler_two_param(
                params.alpha, params.alpha, -params.k * (t_safe ** params.alpha)
            )
        )
        return -params.k * (t_safe ** (params.alpha - 1.0)) * ml_2p

    t = max(initial_guess, 1e-12)
    for _ in range(max_iter):
        f_val = func(t)
        if abs(f_val) < tol:
            return t
        df_val = dfunc(t)
        if not np.isfinite(df_val) or df_val == 0.0:
            break
        step = f_val / df_val
        t_new = t - step
        if t_new <= 0 or not np.isfinite(t_new):
            break
        if abs(t_new - t) < tol * max(1.0, t):
            return t_new
        t = t_new

    # Fallback to bisection
    f_low = func(0.0)
    if f_low < 0:
        return 0.0
    t_low = 0.0
    t_high = max(t, 1.0)
    f_high = func(t_high)
    expand_count = 0
    while f_high > 0 and expand_count < 60:
        t_high *= 2.0
        f_high = func(t_high)
        expand_count += 1
    if f_high > 0:
        raise RuntimeError("Failed to bracket the threshold crossing.")

    for _ in range(max_iter * 2):
        mid = 0.5 * (t_low + t_high)
        f_mid = func(mid)
        if abs(f_mid) < tol or (t_high - t_low) < tol * max(1.0, mid):
            return mid
        if f_mid > 0:
            t_low = mid
        else:
            t_high = mid

    return 0.5 * (t_low + t_high)


def ensure_monotonic(
    t: Sequence[float],
    params: FKParams,
    atol: float = 1e-8,
) -> bool:
    """Check that C(t) is non-increasing over the provided times."""
    values = fractional_capacitance(t, params)
    diffs = np.diff(values)
    return bool(np.all(diffs <= atol))

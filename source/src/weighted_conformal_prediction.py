"""Weighted conformal prediction utilities used by :mod:`core`.

The original project shipped a fairly involved implementation.  This module
captures the essential behaviour required by the modernised code-base: it
supports phase-aware conformal intervals with a simple MAD-based scaling rule
and exposes diagnostics consumed by the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class ConformalResult:
    lower: np.ndarray
    upper: np.ndarray
    q_hats: Dict[int, float]
    phase_assignment: np.ndarray
    phase_counts: Dict[int, int]
    phase_fallback_used: bool
    scale_mode_used: str


@dataclass
class _Cache:
    t_cal: np.ndarray
    residuals: np.ndarray
    scale_values: np.ndarray
    phases_cal: np.ndarray
    q_hats: Dict[int, float]
    global_q: float
    phase_edges: np.ndarray
    scale_mode: str


def _mad(values: np.ndarray) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(mad / 0.6744897501960817 if mad > 0 else 0.0)


def _phase_edges(times: np.ndarray, spec: str) -> np.ndarray:
    if spec is None:
        return np.array([], dtype=float)
    spec_lower = spec.lower()
    if spec_lower in {"none", "global"}:
        return np.array([], dtype=float)
    if spec_lower in {"tertiles", "tertile"}:
        quantiles = [1 / 3, 2 / 3]
    elif spec_lower in {"quartiles", "quartile"}:
        quantiles = [0.25, 0.5, 0.75]
    else:
        # default to tertiles for unrecognised options
        quantiles = [1 / 3, 2 / 3]
    edges = np.quantile(times, quantiles)
    return np.unique(edges)


def _assign_phases(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(times.size, dtype=int)
    return np.digitize(times, edges, right=False)


def _quantile(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        return float("nan")
    sorted_scores = np.sort(scores)
    level = (1 - alpha) * (scores.size + 1)
    rank = max(0, min(int(np.ceil(level) - 1), scores.size - 1))
    return float(sorted_scores[rank])


class WeightedConformalPredictor:
    """Phase-aware conformal predictor with MAD scaling."""

    def __init__(self, *, alpha: float, phases: str = "tertiles", scale_mode: str = "auto") -> None:
        self.alpha = float(alpha)
        self.phases = phases
        self.scale_mode = scale_mode
        self._cache: Optional[_Cache] = None

    def fit(
        self,
        y_train: Iterable[float],
        t_train: Iterable[float],
        *,
        yhat_train: Optional[Iterable[float]] = None,
        cal_fraction: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        y_arr = np.asarray(y_train, dtype=float)
        t_arr = np.asarray(t_train, dtype=float)
        if yhat_train is None:
            raise ValueError("yhat_train must be supplied for weighted conformal fitting.")
        mu_arr = np.asarray(yhat_train, dtype=float)
        if y_arr.shape != mu_arr.shape:
            raise ValueError("y_train and yhat_train must share the same shape.")
        if y_arr.size != t_arr.size:
            raise ValueError("y_train and t_train must share the same shape.")
        if y_arr.size < 4:
            raise ValueError("Need at least four observations for conformal calibration.")

        cal_size = max(1, int(cal_fraction * y_arr.size))
        cal_size = min(cal_size, y_arr.size - 1)
        fit_size = y_arr.size - cal_size

        cal_y = y_arr[fit_size:]
        cal_t = t_arr[fit_size:]
        cal_mu = mu_arr[fit_size:]

        residuals = cal_y - cal_mu
        if self.scale_mode == "std":
            scale = float(np.std(residuals, ddof=1) or 0.0)
        else:
            scale = _mad(residuals)
            if scale == 0.0 and self.scale_mode == "auto":
                scale = float(np.std(residuals, ddof=1) or 0.0)
        scale = max(scale, 1e-12)
        scale_values = np.full(cal_y.size, scale, dtype=float)

        edges = _phase_edges(cal_t, self.phases)
        phases = _assign_phases(cal_t, edges)
        scores = np.abs(residuals) / scale

        rng = np.random.default_rng(random_state)
        phase_q: Dict[int, float] = {}
        for phase in np.unique(phases):
            phase_scores = scores[phases == phase]
            if phase_scores.size == 0:
                phase_q[phase] = float("nan")
                continue
            phase_q[phase] = _quantile(phase_scores, self.alpha)
        global_q = _quantile(scores, self.alpha)

        self._cache = _Cache(
            t_cal=cal_t,
            residuals=residuals,
            scale_values=scale_values,
            phases_cal=phases,
            q_hats=phase_q,
            global_q=global_q,
            phase_edges=edges,
            scale_mode="std" if self.scale_mode == "std" else "mad",
        )

    def _intervals(
        self,
        t_new: np.ndarray,
        mu_new: np.ndarray,
        scale_new: np.ndarray,
        edges: np.ndarray,
        phase_q: Dict[int, float],
        global_q: float,
    ) -> ConformalResult:
        phases = _assign_phases(t_new, edges)
        lower = np.empty_like(mu_new, dtype=float)
        upper = np.empty_like(mu_new, dtype=float)
        phase_counts: Dict[int, int] = {}
        fallback_used = False
        for idx, phase in enumerate(phases):
            q_hat = phase_q.get(phase)
            if not np.isfinite(q_hat):
                q_hat = global_q
                fallback_used = True
            lower[idx] = mu_new[idx] - q_hat * scale_new[idx]
            upper[idx] = mu_new[idx] + q_hat * scale_new[idx]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        return ConformalResult(
            lower=lower,
            upper=upper,
            q_hats=phase_q,
            phase_assignment=phases,
            phase_counts=phase_counts,
            phase_fallback_used=fallback_used,
            scale_mode_used=self._cache.scale_mode if self._cache else "mad",
        )

    def predict(self, t_new: Iterable[float], model_predict) -> ConformalResult:
        if self._cache is None:
            raise RuntimeError("fit() must be called before predict().")
        t_arr = np.asarray(t_new, dtype=float)
        if t_arr.size == 0:
            return ConformalResult(
                lower=np.array([], dtype=float),
                upper=np.array([], dtype=float),
                q_hats=self._cache.q_hats,
                phase_assignment=np.array([], dtype=int),
                phase_counts={},
                phase_fallback_used=False,
                scale_mode_used=self._cache.scale_mode,
            )
        mu_new = np.asarray(model_predict(t_arr), dtype=float)
        scale_new = self._project_scale(t_arr, self._cache.scale_values, self._cache.t_cal)
        return self._intervals(t_arr, mu_new, scale_new, self._cache.phase_edges, self._cache.q_hats, self._cache.global_q)

    def _project_scale(self, t_new: np.ndarray, scale_cal: np.ndarray, t_cal: np.ndarray) -> np.ndarray:
        if t_new.size == 0 or scale_cal.size == 0 or t_cal.size == 0:
            return np.zeros_like(t_new, dtype=float)
        if np.allclose(scale_cal, scale_cal[0]):
            return np.full_like(t_new, scale_cal[0], dtype=float)
        return np.interp(t_new, t_cal, scale_cal, left=scale_cal[0], right=scale_cal[-1])

    def calibration_metrics(
        self,
        y_cal: Iterable[float],
        mu_cal: Iterable[float],
        conformal_calib: ConformalResult,
    ) -> Dict[str, object]:
        y_arr = np.asarray(y_cal, dtype=float)
        mu_arr = np.asarray(mu_cal, dtype=float)
        lower = conformal_calib.lower
        upper = conformal_calib.upper
        if y_arr.size == 0 or lower.size == 0:
            return {"coverage": float("nan"), "per_phase_coverage": {}}
        inside = (y_arr >= lower) & (y_arr <= upper)
        coverage = float(np.mean(inside))
        per_phase: Dict[str, float] = {}
        for phase, count in conformal_calib.phase_counts.items():
            mask = conformal_calib.phase_assignment == phase
            if np.any(mask):
                per_phase[str(phase)] = float(np.mean(inside[mask]))
        residuals = y_arr - mu_arr
        return {
            "coverage": coverage,
            "per_phase_coverage": per_phase,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals, ddof=1)),
        }


__all__ = ["ConformalResult", "WeightedConformalPredictor"]

"""
Physics-Informed Conformal Prediction (PICP) core engine.

Provides :class:`PICPCore` for data loading, physics-model fitting, and
phase-aware conformal uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from math_utils import mittag_leffler
from models import DonorProny2Model, FKModel, FitResult, KWWModel
from weighted_conformal_prediction import ConformalResult, WeightedConformalPredictor
from legacy_core import PICPCore as LegacyPipelineCore

LOGGER = logging.getLogger(__name__)

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover
    acorr_ljungbox = None  # type: ignore
    adfuller = None  # type: ignore
    kpss = None  # type: ignore


DataLike = pd.DataFrame | str | Path
ModelName = Literal["fractional", "classical", "kww", "donor"]

_TIME_ALIASES = ("time", "hour", "age", "t")
_TARGET_ALIASES = ("capacitance", "cap", "c", "measurement")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: str(col).strip() for col in df.columns})


def _detect_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower_map = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    for col in df.columns:
        lowered = str(col).lower()
        if any(alias in lowered for alias in aliases):
            return col
    return None


def load_dataset(
    data: DataLike,
    time_column: Optional[str] = None,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """Load a capacitor degradation dataset."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                from excel_reader import load_excel
            except ImportError:
                df = pd.read_excel(path)
            else:
                times, channels = load_excel(path)
                df = pd.DataFrame({'time': times})
                for name, values in channels.items():
                    df[name] = values
                if time_column is None:
                    time_column = 'time'
                if target_column is None and len(df.columns) > 1:
                    target_column = df.columns[1]
                df = df.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
                df = df.sort_values(time_column).reset_index(drop=True)
                return df, time_column or 'time', target_column or df.columns[1]
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    if df.empty:
        raise ValueError("Dataset is empty")

    df = _normalise_columns(df)
    df = df.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()

    time_column = time_column or _detect_column(df, _TIME_ALIASES)
    target_column = target_column or _detect_column(df, _TARGET_ALIASES)

    if time_column is None or target_column is None:
        raise ValueError("Unable to infer time/target columns from dataset")
    if time_column not in df.columns or target_column not in df.columns:
        raise KeyError("Specified columns not present in dataset")

    df = df.sort_values(time_column).reset_index(drop=True)
    return df, time_column, target_column


# ---------------------------------------------------------------------------
# Classical baseline model
# ---------------------------------------------------------------------------


@dataclass
class ClassicalMassBalanceParams:
    c0: float
    lambda_: float
    kappa: float


class ClassicalMassBalanceModel:
    """Exponential mass balance baseline."""

    def __init__(self) -> None:
        self.params: Optional[ClassicalMassBalanceParams] = None

    def fit(self, t: np.ndarray, y: np.ndarray) -> FitResult:
        from scipy.optimize import least_squares  # lazy import

        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
            raise ValueError("t and y must be 1-D arrays of equal length")
        if t.size < 4:
            raise ValueError("Need at least four observations")

        c0_guess = float(y[0])
        lambda_guess = 1.0 / max(np.median(np.diff(t)), 1.0)
        kappa_guess = float(max(y[0] - y[-1], 1e-3))

        def residuals(x: np.ndarray) -> np.ndarray:
            c0, lambda_, kappa = x
            lambda_ = max(lambda_, 1e-6)
            pred = c0 - kappa * (1 - np.exp(-lambda_ * np.clip(t, a_min=0.0, a_max=None)))
            return pred - y

        bounds = (
            np.array([c0_guess - 5 * abs(kappa_guess), 1e-6, 0.0]),
            np.array([c0_guess + 5 * abs(kappa_guess), 10.0, 5 * abs(kappa_guess)]),
        )
        result = least_squares(residuals, x0=np.array([c0_guess, lambda_guess, kappa_guess]), bounds=bounds)
        params = ClassicalMassBalanceParams(
            c0=float(result.x[0]),
            lambda_=float(max(result.x[1], 1e-6)),
            kappa=float(max(result.x[2], 0.0)),
        )
        fitted = self.predict(t, params)
        resid = y - fitted
        diagnostics = {
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "mae": float(np.mean(np.abs(resid))),
            "r2": float(1 - np.sum(resid**2) / (np.sum((y - y.mean()) ** 2) + 1e-12)),
        }
        fit_result = FitResult(
            params={"C0": params.c0, "lambda": params.lambda_, "kappa": params.kappa},
            fitted=fitted,
            residuals=resid,
            aicc=np.nan,
            converged=bool(result.success),
            diagnostics=diagnostics,
        )
        self.params = params
        return fit_result

    def predict(self, t: np.ndarray, params: Optional[ClassicalMassBalanceParams] = None) -> np.ndarray:
        if params is None:
            if self.params is None:
                raise RuntimeError("Model not fitted")
            params = self.params
        t = np.asarray(t, dtype=float)
        return params.c0 - params.kappa * (1 - np.exp(-params.lambda_ * np.clip(t, a_min=0.0, a_max=None)))


# ---------------------------------------------------------------------------
# Parameter uncertainty
# ---------------------------------------------------------------------------


def compute_parameter_cis(
    fit_result: FitResult,
    model_type: str,
    *,
    method: Literal["bootstrap", "profile"] = "bootstrap",
    alpha: float = 0.05,
    fit_fn: Optional[Callable[[np.ndarray], FitResult]] = None,
    y_true: Optional[np.ndarray] = None,
    block_size: int = 8,
    n_resamples: int = 500,
) -> Dict[str, Tuple[float, float]]:
    params = fit_result.params
    if not params:
        return {}
    if method == "profile":
        raise NotImplementedError("Profile likelihood CIs are not implemented")
    if fit_fn is None or y_true is None:
        residual_std = float(np.std(fit_result.residuals, ddof=1) or 1.0)
        z = stats.norm.ppf(1 - alpha / 2) if stats else 1.96
        return {
            name: (float(value) - z * residual_std, float(value) + z * residual_std)
            for name, value in params.items()
        }
    return block_bootstrap_ci(
        residuals=fit_result.residuals,
        block_size=block_size,
        fit_fn=fit_fn,
        y_true=y_true,
        B=n_resamples,
        alpha=alpha,
    )


def block_bootstrap_ci(
    residuals: np.ndarray,
    block_size: int,
    fit_fn: Callable[[np.ndarray], FitResult],
    y_true: np.ndarray,
    *,
    B: int = 500,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    residuals = np.asarray(residuals, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    n = residuals.size
    if n == 0:
        return {}
    block_size = max(1, min(block_size, n))
    rng = np.random.default_rng()

    base_fit = fit_fn(y_true)
    fitted_base = base_fit.fitted
    param_samples: Dict[str, list[float]] = {name: [] for name in base_fit.params}

    def resample() -> np.ndarray:
        starts = rng.integers(0, n - block_size + 1, size=int(np.ceil(n / block_size)))
        pieces = [residuals[s : s + block_size] for s in starts]
        return np.concatenate(pieces)[:n]

    for _ in range(B):
        boot_resid = resample()
        boot_series = fitted_base + boot_resid
        result = fit_fn(boot_series)
        for name, value in result.params.items():
            param_samples[name].append(float(value))

    ci: Dict[str, Tuple[float, float]] = {}
    for name, samples in param_samples.items():
        if not samples:
            continue
        lower, upper = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci[name] = (float(lower), float(upper))
    return ci


def donor_bootstrap(
    target_series: np.ndarray,
    donor_population: Optional[pd.DataFrame],
    *,
    B: int = 200,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if donor_population is None or donor_population.empty:
        return {
            "prediction_bands": (np.array([]), np.array([])),
            "donor_frequency": {},
            "convergence_rate": 0.0,
        }
    rng = np.random.default_rng(seed)
    donors = donor_population.columns.tolist()
    counts = {donor: 0 for donor in donors}
    predictions = []
    for _ in range(B):
        sample = donor_population.sample(3, axis=1, replace=True, random_state=rng.integers(0, 2**32))
        for col in sample.columns:
            counts[col] += 1
        predictions.append(sample.mean(axis=1).to_numpy())
    preds = np.array(predictions)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return {
        "prediction_bands": (lower, upper),
        "donor_frequency": counts,
        "convergence_rate": float(sum(counts.values()) > 0) / B,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def exchangeability_report(residuals: np.ndarray) -> Dict[str, Any]:
    residuals = np.asarray(residuals, dtype=float)
    report: Dict[str, Any] = {}
    if residuals.size < 10:
        return {"note": "Insufficient points for diagnostics"}

    if adfuller is not None:
        try:
            stat, pvalue, *_ = adfuller(residuals)
            report["adf_stat"] = float(stat)
            report["adf_pvalue"] = float(pvalue)
        except Exception:  # pragma: no cover
            report["adf_pvalue"] = None
    else:
        report["adf_pvalue"] = None

    if kpss is not None:
        try:
            stat, pvalue, *_ = kpss(residuals, regression="c", nlags="auto")
            report["kpss_stat"] = float(stat)
            report["kpss_pvalue"] = float(pvalue)
        except Exception:  # pragma: no cover
            report["kpss_pvalue"] = None
    else:
        report["kpss_pvalue"] = None

    if acorr_ljungbox is not None:
        try:
            lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
            report["ljung_box_pvalue"] = float(lb["lb_pvalue"].iloc[0])
        except Exception:  # pragma: no cover
            report["ljung_box_pvalue"] = None
    else:
        report["ljung_box_pvalue"] = None

    runs_p = None
    if stats is not None:
        try:
            median = np.median(residuals)
            signs = residuals > median
            runs, expected, variance = _runs_statistics(signs)
            z = (runs - expected) / np.sqrt(variance)
            runs_p = 2 * (1 - stats.norm.cdf(abs(z)))
        except Exception:  # pragma: no cover
            runs_p = None
    report["runs_pvalue"] = None if runs_p is None else float(runs_p)

    report["acf_lag1"] = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if residuals.size > 1 else None
    return report


def _runs_statistics(signs: np.ndarray) -> Tuple[float, float, float]:
    signs = np.asarray(signs, dtype=bool)
    n1 = signs.sum()
    n2 = signs.size - n1
    runs = 1
    for i in range(1, signs.size):
        if signs[i] != signs[i - 1]:
            runs += 1
    expected = 1 + (2 * n1 * n2) / (n1 + n2)
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1) + 1e-12)
    return float(runs), float(expected), float(max(variance, 1e-12))


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


@dataclass
class _Split:
    fit_t: np.ndarray
    fit_y: np.ndarray
    calib_t: np.ndarray
    calib_y: np.ndarray
    forecast_t: np.ndarray
    forecast_y: np.ndarray


@dataclass
class DataPreparationResult:
    frame: pd.DataFrame
    time_column: str
    target_column: str
    monotonic: bool


class PICPCore:
    """High-level orchestrator for degradation forecasting."""

    def __init__(
        self,
        *,
        cal_fraction: float = 0.2,
        phases: str = "tertiles",
        scale_mode: str = "auto",
        random_state: Optional[int] = None,
        use_legacy: bool = True,
        fallback_to_modern: bool = True,
        legacy_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cal_fraction = cal_fraction
        self.phases = phases
        self.scale_mode = scale_mode
        self.random_state = random_state
        self.use_legacy = use_legacy
        self.fallback_to_modern = fallback_to_modern
        self.legacy_options = dict(legacy_options or {})
        self._legacy_core: Optional[LegacyPipelineCore] = None


    def run_forecast(
        self,
        data: DataLike,
        *,
        time_column: Optional[str] = None,
        target_column: Optional[str] = None,
        confidence: float = 0.90,
        train_ratio: float = 0.70,
        model_type: ModelName = "classical",
        bayesian_samples: int = 0,
    ) -> Dict[str, Any]:
        prep = self._prepare_data(data, time_column, target_column)

        if self.use_legacy:
            try:
                return self._run_legacy_forecast(
                    preparation=prep,
                    confidence=confidence,
                    train_ratio=train_ratio,
                    model_type=model_type,
                )
            except Exception as legacy_exc:  # pragma: no cover - legacy fallback
                LOGGER.warning("Legacy pipeline failed (%s); falling back to modern core.", legacy_exc)
                if not self.fallback_to_modern:
                    raise

        df = prep.frame
        time_column = prep.time_column
        target_column = prep.target_column

        times = df[time_column].to_numpy(dtype=float)
        values = df[target_column].to_numpy(dtype=float)
        split = self._split_series(times, values, train_ratio)

        fit_result, model_label, model_predict = self._fit_model(model_type, split.fit_t, split.fit_y)
        alpha = 1.0 - confidence

        mu_fit = model_predict(split.fit_t)
        mu_cal = model_predict(split.calib_t)
        mu_fore = model_predict(split.forecast_t) if split.forecast_t.size else np.array([], dtype=float)

        train_t = np.concatenate([split.fit_t, split.calib_t])
        train_y = np.concatenate([split.fit_y, split.calib_y])
        predictor = WeightedConformalPredictor(alpha=alpha, phases=self.phases, scale_mode=self.scale_mode)
        yhat_train = model_predict(train_t)
        predictor.fit(train_y, train_t, yhat_train=yhat_train, cal_fraction=self.cal_fraction, random_state=self.random_state)

        cache = predictor._cache
        if cache is None:
            raise RuntimeError("Conformal predictor cache is unavailable after fit().")

        conformal_calib = predictor.predict(split.calib_t, model_predict)

        if split.forecast_t.size:
            conformal_forecast = predictor.predict(split.forecast_t, model_predict)
            sigma_new = predictor._project_scale(split.forecast_t, cache.scale_values, cache.t_cal)
        else:
            conformal_forecast = ConformalResult(
                lower=np.array([], dtype=float),
                upper=np.array([], dtype=float),
                q_hats={},
                phase_assignment=np.array([], dtype=int),
                phase_fallback_used=False,
                phase_counts={},
                scale_mode_used=cache.scale_mode,
            )
            sigma_new = np.array([], dtype=float)

        residuals_cal = cache.residuals.astype(float)
        sigma_cal = cache.scale_values.astype(float)
        phases_cal = cache.phases_cal.astype(int)
        scores_cal = np.abs(residuals_cal / np.maximum(sigma_cal, 1e-12))

        param_cis = compute_parameter_cis(
            fit_result,
            model_type,
            alpha=alpha,
            fit_fn=lambda series: self._refit_model(model_type, split.fit_t, series),
            y_true=split.fit_y,
            block_size=max(4, int(np.sqrt(split.fit_y.size))),
            n_resamples=200,
        )

        coverage_diag = predictor.calibration_metrics(split.calib_y, mu_cal, conformal_calib)
        metrics = self._compute_metrics(split, model_predict, conformal_forecast, coverage_diag, confidence)
        exchangeability = exchangeability_report(split.calib_y - mu_cal)
        bayesian = self._bayesian_summary(fit_result, samples=bayesian_samples, alpha=alpha)
        donor_summary = donor_bootstrap(split.fit_y, donor_population=None) if model_type == "donor" else None

        forecast_df = pd.DataFrame(
            {
                "time": split.forecast_t,
                "prediction": mu_fore,
                "lower": conformal_forecast.lower,
                "upper": conformal_forecast.upper,
                "observed": split.forecast_y,
            }
        )

        radius = float(np.mean(conformal_forecast.upper - conformal_forecast.lower) / 2) if conformal_forecast.upper.size else 0.0

        phase_quantiles_cal = {str(k): float(v) for k, v in (conformal_calib.q_hats or {}).items()}
        phase_counts_cal = {str(k): int(v) for k, v in conformal_calib.phase_counts.items()}
        phase_quantiles_forecast = {str(k): float(v) for k, v in conformal_forecast.q_hats.items()}
        phase_counts_forecast = {str(k): int(v) for k, v in conformal_forecast.phase_counts.items()}

        pipeline_summary = {
            "data_preparation": {
                "rows": int(len(df)),
                "time_column": time_column,
                "target_column": target_column,
                "sorted_by_time": True,
                "monotonic": prep.monotonic,
            },
            "split": {
                "fit_count": int(split.fit_t.size),
                "calibration_count": int(split.calib_t.size),
                "forecast_count": int(split.forecast_t.size),
                "train_ratio": train_ratio,
            },
            "model_fitting": {
                "model_label": model_label,
                "theta_hat": {k: float(v) for k, v in fit_result.params.items()},
                "param_cis": param_cis,
                "mu_hat_fit": mu_fit.tolist(),
                "mu_hat_cal": mu_cal.tolist(),
            },
            "conformal_calibration": {
                "residuals": residuals_cal.tolist(),
                "sigma_hat": sigma_cal.tolist(),
                "scores": scores_cal.tolist(),
                "phase_assignment": phases_cal.tolist(),
                "phase_quantiles": phase_quantiles_cal,
                "phase_counts": phase_counts_cal,
                "scale_mode": cache.scale_mode,
                "coverage_curve": coverage_diag,
            },
            "prediction": {
                "mu_hat_forecast": mu_fore.tolist(),
                "sigma_hat_new": sigma_new.tolist(),
                "phase_assignment": conformal_forecast.phase_assignment.astype(int).tolist(),
                "phase_quantiles": phase_quantiles_forecast,
                "phase_counts": phase_counts_forecast,
                "intervals": {
                    "lower": conformal_forecast.lower.tolist(),
                    "upper": conformal_forecast.upper.tolist(),
                },
            },
            "validation": {
                "achieved_coverage": metrics.get("coverage_forecast"),
                "target_coverage": confidence,
                "forecast_count": int(split.forecast_t.size),
            },
        }

        result = {
            "success": True,
            "time_column": time_column,
            "target_column": target_column,
            "confidence": confidence,
            "train_ratio": train_ratio,
            "radius": radius,
            "model": {
                "name": model_type,
                "label": model_label,
                "params": {**fit_result.params, "param_cis": param_cis},
                "diagnostics": fit_result.diagnostics,
            },
            "metrics": metrics,
            "forecast_segment": forecast_df.to_dict(orient="list"),
            "diagnostics": {
                "scale_mode": conformal_forecast.scale_mode_used,
                "exchangeability": exchangeability,
                "phase_counts": coverage_diag.get("per_phase_coverage"),
                "monotonic_series": prep.monotonic,
            },
            "conformal": {
                "lower": conformal_forecast.lower.tolist(),
                "upper": conformal_forecast.upper.tolist(),
                "q_hats": {str(k): v for k, v in conformal_forecast.q_hats.items()},
                "phase_assignment": conformal_forecast.phase_assignment.tolist(),
                "coverage": coverage_diag,
            },
            "bayesian": bayesian,
            "donor_bootstrap": donor_summary,
            "pipeline": pipeline_summary,
        }
        return result

    # ------------------------------------------------------------------
    # Legacy pipeline integration
    # ------------------------------------------------------------------

    def _run_legacy_forecast(
        self,
        *,
        preparation: DataPreparationResult,
        confidence: float,
        train_ratio: float,
        model_type: ModelName,
    ) -> Dict[str, Any]:
        if self._legacy_core is None:
            self._legacy_core = LegacyPipelineCore()

        legacy_method_map = {
            "fractional": "FK",
            "kww": "KWW",
            "donor": "DONOR_PRONY2",
            "classical": "auto",
        }
        legacy_method = legacy_method_map.get(model_type, "auto")

        legacy_opts = {
            "stage_widening": self.legacy_options.get("stage_widening", True),
            "scale_override": self.legacy_options.get("scale_override", "auto"),
            "donor_pooling": self.legacy_options.get("donor_pooling", model_type == "donor"),
            "prior_tau": float(self.legacy_options.get("prior_tau", 0.1)),
            "hybrid": bool(self.legacy_options.get("hybrid", False)),
        }

        legacy_result = self._legacy_core.run_enhanced_forecast_pipeline(
            preparation.frame.copy(),
            channel=preparation.target_column,
            cut_percent=float(np.clip(train_ratio, 0.40, 0.95)),
            c0_method=legacy_method,
            stage_widening=legacy_opts["stage_widening"],
            scale_override=legacy_opts["scale_override"],
            donor_pooling=legacy_opts["donor_pooling"],
            prior_tau=legacy_opts["prior_tau"],
            hybrid=legacy_opts["hybrid"],
        )
        if not legacy_result.get("success", False):
            raise RuntimeError(legacy_result.get("error", "Legacy pipeline failed"))

        return self._adapt_legacy_result(
            legacy_result=legacy_result,
            preparation=preparation,
            confidence=confidence,
            train_ratio=train_ratio,
            model_type=model_type,
        )

    def _adapt_legacy_result(
        self,
        *,
        legacy_result: Dict[str, Any],
        preparation: DataPreparationResult,
        confidence: float,
        train_ratio: float,
        model_type: ModelName,
    ) -> Dict[str, Any]:
        tail = legacy_result.get("tail", {})
        prefix = legacy_result.get("prefix", {})
        uncertainty = legacy_result.get("uncertainty", {})
        bands = legacy_result.get("bands", {})
        metrics_legacy = legacy_result.get("metrics", {})

        times = np.asarray(tail.get("t", []), dtype=float)
        predictions = np.asarray(
            tail.get("mu_monotone", tail.get("mu_raw", [])),
            dtype=float,
        )
        observed = np.asarray(
            tail.get("y_actual", tail.get("y", [])),
            dtype=float,
        )

        stage_factors = np.asarray(
            uncertainty.get("cvplus_debug", {}).get("stage", []),
            dtype=float,
        )
        scale_mode = uncertainty.get("scale_mode", "additive")

        levels = [float(level) for level in bands.get("levels", [])]
        lower_sets = bands.get("lower", [])
        upper_sets = bands.get("upper", [])

        lower_band = np.full_like(predictions, np.nan, dtype=float)
        upper_band = np.full_like(predictions, np.nan, dtype=float)
        if levels and lower_sets and upper_sets:
            # choose level closest to requested confidence
            diffs = [abs(level - confidence) for level in levels]
            idx = int(np.argmin(diffs))
            lower_band = np.asarray(lower_sets[idx], dtype=float)
            upper_band = np.asarray(upper_sets[idx], dtype=float)

        # Ensure lengths align
        target_len = predictions.shape[0]
        def _ensure_length(arr: np.ndarray) -> np.ndarray:
            if arr.size == target_len:
                return arr
            if arr.size == 0:
                return np.full(target_len, np.nan, dtype=float)
            return np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, arr.size),
                arr.astype(float),
            )

        lower_band = _ensure_length(lower_band)
        upper_band = _ensure_length(upper_band)
        observed = _ensure_length(observed)
        if stage_factors.size:
            stage_factors = _ensure_length(stage_factors)

        radius = float(np.nanmean((upper_band - lower_band) / 2.0)) if target_len else 0.0

        coverage_dict = uncertainty.get("coverage", {})
        coverage_key = str(int(round(confidence * 100)))
        coverage_forecast = coverage_dict.get(coverage_key)
        if coverage_forecast is None and target_len:
            mask = ~np.isnan(observed)
            if mask.any():
                within = (observed[mask] >= lower_band[mask]) & (observed[mask] <= upper_band[mask])
                coverage_forecast = float(np.mean(within))

        metrics = {
            "rmse_calib": None,
            "mae_calib": None,
            "coverage_calib": None,
            "rmse_forecast": metrics_legacy.get("rmse"),
            "mae_forecast": metrics_legacy.get("mae"),
            "coverage_forecast": coverage_forecast,
            "confidence": confidence,
        }

        q_hats = {}
        if "r90" in uncertainty:
            q_hats["90"] = float(uncertainty["r90"])
        if "r95" in uncertainty:
            q_hats["95"] = float(uncertainty["r95"])

        forecast_df = pd.DataFrame(
            {
                "time": times.tolist(),
                "prediction": predictions.tolist(),
                "lower": lower_band.tolist(),
                "upper": upper_band.tolist(),
                "observed": observed.tolist(),
            }
        )

        pipeline_summary = {
            "data_preparation": {
                "rows": int(len(preparation.frame)),
                "time_column": preparation.time_column,
                "target_column": preparation.target_column,
                "sorted_by_time": True,
                "monotonic": preparation.monotonic,
            },
            "split": {
                "fit_count": int(len(prefix.get("t", []))),
                "calibration_count": int(len(prefix.get("t", []))),  # legacy uses fold CV
                "forecast_count": int(target_len),
                "train_ratio": train_ratio,
            },
            "model_fitting": {
                "model_label": legacy_result.get("model", {}).get("selected"),
                "theta_hat": legacy_result.get("model", {}).get("params", {}),
                "mu_hat_fit": prefix.get("mu_chosen", []),
            },
            "conformal_calibration": {
                "scale_mode": scale_mode,
                "coverage": coverage_dict,
                "stage_widening": stage_factors.tolist() if stage_factors.size else [],
                "base_quantiles": uncertainty.get("cvplus_debug", {}).get("base_quantiles", {}),
            },
            "prediction": {
                "mu_hat_forecast": predictions.tolist(),
                "intervals": {
                    "lower": lower_band.tolist(),
                    "upper": upper_band.tolist(),
                },
                "stage_factors": stage_factors.tolist() if stage_factors.size else [],
            },
            "validation": legacy_result.get("validation", {}),
        }

        diagnostics = {
            "scale_mode": scale_mode,
            "stage_factors": stage_factors.tolist() if stage_factors.size else [],
            "monotonic_series": preparation.monotonic,
            "performance_benchmark": legacy_result.get("performance_benchmark", {}),
        }

        conformal = {
            "lower": lower_band.tolist(),
            "upper": upper_band.tolist(),
            "q_hats": q_hats,
            "phase_assignment": [0] * target_len,
            "coverage": {
                "legacy": coverage_dict,
                "achieved": coverage_forecast,
            },
        }

        model_info = legacy_result.get("model", {})
        model_label = model_info.get("selected", model_type)
        model_params = model_info.get("params", {})

        result = {
            "success": True,
            "time_column": preparation.time_column,
            "target_column": preparation.target_column,
            "confidence": confidence,
            "train_ratio": train_ratio,
            "radius": radius,
            "model": {
                "name": model_type,
                "label": f"{model_label} (legacy)",
                "params": model_params,
                "diagnostics": {
                    "fit_quality": model_info.get("fit_quality"),
                },
            },
            "metrics": metrics,
            "forecast_segment": forecast_df.to_dict(orient="list"),
            "diagnostics": diagnostics,
            "conformal": conformal,
            "bayesian": {},
            "donor_bootstrap": None,
            "pipeline": pipeline_summary,
        }

        return result
    def _prepare_data(
        self,
        data: DataLike,
        time_column: Optional[str],
        target_column: Optional[str],
    ) -> DataPreparationResult:
        df, resolved_time, resolved_target = load_dataset(data, time_column, target_column)
        monotonic = self._is_monotonic(df[resolved_target].to_numpy(dtype=float))
        return DataPreparationResult(df, resolved_time, resolved_target, monotonic)

    @staticmethod
    def _is_monotonic(values: pd.Series | np.ndarray) -> bool:
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            return True
        diffs = np.diff(arr)
        return bool((diffs >= 0).all() or (diffs <= 0).all())

    def _split_series(self, t: np.ndarray, y: np.ndarray, train_ratio: float) -> _Split:
        if not 0.4 <= train_ratio <= 0.9:
            raise ValueError("train_ratio must lie in [0.4, 0.9]")
        n_total = y.size
        if n_total < 12:
            raise ValueError("Need at least 12 observations for forecasting")
        n_train_total = max(int(train_ratio * n_total), 8)
        n_train_total = min(n_train_total, n_total - 2)
        cal_size = max(2, int(self.cal_fraction * n_train_total))
        fit_size = n_train_total - cal_size
        if fit_size < 4:
            raise ValueError("Training window too small after calibration split")

        fit_t = t[:fit_size]
        fit_y = y[:fit_size]
        calib_t = t[fit_size:n_train_total]
        calib_y = y[fit_size:n_train_total]
        forecast_t = t[n_train_total:]
        forecast_y = y[n_train_total:]
        return _Split(fit_t, fit_y, calib_t, calib_y, forecast_t, forecast_y)

    def _fit_model(
        self,
        model_type: ModelName,
        t: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[FitResult, str, Callable[[np.ndarray], np.ndarray]]:
        model_type = model_type.lower()
        if model_type in {"fractional", "fk"}:
            model = FKModel()
            result = model.fit(t, y)
            return result, "Fractional kinetics", lambda x: model.predict(x)
        if model_type == "classical":
            model = ClassicalMassBalanceModel()
            result = model.fit(t, y)
            return result, "Classical mass balance", lambda x: model.predict(x)
        if model_type == "kww":
            model = KWWModel()
            result = model.fit(t, y)
            return result, "KWW stretched exponential", lambda x: model.predict(x)
        if model_type == "donor":
            model = DonorProny2Model()
            result = model.fit(t, y)
            return result, "Donor Prony-2", lambda x: model.predict(x)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def _refit_model(self, model_type: ModelName, t: np.ndarray, y: np.ndarray) -> FitResult:
        return self._fit_model(model_type, t, y)[0]

    def _compute_metrics(
        self,
        split: _Split,
        model_predict: Callable[[np.ndarray], np.ndarray],
        conformal: ConformalResult,
        coverage_diag: Dict[str, Any],
        confidence: float,
    ) -> Dict[str, Optional[float]]:
        calib_pred = model_predict(split.calib_t)
        forecast_pred = model_predict(split.forecast_t) if split.forecast_t.size else np.array([])

        rmse_calib = float(np.sqrt(np.mean((calib_pred - split.calib_y) ** 2))) if split.calib_y.size else None
        mae_calib = float(np.mean(np.abs(calib_pred - split.calib_y))) if split.calib_y.size else None
        coverage_calib = coverage_diag.get("coverage") if split.calib_y.size else None

        if split.forecast_y.size:
            rmse_forecast = float(np.sqrt(np.mean((forecast_pred - split.forecast_y) ** 2)))
            mae_forecast = float(np.mean(np.abs(forecast_pred - split.forecast_y)))
            inside = (split.forecast_y >= conformal.lower) & (split.forecast_y <= conformal.upper)
            coverage_forecast = float(np.mean(inside))
        else:
            rmse_forecast = None
            mae_forecast = None
            coverage_forecast = None

        return {
            "rmse_calib": rmse_calib,
            "mae_calib": mae_calib,
            "coverage_calib": coverage_calib,
            "rmse_forecast": rmse_forecast,
            "mae_forecast": mae_forecast,
            "coverage_forecast": coverage_forecast,
            "confidence": confidence,
        }

    def _bayesian_summary(
        self,
        fit_result: FitResult,
        *,
        samples: int,
        alpha: float,
    ) -> Dict[str, Any]:
        if samples <= 0 or stats is None:
            return {}
        rng = np.random.default_rng(self.random_state)
        residual_std = float(np.std(fit_result.residuals, ddof=1) or 1.0)
        draws: Dict[str, np.ndarray] = {}
        for name, value in fit_result.params.items():
            draws[name] = rng.normal(loc=value, scale=residual_std * 0.1, size=samples)
        means = {name: float(np.mean(vals)) for name, vals in draws.items()}
        lower = {name: float(np.percentile(vals, 100 * (alpha / 2))) for name, vals in draws.items()}
        upper = {name: float(np.percentile(vals, 100 * (1 - alpha / 2))) for name, vals in draws.items()}
        credible = {name: (lower[name], upper[name]) for name in draws}
        return {
            "posterior_means": means,
            "credible_intervals": credible,
            "acceptance_rate": 1.0,
            "sample_count": samples,
        }


__all__ = [
    "PICPCore",
    "load_dataset",
    "ClassicalMassBalanceModel",
    "ClassicalMassBalanceParams",
    "compute_parameter_cis",
    "donor_bootstrap",
    "exchangeability_report",
]

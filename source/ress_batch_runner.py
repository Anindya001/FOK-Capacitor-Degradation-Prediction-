"""RESS batch analysis runner for the CAPDATA3 / AEC-AST benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data" / "reference"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fractional_conformal import conformal_intervals
from fractional_core import FractionalConfig, _split_series
from fractional_diagnostics import empirical_coverage, information_criteria, mae, mean_interval_width, rmse, weighted_interval_score
from fractional_estimation import FKFitResult, fit_fractional_model
from fractional_model import FKParams, fractional_capacitance
from fractional_uq import PredictiveResults, failure_time_samples, laplace_draws, posterior_predictive
from surrogate_models import fit_classical_series, fit_kww_series, predict_classical, predict_kww


DEFAULT_EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "CAPDATA3.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "analysis"
DEFAULT_RANDOM_STATE = 42
DEFAULT_CONFIDENCE = 0.90
DEFAULT_THRESHOLD = 0.80


PUBLISHED_HYBRID_UNIT_BENCHMARK: tuple[dict[str, Any], ...] = (
    {"split": "50/50", "specimen": "C1", "rmse": 6.3954, "mae": 5.5215},
    {"split": "50/50", "specimen": "C2", "rmse": 9.9106, "mae": 8.1673},
    {"split": "50/50", "specimen": "C3", "rmse": 12.3606, "mae": 9.9389},
    {"split": "50/50", "specimen": "C4", "rmse": 9.4962, "mae": 8.2176},
    {"split": "50/50", "specimen": "C5", "rmse": 9.2643, "mae": 8.3541},
    {"split": "50/50", "specimen": "C6", "rmse": 5.9526, "mae": 4.9733},
    {"split": "50/50", "specimen": "C7", "rmse": 10.6405, "mae": 9.2614},
    {"split": "50/50", "specimen": "C8", "rmse": 7.3282, "mae": 6.6799},
    {"split": "60/40", "specimen": "C1", "rmse": 5.5669, "mae": 4.7351},
    {"split": "60/40", "specimen": "C2", "rmse": 8.7561, "mae": 7.2527},
    {"split": "60/40", "specimen": "C3", "rmse": 11.2429, "mae": 8.7198},
    {"split": "60/40", "specimen": "C4", "rmse": 8.6956, "mae": 7.8099},
    {"split": "60/40", "specimen": "C5", "rmse": 8.0516, "mae": 7.1081},
    {"split": "60/40", "specimen": "C6", "rmse": 5.7398, "mae": 4.8341},
    {"split": "60/40", "specimen": "C7", "rmse": 9.4294, "mae": 7.8698},
    {"split": "60/40", "specimen": "C8", "rmse": 6.3342, "mae": 5.7604},
    {"split": "70/30", "specimen": "C1", "rmse": 5.1711, "mae": 4.5464},
    {"split": "70/30", "specimen": "C2", "rmse": 8.8159, "mae": 7.6748},
    {"split": "70/30", "specimen": "C3", "rmse": 10.0386, "mae": 7.2268},
    {"split": "70/30", "specimen": "C4", "rmse": 7.8867, "mae": 7.1445},
    {"split": "70/30", "specimen": "C5", "rmse": 6.8703, "mae": 5.8415},
    {"split": "70/30", "specimen": "C6", "rmse": 5.7795, "mae": 5.1134},
    {"split": "70/30", "specimen": "C7", "rmse": 8.3404, "mae": 6.8401},
    {"split": "70/30", "specimen": "C8", "rmse": 6.1266, "mae": 5.7205},
)


def split_label(train_ratio: float) -> str:
    train_pct = int(round(train_ratio * 100.0))
    return f"{train_pct}/{100 - train_pct}"


def load_aec_ast_retention(excel_path: str | Path = DEFAULT_EXCEL_PATH) -> pd.DataFrame:
    """Load the CAPDATA3 workbook and convert loss to retention."""
    path = Path(excel_path)
    frame = pd.read_excel(path)
    time_col = "Ageing Time"
    value_cols = [col for col in frame.columns if col != time_col]
    renamed = {col: str(col).replace("% C@", "").replace("%", "").replace("@", "").strip() for col in value_cols}
    loss = frame.rename(columns={time_col: "time", **renamed})
    retention = loss.copy()
    for col in retention.columns:
        if col != "time":
            retention[col] = 100.0 - retention[col]
    return retention.sort_values("time").reset_index(drop=True)


def load_published_hybrid_unit_benchmark() -> pd.DataFrame:
    return pd.DataFrame.from_records(PUBLISHED_HYBRID_UNIT_BENCHMARK).sort_values(["split", "specimen"]).reset_index(drop=True)


def repo_relative_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_bundled_reference(filename: str, description: str) -> Path:
    csv_path = REFERENCE_DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"{description} not found: {csv_path}")
    return csv_path


def load_published_comparator_summary() -> pd.DataFrame:
    """Load bundled published splitwise averages used in the manuscript summary table."""
    csv_path = _require_bundled_reference("published_compact_summary.csv", "Published comparator summary table")
    return pd.read_csv(csv_path)


def load_published_comparator_unitwise(split: str = "60/40") -> pd.DataFrame:
    """Load the bundled specimen-wise published comparator table for a supported split."""
    if split != "60/40":
        raise ValueError("Only the 60/40 specimen-wise published comparator table is bundled in this submission.")
    csv_path = _require_bundled_reference("published_full_comparison_6040.csv", "Published 60/40 comparator table")
    return pd.read_csv(csv_path).sort_values(["metric", "specimen"]).reset_index(drop=True)


def load_published_comparator_suite(split: str = "60/40") -> pd.DataFrame:
    """Return the bundled specimen-wise published comparator table in tidy format."""
    frame = load_published_comparator_unitwise(split=split)
    value_columns = [col for col in frame.columns if col not in {"specimen", "metric"}]
    tidy = frame.melt(
        id_vars=["specimen", "metric"],
        value_vars=value_columns,
        var_name="model",
        value_name="published_value",
    )
    tidy["split_label"] = split
    return tidy.sort_values(["metric", "specimen", "model"]).reset_index(drop=True)


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if sorted_values.size == 0:
        return float("nan")
    quantile_level = min(1.0, max(0.0, (1.0 - alpha) * (1.0 + 1.0 / (sorted_values.size + 1.0))))
    rank = int(np.ceil(quantile_level * sorted_values.size)) - 1
    rank = min(max(rank, 0), sorted_values.size - 1)
    return float(sorted_values[rank])


def compute_nonconformity_scores(
    calibration_obs: np.ndarray,
    calibration_samples: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    lower_level = alpha / 2.0
    upper_level = 1.0 - lower_level
    base_lower = np.quantile(calibration_samples, lower_level, axis=0)
    base_upper = np.quantile(calibration_samples, upper_level, axis=0)
    scores = np.maximum.reduce(
        [
            base_lower - calibration_obs,
            calibration_obs - base_upper,
            np.zeros_like(calibration_obs, dtype=float),
        ]
    )
    q_hat = finite_sample_quantile(scores, alpha)
    return {
        "scores": scores,
        "base_lower": base_lower,
        "base_upper": base_upper,
        "q_hat": q_hat,
    }


def conformal_interval_from_quantile(
    test_samples: np.ndarray,
    q_hat: float,
    *,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    lower_level = alpha / 2.0
    upper_level = 1.0 - lower_level
    base_lower = np.quantile(test_samples, lower_level, axis=0)
    base_upper = np.quantile(test_samples, upper_level, axis=0)
    return {
        "base_lower": base_lower,
        "base_upper": base_upper,
        "lower": base_lower - q_hat,
        "upper": base_upper + q_hat,
        "q_hat": float(q_hat),
    }


@dataclass
class FOKCaseArtifacts:
    specimen: str
    train_ratio: float
    split: str
    time: np.ndarray
    retention: np.ndarray
    splits: dict[str, np.ndarray | int]
    fit: FKFitResult
    params_draws: list[FKParams]
    sigma_draws: np.ndarray
    predictive: PredictiveResults
    conformal: dict[str, np.ndarray | float]
    calibration_detail: dict[str, np.ndarray | float]
    forecast_prediction: np.ndarray
    info: dict[str, float]
    failure_samples: dict[float, np.ndarray]
    threshold_probability_by_end: float | None


def run_fractional_case(
    specimen: str,
    times: np.ndarray,
    retention: np.ndarray,
    *,
    train_ratio: float,
    confidence: float = DEFAULT_CONFIDENCE,
    n_draws: int = 2000,
    calibration_fraction: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold: float = DEFAULT_THRESHOLD,
) -> FOKCaseArtifacts:
    """Run the full FOK case analysis for one specimen and one split."""
    config = FractionalConfig(
        train_ratio=train_ratio,
        calibration_fraction=calibration_fraction,
        confidence=confidence,
        n_draws=n_draws,
        bootstrap_draws=0,
        thresholds=(threshold,),
        random_state=random_state,
        run_prequential=False,
    )
    split = _split_series(np.asarray(times, dtype=float), np.asarray(retention, dtype=float), config)
    fit = fit_fractional_model(split["fit_t"], split["fit_y"])
    forecast_prediction = fractional_capacitance(split["forecast_t"], fit.params)
    params_draws, sigma_draws = laplace_draws(fit, n_draws=n_draws, random_state=random_state)
    predictive = posterior_predictive(params_draws, sigma_draws, np.asarray(times, dtype=float), random_state=random_state)
    info = information_criteria(fit.residuals, fit.sigma, 4)

    fit_size = int(split["fit_size"])
    train_idx = int(split["train_idx"])
    cal_count = split["cal_t"].size
    forecast_count = split["forecast_t"].size
    alpha = 1.0 - confidence

    if cal_count and forecast_count:
        calibration_samples = predictive.total_samples[:, fit_size:train_idx]
        forecast_samples = predictive.total_samples[:, train_idx : train_idx + forecast_count]
        calibration_detail = compute_nonconformity_scores(split["cal_y"], calibration_samples, alpha=alpha)
        conformal = conformal_interval_from_quantile(forecast_samples, float(calibration_detail["q_hat"]), alpha=alpha)
        conformal["center"] = np.median(forecast_samples, axis=0)
        conformal["scale"] = 0.5 * (np.asarray(conformal["base_upper"]) - np.asarray(conformal["base_lower"]))
    else:
        calibration_detail = {"scores": np.array([], dtype=float), "base_lower": np.array([], dtype=float), "base_upper": np.array([], dtype=float), "q_hat": float("nan")}
        conformal = {"base_lower": np.array([], dtype=float), "base_upper": np.array([], dtype=float), "lower": np.array([], dtype=float), "upper": np.array([], dtype=float), "q_hat": float("nan"), "center": np.array([], dtype=float), "scale": np.array([], dtype=float)}

    failures = failure_time_samples(params_draws, [threshold])
    threshold_samples = failures[threshold]
    t_end = float(np.asarray(times, dtype=float)[-1])
    threshold_probability_by_end = None
    finite_threshold = threshold_samples[np.isfinite(threshold_samples)]
    if finite_threshold.size:
        threshold_probability_by_end = float(np.mean(finite_threshold <= t_end))

    return FOKCaseArtifacts(
        specimen=specimen,
        train_ratio=float(train_ratio),
        split=split_label(train_ratio),
        time=np.asarray(times, dtype=float),
        retention=np.asarray(retention, dtype=float),
        splits=split,
        fit=fit,
        params_draws=params_draws,
        sigma_draws=np.asarray(sigma_draws, dtype=float),
        predictive=predictive,
        conformal=conformal,
        calibration_detail=calibration_detail,
        forecast_prediction=np.asarray(forecast_prediction, dtype=float),
        info=info,
        failure_samples=failures,
        threshold_probability_by_end=threshold_probability_by_end,
    )


def _empty_param_columns() -> dict[str, float]:
    return {
        "C0": np.nan,
        "k": np.nan,
        "alpha": np.nan,
        "f_inf": np.nan,
        "tau": np.nan,
        "beta": np.nan,
        "C_inf": np.nan,
        "Delta": np.nan,
        "lambda": np.nan,
    }


def _fok_row(case: FOKCaseArtifacts) -> dict[str, Any]:
    split = case.splits
    forecast_y = np.asarray(split["forecast_y"], dtype=float)
    conformal_lower = np.asarray(case.conformal["lower"], dtype=float)
    conformal_upper = np.asarray(case.conformal["upper"], dtype=float)
    alpha = 1.0 - DEFAULT_CONFIDENCE
    row = {
        "specimen": case.specimen,
        "split": case.split,
        "train_ratio": case.train_ratio,
        "model": "FOK",
        "rmse_forecast": rmse(forecast_y, case.forecast_prediction),
        "mae_forecast": mae(forecast_y, case.forecast_prediction),
        "sigma": float(case.fit.sigma),
        "loglik": float(case.info["loglik"]),
        "AIC": float(case.info["AIC"]),
        "BIC": float(case.info["BIC"]),
        "n_fit": int(np.asarray(split["fit_t"]).size),
        "n_calibration": int(np.asarray(split["cal_t"]).size),
        "n_forecast": int(np.asarray(split["forecast_t"]).size),
        "coverage_90": empirical_coverage(forecast_y, conformal_lower, conformal_upper),
        "interval_width_90": mean_interval_width(conformal_lower, conformal_upper),
        "wis_90": weighted_interval_score(forecast_y, conformal_lower, conformal_upper, alpha=alpha),
        "threshold_q80_median": float(np.nanmedian(case.failure_samples[DEFAULT_THRESHOLD])),
        "threshold_q80_q05": float(np.nanpercentile(case.failure_samples[DEFAULT_THRESHOLD], 5)),
        "threshold_q80_q95": float(np.nanpercentile(case.failure_samples[DEFAULT_THRESHOLD], 95)),
        "p_threshold_q80_by_end": case.threshold_probability_by_end,
        "published_hybrid_available": True,
    }
    row.update(_empty_param_columns())
    row.update(
        {
            "C0": float(case.fit.params.C0),
            "k": float(case.fit.params.k),
            "alpha": float(case.fit.params.alpha),
            "f_inf": float(case.fit.params.f_inf),
        }
    )
    return row


def _baseline_row(
    *,
    specimen: str,
    train_ratio: float,
    model_name: str,
    fit_result: dict[str, Any],
    forecast_obs: np.ndarray,
    forecast_pred: np.ndarray,
    n_fit: int,
    n_calibration: int,
    n_forecast: int,
) -> dict[str, Any]:
    row = {
        "specimen": specimen,
        "split": split_label(train_ratio),
        "train_ratio": float(train_ratio),
        "model": model_name,
        "rmse_forecast": rmse(forecast_obs, forecast_pred),
        "mae_forecast": mae(forecast_obs, forecast_pred),
        "sigma": float(fit_result["sigma"]),
        "loglik": float(fit_result["loglik"]),
        "AIC": float(fit_result["AIC"]),
        "BIC": float(fit_result["BIC"]),
        "n_fit": int(n_fit),
        "n_calibration": int(n_calibration),
        "n_forecast": int(n_forecast),
        "coverage_90": np.nan,
        "interval_width_90": np.nan,
        "wis_90": np.nan,
        "threshold_q80_median": np.nan,
        "threshold_q80_q05": np.nan,
        "threshold_q80_q95": np.nan,
        "p_threshold_q80_by_end": np.nan,
        "published_hybrid_available": True,
    }
    row.update(_empty_param_columns())
    params = fit_result["params"]
    if model_name == "Classical":
        row.update(
            {
                "C_inf": float(params["C_inf"]),
                "Delta": float(params["Delta"]),
                "lambda": float(params["lambda"]),
            }
        )
    elif model_name == "KWW":
        row.update(
            {
                "C_inf": float(params["C_inf"]),
                "tau": float(params["tau"]),
                "beta": float(params["beta"]),
                "C0": float(params["C0"]),
            }
        )
    return row


def run_batch(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    train_ratios: list[float] | tuple[float, ...] = (0.5, 0.6, 0.7),
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence: float = DEFAULT_CONFIDENCE,
    n_draws: int = 2000,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_csv: bool = True,
    return_case_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple[str, str], FOKCaseArtifacts]]:
    """Run FOK, Classical, and KWW across all specimens and requested splits."""
    retention = load_aec_ast_retention(excel_path)
    times = retention["time"].to_numpy(dtype=float)
    specimens = [col for col in retention.columns if col != "time"]

    rows: list[dict[str, Any]] = []
    case_details: dict[tuple[str, str], FOKCaseArtifacts] = {}

    for specimen in specimens:
        values = retention[specimen].to_numpy(dtype=float)
        for ratio in train_ratios:
            case = run_fractional_case(
                specimen,
                times,
                values,
                train_ratio=float(ratio),
                confidence=confidence,
                n_draws=n_draws,
                random_state=random_state,
            )
            case_details[(specimen, case.split)] = case
            rows.append(_fok_row(case))

            split = case.splits
            classical_fit = fit_classical_series(split["fit_t"], split["fit_y"])
            kww_fit = fit_kww_series(split["fit_t"], split["fit_y"])

            classical_forecast = predict_classical(split["forecast_t"], classical_fit["params"])
            kww_forecast = predict_kww(split["forecast_t"], kww_fit["params"])
            forecast_obs = np.asarray(split["forecast_y"], dtype=float)

            rows.append(
                _baseline_row(
                    specimen=specimen,
                    train_ratio=float(ratio),
                    model_name="Classical",
                    fit_result=classical_fit,
                    forecast_obs=forecast_obs,
                    forecast_pred=classical_forecast,
                    n_fit=np.asarray(split["fit_t"]).size,
                    n_calibration=np.asarray(split["cal_t"]).size,
                    n_forecast=np.asarray(split["forecast_t"]).size,
                )
            )
            rows.append(
                _baseline_row(
                    specimen=specimen,
                    train_ratio=float(ratio),
                    model_name="KWW",
                    fit_result=kww_fit,
                    forecast_obs=forecast_obs,
                    forecast_pred=kww_forecast,
                    n_fit=np.asarray(split["fit_t"]).size,
                    n_calibration=np.asarray(split["cal_t"]).size,
                    n_forecast=np.asarray(split["forecast_t"]).size,
                )
            )

    results_df = pd.DataFrame.from_records(rows).sort_values(["specimen", "train_ratio", "model"]).reset_index(drop=True)
    if save_csv:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_dir / "ress_batch_results.csv", index=False)
        load_published_hybrid_unit_benchmark().to_csv(out_dir / "published_hybrid_unit_benchmark.csv", index=False)
        manifest = {
            "excel_path": repo_relative_path(excel_path),
            "train_ratios": [float(r) for r in train_ratios],
            "confidence": float(confidence),
            "n_draws": int(n_draws),
            "random_state": int(random_state),
            "rows": int(results_df.shape[0]),
        }
        (out_dir / "ress_batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if return_case_details:
        return results_df, case_details
    return results_df


if __name__ == "__main__":
    frame = run_batch()
    print(f"Wrote {repo_relative_path(DEFAULT_OUTPUT_DIR / 'ress_batch_results.csv')}")
    print(frame.head().to_string(index=False))

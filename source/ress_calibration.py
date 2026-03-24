"""Calibration diagnostics for the RESS CAPDATA3 revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ress_batch_runner import (
    DEFAULT_CONFIDENCE,
    DEFAULT_OUTPUT_DIR,
    FOKCaseArtifacts,
    compute_nonconformity_scores,
    conformal_interval_from_quantile,
    finite_sample_quantile,
)
from src.fractional_diagnostics import empirical_coverage, mean_interval_width, weighted_interval_score


def _wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    spread = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return float(center - spread), float(center + spread)


def reliability_diagram(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    nominal_levels: np.ndarray = np.arange(0.1, 1.0, 0.1),
) -> dict[str, pd.DataFrame]:
    """Compute pooled conformal reliability curves across specimens."""
    rows: list[dict[str, object]] = []
    mace_rows: list[dict[str, object]] = []
    for split in sorted({case.split for case in case_details.values()}):
        split_cases = [case for case in case_details.values() if case.split == split]
        abs_errors = []
        for nominal in nominal_levels:
            alpha = 1.0 - float(nominal)
            success = 0
            total = 0
            widths: list[float] = []
            for case in split_cases:
                fit_size = int(case.splits["fit_size"])
                train_idx = int(case.splits["train_idx"])
                forecast_obs = np.asarray(case.splits["forecast_y"], dtype=float)
                calibration_obs = np.asarray(case.splits["cal_y"], dtype=float)
                cal_samples = case.predictive.total_samples[:, fit_size:train_idx]
                forecast_samples = case.predictive.total_samples[:, train_idx : train_idx + forecast_obs.size]
                cal = compute_nonconformity_scores(calibration_obs, cal_samples, alpha=alpha)
                interval = conformal_interval_from_quantile(forecast_samples, float(cal["q_hat"]), alpha=alpha)
                inside = (forecast_obs >= interval["lower"]) & (forecast_obs <= interval["upper"])
                success += int(np.sum(inside))
                total += int(forecast_obs.size)
                widths.append(mean_interval_width(interval["lower"], interval["upper"]))
            empirical = success / total if total else float("nan")
            ci_low, ci_high = _wilson_interval(success, total)
            abs_errors.append(abs(empirical - nominal))
            rows.append(
                {
                    "split": split,
                    "nominal": float(nominal),
                    "empirical": float(empirical),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "pooled_points": int(total),
                    "mean_interval_width": float(np.mean(widths)),
                }
            )
        mace_rows.append({"split": split, "MACE": float(np.mean(abs_errors))})
    return {
        "curve": pd.DataFrame.from_records(rows),
        "mace": pd.DataFrame.from_records(mace_rows),
    }


def hierarchical_conformal_ablation(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    alpha = 1.0 - confidence
    for split in sorted({case.split for case in case_details.values()}):
        split_cases = [case for case in case_details.values() if case.split == split]
        local_coverages: list[float] = []
        local_widths: list[float] = []
        local_wis: list[float] = []
        pooled_scores = []
        for case in split_cases:
            forecast_obs = np.asarray(case.splits["forecast_y"], dtype=float)
            lower = np.asarray(case.conformal["lower"], dtype=float)
            upper = np.asarray(case.conformal["upper"], dtype=float)
            local_coverages.append(empirical_coverage(forecast_obs, lower, upper))
            local_widths.append(mean_interval_width(lower, upper))
            local_wis.append(weighted_interval_score(forecast_obs, lower, upper, alpha=alpha))
            pooled_scores.append(np.asarray(case.calibration_detail["scores"], dtype=float))
        pooled_q = finite_sample_quantile(np.concatenate(pooled_scores), alpha)
        pooled_coverages: list[float] = []
        pooled_widths: list[float] = []
        pooled_wis: list[float] = []
        for case in split_cases:
            train_idx = int(case.splits["train_idx"])
            forecast_obs = np.asarray(case.splits["forecast_y"], dtype=float)
            forecast_samples = case.predictive.total_samples[:, train_idx : train_idx + forecast_obs.size]
            interval = conformal_interval_from_quantile(forecast_samples, pooled_q, alpha=alpha)
            pooled_coverages.append(empirical_coverage(forecast_obs, interval["lower"], interval["upper"]))
            pooled_widths.append(mean_interval_width(interval["lower"], interval["upper"]))
            pooled_wis.append(weighted_interval_score(forecast_obs, interval["lower"], interval["upper"], alpha=alpha))
        rows.extend(
            [
                {
                    "split": split,
                    "scheme": "per-specimen",
                    "mean_coverage": float(np.mean(local_coverages)),
                    "mean_width": float(np.mean(local_widths)),
                    "mean_wis": float(np.mean(local_wis)),
                },
                {
                    "split": split,
                    "scheme": "hierarchical-pooled",
                    "mean_coverage": float(np.mean(pooled_coverages)),
                    "mean_width": float(np.mean(pooled_widths)),
                    "mean_wis": float(np.mean(pooled_wis)),
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def save_calibration_outputs(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reliability = reliability_diagram(case_details)
    curve_path = out_dir / "ress_reliability_curve.csv"
    mace_path = out_dir / "ress_reliability_mace.csv"
    reliability["curve"].to_csv(curve_path, index=False)
    reliability["mace"].to_csv(mace_path, index=False)

    hierarchical = hierarchical_conformal_ablation(case_details)
    hierarchical_path = out_dir / "ress_hierarchical_conformal_ablation.csv"
    hierarchical.to_csv(hierarchical_path, index=False)

    manifest = {
        "curve_rows": int(reliability["curve"].shape[0]),
        "splits": sorted(reliability["curve"]["split"].unique().tolist()),
    }
    manifest_path = out_dir / "ress_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "curve": curve_path,
        "mace": mace_path,
        "hierarchical": hierarchical_path,
        "manifest": manifest_path,
    }

"""Statistical validation utilities for the RESS CAPDATA3 revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from ress_batch_runner import DEFAULT_OUTPUT_DIR, FOKCaseArtifacts, load_published_hybrid_unit_benchmark


def _rank_biserial_from_differences(differences: np.ndarray) -> float:
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.abs(diff) > 1e-12]
    if diff.size == 0:
        return 0.0
    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    ranks = np.empty_like(abs_diff, dtype=float)
    i = 0
    while i < diff.size:
        j = i
        while j + 1 < diff.size and abs_diff[order[j + 1]] == abs_diff[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    denom = positive + negative
    return float((positive - negative) / denom) if denom else 0.0


def _safe_norm_ppf(value: float) -> float:
    clipped = min(max(value, 1e-10), 1.0 - 1e-10)
    return float(stats.norm.ppf(clipped))


def _bca_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int,
    alpha: float,
    random_state: int,
) -> tuple[float, float]:
    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size < 2:
        theta = statistic(data)
        return float(theta), float(theta)

    rng = np.random.default_rng(random_state)
    theta_hat = float(statistic(data))
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=data.size, replace=True)
        boot_stats[i] = float(statistic(sample))

    prop_less = (np.sum(boot_stats < theta_hat) + 0.5 * np.sum(boot_stats == theta_hat)) / n_bootstrap
    z0 = _safe_norm_ppf(prop_less)

    jackknife = np.empty(data.size, dtype=float)
    for i in range(data.size):
        jackknife[i] = float(statistic(np.delete(data, i)))
    jack_mean = float(np.mean(jackknife))
    numerator = np.sum((jack_mean - jackknife) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
    acceleration = float(numerator / denominator) if denominator > 0 else 0.0

    z_low = _safe_norm_ppf(alpha / 2.0)
    z_high = _safe_norm_ppf(1.0 - alpha / 2.0)

    def _adjust(z_alpha: float) -> float:
        denom = 1.0 - acceleration * (z0 + z_alpha)
        if abs(denom) < 1e-12:
            return 0.5
        return float(stats.norm.cdf(z0 + (z0 + z_alpha) / denom))

    adj_low = min(max(_adjust(z_low), 0.0), 1.0)
    adj_high = min(max(_adjust(z_high), 0.0), 1.0)
    low = float(np.quantile(boot_stats, adj_low))
    high = float(np.quantile(boot_stats, adj_high))
    return low, high


def hodges_lehmann_paired(differences: np.ndarray) -> float:
    diff = np.asarray(differences, dtype=float).reshape(-1)
    walsh = []
    for i in range(diff.size):
        for j in range(i, diff.size):
            walsh.append(0.5 * (diff[i] + diff[j]))
    return float(np.median(np.asarray(walsh, dtype=float)))


def cliffs_delta_against_zero(differences: np.ndarray) -> float:
    diff = np.asarray(differences, dtype=float)
    if diff.size == 0:
        return 0.0
    positive = float(np.sum(diff > 0))
    negative = float(np.sum(diff < 0))
    return float((positive - negative) / diff.size)


def bayesian_bootstrap_probability_superiority(
    differences: np.ndarray,
    *,
    n_draws: int = 20_000,
    rope: float = 0.0,
    random_state: int = 42,
) -> dict[str, float]:
    diff = np.asarray(differences, dtype=float).reshape(-1)
    if diff.size == 0:
        return {"p_superior": float("nan"), "p_equivalent": float("nan"), "p_inferior": float("nan")}
    rng = np.random.default_rng(random_state)
    weights = rng.dirichlet(np.ones(diff.size), size=n_draws)
    weighted_mean = weights @ diff
    return {
        "p_superior": float(np.mean(weighted_mean > rope)),
        "p_equivalent": float(np.mean(np.abs(weighted_mean) <= rope)),
        "p_inferior": float(np.mean(weighted_mean < -rope)),
    }


def bootstrap_rmse_comparison(
    rmse_fok: np.ndarray,
    rmse_comp: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """BCa bootstrap CI on the mean paired RMSE difference."""
    fok = np.asarray(rmse_fok, dtype=float).reshape(-1)
    comp = np.asarray(rmse_comp, dtype=float).reshape(-1)
    if fok.shape != comp.shape:
        raise ValueError("rmse_fok and rmse_comp must share the same shape.")
    diff = comp - fok

    mean_diff = float(np.mean(diff))
    bca_low, bca_high = _bca_interval(diff, np.mean, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state)
    hl_est = hodges_lehmann_paired(diff)
    hl_low, hl_high = _bca_interval(diff, hodges_lehmann_paired, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state + 1)
    cliffs = cliffs_delta_against_zero(diff)
    cliffs_low, cliffs_high = _bca_interval(diff, cliffs_delta_against_zero, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state + 2)
    bayes = bayesian_bootstrap_probability_superiority(diff, random_state=random_state + 3)
    return {
        "mean_diff": mean_diff,
        "bca_ci_low": float(bca_low),
        "bca_ci_high": float(bca_high),
        "hodges_lehmann": float(hl_est),
        "hl_ci_low": float(hl_low),
        "hl_ci_high": float(hl_high),
        "cliffs_delta": float(cliffs),
        "cliffs_ci_low": float(cliffs_low),
        "cliffs_ci_high": float(cliffs_high),
        **bayes,
    }


def _comparison_frame(results_df: pd.DataFrame, comparator: str) -> pd.DataFrame:
    fok = results_df.loc[results_df["model"] == "FOK", ["specimen", "split", "rmse_forecast", "mae_forecast"]].copy()
    if comparator == "Hybrid":
        comp = load_published_hybrid_unit_benchmark().rename(columns={"specimen": "specimen", "split": "split", "rmse": "rmse_forecast", "mae": "mae_forecast"})
    else:
        comp = results_df.loc[results_df["model"] == comparator, ["specimen", "split", "rmse_forecast", "mae_forecast"]].copy()
    merged = fok.merge(comp, on=["specimen", "split"], suffixes=("_fok", "_comp")).sort_values(["split", "specimen"]).reset_index(drop=True)
    return merged


def _comparison_pairs(results_df: pd.DataFrame, comparator: str) -> tuple[np.ndarray, np.ndarray]:
    merged = _comparison_frame(results_df, comparator)
    return merged["rmse_forecast_fok"].to_numpy(dtype=float), merged["rmse_forecast_comp"].to_numpy(dtype=float)


def compute_splitwise_comparisons(
    results_df: pd.DataFrame,
    comparator: str,
    *,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = _comparison_frame(results_df, comparator)
    rows: list[dict[str, object]] = []
    for split, group in merged.groupby("split", sort=True):
        fok_rmse = group["rmse_forecast_fok"].to_numpy(dtype=float)
        comp_rmse = group["rmse_forecast_comp"].to_numpy(dtype=float)
        diff = comp_rmse - fok_rmse
        stat = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox", method="exact")
        boot = bootstrap_rmse_comparison(fok_rmse, comp_rmse, n_bootstrap=n_bootstrap, random_state=random_state)
        rows.append(
            {
                "comparison": f"FOK vs {comparator}",
                "split": split,
                "n_pairs": int(diff.size),
                "mean_fok_rmse": float(np.mean(fok_rmse)),
                "mean_comp_rmse": float(np.mean(comp_rmse)),
                "mean_delta_rmse": float(np.mean(diff)),
                "median_delta_rmse": float(np.median(diff)),
                "p_value_exact": float(stat.pvalue),
                "rank_biserial": _rank_biserial_from_differences(diff),
                "wins": int(np.sum(diff > 0.0)),
                "ties": int(np.sum(np.abs(diff) <= 1e-12)),
                **boot,
            }
        )

    pooled_diff = merged["rmse_forecast_comp"].to_numpy(dtype=float) - merged["rmse_forecast_fok"].to_numpy(dtype=float)
    summary = {
        "comparison": f"FOK vs {comparator}",
        "scope": "splitwise_inference",
        "pooled_descriptive_mean_delta_rmse": float(np.mean(pooled_diff)),
        "pooled_descriptive_median_delta_rmse": float(np.median(pooled_diff)),
        "pooled_wins": int(np.sum(pooled_diff > 0.0)),
        "pooled_pairs": int(pooled_diff.size),
        "note": "The same specimens contribute to all three nested splits; pooled 24-case summaries are descriptive only.",
    }
    return pd.DataFrame.from_records(rows), summary


def compute_pairwise_comparisons(
    results_df: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparator in ("Hybrid", "Classical", "KWW"):
        fok_rmse, comp_rmse = _comparison_pairs(results_df, comparator)
        diff = comp_rmse - fok_rmse
        stat = stats.wilcoxon(fok_rmse, comp_rmse, alternative="less", method="exact")
        boot = bootstrap_rmse_comparison(fok_rmse, comp_rmse, n_bootstrap=n_bootstrap, random_state=random_state)
        rows.append(
            {
                "comparison": f"FOK vs {comparator}",
                "scope": "pooled_descriptive_only",
                "inferentially_independent": False,
                "n_pairs": int(fok_rmse.size),
                "mean_fok_rmse": float(np.mean(fok_rmse)),
                "mean_comp_rmse": float(np.mean(comp_rmse)),
                "mean_delta_rmse": float(np.mean(diff)),
                "median_delta_rmse": float(np.median(diff)),
                "p_value_exact": float(stat.pvalue),
                "rank_biserial": _rank_biserial_from_differences(diff),
                "note": "Pooled 24-case summaries aggregate nested splits from the same specimens and are descriptive only.",
                **boot,
            }
        )
    return pd.DataFrame.from_records(rows)


def compute_fok_kww_alpha_correlation(results_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    fok = results_df.loc[results_df["model"] == "FOK", ["specimen", "split", "alpha", "rmse_forecast"]].rename(columns={"rmse_forecast": "rmse_fok"})
    kww = results_df.loc[results_df["model"] == "KWW", ["specimen", "split", "rmse_forecast"]].rename(columns={"rmse_forecast": "rmse_kww"})
    merged = fok.merge(kww, on=["specimen", "split"], how="inner").sort_values(["split", "specimen"]).reset_index(drop=True)
    merged["one_minus_alpha"] = 1.0 - merged["alpha"]
    merged["rmse_gain"] = merged["rmse_kww"] - merged["rmse_fok"]
    spearman = stats.spearmanr(merged["one_minus_alpha"], merged["rmse_gain"])
    kendall = stats.kendalltau(merged["one_minus_alpha"], merged["rmse_gain"])
    summary = {
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
    }
    return merged, summary


def compute_residual_bias(case_details: dict[tuple[str, str], FOKCaseArtifacts]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for (specimen, split), case in sorted(case_details.items()):
        forecast_t = np.asarray(case.splits["forecast_t"], dtype=float)
        forecast_y = np.asarray(case.splits["forecast_y"], dtype=float)
        residual = forecast_y - case.forecast_prediction
        lead_time = forecast_t - float(forecast_t[0])
        case_rows.append(
            {
                "specimen": specimen,
                "split": split,
                "mean_signed_residual": float(np.mean(residual)),
                "median_signed_residual": float(np.median(residual)),
                "n_forecast": int(residual.size),
            }
        )
        for t, lead, obs, pred, res in zip(forecast_t, lead_time, forecast_y, case.forecast_prediction, residual, strict=True):
            point_rows.append(
                {
                    "specimen": specimen,
                    "split": split,
                    "time": float(t),
                    "lead_time": float(lead),
                    "observed": float(obs),
                    "predicted": float(pred),
                    "signed_residual": float(res),
                }
            )

    point_df = pd.DataFrame.from_records(point_rows)
    case_df = pd.DataFrame.from_records(case_rows)

    for split, group in case_df.groupby("split", sort=True):
        point_group = point_df.loc[point_df["split"] == split].copy()
        stat = stats.wilcoxon(group["mean_signed_residual"].to_numpy(dtype=float), alternative="two-sided", method="exact")
        regression = stats.linregress(point_group["lead_time"].to_numpy(dtype=float), point_group["signed_residual"].to_numpy(dtype=float))
        split_rows.append(
            {
                "split": split,
                "mean_case_bias": float(group["mean_signed_residual"].mean()),
                "median_case_bias": float(group["mean_signed_residual"].median()),
                "wilcoxon_p": float(stat.pvalue),
                "n_cases": int(group.shape[0]),
                "pooled_points": int(point_group.shape[0]),
                "lead_slope": float(regression.slope),
                "lead_intercept": float(regression.intercept),
                "lead_rvalue": float(regression.rvalue),
                "lead_pvalue": float(regression.pvalue),
            }
        )
    split_df = pd.DataFrame.from_records(split_rows)
    return point_df, case_df, split_df


def save_statistical_outputs(
    results_df: pd.DataFrame,
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairwise = compute_pairwise_comparisons(results_df, n_bootstrap=n_bootstrap, random_state=random_state)
    pairwise_path = out_dir / "ress_pairwise_statistics.csv"
    pairwise.to_csv(pairwise_path, index=False)

    classical_splitwise, classical_splitwise_summary = compute_splitwise_comparisons(
        results_df,
        "Classical",
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    classical_splitwise_path = out_dir / "ress_classical_splitwise_statistics.csv"
    classical_splitwise.to_csv(classical_splitwise_path, index=False)
    classical_splitwise_summary_path = out_dir / "ress_classical_splitwise_summary.json"
    classical_splitwise_summary_path.write_text(json.dumps(classical_splitwise_summary, indent=2), encoding="utf-8")

    alpha_pairs, alpha_summary = compute_fok_kww_alpha_correlation(results_df)
    alpha_pairs_path = out_dir / "ress_fok_kww_gain_vs_alpha.csv"
    alpha_pairs.to_csv(alpha_pairs_path, index=False)
    alpha_summary_path = out_dir / "ress_fok_kww_gain_vs_alpha_summary.json"
    alpha_summary_path.write_text(json.dumps(alpha_summary, indent=2), encoding="utf-8")

    residual_points, residual_cases, residual_summary = compute_residual_bias(case_details)
    residual_points_path = out_dir / "ress_residual_bias_points.csv"
    residual_cases_path = out_dir / "ress_residual_bias_cases.csv"
    residual_summary_path = out_dir / "ress_residual_bias_summary.csv"
    residual_points.to_csv(residual_points_path, index=False)
    residual_cases.to_csv(residual_cases_path, index=False)
    residual_summary.to_csv(residual_summary_path, index=False)

    return {
        "pairwise": pairwise_path,
        "classical_splitwise": classical_splitwise_path,
        "classical_splitwise_summary": classical_splitwise_summary_path,
        "alpha_pairs": alpha_pairs_path,
        "alpha_summary": alpha_summary_path,
        "residual_points": residual_points_path,
        "residual_cases": residual_cases_path,
        "residual_summary": residual_summary_path,
    }

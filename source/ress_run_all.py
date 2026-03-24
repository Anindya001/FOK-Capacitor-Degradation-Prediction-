"""Master runner for the RESS CAPDATA3 resubmission analyses."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ress_batch_runner import (
    DEFAULT_EXCEL_PATH,
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    load_published_comparator_summary,
    load_published_comparator_suite,
    load_published_hybrid_unit_benchmark,
    repo_relative_path,
    run_batch,
)
from ress_calibration import reliability_diagram, save_calibration_outputs
from ress_decomposition import epistemic_aleatoric_width_ratio, save_decomposition_outputs
from ress_figures import save_figure_bundle
from ress_sensitivity import save_sensitivity_outputs
from ress_model_selection import bootstrap_split_mean_metric, save_model_selection_outputs
from ress_statistical_tests import (
    compute_fok_kww_alpha_correlation,
    compute_pairwise_comparisons,
    compute_residual_bias,
    compute_splitwise_comparisons,
    save_statistical_outputs,
)
from ress_threshold_times import ranking_validation, save_threshold_outputs, threshold_time_table


PACKAGE_ROOT = REPO_ROOT / "output"


def _relative_path_mapping(paths: dict[str, str | Path]) -> dict[str, str]:
    return {key: repo_relative_path(value) for key, value in paths.items()}


def package_submission_outputs(analysis_dir: Path, *, exclude_names: set[str] | None = None) -> dict[str, list[str]]:
    exclude_names = exclude_names or set()
    package_dirs = {
        ".png": PACKAGE_ROOT / "figures",
        ".csv": PACKAGE_ROOT / "tables_csv",
        ".json": PACKAGE_ROOT / "metadata",
    }
    mirrored: dict[str, list[str]] = {"figures": [], "tables_csv": [], "metadata": []}
    for path in sorted(analysis_dir.iterdir()):
        if not path.is_file() or path.name in exclude_names:
            continue
        target_dir = package_dirs.get(path.suffix.lower())
        if target_dir is None:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        mirrored[target_dir.name].append(repo_relative_path(target))
    for key in mirrored:
        mirrored[key] = sorted(set(mirrored[key]))
    return mirrored


def build_full_comparison_table(results_df: pd.DataFrame, *, split: str = "60/40") -> pd.DataFrame:
    published = load_published_comparator_suite(split=split)
    sub = published.loc[
        (published["split_label"] == split)
        & (published["metric"].isin(["RMSE", "MAE"]))
        & (~published["specimen"].str.contains("avg", case=False, na=False))
    ].copy()
    pivot = sub.pivot_table(index=["specimen", "metric"], columns="model", values="published_value").reset_index()
    fok = (
        results_df.loc[results_df["split"] == split, ["specimen", "rmse_forecast", "mae_forecast", "model"]]
        .query("model == 'FOK'")
        .drop(columns="model")
        .copy()
    )
    fok_long = pd.concat(
        [
            fok.rename(columns={"rmse_forecast": "FOK"})[["specimen", "FOK"]].assign(metric="RMSE"),
            fok.rename(columns={"mae_forecast": "FOK"})[["specimen", "FOK"]].assign(metric="MAE"),
        ],
        ignore_index=True,
    )
    merged = pivot.merge(fok_long, on=["specimen", "metric"], how="left")
    return merged.sort_values(["metric", "specimen"]).reset_index(drop=True)


def build_compact_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    published_summary = load_published_comparator_summary().set_index("Model")

    ours = (
        results_df.loc[results_df["model"].isin(["FOK", "KWW", "Classical"])]
        .groupby(["model", "split"], as_index=False)[["rmse_forecast", "mae_forecast"]]
        .mean()
    )
    ours_rmse = ours.pivot(index="model", columns="split", values="rmse_forecast")
    ours_mae = ours.pivot(index="model", columns="split", values="mae_forecast")

    rows = []
    for row_label in ("ARIMA", "SES", "Bi-LSTM", "CTB", "Hybrid"):
        row = {"Model": row_label}
        for split in ("50/50", "60/40", "70/30"):
            row[f"{split}_RMSE"] = float(published_summary.loc[row_label, f"{split}_RMSE"])
            row[f"{split}_MAE"] = float(published_summary.loc[row_label, f"{split}_MAE"])
        rows.append(row)
    for row_label in ("KWW", "Classical", "FOK"):
        row = {"Model": row_label}
        for split in ("50/50", "60/40", "70/30"):
            row[f"{split}_RMSE"] = float(ours_rmse.loc[row_label, split])
            row[f"{split}_MAE"] = float(ours_mae.loc[row_label, split])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_direct_hybrid_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    hybrid = load_published_hybrid_unit_benchmark()
    fok = results_df.loc[results_df["model"] == "FOK", ["specimen", "split", "rmse_forecast", "mae_forecast"]].copy()
    merged = fok.merge(hybrid, on=["specimen", "split"], how="inner", suffixes=("_fok", "_hybrid"))
    rows = []
    for split, group in merged.groupby("split", sort=True):
        diff = group["rmse"].to_numpy(dtype=float) - group["rmse_forecast"].to_numpy(dtype=float)
        boots = np.empty(10_000, dtype=float)
        rng = np.random.default_rng(42)
        for i in range(10_000):
            sample = rng.choice(diff, size=diff.size, replace=True)
            boots[i] = float(np.mean(sample))
        rows.append(
            {
                "split": split,
                "fok_rmse_mean": float(group["rmse_forecast"].mean()),
                "hybrid_rmse_mean": float(group["rmse"].mean()),
                "mean_rmse_gain": float(np.mean(diff)),
                "rmse_gain_ci_low": float(np.quantile(boots, 0.025)),
                "rmse_gain_ci_high": float(np.quantile(boots, 0.975)),
                "fok_mae_mean": float(group["mae_forecast"].mean()),
                "hybrid_mae_mean": float(group["mae"].mean()),
                "unit_wins_rmse": int(np.sum(group["rmse_forecast"] < group["rmse"])),
                "unit_wins_mae": int(np.sum(group["mae_forecast"] < group["mae"])),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("split").reset_index(drop=True)


def build_manuscript_metrics(
    pairwise: pd.DataFrame,
    classical_splitwise: pd.DataFrame,
    threshold_table_df: pd.DataFrame,
    threshold_ranking: dict[str, float],
    ratio_df: pd.DataFrame,
    reliability_mace_df: pd.DataFrame,
    hierarchical_df: pd.DataFrame,
) -> dict[str, object]:
    hybrid_row = pairwise.loc[pairwise["comparison"] == "FOK vs Hybrid"].iloc[0]
    ratio_summary = ratio_df.groupby("location")["ratio"].mean().to_dict()
    best_split = reliability_mace_df.sort_values("MACE").iloc[0]
    hierarchical_best = hierarchical_df.sort_values(["split", "scheme"]).to_dict(orient="records")
    highest_priority = threshold_table_df.iloc[0]
    lowest_priority = threshold_table_df.iloc[-1]
    return {
        "hybrid_mean_diff": float(hybrid_row["mean_diff"]),
        "hybrid_ci_low": float(hybrid_row["bca_ci_low"]),
        "hybrid_ci_high": float(hybrid_row["bca_ci_high"]),
        "hybrid_scope": "descriptive_only_at_specimen_level",
        "classical_splitwise": classical_splitwise.to_dict(orient="records"),
        "threshold_highest_priority": highest_priority.to_dict(),
        "threshold_lowest_priority": lowest_priority.to_dict(),
        "threshold_ranking_validation": threshold_ranking,
        "ratio_summary": ratio_summary,
        "best_calibration_split": best_split.to_dict(),
        "hierarchical_summary": hierarchical_best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel-path", default=str(DEFAULT_EXCEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--sa-time-grid", type=int, default=50)
    parser.add_argument("--sa-rho", type=float, default=0.20)
    parser.add_argument("--sa-samples", type=int, default=512)
    parser.add_argument("--sa-bootstrap", type=int, default=200)
    parser.add_argument("--sa-trajectories", type=int, default=20)
    parser.add_argument("--sa-levels", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df, case_details = run_batch(
        args.excel_path,
        n_draws=args.n_draws,
        output_dir=output_dir,
        save_csv=True,
        return_case_details=True,
    )

    statistical_paths = save_statistical_outputs(results_df, case_details, output_dir=output_dir, n_bootstrap=args.n_bootstrap)
    pairwise = compute_pairwise_comparisons(results_df, n_bootstrap=args.n_bootstrap)
    classical_splitwise, _ = compute_splitwise_comparisons(results_df, "Classical", n_bootstrap=args.n_bootstrap)
    alpha_pairs, alpha_summary = compute_fok_kww_alpha_correlation(results_df)
    residual_points, _, residual_summary = compute_residual_bias(case_details)

    model_selection_paths = save_model_selection_outputs(results_df, output_dir=output_dir)
    mechanistic_rmse = bootstrap_split_mean_metric(results_df, metric="rmse_forecast", n_bootstrap=args.n_bootstrap)

    calibration_paths = save_calibration_outputs(case_details, output_dir=output_dir)
    reliability = reliability_diagram(case_details)
    hierarchical = pd.read_csv(calibration_paths["hierarchical"])

    threshold_paths = save_threshold_outputs(case_details, output_dir=output_dir)
    threshold_table_df = threshold_time_table(case_details)
    threshold_ranking = ranking_validation(threshold_table_df)

    decomposition_paths = save_decomposition_outputs(case_details, output_dir=output_dir)
    ratio_df = epistemic_aleatoric_width_ratio(case_details)

    sensitivity_paths = save_sensitivity_outputs(
        case_details,
        output_dir=output_dir,
        time_grid_size=args.sa_time_grid,
        rho=args.sa_rho,
        sobol_samples=args.sa_samples,
        sobol_bootstrap=args.sa_bootstrap,
        morris_trajectories=args.sa_trajectories,
        morris_levels=args.sa_levels,
        random_state=42,
    )

    figures = save_figure_bundle(
        results_df=results_df,
        case_details=case_details,
        alpha_pairs=alpha_pairs,
        alpha_summary=alpha_summary,
        mechanistic_rmse=mechanistic_rmse,
        reliability_curve=reliability["curve"],
        reliability_mace=reliability["mace"],
        hierarchical_summary=hierarchical,
        threshold_table=threshold_table_df,
        residual_points=residual_points,
        residual_summary=residual_summary,
        output_dir=output_dir,
    )

    full_6040 = build_full_comparison_table(results_df)
    full_6040_path = output_dir / "ress_full_comparison_6040.csv"
    full_6040.to_csv(full_6040_path, index=False)

    compact = build_compact_summary_table(results_df)
    compact_path = output_dir / "ress_compact_summary_table.csv"
    compact.to_csv(compact_path, index=False)

    direct_hybrid = build_direct_hybrid_comparison(results_df)
    direct_hybrid_path = output_dir / "ress_direct_hybrid_comparison.csv"
    direct_hybrid.to_csv(direct_hybrid_path, index=False)

    manuscript_metrics = build_manuscript_metrics(
        pairwise=pairwise,
        classical_splitwise=classical_splitwise,
        threshold_table_df=threshold_table_df,
        threshold_ranking=threshold_ranking,
        ratio_df=ratio_df,
        reliability_mace_df=reliability["mace"],
        hierarchical_df=hierarchical,
    )
    manuscript_metrics_path = output_dir / "ress_manuscript_metrics.json"
    manuscript_metrics_path.write_text(json.dumps(manuscript_metrics, indent=2), encoding="utf-8")

    packaged_outputs = package_submission_outputs(output_dir, exclude_names={"ress_run_all_manifest.json"})
    packaged_manifest_path = PACKAGE_ROOT / "metadata" / "ress_run_all_manifest.json"
    packaged_outputs["metadata"] = sorted(set([*packaged_outputs["metadata"], repo_relative_path(packaged_manifest_path)]))

    manifest = {
        "analysis_dir": repo_relative_path(output_dir),
        "source_excel": repo_relative_path(args.excel_path),
        "batch_rows": int(results_df.shape[0]),
        "statistical_outputs": _relative_path_mapping(statistical_paths),
        "model_selection_outputs": _relative_path_mapping(model_selection_paths),
        "calibration_outputs": _relative_path_mapping(calibration_paths),
        "threshold_outputs": _relative_path_mapping(threshold_paths),
        "decomposition_outputs": _relative_path_mapping(decomposition_paths),
        "sensitivity_outputs": _relative_path_mapping(sensitivity_paths),
        "figures": _relative_path_mapping(figures),
        "tables": {
            "full_6040": repo_relative_path(full_6040_path),
            "compact_summary": repo_relative_path(compact_path),
            "direct_hybrid": repo_relative_path(direct_hybrid_path),
        },
        "manuscript_metrics": repo_relative_path(manuscript_metrics_path),
        "packaged_outputs": packaged_outputs,
    }
    manifest_path = output_dir / "ress_run_all_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    packaged_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.resolve() != packaged_manifest_path.resolve():
        shutil.copy2(manifest_path, packaged_manifest_path)
    print(f"Wrote RESS resubmission outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

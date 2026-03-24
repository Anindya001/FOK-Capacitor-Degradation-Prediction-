"""Mechanistic model-selection utilities for the RESS revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ress_batch_runner import DEFAULT_OUTPUT_DIR


def mechanistic_subset(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.loc[results_df["model"].isin(["FOK", "KWW", "Classical"])].copy()


def build_mechanistic_aic_bic_table(results_df: pd.DataFrame, *, split: str = "60/40") -> pd.DataFrame:
    frame = mechanistic_subset(results_df)
    frame = frame.loc[frame["split"] == split].copy()
    frame["delta_AIC"] = frame.groupby("specimen")["AIC"].transform(lambda s: s - s.min())
    frame["delta_BIC"] = frame.groupby("specimen")["BIC"].transform(lambda s: s - s.min())
    return frame.sort_values(["specimen", "model"]).reset_index(drop=True)


def summarize_aic_bic_wins(aic_bic_table: pd.DataFrame) -> dict[str, dict[str, int]]:
    best_aic = aic_bic_table.sort_values(["specimen", "AIC"]).groupby("specimen").first().reset_index()
    best_bic = aic_bic_table.sort_values(["specimen", "BIC"]).groupby("specimen").first().reset_index()
    return {
        "best_AIC_count": best_aic["model"].value_counts().sort_index().to_dict(),
        "best_BIC_count": best_bic["model"].value_counts().sort_index().to_dict(),
    }


def bootstrap_split_mean_metric(
    results_df: pd.DataFrame,
    *,
    metric: str = "rmse_forecast",
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    frame = mechanistic_subset(results_df)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    for split, split_group in frame.groupby("split", sort=True):
        for model, group in split_group.groupby("model", sort=True):
            values = group[metric].to_numpy(dtype=float)
            boots = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                sample = rng.choice(values, size=values.size, replace=True)
                boots[i] = float(np.mean(sample))
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "ci_low": float(np.quantile(boots, 0.025)),
                    "ci_high": float(np.quantile(boots, 0.975)),
                    "n_units": int(values.size),
                }
            )
    return pd.DataFrame.from_records(rows)


def save_model_selection_outputs(
    results_df: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "60/40",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_mechanistic_aic_bic_table(results_df, split=split)
    table_path = out_dir / f"ress_mechanistic_aic_bic_{split.replace('/', '')}.csv"
    table.to_csv(table_path, index=False)

    wins = summarize_aic_bic_wins(table)
    wins_path = out_dir / f"ress_mechanistic_aic_bic_wins_{split.replace('/', '')}.json"
    wins_path.write_text(json.dumps(wins, indent=2), encoding="utf-8")

    rmse_summary = bootstrap_split_mean_metric(results_df, metric="rmse_forecast")
    rmse_summary_path = out_dir / "ress_mechanistic_split_mean_rmse.csv"
    rmse_summary.to_csv(rmse_summary_path, index=False)

    mae_summary = bootstrap_split_mean_metric(results_df, metric="mae_forecast")
    mae_summary_path = out_dir / "ress_mechanistic_split_mean_mae.csv"
    mae_summary.to_csv(mae_summary_path, index=False)

    return {
        "aic_bic_table": table_path,
        "aic_bic_wins": wins_path,
        "rmse_summary": rmse_summary_path,
        "mae_summary": mae_summary_path,
    }

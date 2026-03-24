"""Publication-quality figure generation for the RESS CAPDATA3 revision."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ress_batch_runner import (
    DEFAULT_CONFIDENCE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLD,
    FOKCaseArtifacts,
    REPO_ROOT,
    conformal_interval_from_quantile,
    finite_sample_quantile,
    load_published_hybrid_unit_benchmark,
)
from fractional_model import fractional_capacitance


MANUSCRIPT_FIGURE_DIR = REPO_ROOT / "output" / "figures"

PALETTE = {
    "ink": "#1F2933",
    "grid": "#D7DEE5",
    "fok": "#138A61",
    "classical": "#C8572A",
    "kww": "#7A6AAE",
    "hybrid": "#2C7FB8",
    "train": "#111111",
    "heldout": "#0B63A5",
    "epistemic": "#6BAED6",
    "total": "#BFDDED",
    "conformal": "#19A7CE",
    "boundary": "#B23A48",
    "threshold": "#C98B00",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
            "mathtext.default": "regular",
        }
    )


def _save_figure(
    fig: plt.Figure,
    out_path: str | Path,
    *,
    copy_to_manuscript: bool = True,
    tight_rect: tuple[float, float, float, float] | None = None,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if copy_to_manuscript:
        MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, MANUSCRIPT_FIGURE_DIR / path.name)
    return path


def _split_sort_key(label: str) -> tuple[int, int]:
    left, right = label.split("/")
    return int(left), int(right)


def _deterministic_curve(case: FOKCaseArtifacts) -> np.ndarray:
    return np.asarray(fractional_capacitance(case.time, case.fit.params), dtype=float)


def _display_limits(curves: list[np.ndarray], *, lower: float = 50.0, upper: float = 102.0) -> tuple[float, float]:
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in curves if np.size(arr)])
    if finite.size == 0:
        return lower, upper
    y_min = max(lower, float(np.floor(np.min(finite) - 3.0)))
    y_max = min(upper, float(np.ceil(np.max(finite) + 2.0)))
    if y_max <= y_min:
        return lower, upper
    return y_min, y_max


def _pooled_q_by_split(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float]:
    alpha = 1.0 - float(confidence)
    q_by_split: dict[str, float] = {}
    for split in sorted({case.split for case in case_details.values()}, key=_split_sort_key):
        scores = [
            np.asarray(case.calibration_detail["scores"], dtype=float)
            for case in case_details.values()
            if case.split == split and np.asarray(case.calibration_detail["scores"]).size
        ]
        if not scores:
            q_by_split[split] = float("nan")
            continue
        q_by_split[split] = float(finite_sample_quantile(np.concatenate(scores), alpha))
    return q_by_split


def _interval_for_case(
    case: FOKCaseArtifacts,
    *,
    interval_mode: str,
    pooled_q_by_split: dict[str, float] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, np.ndarray | float]:
    if interval_mode == "per-specimen":
        return case.conformal
    if interval_mode != "hierarchical-pooled":
        raise ValueError(f"Unknown interval_mode: {interval_mode}")
    if pooled_q_by_split is None or not np.isfinite(pooled_q_by_split.get(case.split, np.nan)):
        return case.conformal

    train_idx = int(case.splits["train_idx"])
    forecast_obs = np.asarray(case.splits["forecast_y"], dtype=float)
    forecast_samples = case.predictive.total_samples[:, train_idx : train_idx + forecast_obs.size]
    alpha = 1.0 - float(confidence)
    interval = conformal_interval_from_quantile(forecast_samples, float(pooled_q_by_split[case.split]), alpha=alpha)
    interval["center"] = np.median(forecast_samples, axis=0)
    interval["scale"] = 0.5 * (np.asarray(interval["base_upper"]) - np.asarray(interval["base_lower"]))
    return interval


def _plot_forecast_grid(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    out_path: str | Path,
    *,
    specimens: tuple[str, ...],
    splits: tuple[str, ...],
    interval_mode: str,
    figsize: tuple[float, float],
    rmse_fontsize: float,
    train_markersize: float,
    heldout_size: float,
    line_width: float,
    tight_rect: tuple[float, float, float, float],
) -> Path:
    _configure_matplotlib()
    fig, axes = plt.subplots(len(specimens), len(splits), figsize=figsize, sharex=True, sharey=True)
    pooled_q = _pooled_q_by_split(case_details) if interval_mode == "hierarchical-pooled" else None
    if len(specimens) == 1 and len(splits) == 1:
        axes = np.array([[axes]])
    elif len(specimens) == 1:
        axes = np.array([axes])
    elif len(splits) == 1:
        axes = np.array([[ax] for ax in axes])
    y_curves: list[np.ndarray] = []
    deterministic_by_case: dict[tuple[str, str], np.ndarray] = {}
    for specimen in specimens:
        for split in splits:
            case = case_details[(specimen, split)]
            deterministic = _deterministic_curve(case)
            deterministic_by_case[(specimen, split)] = deterministic
            y_curves.extend([np.asarray(case.retention, dtype=float), deterministic])
    y_min, y_max = _display_limits(y_curves)

    for row, specimen in enumerate(specimens):
        for col, split in enumerate(splits):
            case = case_details[(specimen, split)]
            ax = axes[row, col]
            train_idx = int(case.splits["train_idx"])
            forecast_t = np.asarray(case.splits["forecast_t"], dtype=float)
            forecast_y = np.asarray(case.splits["forecast_y"], dtype=float)
            deterministic = deterministic_by_case[(specimen, split)]
            interval = _interval_for_case(case, interval_mode=interval_mode, pooled_q_by_split=pooled_q)
            lower = np.clip(np.asarray(interval["lower"], dtype=float), y_min, y_max)
            upper = np.clip(np.asarray(interval["upper"], dtype=float), y_min, y_max)
            rmse_val = float(np.sqrt(np.mean((forecast_y - case.forecast_prediction) ** 2)))

            label_train = "Training data" if row == 0 and col == 0 else None
            label_holdout = "Held-out" if row == 0 and col == 0 else None
            label_mean = "FOK mean" if row == 0 and col == 0 else None
            band_label = "Pooled conformal 90% band" if interval_mode == "hierarchical-pooled" else "Conformal 90% band"
            label_band = band_label if row == 0 and col == 0 else None
            ax.plot(case.time[:train_idx], case.retention[:train_idx], color=PALETTE["train"], marker="o", markersize=train_markersize, linewidth=line_width, label=label_train)
            ax.scatter(forecast_t, forecast_y, color=PALETTE["heldout"], s=heldout_size, zorder=4, label=label_holdout)
            ax.plot(case.time, np.clip(deterministic, y_min, y_max), color=PALETTE["fok"], linewidth=line_width + 0.3, label=label_mean)
            ax.fill_between(forecast_t, lower, upper, color=PALETTE["total"], alpha=0.55, label=label_band)
            ax.axvline(case.time[train_idx], color=PALETTE["boundary"], linestyle="--", linewidth=1.0)
            ax.axhline(DEFAULT_THRESHOLD * case.retention[0], color=PALETTE["threshold"], linestyle=":", linewidth=1.0)
            ax.set_title(f"{specimen} | {split}")
            ax.set_ylim(y_min, y_max)
            ax.text(
                0.03,
                0.05,
                f"RMSE = {rmse_val:.2f}",
                transform=ax.transAxes,
                fontsize=rmse_fontsize,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#B8C1CC", alpha=0.92),
            )
            if row == len(specimens) - 1:
                ax.set_xlabel("Ageing time (h)")
            if col == 0:
                ax.set_ylabel("Retention (%)")
    legend_handles = [
        Line2D([0], [0], color=PALETTE["train"], marker="o", markersize=4, linewidth=1.2, label="Training data"),
        Line2D([0], [0], color=PALETTE["heldout"], marker="o", linestyle="None", markersize=5, label="Held-out"),
        Line2D([0], [0], color=PALETTE["fok"], linewidth=1.6, label="FOK mean"),
        Patch(facecolor=PALETTE["total"], alpha=0.55, label="Pooled conformal 90% band" if interval_mode == "hierarchical-pooled" else "Conformal 90% band"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    return _save_figure(fig, out_path, tight_rect=tight_rect)


def plot_reduced_forecast_panels(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    out_path: str | Path,
    *,
    specimens: tuple[str, ...] = ("C2", "C8"),
    splits: tuple[str, ...] = ("50/50", "60/40", "70/30"),
    interval_mode: str = "hierarchical-pooled",
) -> Path:
    return _plot_forecast_grid(
        case_details,
        out_path,
        specimens=specimens,
        splits=splits,
        interval_mode=interval_mode,
        figsize=(10.6, 6.35),
        rmse_fontsize=8.5,
        train_markersize=3.0,
        heldout_size=18.0,
        line_width=1.2,
        tight_rect=(0.0, 0.0, 1.0, 0.96),
    )


def plot_full_forecast_panel_groups(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    output_dir: str | Path,
    *,
    interval_mode: str = "hierarchical-pooled",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "full_forecast_group1": ("C1", "C2", "C3", "C4"),
        "full_forecast_group2": ("C5", "C6", "C7", "C8"),
    }
    outputs: dict[str, Path] = {}
    for name, specimens in groups.items():
        outputs[name] = _plot_forecast_grid(
            case_details,
            out_dir / f"ress_{name}.png",
            specimens=specimens,
            splits=("50/50", "60/40", "70/30"),
            interval_mode=interval_mode,
            figsize=(10.8, 11.0),
            rmse_fontsize=7.2,
            train_markersize=2.4,
            heldout_size=11.0,
            line_width=1.0,
            tight_rect=(0.0, 0.0, 1.0, 0.975),
        )
    return outputs


def plot_decomposition_panels(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    out_path: str | Path,
    *,
    split: str = "60/40",
    specimens: tuple[str, ...] = ("C3", "C5", "C2"),
    interval_mode: str = "hierarchical-pooled",
) -> Path:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, len(specimens), figsize=(8.2, 4.1), sharey=True)
    pooled_q = _pooled_q_by_split(case_details) if interval_mode == "hierarchical-pooled" else None
    y_curves: list[np.ndarray] = []
    deterministic_by_specimen: dict[str, np.ndarray] = {}
    for specimen in specimens:
        case = case_details[(specimen, split)]
        deterministic = _deterministic_curve(case)
        deterministic_by_specimen[specimen] = deterministic
        y_curves.extend([np.asarray(case.retention, dtype=float), deterministic])
    y_min, y_max = _display_limits(y_curves, lower=55.0, upper=106.0)

    for ax, specimen in zip(axes, specimens, strict=True):
        case = case_details[(specimen, split)]
        train_idx = int(case.splits["train_idx"])
        forecast_t = np.asarray(case.splits["forecast_t"], dtype=float)
        forecast_y = np.asarray(case.splits["forecast_y"], dtype=float)
        deterministic = deterministic_by_specimen[specimen]
        interval = _interval_for_case(case, interval_mode=interval_mode, pooled_q_by_split=pooled_q)
        epistemic = np.quantile(case.predictive.mu_samples[:, train_idx:], [0.05, 0.95], axis=0)
        total = np.quantile(case.predictive.total_samples[:, train_idx:], [0.05, 0.95], axis=0)
        epistemic_lower = np.clip(epistemic[0], y_min, y_max)
        epistemic_upper = np.clip(epistemic[1], y_min, y_max)
        total_lower = np.clip(total[0], y_min, y_max)
        total_upper = np.clip(total[1], y_min, y_max)
        rmse_val = float(np.sqrt(np.mean((forecast_y - case.forecast_prediction) ** 2)))

        ax.plot(case.time[:train_idx], case.retention[:train_idx], color=PALETTE["train"], marker="o", markersize=3, linewidth=1.3)
        ax.scatter(forecast_t, forecast_y, color=PALETTE["heldout"], s=18, zorder=5)
        ax.plot(case.time, np.clip(deterministic, y_min, y_max), color=PALETTE["fok"], linewidth=1.7)
        ax.fill_between(forecast_t, total_lower, total_upper, color=PALETTE["total"], alpha=0.48)
        ax.plot(forecast_t, total_lower, color="#86B8D1", linewidth=0.9, alpha=0.95)
        ax.plot(forecast_t, total_upper, color="#86B8D1", linewidth=0.9, alpha=0.95)
        ax.fill_between(forecast_t, epistemic_lower, epistemic_upper, color=PALETTE["epistemic"], alpha=0.72)
        ax.plot(forecast_t, epistemic_lower, color="#377EB8", linewidth=0.95, alpha=0.98)
        ax.plot(forecast_t, epistemic_upper, color="#377EB8", linewidth=0.95, alpha=0.98)
        ax.plot(forecast_t, np.clip(np.asarray(interval["lower"], dtype=float), y_min, y_max), color=PALETTE["conformal"], linestyle="--", linewidth=1.15)
        ax.plot(forecast_t, np.clip(np.asarray(interval["upper"], dtype=float), y_min, y_max), color=PALETTE["conformal"], linestyle="--", linewidth=1.15)
        ax.axvline(case.time[train_idx], color=PALETTE["boundary"], linestyle="--", linewidth=1.0)
        ax.axhline(DEFAULT_THRESHOLD * case.retention[0], color=PALETTE["threshold"], linestyle=":", linewidth=1.0)
        ax.text(0.03, 0.04, f"RMSE = {rmse_val:.2f}", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B8C1CC", alpha=0.9))
        ax.set_title(f"{specimen} | {split}")
        ax.set_xlabel("Ageing time (h)")
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("Retention (%)")
    legend_handles = [
        Line2D([0], [0], color=PALETTE["train"], marker="o", markersize=4, linewidth=1.3, label="Training data"),
        Line2D([0], [0], color=PALETTE["heldout"], marker="o", linestyle="None", markersize=5, label="Held-out"),
        Line2D([0], [0], color=PALETTE["fok"], linewidth=1.7, label="FOK mean"),
        Patch(facecolor=PALETTE["total"], edgecolor="#86B8D1", alpha=0.48, label="Total 90% band"),
        Patch(facecolor=PALETTE["epistemic"], edgecolor="#377EB8", alpha=0.72, label="Epistemic 90% band"),
        Line2D([0], [0], color=PALETTE["conformal"], linestyle="--", linewidth=1.15, label="Pooled conformal 90%" if interval_mode == "hierarchical-pooled" else "Conformal 90%"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    return _save_figure(fig, out_path, tight_rect=(0.0, 0.0, 1.0, 0.9))


def summarize_fok_calibration(results_df: pd.DataFrame) -> pd.DataFrame:
    frame = results_df.loc[results_df["model"] == "FOK", ["split", "coverage_90", "interval_width_90", "wis_90"]].copy()
    summary = (
        frame.groupby("split", as_index=False)
        .agg(
            coverage_90=("coverage_90", "mean"),
            interval_width_90=("interval_width_90", "mean"),
            wis_90=("wis_90", "mean"),
        )
        .sort_values("split", key=lambda s: s.map(_split_sort_key))
        .reset_index(drop=True)
    )
    return summary


def plot_coverage_diagnostics(results_df: pd.DataFrame, out_path: str | Path) -> Path:
    _configure_matplotlib()
    summary = summarize_fok_calibration(results_df)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.25))
    split_order = summary["split"].tolist()
    colours = [PALETTE["classical"], PALETTE["hybrid"], PALETTE["fok"]]
    x = np.arange(len(split_order))

    panels = [
        ("coverage_90", "Empirical coverage", (0.0, 1.0)),
        ("interval_width_90", "Mean interval width", None),
        ("wis_90", "Weighted interval score", None),
    ]
    for ax, (column, title, y_limits) in zip(axes, panels, strict=True):
        ax.bar(x, summary[column].to_numpy(dtype=float), color=colours, alpha=0.88, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(split_order)
        ax.set_xlabel("Train/test split")
        ax.set_title(title)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if column == "coverage_90":
            ax.axhline(0.90, color=PALETTE["boundary"], linestyle="--", linewidth=1.0)
            ax.set_ylabel("Empirical coverage")
        elif column == "interval_width_90":
            ax.set_ylabel("Mean width")
        else:
            ax.set_ylabel("WIS")
    return _save_figure(fig, out_path)


def plot_gain_vs_alpha(alpha_pairs: pd.DataFrame, summary: dict[str, float], out_path: str | Path) -> Path:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    split_order = sorted(alpha_pairs["split"].unique().tolist(), key=_split_sort_key)
    colours = {"50/50": PALETTE["classical"], "60/40": PALETTE["hybrid"], "70/30": PALETTE["fok"]}
    for split in split_order:
        group = alpha_pairs.loc[alpha_pairs["split"] == split]
        ax.scatter(group["one_minus_alpha"], group["rmse_gain"], label=split, color=colours[split], s=38, alpha=0.85)
    x = alpha_pairs["one_minus_alpha"].to_numpy(dtype=float)
    y = alpha_pairs["rmse_gain"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, color=PALETTE["ink"], linewidth=1.2)
    ax.axhline(0.0, color=PALETTE["boundary"], linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$1 - \hat{\alpha}$")
    ax.set_ylabel(r"$\mathrm{RMSE}_{KWW} - \mathrm{RMSE}_{FOK}$")
    ax.legend(frameon=False, title="Split")
    ax.text(0.04, 0.96, rf"Spearman $\rho$ = {summary['spearman_rho']:.2f}" + "\n" + rf"$p$ = {summary['spearman_p']:.3g}", transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B8C1CC", alpha=0.92))
    return _save_figure(fig, out_path)


def plot_fok_vs_hybrid_split_metrics(
    results_df: pd.DataFrame,
    out_path: str | Path,
    *,
    copy_to_manuscript: bool = True,
) -> Path:
    _configure_matplotlib()
    hybrid = load_published_hybrid_unit_benchmark()
    fok = (
        results_df.loc[results_df["model"] == "FOK", ["split", "rmse_forecast", "mae_forecast"]]
        .groupby("split", as_index=False)
        .mean()
        .rename(columns={"rmse_forecast": "rmse_fok", "mae_forecast": "mae_fok"})
    )
    hybrid_summary = (
        hybrid.groupby("split", as_index=False)[["rmse", "mae"]]
        .mean()
        .rename(columns={"rmse": "rmse_hybrid", "mae": "mae_hybrid"})
    )
    merged = fok.merge(hybrid_summary, on="split", how="inner")
    split_order = sorted(merged["split"].tolist(), key=_split_sort_key)
    merged = merged.set_index("split").loc[split_order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True)
    x = np.arange(len(split_order))
    width = 0.32
    panels = [
        ("rmse_fok", "rmse_hybrid", "RMSE", axes[0]),
        ("mae_fok", "mae_hybrid", "MAE", axes[1]),
    ]
    for fok_col, hybrid_col, title, ax in panels:
        fok_vals = merged[fok_col].to_numpy(dtype=float)
        hybrid_vals = merged[hybrid_col].to_numpy(dtype=float)
        ax.bar(x - width / 2, hybrid_vals, width=width, color=PALETTE["hybrid"], alpha=0.9, label="Hybrid")
        ax.bar(x + width / 2, fok_vals, width=width, color=PALETTE["fok"], alpha=0.92, label="FOK")
        reduction = 100.0 * (hybrid_vals - fok_vals) / hybrid_vals
        for xpos, y_hybrid, pct in zip(x, hybrid_vals, reduction, strict=True):
            ax.text(
                xpos,
                y_hybrid + max(0.12, 0.03 * float(np.max(hybrid_vals))),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(split_order)
        ax.set_xlabel("Train/test split")
        ax.set_ylabel(f"Mean forecast {title}")
        ax.set_title(title)
    axes[0].legend(frameon=False, loc="upper left")
    return _save_figure(fig, out_path, copy_to_manuscript=copy_to_manuscript)


def plot_fok_vs_hybrid_gain_heatmap(
    results_df: pd.DataFrame,
    out_path: str | Path,
    *,
    copy_to_manuscript: bool = True,
) -> Path:
    _configure_matplotlib()
    hybrid = load_published_hybrid_unit_benchmark()
    fok = results_df.loc[results_df["model"] == "FOK", ["specimen", "split", "rmse_forecast"]].copy()
    merged = fok.merge(hybrid.loc[:, ["specimen", "split", "rmse"]], on=["specimen", "split"], how="inner")
    merged["rmse_gain_pct"] = 100.0 * (merged["rmse"] - merged["rmse_forecast"]) / merged["rmse"]

    split_order = sorted(merged["split"].unique().tolist(), key=_split_sort_key)
    specimen_order = sorted(
        merged["specimen"].unique().tolist(),
        key=lambda label: int(str(label).replace("C", "")),
    )
    matrix = (
        merged.pivot(index="specimen", columns="split", values="rmse_gain_pct")
        .reindex(index=specimen_order, columns=split_order)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    image = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0.0, vmax=float(np.nanmax(matrix)))
    ax.set_xticks(np.arange(len(split_order)))
    ax.set_xticklabels(split_order)
    ax.set_yticks(np.arange(len(specimen_order)))
    ax.set_yticklabels(specimen_order)
    ax.set_xlabel("Train/test split")
    ax.set_ylabel("Specimen")
    for row, specimen in enumerate(specimen_order):
        for col, split in enumerate(split_order):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=PALETTE["ink"],
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("RMSE gain over Hybrid (%)")
    return _save_figure(fig, out_path, copy_to_manuscript=copy_to_manuscript)


def plot_mechanistic_rmse_bars(rmse_summary: pd.DataFrame, out_path: str | Path) -> Path:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    split_order = sorted(rmse_summary["split"].unique().tolist(), key=_split_sort_key)
    model_order = ["Classical", "KWW", "FOK"]
    colour_map = {"Classical": PALETTE["classical"], "KWW": PALETTE["kww"], "FOK": PALETTE["fok"]}
    x = np.arange(len(split_order))
    width = 0.23
    for i, model in enumerate(model_order):
        group = rmse_summary.loc[rmse_summary["model"] == model].set_index("split").loc[split_order]
        mean = group["mean"].to_numpy(dtype=float)
        low = group["ci_low"].to_numpy(dtype=float)
        high = group["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([mean - low, high - mean])
        ax.bar(x + (i - 1) * width, mean, width=width, color=colour_map[model], alpha=0.88, label=model, yerr=yerr, capsize=4, error_kw={"linewidth": 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(split_order)
    ax.set_xlabel("Train/test split")
    ax.set_ylabel("Mean forecast RMSE")
    ax.legend(frameon=False)
    return _save_figure(fig, out_path)


def plot_reliability_diagram(curve: pd.DataFrame, mace: pd.DataFrame, out_path: str | Path) -> Path:
    _configure_matplotlib()
    split_order = sorted(curve["split"].unique().tolist(), key=_split_sort_key)
    fig, axes = plt.subplots(1, len(split_order), figsize=(8.6, 3.2), sharey=True)
    for ax, split in zip(axes, split_order, strict=True):
        group = curve.loc[curve["split"] == split].sort_values("nominal")
        yerr = np.vstack(
            [
                group["empirical"].to_numpy(dtype=float) - group["ci_low"].to_numpy(dtype=float),
                group["ci_high"].to_numpy(dtype=float) - group["empirical"].to_numpy(dtype=float),
            ]
        )
        ax.errorbar(group["nominal"], group["empirical"], yerr=yerr, marker="o", color=PALETTE["hybrid"], linewidth=1.2, capsize=3)
        ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["boundary"], linewidth=1.0)
        mace_value = float(mace.loc[mace["split"] == split, "MACE"].iloc[0])
        ax.text(0.04, 0.96, f"MACE = {mace_value:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#B8C1CC", alpha=0.92))
        ax.set_title(split)
        ax.set_xlabel("Nominal coverage")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Empirical coverage")
    return _save_figure(fig, out_path)


def plot_calibration_overview(
    reliability_curve: pd.DataFrame,
    reliability_mace: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    split_order = sorted(reliability_curve["split"].unique().tolist(), key=_split_sort_key)
    colour_map = {"50/50": PALETTE["classical"], "60/40": PALETTE["hybrid"], "70/30": PALETTE["fok"]}

    ax = axes[0]
    for split in split_order:
        group = reliability_curve.loc[reliability_curve["split"] == split].sort_values("nominal")
        mace_value = float(reliability_mace.loc[reliability_mace["split"] == split, "MACE"].iloc[0])
        ax.plot(
            group["nominal"],
            group["empirical"],
            marker="o",
            markersize=3.6,
            linewidth=1.25,
            color=colour_map[split],
            label=f"{split} (MACE {mace_value:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["boundary"], linewidth=1.0)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Reliability")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    coverage_df = (
        hierarchical_summary.loc[:, ["split", "scheme", "mean_coverage", "mean_width"]]
        .copy()
        .sort_values(["split", "scheme"], key=lambda s: s.map(_split_sort_key) if s.name == "split" else s)
    )
    x = np.arange(len(split_order))
    width = 0.33
    scheme_order = ["hierarchical-pooled", "per-specimen"]
    scheme_labels = {"per-specimen": "Per-specimen", "hierarchical-pooled": "Pooled"}
    scheme_colours = {"per-specimen": PALETTE["hybrid"], "hierarchical-pooled": PALETTE["fok"]}
    for idx, scheme in enumerate(scheme_order):
        values = (
            coverage_df.loc[coverage_df["scheme"] == scheme]
            .set_index("split")
            .loc[split_order]
        )
        xpos = x + (idx - 0.5) * width
        heights = values["mean_coverage"].to_numpy(dtype=float)
        widths_text = values["mean_width"].to_numpy(dtype=float)
        ax.bar(xpos, heights, width=width, color=scheme_colours[scheme], alpha=0.88, label=scheme_labels[scheme])
        for xp, h, w_text in zip(xpos, heights, widths_text, strict=True):
            ax.text(xp, h + 0.02, f"w={w_text:.1f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.90, color=PALETTE["boundary"], linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(split_order)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Train/test split")
    ax.set_ylabel("Mean 90% coverage")
    ax.set_title("90% coverage and width")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save_figure(fig, out_path)


def plot_threshold_forest(table: pd.DataFrame, out_path: str | Path) -> Path:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ordered = table.sort_values("T_0p80_median_h", ascending=True).reset_index(drop=True)
    y = np.arange(ordered.shape[0])
    lower = ordered["T_0p80_median_h"] - ordered["T_0p80_q05_h"]
    upper = ordered["T_0p80_q95_h"] - ordered["T_0p80_median_h"]
    ax.errorbar(
        ordered["T_0p80_median_h"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=PALETTE["fok"],
        ecolor=PALETTE["hybrid"],
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(560.0, color=PALETTE["boundary"], linestyle="--", linewidth=1.0)
    xmin = float(ordered["T_0p80_q05_h"].min()) * 0.8
    xmax = float(ordered["T_0p80_q95_h"].max()) * 1.12
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Specimen"])
    ax.invert_yaxis()
    tick_candidates = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000], dtype=float)
    visible_ticks = tick_candidates[(tick_candidates >= xmin) & (tick_candidates <= xmax)]
    if visible_ticks.size:
        ax.set_xticks(visible_ticks)
        ax.set_xticklabels([f"{int(t):,}" for t in visible_ticks])
    ax.set_xlabel(r"$T_{0.80}$ (h)")
    ax.set_ylabel("Specimen")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    return _save_figure(fig, out_path)


def plot_residual_bias_trend(point_df: pd.DataFrame, residual_summary: pd.DataFrame, out_path: str | Path, *, split: str = "60/40") -> Path:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    group = point_df.loc[point_df["split"] == split].copy()
    lead_summary = (
        group.groupby("lead_time", as_index=False)
        .agg(
            mean=("signed_residual", "mean"),
            q10=("signed_residual", lambda s: np.quantile(s, 0.1)),
            q90=("signed_residual", lambda s: np.quantile(s, 0.9)),
        )
        .reset_index(drop=True)
    )
    ax.plot(lead_summary["lead_time"], lead_summary["mean"], color=PALETTE["hybrid"], linewidth=1.5, marker="o")
    ax.fill_between(lead_summary["lead_time"], lead_summary["q10"], lead_summary["q90"], color=PALETTE["total"], alpha=0.8)
    slope = float(residual_summary.loc[residual_summary["split"] == split, "lead_slope"].iloc[0])
    intercept = float(residual_summary.loc[residual_summary["split"] == split, "lead_intercept"].iloc[0])
    x = lead_summary["lead_time"].to_numpy(dtype=float)
    ax.plot(x, intercept + slope * x, color=PALETTE["ink"], linestyle="--", linewidth=1.1)
    ax.axhline(0.0, color=PALETTE["boundary"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Observed - predicted retention")
    return _save_figure(fig, out_path)


def save_figure_bundle(
    *,
    results_df: pd.DataFrame,
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    alpha_pairs: pd.DataFrame,
    alpha_summary: dict[str, float],
    mechanistic_rmse: pd.DataFrame,
    reliability_curve: pd.DataFrame,
    reliability_mace: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    threshold_table: pd.DataFrame,
    residual_points: pd.DataFrame,
    residual_summary: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fok_vs_hybrid_split_metrics": plot_fok_vs_hybrid_split_metrics(results_df, out_dir / "ress_fok_vs_hybrid_split_metrics.png"),
        "fok_vs_hybrid_gain_heatmap": plot_fok_vs_hybrid_gain_heatmap(results_df, out_dir / "ress_hybrid_rmse_gain_heatmap.png"),
        "forecast_panels": plot_reduced_forecast_panels(case_details, out_dir / "ress_reduced_forecast_panels.png"),
        "decomposition": plot_decomposition_panels(case_details, out_dir / "ress_decomposition_panels.png"),
        "calibration_overview": plot_calibration_overview(
            reliability_curve,
            reliability_mace,
            hierarchical_summary,
            out_dir / "ress_calibration_overview.png",
        ),
        "coverage_diagnostics": plot_coverage_diagnostics(results_df, out_dir / "capdata3_coverage_interval_diagnostics.png"),
        "gain_vs_alpha": plot_gain_vs_alpha(alpha_pairs, alpha_summary, out_dir / "ress_fok_kww_gain_vs_alpha.png"),
        "mechanistic_rmse": plot_mechanistic_rmse_bars(mechanistic_rmse, out_dir / "ress_mechanistic_rmse_with_ci.png"),
        "reliability": plot_reliability_diagram(reliability_curve, reliability_mace, out_dir / "ress_reliability_diagram.png"),
        "threshold_forest": plot_threshold_forest(threshold_table, out_dir / "ress_threshold_forest.png"),
        "residual_bias": plot_residual_bias_trend(residual_points, residual_summary, out_dir / "ress_residual_bias_trend.png"),
    }
    outputs.update(plot_full_forecast_panel_groups(case_details, out_dir))
    return outputs

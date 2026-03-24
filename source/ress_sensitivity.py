"""Time-resolved Sobol and Morris sensitivity analysis for the RESS revision."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fractional_model import FKParams, time_to_threshold
from math_utils import mittag_leffler
from ress_batch_runner import DEFAULT_OUTPUT_DIR, DEFAULT_THRESHOLD, FOKCaseArtifacts


MANUSCRIPT_FIGURE_DIR = REPO_ROOT / "output" / "figures"
PARAMETER_ORDER = ("k", "alpha", "f_inf")
PARAMETER_LABELS = {
    "k": r"Rate $k$",
    "alpha": r"Order $\alpha$",
    "f_inf": r"Plateau $f_\infty$",
}
PARAMETER_COLOURS = {
    "k": "#2C7FB8",
    "alpha": "#138A61",
    "f_inf": "#C8572A",
}


@dataclass(frozen=True)
class LocalSensitivityBox:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    base_params: FKParams


def _split_sort_key(label: str) -> tuple[int, int]:
    left, right = label.split("/")
    return int(left), int(right)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D7DEE5",
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
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if copy_to_manuscript:
        MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, MANUSCRIPT_FIGURE_DIR / path.name)
    return path


def _nanpercentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _nanmedian(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _bounded_relative_interval(
    value: float,
    rho: float,
    *,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    lower = max(lower_bound, value * (1.0 - rho))
    upper = min(upper_bound, value * (1.0 + rho))
    if upper <= lower:
        width = max(abs(value) * rho, 1e-4)
        lower = max(lower_bound, value - width)
        upper = min(upper_bound, value + width)
    if upper <= lower:
        upper = min(upper_bound, lower + 1e-4)
        lower = max(lower_bound, upper - 1e-4)
    return float(lower), float(upper)


def _build_local_box(
    case: FOKCaseArtifacts,
    *,
    rho: float,
    threshold_cap: float | None = None,
) -> LocalSensitivityBox:
    theta = case.fit.params
    bounds: list[tuple[float, float]] = []
    for name in PARAMETER_ORDER:
        value = float(getattr(theta, name))
        if name == "k":
            lower, upper = _bounded_relative_interval(
                value,
                rho,
                lower_bound=1e-10,
                upper_bound=10.0,
            )
        elif name == "alpha":
            lower, upper = _bounded_relative_interval(
                value,
                rho,
                lower_bound=0.01,
                upper_bound=0.995,
            )
        else:
            upper_limit = 0.995
            if threshold_cap is not None:
                upper_limit = min(upper_limit, float(threshold_cap) - 1e-4)
            lower, upper = _bounded_relative_interval(
                value,
                rho,
                lower_bound=1e-6,
                upper_bound=upper_limit,
            )
        bounds.append((lower, upper))
    return LocalSensitivityBox(
        names=PARAMETER_ORDER,
        lower=np.array([item[0] for item in bounds], dtype=float),
        upper=np.array([item[1] for item in bounds], dtype=float),
        base_params=theta,
    )


def _map_to_bounds(unit_matrix: np.ndarray, box: LocalSensitivityBox) -> np.ndarray:
    return box.lower + np.asarray(unit_matrix, dtype=float) * (box.upper - box.lower)


def _row_to_params(row: np.ndarray, box: LocalSensitivityBox) -> FKParams:
    values = {
        "C0": float(box.base_params.C0),
        "k": float(box.base_params.k),
        "alpha": float(box.base_params.alpha),
        "f_inf": float(box.base_params.f_inf),
    }
    for idx, name in enumerate(box.names):
        values[name] = float(row[idx])
    return FKParams(**values)


def _evaluate_design(
    unit_matrix: np.ndarray,
    box: LocalSensitivityBox,
    evaluator: Callable[[FKParams], np.ndarray | float],
) -> np.ndarray:
    parameter_matrix = _map_to_bounds(unit_matrix, box)
    outputs: list[np.ndarray] = []
    output_size: int | None = None
    for row in parameter_matrix:
        theta = _row_to_params(row, box)
        try:
            values = np.atleast_1d(np.asarray(evaluator(theta), dtype=float))
        except Exception:
            if output_size is None:
                raise
            values = np.full(output_size, np.nan, dtype=float)
        if output_size is None:
            output_size = int(values.size)
        outputs.append(values.reshape(-1))
    return np.vstack(outputs)


def _normalised_response(theta: FKParams, time_grid: np.ndarray) -> np.ndarray:
    z = -theta.k * np.power(time_grid, theta.alpha)
    ml = mittag_leffler(theta.alpha, z)
    return theta.f_inf + (1.0 - theta.f_inf) * ml


def _threshold_response(theta: FKParams, threshold: float) -> float:
    return float(time_to_threshold(theta, threshold))


def _bootstrap_interval(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        raise ValueError("bootstrap samples must not be empty")
    stacked = np.stack(samples, axis=0)
    lower = np.full(stacked.shape[1:], np.nan, dtype=float)
    upper = np.full(stacked.shape[1:], np.nan, dtype=float)
    for idx in np.ndindex(stacked.shape[1:]):
        values = stacked[(slice(None),) + idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        lower[idx] = float(np.percentile(values, 2.5))
        upper[idx] = float(np.percentile(values, 97.5))
    return np.stack([lower, upper], axis=0)


def _sobol_jansen(
    box: LocalSensitivityBox,
    evaluator: Callable[[FKParams], np.ndarray | float],
    *,
    n_samples: int,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_dim = len(box.names)
    A = rng.random((int(n_samples), n_dim))
    B = rng.random((int(n_samples), n_dim))
    y_a = _evaluate_design(A, box, evaluator)
    y_b = _evaluate_design(B, box, evaluator)

    y_ab: list[np.ndarray] = []
    for idx in range(n_dim):
        ab = A.copy()
        ab[:, idx] = B[:, idx]
        y_ab.append(_evaluate_design(ab, box, evaluator))

    finite_rows = np.isfinite(y_a).all(axis=1) & np.isfinite(y_b).all(axis=1)
    for arr in y_ab:
        finite_rows &= np.isfinite(arr).all(axis=1)
    y_a = y_a[finite_rows]
    y_b = y_b[finite_rows]
    y_ab = [arr[finite_rows] for arr in y_ab]
    if y_a.shape[0] < max(16, int(n_samples) // 20):
        raise RuntimeError("Too few finite samples available for Sobol analysis.")

    variance = np.var(np.vstack([y_a, y_b]), axis=0, ddof=1)
    valid_variance = variance > 1e-12
    output_size = y_a.shape[1]
    first = np.full((output_size, n_dim), np.nan, dtype=float)
    total = np.full((output_size, n_dim), np.nan, dtype=float)

    for idx, arr in enumerate(y_ab):
        if np.any(valid_variance):
            numerator_first = 0.5 * np.mean((y_b - arr) ** 2, axis=0)
            numerator_total = 0.5 * np.mean((y_a - arr) ** 2, axis=0)
            first[valid_variance, idx] = 1.0 - numerator_first[valid_variance] / variance[valid_variance]
            total[valid_variance, idx] = numerator_total[valid_variance] / variance[valid_variance]

    if n_bootstrap <= 0:
        first_ci = np.full((2, output_size, n_dim), np.nan, dtype=float)
        total_ci = np.full((2, output_size, n_dim), np.nan, dtype=float)
    else:
        first_samples: list[np.ndarray] = []
        total_samples: list[np.ndarray] = []
        for _ in range(int(n_bootstrap)):
            sample_idx = rng.integers(0, y_a.shape[0], size=y_a.shape[0])
            y_a_b = y_a[sample_idx]
            y_b_b = y_b[sample_idx]
            y_ab_b = [arr[sample_idx] for arr in y_ab]
            variance_b = np.var(np.vstack([y_a_b, y_b_b]), axis=0, ddof=1)
            valid_b = variance_b > 1e-12
            first_b = np.full((output_size, n_dim), np.nan, dtype=float)
            total_b = np.full((output_size, n_dim), np.nan, dtype=float)
            for idx, arr_b in enumerate(y_ab_b):
                if np.any(valid_b):
                    num_first_b = 0.5 * np.mean((y_b_b - arr_b) ** 2, axis=0)
                    num_total_b = 0.5 * np.mean((y_a_b - arr_b) ** 2, axis=0)
                    first_b[valid_b, idx] = 1.0 - num_first_b[valid_b] / variance_b[valid_b]
                    total_b[valid_b, idx] = num_total_b[valid_b] / variance_b[valid_b]
            first_samples.append(first_b)
            total_samples.append(total_b)
        first_ci = _bootstrap_interval(first_samples)
        total_ci = _bootstrap_interval(total_samples)

    return {
        "first": first,
        "total": total,
        "first_ci": first_ci,
        "total_ci": total_ci,
    }


def _morris_screen(
    box: LocalSensitivityBox,
    evaluator: Callable[[FKParams], np.ndarray | float],
    *,
    n_trajectories: int,
    levels: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    if levels < 3:
        raise ValueError("Morris screening requires at least 3 grid levels.")

    rng = np.random.default_rng(random_state)
    n_dim = len(box.names)
    delta = levels / (2.0 * (levels - 1.0))
    grid = np.linspace(0.0, 1.0, int(levels))

    trajectories: list[np.ndarray] = []
    for _ in range(int(n_trajectories)):
        current = rng.choice(grid, size=n_dim, replace=True).astype(float)
        points = [current.copy()]
        changes: list[tuple[int, float]] = []
        for param_idx in rng.permutation(n_dim):
            candidates: list[float] = []
            if current[param_idx] + delta <= 1.0 + 1e-12:
                candidates.append(delta)
            if current[param_idx] - delta >= -1e-12:
                candidates.append(-delta)
            if not candidates:
                continue
            step = float(candidates[int(rng.integers(0, len(candidates)))])
            next_point = current.copy()
            next_point[param_idx] = np.clip(current[param_idx] + step, 0.0, 1.0)
            actual_step = float(next_point[param_idx] - current[param_idx])
            if actual_step == 0.0:
                continue
            points.append(next_point.copy())
            changes.append((param_idx, actual_step))
            current = next_point
        if len(points) <= 1:
            continue
        responses = _evaluate_design(np.vstack(points), box, evaluator)
        effects = np.full((n_dim, responses.shape[1]), np.nan, dtype=float)
        for step_idx, (param_idx, actual_step) in enumerate(changes, start=1):
            effects[param_idx] = (responses[step_idx] - responses[step_idx - 1]) / actual_step
        trajectories.append(effects)

    if not trajectories:
        raise RuntimeError("No valid Morris trajectories were generated.")

    ee = np.stack(trajectories, axis=0)
    return {
        "mu_star": np.nanmean(np.abs(ee), axis=0).T,
        "sigma": np.nanstd(ee, axis=0, ddof=1).T,
    }


def _summarize_frame(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(group_columns, keys))
        values = group["value"].to_numpy(dtype=float)
        row.update(
            {
                "mean": _nanmean(values),
                "median": _nanmedian(values),
                "q25": _nanpercentile(values, 25.0),
                "q75": _nanpercentile(values, 75.0),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _build_summary_payload(
    time_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    split_boundaries: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "threshold": float(threshold),
        "split_boundaries_h": {
            row["split"]: float(row["forecast_boundary_h"]) for _, row in split_boundaries.iterrows()
        },
        "splits": {},
    }
    grid_times = np.unique(time_summary["time_h"].to_numpy(dtype=float))
    end_time = float(grid_times[-1])
    midpoint_target = 0.5 * end_time
    midpoint_time = float(grid_times[np.argmin(np.abs(grid_times - midpoint_target))])
    for split in sorted(time_summary["split"].unique().tolist(), key=_split_sort_key):
        split_payload: dict[str, object] = {}
        for target_time, label in ((midpoint_time, "mid"), (end_time, "late")):
            subset = time_summary.loc[
                (time_summary["split"] == split)
                & (time_summary["method"] == "sobol")
                & (time_summary["index_type"] == "total")
                & np.isclose(time_summary["time_h"], target_time)
            ].sort_values("median", ascending=False)
            split_payload[f"{label}_horizon_sobol_total"] = subset.loc[:, ["parameter", "median"]].to_dict(orient="records")
        threshold_subset = threshold_summary.loc[
            (threshold_summary["split"] == split)
            & (threshold_summary["method"] == "sobol")
            & (threshold_summary["index_type"] == "total")
        ].sort_values("median", ascending=False)
        split_payload["threshold_sobol_total"] = threshold_subset.loc[:, ["parameter", "median"]].to_dict(orient="records")
        payload["splits"][split] = split_payload
    return payload


def _plot_time_resolved_sensitivity(
    time_summary: pd.DataFrame,
    split_boundaries: pd.DataFrame,
    out_path: str | Path,
    *,
    copy_to_manuscript: bool,
) -> Path:
    _configure_matplotlib()
    splits = sorted(time_summary["split"].unique().tolist(), key=_split_sort_key)
    panels = [
        ("sobol", "first", "Sobol First-Order"),
        ("sobol", "total", "Sobol Total-Effect"),
        ("morris", "mu_star", r"Morris $\mu^\star$"),
    ]
    fig, axes = plt.subplots(len(splits), len(panels), figsize=(12.4, 8.2), sharex=True)
    if len(splits) == 1:
        axes = np.array([axes])

    for row_idx, split in enumerate(splits):
        boundary = float(
            split_boundaries.loc[split_boundaries["split"] == split, "forecast_boundary_h"].iloc[0]
        )
        for col_idx, (method, index_type, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            subset = time_summary.loc[
                (time_summary["split"] == split)
                & (time_summary["method"] == method)
                & (time_summary["index_type"] == index_type)
            ].sort_values("time_h")
            for parameter in PARAMETER_ORDER:
                param_frame = subset.loc[subset["parameter"] == parameter].sort_values("time_h")
                if param_frame.empty:
                    continue
                x = param_frame["time_h"].to_numpy(dtype=float)
                median = param_frame["median"].to_numpy(dtype=float)
                q25 = param_frame["q25"].to_numpy(dtype=float)
                q75 = param_frame["q75"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    median,
                    color=PARAMETER_COLOURS[parameter],
                    linewidth=1.8,
                    label=PARAMETER_LABELS[parameter] if row_idx == 0 and col_idx == 0 else None,
                )
                ax.fill_between(
                    x,
                    q25,
                    q75,
                    color=PARAMETER_COLOURS[parameter],
                    alpha=0.12,
                    linewidth=0.0,
                )
            ax.axvline(boundary, color="#B23A48", linestyle="--", linewidth=1.0)
            if method == "sobol":
                ax.set_ylim(-0.05, 1.05)
            ax.set_title(title if row_idx == 0 else "")
            if col_idx == 0:
                ax.set_ylabel(f"{split}\nIndex Value")
            if row_idx == len(splits) - 1:
                ax.set_xlabel("Ageing time (h)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    return _save_figure(fig, out_path, copy_to_manuscript=copy_to_manuscript)


def _plot_threshold_sensitivity(
    threshold_summary: pd.DataFrame,
    out_path: str | Path,
    *,
    copy_to_manuscript: bool,
    threshold: float,
) -> Path:
    _configure_matplotlib()
    splits = sorted(threshold_summary["split"].unique().tolist(), key=_split_sort_key)
    panels = [
        ("sobol", "total", rf"Total Sobol on $T_{{{threshold:.2f}}}$"),
        ("morris", "mu_star", rf"Morris $\mu^\star$ on $T_{{{threshold:.2f}}}$"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(10.6, 3.8), sharey=False)
    if len(panels) == 1:
        axes = np.array([axes])
    x = np.arange(len(splits), dtype=float)
    width = 0.22
    for ax, (method, index_type, title) in zip(axes, panels):
        subset = threshold_summary.loc[
            (threshold_summary["method"] == method)
            & (threshold_summary["index_type"] == index_type)
        ]
        for param_idx, parameter in enumerate(PARAMETER_ORDER):
            param_frame = (
                subset.loc[subset["parameter"] == parameter]
                .set_index("split")
                .reindex(splits)
                .reset_index()
            )
            median = param_frame["median"].to_numpy(dtype=float)
            q25 = param_frame["q25"].to_numpy(dtype=float)
            q75 = param_frame["q75"].to_numpy(dtype=float)
            ax.bar(
                x + (param_idx - 1) * width,
                median,
                width=width,
                color=PARAMETER_COLOURS[parameter],
                alpha=0.85,
                label=PARAMETER_LABELS[parameter] if ax is axes[0] else None,
                yerr=np.vstack([median - q25, q75 - median]),
                error_kw={"elinewidth": 0.8, "capsize": 2},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_title(title)
        ax.set_xlabel("Forward split")
        ax.set_ylabel("Index Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    return _save_figure(fig, out_path, copy_to_manuscript=copy_to_manuscript)


def run_sensitivity_analysis(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    time_grid_size: int = 50,
    rho: float = 0.20,
    sobol_samples: int = 512,
    sobol_bootstrap: int = 200,
    morris_trajectories: int = 20,
    morris_levels: int = 4,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = 42,
) -> dict[str, object]:
    if not case_details:
        raise ValueError("case_details must not be empty.")

    time_max = max(float(case.time[-1]) for case in case_details.values())
    time_grid = np.linspace(0.0, time_max, int(time_grid_size), dtype=float)
    split_boundaries = pd.DataFrame.from_records(
        [
            {
                "split": split,
                "forecast_boundary_h": _nanmedian(
                    [
                        float(case.time[int(case.splits["train_idx"])])
                        for case in case_details.values()
                        if case.split == split
                    ]
                ),
            }
            for split in sorted({case.split for case in case_details.values()}, key=_split_sort_key)
        ]
    )

    time_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    ordered_cases = sorted(case_details.items(), key=lambda item: (_split_sort_key(item[0][1]), item[0][0]))

    for case_idx, ((specimen, split), case) in enumerate(ordered_cases):
        response_box = _build_local_box(case, rho=rho, threshold_cap=None)
        threshold_box = _build_local_box(case, rho=rho, threshold_cap=threshold)
        seed_base = int(random_state + case_idx * 1000)

        sobol_time = _sobol_jansen(
            response_box,
            lambda theta, grid=time_grid: _normalised_response(theta, grid),
            n_samples=sobol_samples,
            n_bootstrap=sobol_bootstrap,
            random_state=seed_base,
        )
        morris_time = _morris_screen(
            response_box,
            lambda theta, grid=time_grid: _normalised_response(theta, grid),
            n_trajectories=morris_trajectories,
            levels=morris_levels,
            random_state=seed_base + 1,
        )
        sobol_threshold = _sobol_jansen(
            threshold_box,
            lambda theta, q=threshold: _threshold_response(theta, q),
            n_samples=sobol_samples,
            n_bootstrap=sobol_bootstrap,
            random_state=seed_base + 2,
        )
        morris_threshold = _morris_screen(
            threshold_box,
            lambda theta, q=threshold: _threshold_response(theta, q),
            n_trajectories=morris_trajectories,
            levels=morris_levels,
            random_state=seed_base + 3,
        )

        for time_idx, time_h in enumerate(time_grid):
            for param_idx, parameter in enumerate(PARAMETER_ORDER):
                time_rows.extend(
                    [
                        {
                            "split": split,
                            "specimen": specimen,
                            "qoi": "normalized_retention",
                            "time_h": float(time_h),
                            "parameter": parameter,
                            "method": "sobol",
                            "index_type": "first",
                            "value": float(sobol_time["first"][time_idx, param_idx]),
                            "ci_low": float(sobol_time["first_ci"][0, time_idx, param_idx]),
                            "ci_high": float(sobol_time["first_ci"][1, time_idx, param_idx]),
                        },
                        {
                            "split": split,
                            "specimen": specimen,
                            "qoi": "normalized_retention",
                            "time_h": float(time_h),
                            "parameter": parameter,
                            "method": "sobol",
                            "index_type": "total",
                            "value": float(sobol_time["total"][time_idx, param_idx]),
                            "ci_low": float(sobol_time["total_ci"][0, time_idx, param_idx]),
                            "ci_high": float(sobol_time["total_ci"][1, time_idx, param_idx]),
                        },
                        {
                            "split": split,
                            "specimen": specimen,
                            "qoi": "normalized_retention",
                            "time_h": float(time_h),
                            "parameter": parameter,
                            "method": "morris",
                            "index_type": "mu_star",
                            "value": float(morris_time["mu_star"][time_idx, param_idx]),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        },
                        {
                            "split": split,
                            "specimen": specimen,
                            "qoi": "normalized_retention",
                            "time_h": float(time_h),
                            "parameter": parameter,
                            "method": "morris",
                            "index_type": "sigma",
                            "value": float(morris_time["sigma"][time_idx, param_idx]),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        },
                    ]
                )

        for param_idx, parameter in enumerate(PARAMETER_ORDER):
            threshold_rows.extend(
                [
                    {
                        "split": split,
                        "specimen": specimen,
                        "qoi": f"T_{threshold:.2f}",
                        "parameter": parameter,
                        "method": "sobol",
                        "index_type": "first",
                        "value": float(sobol_threshold["first"][0, param_idx]),
                        "ci_low": float(sobol_threshold["first_ci"][0, 0, param_idx]),
                        "ci_high": float(sobol_threshold["first_ci"][1, 0, param_idx]),
                    },
                    {
                        "split": split,
                        "specimen": specimen,
                        "qoi": f"T_{threshold:.2f}",
                        "parameter": parameter,
                        "method": "sobol",
                        "index_type": "total",
                        "value": float(sobol_threshold["total"][0, param_idx]),
                        "ci_low": float(sobol_threshold["total_ci"][0, 0, param_idx]),
                        "ci_high": float(sobol_threshold["total_ci"][1, 0, param_idx]),
                    },
                    {
                        "split": split,
                        "specimen": specimen,
                        "qoi": f"T_{threshold:.2f}",
                        "parameter": parameter,
                        "method": "morris",
                        "index_type": "mu_star",
                        "value": float(morris_threshold["mu_star"][0, param_idx]),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                    },
                    {
                        "split": split,
                        "specimen": specimen,
                        "qoi": f"T_{threshold:.2f}",
                        "parameter": parameter,
                        "method": "morris",
                        "index_type": "sigma",
                        "value": float(morris_threshold["sigma"][0, param_idx]),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                    },
                ]
            )

    time_case = pd.DataFrame.from_records(time_rows)
    threshold_case = pd.DataFrame.from_records(threshold_rows)
    time_summary = _summarize_frame(
        time_case,
        ["split", "qoi", "time_h", "parameter", "method", "index_type"],
    )
    threshold_summary = _summarize_frame(
        threshold_case,
        ["split", "qoi", "parameter", "method", "index_type"],
    )
    summary_payload = _build_summary_payload(
        time_summary,
        threshold_summary,
        split_boundaries,
        threshold=threshold,
    )
    return {
        "time_grid": time_grid,
        "split_boundaries": split_boundaries,
        "time_case": time_case,
        "time_summary": time_summary,
        "threshold_case": threshold_case,
        "threshold_summary": threshold_summary,
        "summary": summary_payload,
    }


def save_sensitivity_outputs(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    time_grid_size: int = 50,
    rho: float = 0.20,
    sobol_samples: int = 512,
    sobol_bootstrap: int = 200,
    morris_trajectories: int = 20,
    morris_levels: int = 4,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = 42,
    copy_to_manuscript: bool = True,
) -> dict[str, Path]:
    analysis = run_sensitivity_analysis(
        case_details,
        time_grid_size=time_grid_size,
        rho=rho,
        sobol_samples=sobol_samples,
        sobol_bootstrap=sobol_bootstrap,
        morris_trajectories=morris_trajectories,
        morris_levels=morris_levels,
        threshold=threshold,
        random_state=random_state,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_case_path = out_dir / "ress_sensitivity_time_case.csv"
    time_summary_path = out_dir / "ress_sensitivity_time_summary.csv"
    threshold_case_path = out_dir / "ress_sensitivity_threshold_case.csv"
    threshold_summary_path = out_dir / "ress_sensitivity_threshold_summary.csv"
    boundary_path = out_dir / "ress_sensitivity_split_boundaries.csv"
    summary_path = out_dir / "ress_sensitivity_summary.json"
    time_figure_path = out_dir / "ress_sensitivity_time_resolved.png"
    threshold_figure_path = out_dir / "ress_sensitivity_threshold.png"
    manifest_path = out_dir / "ress_sensitivity_manifest.json"

    analysis["time_case"].to_csv(time_case_path, index=False)
    analysis["time_summary"].to_csv(time_summary_path, index=False)
    analysis["threshold_case"].to_csv(threshold_case_path, index=False)
    analysis["threshold_summary"].to_csv(threshold_summary_path, index=False)
    analysis["split_boundaries"].to_csv(boundary_path, index=False)
    summary_path.write_text(json.dumps(analysis["summary"], indent=2), encoding="utf-8")

    _plot_time_resolved_sensitivity(
        analysis["time_summary"],
        analysis["split_boundaries"],
        time_figure_path,
        copy_to_manuscript=copy_to_manuscript,
    )
    _plot_threshold_sensitivity(
        analysis["threshold_summary"],
        threshold_figure_path,
        copy_to_manuscript=copy_to_manuscript,
        threshold=threshold,
    )

    manifest = {
        "time_grid_size": int(time_grid_size),
        "rho": float(rho),
        "sobol_samples": int(sobol_samples),
        "sobol_bootstrap": int(sobol_bootstrap),
        "morris_trajectories": int(morris_trajectories),
        "morris_levels": int(morris_levels),
        "threshold": float(threshold),
        "splits": sorted(analysis["split_boundaries"]["split"].tolist(), key=_split_sort_key),
        "files": {
            "time_case": str(time_case_path),
            "time_summary": str(time_summary_path),
            "threshold_case": str(threshold_case_path),
            "threshold_summary": str(threshold_summary_path),
            "split_boundaries": str(boundary_path),
            "summary": str(summary_path),
            "time_figure": str(time_figure_path),
            "threshold_figure": str(threshold_figure_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "time_case": time_case_path,
        "time_summary": time_summary_path,
        "threshold_case": threshold_case_path,
        "threshold_summary": threshold_summary_path,
        "split_boundaries": boundary_path,
        "summary": summary_path,
        "time_figure": time_figure_path,
        "threshold_figure": threshold_figure_path,
        "manifest": manifest_path,
    }


__all__ = ["run_sensitivity_analysis", "save_sensitivity_outputs"]

"""Epistemic-versus-aleatoric decomposition summaries for the RESS revision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ress_batch_runner import DEFAULT_OUTPUT_DIR, FOKCaseArtifacts


def epistemic_aleatoric_width_ratio(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    split: str = "60/40",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (specimen, case_split), case in sorted(case_details.items()):
        if case_split != split:
            continue
        train_idx = int(case.splits["train_idx"])
        forecast_count = int(np.asarray(case.splits["forecast_t"]).size)
        if forecast_count == 0:
            continue
        midpoint_idx = train_idx + forecast_count // 2
        endpoint_idx = train_idx + forecast_count - 1

        for label, idx in (("midpoint", midpoint_idx), ("endpoint", endpoint_idx)):
            mu_band = np.quantile(case.predictive.mu_samples[:, idx], [0.05, 0.95])
            total_band = np.quantile(case.predictive.total_samples[:, idx], [0.05, 0.95])
            epistemic_width = float(mu_band[1] - mu_band[0])
            total_width = float(total_band[1] - total_band[0])
            rows.append(
                {
                    "specimen": specimen,
                    "split": split,
                    "location": label,
                    "time": float(case.time[idx]),
                    "epistemic_width": epistemic_width,
                    "total_width": total_width,
                    "ratio": float(epistemic_width / total_width) if total_width > 0 else float("nan"),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values(["specimen", "location"]).reset_index(drop=True)


def save_decomposition_outputs(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "60/40",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = epistemic_aleatoric_width_ratio(case_details, split=split)
    ratio_path = out_dir / f"ress_epistemic_aleatoric_ratio_{split.replace('/', '')}.csv"
    ratios.to_csv(ratio_path, index=False)
    summary = (
        ratios.groupby(["split", "location"], as_index=False)
        .agg(
            mean_ratio=("ratio", "mean"),
            median_ratio=("ratio", "median"),
            min_ratio=("ratio", "min"),
            max_ratio=("ratio", "max"),
        )
        .reset_index(drop=True)
    )
    summary_path = out_dir / f"ress_epistemic_aleatoric_ratio_summary_{split.replace('/', '')}.csv"
    summary.to_csv(summary_path, index=False)
    return {"ratios": ratio_path, "summary": summary_path}

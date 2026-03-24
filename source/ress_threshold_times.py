"""Threshold-crossing time summaries for the RESS revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ress_batch_runner import DEFAULT_OUTPUT_DIR, DEFAULT_THRESHOLD, FOKCaseArtifacts


def threshold_time_table(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    split: str = "70/30",
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (specimen, case_split), case in sorted(case_details.items()):
        if case_split != split:
            continue
        samples = np.asarray(case.failure_samples[threshold], dtype=float)
        finite = samples[np.isfinite(samples)]
        terminal_retention = float(case.retention[-1])
        rows.append(
            {
                "Specimen": specimen,
                "Terminal_Retention_pct": terminal_retention,
                "T_0p80_q05_h": float(np.nanpercentile(finite, 5)),
                "T_0p80_median_h": float(np.nanpercentile(finite, 50)),
                "T_0p80_q95_h": float(np.nanpercentile(finite, 95)),
                "P_cross_by_560h": float(np.mean(finite <= 560.0)),
            }
        )
    frame = pd.DataFrame.from_records(rows).sort_values("T_0p80_median_h").reset_index(drop=True)
    frame["Rank"] = np.arange(1, frame.shape[0] + 1)
    return frame


def ranking_validation(table: pd.DataFrame) -> dict[str, float]:
    t_rank = table["T_0p80_median_h"].rank(method="average", ascending=True)
    terminal_rank = table["Terminal_Retention_pct"].rank(method="average", ascending=False)
    spearman = stats.spearmanr(t_rank, terminal_rank)
    return {
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def save_threshold_outputs(
    case_details: dict[tuple[str, str], FOKCaseArtifacts],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "70/30",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = threshold_time_table(case_details, split=split)
    table_path = out_dir / f"ress_threshold_times_{split.replace('/', '')}.csv"
    table.to_csv(table_path, index=False)
    ranking = ranking_validation(table)
    ranking_path = out_dir / f"ress_threshold_times_ranking_{split.replace('/', '')}.json"
    ranking_path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    return {"table": table_path, "ranking": ranking_path}

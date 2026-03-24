# FOK REPRODUCIBILITY 

This repository is a  reproducibility package for the fractional-order kinetics (FOK) capacitor prognostics study on the CAPDATA3 dataset. It contains the raw Excel workbook, the full analysis code, precomputed outputs, and a one-command pipeline that regenerates the tables, figures, and manuscript metrics shipped in `output/`.

The full pipeline was re-run from this folder on March 24, 2026 in the current workspace. The regenerated manifests are portable and repo-relative, so the package can be cloned and executed on a different machine without editing hard-coded paths.

## Repository highlights

- `source/data/raw/CAPDATA3.xlsx` is the bundled source dataset with 25 ageing-time points and 8 specimens (`C1` to `C8`).
- `source/ress_run_all.py` is the main entrypoint. It executes the end-to-end analysis and repopulates `output/analysis/`.
- `output/figures/`, `output/tables_csv/`, and `output/metadata/` are reviewer-friendly mirrors of the raw analysis products in `output/analysis/`.
- `.github/workflows/reproducibility.yml` runs the full pipeline on GitHub Actions and checks that the expected artefacts are created.

## Verified headline results

| Result | Evidence |
| --- | --- |
| FOK outperforms the published Hybrid benchmark in pooled RMSE by `5.3753` units | `output/tables_csv/ress_pairwise_statistics.csv` |
| FOK outperforms the Classical baseline in pooled RMSE by `5.1015` units | `output/tables_csv/ress_pairwise_statistics.csv` |
| FOK remains slightly better than KWW on pooled RMSE (`0.2077` mean gain), but the CI crosses zero | `output/tables_csv/ress_pairwise_statistics.csv` |
| The best calibration split is `70/30` with `MACE = 0.1358` | `output/tables_csv/ress_reliability_mace.csv` |
| Threshold ranking places `C6` as highest priority and `C3` as lowest priority at the 80% retention threshold | `output/tables_csv/ress_threshold_times_7030.csv` |
| Epistemic uncertainty dominates total predictive width at the 60/40 split with mean ratios `0.8878` at the forecast midpoint and `0.9551` at the endpoint | `output/tables_csv/ress_epistemic_aleatoric_ratio_summary_6040.csv` |

## Quick start

1. Create and activate a Python environment with Python `3.11+`.
2. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Reproduce all artefacts:

```bash
python source/ress_run_all.py
```

4. Inspect the reviewer-facing outputs:

- Figures: `output/figures/`
- Tables: `output/tables_csv/`
- Metadata and manifests: `output/metadata/`

The local rerun in this workspace completed in about 4.5 minutes.

## Output structure

```text
output/
├── analysis/      # Raw outputs written directly by the analysis scripts
├── figures/       # PNG mirrors for reviewer browsing
├── metadata/      # JSON summaries and manifests
└── tables_csv/    # CSV mirrors for reviewer browsing
```

## Figure preview

### Splitwise performance

![Splitwise performance](output/figures/ress_fok_vs_hybrid_split_metrics.png)

### Reliability

![Reliability diagram](output/figures/ress_reliability_diagram.png)

### Threshold ranking

![Threshold forest](output/figures/ress_threshold_forest.png)

## Important submission notes

- The original code expected external published-comparator files that were not bundled in the folder. This submission now ships the needed reference tables in `source/data/reference/`.
- `output/metadata/ress_run_all_manifest.json` now stores repo-relative paths instead of machine-specific absolute paths.
- The legacy module `source/src/core.py` is not on the critical execution path for this reproducibility package; the reviewer workflow is centered on `source/ress_run_all.py`.

## Documentation

- Detailed run instructions and code/results analysis: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Main workflow file for CI execution: [.github/workflows/reproducibility.yml](.github/workflows/reproducibility.yml)

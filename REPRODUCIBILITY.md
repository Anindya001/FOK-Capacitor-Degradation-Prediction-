# Reproducibility Guide

## Scope

This package is organized to let a reviewer do three things quickly:

1. Re-run the complete computational pipeline from the bundled CAPDATA3 workbook.
2. Inspect every generated figure, table, and metadata file from a clean `output/` tree.
3. Trace each manuscript claim back to the Python module and artefact that produced it.

The primary entrypoint is:

```bash
python source/ress_run_all.py
```

## Verified environment

The package was re-executed successfully in this workspace on March 24, 2026 with:

- Python `3.13.12`
- Windows PowerShell host
- Local wall-clock runtime of roughly `4.5 minutes`

For GitHub and cross-platform use, the repository ships:

- `requirements.txt`
- `.github/workflows/reproducibility.yml`
- Repo-relative manifests under `output/metadata/`

## How to run

### Fresh environment

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### End-to-end rerun

```bash
python source/ress_run_all.py
```

### Optional CLI knobs

The driver exposes the main numerical settings:

```bash
python source/ress_run_all.py --help
```

Important flags:

- `--excel-path`: alternate workbook path
- `--output-dir`: alternate analysis directory
- `--n-draws`: posterior predictive draw count
- `--n-bootstrap`: bootstrap size for statistics and RMSE confidence intervals
- `--sa-*`: sensitivity-analysis controls

## What gets regenerated

`source/ress_run_all.py` writes all primary outputs to `output/analysis/`, then mirrors them into reviewer-friendly folders:

- `output/figures/`: PNGs only
- `output/tables_csv/`: CSVs only
- `output/metadata/`: JSON summaries and manifests

The most important files are:

| Artefact | Purpose |
| --- | --- |
| `output/tables_csv/ress_batch_results.csv` | Master per-specimen, per-split, per-model result table |
| `output/tables_csv/ress_pairwise_statistics.csv` | Pooled FOK vs Hybrid/Classical/KWW comparisons |
| `output/tables_csv/ress_compact_summary_table.csv` | Manuscript-style splitwise summary table |
| `output/tables_csv/ress_full_comparison_6040.csv` | 60/40 unit-level comparison table with published comparators and FOK |
| `output/tables_csv/ress_threshold_times_7030.csv` | 80% retention threshold-crossing ranking |
| `output/tables_csv/ress_reliability_mace.csv` | Calibration summary by split |
| `output/metadata/ress_manuscript_metrics.json` | Small machine-readable bundle of manuscript-ready headline numbers |
| `output/metadata/ress_run_all_manifest.json` | Portable manifest of the complete package |

## Codebase analysis

### Top-level drivers

- `source/ress_run_all.py`
  - Orchestrates the entire pipeline.
  - Regenerates all tables, figures, and manuscript metrics.
  - Packages outputs into the final reviewer-facing `output/` layout.

- `source/ress_batch_runner.py`
  - Loads `CAPDATA3.xlsx`.
  - Converts loss measurements to retention.
  - Runs the FOK, Classical, and KWW models across `8 specimens x 3 splits = 24` cases.
  - Produces `72` result rows in `ress_batch_results.csv`.

### Analysis modules

- `source/ress_statistical_tests.py`
  - Pooled and splitwise comparisons.
  - Wilcoxon tests, BCa bootstrap intervals, Hodges-Lehmann estimates, Cliff's delta, Bayesian bootstrap superiority probabilities.

- `source/ress_model_selection.py`
  - Mechanistic AIC/BIC tables and splitwise RMSE/MAE bootstrap summaries.

- `source/ress_calibration.py`
  - Reliability curves, MACE summaries, and pooled-vs-local conformal ablation.

- `source/ress_threshold_times.py`
  - Threshold-crossing time quantiles and ranking validation.

- `source/ress_decomposition.py`
  - Epistemic-vs-total predictive interval width ratios.

- `source/ress_sensitivity.py`
  - Time-resolved Sobol and Morris sensitivity analysis.

- `source/ress_figures.py`
  - All manuscript-style figure rendering and export.

### Core modeling stack

- `source/src/fractional_model.py`
  - Fractional-order kinetics model definition and threshold-time utilities.

- `source/src/fractional_estimation.py`
  - FOK parameter estimation.

- `source/src/fractional_uq.py`
  - Laplace draws, posterior predictive sampling, and failure-time sampling.

- `source/src/fractional_diagnostics.py`
  - RMSE, MAE, coverage, interval width, WIS, information criteria, and prequential tools.

- `source/src/surrogate_models.py`
  - Classical and KWW baseline fitting/prediction.

- `source/src/fractional_core.py`
  - Higher-level fractional pipeline utilities used by the batch runner.

## Results analysis

### Performance summary

From `output/tables_csv/ress_compact_summary_table.csv`:

| Model | 50/50 RMSE | 60/40 RMSE | 70/30 RMSE |
| --- | ---: | ---: | ---: |
| FOK | 2.7448 | 2.9192 | 2.4843 |
| KWW | 2.8089 | 3.1943 | 2.7680 |
| Classical | 9.5372 | 7.7046 | 6.2111 |
| Hybrid | 8.9186 | 7.9771 | 7.3786 |

Interpretation:

- FOK is the best forecasting model in the bundled results table across all three train/test splits.
- KWW is competitive, but FOK remains better on mean forecast RMSE and MAE at every split.
- Classical and published Hybrid comparators are substantially worse in pure forecast accuracy.

### Pairwise evidence

From `output/tables_csv/ress_pairwise_statistics.csv`:

| Comparison | Mean RMSE gain | 95% BCa CI | Exact p-value |
| --- | ---: | --- | ---: |
| FOK vs Hybrid | 5.3753 | [4.6891, 6.1367] | 5.96e-08 |
| FOK vs Classical | 5.1015 | [4.4380, 5.7767] | 5.96e-08 |
| FOK vs KWW | 0.2077 | [-0.2689, 0.8328] | 4.28e-01 |

Interpretation:

- The strongest claims in the package are FOK vs Hybrid and FOK vs Classical.
- FOK vs KWW is directionally favorable to FOK but not decisive under the packaged pooled comparison.
- `output/metadata/ress_classical_splitwise_summary.json` confirms that every splitwise FOK-vs-Classical comparison is a clean `24/24` pooled win descriptively and `8/8` win within each split.

### Calibration

From `output/tables_csv/ress_reliability_mace.csv` and `output/metadata/ress_manuscript_metrics.json`:

- Best split: `70/30`
- Best MACE: `0.1357638888888889`

From `output/tables_csv/ress_hierarchical_conformal_ablation.csv`:

- The hierarchical pooled conformal scheme improves mean coverage over per-specimen calibration for the 60/40 and 70/30 splits, but with somewhat wider intervals.

### Threshold ranking

From `output/tables_csv/ress_threshold_times_7030.csv`:

- Highest priority specimen: `C6`
  - Median time to 80% retention: `123.11 h`
- Lowest priority specimen: `C3`
  - Median time to 80% retention: `410.43 h`

From `output/metadata/ress_manuscript_metrics.json`:

- Ranking validation Spearman statistic: `-0.8333`
- The sign is negative because the code ranks terminal retention in descending order while threshold time is ranked ascending.

### Uncertainty decomposition

From `output/tables_csv/ress_epistemic_aleatoric_ratio_summary_6040.csv`:

- Mean epistemic/total width ratio at forecast midpoint: `0.8878`
- Mean epistemic/total width ratio at forecast endpoint: `0.9551`

Interpretation:

- Most of the predictive width is epistemic rather than purely observational at the 60/40 split.
- That pattern becomes even stronger near the forecast endpoint.

### Sensitivity analysis

From `output/metadata/ress_sensitivity_summary.json`:

- `alpha` is the dominant parameter across mid-horizon, late-horizon, and threshold-based Sobol totals in all three splits.
- `k` is consistently secondary.
- `f_inf` contributes almost nothing to the total Sobol index under the local analysis box.

### Mechanistic selection vs forecast accuracy

From `output/metadata/ress_mechanistic_aic_bic_wins_6040.json`:

- KWW wins AIC and BIC for all 8 specimens at the 60/40 split.

Interpretation:

- This is an important nuance for reviewers.
- In-sample information criteria favor KWW, while out-of-sample forecast RMSE still favors FOK.
- The package therefore supports a stronger forecasting claim than a universal mechanistic-selection claim.

## Figure map

| Figure file | What it shows |
| --- | --- |
| `output/figures/ress_fok_vs_hybrid_split_metrics.png` | Mean RMSE and MAE by split for FOK vs the published Hybrid benchmark |
| `output/figures/ress_hybrid_rmse_gain_heatmap.png` | Specimen-level RMSE gain of FOK over Hybrid |
| `output/figures/ress_reduced_forecast_panels.png` | Reduced multi-panel forecast visualization across representative cases |
| `output/figures/ress_full_forecast_group1.png` | First group of full forecast panels |
| `output/figures/ress_full_forecast_group2.png` | Second group of full forecast panels |
| `output/figures/ress_decomposition_panels.png` | Epistemic and total uncertainty decomposition panels |
| `output/figures/ress_reliability_diagram.png` | Calibration reliability diagram with error bars |
| `output/figures/ress_calibration_overview.png` | Combined calibration overview and pooled-vs-local conformal comparison |
| `output/figures/ress_threshold_forest.png` | Threshold-crossing interval forest plot |
| `output/figures/ress_sensitivity_time_resolved.png` | Time-resolved sensitivity profiles |
| `output/figures/ress_sensitivity_threshold.png` | Threshold-focused sensitivity summary |
| `output/figures/ress_residual_bias_trend.png` | Residual bias versus forecast lead time |

## Submission hardening performed in this package

The current folder was upgraded to behave like a proper GitHub reproducibility repository:

- Added a top-level `README.md`.
- Added this detailed `REPRODUCIBILITY.md`.
- Added `requirements.txt`.
- Added `.gitignore`.
- Added `.github/workflows/reproducibility.yml`.
- Switched manifests from absolute machine paths to repo-relative paths.
- Bundled the missing published comparator references under `source/data/reference/`.
- Standardized the repository-level `output/` layout used by reruns.

## Caveats reviewers should know

- The specimen-wise published comparator table bundled here is only available for the `60/40` split.
- For `50/50` and `70/30`, the package bundles published splitwise summary values instead of a specimen-wise comparator matrix.
- That is sufficient for reproducing the manuscript summary table, but it should not be misread as full raw comparator availability for every external baseline.
- `source/src/core.py` contains legacy functionality with optional imports that are not required by the main reproducibility path.

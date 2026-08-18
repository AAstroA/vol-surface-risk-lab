# Reproducibility and operating notes

## Environments

- Python 3.10 or newer.
- Dependencies and lower bounds are declared in `pyproject.toml`.
- All stochastic steps use `project.random_seed`.
- Matplotlib uses a non-interactive backend, so the pipeline also runs on WRDS Cloud and CI.

## Commands

```bash
make install
make test
make demo
make wrds
```

Equivalent CLI commands are shown in the main README. Every run writes a `run_summary.json` containing its source, interval, row counts, surface size, PCA coverage, breach counts, and artifact paths.

## Data policy

Raw WRDS downloads are cached below `data/raw/wrds/<ticker>/<start>_<end>/` and ignored by Git. This avoids publishing licensed WRDS data. Processed real-data outputs below `reports/` are also ignored. The committed `sample_outputs/demo/` data are synthetic and safe to share.

The 2005–2021 configuration instead uses `data/raw/wrds/<ticker>/by_year/<year>/`. Finished years are never re-downloaded unless explicitly forced. Surface partitions are keyed by a hash of the option types, filters, and surface configuration, so changing a research choice cannot silently reuse incompatible fitted values.

Credentials live only in `.env`, which is owner-readable and Git-ignored. `.env.example` contains placeholders only.

## Expected artifacts

- `processed/clean_options.parquet`: filtered, normalized option observations.
- `processed/fitted_surface.parquet`: daily fixed-grid IV surfaces.
  The `extrapolated` flag identifies nodes outside the daily quote cloud.
- `tables/surface_regression_diagnostics.csv`: daily fit quality and coefficients.
- `tables/surface_method_comparison.csv`: deterministic held-out errors for every reconstruction family.
- `tables/daily_data_coverage.csv`: compact date-wise coverage used by long-horizon plots.
- `tables/pca_explained_variance.csv`: factor variance and eigenvalues.
- `tables/vega_exposures.csv`: the exact exposure mapping used for P&L.
- `tables/var_backtest.csv`: realized P&L, VaR, ES, and breach indicators.
- `tables/backtest_diagnostics.csv`: exception severity, quantile loss, Kupiec, independence, and joint conditional-coverage tests at every configured level.
- `tables/var_model_ranking.csv`: level-specific coverage, quantile-loss, and combined ranks.
- `tables/diebold_mariano_comparison.csv`: pairwise PSP/PCA/PWG pinball-loss comparisons with Newey–West long-run variance.
- `figures/01`–`07`: coverage, surface, fit, PCA, backtest, and P&L diagnostics.
- `figures/08`–`12`: reconstructed surface, 3D PCA factors, PCA reconstruction/error, method surfaces, and held-out method metrics.
- `figures/13`–`17`: the remaining confidence-level backtests, coverage intervals, UC/IND/CC heat map, and Murphy diagrams.
- `reports/SPX_WRDS_Thesis_Report_2005_2021.pdf`: compiled professional report.
- `reports/latex/`: reproducible LaTeX source and CSV-derived table fragments.

Refresh the multi-level evaluation without re-running WRDS extraction or scenario generation:

```bash
PYTHONPATH=src python scripts/refresh_var_evaluation.py \
  --config configs/long_horizon.yaml
```

## Legacy traceability

Legacy notebooks, benchmark files, figures, MATLAB code, original reports, and the full migration
manifest are retained only in the private local project. They are deliberately excluded from the
public repository because they are not required to reproduce the maintained pipeline and may embed
private paths, licensed data, document metadata, or obsolete credentials. `archive/README.md`
records this boundary without publishing the private artifacts.

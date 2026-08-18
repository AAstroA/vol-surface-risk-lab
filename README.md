# MSc Thesis — Volatility Surface Risk Lab

This repository contains **Alireza Moslemi Haghighi's MSc thesis project and subsequent research extensions** on SPX implied-volatility-surface risk. The project is based on the Parametric Surface Projection methodology developed with Shiva Zamani and Hamid Arian and documented in [arXiv:2311.14985](https://arxiv.org/abs/2311.14985).

The maintained research platform studies how alternative surface representations and transition laws affect option-portfolio risk. It replaces former distributed OptionsData inputs with WRDS OptionMetrics IvyDB US and separates data access, surface estimation, PCA, stochastic-volatility modeling, scenario generation, portfolio valuation, backtesting, and reporting.

The legacy PwG interest-rate spreadsheets are deliberately excluded. The pipeline obtains the daily zero curve from WRDS OptionMetrics (`zerocd`) and interpolates it to each option maturity.

## Research report

The complete empirical report is available at:

[`reports/SPX_WRDS_Thesis_Report_2005_2021.pdf`](reports/SPX_WRDS_Thesis_Report_2005_2021.pdf)

The LaTeX source and generated report fragments are under `reports/latex/`.

## Public browser versus empirical research

The accompanying online teaching laboratory does **not** load or expose WRDS/OptionMetrics observations. Its displayed surfaces are generated from visible ATM-volatility, skew, curvature, term-slope, and model-shape controls. Its scenario paths use seeded stylized Heston-, GBM/FHS-, or PCA-inspired equations, and its portfolio P&L uses disclosed delta, vega, gamma, vanna, and hedging assumptions.

The browser's fixed backtest table reproduces aggregate results from this empirical report. The underlying option records, credentials, and licensed caches remain outside the public browser and repository.

## Reproduce the included sample

The demo is deterministic, uses no credentials, and creates the same schema and artifacts as a WRDS run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
spx-risk demo --config configs/demo.yaml
```

The checked sample is in `sample_outputs/demo/`. It includes cleaned synthetic demo data, fitted surfaces, reconstruction-method validation, PCA results, PSP/PCA/PWG VaR forecasts, and seventeen publication-ready figures, including 3D surfaces, PCA reconstructions, separate 90%/95%/99% backtests, coverage tests, and Murphy diagrams.

## Run with WRDS OptionMetrics

Create the private credential file and restrict its permissions:

```bash
cp .env.example .env
chmod 600 .env
```

Fill in `WRDS_USERNAME` and `WRDS_PASSWORD`. `.env` is ignored by Git and must never be committed. The downloader discovers an accessible OptionMetrics schema, resolves annual tables, downloads only the selected dates and SPX security IDs, and caches Parquet files under `data/raw/`.

Direct database access:

```bash
spx-risk probe-wrds --config configs/default.yaml
spx-risk all --config configs/default.yaml
```

## Choose any research interval

Edit `data.start_date` and `data.end_date` in YAML, or override either from the command line:

```bash
spx-risk all --config configs/default.yaml \
  --start-date 2012-01-03 --end-date 2014-12-31
```

All research choices—filters, fixed maturities, moneyness grid, PCA components, VaR levels, rolling history, scenario count, and portfolio vega—are configurable in YAML.

For the full 2005–2021 analysis, use the annual, resumable configuration:

```bash
spx-risk all --config configs/long_horizon.yaml
```

Each completed WRDS year is written immediately to `data/raw/wrds/SPX/by_year/`. Server-side quote filters reduce transfer volume, and daily reconstructed surfaces are cached under a configuration fingerprint in `data/interim/`. A rerun therefore resumes at the first missing year or analysis partition.

## Reconstruction and risk methods

`surface.method` selects the daily production surface: `b_spline` (the paper-aligned default), `polynomial` (Dumas-style parametric form), `thin_plate`, or `linear`. `surface.comparison_methods` evaluates the alternatives on deterministic date-wise holdouts.

Risk forecasts can include:

- `psp`: raw historical surface shocks, with uniform or exponential recency weights;
- `pca`: rolling PCA reconstruction and historical factor-score bootstrap;
- `pwg`: a shrinkage-estimated full-grid Gaussian benchmark.

The long-horizon robustness layer adds:

- `PCA-C`: correlation-PCA after node-wise standardization, used to test canonical level, slope, term-slope, and curvature interpretations;
- `Heston-HS`: daily warm-started Heston calibration, Fourier pricing, Black-equivalent IV inversion, and exponentially weighted historical simulation of the Heston-implied surface;
- `Heston-MC-P`: joint one-day spot/variance simulation under trailing physical parameters and conditional risk-neutral Heston repricing of a fixed-strike, delta-hedged call strip;
- `Heston-MC-Q`: a deliberately labeled risk-neutral sensitivity, not a physical VaR forecast.

Run the extensions from the saved 45-node market surface—no WRDS download is repeated:

```bash
PYTHONPATH=src python scripts/run_extended_analysis.py \
  --config configs/long_horizon.yaml
```

The Heston cache is fingerprinted by its numerical settings, dates, and grid. The pricer uses a damped Carr–Madan transform, Gauss–Laguerre integration, vectorized strikes by maturity, and daily warm starts. Calibration success, IV RMSE, Feller margins, and parameter-bound contacts remain in machine-readable tables.

Two portfolio experiments are kept separate. The controlled surface-risk experiment maps PSP/PCA/PWG/PCA-C/Heston-HS shocks to linear vega P&L. The full-revaluation experiment fixes strikes, rolls maturity, includes the daily delta hedge, cash and dividends, and reprices every call. Its GBM+PSP-FHS benchmark samples a standardized spot innovation and complete surface shock from the same historical date, preserving their leverage dependence; Heston-MC-P/Q simulate spot and variance jointly. Dollar losses and ranks are never pooled across the two experiments.

`analysis/black_scholes.py` provides tested European call/put pricing, delta, gamma, vega, theta, rho, and Brent implied-volatility inversion. The long run uses valid OptionMetrics IV directly; set `filters.impute_missing_iv: true` to enable the price-to-IV fallback.

Every configured VaR level is evaluated with Kupiec unconditional coverage, Christoffersen independence and joint conditional coverage, consistent quantile loss, HAC Diebold–Mariano comparisons, and Murphy diagrams. Run `scripts/refresh_var_evaluation.py` to rebuild those tables and figures from a saved backtest without downloading WRDS data again.

## Project map

```text
configs/                  dated, reviewable analysis settings
src/spx_risk/data/        WRDS adapter, schema normalization, demo data
src/spx_risk/analysis/    surface fit, PCA, Heston, risk, statistical tests
src/spx_risk/visualization/ thesis plots and diagnostic dashboards
sample_outputs/demo/      reproducible synthetic sample figures, tables, and Parquet data
reports/                  empirical report, LaTeX source, and table fragments
archive/                  public provenance note; private legacy material stays local and ignored
docs/                     roadmap, WRDS mapping, audit, and reproducibility notes
tests/                    fast deterministic checks
```

Start with [the research roadmap](docs/roadmap.md), [GUI blueprint](docs/gui_blueprint.md), [paper alignment](docs/paper_alignment.md), [WRDS mapping](docs/wrds_mapping.md), and [reproducibility notes](docs/reproducibility.md).

The live OptionMetrics adapter has been validated against `optionm_all` for 2013-01-02 through 2013-06-28. See [the WRDS validation record](docs/wrds_validation.md) for row counts, schema adaptations, and headline results.

## Important interpretation note

`PWG` in the new outputs is a transparent full-grid multivariate-Gaussian benchmark with Ledoit–Wolf covariance shrinkage. It is methodologically related to the legacy PwG comparison but is not presented as a bit-for-bit reproduction of the archived MATLAB implementation. PSP, PCA, and PWG are evaluated against the same realized vega P&L and rolling information set; PCA is re-estimated at every forecast date to prevent look-ahead bias.

Raw covariance PCA should not automatically be labeled level–slope–curvature. In this sample its geometry is dominated by high-variance short-dated wing nodes. Correlation-PCA produces level-like PC1, moneyness-slope PC2, term-slope PC3, and the clearest smile curvature in PC5; both bases and their reconstruction/risk trade-offs are reported rather than selecting the visually simpler basis.

## Data acknowledgment

Wharton Research Data Services (WRDS) was used in preparing this research. This service and the data available thereon constitute valuable intellectual property and trade secrets of WRDS and/or its third-party suppliers.

The repository contains project-authored code, documentation, deterministic demo outputs, and aggregate research reporting. It does not contain WRDS credentials or raw licensed OptionMetrics extracts.

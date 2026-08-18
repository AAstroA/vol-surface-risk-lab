# GUI blueprint

## Product concept

The application should feel like a volatility-model laboratory: every chart is interactive, every
model choice is explicit, and every experiment can be saved as a reproducible configuration. The
recommended public name is **Volatility Surface Risk Lab**.

## Workspace layout

A persistent experiment bar spans the application. It shows the active market interval, portfolio,
pricing model, scenario law, horizon, confidence levels, random seed, and configuration fingerprint.
Changing a control creates a visible unsaved experiment; saving it writes a portable YAML/JSON file.

### 1. Data & market

- Choose ticker and custom start/end dates with trading-day validation.
- Select demo data or a private WRDS connection; credentials remain server-side.
- Adjust quote, maturity, moneyness, spread, and liquidity filters.
- Inspect coverage, rejected observations, zero curves, dividend yields, and missing-data warnings.

### 2. Surface studio

- Compare B-spline, polynomial, thin-plate, linear, and Heston surfaces on the same date.
- Show raw quotes over an interactive 3D surface plus synchronized smile and term-structure slices.
- Display held-out errors, hull/extrapolation flags, arbitrage diagnostics, and parameter controls.
- Animate the surface through time or pin two dates for a shock comparison.

### 3. Factor lab

- Toggle covariance PCA, correlation PCA, and selected rolling windows.
- View 3D PC loading surfaces, explained variance, factor scores, and reconstruction error.
- Move factor sliders to apply level, slope, term-slope, skew, and curvature shocks to the surface.
- Compare observed and reconstructed surfaces with linked camera angles and color scales.

### 4. Model lab

- Place Black–Scholes, local surface interpolation, Heston-P, and Heston-Q side by side.
- Inspect calibrated Heston parameters, Feller condition, fit residuals, and physical-transition
  estimates.
- Simulate linked spot/variance paths and reveal the leverage effect through selectable paths,
  confidence fans, and joint distributions.
- Expose numerical settings with convergence warnings and a safe expert-mode drawer.

### 5. Portfolio builder

- Build a strip or import positions; filter the option chain by maturity and moneyness.
- Show price, Greeks, notional, total vega, delta hedge, and concentration by surface region.
- Highlight the exact fixed strikes and maturity roll used for full revaluation.

### 6. Scenario engine

- Choose PSP, PCA, PWG, paired GBM + PSP-FHS, Heston-MC-P, or Heston-MC-Q.
- Set horizon, path count, weighting half-life, PCA factors, drift treatment, and random seed.
- Explore scenario paths, terminal distributions, surface shocks, and spot/variance dependence.
- Keep physical VaR forecasts visually distinct from risk-neutral pricing sensitivities.

### 7. Risk dashboard

- Display P&L, VaR, expected shortfall, component risk, breach timeline, and tail scenarios.
- Switch among 90%, 95%, and 99% without recomputing compatible cached results.
- Drill from a tail loss into spot, variance, surface, option, and Greek contributions.
- Compare linear-vega and full-revaluation experiments without pooling their dollar rankings.

### 8. Backtests & robustness

- Present Kupiec, independence, conditional-coverage, quantile-loss, and DM results together.
- Use regime selectors for the financial crisis, expansion, and pandemic periods.
- Show numerical convergence, window sensitivity, extrapolation share, parameter-bound contacts,
  and model-ranking uncertainty.
- Explain failures in plain language next to the formal statistic.

### 9. Compare & export

- Pin experiments and compare inputs, surfaces, factors, risk forecasts, and tests side by side.
- Export figures, CSV/Parquet tables, YAML/JSON configuration, and a timestamped PDF report.
- Include package version, data interval, model fingerprint, and seed in every exported bundle.

## Interaction principles

- **Linked views:** selecting a maturity, date, factor, path, or breach updates every relevant panel.
- **Progressive disclosure:** sensible defaults first; numerical and calibration controls live in an
  expert drawer.
- **Measure clarity:** use persistent labels and color families for physical scenarios, risk-neutral
  pricing, historical simulation, and realized outcomes.
- **No silent state:** every result is determined by visible controls plus an exportable config.
- **Privacy by design:** WRDS credentials and licensed raw data never reach the browser or exports.
- **Honest diagnostics:** failed calibrations, extrapolation, stale curves, and rejected backtests are
  prominent states, not hidden log messages.

## Recommended architecture

Use Plotly Dash for the first durable multi-tab application, with the existing `spx_risk` package as
the model layer. Plotly provides linked 2D/3D charts and Dash provides explicit callback/state
control suited to a research workspace. Keep long jobs behind a task interface so the local MVP can
run synchronously while a later deployment can add a worker queue without rewriting models.

- Model layer: existing typed Python configuration and analysis functions.
- Application layer: Dash pages, reusable control panels, and experiment state.
- Storage: Parquet for arrays, DuckDB for interactive queries, and JSON/YAML for experiment specs.
- Cache: configuration fingerprint plus data-version hash.
- Security: server-side `.env`; never serialize credentials into browser state.
- Testing: unit tests for model callbacks, snapshot tests for configuration round-trips, and one
  end-to-end demo-data flow.

## Delivery phases

1. **Explorer MVP:** Data & market, Surface studio, Factor lab, and deterministic demo mode.
2. **Risk workbench:** Portfolio builder, Scenario engine, Risk dashboard, and saved experiments.
3. **Validation suite:** Backtests, robustness, comparisons, and report export.
4. **Deployment hardening:** background jobs, access control, cache management, and optional remote
   compute while retaining a fully local mode.

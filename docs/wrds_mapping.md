# WRDS OptionMetrics mapping

The adapter normalizes WRDS IvyDB US data to a small internal contract so the analysis is independent of the former OptionsData file layout.

| Analysis field | WRDS source | WRDS field / rule |
|---|---|---|
| underlying | security name | `secnmd.ticker = 'SPX'` |
| secid | security name | distinct historical `secnmd.secid` values |
| quote_date | annual option price | `opprcdYYYY.date` |
| expiration | annual option price | `opprcdYYYY.exdate` |
| option type | annual option price | `cp_flag`: C → call, P → put |
| strike | annual option price | `strike_price / 1000` |
| bid / ask | annual option price | `best_bid`, `best_offer` |
| volume / open interest | annual option price | `volume`, `open_interest` |
| implied volatility | annual option price | `impl_volatility` |
| Greeks | annual option price | `delta`, `gamma`, `vega`, `theta` |
| forward | annual option price | dividend-aware `forward_price`; spot/rate model only as fallback |
| underlying close | annual security price | `secprdYYYY.close` |
| underlying return | annual security price | `secprdYYYY.return`, normalized as `total_return` |
| index dividend yield | index dividend | `idxdvd.rate / 100` |
| risk-free rate | annual zero curve | `zerocdYYYY.rate`, linearly interpolated by `days` |

The current WRDS product schema is expected to be `optionm_all`; `optionm` remains a compatibility candidate for older entitlements. Table names are discovered from the logged-in account rather than assumed to be available.

## Replacing OptionsData behavior

- The old per-day/per-expiry CSV discovery is replaced with date-bounded SQL and Parquet caching.
- Existing OptionMetrics implied volatility is used when valid. Missing vendor IV is excluded by default, matching the thesis model-quality rule; optional Black–Scholes inversion can be enabled with `filters.impute_missing_iv`.
- Strike scaling is handled once in the adapter.
- Zero-curve percentage-point exports are normalized to decimal continuously compounded rates before interpolation.
- Mid prices are consistently defined as `(best_bid + best_offer) / 2`.
- Moneyness uses the vendor forward when populated. Otherwise SPX uses the OptionMetrics zero curve and continuous index dividend yield: `S × exp((r − q)T)`.
- Raw credentials, phone numbers, and MFA passcodes never appear in queries, caches, metadata, or output files.

## Validation

Run `spx-risk probe-wrds` when using a different account because table availability is subscription-specific. This installation has been validated against `optionm_all`; its full run records the actual schema, SPX security IDs, date interval, row counts, rate source, and dividend source in `metadata.json` and `run_summary.json`.

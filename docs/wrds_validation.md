# WRDS validation record

The maintained pipeline was successfully run against the account's live WRDS OptionMetrics IvyDB US entitlement on 2026-08-18.

## Validated input

| Item | Result |
|---|---:|
| Schema | `optionm_all` |
| SPX security ID | 108105 |
| Interval | 2013-01-02 to 2013-06-28 |
| Option-price rows | 370,687 |
| Underlying-price rows | 124 |
| Zero-curve rows | 5,689 |
| Index-dividend rows | 124 |
| Eligible option rows after filters | 168,937 |
| Fitted surface dates | 124 |
| Grid points per date | 36 |

The raw licensed files are cached locally under `data/raw/` and excluded from Git. The live reports under `reports/` are also ignored; the synthetic sample under `sample_outputs/demo/` remains safe to publish.

## Schema adaptations verified live

- `secprd2013` calls its return field `return`; the adapter discovers and aliases it as `total_return`.
- The zero curve is the unpartitioned `zerocd` table and its rates are percentage points; the pipeline normalizes them to decimals.
- The index dividend source is `idxdvd`; its percentage rate is normalized to a decimal continuous yield.
- `opprcd2013.forward_price` exists but is null for this slice, so SPX forward moneyness correctly falls back to `S × exp((r − q)T)`.
- Missing OptionMetrics implied volatility is excluded by default, consistent with the thesis model-quality filter. Optional Black–Scholes imputation remains configurable.

## Headline output

- PC1–PC3 explain 85.5%, 10.6%, and 3.3% of daily fixed-grid IV-surface variation; together they explain 99.4%.
- At 95% confidence, the 103-day out-of-sample evaluation has 6 PCA breaches and 7 PWG-benchmark breaches, versus 5.15 expected.
- The 95% Kupiec p-values are 0.708 for PCA and 0.427 for PWG; the corresponding independence p-values are 0.386 and 0.310.

These are research outputs for the configured vega exposure model, not investment advice or a claim that the legacy MATLAB PwG implementation has been reproduced bit-for-bit.

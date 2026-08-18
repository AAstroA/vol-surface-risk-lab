# Alignment with the PSP paper

The implementation is enriched from Zamani et al., *Temporal Volatility Surface Projection: Parametric Surface Projection Method for Derivatives Portfolio Risk Management* (arXiv:2311.14985), while retaining explicit boundaries between reproduced and extended elements.

## Surface reconstruction

All models target log implied volatility using normalized forward moneyness

`x = log(F/K) / sqrt(T)`.

The available daily reconstruction families are:

| Configuration | Role | Implementation |
|---|---|---|
| `polynomial` | Paper's Dumas-style parametric surface | `1, x, T, x², xT`, fitted by least squares |
| `b_spline` | Paper's preferred non-parametric family | cubic tensor-product bases in `x` and `T`, ridge-stabilized |
| `thin_plate` | Smooth non-parametric benchmark | scaled thin-plate radial basis with quantile centers |
| `linear` | Low-assumption benchmark | Delaunay linear interpolation with nearest-neighbor hull fallback |

The pipeline reports deterministic 80/20 date-wise holdout RMSE, MAE, and R². This separates visual smoothness from predictive performance and prevents choosing a method only because it interpolates the training quotes closely.

A fixed rectangular grid can contain nodes outside that day's observed option cloud, especially deep-in-the-money short-dated calls. The code tests every grid node against the convex hull of observed normalized moneyness and maturity. In-hull values retain the selected reconstruction; out-of-hull nodes receive the nearest fitted observation and are marked with `extrapolated = true`. This prevents unsupported spline tails from creating artificial volatility spikes while keeping the data limitation auditable.

## Parametric Surface Projection scenarios

For forecast date `t`, PSP samples historical surface shocks

`ΔIVS_s = IVS_s − IVS_(s−1)`

from the rolling information window. `psp_weighting: uniform` gives each historical date equal probability; `exponential` assigns more mass to recent shocks using `psp_decay`. No future surface is used.

The paper then combines the shocked surface with simulated next-day spot/moneyness and full Black–Scholes repricing of a call portfolio. This project now reports both views without mixing them. The controlled experiment maps `ΔIVS_s` to P&L through a fixed, disclosed vega vector. The separate full-revaluation experiment fixes strikes, rolls maturity, and includes the delta hedge, cash, dividends, spot movement, and nonlinear call repricing. Its GBM+PSP-FHS benchmark pairs the standardized spot innovation with the complete surface shock from the same historical day, preserving empirical spot/surface dependence.

## Heston extension

`Heston-HS` maps each daily Heston price surface back to Black-equivalent IV and applies historical simulation to its changes; it is a structural surface filter only. `Heston-MC-P` answers the broader dynamics question by jointly simulating future spot and variance under trailing physical parameters, then pricing horizon options with current risk-neutral Heston parameters conditional on each simulated state. `Heston-MC-Q` is retained only as a pricing-measure sensitivity. All three full-revaluation engines forecast the same realized fixed-strike, delta-hedged portfolio, and their results are not pooled with the vega-only study.

## PCA and PWG extensions

PCA is fitted to fixed-grid daily IV changes and re-estimated at every forecast origin. The 3D outputs show the first three loading surfaces and the latest observed surface versus its truncated PCA reconstruction and residual.

PWG is a full-grid Gaussian benchmark estimated with Ledoit–Wolf covariance shrinkage. It is not a claim of bit-identical reproduction of the legacy MATLAB files, and the legacy PwG interest-rate workbooks remain excluded. WRDS OptionMetrics `zerocd` is the only risk-free curve source.

## Paper filters reflected in the long run

`configs/long_horizon.yaml` covers 2005–2021 and uses calls, a $1 minimum mid price, a 15-day minimum maturity, forward moneyness, mid bid/ask prices, positive vendor IV, and date-bounded WRDS tables. Additional configurable spread, open-interest, maturity, and moneyness controls make the data contract auditable.

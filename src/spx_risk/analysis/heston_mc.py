"""Joint Heston spot--variance scenarios and fixed-strike option revaluation.

This module deliberately separates two experiments:

* the existing surface-risk study evaluates linear vega P&L; and
* this full-revaluation study simulates spot and variance jointly, holds the
  option strikes fixed for one trading day, rolls time to expiry, and reprices
  a newly delta-hedged call strip.

Daily option-surface calibration identifies risk-neutral (Q) parameters.  A
VaR forecast, however, requires a physical (P) transition law.  The primary
Heston-MC-P benchmark therefore estimates the P drift and square-root variance
dynamics from a trailing window of SPX returns and the filtered variance state,
using only information available at the forecast date.  Heston-MC-Q is retained
as a clearly labelled pricing-measure sensitivity check, not as a claim that Q
is the real-world return law.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from spx_risk.analysis.heston import (
    HestonParameters,
    HestonSurfaceResult,
    heston_normalized_call_price_grid,
    heston_normalized_call_prices,
    normalized_black_call_prices,
)
from spx_risk.analysis.risk import (
    RiskResult,
    evaluate_var_backtest,
    make_vega_exposures,
)
from spx_risk.config import AppConfig


TRADING_YEAR = 252.0
CALENDAR_YEAR = 365.25


@dataclass(frozen=True)
class HestonMonteCarloResult:
    """Full-revaluation forecasts plus the audit trail used to create them."""

    risk: RiskResult
    physical_parameters: pd.DataFrame
    forecast_diagnostics: pd.DataFrame
    numerical_robustness: pd.DataFrame


def _black_call(
    spot: np.ndarray | float,
    strike: np.ndarray | float,
    maturity: np.ndarray | float,
    rate: np.ndarray | float,
    dividend_yield: np.ndarray | float,
    volatility: np.ndarray | float,
) -> np.ndarray:
    spot, strike, maturity, rate, dividend_yield, volatility = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(dividend_yield, dtype=float),
        np.asarray(volatility, dtype=float),
    )
    root = np.sqrt(np.maximum(maturity, 1e-12))
    sigma = np.maximum(volatility, 1e-8)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * sigma * sigma) * maturity
    ) / (sigma * root)
    d2 = d1 - sigma * root
    return (
        spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1)
        - strike * np.exp(-rate * maturity) * norm.cdf(d2)
    )


def _black_delta_vega(
    spot: float,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    dividend_yield: float,
    volatility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root)
    discount = np.exp(-dividend_yield * maturity)
    delta = discount * norm.cdf(d1)
    vega = spot * discount * norm.pdf(d1) * root
    return delta, vega


def load_wrds_market_history(
    config: AppConfig,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, np.ndarray]]:
    """Load cached WRDS SPX closes/dividends and OptionMetrics zero curves."""

    root = config.data.cache_dir / config.data.underlying_ticker / "by_year"
    years = range(dates.min().year, dates.max().year + 1)
    underlying_paths = [root / str(year) / "underlying.parquet" for year in years]
    curve_paths = [root / str(year) / "zero_curve.parquet" for year in years]
    missing = [path for path in (*underlying_paths, *curve_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WRDS annual cache files: {missing[:3]}")

    underlying = pd.concat(
        [pd.read_parquet(path) for path in underlying_paths], ignore_index=True
    )
    underlying["quote_date"] = pd.to_datetime(underlying["quote_date"])
    underlying = (
        underlying.sort_values("quote_date")
        .drop_duplicates("quote_date", keep="last")
        .set_index("quote_date")
        .reindex(dates)
    )
    required = ["close", "dividend_yield"]
    if underlying[required].isna().any().any():
        raise ValueError("WRDS underlying history does not cover every surface date")

    curve = pd.concat([pd.read_parquet(path) for path in curve_paths], ignore_index=True)
    curve["quote_date"] = pd.to_datetime(curve["quote_date"])
    curve["days"] = pd.to_numeric(curve["days"], errors="coerce")
    curve["rate"] = pd.to_numeric(curve["rate"], errors="coerce")
    if curve["rate"].abs().quantile(0.99) > 0.25:
        curve["rate"] /= 100.0
    curves: dict[pd.Timestamp, np.ndarray] = {}
    for date, group in curve.dropna().groupby("quote_date", sort=False):
        values = (
            group.sort_values("days")[["days", "rate"]]
            .drop_duplicates("days", keep="last")
            .to_numpy(float)
        )
        curves[pd.Timestamp(date)] = values
    missing_curve_dates = dates.difference(pd.DatetimeIndex(curves))
    underlying["zero_curve_source_date"] = underlying.index
    underlying["zero_curve_stale_days"] = 0
    available_dates = pd.DatetimeIndex(sorted(curves))
    for date in missing_curve_dates:
        prior = available_dates[available_dates < date]
        if not len(prior):
            raise ValueError(f"No current or prior zero curve is available for {date:%Y-%m-%d}")
        source_date = prior[-1]
        curves[date] = curves[source_date]
        underlying.loc[date, "zero_curve_source_date"] = source_date
        underlying.loc[date, "zero_curve_stale_days"] = int((date - source_date).days)
    return underlying, curves


def _curve_rates(curve: np.ndarray, days: np.ndarray | float) -> np.ndarray:
    return np.interp(
        np.asarray(days, dtype=float),
        curve[:, 0],
        curve[:, 1],
        left=curve[0, 1],
        right=curve[-1, 1],
    )


def estimate_physical_heston_history(
    q_parameters: pd.DataFrame,
    underlying: pd.DataFrame,
    *,
    rolling_window: int = 500,
    minimum_history: int = 252,
) -> pd.DataFrame:
    """Estimate a trailing physical transition law from observable history.

    The variance state is the option-implied Heston ``v0`` proxy.  Its use is
    transparent and pragmatic: it is observable at the forecast origin, but it
    is not a claim that risk-neutral instantaneous variance equals latent
    physical variance.  Parameter bounds stabilize noisy daily regressions and
    are reported rather than hidden.
    """

    parameters = q_parameters.copy()
    parameters["quote_date"] = pd.to_datetime(parameters["quote_date"])
    parameters = parameters.set_index("quote_date").sort_index()
    common = parameters.index.intersection(underlying.index)
    parameters = parameters.loc[common]
    state = parameters["v0"].astype(float)
    returns = np.log(underlying.loc[common, "close"].astype(float)).diff()
    dt = 1.0 / TRADING_YEAR
    rows: list[dict[str, object]] = []

    for position in range(minimum_history, len(common)):
        start = max(0, position - rolling_window)
        variance_window = state.iloc[start : position + 1].to_numpy(float)
        return_window = returns.iloc[start : position + 1].to_numpy(float)
        dividend_window = underlying.loc[
            common[start : position + 1], "dividend_yield"
        ].to_numpy(float)
        variance_lag = variance_window[:-1]
        variance_change = np.diff(variance_window)
        aligned_returns = return_window[1:]
        aligned_dividends = dividend_window[1:]
        valid = (
            np.isfinite(variance_lag)
            & np.isfinite(variance_change)
            & np.isfinite(aligned_returns)
            & np.isfinite(aligned_dividends)
        )
        variance_lag = variance_lag[valid]
        variance_change = variance_change[valid]
        aligned_returns = aligned_returns[valid]
        aligned_dividends = aligned_dividends[valid]
        design = np.column_stack([np.ones(len(variance_lag)), variance_lag])
        intercept, slope = np.linalg.lstsq(design, variance_change, rcond=None)[0]
        raw_kappa = -slope / dt
        raw_theta = (
            intercept / (raw_kappa * dt) if raw_kappa > 1e-8 else np.nan
        )
        fitted = intercept + slope * variance_lag
        residual = variance_change - fitted
        standardized_variance = residual / np.sqrt(
            np.maximum(variance_lag, 1e-6) * dt
        )
        raw_xi = float(np.std(standardized_variance, ddof=2))

        lower, upper = np.quantile(aligned_returns, [0.01, 0.99])
        robust_returns = np.clip(aligned_returns, lower, upper)
        # Close-to-close returns are ex-dividend.  Convert their annualized
        # drift to a total-return drift before the simulator subtracts today's
        # dividend yield in the ex-dividend spot equation.
        mu_total = float(np.mean(robust_returns) / dt + np.mean(aligned_dividends))
        standardized_spot = (aligned_returns - aligned_returns.mean()) / np.sqrt(
            np.maximum(variance_lag, 1e-6) * dt
        )
        raw_rho = float(
            np.corrcoef(standardized_spot, standardized_variance)[0, 1]
        )
        current = parameters.iloc[position]
        kappa = float(np.clip(raw_kappa, 0.05, 100.0))
        theta = float(
            np.clip(
                raw_theta if np.isfinite(raw_theta) and raw_theta > 0 else current.theta,
                0.001,
                1.0,
            )
        )
        xi = float(np.clip(raw_xi, 0.03, 3.0))
        rho = float(np.clip(raw_rho, -0.995, 0.25))
        date = common[position]
        rows.append(
            {
                "quote_date": date,
                "window_start": common[start],
                "window_end": date,
                "observations": len(variance_lag),
                "mu_total": float(np.clip(mu_total, -0.50, 0.50)),
                "kappa_p": kappa,
                "theta_p": theta,
                "xi_p": xi,
                "rho_p": rho,
                "raw_kappa_p": raw_kappa,
                "raw_theta_p": raw_theta,
                "raw_xi_p": raw_xi,
                "raw_rho_p": raw_rho,
                "kappa_bound": not 0.05 < raw_kappa < 100.0,
                "theta_bound": not (
                    np.isfinite(raw_theta) and 0.001 < raw_theta < 1.0
                ),
                "xi_bound": not 0.03 < raw_xi < 3.0,
                "rho_bound": not -0.995 < raw_rho < 0.25,
            }
        )
    return pd.DataFrame(rows).set_index("quote_date")


def simulate_heston_states(
    *,
    spot: float,
    variance: float,
    mu_total: float,
    dividend_yield: float,
    parameters: HestonParameters,
    standard_normals: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Projected Euler simulation with full-truncation variance coefficients.

    Drift and diffusion use the nonnegative part of variance, and each update
    is projected back to zero.  The explicit projection is slightly different
    from the canonical scheme that permits a negative auxiliary state; the
    time-step sensitivity table therefore treats discretization bias as an
    empirical diagnostic rather than assuming it away.
    """

    if standard_normals.shape[0] != steps or standard_normals.shape[2] != 2:
        raise ValueError("standard_normals must have shape (steps, scenarios, 2)")
    scenarios = standard_normals.shape[1]
    horizon_step = 1.0 / (TRADING_YEAR * steps)
    simulated_spot = np.full(scenarios, float(spot))
    simulated_variance = np.full(scenarios, float(variance))
    rho = float(np.clip(parameters.rho, -0.999999, 0.999999))
    orthogonal_scale = np.sqrt(1.0 - rho * rho)
    for step in range(steps):
        variance_positive = np.maximum(simulated_variance, 0.0)
        z_variance = standard_normals[step, :, 0]
        z_spot = rho * z_variance + orthogonal_scale * standard_normals[step, :, 1]
        root = np.sqrt(variance_positive * horizon_step)
        simulated_spot *= np.exp(
            (mu_total - dividend_yield - 0.5 * variance_positive) * horizon_step
            + root * z_spot
        )
        simulated_variance = np.maximum(
            simulated_variance
            + parameters.kappa
            * (parameters.theta - variance_positive)
            * horizon_step
            + parameters.xi * root * z_variance,
            0.0,
        )
    return simulated_spot, simulated_variance


def _rectangular_interpolate(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Vectorized bilinear interpolation on a rectangular grid."""

    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    x_values = np.clip(x, x_grid[0], x_grid[-1]) if clip else x
    y_values = np.clip(y, y_grid[0], y_grid[-1]) if clip else y
    ix = np.clip(np.searchsorted(x_grid, x_values, side="right") - 1, 0, len(x_grid) - 2)
    iy = np.clip(np.searchsorted(y_grid, y_values, side="right") - 1, 0, len(y_grid) - 2)
    wx = (x_values - x_grid[ix]) / np.maximum(x_grid[ix + 1] - x_grid[ix], 1e-15)
    wy = (y_values - y_grid[iy]) / np.maximum(y_grid[iy + 1] - y_grid[iy], 1e-15)
    v00 = values[ix, iy]
    v01 = values[ix, iy + 1]
    v10 = values[ix + 1, iy]
    v11 = values[ix + 1, iy + 1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + (1.0 - wx) * wy * v01
        + wx * (1.0 - wy) * v10
        + wx * wy * v11
    )


def _surface_interpolate(
    values: np.ndarray,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
    maturities: np.ndarray,
    moneyness: np.ndarray,
) -> tuple[np.ndarray, int]:
    clipped = int(
        np.sum(
            (maturities < maturity_grid[0])
            | (maturities > maturity_grid[-1])
            | (moneyness < moneyness_grid[0])
            | (moneyness > moneyness_grid[-1])
        )
    )
    result = _rectangular_interpolate(
        maturity_grid,
        moneyness_grid,
        values,
        maturities,
        moneyness,
        clip=False,
    )
    return result, clipped


def _batched_surface_interpolate(
    values: np.ndarray,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
    maturity_targets: np.ndarray,
    moneyness_targets: np.ndarray,
) -> np.ndarray:
    """Interpolate one surface per scenario at one point per contract."""

    scenarios, _, _ = values.shape
    output = np.empty_like(moneyness_targets, dtype=float)
    for contract, maturity in enumerate(maturity_targets):
        maturity_value = float(maturity)
        low_t = int(
            np.clip(
                np.searchsorted(maturity_grid, maturity_value, side="right") - 1,
                0,
                len(maturity_grid) - 2,
            )
        )
        weight_t = (maturity_value - maturity_grid[low_t]) / (
            maturity_grid[low_t + 1] - maturity_grid[low_t]
        )
        row = (1.0 - weight_t) * values[:, low_t, :] + weight_t * values[:, low_t + 1, :]
        money = moneyness_targets[:, contract]
        low_m = np.clip(
            np.searchsorted(moneyness_grid, money, side="right") - 1,
            0,
            len(moneyness_grid) - 2,
        )
        weight_m = (money - moneyness_grid[low_m]) / (
            moneyness_grid[low_m + 1] - moneyness_grid[low_m]
        )
        rows = np.arange(scenarios)
        output[:, contract] = (
            (1.0 - weight_m) * row[rows, low_m]
            + weight_m * row[rows, low_m + 1]
        )
    return output


def _heston_reprice(
    *,
    simulated_spot: np.ndarray,
    simulated_variance: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    dividend_yield: float,
    parameters_q: HestonParameters,
    price_basis: np.ndarray,
    integration_nodes: int,
    integration_upper: float,
    variance_grid_size: int,
    strike_grid_size: int,
) -> np.ndarray:
    scenarios = len(simulated_spot)
    option_values = np.empty((scenarios, len(strike)), dtype=float)
    for maturity_value in np.unique(maturity):
        mask = maturity == maturity_value
        local_rate = float(rate[mask][0])
        forward = simulated_spot * np.exp(
            (local_rate - dividend_yield) * maturity_value
        )
        normalized_strike = strike[mask][None, :] / forward[:, None]
        k_min = max(0.20, float(normalized_strike.min()) * 0.999)
        k_max = min(5.00, float(normalized_strike.max()) * 1.001)
        strike_grid = np.linspace(k_min, k_max, strike_grid_size)
        v_min = max(1e-8, float(simulated_variance.min()) * 0.999)
        v_max = max(v_min + 1e-6, float(simulated_variance.max()) * 1.001)
        variance_grid = np.linspace(v_min, v_max, variance_grid_size)
        normalized_grid = heston_normalized_call_price_grid(
            strike_grid,
            float(maturity_value),
            variance_grid,
            parameters_q,
            integration_nodes=integration_nodes,
            integration_upper=integration_upper,
        )
        interpolated = _rectangular_interpolate(
            variance_grid,
            strike_grid,
            normalized_grid,
            simulated_variance[:, None],
            normalized_strike,
        )
        price = (
            simulated_spot[:, None]
            * np.exp(-dividend_yield * maturity_value)
            * interpolated
            + price_basis[mask][None, :]
        )
        lower = np.maximum(
            simulated_spot[:, None] * np.exp(-dividend_yield * maturity_value)
            - strike[mask][None, :] * np.exp(-local_rate * maturity_value),
            0.0,
        )
        upper = simulated_spot[:, None] * np.exp(-dividend_yield * maturity_value)
        option_values[:, mask] = np.clip(price, lower, upper)
    return option_values


def _gbm_psp_reprice(
    *,
    simulated_spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    maturity_days: np.ndarray,
    rate: np.ndarray,
    dividend_yield: float,
    current_surface: np.ndarray,
    sampled_shocks: np.ndarray,
    maturity_grid: np.ndarray,
    moneyness_grid: np.ndarray,
) -> np.ndarray:
    forward = simulated_spot[:, None] * np.exp(
        (rate[None, :] - dividend_yield) * maturity[None, :]
    )
    future_moneyness = forward / strike[None, :]
    baseline, _ = _surface_interpolate(
        current_surface,
        maturity_grid,
        moneyness_grid,
        np.broadcast_to(maturity_days, future_moneyness.shape),
        future_moneyness,
    )
    shock = _batched_surface_interpolate(
        sampled_shocks,
        maturity_grid,
        moneyness_grid,
        maturity_days,
        future_moneyness,
    )
    volatility = np.clip(baseline + shock, 0.03, 2.0)
    return _black_call(
        simulated_spot[:, None],
        strike[None, :],
        maturity[None, :],
        rate[None, :],
        dividend_yield,
        volatility,
    )


def _tail_risk(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    threshold = float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= threshold]
    return max(0.0, -threshold), max(0.0, -float(tail.mean()))


def _numerical_checks(
    q_parameters: pd.DataFrame,
    physical_parameters: pd.DataFrame,
    underlying: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """Check time-step convergence, mean-return error, and price-grid accuracy."""

    dates = physical_parameters.index
    positions = np.linspace(0, len(dates) - 1, 12, dtype=int)
    rng = np.random.default_rng(config.project.random_seed + 991)
    rows: list[dict[str, object]] = []
    q_frame = q_parameters.set_index("quote_date")
    scenarios = max(5000, config.risk.scenarios)
    for position in positions:
        date = dates[position]
        q_row = q_frame.loc[date]
        p_row = physical_parameters.loc[date]
        q_parameters_date = HestonParameters(
            q_row.v0, q_row.kappa, q_row.theta, q_row.xi, q_row.rho
        )
        p_parameters_date = HestonParameters(
            q_row.v0, p_row.kappa_p, p_row.theta_p, p_row.xi_p, p_row.rho_p
        )
        fine_normals = rng.standard_normal((16, scenarios, 2))
        for measure, parameters, drift in (
            ("P", p_parameters_date, p_row.mu_total),
            # Set total-return drift equal to q so the ex-dividend spot is a
            # zero-carry martingale for this numerical check.  Production Q
            # scenarios use the actual one-day OptionMetrics rate.
            ("Q", q_parameters_date, underlying.loc[date, "dividend_yield"]),
        ):
            for steps in (4, 8, 16):
                block = 16 // steps
                normals = fine_normals.reshape(
                    steps, block, scenarios, 2
                ).sum(axis=1) / np.sqrt(block)
                spot, variance = simulate_heston_states(
                    spot=float(underlying.loc[date, "close"]),
                    variance=q_row.v0,
                    mu_total=float(drift),
                    dividend_yield=float(underlying.loc[date, "dividend_yield"]),
                    parameters=parameters,
                    standard_normals=normals,
                    steps=steps,
                )
                returns = spot / float(underlying.loc[date, "close"]) - 1.0
                expected_return = np.expm1(
                    (
                        float(drift)
                        - float(underlying.loc[date, "dividend_yield"])
                    )
                    / TRADING_YEAR
                )
                rows.append(
                    {
                        "quote_date": date,
                        "measure": measure,
                        "steps": steps,
                        "return_q01": np.quantile(returns, 0.01),
                        "return_q05": np.quantile(returns, 0.05),
                        "return_mean": returns.mean(),
                        "mean_return_error": returns.mean() - expected_return,
                        "variance_q50": np.quantile(variance, 0.50),
                        "variance_q99": np.quantile(variance, 0.99),
                        "zero_variance_share": np.mean(variance == 0.0),
                    }
                )

        variance_grid = np.linspace(
            max(1e-6, 0.40 * q_row.v0),
            max(2e-6, 1.80 * q_row.v0),
            config.heston.variance_grid_size,
        )
        strike_grid = np.linspace(0.70, 1.40, config.heston.strike_grid_size)
        maturity = 90.0 / CALENDAR_YEAR
        price_grid = heston_normalized_call_price_grid(
            strike_grid,
            maturity,
            variance_grid,
            q_parameters_date,
            integration_nodes=config.heston.simulation_integration_nodes,
            integration_upper=config.heston.integration_upper,
        )
        test_variance = rng.uniform(variance_grid[0], variance_grid[-1], 32)
        test_strike = rng.uniform(strike_grid[0], strike_grid[-1], 32)
        interpolated = _rectangular_interpolate(
            variance_grid,
            strike_grid,
            price_grid,
            test_variance,
            test_strike,
        )
        direct = np.array(
            [
                heston_normalized_call_prices(
                    np.array([strike]),
                    np.array([maturity]),
                    HestonParameters(
                        variance,
                        q_parameters_date.kappa,
                        q_parameters_date.theta,
                        q_parameters_date.xi,
                        q_parameters_date.rho,
                    ),
                    integration_nodes=config.heston.simulation_integration_nodes,
                    integration_upper=config.heston.integration_upper,
                )[0]
                for variance, strike in zip(test_variance, test_strike, strict=True)
            ]
        )
        error = interpolated - direct
        rows.append(
            {
                "quote_date": date,
                "measure": "PricingGrid",
                "steps": config.heston.simulation_steps,
                "price_grid_mean_abs_error": np.mean(np.abs(error)),
                "price_grid_max_abs_error": np.max(np.abs(error)),
            }
        )
    return pd.DataFrame(rows)


def run_heston_full_revaluation(
    market_surface: pd.DataFrame,
    heston: HestonSurfaceResult,
    config: AppConfig,
) -> HestonMonteCarloResult:
    """Run paired GBM-PSP FHS and Heston P/Q on one common realized P&L."""

    market = market_surface.sort_index().dropna(axis=0, how="any")
    columns = market.columns.sort_values()
    market = market.loc[:, columns]
    dates = pd.DatetimeIndex(market.index)
    underlying, curves = load_wrds_market_history(config, dates)
    q_parameters = heston.parameters.copy()
    q_parameters["quote_date"] = pd.to_datetime(q_parameters["quote_date"])
    q_parameters = q_parameters.sort_values("quote_date")
    q_by_date = q_parameters.set_index("quote_date")
    physical = estimate_physical_heston_history(
        q_parameters,
        underlying,
        rolling_window=config.heston.physical_window,
        minimum_history=config.risk.minimum_history,
    )

    maturity_grid = np.asarray(sorted(columns.get_level_values("maturity_days").unique()), float)
    moneyness_grid = np.asarray(sorted(columns.get_level_values("moneyness").unique()), float)
    maturity_days = np.asarray(columns.get_level_values("maturity_days"), float)
    moneyness = np.asarray(columns.get_level_values("moneyness"), float)
    target_vega = make_vega_exposures(columns, config.risk.total_vega).to_numpy(float)
    # Use the disclosed vega profile as a *quantity* shape, then apply one
    # common daily scale so total portfolio vega is exactly the configured
    # amount.  Dividing node by node by tiny deep-wing vegas would create
    # economically meaningless trillion-contract positions.
    quantity_shape = target_vega / target_vega.sum()
    changes = market.diff().dropna().to_numpy(float)
    log_returns = np.log(underlying["close"].astype(float)).diff()
    lagged_variance = q_by_date["v0"].reindex(dates).shift(1)
    standardized_spot_innovations = (
        log_returns / np.sqrt(np.maximum(lagged_variance, 1e-8) / TRADING_YEAR)
    ).iloc[1:].to_numpy(float)
    methods = ("gbm_psp_fr", "heston_mc_p", "heston_mc_q")
    output = pd.DataFrame(index=dates[1:], data={"actual_pnl": np.nan})
    for confidence in config.risk.confidence_levels:
        label = int(round(100 * confidence))
        for method in methods:
            output[f"var_{label}_{method}"] = np.nan
            output[f"es_{label}_{method}"] = np.nan

    rng = np.random.default_rng(config.project.random_seed + 7301)
    diagnostics: list[dict[str, object]] = []
    scenario_count = config.risk.scenarios
    steps = config.heston.simulation_steps
    heston_surface = heston.surface_matrix.reindex(index=dates, columns=columns)

    for position in range(config.risk.minimum_history, len(dates) - 1):
        if (position - config.risk.minimum_history) % 250 == 0:
            print(
                "Heston full revaluation: "
                f"{position - config.risk.minimum_history + 1}/"
                f"{len(dates) - 1 - config.risk.minimum_history} forecast dates"
            )
        date = dates[position]
        horizon_date = dates[position + 1]
        if date not in physical.index:
            continue
        calendar_gap = int((horizon_date - date).days)
        calendar_dt = calendar_gap / CALENDAR_YEAR
        surface0 = market.iloc[position].to_numpy(float).reshape(
            len(maturity_grid), len(moneyness_grid)
        )
        surface1 = market.iloc[position + 1].to_numpy(float).reshape(
            len(maturity_grid), len(moneyness_grid)
        )
        spot0 = float(underlying.loc[date, "close"])
        spot1 = float(underlying.loc[horizon_date, "close"])
        dividend0 = float(underlying.loc[date, "dividend_yield"])
        dividend1 = float(underlying.loc[horizon_date, "dividend_yield"])
        rate0 = _curve_rates(curves[date], maturity_days)
        remaining_days = np.maximum(maturity_days - calendar_gap, 1.0)
        maturity0 = maturity_days / CALENDAR_YEAR
        maturity1 = remaining_days / CALENDAR_YEAR
        rate1 = _curve_rates(curves[horizon_date], remaining_days)
        scenario_rate1 = _curve_rates(curves[date], remaining_days)
        forward0 = spot0 * np.exp((rate0 - dividend0) * maturity0)
        strike = forward0 / moneyness
        volatility0 = market.iloc[position].to_numpy(float)
        price0 = _black_call(
            spot0, strike, maturity0, rate0, dividend0, volatility0
        )
        delta0, unit_vega = _black_delta_vega(
            spot0, strike, maturity0, rate0, dividend0, volatility0
        )
        quantity_scale = config.risk.total_vega / float(quantity_shape @ unit_vega)
        quantity = quantity_scale * quantity_shape
        hedge_units = -float(quantity @ delta0)
        initial_option_value = float(quantity @ price0)
        initial_cash = -initial_option_value - hedge_units * spot0
        cash_rate = float(_curve_rates(curves[date], max(calendar_gap, 1)))
        cash_horizon = initial_cash * np.exp(cash_rate * calendar_dt)
        dividend_cash = hedge_units * spot0 * np.expm1(dividend0 * calendar_dt)

        forward1 = spot1 * np.exp((rate1 - dividend1) * maturity1)
        realized_moneyness = forward1 / strike
        volatility1, clip_count = _surface_interpolate(
            surface1,
            maturity_grid,
            moneyness_grid,
            remaining_days,
            realized_moneyness,
        )
        price1 = _black_call(
            spot1, strike, maturity1, rate1, dividend1, np.clip(volatility1, 0.03, 2.0)
        )
        actual_pnl = float(
            quantity @ price1
            + hedge_units * spot1
            + dividend_cash
            + cash_horizon
        )
        output.loc[horizon_date, "actual_pnl"] = actual_pnl

        q_row = q_by_date.loc[date]
        p_row = physical.loc[date]
        parameters_q = HestonParameters(
            q_row.v0, q_row.kappa, q_row.theta, q_row.xi, q_row.rho
        )
        parameters_p = HestonParameters(
            q_row.v0, p_row.kappa_p, p_row.theta_p, p_row.xi_p, p_row.rho_p
        )
        normals = rng.standard_normal((steps, scenario_count, 2))
        spot_p, variance_p = simulate_heston_states(
            spot=spot0,
            variance=q_row.v0,
            mu_total=p_row.mu_total,
            dividend_yield=dividend0,
            parameters=parameters_p,
            standard_normals=normals,
            steps=steps,
        )
        short_rate = float(_curve_rates(curves[date], 1.0))
        spot_q, variance_q = simulate_heston_states(
            spot=spot0,
            variance=q_row.v0,
            mu_total=short_rate,
            dividend_yield=dividend0,
            parameters=parameters_q,
            standard_normals=normals,
            steps=steps,
        )

        heston_iv0 = heston_surface.loc[date].to_numpy(float)
        heston_normalized0 = normalized_black_call_prices(
            1.0 / moneyness, maturity0, heston_iv0
        )
        heston_price0 = spot0 * np.exp(-dividend0 * maturity0) * heston_normalized0
        price_basis = price0 - heston_price0
        option_p = _heston_reprice(
            simulated_spot=spot_p,
            simulated_variance=variance_p,
            strike=strike,
            maturity=maturity1,
            rate=scenario_rate1,
            dividend_yield=dividend0,
            parameters_q=parameters_q,
            price_basis=price_basis,
            integration_nodes=config.heston.simulation_integration_nodes,
            integration_upper=config.heston.integration_upper,
            variance_grid_size=config.heston.variance_grid_size,
            strike_grid_size=config.heston.strike_grid_size,
        )
        option_q = _heston_reprice(
            simulated_spot=spot_q,
            simulated_variance=variance_q,
            strike=strike,
            maturity=maturity1,
            rate=scenario_rate1,
            dividend_yield=dividend0,
            parameters_q=parameters_q,
            price_basis=price_basis,
            integration_nodes=config.heston.simulation_integration_nodes,
            integration_upper=config.heston.integration_upper,
            variance_grid_size=config.heston.variance_grid_size,
            strike_grid_size=config.heston.strike_grid_size,
        )
        common_carry = dividend_cash + cash_horizon
        pnl_p = option_p @ quantity + hedge_units * spot_p + common_carry
        pnl_q = option_q @ quantity + hedge_units * spot_q + common_carry

        history_start = max(0, position - config.risk.rolling_window)
        history = changes[history_start:position]
        spot_history = standardized_spot_innovations[history_start:position]
        ages = len(history) - 1 - np.arange(len(history))
        probabilities = np.power(config.risk.psp_decay, ages, dtype=float)
        probabilities /= probabilities.sum()
        sampled = rng.choice(
            len(history), size=scenario_count, replace=True, p=probabilities
        )
        # Filtered historical simulation: the spot innovation and complete
        # surface shock come from the same historical day.  This preserves the
        # empirical leverage/dependence structure instead of combining two
        # independently sampled sources of risk.
        weighted_spot_mean = float(probabilities @ spot_history)
        centered_spot_history = spot_history - weighted_spot_mean
        weighted_spot_scale = float(
            np.sqrt(probabilities @ np.square(centered_spot_history))
        )
        if not np.isfinite(weighted_spot_scale) or weighted_spot_scale < 1e-8:
            raise ValueError(f"Degenerate historical spot innovations on {date:%Y-%m-%d}")
        sampled_spot_innovation = centered_spot_history[sampled] / weighted_spot_scale
        spot_volatility = np.sqrt(max(float(q_row.v0), 1e-8))
        spot_gbm = spot0 * np.exp(
            (p_row.mu_total - dividend0 - 0.5 * spot_volatility**2)
            / TRADING_YEAR
            + spot_volatility
            / np.sqrt(TRADING_YEAR)
            * sampled_spot_innovation
        )
        sampled_shocks = history[sampled].reshape(
            scenario_count, len(maturity_grid), len(moneyness_grid)
        )
        option_gbm = _gbm_psp_reprice(
            simulated_spot=spot_gbm,
            strike=strike,
            maturity=maturity1,
            maturity_days=remaining_days,
            rate=scenario_rate1,
            dividend_yield=dividend0,
            current_surface=surface0,
            sampled_shocks=sampled_shocks,
            maturity_grid=maturity_grid,
            moneyness_grid=moneyness_grid,
        )
        pnl_gbm = option_gbm @ quantity + hedge_units * spot_gbm + common_carry
        pnl_by_method = {
            "gbm_psp_fr": pnl_gbm,
            "heston_mc_p": pnl_p,
            "heston_mc_q": pnl_q,
        }
        for confidence in config.risk.confidence_levels:
            label = int(round(100 * confidence))
            for method, pnl in pnl_by_method.items():
                value_at_risk, expected_shortfall = _tail_risk(pnl, confidence)
                output.loc[horizon_date, f"var_{label}_{method}"] = value_at_risk
                output.loc[horizon_date, f"es_{label}_{method}"] = expected_shortfall
        diagnostics.append(
            {
                "forecast_date": date,
                "horizon_date": horizon_date,
                "calendar_gap": calendar_gap,
                "spot": spot0,
                "realized_spot_return": spot1 / spot0 - 1.0,
                "actual_pnl": actual_pnl,
                "surface_extrapolated_nodes": clip_count,
                "surface_extrapolated_share": clip_count / len(columns),
                "zero_curve_stale_days": int(
                    underlying.loc[date, "zero_curve_stale_days"]
                ),
                "portfolio_option_value": initial_option_value,
                "portfolio_delta_before_hedge": -hedge_units,
                "portfolio_contract_equivalents": float(np.abs(quantity).sum()),
                "portfolio_vega": float(quantity @ unit_vega),
                "pnl_mean_gbm_psp_fr": float(pnl_gbm.mean()),
                "pnl_std_gbm_psp_fr": float(pnl_gbm.std(ddof=1)),
                "pnl_mean_heston_mc_p": float(pnl_p.mean()),
                "pnl_std_heston_mc_p": float(pnl_p.std(ddof=1)),
                "pnl_mean_heston_mc_q": float(pnl_q.mean()),
                "pnl_std_heston_mc_q": float(pnl_q.std(ddof=1)),
                "paired_spot_surface_level_correlation": float(
                    np.corrcoef(spot_history, history.mean(axis=1))[0, 1]
                ),
                "spot_p_q01": float(np.quantile(spot_p / spot0 - 1.0, 0.01)),
                "spot_q_q01": float(np.quantile(spot_q / spot0 - 1.0, 0.01)),
                "variance_p_q99": float(np.quantile(variance_p, 0.99)),
                "variance_q_q99": float(np.quantile(variance_q, 0.99)),
            }
        )

    for confidence in config.risk.confidence_levels:
        label = int(round(100 * confidence))
        for method in methods:
            value = output[f"var_{label}_{method}"]
            output[f"breach_{label}_{method}"] = np.where(
                value.notna(), output["actual_pnl"] < -value, np.nan
            )
    local = replace(config, risk=replace(config.risk, methods=methods))
    risk = RiskResult(
        exposures=pd.Series(target_vega, index=columns, name="target_vega"),
        backtest=output,
        diagnostics=evaluate_var_backtest(output, local),
    )
    numerical = _numerical_checks(q_parameters, physical, underlying, config)
    return HestonMonteCarloResult(
        risk=risk,
        physical_parameters=physical.reset_index(),
        forecast_diagnostics=pd.DataFrame(diagnostics),
        numerical_robustness=numerical,
    )

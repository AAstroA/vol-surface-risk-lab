"""Heston stochastic-volatility calibration on a forward-normalized IV grid.

The market surface is already expressed in forward moneyness ``F / K``.  This
module therefore prices normalized calls with ``F = 1`` and discount factor
one, calibrates the five Heston parameters to Black-Scholes prices implied by
the observed grid, and converts the fitted prices back to Black-Scholes
implied volatilities.  Spot scale, rates, and dividends have already entered
the construction of forward moneyness upstream.

For long samples, the four persistent parameters are recalibrated at a fixed
frequency and carried forward.  Instantaneous variance is filtered every day
from the short-maturity ATM volatility.  This keeps the structural benchmark
fast, stable, and strictly real-time: no future calibration is interpolated
backward.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import roots_laguerre
from scipy.stats import norm


@dataclass(frozen=True)
class HestonParameters:
    """Risk-neutral Heston parameters in variance units."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    @property
    def feller_margin(self) -> float:
        return float(2.0 * self.kappa * self.theta - self.xi * self.xi)

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.v0, self.kappa, self.theta, self.xi, self.rho], dtype=float
        )


@dataclass(frozen=True)
class HestonSurfaceResult:
    """Daily Heston-implied surface plus calibration audit information."""

    surface_matrix: pd.DataFrame
    parameters: pd.DataFrame
    calibration: pd.DataFrame


@lru_cache(maxsize=16)
def _quadrature(nodes: int, upper: float) -> tuple[np.ndarray, np.ndarray]:
    del upper  # retained in the public API for reproducible configuration files
    frequencies, laguerre_weights = roots_laguerre(nodes)
    # Gauss-Laguerre integrates exp(-u) f(u); undo that weight to recover the
    # Fourier integral.  It is materially more stable than truncating a highly
    # oscillatory short-maturity integral at an arbitrary finite frequency.
    return frequencies, laguerre_weights * np.exp(frequencies)


def _characteristic_coefficients(
    frequency: np.ndarray,
    maturity: float,
    parameters: HestonParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine coefficients of the log normalized-forward transform.

    The branch with non-negative real part implements the numerically stable
    form of the Heston transform (the so-called little-trap representation).
    """

    u = np.asarray(frequency, dtype=complex)
    iu = 1j * u
    v0, kappa, theta, xi, rho = parameters.as_array()
    d = np.sqrt((kappa - rho * xi * iu) ** 2 + xi * xi * (u * u + iu))
    d = np.where(np.real(d) < 0.0, -d, d)
    b = kappa - rho * xi * iu
    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * maturity)
    c = (kappa * theta / (xi * xi)) * (
        (b - d) * maturity
        - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    d_coefficient = ((b - d) / (xi * xi)) * (
        (1.0 - exp_dt) / (1.0 - g * exp_dt)
    )
    return c, d_coefficient


def _characteristic_function(
    frequency: np.ndarray,
    maturity: float,
    parameters: HestonParameters,
) -> np.ndarray:
    """Characteristic function of log normalized-forward price."""

    c, d_coefficient = _characteristic_coefficients(
        frequency, maturity, parameters
    )
    return np.exp(c + d_coefficient * parameters.v0)


def heston_normalized_call_prices(
    strikes: np.ndarray,
    maturities: np.ndarray,
    parameters: HestonParameters,
    *,
    integration_nodes: int = 64,
    integration_upper: float = 160.0,
) -> np.ndarray:
    """Price normalized calls with the damped Carr--Madan inversion.

    Damping avoids the large cancellation errors of the raw ``P1/P2`` formula
    for short-dated, far out-of-the-money nodes--the exact corner that drove
    the unstable PCA picture in the original report.
    """

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    if strikes.shape != maturities.shape:
        raise ValueError("strikes and maturities must have the same shape")
    if np.any(strikes <= 0.0) or np.any(maturities <= 0.0):
        raise ValueError("strikes and maturities must be strictly positive")

    frequencies, weights = _quadrature(integration_nodes, float(integration_upper))
    output = np.empty_like(strikes)
    damping = 0.75
    for maturity in np.unique(maturities):
        mask = maturities == maturity
        strike_group = strikes[mask]
        phi_shifted = _characteristic_function(
            frequencies - (damping + 1.0) * 1j,
            float(maturity),
            parameters,
        )
        denominator = (
            damping * damping
            + damping
            - frequencies * frequencies
            + 1j * (2.0 * damping + 1.0) * frequencies
        )
        log_strike = np.log(strike_group)
        kernel = np.exp(-1j * np.outer(log_strike, frequencies))
        prices = (
            np.exp(-damping * log_strike)
            * np.real(kernel * (phi_shifted / denominator) * weights).sum(axis=1)
            / np.pi
        )
        intrinsic = np.maximum(1.0 - strike_group, 0.0)
        output[mask] = np.clip(prices, intrinsic, 1.0)
    return output


def heston_normalized_call_price_grid(
    strikes: np.ndarray,
    maturity: float,
    state_variances: np.ndarray,
    structural_parameters: HestonParameters,
    *,
    integration_nodes: int = 64,
    integration_upper: float = 160.0,
) -> np.ndarray:
    """Price a rectangular ``(variance, strike)`` grid efficiently.

    Conditional Heston repricing repeatedly changes only the current variance
    state while keeping ``kappa``, ``theta``, ``xi``, and ``rho`` fixed over a
    one-day risk horizon.  The affine characteristic function is exponential
    affine in ``v0``; exploiting that structure avoids one Fourier inversion
    per Monte Carlo path.
    """

    strikes = np.asarray(strikes, dtype=float)
    state_variances = np.asarray(state_variances, dtype=float)
    if strikes.ndim != 1 or state_variances.ndim != 1:
        raise ValueError("strikes and state_variances must be one-dimensional")
    if np.any(strikes <= 0.0) or np.any(state_variances < 0.0) or maturity <= 0.0:
        raise ValueError("strikes and maturity must be positive; variances non-negative")

    frequencies, weights = _quadrature(integration_nodes, float(integration_upper))
    damping = 0.75
    shifted = frequencies - (damping + 1.0) * 1j
    affine_parameters = HestonParameters(
        v0=0.0,
        kappa=structural_parameters.kappa,
        theta=structural_parameters.theta,
        xi=structural_parameters.xi,
        rho=structural_parameters.rho,
    )
    c, d_coefficient = _characteristic_coefficients(
        shifted,
        float(maturity),
        affine_parameters,
    )
    characteristic = np.exp(
        c[None, :] + state_variances[:, None] * d_coefficient[None, :]
    )
    denominator = (
        damping * damping
        + damping
        - frequencies * frequencies
        + 1j * (2.0 * damping + 1.0) * frequencies
    )
    log_strike = np.log(strikes)
    kernel = np.exp(-1j * np.outer(log_strike, frequencies))
    prices = (
        np.exp(-damping * log_strike)[None, :]
        * np.real((characteristic * (weights / denominator)[None, :]) @ kernel.T)
        / np.pi
    )
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    return np.clip(prices, intrinsic[None, :], 1.0)


def normalized_black_call_prices(
    strikes: np.ndarray, maturities: np.ndarray, volatility: np.ndarray
) -> np.ndarray:
    """Black-Scholes calls for normalized forward one and discount factor one."""

    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    total = volatility * np.sqrt(maturities)
    d1 = (-np.log(strikes) + 0.5 * total * total) / total
    d2 = d1 - total
    return norm.cdf(d1) - strikes * norm.cdf(d2)


def normalized_black_vega(
    strikes: np.ndarray, maturities: np.ndarray, volatility: np.ndarray
) -> np.ndarray:
    total = np.asarray(volatility, dtype=float) * np.sqrt(maturities)
    d1 = (-np.log(np.asarray(strikes, dtype=float)) + 0.5 * total * total) / total
    return norm.pdf(d1) * np.sqrt(maturities)


def normalized_implied_volatilities(
    prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    iterations: int = 36,
) -> np.ndarray:
    """Vectorized, bracketed inversion of normalized Black call prices."""

    prices = np.asarray(prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    targets = np.clip(prices, intrinsic + 1e-12, 1.0 - 1e-12)
    lower = np.full_like(targets, 1e-4)
    upper = np.full_like(targets, 3.0)
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        value = normalized_black_call_prices(strikes, maturities, middle)
        below = value < targets
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return 0.5 * (lower + upper)


def _parameters(values: np.ndarray) -> HestonParameters:
    return HestonParameters(*(float(value) for value in values))


def calibrate_heston_surface(
    market_iv: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    *,
    initial: HestonParameters | None = None,
    integration_nodes: int = 64,
    integration_upper: float = 160.0,
    max_nfev: int = 90,
) -> tuple[HestonParameters, dict[str, float | int | bool]]:
    """Calibrate Heston to market-IV-equivalent prices with vega scaling."""

    market_iv = np.asarray(market_iv, dtype=float)
    market_prices = normalized_black_call_prices(strikes, maturities, market_iv)
    vegas = np.maximum(normalized_black_vega(strikes, maturities, market_iv), 0.025)
    atm_short = float(market_iv[np.argmin(np.abs(strikes - 1.0) + maturities)] ** 2)
    atm_long = float(
        market_iv[np.argmin(np.abs(strikes - 1.0) + np.abs(maturities - maturities.max()))]
        ** 2
    )
    if initial is None:
        initial = HestonParameters(
            v0=float(np.clip(atm_short, 0.003, 0.64)),
            kappa=2.0,
            theta=float(np.clip(atm_long, 0.003, 0.64)),
            xi=0.55,
            rho=-0.70,
        )
    x0 = initial.as_array()
    lower = np.array([0.001, 0.05, 0.001, 0.03, -0.995])
    # The vol-of-vol cap excludes numerically weak extreme solutions whose
    # Fourier tails decay too slowly to be reliable on this coarse 45-node
    # surface.  The bound is disclosed and hitting rates are reported.
    upper = np.array([1.00, 12.0, 1.00, 1.50, 0.25])
    x0 = np.clip(x0, lower + 1e-8, upper - 1e-8)
    scale = np.array([0.04, 2.0, 0.04, 0.50, 0.50])

    def residual(values: np.ndarray) -> np.ndarray:
        fitted = heston_normalized_call_prices(
            strikes,
            maturities,
            _parameters(values),
            integration_nodes=integration_nodes,
            integration_upper=integration_upper,
        )
        price_residual = (fitted - market_prices) / vegas
        # A very light warm-start penalty stabilizes weakly identified daily
        # parameters without forcing the Feller inequality or hiding misfit.
        regularization = 0.0025 * (values - x0) / scale
        return np.concatenate([price_residual, regularization])

    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-7,
        xtol=2e-7,
        gtol=2e-7,
        max_nfev=max_nfev,
    )
    parameters = _parameters(fit.x)
    fitted_prices = heston_normalized_call_prices(
        strikes,
        maturities,
        parameters,
        integration_nodes=max(160, integration_nodes),
        integration_upper=integration_upper,
    )
    fitted_iv = normalized_implied_volatilities(fitted_prices, strikes, maturities)
    errors = fitted_iv - market_iv
    diagnostics: dict[str, float | int | bool] = {
        "success": bool(fit.success),
        "function_evaluations": int(fit.nfev),
        "iv_rmse": float(np.sqrt(np.mean(errors * errors))),
        "iv_mae": float(np.mean(np.abs(errors))),
        "max_abs_iv_error": float(np.max(np.abs(errors))),
        "cost": float(fit.cost),
        "feller_margin": parameters.feller_margin,
        "feller_satisfied": bool(parameters.feller_margin >= 0.0),
    }
    return parameters, diagnostics


def _daily_v0(
    observed_row: pd.Series,
    theta: float,
    kappa: float,
) -> float:
    """Infer current variance from the 30-day ATM average-variance proxy."""

    columns = observed_row.index
    maturities = np.asarray(columns.get_level_values("maturity_days"), dtype=float)
    moneyness = np.asarray(columns.get_level_values("moneyness"), dtype=float)
    target = np.argmin(np.abs(maturities - maturities.min()) + 1000.0 * np.abs(moneyness - 1.0))
    maturity = maturities[target] / 365.25
    average_variance = float(observed_row.iloc[target] ** 2)
    loading = (1.0 - np.exp(-kappa * maturity)) / (kappa * maturity)
    v0 = theta + (average_variance - theta) / max(loading, 1e-8)
    return float(np.clip(v0, 0.001, 1.0))


def build_heston_surface_history(
    market_surface_matrix: pd.DataFrame,
    *,
    recalibration_frequency: int = 21,
    integration_nodes: int = 64,
    integration_upper: float = 160.0,
    max_nfev: int = 90,
) -> HestonSurfaceResult:
    """Create a real-time Heston-filtered history for a daily IV surface."""

    if recalibration_frequency < 1:
        raise ValueError("recalibration_frequency must be positive")
    matrix = market_surface_matrix.sort_index().dropna(axis=0, how="any")
    maturity_days = np.asarray(
        matrix.columns.get_level_values("maturity_days"), dtype=float
    )
    moneyness = np.asarray(matrix.columns.get_level_values("moneyness"), dtype=float)
    strikes = 1.0 / moneyness
    maturities = maturity_days / 365.25

    fitted_rows: list[np.ndarray] = []
    parameter_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    structural: HestonParameters | None = None
    last_calibration_date: pd.Timestamp | None = None

    for position, (date, row) in enumerate(matrix.iterrows()):
        recalibrate = structural is None or position % recalibration_frequency == 0
        if recalibrate:
            structural, diagnostics = calibrate_heston_surface(
                row.to_numpy(float),
                strikes,
                maturities,
                initial=structural,
                integration_nodes=integration_nodes,
                integration_upper=integration_upper,
                max_nfev=max_nfev,
            )
            last_calibration_date = pd.Timestamp(date)
            calibration_rows.append(
                {
                    "quote_date": date,
                    **diagnostics,
                    "v0": structural.v0,
                    "kappa": structural.kappa,
                    "theta": structural.theta,
                    "xi": structural.xi,
                    "rho": structural.rho,
                }
            )
        assert structural is not None and last_calibration_date is not None
        daily = HestonParameters(
            # On calibration dates use the full-surface estimate.  Between
            # calibration dates update only v0 from short-ATM average variance.
            v0=(
                structural.v0
                if recalibrate
                else _daily_v0(row, structural.theta, structural.kappa)
            ),
            kappa=structural.kappa,
            theta=structural.theta,
            xi=structural.xi,
            rho=structural.rho,
        )
        prices = heston_normalized_call_prices(
            strikes,
            maturities,
            daily,
            integration_nodes=max(160, integration_nodes),
            integration_upper=integration_upper,
        )
        fitted_iv = normalized_implied_volatilities(prices, strikes, maturities)
        fitted_rows.append(fitted_iv)
        errors = fitted_iv - row.to_numpy(float)
        parameter_rows.append(
            {
                "quote_date": date,
                "calibration_date": last_calibration_date,
                "recalibrated": recalibrate,
                "v0": daily.v0,
                "kappa": daily.kappa,
                "theta": daily.theta,
                "xi": daily.xi,
                "rho": daily.rho,
                "feller_margin": daily.feller_margin,
                "feller_satisfied": daily.feller_margin >= 0.0,
                "daily_iv_rmse": float(np.sqrt(np.mean(errors * errors))),
                "daily_iv_mae": float(np.mean(np.abs(errors))),
            }
        )

    fitted = pd.DataFrame(fitted_rows, index=matrix.index, columns=matrix.columns)
    fitted.index.name = "quote_date"
    return HestonSurfaceResult(
        surface_matrix=fitted,
        parameters=pd.DataFrame(parameter_rows),
        calibration=pd.DataFrame(calibration_rows),
    )

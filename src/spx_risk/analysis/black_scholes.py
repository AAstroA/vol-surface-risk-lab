"""Black-Scholes-Merton pricing, Greeks, and implied-volatility inversion.

The WRDS production run consumes OptionMetrics' standardized implied
volatility.  These functions make the pricing theory explicit, support audit
checks, and provide a reproducible fallback for price-to-IV validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate(spot: float, strike: float, maturity: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive")
    if maturity < 0:
        raise ValueError("Maturity cannot be negative")
    if volatility <= 0:
        raise ValueError("Volatility must be positive")


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[float, float]:
    _validate(spot, strike, maturity, volatility)
    if maturity == 0:
        return (np.inf if spot > strike else -np.inf), (np.inf if spot > strike else -np.inf)
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    return float(d1), float(d1 - volatility * root_t)


def black_scholes_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put with continuous dividend yield."""
    _validate(spot, strike, maturity, volatility)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if maturity == 0:
        return float(max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0))
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend_yield, volatility)
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def black_scholes_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> BlackScholesGreeks:
    """Return standard European-option Greeks; vega is per unit volatility."""
    if maturity <= 0:
        raise ValueError("Greeks require strictly positive maturity")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend_yield, volatility)
    root_t = np.sqrt(maturity)
    discounted_spot = np.exp(-dividend_yield * maturity)
    discounted_strike = np.exp(-rate * maturity)
    density = norm.pdf(d1)
    gamma = discounted_spot * density / (spot * volatility * root_t)
    vega = spot * discounted_spot * density * root_t
    if option_type == "call":
        delta = discounted_spot * norm.cdf(d1)
        theta = (
            -spot * discounted_spot * density * volatility / (2.0 * root_t)
            - rate * strike * discounted_strike * norm.cdf(d2)
            + dividend_yield * spot * discounted_spot * norm.cdf(d1)
        )
        rho = strike * maturity * discounted_strike * norm.cdf(d2)
    else:
        delta = discounted_spot * (norm.cdf(d1) - 1.0)
        theta = (
            -spot * discounted_spot * density * volatility / (2.0 * root_t)
            + rate * strike * discounted_strike * norm.cdf(-d2)
            - dividend_yield * spot * discounted_spot * norm.cdf(-d1)
        )
        rho = -strike * maturity * discounted_strike * norm.cdf(-d2)
    return BlackScholesGreeks(
        delta=float(delta), gamma=float(gamma), vega=float(vega),
        theta=float(theta), rho=float(rho)
    )


def implied_volatility(
    market_price: float,
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    """Invert Black-Scholes with a bracketed Brent root finder."""
    if market_price <= 0:
        raise ValueError("Market price must be positive")

    def error(volatility: float) -> float:
        return black_scholes_price(
            option_type, spot, strike, maturity, rate, volatility, dividend_yield
        ) - market_price

    low_error, high_error = error(lower), error(upper)
    if low_error * high_error > 0:
        raise ValueError("Market price lies outside the configured implied-volatility bracket")
    return float(brentq(error, lower, upper, xtol=1e-12, rtol=1e-12))

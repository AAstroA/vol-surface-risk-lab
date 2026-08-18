"""Deterministic OptionMetrics-shaped demo data for tests and sample outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from spx_risk.config import AppConfig
from spx_risk.data.wrds import WRDSDataset


def _black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str,
) -> float:
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))
    return float(strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def generate_demo_dataset(config: AppConfig) -> WRDSDataset:
    rng = np.random.default_rng(config.project.random_seed)
    dates = pd.bdate_range(config.data.start_date, config.data.end_date)
    returns = rng.normal(0.00015, 0.008, len(dates))
    close = 1450.0 * np.exp(np.cumsum(returns))
    dividend_yield = 0.018 + 0.0015 * np.sin(np.arange(len(dates)) / 35)

    underlying = pd.DataFrame(
        {
            "secid": 108105,
            "quote_date": dates,
            "close": close,
            "volume": rng.integers(800_000, 1_400_000, len(dates)),
            "total_return": np.r_[np.nan, np.diff(close) / close[:-1]],
            "dividend_yield": dividend_yield,
        }
    )

    maturity_grid = sorted(set(config.surface.maturity_days + (270, 365)))
    rate_rows: list[dict[str, object]] = []
    option_rows: list[dict[str, object]] = []
    factor_state = np.zeros(3)
    option_id = 1_000_000

    for date_index, (quote_date, spot, dividend) in enumerate(
        zip(dates, close, dividend_yield, strict=True)
    ):
        factor_state = np.array([0.93, 0.88, 0.80]) * factor_state + rng.normal(
            0.0, [0.0040, 0.0025, 0.0018]
        )
        for maturity_days in maturity_grid:
            maturity = maturity_days / 365.25
            rate = 0.0015 + 0.000055 * maturity_days + 0.0008 * np.sin(date_index / 29)
            rate_rows.append({"quote_date": quote_date, "days": maturity_days, "rate": rate})
            forward = spot * np.exp((rate - dividend) * maturity)
            for money in np.linspace(0.72, 1.28, 15):
                strike = forward / money
                log_money = np.log(money)
                implied_volatility = (
                    0.185
                    + factor_state[0]
                    + (-0.13 + factor_state[1]) * log_money
                    + (0.72 + factor_state[2]) * log_money**2
                    + 0.025 * np.sqrt(maturity)
                )
                implied_volatility = float(np.clip(implied_volatility, 0.07, 0.80))
                for option_type, cp_flag in (("call", "C"), ("put", "P")):
                    mid = _black_scholes(
                        float(spot), float(strike), maturity, rate, implied_volatility, option_type
                    )
                    spread = max(0.05, 0.025 * mid)
                    option_rows.append(
                        {
                            "secid": 108105,
                            "quote_date": quote_date,
                            "expiration": quote_date + pd.Timedelta(days=maturity_days),
                            "cp_flag": cp_flag,
                            "type": option_type,
                            "strike": round(strike, 2),
                            "bid": max(0.01, mid - spread / 2),
                            "ask": mid + spread / 2,
                            "volume": int(rng.integers(0, 8000)),
                            "open_interest": int(rng.integers(25, 45_000)),
                            "implied_volatility": implied_volatility,
                            "delta": np.nan,
                            "gamma": np.nan,
                            "vega": np.nan,
                            "theta": np.nan,
                            "forward_price": forward,
                            "optionid": option_id,
                            "contract": str(option_id),
                            "underlying": config.data.underlying_ticker,
                        }
                    )
                    option_id += 1

    options = pd.DataFrame(option_rows)
    zero_curve = pd.DataFrame(rate_rows)
    metadata = {
        "source": "deterministic synthetic demo shaped like WRDS OptionMetrics",
        "is_demo": True,
        "underlying": config.data.underlying_ticker,
        "start_date": config.data.start_date.isoformat(),
        "end_date": config.data.end_date.isoformat(),
        "interest_rate_source": "synthetic OptionMetrics-like zero curve; legacy PwG rate files excluded",
        "option_rows": int(len(options)),
        "underlying_rows": int(len(underlying)),
        "zero_curve_rows": int(len(zero_curve)),
        "index_dividend_rows": int(len(underlying)),
        "dividend_yield_source": "synthetic OptionMetrics-like index dividend yield",
    }
    return WRDSDataset(options, underlying, zero_curve, metadata)

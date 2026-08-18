"""PSP, PCA, and full-grid Gaussian surface-shock VaR backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spx_risk.analysis.pca import PCAResult
from spx_risk.config import AppConfig


@dataclass(frozen=True)
class RiskResult:
    exposures: pd.Series
    backtest: pd.DataFrame
    diagnostics: pd.DataFrame


def make_vega_exposures(columns: pd.MultiIndex, total_vega: float) -> pd.Series:
    maturity = columns.get_level_values("maturity_days").to_numpy(float)
    moneyness = columns.get_level_values("moneyness").to_numpy(float)
    weights = np.sqrt(maturity / 365.25) * np.exp(-0.5 * np.square((moneyness - 1.0) / 0.14))
    weights /= weights.sum()
    return pd.Series(total_vega * weights, index=columns, name="vega_exposure")


def _kupiec_test(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    count = int(np.sum(breaches))
    observations = len(breaches)
    if observations == 0:
        return np.nan, np.nan
    observed = np.clip(count / observations, 1e-12, 1 - 1e-12)
    expected = alpha
    null_log_likelihood = (observations - count) * np.log(1 - expected) + count * np.log(expected)
    alt_log_likelihood = (observations - count) * np.log(1 - observed) + count * np.log(observed)
    statistic = -2 * (null_log_likelihood - alt_log_likelihood)
    return float(statistic), float(1 - chi2.cdf(statistic, df=1))


def _christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    if len(breaches) < 3:
        return np.nan, np.nan
    previous = breaches[:-1].astype(int)
    current = breaches[1:].astype(int)
    counts = np.array(
        [
            np.sum((previous == 0) & (current == 0)),
            np.sum((previous == 0) & (current == 1)),
            np.sum((previous == 1) & (current == 0)),
            np.sum((previous == 1) & (current == 1)),
        ],
        dtype=float,
    )
    n00, n01, n10, n11 = counts
    pi0 = np.clip(n01 / max(n00 + n01, 1), 1e-12, 1 - 1e-12)
    pi1 = np.clip(n11 / max(n10 + n11, 1), 1e-12, 1 - 1e-12)
    pi = np.clip((n01 + n11) / max(counts.sum(), 1), 1e-12, 1 - 1e-12)
    independent = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    markov = n00 * np.log(1 - pi0) + n01 * np.log(pi0) + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
    statistic = -2 * (independent - markov)
    return float(statistic), float(1 - chi2.cdf(statistic, df=1))


def _quantile_loss(actual: np.ndarray, forecast: np.ndarray, alpha: float) -> np.ndarray:
    """Strictly consistent pinball loss for a lower-tail quantile forecast."""
    return (alpha - (actual < forecast).astype(float)) * (actual - forecast)


def evaluate_var_backtest(backtest: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Evaluate every model and confidence level with coverage and loss tests.

    Besides the exception count, this implements the thesis' Kupiec
    unconditional-coverage and Christoffersen independence tests.  Their sum is
    the two-degree-of-freedom conditional-coverage statistic.  Magnitude and
    scoring columns make models with similar exception counts distinguishable.
    """
    rows: list[dict[str, object]] = []
    for method in config.risk.methods:
        for confidence in config.risk.confidence_levels:
            label = int(round(confidence * 100))
            var_column = f"var_{label}_{method}"
            es_column = f"es_{label}_{method}"
            breach_column = f"breach_{label}_{method}"
            valid_mask = backtest[var_column].notna() & backtest["actual_pnl"].notna()
            actual = backtest.loc[valid_mask, "actual_pnl"].to_numpy(float)
            value_at_risk = backtest.loc[valid_mask, var_column].to_numpy(float)
            expected_shortfall = backtest.loc[valid_mask, es_column].to_numpy(float)
            forecast = -value_at_risk
            valid = backtest.loc[valid_mask, breach_column].astype(bool).to_numpy()
            alpha = 1 - confidence
            kupiec_stat, kupiec_p = _kupiec_test(valid, alpha)
            independence_stat, independence_p = _christoffersen_independence(valid)
            conditional_stat = kupiec_stat + independence_stat
            conditional_p = float(1 - chi2.cdf(conditional_stat, df=2))
            breach_excess = -(actual[valid] + value_at_risk[valid])
            quantile_loss = _quantile_loss(actual, forecast, alpha)
            expected_breaches = len(valid) * alpha
            rows.append(
                {
                    "method": method.upper(),
                    "confidence": confidence,
                    "observations": len(valid),
                    "breaches": int(valid.sum()),
                    "expected_breaches": expected_breaches,
                    "breach_rate": float(valid.mean()) if len(valid) else np.nan,
                    "coverage_ratio": (
                        float(valid.sum() / expected_breaches) if expected_breaches else np.nan
                    ),
                    "mean_var": float(value_at_risk.mean()) if len(valid) else np.nan,
                    "mean_es": float(expected_shortfall.mean()) if len(valid) else np.nan,
                    "mean_quantile_loss": float(quantile_loss.mean()) if len(valid) else np.nan,
                    "mean_breach_excess": (
                        float(breach_excess.mean()) if len(breach_excess) else 0.0
                    ),
                    "kupiec_lr": kupiec_stat,
                    "kupiec_p_value": kupiec_p,
                    "independence_lr": independence_stat,
                    "independence_p_value": independence_p,
                    "conditional_coverage_lr": conditional_stat,
                    "conditional_coverage_p_value": conditional_p,
                    "kupiec_reject_5pct": bool(kupiec_p < 0.05),
                    "independence_reject_5pct": bool(independence_p < 0.05),
                    "conditional_coverage_reject_5pct": bool(conditional_p < 0.05),
                }
            )
    return pd.DataFrame(rows)


def rank_var_models(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Create an interpretable level-by-level ranking without hiding trade-offs."""
    frame = diagnostics.copy()
    frame["absolute_coverage_error"] = (
        frame["breach_rate"] - (1.0 - frame["confidence"])
    ).abs()
    frame["coverage_rank"] = frame.groupby("confidence")["absolute_coverage_error"].rank(
        method="min"
    )
    frame["quantile_loss_rank"] = frame.groupby("confidence")["mean_quantile_loss"].rank(
        method="min"
    )
    frame["combined_rank_score"] = (
        frame["coverage_rank"] + frame["quantile_loss_rank"]
    ) / 2.0
    frame["overall_rank"] = frame.groupby("confidence")["combined_rank_score"].rank(
        method="min"
    )
    columns = [
        "confidence",
        "method",
        "breach_rate",
        "absolute_coverage_error",
        "mean_quantile_loss",
        "coverage_rank",
        "quantile_loss_rank",
        "combined_rank_score",
        "overall_rank",
    ]
    return frame[columns].sort_values(["confidence", "overall_rank", "method"])


def _rolling_pca_scenarios(
    historical_changes: np.ndarray,
    exposures: np.ndarray,
    config: AppConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fit PCA using past information and bootstrap reconstructed portfolio shocks."""
    values = historical_changes
    scaler: StandardScaler | None = None
    if config.pca.standardize:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    component_count = min(config.pca.components, values.shape[0], values.shape[1])
    model = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=config.project.random_seed,
    )
    historical_scores = model.fit_transform(values)
    reconstructed = model.inverse_transform(historical_scores)
    if scaler is not None:
        reconstructed = scaler.inverse_transform(reconstructed)
    historical_pnl = reconstructed @ exposures
    sample_indices = rng.integers(0, len(historical_scores), config.risk.scenarios)
    return historical_pnl[sample_indices]


def _psp_scenarios(
    historical_changes: np.ndarray,
    exposures: np.ndarray,
    config: AppConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap raw historical surface shocks as in the paper's PSP block.

    Under the vega-isolated portfolio used here, applying a historical shock to
    today's surface and repricing linearly is exactly exposure dot surface shock.
    The implementation supports both equal and exponentially decaying scenario
    weights, making the forecast origin explicit and free of look-ahead.
    """
    probabilities: np.ndarray | None = None
    if config.risk.psp_weighting == "exponential":
        age = len(historical_changes) - 1 - np.arange(len(historical_changes))
        probabilities = np.power(config.risk.psp_decay, age, dtype=float)
        probabilities /= probabilities.sum()
    indices = rng.choice(
        len(historical_changes), size=config.risk.scenarios, replace=True, p=probabilities
    )
    return historical_changes[indices] @ exposures


def _pwg_scenarios(
    historical_changes: np.ndarray,
    exposures: np.ndarray,
    config: AppConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw portfolio P&L directly from the implied full-grid Gaussian law."""
    if config.risk.covariance_shrinkage and len(historical_changes) > 2:
        estimate = LedoitWolf().fit(historical_changes)
        mean, covariance = estimate.location_, estimate.covariance_
    else:
        mean = historical_changes.mean(axis=0)
        covariance = np.atleast_2d(np.cov(historical_changes, rowvar=False))
    portfolio_mean = float(mean @ exposures)
    portfolio_variance = float(exposures @ covariance @ exposures)
    return rng.normal(
        portfolio_mean,
        np.sqrt(max(portfolio_variance, 0.0)),
        size=config.risk.scenarios,
    )


def run_var_backtest(pca: PCAResult, config: AppConfig) -> RiskResult:
    rng = np.random.default_rng(config.project.random_seed)
    changes = pca.changes
    exposures = make_vega_exposures(changes.columns, config.risk.total_vega)
    actual_pnl = changes.to_numpy(float) @ exposures.to_numpy(float)
    output = pd.DataFrame(index=changes.index, data={"actual_pnl": actual_pnl})

    for confidence in config.risk.confidence_levels:
        label = int(round(confidence * 100))
        for method in config.risk.methods:
            output[f"var_{label}_{method}"] = np.nan
            output[f"es_{label}_{method}"] = np.nan

    minimum = config.risk.minimum_history
    for index in range(minimum, len(changes)):
        start = max(0, index - config.risk.rolling_window)
        historical_changes = changes.iloc[start:index].to_numpy(float)
        exposure_values = exposures.to_numpy(float)
        scenario_pnl: dict[str, np.ndarray] = {}
        if "psp" in config.risk.methods:
            scenario_pnl["psp"] = _psp_scenarios(
                historical_changes, exposure_values, config, rng
            )
        if "pca" in config.risk.methods:
            scenario_pnl["pca"] = _rolling_pca_scenarios(
                historical_changes, exposure_values, config, rng
            )
        if "pwg" in config.risk.methods:
            scenario_pnl["pwg"] = _pwg_scenarios(
                historical_changes, exposure_values, config, rng
            )
        for confidence in config.risk.confidence_levels:
            label = int(round(confidence * 100))
            alpha = 1 - confidence
            for method, pnl in scenario_pnl.items():
                threshold = float(np.quantile(pnl, alpha))
                tail = pnl[pnl <= threshold]
                output.iloc[index, output.columns.get_loc(f"var_{label}_{method}")] = max(
                    0.0, -threshold
                )
                output.iloc[index, output.columns.get_loc(f"es_{label}_{method}")] = max(
                    0.0, -float(tail.mean()) if len(tail) else -threshold
                )

    for confidence in config.risk.confidence_levels:
        label = int(round(confidence * 100))
        for method in config.risk.methods:
            value = output[f"var_{label}_{method}"]
            output[f"breach_{label}_{method}"] = np.where(
                value.notna(), output["actual_pnl"] < -value, np.nan
            )

    return RiskResult(exposures, output, evaluate_var_backtest(output, config))


def diebold_mariano_loss_comparison(
    backtest: pd.DataFrame,
    confidence: float = 0.95,
    method_a: str = "pca",
    method_b: str = "pwg",
) -> dict[str, float]:
    label = int(round(confidence * 100))
    alpha = 1 - confidence
    actual = backtest["actual_pnl"]
    forecast_a = -backtest[f"var_{label}_{method_a}"]
    forecast_b = -backtest[f"var_{label}_{method_b}"]
    valid = actual.notna() & forecast_a.notna() & forecast_b.notna()
    if valid.sum() < 3:
        return {"statistic": np.nan, "p_value": np.nan, "observations": int(valid.sum())}
    actual = actual[valid].to_numpy(float)
    loss_a = (alpha - (actual < forecast_a[valid]).astype(float)) * (
        actual - forecast_a[valid].to_numpy(float)
    )
    loss_b = (alpha - (actual < forecast_b[valid]).astype(float)) * (
        actual - forecast_b[valid].to_numpy(float)
    )
    difference = np.asarray(loss_a - loss_b, dtype=float)
    # Newey-West long-run variance for the quantile-loss differential. This
    # avoids treating serially correlated rolling forecasts as independent.
    centered = difference - difference.mean()
    lag_count = min(int(np.floor(len(difference) ** (1 / 3))), len(difference) - 1)
    long_run_variance = float(centered @ centered / len(centered))
    for lag in range(1, lag_count + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / len(centered))
        long_run_variance += 2 * (1 - lag / (lag_count + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / len(difference))
    statistic = difference.mean() / standard_error if standard_error > 0 else np.nan
    return {
        "statistic": float(statistic),
        "p_value": float(2 * (1 - norm.cdf(abs(statistic)))),
        "observations": int(len(difference)),
        "hac_lags": int(lag_count),
        "mean_loss_difference": float(difference.mean()),
    }

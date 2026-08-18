"""PCA interpretation and out-of-sample robustness checks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spx_risk.analysis.heston import HestonSurfaceResult
from spx_risk.analysis.pca import PCAResult
from spx_risk.analysis.risk import (
    RiskResult,
    evaluate_var_backtest,
    make_vega_exposures,
)
from spx_risk.config import AppConfig


def canonical_surface_templates(columns: pd.MultiIndex) -> pd.DataFrame:
    """Return orthonormal level, skew, term-slope, and curvature templates."""

    maturity = np.asarray(columns.get_level_values("maturity_days"), dtype=float)
    moneyness = np.asarray(columns.get_level_values("moneyness"), dtype=float)
    log_moneyness = np.log(moneyness)
    candidates = {
        "Level": np.ones(len(columns)),
        "Moneyness slope": log_moneyness - log_moneyness.mean(),
        "Term slope": np.log(maturity) - np.log(maturity).mean(),
        "Smile curvature": np.square(log_moneyness - log_moneyness.mean()),
    }
    basis: list[np.ndarray] = []
    output: dict[str, np.ndarray] = {}
    for name, candidate in candidates.items():
        value = candidate.astype(float).copy()
        for existing in basis:
            value -= np.dot(value, existing) * existing
        value /= np.linalg.norm(value)
        basis.append(value)
        output[name] = value
    return pd.DataFrame(output, index=columns)


def pca_interpretation_diagnostics(
    changes: pd.DataFrame,
    *,
    components: int = 10,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Compare covariance and correlation PCA with canonical surface shapes."""

    templates = canonical_surface_templates(changes.columns)
    records: list[dict[str, object]] = []
    loading_frames: dict[str, pd.DataFrame] = {}
    reconstruction_rows: list[dict[str, object]] = []
    for mode in ("Covariance", "Correlation"):
        raw = changes.to_numpy(float)
        scaler: StandardScaler | None = None
        values = raw
        if mode == "Correlation":
            scaler = StandardScaler()
            values = scaler.fit_transform(raw)
        count = min(components, values.shape[0], values.shape[1])
        model = PCA(n_components=count, svd_solver="full").fit(values)
        shape_loadings = model.components_.copy()
        names = [f"PC{number}" for number in range(1, count + 1)]
        loading_frames[mode] = pd.DataFrame(
            shape_loadings, index=names, columns=changes.columns
        )
        cosine = np.abs(shape_loadings @ templates.to_numpy(float))
        for index, name in enumerate(names):
            best = int(np.argmax(cosine[index]))
            row: dict[str, object] = {
                "mode": mode,
                "component": name,
                "explained_variance_ratio": model.explained_variance_ratio_[index],
                "cumulative_explained_variance": model.explained_variance_ratio_[: index + 1].sum(),
                "best_template": templates.columns[best],
                "best_abs_cosine": cosine[index, best],
            }
            row.update(
                {
                    f"cosine_{column.lower().replace(' ', '_')}": cosine[index, number]
                    for number, column in enumerate(templates.columns)
                }
            )
            records.append(row)

        full_scores = model.transform(values)
        for retained in (1, 3, 5, min(10, count)):
            truncated = np.zeros_like(full_scores)
            truncated[:, :retained] = full_scores[:, :retained]
            reconstruction = model.inverse_transform(truncated)
            if scaler is not None:
                reconstruction = scaler.inverse_transform(reconstruction)
            error = reconstruction - raw
            reconstruction_rows.append(
                {
                    "mode": mode,
                    "components": retained,
                    "rmse_iv_points": float(np.sqrt(np.mean(error * error))),
                    "relative_frobenius_error": float(
                        np.linalg.norm(error) / np.linalg.norm(raw)
                    ),
                }
            )
    return (
        pd.DataFrame(records),
        loading_frames,
        pd.DataFrame(reconstruction_rows).drop_duplicates(["mode", "components"]),
    )


def pca_subperiod_stability(changes: pd.DataFrame) -> pd.DataFrame:
    """Measure standardized-PCA subspace stability by principal-angle cosines."""

    periods = {
        "Pre-GFC (2005-2006)": ("2005-01-01", "2006-12-31"),
        "GFC (2007-2009)": ("2007-01-01", "2009-12-31"),
        "Expansion (2010-2019)": ("2010-01-01", "2019-12-31"),
        "COVID (2020-2021)": ("2020-01-01", "2021-12-31"),
    }

    def components(frame: pd.DataFrame, count: int) -> np.ndarray:
        values = StandardScaler().fit_transform(frame.to_numpy(float))
        return PCA(n_components=count, svd_solver="full").fit(values).components_

    full = {count: components(changes, count) for count in (3, 5)}
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        subset = changes.loc[start:end]
        for count in (3, 5):
            local = components(subset, count)
            singular = np.linalg.svd(full[count] @ local.T, compute_uv=False)
            rows.append(
                {
                    "period": period,
                    "components": count,
                    "observations": len(subset),
                    "mean_principal_cosine": float(singular.mean()),
                    "minimum_principal_cosine": float(singular.min()),
                    "maximum_principal_angle_degrees": float(
                        np.degrees(np.arccos(np.clip(singular.min(), -1.0, 1.0)))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _weighted_tail(
    values: np.ndarray,
    confidence: float,
    decay: float,
) -> tuple[float, float]:
    order = np.argsort(values)
    sorted_values = values[order]
    if decay < 1.0:
        age = len(values) - 1 - np.arange(len(values))
        probabilities = np.power(decay, age, dtype=float)
    else:
        probabilities = np.ones(len(values), dtype=float)
    probabilities = probabilities[order]
    probabilities /= probabilities.sum()
    alpha = 1.0 - confidence
    cumulative = np.cumsum(probabilities)
    position = min(int(np.searchsorted(cumulative, alpha, side="left")), len(values) - 1)
    threshold = float(sorted_values[position])
    tail_mask = sorted_values <= threshold
    tail_weights = probabilities[tail_mask]
    expected_shortfall = float(
        np.average(sorted_values[tail_mask], weights=tail_weights)
    )
    return threshold, expected_shortfall


def historical_surface_risk_backtest(
    scenario_changes: pd.DataFrame,
    actual_changes: pd.DataFrame,
    config: AppConfig,
    *,
    method: str,
    rolling_window: int | None = None,
    decay: float | None = None,
) -> RiskResult:
    """Deterministic exponentially weighted surface-shock VaR and ES."""

    scenario_changes = scenario_changes.reindex(actual_changes.index).dropna()
    common = actual_changes.index.intersection(scenario_changes.index)
    actual_changes = actual_changes.loc[common]
    scenario_changes = scenario_changes.loc[common]
    exposures = make_vega_exposures(actual_changes.columns, config.risk.total_vega)
    exposure_values = exposures.to_numpy(float)
    actual_pnl = actual_changes.to_numpy(float) @ exposure_values
    scenario_pnl = scenario_changes.to_numpy(float) @ exposure_values
    output = pd.DataFrame(index=common, data={"actual_pnl": actual_pnl})
    method = method.lower()
    window = rolling_window or config.risk.rolling_window
    decay = config.risk.psp_decay if decay is None else decay
    for confidence in config.risk.confidence_levels:
        label = int(round(confidence * 100))
        output[f"var_{label}_{method}"] = np.nan
        output[f"es_{label}_{method}"] = np.nan
    for index in range(config.risk.minimum_history, len(output)):
        start = max(0, index - window)
        history = scenario_pnl[start:index]
        for confidence in config.risk.confidence_levels:
            label = int(round(confidence * 100))
            threshold, expected_shortfall = _weighted_tail(history, confidence, decay)
            output.iat[index, output.columns.get_loc(f"var_{label}_{method}")] = max(
                0.0, -threshold
            )
            output.iat[index, output.columns.get_loc(f"es_{label}_{method}")] = max(
                0.0, -expected_shortfall
            )
    for confidence in config.risk.confidence_levels:
        label = int(round(confidence * 100))
        value = output[f"var_{label}_{method}"]
        output[f"breach_{label}_{method}"] = np.where(
            value.notna(), output["actual_pnl"] < -value, np.nan
        )
    local = replace(config, risk=replace(config.risk, methods=(method,)))
    return RiskResult(exposures, output, evaluate_var_backtest(output, local))


def psp_specification_robustness(
    changes: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-estimate PSP across transparent window/decay choices."""

    specifications = [
        (250, 0.995),
        (500, 0.985),
        (500, 0.995),
        (500, 1.000),
        (750, 0.995),
    ]
    diagnostics: list[pd.DataFrame] = []
    backtests: list[pd.DataFrame] = []
    for window, decay in specifications:
        label = f"PSP-W{window}-{'EQ' if decay == 1.0 else f'D{decay:.3f}'}"
        method = "psp_robust"
        result = historical_surface_risk_backtest(
            changes,
            changes,
            config,
            method=method,
            rolling_window=window,
            decay=decay,
        )
        frame = result.diagnostics.copy()
        frame["specification"] = label
        frame["rolling_window"] = window
        frame["decay"] = decay
        diagnostics.append(frame)
        bt = result.backtest.copy()
        bt["specification"] = label
        backtests.append(bt.reset_index())
    return pd.concat(diagnostics, ignore_index=True), pd.concat(backtests, ignore_index=True)


def heston_risk_backtest(
    heston: HestonSurfaceResult,
    market_changes: pd.DataFrame,
    config: AppConfig,
) -> RiskResult:
    heston_changes = heston.surface_matrix.diff().dropna()
    return historical_surface_risk_backtest(
        heston_changes,
        market_changes,
        config,
        method="heston",
    )


def subperiod_backtest_diagnostics(
    backtest: pd.DataFrame,
    methods: tuple[str, ...],
    config: AppConfig,
) -> pd.DataFrame:
    periods = {
        "GFC (2007-2009)": ("2007-01-01", "2009-12-31"),
        "Expansion (2010-2019)": ("2010-01-01", "2019-12-31"),
        "COVID (2020-2021)": ("2020-01-01", "2021-12-31"),
    }
    local = replace(config, risk=replace(config.risk, methods=methods))
    rows: list[pd.DataFrame] = []
    full = evaluate_var_backtest(backtest, local)
    full["period"] = "Full OOS"
    rows.append(full)
    for name, (start, end) in periods.items():
        frame = evaluate_var_backtest(backtest.loc[start:end], local)
        frame["period"] = name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)

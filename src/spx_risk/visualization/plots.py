"""Informative, consistent figures for data, surfaces, PCA, and VaR."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spx_risk.analysis.pca import PCAResult
from spx_risk.analysis.risk import RiskResult
from spx_risk.analysis.surface import SurfaceFit
from spx_risk.visualization.style import COLORS, apply_style


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_data_coverage(coverage: pd.DataFrame, path: Path) -> Path:
    apply_style()
    daily = coverage.set_index("quote_date").sort_index()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    series = (
        ("contracts", "Eligible contracts", COLORS["navy"]),
        ("strikes", "Distinct strikes", COLORS["blue"]),
        ("expirations", "Distinct expirations", COLORS["teal"]),
    )
    for axis, (column, label, color) in zip(axes, series, strict=True):
        axis.fill_between(daily.index, daily[column], color=color, alpha=0.14)
        axis.plot(daily.index, daily[column], color=color, linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(axis="x", visible=False)
    axes[0].set_title("SPX option-data coverage after quality filters", loc="left")
    axes[-1].set_xlabel("Quote date")
    span = daily.index.max() - daily.index.min()
    if span.days > 900:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return _save(fig, path)


def plot_surface_snapshot(surface: pd.DataFrame, path: Path, quote_date: pd.Timestamp | None = None) -> Path:
    apply_style()
    if quote_date is None:
        quote_date = pd.Timestamp(surface["quote_date"].max())
    snapshot = surface[surface["quote_date"] == quote_date]
    pivot = snapshot.pivot(index="maturity_days", columns="moneyness", values="predicted_iv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    sns.heatmap(
        pivot,
        cmap="mako",
        annot=True,
        fmt=".1%",
        cbar_kws={"label": "Implied volatility"},
        ax=axes[0],
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Interpolated surface", loc="left")
    axes[0].set_xlabel("Forward moneyness F/K")
    axes[0].set_ylabel("Maturity (days)")
    for maturity, group in snapshot.groupby("maturity_days"):
        axes[1].plot(
            group["moneyness"],
            group["predicted_iv"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{maturity} days",
        )
    axes[1].axvline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("Smile slices", loc="left")
    axes[1].set_xlabel("Forward moneyness F/K")
    axes[1].set_ylabel("Implied volatility")
    axes[1].legend(ncol=2)
    fig.suptitle(f"SPX implied-volatility surface — {quote_date:%Y-%m-%d}", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_regression_quality(diagnostics: pd.DataFrame, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(diagnostics["quote_date"], diagnostics["r_squared"], color=COLORS["blue"])
    axes[0].axhline(diagnostics["r_squared"].median(), color=COLORS["gold"], linestyle="--")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Daily surface fit", loc="left")
    axes[0].set_ylabel("R²")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    sns.histplot(diagnostics["log_iv_rmse"], bins=18, color=COLORS["teal"], ax=axes[1])
    axes[1].axvline(
        diagnostics["log_iv_rmse"].median(), color=COLORS["gold"], linestyle="--", label="Median"
    )
    axes[1].set_title("Cross-sectional residual error", loc="left")
    axes[1].set_xlabel("RMSE of log implied volatility")
    axes[1].legend()
    fig.suptitle("Regression diagnostics", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_explained_variance(pca: PCAResult, path: Path) -> Path:
    apply_style()
    frame = pca.explained_variance
    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    axis.bar(
        frame["component"],
        frame["explained_variance_ratio"],
        color=COLORS["blue"],
        alpha=0.9,
        label="Individual",
    )
    axis.plot(
        frame["component"],
        frame["cumulative_explained_variance"],
        color=COLORS["orange"],
        marker="o",
        linewidth=2,
        label="Cumulative",
    )
    axis.axhline(0.95, color=COLORS["gray"], linestyle="--", linewidth=1, label="95%")
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Share of variation")
    axis.set_title("How much surface variation each PCA factor explains", loc="left")
    axis.legend(ncol=3, loc="lower right")
    return _save(fig, path)


def plot_pca_components(pca: PCAResult, path: Path, count: int = 3) -> Path:
    apply_style()
    count = min(count, len(pca.loadings))
    maximum = np.abs(pca.loadings.iloc[:count].to_numpy()).max()
    fig, axes = plt.subplots(1, count, figsize=(4.2 * count, 4), constrained_layout=True, squeeze=False)
    for index in range(count):
        component = pca.loadings.iloc[index].unstack("moneyness")
        sns.heatmap(
            component,
            cmap="vlag",
            center=0,
            vmin=-maximum,
            vmax=maximum,
            cbar=index == count - 1,
            cbar_kws={"label": "Loading (IV points)"},
            ax=axes[0, index],
        )
        axes[0, index].invert_yaxis()
        share = pca.explained_variance.iloc[index]["explained_variance_ratio"]
        axes[0, index].set_title(f"PC{index + 1} — {share:.1%}", loc="left")
        axes[0, index].set_xlabel("Forward moneyness F/K")
        axes[0, index].set_ylabel("Maturity (days)" if index == 0 else "")
    fig.suptitle("Leading implied-volatility-surface factors", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_var_backtest(risk: RiskResult, path: Path, confidence: float = 0.95) -> Path:
    apply_style()
    label = int(round(confidence * 100))
    frame = risk.backtest
    methods = [column.removeprefix(f"var_{label}_") for column in frame if column.startswith(f"var_{label}_")]
    titles = {
        "psp": "Parametric Surface Projection / historical shocks",
        "pca": "PCA factor scenarios",
        "pwg": "Full-grid Gaussian / PwG benchmark",
    }
    colors = {"psp": COLORS["orange"], "pca": COLORS["blue"], "pwg": COLORS["teal"]}
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(12, 3.2 * len(methods)), sharex=True, constrained_layout=True,
        squeeze=False,
    )
    for axis, method in zip(axes[:, 0], methods, strict=True):
        title, color = titles.get(method, method.upper()), colors.get(method, COLORS["blue"])
        threshold = -frame[f"var_{label}_{method}"]
        breach = frame[f"breach_{label}_{method}"] == 1
        axis.plot(frame.index, frame["actual_pnl"], color=COLORS["navy"], linewidth=1, label="Realized P&L")
        axis.plot(frame.index, threshold, color=color, linewidth=1.7, label=f"{label}% VaR threshold")
        axis.fill_between(frame.index, threshold, 0, color=color, alpha=0.08)
        axis.scatter(
            frame.index[breach], frame.loc[breach, "actual_pnl"], color=COLORS["red"], s=25, zorder=4, label="Breach"
        )
        axis.axhline(0, color=COLORS["gray"], linewidth=0.8)
        axis.set_title(title, loc="left")
        axis.set_ylabel("Daily vega P&L")
        axis.legend(ncol=3, loc="lower left")
    final_axis = axes[-1, 0]
    span = frame.index.max() - frame.index.min()
    if span.days > 900:
        final_axis.xaxis.set_major_locator(mdates.YearLocator(2))
        final_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        final_axis.xaxis.set_major_locator(mdates.MonthLocator())
        final_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    final_axis.set_xlabel("Quote date")
    fig.suptitle(f"Out-of-sample {label}% VaR backtest", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_var_coverage_summary(risk: RiskResult, path: Path) -> Path:
    """Observed exception rates with Wilson intervals at every VaR level."""
    apply_style()
    diagnostics = risk.diagnostics.copy()
    confidences = sorted(diagnostics["confidence"].unique())
    methods = list(diagnostics["method"].drop_duplicates())
    colors = {"PSP": COLORS["orange"], "PCA": COLORS["blue"], "PWG": COLORS["teal"]}
    fig, axes = plt.subplots(
        1, len(confidences), figsize=(4.1 * len(confidences), 4.5),
        constrained_layout=True, squeeze=False, sharey=True,
    )
    for axis, confidence in zip(axes[0], confidences, strict=True):
        subset = diagnostics[diagnostics["confidence"] == confidence].set_index("method").reindex(methods)
        rates = subset["breach_rate"].to_numpy(float)
        n = subset["observations"].to_numpy(float)
        z = 1.96
        denominator = 1.0 + z * z / n
        center = (rates + z * z / (2.0 * n)) / denominator
        half_width = z * np.sqrt(rates * (1.0 - rates) / n + z * z / (4.0 * n * n)) / denominator
        positions = np.arange(len(methods))
        axis.bar(
            positions, rates, color=[colors.get(method, COLORS["gray"]) for method in methods],
            alpha=0.9, width=0.68,
        )
        axis.errorbar(
            positions, center, yerr=half_width, fmt="none", ecolor=COLORS["navy"],
            capsize=4, linewidth=1.2,
        )
        nominal = 1.0 - confidence
        axis.axhline(nominal, color=COLORS["red"], linestyle="--", linewidth=1.5,
                     label=f"Nominal {nominal:.0%}")
        axis.set_xticks(positions, methods)
        axis.set_title(f"{confidence:.0%} VaR", loc="left")
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        axis.set_xlabel("Model")
        axis.legend(loc="upper right")
    axes[0, 0].set_ylabel("Observed exception rate (95% Wilson CI)")
    fig.suptitle("VaR coverage across confidence levels", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_var_test_heatmap(risk: RiskResult, path: Path) -> Path:
    """Display the three Christoffersen-family test p-values together."""
    apply_style()
    diagnostics = risk.diagnostics.copy()
    diagnostics["confidence_label"] = diagnostics["confidence"].map(lambda x: f"{x:.0%}")
    tests = (
        ("kupiec_p_value", "Kupiec unconditional coverage"),
        ("independence_p_value", "Christoffersen independence"),
        ("conditional_coverage_p_value", "Conditional coverage (joint)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.1), constrained_layout=True)
    for index, (axis, (column, title)) in enumerate(zip(axes, tests, strict=True)):
        pivot = diagnostics.pivot(index="method", columns="confidence_label", values=column)
        pivot = pivot.reindex(index=[m for m in ("PSP", "PCA", "PWG") if m in pivot.index])
        sns.heatmap(
            pivot.clip(upper=0.10), annot=pivot, fmt=".3f", cmap="RdYlGn", vmin=0.0, vmax=0.10,
            linewidths=1.0, linecolor="white", cbar=index == 2,
            cbar_kws={"label": "p-value (color capped at 0.10)"}, ax=axis,
        )
        axis.set_title(title, loc="left")
        axis.set_xlabel("VaR confidence")
        axis.set_ylabel("Model" if index == 0 else "")
    fig.suptitle("Coverage-test evidence: values below 0.05 reject the null", fontsize=14, weight="bold")
    return _save(fig, path)


def _elementary_quantile_score(
    actual: np.ndarray, forecast: np.ndarray, alpha: float, theta: np.ndarray
) -> np.ndarray:
    actual_2d = actual[:, None]
    forecast_2d = forecast[:, None]
    theta_2d = theta[None, :]
    return np.mean(
        ((actual_2d < forecast_2d).astype(float) - alpha)
        * ((theta_2d < forecast_2d).astype(float) - (theta_2d < actual_2d).astype(float)),
        axis=0,
    )


def plot_var_murphy_diagrams(risk: RiskResult, path: Path) -> Path:
    """Murphy diagrams for lower-tail quantiles at 90%, 95%, and 99%."""
    apply_style()
    frame = risk.backtest
    confidences = sorted(risk.diagnostics["confidence"].unique())
    colors = {"psp": COLORS["orange"], "pca": COLORS["blue"], "pwg": COLORS["teal"]}
    fig, axes = plt.subplots(
        1, len(confidences), figsize=(4.6 * len(confidences), 4.5),
        constrained_layout=True, squeeze=False,
    )
    for axis, confidence in zip(axes[0], confidences, strict=True):
        label = int(round(confidence * 100))
        alpha = 1.0 - confidence
        methods = [
            column.removeprefix(f"var_{label}_")
            for column in frame.columns if column.startswith(f"var_{label}_")
        ]
        valid = frame[f"var_{label}_{methods[0]}"].notna() & frame["actual_pnl"].notna()
        actual = frame.loc[valid, "actual_pnl"].to_numpy(float)
        all_values = [actual]
        forecasts: dict[str, np.ndarray] = {}
        for method in methods:
            forecast = -frame.loc[valid, f"var_{label}_{method}"].to_numpy(float)
            forecasts[method] = forecast
            all_values.append(forecast)
        pooled = np.concatenate(all_values)
        lower, upper = np.quantile(pooled, [0.002, 0.40])
        theta = np.linspace(lower, upper, 260)
        for method in methods:
            score = _elementary_quantile_score(actual, forecasts[method], alpha, theta)
            axis.plot(theta, score, color=colors.get(method, COLORS["gray"]), linewidth=2,
                      label=method.upper())
        axis.set_title(f"{confidence:.0%} VaR", loc="left")
        axis.set_xlabel(r"Threshold $\theta$ (daily vega P&L)")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axis.legend()
    axes[0, 0].set_ylabel("Mean elementary quantile score (lower is better)")
    fig.suptitle("Murphy diagrams for robust VaR forecast comparison", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_pnl_distribution(risk: RiskResult, path: Path) -> Path:
    apply_style()
    values = risk.backtest["actual_pnl"].dropna()
    fig, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    sns.histplot(values, bins=24, stat="density", color=COLORS["blue"], alpha=0.35, ax=axis)
    sns.kdeplot(values, color=COLORS["navy"], linewidth=2, ax=axis)
    axis.axvline(values.quantile(0.05), color=COLORS["red"], linestyle="--", label="Empirical 5% quantile")
    axis.axvline(0, color=COLORS["gray"], linewidth=1)
    axis.set_title("Distribution of realized daily vega P&L", loc="left")
    axis.set_xlabel("Daily vega P&L")
    axis.legend()
    return _save(fig, path)


def _mesh(frame: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot(index="maturity_days", columns="moneyness", values=value).sort_index()
    x, y = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
    return x, y, pivot.to_numpy(float)


def _format_3d(axis: plt.Axes, title: str, z_label: str) -> None:
    axis.set_title(title, loc="left", pad=10)
    axis.set_xlabel("Forward moneyness F/K", labelpad=7)
    axis.set_ylabel("Maturity (days)", labelpad=7)
    axis.set_zlabel(z_label, labelpad=7)
    axis.view_init(elev=25, azim=-128)


def plot_surface_3d(surface: pd.DataFrame, path: Path) -> Path:
    apply_style()
    quote_date = pd.Timestamp(surface["quote_date"].max())
    snapshot = surface[surface["quote_date"] == quote_date]
    x, y, z = _mesh(snapshot, "predicted_iv")
    fig = plt.figure(figsize=(10, 6.2), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    plotted = axis.plot_surface(x, y, z, cmap="viridis", linewidth=0.25, edgecolor="white", alpha=0.96)
    axis.contour(x, y, z, zdir="z", offset=float(z.min()), cmap="viridis", alpha=0.6)
    _format_3d(axis, f"{snapshot['method'].iloc[0].replace('_', ' ').title()} — {quote_date:%Y-%m-%d}", "Implied volatility")
    axis.zaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    fig.colorbar(plotted, ax=axis, shrink=0.65, pad=0.09, label="Implied volatility")
    fig.suptitle("Reconstructed SPX implied-volatility surface", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_pca_components_3d(pca: PCAResult, path: Path, count: int = 3) -> Path:
    apply_style()
    count = min(count, len(pca.loadings))
    maximum = float(np.abs(pca.loadings.iloc[:count].to_numpy()).max())
    fig = plt.figure(figsize=(5.1 * count, 5), constrained_layout=True)
    for index in range(count):
        axis = fig.add_subplot(1, count, index + 1, projection="3d")
        component = pca.loadings.iloc[index].rename("loading").reset_index()
        x, y, z = _mesh(component, "loading")
        plotted = axis.plot_surface(
            x, y, z, cmap="coolwarm", vmin=-maximum, vmax=maximum, linewidth=0.2, edgecolor="white"
        )
        share = pca.explained_variance.iloc[index]["explained_variance_ratio"]
        _format_3d(axis, f"PC{index + 1} — {share:.1%}", "Loading")
    fig.colorbar(plotted, ax=fig.axes, shrink=0.55, pad=0.03, label="Loading (IV points)")
    fig.suptitle("Three-dimensional PCA volatility-surface factors", fontsize=14, weight="bold")
    return _save(fig, path)


def plot_pca_reconstruction_3d(pca: PCAResult, path: Path, components: int = 3) -> Path:
    apply_style()
    components = min(components, pca.model.n_components_)
    latest_change = pca.changes.iloc[[-1]].to_numpy(float)
    transformed = pca.scaler.transform(latest_change) if pca.scaler is not None else latest_change
    scores = pca.model.transform(transformed)
    scores[:, components:] = 0.0
    reconstructed_change = pca.inverse_transform(scores)[0]
    previous = pca.surface_matrix.iloc[-2].to_numpy(float)
    actual = pca.surface_matrix.iloc[-1].to_numpy(float)
    reconstructed = previous + reconstructed_change
    columns = pca.surface_matrix.columns
    frames = []
    for name, values in (
        ("Observed surface", actual),
        (f"PC1–PC{components} reconstruction", reconstructed),
        ("Reconstruction error", actual - reconstructed),
    ):
        frame = pd.Series(values, index=columns, name="value").reset_index()
        frames.append((name, frame))

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    main_values = np.concatenate([actual, reconstructed])
    for index, (name, frame) in enumerate(frames):
        axis = fig.add_subplot(1, 3, index + 1, projection="3d")
        x, y, z = _mesh(frame, "value")
        if index < 2:
            plotted = axis.plot_surface(
                x, y, z, cmap="viridis", vmin=float(main_values.min()), vmax=float(main_values.max()),
                linewidth=0.2, edgecolor="white"
            )
            axis.zaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            z_label = "Implied volatility"
        else:
            maximum_error = max(float(np.abs(z).max()), 1e-8)
            plotted = axis.plot_surface(
                x, y, z, cmap="coolwarm", vmin=-maximum_error, vmax=maximum_error,
                linewidth=0.2, edgecolor="white"
            )
            z_label = "IV error"
        _format_3d(axis, name, z_label)
    fig.suptitle(
        f"PCA reconstruction of the latest daily surface ({pca.surface_matrix.index[-1]:%Y-%m-%d})",
        fontsize=14,
        weight="bold",
    )
    return _save(fig, path)


def plot_method_comparison_3d(comparison: pd.DataFrame, path: Path) -> Path:
    apply_style()
    methods = list(comparison["method"].drop_duplicates())
    columns = min(2, len(methods))
    rows = int(np.ceil(len(methods) / columns))
    figure = plt.figure(figsize=(7.2 * columns, 5.0 * rows), constrained_layout=True)
    z_min = float(comparison["predicted_iv"].min())
    z_max = float(comparison["predicted_iv"].max())
    for index, method in enumerate(methods):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        x, y, z = _mesh(comparison[comparison["method"] == method], "predicted_iv")
        axis.plot_surface(
            x, y, z, cmap="viridis", vmin=z_min, vmax=z_max, linewidth=0.2, edgecolor="white"
        )
        axis.set_zlim(z_min, z_max)
        axis.zaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        _format_3d(axis, method.replace("_", " ").title(), "Implied volatility")
    date = pd.Timestamp(comparison["quote_date"].max())
    figure.suptitle(
        f"Volatility-surface reconstruction methods — {date:%Y-%m-%d}", fontsize=14, weight="bold"
    )
    return _save(figure, path)


def plot_method_metrics(comparison: pd.DataFrame, path: Path) -> Path:
    apply_style()
    order = comparison.groupby("method")["iv_rmse"].median().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    sns.boxplot(data=comparison, x="method", y="iv_rmse", order=order, color=COLORS["blue"], ax=axes[0])
    axes[0].set_title("Out-of-sample IV error", loc="left")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE (IV points)")
    axes[0].tick_params(axis="x", rotation=20)
    medians = comparison.groupby("method")["r_squared"].median().reindex(order)
    axes[1].bar(medians.index, medians.values, color=COLORS["teal"])
    axes[1].set_title("Median validation R²", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Reconstruction-method validation", fontsize=14, weight="bold")
    return _save(fig, path)


def create_all_plots(
    surface_fit: SurfaceFit,
    pca: PCAResult,
    risk: RiskResult,
    output_directory: Path,
) -> list[Path]:
    confidences = sorted(float(value) for value in risk.diagnostics["confidence"].unique())
    primary_confidence = min(confidences, key=lambda value: abs(value - 0.95))
    primary_label = int(round(primary_confidence * 100))
    figures = [
        plot_data_coverage(surface_fit.coverage, output_directory / "01_data_coverage.png"),
        plot_surface_snapshot(surface_fit.surface, output_directory / "02_surface_snapshot.png"),
        plot_regression_quality(surface_fit.diagnostics, output_directory / "03_regression_quality.png"),
        plot_explained_variance(pca, output_directory / "04_pca_explained_variance.png"),
        plot_pca_components(pca, output_directory / "05_pca_components.png"),
        plot_var_backtest(
            risk,
            output_directory / f"06_var_backtest_{primary_label}.png",
            confidence=primary_confidence,
        ),
        plot_pnl_distribution(risk, output_directory / "07_pnl_distribution.png"),
        plot_surface_3d(surface_fit.surface, output_directory / "08_reconstructed_surface_3d.png"),
        plot_pca_components_3d(pca, output_directory / "09_pca_components_3d.png"),
        plot_pca_reconstruction_3d(pca, output_directory / "10_pca_reconstruction_3d.png"),
    ]
    for figure_number, confidence in enumerate(
        (value for value in confidences if value != primary_confidence), start=13
    ):
        label = int(round(confidence * 100))
        figures.append(
            plot_var_backtest(
                risk,
                output_directory / f"{figure_number:02d}_var_backtest_{label}.png",
                confidence=confidence,
            )
        )
    summary_number = 13 + len(confidences) - 1
    figures.extend(
        [
            plot_var_coverage_summary(
                risk, output_directory / f"{summary_number:02d}_var_coverage_summary.png"
            ),
            plot_var_test_heatmap(
                risk, output_directory / f"{summary_number + 1:02d}_var_test_heatmap.png"
            ),
            plot_var_murphy_diagrams(
                risk, output_directory / f"{summary_number + 2:02d}_var_murphy_diagrams.png"
            ),
        ]
    )
    if not surface_fit.comparison_surfaces.empty:
        figures.append(
            plot_method_comparison_3d(
                surface_fit.comparison_surfaces,
                output_directory / "11_surface_methods_3d.png",
            )
        )
    if not surface_fit.method_comparison.empty:
        figures.append(
            plot_method_metrics(
                surface_fit.method_comparison,
                output_directory / "12_surface_method_metrics.png",
            )
        )
    return sorted(figures, key=lambda figure: figure.name)

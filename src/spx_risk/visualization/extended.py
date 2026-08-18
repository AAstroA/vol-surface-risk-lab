"""Publication figures for PCA robustness and the Heston extension."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from spx_risk.visualization.style import COLORS, apply_style


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _surface_coordinates(columns: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
    maturity = np.unique(columns.get_level_values("maturity_days").to_numpy(float))
    moneyness = np.unique(columns.get_level_values("moneyness").to_numpy(float))
    return np.meshgrid(moneyness, maturity)


def plot_pca_standardization_3d(
    loadings: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    path: Path,
) -> Path:
    """Show why raw covariance PCA and correlation PCA look different."""

    apply_style()
    selected = (0, 1, 2, 4)
    fig = plt.figure(figsize=(16, 8.3), constrained_layout=True)
    all_values = np.concatenate(
        [loadings[mode].iloc[list(selected)].to_numpy().ravel() for mode in ("Covariance", "Correlation")]
    )
    limit = np.quantile(np.abs(all_values), 0.99)
    for row, mode in enumerate(("Covariance", "Correlation")):
        frame = loadings[mode]
        x, y = _surface_coordinates(frame.columns)
        subset = diagnostics[diagnostics["mode"] == mode].set_index("component")
        for column, component_index in enumerate(selected):
            axis = fig.add_subplot(2, 4, row * 4 + column + 1, projection="3d")
            z = frame.iloc[component_index].unstack("moneyness").to_numpy(float)
            axis.plot_surface(
                x, y, z, cmap="coolwarm", vmin=-limit, vmax=limit,
                linewidth=0.25, edgecolor="white", antialiased=True,
            )
            component = f"PC{component_index + 1}"
            result = subset.loc[component]
            axis.set_title(
                f"{component}: {result.explained_variance_ratio:.1%}\n"
                f"closest to {result.best_template} (|cos|={result.best_abs_cosine:.2f})",
                pad=8,
            )
            axis.set_xlabel("F/K", labelpad=5)
            axis.set_ylabel("Days", labelpad=5)
            axis.set_zlabel("Direction", labelpad=4)
            axis.view_init(elev=24, azim=-58)
            axis.tick_params(labelsize=8)
        fig.text(
            0.012,
            0.73 if row == 0 else 0.27,
            f"{mode}\nPCA",
            color=COLORS["navy"] if row == 0 else COLORS["teal"],
            fontsize=13,
            weight="bold",
            ha="center",
            va="center",
            rotation=90,
        )
    fig.suptitle(
        "Raw node variance distorts PCA geometry; correlation scaling restores a level factor",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_pca_diagnostic_dashboard(
    diagnostics: pd.DataFrame,
    reconstruction: pd.DataFrame,
    stability: pd.DataFrame,
    path: Path,
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    correlation = diagnostics[diagnostics["mode"] == "Correlation"].iloc[:5]
    cosine_columns = [column for column in correlation if column.startswith("cosine_")]
    labels = [
        column.removeprefix("cosine_").replace("_", " ").title()
        for column in cosine_columns
    ]
    cosine = correlation.set_index("component")[cosine_columns]
    cosine.columns = labels
    sns.heatmap(
        cosine,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="mako",
        cbar_kws={"label": "Absolute cosine"},
        ax=axes[0],
    )
    axes[0].set_title("Canonical-shape alignment", loc="left")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.lineplot(
        data=reconstruction,
        x="components",
        y="relative_frobenius_error",
        hue="mode",
        marker="o",
        palette={"Covariance": COLORS["blue"], "Correlation": COLORS["teal"]},
        ax=axes[1],
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("Change reconstruction error", loc="left")
    axes[1].set_xlabel("Retained components")
    axes[1].set_ylabel("Relative Frobenius error")
    axes[1].legend(title="PCA basis")

    stable = stability[stability["components"] == 5].copy()
    axes[2].barh(
        stable["period"], stable["mean_principal_cosine"], color=COLORS["gold"]
    )
    axes[2].scatter(
        stable["minimum_principal_cosine"], stable["period"],
        color=COLORS["red"], label="Weakest direction", zorder=3,
    )
    axes[2].set_xlim(0, 1.02)
    axes[2].set_title("Five-factor subspace stability", loc="left")
    axes[2].set_xlabel("Principal-angle cosine (1 = identical)")
    axes[2].set_ylabel("")
    axes[2].legend(loc="lower right")
    fig.suptitle(
        "PCA interpretation and stability checks",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_heston_surface_comparison(
    observed: pd.DataFrame,
    heston: pd.DataFrame,
    path: Path,
) -> Path:
    apply_style()
    date = observed.index.intersection(heston.index).max()
    observed_row = observed.loc[date]
    heston_row = heston.loc[date]
    x, y = _surface_coordinates(observed.columns)
    panels = [
        (observed_row, "Observed B-spline surface", "viridis", "Implied volatility"),
        (heston_row, "Heston-implied surface", "viridis", "Implied volatility"),
        (heston_row - observed_row, "Heston minus observed", "coolwarm", "IV error"),
    ]
    fig = plt.figure(figsize=(14, 4.9), constrained_layout=True)
    for index, (series, title, cmap, zlabel) in enumerate(panels, start=1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        z = series.unstack("moneyness").to_numpy(float)
        axis.plot_surface(x, y, z, cmap=cmap, linewidth=.3, edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel("Forward moneyness F/K")
        axis.set_ylabel("Maturity (days)")
        axis.set_zlabel(zlabel)
        axis.view_init(elev=24, azim=-58)
    rmse = float(np.sqrt(np.mean(np.square((heston_row - observed_row).to_numpy(float)))))
    fig.suptitle(
        f"Stochastic-volatility surface reconstruction — {date:%Y-%m-%d} (RMSE {rmse:.3%})",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_heston_parameters(parameters: pd.DataFrame, path: Path) -> Path:
    apply_style()
    frame = parameters.copy()
    frame["quote_date"] = pd.to_datetime(frame["quote_date"])
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8), sharex=True, constrained_layout=True)
    series = [
        ("v0", "Filtered instantaneous volatility", COLORS["navy"], True),
        ("theta", "Long-run volatility", COLORS["blue"], True),
        ("kappa", "Mean reversion κ", COLORS["teal"], False),
        ("xi", "Vol-of-vol ξ", COLORS["orange"], False),
        ("rho", "Spot/variance correlation ρ", COLORS["red"], False),
        ("daily_iv_rmse", "Daily surface RMSE", COLORS["gray"], True),
    ]
    for axis, (column, title, color, square_root) in zip(axes.ravel(), series, strict=True):
        values = frame[column].to_numpy(float)
        if square_root and column in {"v0", "theta"}:
            values = np.sqrt(np.clip(values, 0, None))
        axis.plot(frame["quote_date"], values, color=color, linewidth=.9)
        axis.set_title(title, loc="left")
        if square_root:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.xaxis.set_major_locator(mdates.YearLocator(3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Real-time Heston parameter filter and fit quality",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_extended_var_coverage(diagnostics: pd.DataFrame, path: Path) -> Path:
    apply_style()
    methods = list(diagnostics["method"].drop_duplicates())
    confidences = sorted(diagnostics["confidence"].unique())
    palette = [COLORS["orange"], COLORS["blue"], COLORS["gold"], COLORS["teal"], COLORS["red"]]
    fig, axes = plt.subplots(1, len(confidences), figsize=(14, 4.5), sharey=False, constrained_layout=True)
    for axis, confidence in zip(np.atleast_1d(axes), confidences, strict=True):
        subset = diagnostics[diagnostics["confidence"] == confidence].set_index("method").loc[methods]
        expected = 1.0 - confidence
        axis.bar(methods, subset["breach_rate"], color=palette[: len(methods)], alpha=.9)
        axis.axhline(expected, color=COLORS["navy"], linestyle="--", linewidth=1.4, label="Nominal")
        for position, (_, row) in enumerate(subset.iterrows()):
            axis.text(
                position, row.breach_rate + expected * .04,
                "pass" if not row.conditional_coverage_reject_5pct else "reject",
                ha="center", va="bottom", fontsize=8,
                color=COLORS["teal"] if not row.conditional_coverage_reject_5pct else COLORS["red"],
                weight="bold",
            )
        axis.set_title(f"{confidence:.0%} VaR", loc="left")
        axis.set_ylabel("Observed breach rate")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle(
        "Out-of-sample coverage: original, correlation-PCA, and Heston-filtered models",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_robustness_dashboard(
    psp: pd.DataFrame,
    subperiod: pd.DataFrame,
    path: Path,
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    psp_pivot = psp.pivot(index="specification", columns="confidence", values="coverage_ratio")
    psp_pivot.columns = [f"{value:.0%}" for value in psp_pivot.columns]
    sns.heatmap(
        psp_pivot,
        annot=True,
        fmt=".2f",
        center=1.0,
        cmap="vlag_r",
        vmin=.65,
        vmax=1.35,
        cbar_kws={"label": "Observed / expected breaches"},
        ax=axes[0],
    )
    axes[0].set_title("PSP window and decay sensitivity", loc="left")
    axes[0].set_xlabel("VaR level")
    axes[0].set_ylabel("")

    subset = subperiod[subperiod["confidence"] == .99].copy()
    pivot = subset.pivot(index="method", columns="period", values="coverage_ratio")
    order = [column for column in ["Full OOS", "GFC (2007-2009)", "Expansion (2010-2019)", "COVID (2020-2021)"] if column in pivot]
    pivot = pivot[order]
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        center=1.0,
        cmap="vlag_r",
        vmin=.25,
        vmax=2.25,
        cbar_kws={"label": "Observed / expected breaches"},
        ax=axes[1],
    )
    axes[1].set_title("99% VaR regime sensitivity", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle(
        "Specification and crisis-regime robustness",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_extended_var_backtest(backtest: pd.DataFrame, path: Path, confidence: float = .95) -> Path:
    apply_style()
    label = int(round(confidence * 100))
    methods = ["psp", "pca", "pca_corr", "pwg", "heston"]
    names = {"psp": "PSP", "pca": "PCA", "pca_corr": "PCA-C", "pwg": "PWG", "heston": "Heston-HS"}
    colors = {"psp": COLORS["orange"], "pca": COLORS["blue"], "pca_corr": COLORS["gold"], "pwg": COLORS["teal"], "heston": COLORS["red"]}
    fig, axis = plt.subplots(figsize=(13, 5.1), constrained_layout=True)
    pnl = backtest["actual_pnl"] / 1_000
    axis.plot(backtest.index, pnl, color="#B5BEC8", linewidth=.7, alpha=.65, label="Actual P&L")
    for method in methods:
        axis.plot(
            backtest.index,
            -backtest[f"var_{label}_{method}"] / 1_000,
            color=colors[method], linewidth=1.0, alpha=.9, label=names[method],
        )
    axis.axhline(0, color=COLORS["navy"], linewidth=.7)
    axis.set_ylabel("Daily vega P&L / VaR threshold (USD thousands)")
    axis.set_xlabel("Forecast date")
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.legend(ncol=6, loc="lower center", bbox_to_anchor=(.5, 1.01))
    axis.set_title(
        f"{confidence:.0%} VaR paths including correlation-PCA and Heston-filtered history",
        loc="left",
    )
    return _save(fig, path)


def plot_heston_mc_diagnostics(
    physical: pd.DataFrame,
    forecasts: pd.DataFrame,
    numerical: pd.DataFrame,
    path: Path,
) -> Path:
    """Show the P-measure filter, leverage, state tails, and step convergence."""

    apply_style()
    p = physical.copy()
    p["quote_date"] = pd.to_datetime(p["quote_date"])
    f = forecasts.copy()
    f["forecast_date"] = pd.to_datetime(f["forecast_date"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), constrained_layout=True)

    axes[0, 0].plot(p["quote_date"], p["kappa_p"], color=COLORS["teal"], linewidth=.8)
    axes[0, 0].set_title("Trailing physical mean reversion", loc="left")
    axes[0, 0].set_ylabel(r"$\kappa_P$")

    axes[0, 1].plot(p["quote_date"], p["rho_p"], color=COLORS["red"], linewidth=.8)
    axes[0, 1].axhline(0, color=COLORS["gray"], linewidth=.7)
    axes[0, 1].set_title("Physical leverage correlation", loc="left")
    axes[0, 1].set_ylabel(r"$\rho_P$")

    axes[1, 0].plot(
        f["forecast_date"], f["realized_spot_return"], color="#B5BEC8",
        linewidth=.55, label="Realized",
    )
    axes[1, 0].plot(
        f["forecast_date"], f["spot_p_q01"], color=COLORS["red"],
        linewidth=.8, label="Heston-P simulated 1%",
    )
    axes[1, 0].plot(
        f["forecast_date"], f["spot_q_q01"], color=COLORS["blue"],
        linewidth=.7, alpha=.85, label="Heston-Q simulated 1%",
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set_title("One-day SPX return tails", loc="left")
    axes[1, 0].legend(fontsize=8, ncol=3)

    convergence = numerical.groupby(["measure", "steps"])["return_q01"].median().reset_index()
    for measure, color in (("P", COLORS["red"]), ("Q", COLORS["blue"])):
        subset = convergence[convergence["measure"] == measure]
        axes[1, 1].plot(
            subset["steps"], subset["return_q01"], marker="o", color=color,
            label=f"Heston-{measure}",
        )
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].set_xticks([4, 8, 16])
    axes[1, 1].set_xlabel("Projected-Euler substeps per day")
    axes[1, 1].set_ylabel("Median simulated 1% return")
    axes[1, 1].set_title("Discretization convergence", loc="left")
    axes[1, 1].legend()

    for axis in axes.ravel()[:3]:
        axis.xaxis.set_major_locator(mdates.YearLocator(3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Heston joint spot--variance Monte Carlo diagnostics",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_full_revaluation_coverage(diagnostics: pd.DataFrame, path: Path) -> Path:
    apply_style()
    labels = {
        "GBM_PSP_FR": "GBM + PSP-FHS",
        "HESTON_MC_P": "Heston-MC-P",
        "HESTON_MC_Q": "Heston-MC-Q",
    }
    confidences = sorted(diagnostics["confidence"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    colors = [COLORS["orange"], COLORS["red"], COLORS["blue"]]
    for axis, confidence in zip(axes, confidences, strict=True):
        subset = diagnostics[diagnostics["confidence"] == confidence].copy()
        names = [labels[value] for value in subset["method"]]
        axis.bar(names, subset["breach_rate"], color=colors)
        expected = 1.0 - confidence
        axis.axhline(expected, color=COLORS["navy"], linestyle="--", linewidth=1.2)
        for location, row in enumerate(subset.itertuples(index=False)):
            axis.text(
                location,
                row.breach_rate + expected * .045,
                "pass" if not row.conditional_coverage_reject_5pct else "reject",
                ha="center",
                fontsize=8,
                weight="bold",
                color=(
                    COLORS["teal"]
                    if not row.conditional_coverage_reject_5pct
                    else COLORS["red"]
                ),
            )
        axis.set_title(f"{confidence:.0%} VaR", loc="left")
        axis.set_ylabel("Observed breach rate")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.tick_params(axis="x", rotation=24)
    fig.suptitle(
        "Full-revaluation VaR: one common delta-hedged fixed-strike portfolio",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)


def plot_full_revaluation_backtest(
    backtest: pd.DataFrame,
    path: Path,
    confidence: float = .95,
) -> Path:
    apply_style()
    label = int(round(confidence * 100))
    methods = ["gbm_psp_fr", "heston_mc_p", "heston_mc_q"]
    names = {
        "gbm_psp_fr": "GBM + PSP-FHS",
        "heston_mc_p": "Heston-MC-P",
        "heston_mc_q": "Heston-MC-Q",
    }
    colors = {
        "gbm_psp_fr": COLORS["orange"],
        "heston_mc_p": COLORS["red"],
        "heston_mc_q": COLORS["blue"],
    }
    fig, axis = plt.subplots(figsize=(13, 5.4))
    axis.plot(
        backtest.index,
        backtest["actual_pnl"] / 1_000,
        color="#AAB4C0",
        linewidth=.65,
        label="Realized full-revaluation P&L",
    )
    for method in methods:
        axis.plot(
            backtest.index,
            -backtest[f"var_{label}_{method}"] / 1_000,
            color=colors[method],
            linewidth=.9,
            label=names[method],
        )
    axis.axhline(0, color=COLORS["navy"], linewidth=.7)
    axis.set_ylabel("Daily P&L / VaR threshold (USD thousands)")
    axis.set_xlabel("Realization date")
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.suptitle(
        f"{confidence:.0%} VaR with joint spot--variance simulation and full repricing",
        x=.07,
        y=.985,
        ha="left",
        fontsize=14,
        weight="bold",
        color=COLORS["navy"],
    )
    fig.legend(
        handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(.58, .945),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, .88))
    return _save(fig, path)


def plot_full_revaluation_regimes(subperiod: pd.DataFrame, path: Path) -> Path:
    apply_style()
    subset = subperiod.copy()
    subset["method"] = subset["method"].replace(
        {
            "GBM_PSP_FR": "GBM + PSP-FHS",
            "HESTON_MC_P": "Heston-MC-P",
            "HESTON_MC_Q": "Heston-MC-Q",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis_number, (axis, confidence) in enumerate(
        zip(axes, sorted(subset["confidence"].unique()), strict=True)
    ):
        pivot = subset[subset["confidence"] == confidence].pivot(
            index="method", columns="period", values="coverage_ratio"
        )
        order = [
            value
            for value in (
                "Full OOS",
                "GFC (2007-2009)",
                "Expansion (2010-2019)",
                "COVID (2020-2021)",
            )
            if value in pivot
        ]
        sns.heatmap(
            pivot[order],
            annot=True,
            fmt=".2f",
            center=1.0,
            cmap="vlag_r",
            vmin=.35,
            vmax=1.65,
            cbar=False,
            ax=axis,
        )
        axis.set_title(f"{confidence:.0%} VaR", loc="left")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=34)
        if axis_number == 0:
            axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
        else:
            axis.set_yticklabels([])
    fig.suptitle(
        "Full-revaluation coverage across crisis regimes (observed / expected)",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    return _save(fig, path)

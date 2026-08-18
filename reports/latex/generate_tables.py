"""Generate compact LaTeX result-table fragments from verified CSV outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "long_horizon_2005_2021"
TABLES = RUN / "tables"
GENERATED = HERE / "generated"


def _write(name: str, body: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text(body.strip() + "\n", encoding="utf-8")


def _p(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def _pct(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}%}".replace("%", r"\%")


def main() -> None:
    diagnostics = pd.read_csv(TABLES / "backtest_diagnostics.csv")
    rows = []
    for row in diagnostics.itertuples(index=False):
        verdict = "Pass" if not row.conditional_coverage_reject_5pct else "Reject"
        rows.append(
            f"{row.method} & {_pct(row.confidence)} & {row.breaches} / {row.expected_breaches:.1f} "
            f"& {_pct(row.breach_rate, 2)} & {_p(row.kupiec_p_value)} & "
            f"{_p(row.independence_p_value)} & {_p(row.conditional_coverage_p_value)} "
            f"& {row.mean_quantile_loss:,.1f} & {verdict} \\\\"
        )
    _write("var_diagnostics.tex", "\n".join(rows) + "\n\\bottomrule")

    ranking = pd.read_csv(TABLES / "var_model_ranking.csv")
    rows = []
    for row in ranking.itertuples(index=False):
        rows.append(
            f"{_pct(row.confidence)} & {row.method} & {_pct(row.absolute_coverage_error, 3)} "
            f"& {row.mean_quantile_loss:,.1f} & {row.coverage_rank:.0f} "
            f"& {row.quantile_loss_rank:.0f} & {row.overall_rank:.0f} \\\\"
        )
    _write("var_ranking.tex", "\n".join(rows) + "\n\\bottomrule")

    dm = pd.read_csv(TABLES / "diebold_mariano_comparison.csv")
    rows = []
    for row in dm.itertuples(index=False):
        preferred = row.method_a if row.mean_loss_difference < 0 else row.method_b
        if row.p_value >= 0.05:
            preferred = "No difference"
        rows.append(
            f"{_pct(row.confidence)} & {row.method_a}--{row.method_b} "
            f"& {row.statistic:.3f} & {_p(row.p_value)} "
            f"& {row.mean_loss_difference:,.1f} & {preferred} \\\\"
        )
    _write("dm_results.tex", "\n".join(rows) + "\n\\bottomrule")

    methods = pd.read_csv(TABLES / "surface_method_comparison.csv")
    aggregate = methods.groupby("method").agg(
        validation_dates=("quote_date", "nunique"),
        median_rmse=("iv_rmse", "median"),
        mean_rmse=("iv_rmse", "mean"),
        median_log_iv_mae=("log_iv_mae", "median"),
        median_r2=("r_squared", "median"),
    ).sort_values("median_rmse")
    rows = []
    for method, row in aggregate.iterrows():
        method_label = method.replace("_", r"\_")
        rows.append(
            f"{method_label} & {int(row.validation_dates)} "
            f"& {row.median_rmse:.5f} & {row.mean_rmse:.5f} "
            f"& {row.median_log_iv_mae:.5f} & {row.median_r2:.4f} \\\\"
        )
    _write("surface_methods.tex", "\n".join(rows) + "\n\\bottomrule")

    pca = pd.read_csv(TABLES / "pca_explained_variance.csv")
    rows = []
    for row in pca.itertuples(index=False):
        rows.append(
            f"{row.component} & {_pct(row.explained_variance_ratio, 2)} "
            f"& {_pct(row.cumulative_explained_variance, 2)} \\\\"
        )
    _write("pca_variance.tex", "\n".join(rows) + "\n\\bottomrule")

    exposures = pd.read_csv(TABLES / "vega_exposures.csv")
    exposure_summary = exposures.groupby("maturity_days")["vega_exposure"].agg(["sum", "min", "max"])
    total = exposures["vega_exposure"].sum()
    rows = []
    for maturity, row in exposure_summary.iterrows():
        rows.append(
            f"{int(maturity)} & {row['sum']:,.0f} & {_pct(row['sum'] / total, 1)} "
            f"& {row['min']:,.0f} & {row['max']:,.0f} \\\\"
        )
    _write("vega_exposure.tex", "\n".join(rows) + "\n\\bottomrule")

    interpretation = pd.read_csv(TABLES / "pca_interpretation_diagnostics.csv")
    selected = interpretation[
        (interpretation["mode"] == "Correlation")
        & interpretation["component"].isin(["PC1", "PC2", "PC3", "PC5"])
    ]
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(
            f"{row.component} & {_pct(row.explained_variance_ratio, 2)} "
            f"& {row.best_template} & {row.best_abs_cosine:.3f} \\\\"
        )
    _write("pca_interpretation.tex", "\n".join(rows) + "\n\\bottomrule")

    extended = pd.read_csv(TABLES / "extended_backtest_diagnostics.csv")
    additions = extended[extended["method"].isin(["PCA_CORR", "HESTON"])]
    rows = []
    for row in additions.itertuples(index=False):
        label = "PCA-C" if row.method == "PCA_CORR" else "Heston-HS"
        verdict = "Pass" if not row.conditional_coverage_reject_5pct else "Reject"
        rows.append(
            f"{label} & {_pct(row.confidence)} & {row.breaches} / {row.expected_breaches:.1f} "
            f"& {_pct(row.breach_rate, 2)} & {_p(row.kupiec_p_value)} & "
            f"{_p(row.independence_p_value)} & {_p(row.conditional_coverage_p_value)} "
            f"& {row.mean_quantile_loss:,.1f} & {verdict} \\\\"
        )
    _write("extended_additions.tex", "\n".join(rows) + "\n\\bottomrule")

    heston_parameters = pd.read_csv(TABLES / "heston_daily_parameters.csv")
    parameter_specs = [
        ("$\\sqrt{v_0}$", "v0", True),
        ("$\\kappa$", "kappa", False),
        ("$\\sqrt{\\theta}$", "theta", True),
        ("$\\xi$", "xi", False),
        ("$\\rho$", "rho", False),
    ]
    rows = []
    for label, column, root in parameter_specs:
        values = heston_parameters[column].clip(lower=0).pow(.5) if root else heston_parameters[column]
        formatter = (lambda value: _pct(value, 1)) if root else (lambda value: f"{value:.3f}")
        rows.append(
            f"{label} & {formatter(values.quantile(.10))} & {formatter(values.median())} "
            f"& {formatter(values.quantile(.90))} \\\\"
        )
    _write("heston_parameters.tex", "\n".join(rows) + "\n\\bottomrule")

    extended_dm = pd.read_csv(TABLES / "extended_diebold_mariano_comparison.csv")
    heston_dm = extended_dm[
        (extended_dm["method_a"] == "PSP") & (extended_dm["method_b"] == "HESTON")
    ]
    rows = []
    for row in heston_dm.itertuples(index=False):
        rows.append(
            f"{_pct(row.confidence)} & {row.statistic:.3f} & {_p(row.p_value)} "
            f"& {row.mean_loss_difference:,.1f} \\\\"
        )
    _write("heston_dm.tex", "\n".join(rows) + "\n\\bottomrule")

    full_revaluation = pd.read_csv(TABLES / "full_revaluation_diagnostics.csv")
    full_labels = {
        "GBM_PSP_FR": "GBM+PSP-FHS",
        "HESTON_MC_P": "Heston-MC-P",
        "HESTON_MC_Q": "Heston-MC-Q",
    }
    rows = []
    for row in full_revaluation.itertuples(index=False):
        verdict = "Pass" if not row.conditional_coverage_reject_5pct else "Reject"
        rows.append(
            f"{full_labels[row.method]} & {_pct(row.confidence)} "
            f"& {row.breaches} / {row.expected_breaches:.1f} "
            f"& {_pct(row.breach_rate, 2)} & {row.mean_var:,.0f} "
            f"& {row.mean_es:,.0f} & {_p(row.kupiec_p_value)} "
            f"& {_p(row.independence_p_value)} "
            f"& {_p(row.conditional_coverage_p_value)} "
            f"& {row.mean_quantile_loss:,.1f} & {verdict} \\\\"
        )
    _write(
        "full_revaluation_diagnostics.tex",
        "\n".join(rows) + "\n\\bottomrule",
    )

    full_dm = pd.read_csv(TABLES / "full_revaluation_diebold_mariano.csv")
    rows = []
    for row in full_dm.itertuples(index=False):
        method_a = full_labels[row.method_a]
        method_b = full_labels[row.method_b]
        preferred = method_a if row.mean_loss_difference < 0 else method_b
        if row.p_value >= 0.05:
            preferred = "No difference"
        rows.append(
            f"{_pct(row.confidence)} & {method_a}--{method_b} "
            f"& {row.statistic:.3f} & {_p(row.p_value)} "
            f"& {row.mean_loss_difference:,.1f} & {preferred} \\\\"
        )
    _write("full_revaluation_dm.tex", "\n".join(rows) + "\n\\bottomrule")

    physical = pd.read_csv(TABLES / "heston_physical_parameters.csv")
    parameter_specs = [
        ("$\\mu_P$", "mu_total", None),
        ("$\\kappa_P$", "kappa_p", "kappa_bound"),
        ("$\\theta_P$", "theta_p", "theta_bound"),
        ("$\\xi_P$", "xi_p", "xi_bound"),
        ("$\\rho_P$", "rho_p", "rho_bound"),
    ]
    rows = []
    for label, column, bound_column in parameter_specs:
        values = physical[column]
        bound_share = "--" if bound_column is None else _pct(physical[bound_column].mean(), 1)
        rows.append(
            f"{label} & {values.quantile(.10):.3f} & {values.median():.3f} "
            f"& {values.quantile(.90):.3f} & {bound_share} \\\\"
        )
    _write("physical_heston_parameters.tex", "\n".join(rows) + "\n\\bottomrule")

    numerical = pd.read_csv(TABLES / "heston_mc_numerical_robustness.csv")
    simulation_numerical = numerical[numerical["measure"].isin(["P", "Q"])]
    rows = []
    for (measure, steps), group in simulation_numerical.groupby(["measure", "steps"]):
        rows.append(
            f"Heston-{measure} & {int(steps)} & {_pct(group['return_q01'].median(), 3)} "
            f"& {_pct(group['mean_return_error'].abs().max(), 3)} "
            f"& {_pct(group['zero_variance_share'].max(), 3)} \\\\"
        )
    _write("heston_mc_numerical.tex", "\n".join(rows) + "\n\\bottomrule")

    summary = json.loads((RUN / "run_summary.json").read_text(encoding="utf-8"))
    heston_summary = pd.read_csv(TABLES / "heston_summary.csv").iloc[0]
    correlation_pc = selected.set_index("component")
    heston_diagnostics = extended[extended["method"] == "HESTON"].set_index("confidence")
    full_by_method = {
        method: group.set_index("confidence")
        for method, group in full_revaluation.groupby("method")
    }
    full_forecasts = pd.read_csv(TABLES / "full_revaluation_forecast_diagnostics.csv")
    pricing_grid = numerical[numerical["measure"] == "PricingGrid"]
    p_numerical = numerical[numerical["measure"] == "P"]
    macros = "\n".join(
        [
            rf"\newcommand{{\RawRows}}{{{summary['option_rows']:,}}}",
            rf"\newcommand{{\EligibleRows}}{{{summary['eligible_option_rows']:,}}}",
            rf"\newcommand{{\SurfaceDates}}{{{summary['surface_dates']:,}}}",
            rf"\newcommand{{\ExtrapolatedShare}}{{{_pct(summary['extrapolated_surface_grid_share'], 2)}}}",
            rf"\newcommand{{\PCAExplained}}{{{_pct(summary['pca_cumulative_explained_variance'], 2)}}}",
            rf"\newcommand{{\CorrelationPCOneCosine}}{{{correlation_pc.loc['PC1', 'best_abs_cosine']:.3f}}}",
            rf"\newcommand{{\CorrelationPCTwoCosine}}{{{correlation_pc.loc['PC2', 'best_abs_cosine']:.3f}}}",
            rf"\newcommand{{\CorrelationPCThreeCosine}}{{{correlation_pc.loc['PC3', 'best_abs_cosine']:.3f}}}",
            rf"\newcommand{{\CorrelationPCFiveCosine}}{{{correlation_pc.loc['PC5', 'best_abs_cosine']:.3f}}}",
            rf"\newcommand{{\HestonMedianRMSE}}{{{_pct(heston_summary['median_daily_iv_rmse'], 2)}}}",
            rf"\newcommand{{\HestonPninetyRMSE}}{{{_pct(heston_summary['p90_daily_iv_rmse'], 2)}}}",
            rf"\newcommand{{\HestonFellerShare}}{{{_pct(heston_summary['feller_satisfaction_rate'], 1)}}}",
            rf"\newcommand{{\HestonXiBoundShare}}{{{_pct(heston_summary['xi_upper_bound_rate'], 1)}}}",
            rf"\newcommand{{\HestonNinetyBreaches}}{{{int(heston_diagnostics.loc[.90, 'breaches'])}}}",
            rf"\newcommand{{\HestonNinetyFiveBreaches}}{{{int(heston_diagnostics.loc[.95, 'breaches'])}}}",
            rf"\newcommand{{\HestonNinetyNineBreaches}}{{{int(heston_diagnostics.loc[.99, 'breaches'])}}}",
            rf"\newcommand{{\FullGBMNinetyBreaches}}{{{int(full_by_method['GBM_PSP_FR'].loc[.90, 'breaches'])}}}",
            rf"\newcommand{{\FullGBMNinetyFiveBreaches}}{{{int(full_by_method['GBM_PSP_FR'].loc[.95, 'breaches'])}}}",
            rf"\newcommand{{\FullGBMNinetyNineBreaches}}{{{int(full_by_method['GBM_PSP_FR'].loc[.99, 'breaches'])}}}",
            rf"\newcommand{{\FullHestonPNinetyBreaches}}{{{int(full_by_method['HESTON_MC_P'].loc[.90, 'breaches'])}}}",
            rf"\newcommand{{\FullHestonPNinetyFiveBreaches}}{{{int(full_by_method['HESTON_MC_P'].loc[.95, 'breaches'])}}}",
            rf"\newcommand{{\FullHestonPNinetyNineBreaches}}{{{int(full_by_method['HESTON_MC_P'].loc[.99, 'breaches'])}}}",
            rf"\newcommand{{\FullHestonPNinetyFiveCCP}}{{{_p(full_by_method['HESTON_MC_P'].loc[.95, 'conditional_coverage_p_value'])}}}",
            rf"\newcommand{{\FullPairedCorrelation}}{{{full_forecasts['paired_spot_surface_level_correlation'].median():.3f}}}",
            rf"\newcommand{{\FullExtrapolatedShare}}{{{_pct(full_forecasts['surface_extrapolated_share'].mean(), 1)}}}",
            rf"\newcommand{{\CurveFallbackCount}}{{{int((full_forecasts['zero_curve_stale_days'] > 0).sum())}}}",
            rf"\newcommand{{\PhysicalKappaBoundShare}}{{{_pct(physical['kappa_bound'].mean(), 1)}}}",
            rf"\newcommand{{\HestonMCMeanError}}{{{_pct(p_numerical['mean_return_error'].abs().max(), 3)}}}",
            rf"\newcommand{{\HestonPriceGridError}}{{{pricing_grid['price_grid_max_abs_error'].max() * 10_000:.2f}}}",
        ]
    )
    _write("result_macros.tex", macros)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Add PCA robustness and a real-time Heston surface-risk benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import pandas as pd

from spx_risk.analysis.heston import HestonSurfaceResult, build_heston_surface_history
from spx_risk.analysis.heston_mc import run_heston_full_revaluation
from spx_risk.analysis.pca import run_pca
from spx_risk.analysis.risk import (
    diebold_mariano_loss_comparison,
    evaluate_var_backtest,
    rank_var_models,
    run_var_backtest,
)
from spx_risk.analysis.robustness import (
    heston_risk_backtest,
    pca_interpretation_diagnostics,
    pca_subperiod_stability,
    psp_specification_robustness,
    subperiod_backtest_diagnostics,
)
from spx_risk.config import load_config
from spx_risk.visualization.extended import (
    plot_full_revaluation_backtest,
    plot_full_revaluation_coverage,
    plot_full_revaluation_regimes,
    plot_extended_var_backtest,
    plot_extended_var_coverage,
    plot_heston_mc_diagnostics,
    plot_heston_parameters,
    plot_heston_surface_comparison,
    plot_pca_diagnostic_dashboard,
    plot_pca_standardization_3d,
    plot_robustness_dashboard,
)


def _load_heston_cache(
    surface_path: Path,
    parameters_path: Path,
    calibration_path: Path,
    metadata_path: Path,
    expected_signature: str,
) -> HestonSurfaceResult | None:
    if not all(
        path.is_file()
        for path in (surface_path, parameters_path, calibration_path, metadata_path)
    ):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != expected_signature:
        return None
    surface = pd.read_parquet(surface_path)
    surface.index = pd.to_datetime(surface.index)
    surface.index.name = "quote_date"
    parameters = pd.read_csv(parameters_path, parse_dates=["quote_date", "calibration_date"])
    calibration = pd.read_csv(calibration_path, parse_dates=["quote_date"])
    return HestonSurfaceResult(surface, parameters, calibration)


def _save_heston(
    result: HestonSurfaceResult,
    surface_path: Path,
    parameters_path: Path,
    calibration_path: Path,
    metadata_path: Path,
    signature: str,
) -> None:
    result.surface_matrix.to_parquet(surface_path)
    result.parameters.to_csv(parameters_path, index=False)
    result.calibration.to_csv(calibration_path, index=False)
    metadata_path.write_text(
        json.dumps({"signature": signature}, indent=2), encoding="utf-8"
    )


def _heston_signature(config, market: pd.DataFrame) -> str:
    # Surface calibration and scenario simulation have separate caches.  Do not
    # invalidate 4,280 daily calibrations when only a Monte Carlo grid changes.
    calibration_settings = {
        name: getattr(config.heston, name)
        for name in (
            "enabled",
            "recalibration_frequency",
            "integration_nodes",
            "integration_upper",
            "max_nfev",
        )
    }
    payload = {
        "implementation_version": 2,
        "heston": calibration_settings,
        "first_date": str(market.index.min()),
        "last_date": str(market.index.max()),
        "shape": market.shape,
        "columns": [tuple(map(float, column)) for column in market.columns],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _heston_mc_signature(config, market: pd.DataFrame, heston_signature: str) -> str:
    payload = {
        "implementation_version": 2,
        "heston_surface_signature": heston_signature,
        "simulation": {
            name: getattr(config.heston, name)
            for name in (
                "simulation_steps",
                "simulation_integration_nodes",
                "variance_grid_size",
                "strike_grid_size",
                "physical_window",
                "integration_upper",
            )
        },
        "risk": {
            "confidence_levels": config.risk.confidence_levels,
            "rolling_window": config.risk.rolling_window,
            "minimum_history": config.risk.minimum_history,
            "scenarios": config.risk.scenarios,
            "total_vega": config.risk.total_vega,
            "psp_decay": config.risk.psp_decay,
        },
        "first_date": str(market.index.min()),
        "last_date": str(market.index.max()),
        "shape": market.shape,
        "columns": [tuple(map(float, column)) for column in market.columns],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _rename_pca_corr(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={column: column.replace("_pca", "_pca_corr") for column in frame.columns}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/long_horizon.yaml")
    parser.add_argument("--force-heston", action="store_true")
    parser.add_argument("--force-heston-mc", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = config.output.root
    tables = output / "tables"
    processed = output / "processed"
    figures = output / "figures"
    for directory in (tables, processed, figures):
        directory.mkdir(parents=True, exist_ok=True)

    print("Loading the verified 2005-2021 surface history ...", flush=True)
    surface_long = pd.read_parquet(processed / "fitted_surface.parquet")
    market = surface_long.pivot(
        index="quote_date",
        columns=["maturity_days", "moneyness"],
        values="predicted_iv",
    ).sort_index()
    market = market.interpolate(limit_direction="both").dropna(axis=0, how="any")
    changes = market.diff().dropna()

    print("Auditing covariance and correlation PCA geometry ...", flush=True)
    pca_diagnostics, loading_frames, reconstruction = pca_interpretation_diagnostics(changes)
    stability = pca_subperiod_stability(changes)
    pca_diagnostics.to_csv(tables / "pca_interpretation_diagnostics.csv", index=False)
    reconstruction.to_csv(tables / "pca_reconstruction_robustness.csv", index=False)
    stability.to_csv(tables / "pca_subperiod_stability.csv", index=False)

    heston_surface_path = processed / "heston_surface.parquet"
    heston_parameters_path = tables / "heston_daily_parameters.csv"
    heston_calibration_path = tables / "heston_calibration_diagnostics.csv"
    heston_metadata_path = processed / "heston_cache_metadata.json"
    heston_signature = _heston_signature(config, market)
    heston = None if args.force_heston else _load_heston_cache(
        heston_surface_path,
        heston_parameters_path,
        heston_calibration_path,
        heston_metadata_path,
        heston_signature,
    )
    if heston is None:
        print(
            "Calibrating the real-time Heston surface history ...",
            flush=True,
        )
        heston = build_heston_surface_history(
            market,
            recalibration_frequency=config.heston.recalibration_frequency,
            integration_nodes=config.heston.integration_nodes,
            integration_upper=config.heston.integration_upper,
            max_nfev=config.heston.max_nfev,
        )
        _save_heston(
            heston,
            heston_surface_path,
            heston_parameters_path,
            heston_calibration_path,
            heston_metadata_path,
            heston_signature,
        )
    else:
        print("Using the cached Heston surface history ...", flush=True)

    heston_summary = pd.DataFrame(
        [
            {
                "surface_dates": len(heston.surface_matrix),
                "calibration_dates": len(heston.calibration),
                "calibration_success_rate": heston.calibration["success"].mean(),
                "median_calibration_iv_rmse": heston.calibration["iv_rmse"].median(),
                "median_daily_iv_rmse": heston.parameters["daily_iv_rmse"].median(),
                "p90_daily_iv_rmse": heston.parameters["daily_iv_rmse"].quantile(.90),
                "feller_satisfaction_rate": heston.parameters["feller_satisfied"].mean(),
                "xi_upper_bound_rate": (heston.parameters["xi"] >= 1.4999).mean(),
                "recalibration_frequency": config.heston.recalibration_frequency,
                "integration_nodes_calibration": config.heston.integration_nodes,
                "integration_nodes_final": 160,
            }
        ]
    )
    heston_summary.to_csv(tables / "heston_summary.csv", index=False)

    print("Running correlation-PCA and Heston-filtered out-of-sample VaR ...", flush=True)
    correlation_config = replace(
        config,
        pca=replace(config.pca, standardize=True),
        risk=replace(config.risk, methods=("pca",)),
    )
    correlation_pca = run_pca(surface_long, correlation_config)
    correlation_risk = run_var_backtest(correlation_pca, correlation_config)
    correlation_backtest = _rename_pca_corr(correlation_risk.backtest.drop(columns="actual_pnl"))
    heston_risk = heston_risk_backtest(heston, changes, config)

    existing = pd.read_csv(tables / "var_backtest.csv", parse_dates=["quote_date"])
    existing = existing.set_index("quote_date").sort_index()
    extended = existing.join(correlation_backtest, how="left")
    extended = extended.join(heston_risk.backtest.drop(columns="actual_pnl"), how="left")
    methods = ("psp", "pca", "pca_corr", "pwg", "heston")
    extended_config = replace(config, risk=replace(config.risk, methods=methods))
    diagnostics = evaluate_var_backtest(extended, extended_config)
    ranking = rank_var_models(diagnostics)
    extended.to_csv(tables / "extended_var_backtest.csv", index_label="quote_date")
    diagnostics.to_csv(tables / "extended_backtest_diagnostics.csv", index=False)
    ranking.to_csv(tables / "extended_var_model_ranking.csv", index=False)

    dm_rows: list[dict[str, object]] = []
    for confidence in config.risk.confidence_levels:
        for first, second in combinations(methods, 2):
            dm_rows.append(
                {
                    "confidence": confidence,
                    "method_a": first.upper(),
                    "method_b": second.upper(),
                    **diebold_mariano_loss_comparison(
                        extended, confidence, first, second
                    ),
                }
            )
    pd.DataFrame(dm_rows).to_csv(
        tables / "extended_diebold_mariano_comparison.csv", index=False
    )

    print("Running joint Heston spot-variance Monte Carlo and full repricing ...", flush=True)
    mc_methods = ("gbm_psp_fr", "heston_mc_p", "heston_mc_q")
    mc_config = replace(config, risk=replace(config.risk, methods=mc_methods))
    mc_signature = _heston_mc_signature(config, market, heston_signature)
    mc_metadata_path = processed / "heston_mc_cache_metadata.json"
    mc_backtest_path = tables / "full_revaluation_backtest.csv"
    mc_physical_path = tables / "heston_physical_parameters.csv"
    mc_forecast_path = tables / "full_revaluation_forecast_diagnostics.csv"
    mc_numerical_path = tables / "heston_mc_numerical_robustness.csv"
    cache_valid = (
        not args.force_heston_mc
        and all(
            path.is_file()
            for path in (
                mc_metadata_path,
                mc_backtest_path,
                mc_physical_path,
                mc_forecast_path,
                mc_numerical_path,
            )
        )
        and json.loads(mc_metadata_path.read_text(encoding="utf-8")).get("signature")
        == mc_signature
    )
    if cache_valid:
        print("Using the cached Heston Monte Carlo full-revaluation backtest ...", flush=True)
        mc_backtest = pd.read_csv(mc_backtest_path, parse_dates=["quote_date"]).set_index(
            "quote_date"
        )
        mc_physical = pd.read_csv(
            mc_physical_path, parse_dates=["quote_date", "window_start", "window_end"]
        )
        mc_forecast = pd.read_csv(
            mc_forecast_path, parse_dates=["forecast_date", "horizon_date"]
        )
        mc_numerical = pd.read_csv(mc_numerical_path, parse_dates=["quote_date"])
    else:
        mc_result = run_heston_full_revaluation(market, heston, config)
        mc_backtest = mc_result.risk.backtest
        mc_physical = mc_result.physical_parameters
        mc_forecast = mc_result.forecast_diagnostics
        mc_numerical = mc_result.numerical_robustness
        mc_backtest.to_csv(mc_backtest_path, index_label="quote_date")
        mc_physical.to_csv(mc_physical_path, index=False)
        mc_forecast.to_csv(mc_forecast_path, index=False)
        mc_numerical.to_csv(mc_numerical_path, index=False)
        mc_metadata_path.write_text(
            json.dumps({"signature": mc_signature}, indent=2), encoding="utf-8"
        )

    mc_diagnostics = evaluate_var_backtest(mc_backtest, mc_config)
    mc_ranking = rank_var_models(mc_diagnostics)
    mc_subperiod = subperiod_backtest_diagnostics(mc_backtest, mc_methods, config)
    mc_dm_rows: list[dict[str, object]] = []
    for confidence in config.risk.confidence_levels:
        for first, second in combinations(mc_methods, 2):
            mc_dm_rows.append(
                {
                    "confidence": confidence,
                    "method_a": first.upper(),
                    "method_b": second.upper(),
                    **diebold_mariano_loss_comparison(
                        mc_backtest, confidence, first, second
                    ),
                }
            )
    mc_diagnostics.to_csv(tables / "full_revaluation_diagnostics.csv", index=False)
    mc_ranking.to_csv(tables / "full_revaluation_model_ranking.csv", index=False)
    mc_subperiod.to_csv(
        tables / "full_revaluation_subperiod_diagnostics.csv", index=False
    )
    pd.DataFrame(mc_dm_rows).to_csv(
        tables / "full_revaluation_diebold_mariano.csv", index=False
    )

    print("Running PSP specification and crisis-subperiod robustness ...", flush=True)
    psp_diagnostics, _ = psp_specification_robustness(changes, config)
    psp_diagnostics.to_csv(tables / "psp_specification_robustness.csv", index=False)
    subperiod = subperiod_backtest_diagnostics(extended, methods, config)
    subperiod.to_csv(tables / "subperiod_backtest_diagnostics.csv", index=False)

    print("Rendering the extended publication figures ...", flush=True)
    new_figures = [
        plot_pca_standardization_3d(
            loading_frames, pca_diagnostics, figures / "18_pca_standardization_3d.png"
        ),
        plot_pca_diagnostic_dashboard(
            pca_diagnostics,
            reconstruction,
            stability,
            figures / "19_pca_diagnostic_dashboard.png",
        ),
        plot_heston_surface_comparison(
            market, heston.surface_matrix, figures / "20_heston_surface_3d.png"
        ),
        plot_heston_parameters(
            heston.parameters, figures / "21_heston_parameters.png"
        ),
        plot_extended_var_coverage(
            diagnostics, figures / "22_extended_var_coverage.png"
        ),
        plot_robustness_dashboard(
            psp_diagnostics, subperiod, figures / "23_robustness_dashboard.png"
        ),
        plot_extended_var_backtest(
            extended, figures / "24_extended_var_backtest_95.png"
        ),
        plot_heston_mc_diagnostics(
            mc_physical,
            mc_forecast,
            mc_numerical,
            figures / "25_heston_mc_diagnostics.png",
        ),
        plot_full_revaluation_coverage(
            mc_diagnostics, figures / "26_full_revaluation_coverage.png"
        ),
        plot_full_revaluation_backtest(
            mc_backtest, figures / "27_full_revaluation_backtest_95.png"
        ),
        plot_full_revaluation_regimes(
            mc_subperiod, figures / "28_full_revaluation_regimes.png"
        ),
    ]

    summary_path = output / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    corr = pca_diagnostics[
        (pca_diagnostics["mode"] == "Correlation")
        & pca_diagnostics["component"].isin(["PC1", "PC2", "PC3", "PC5"])
    ]
    summary["extended_analysis"] = {
        "methods": list(methods),
        "heston": heston_summary.iloc[0].to_dict(),
        "full_revaluation_methods": list(mc_methods),
        "full_revaluation_forecasts": int(mc_diagnostics["observations"].max()),
        "correlation_pca_canonical_matches": corr[
            ["component", "best_template", "best_abs_cosine"]
        ].to_dict(orient="records"),
        "new_figures": [str(path.relative_to(config.project_root)) for path in new_figures],
    }
    summary["figures"] = [
        str(path.relative_to(config.project_root)) for path in sorted(figures.glob("*.png"))
    ]
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Extended analysis complete: {len(diagnostics)} model-level evaluations, "
        f"{len(new_figures)} new figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refresh multi-level VaR tables and figures without re-downloading WRDS data."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from spx_risk.analysis.risk import (
    RiskResult,
    diebold_mariano_loss_comparison,
    evaluate_var_backtest,
    rank_var_models,
)
from spx_risk.config import load_config
from spx_risk.visualization.plots import (
    plot_var_backtest,
    plot_var_coverage_summary,
    plot_var_murphy_diagrams,
    plot_var_test_heatmap,
)


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/long_horizon.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_root = config.output.root
    tables = output_root / "tables"
    figures = output_root / "figures"

    backtest = pd.read_csv(tables / "var_backtest.csv", parse_dates=["quote_date"])
    backtest = backtest.set_index("quote_date").sort_index()
    exposure_frame = pd.read_csv(tables / "vega_exposures.csv")
    exposure_index = pd.MultiIndex.from_frame(
        exposure_frame[["maturity_days", "moneyness"]]
    )
    exposures = pd.Series(
        exposure_frame["vega_exposure"].to_numpy(float),
        index=exposure_index,
        name="vega_exposure",
    )

    diagnostics = evaluate_var_backtest(backtest, config)
    risk = RiskResult(exposures=exposures, backtest=backtest, diagnostics=diagnostics)
    diagnostics.to_csv(tables / "backtest_diagnostics.csv", index=False)
    rank_var_models(diagnostics).to_csv(tables / "var_model_ranking.csv", index=False)

    dm_rows: list[dict[str, object]] = []
    for confidence in config.risk.confidence_levels:
        for method_a, method_b in combinations(config.risk.methods, 2):
            dm_rows.append(
                {
                    "confidence": confidence,
                    "method_a": method_a.upper(),
                    "method_b": method_b.upper(),
                    **diebold_mariano_loss_comparison(
                        backtest, confidence, method_a, method_b
                    ),
                }
            )
    pd.DataFrame(dm_rows).to_csv(tables / "diebold_mariano_comparison.csv", index=False)

    confidences = sorted(config.risk.confidence_levels)
    primary = min(confidences, key=lambda value: abs(value - 0.95))
    primary_label = int(round(primary * 100))
    risk_figures = [
        plot_var_backtest(
            risk, figures / f"06_var_backtest_{primary_label}.png", confidence=primary
        )
    ]
    for number, confidence in enumerate(
        (value for value in confidences if value != primary), start=13
    ):
        label = int(round(confidence * 100))
        risk_figures.append(
            plot_var_backtest(
                risk, figures / f"{number:02d}_var_backtest_{label}.png", confidence=confidence
            )
        )
    summary_number = 13 + len(confidences) - 1
    risk_figures.extend(
        [
            plot_var_coverage_summary(
                risk, figures / f"{summary_number:02d}_var_coverage_summary.png"
            ),
            plot_var_test_heatmap(
                risk, figures / f"{summary_number + 1:02d}_var_test_heatmap.png"
            ),
            plot_var_murphy_diagrams(
                risk, figures / f"{summary_number + 2:02d}_var_murphy_diagrams.png"
            ),
        ]
    )

    obsolete_figure = figures / "06_var_backtest.png"
    if obsolete_figure.is_file():
        obsolete_figure.unlink()

    summary_path = output_root / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["confidence_levels"] = list(confidences)
    summary["var_evaluation_tests"] = [
        "Kupiec unconditional coverage",
        "Christoffersen independence",
        "Christoffersen conditional coverage",
        "Diebold-Mariano HAC quantile-loss comparison",
        "Murphy diagrams",
    ]
    for confidence in confidences:
        label = int(round(confidence * 100))
        for method in config.risk.methods:
            summary[f"{method}_{label}_breaches"] = int(
                backtest[f"breach_{label}_{method}"].dropna().astype(bool).sum()
            )
    summary.setdefault("tables", {})["var_model_ranking"] = _relative(
        tables / "var_model_ranking.csv", config.project_root
    )
    summary["figures"] = [
        _relative(path, config.project_root) for path in sorted(figures.glob("*.png"))
    ]
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Refreshed {len(diagnostics)} VaR evaluations and {len(risk_figures)} risk figures.")


if __name__ == "__main__":
    main()

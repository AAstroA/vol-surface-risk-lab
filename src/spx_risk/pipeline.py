"""End-to-end orchestration and artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import pandas as pd

from spx_risk.analysis.pca import PCAResult, run_pca
from spx_risk.analysis.risk import (
    RiskResult,
    diebold_mariano_loss_comparison,
    rank_var_models,
    run_var_backtest,
)
from spx_risk.analysis.surface import SurfaceFit, fit_daily_surfaces, prepare_options
from spx_risk.config import AppConfig
from spx_risk.data.demo import generate_demo_dataset
from spx_risk.data.wrds import (
    WRDSDataset,
    WRDSOptionMetricsClient,
    annual_cache_directory,
    cache_directory,
    load_dataset,
    load_metadata,
    partition_is_complete,
    save_dataset,
)
from spx_risk.visualization.plots import create_all_plots


@dataclass(frozen=True)
class PipelineResult:
    dataset: WRDSDataset
    surface_fit: SurfaceFit
    pca: PCAResult
    risk: RiskResult
    output_root: Path
    figures: tuple[Path, ...]


def _portable_artifact_path(path: Path, project_root: Path) -> str:
    """Prefer a project-relative path, while allowing an explicit external output root."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def download_wrds(config: AppConfig) -> WRDSDataset:
    with WRDSOptionMetricsClient(config) as client:
        dataset = client.download()
    save_dataset(dataset, cache_directory(config))
    return dataset


def download_wrds_partitions(config: AppConfig, *, force: bool = False) -> list[Path]:
    """Download a long interval year by year, preserving every completed year."""
    years = range(config.data.start_date.year, config.data.end_date.year + 1)
    required = [annual_cache_directory(config, year) for year in years]
    pending = [
        (year, directory)
        for year, directory in zip(years, required, strict=True)
        if force or not partition_is_complete(directory)
    ]
    if not pending:
        return required

    with WRDSOptionMetricsClient(config) as client:
        secids = client._spx_secids()
        for year, directory in pending:
            print(f"Downloading WRDS OptionMetrics partition {year} ...", flush=True)
            dataset = client.download_year(year, secids)
            save_dataset(dataset, directory)
            print(
                f"Cached {year}: {len(dataset.options):,} option rows at {directory}",
                flush=True,
            )
    return required


def resolve_dataset(config: AppConfig, *, force_download: bool = False) -> WRDSDataset:
    if config.data.source == "demo":
        return generate_demo_dataset(config)
    if config.data.source != "wrds":
        raise ValueError(f"Unknown data source: {config.data.source}")
    directory = cache_directory(config)
    if force_download or not (directory / "options.parquet").exists():
        return download_wrds(config)
    return load_dataset(directory)


def _analysis_signature(config: AppConfig) -> str:
    payload = {
        "pipeline_version": 3,
        "option_types": config.data.option_types,
        "filters": asdict(config.filters),
        "surface": asdict(config.surface),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _analysis_partition_directory(config: AppConfig, year: int) -> Path:
    return (
        config.project_root
        / "data"
        / "interim"
        / config.project.name
        / _analysis_signature(config)
        / str(year)
    )


def _analysis_partition_complete(directory: Path) -> bool:
    return all(
        (directory / name).is_file()
        for name in (
            "surface.parquet",
            "diagnostics.parquet",
            "method_comparison.parquet",
            "comparison_surfaces.parquet",
            "coverage.parquet",
        )
    )


def _save_surface_partition(surface_fit: SurfaceFit, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    surface_fit.surface.to_parquet(directory / "surface.parquet", index=False)
    surface_fit.diagnostics.to_parquet(directory / "diagnostics.parquet", index=False)
    surface_fit.method_comparison.to_parquet(
        directory / "method_comparison.parquet", index=False
    )
    surface_fit.comparison_surfaces.to_parquet(
        directory / "comparison_surfaces.parquet", index=False
    )
    surface_fit.coverage.to_parquet(directory / "coverage.parquet", index=False)


def _load_surface_partition(directory: Path) -> SurfaceFit:
    return SurfaceFit(
        observations=pd.DataFrame(),
        surface=pd.read_parquet(directory / "surface.parquet"),
        diagnostics=pd.read_parquet(directory / "diagnostics.parquet"),
        method_comparison=pd.read_parquet(directory / "method_comparison.parquet"),
        comparison_surfaces=pd.read_parquet(directory / "comparison_surfaces.parquet"),
        coverage=pd.read_parquet(directory / "coverage.parquet"),
    )


def _combine_surface_partitions(partitions: list[SurfaceFit]) -> SurfaceFit:
    def combine(attribute: str) -> pd.DataFrame:
        frames = [getattr(partition, attribute) for partition in partitions]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    comparison_surfaces = combine("comparison_surfaces")
    if not comparison_surfaces.empty:
        latest = comparison_surfaces["quote_date"].max()
        comparison_surfaces = comparison_surfaces[
            comparison_surfaces["quote_date"] == latest
        ].copy()
    surface = combine("surface")
    diagnostics = combine("diagnostics")
    method_comparison = combine("method_comparison")
    coverage = combine("coverage")
    if not surface.empty:
        surface = surface.sort_values(["quote_date", "maturity_days", "moneyness"])
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("quote_date")
    if not method_comparison.empty:
        method_comparison = method_comparison.sort_values(["quote_date", "method"])
    if not coverage.empty:
        coverage = coverage.sort_values("quote_date")
    return SurfaceFit(
        observations=pd.DataFrame(),
        surface=surface,
        diagnostics=diagnostics,
        method_comparison=method_comparison,
        comparison_surfaces=comparison_surfaces,
        coverage=coverage,
    )


def _run_partitioned_surfaces(
    config: AppConfig, *, force_download: bool = False
) -> tuple[WRDSDataset, SurfaceFit]:
    raw_directories = download_wrds_partitions(config, force=force_download)
    surface_partitions: list[SurfaceFit] = []
    metadata_rows: list[dict[str, object]] = []
    for year, raw_directory in zip(
        range(config.data.start_date.year, config.data.end_date.year + 1),
        raw_directories,
        strict=True,
    ):
        analysis_directory = _analysis_partition_directory(config, year)
        if _analysis_partition_complete(analysis_directory):
            print(f"Using cached surface partition {year} ...", flush=True)
            surface_fit = _load_surface_partition(analysis_directory)
            metadata_rows.append(load_metadata(raw_directory))
        else:
            print(f"Reconstructing daily surfaces for {year} ...", flush=True)
            dataset = load_dataset(raw_directory)
            metadata_rows.append(dataset.metadata)
            observations = prepare_options(
                dataset.options, dataset.underlying, dataset.zero_curve, config
            )
            surface_fit = fit_daily_surfaces(observations, config)
            _save_surface_partition(surface_fit, analysis_directory)
        surface_partitions.append(surface_fit)

    metadata = {
        "source": "WRDS OptionMetrics IvyDB US (annual partitions)",
        "underlying": config.data.underlying_ticker,
        "start_date": config.data.start_date.isoformat(),
        "end_date": config.data.end_date.isoformat(),
        "years": list(range(config.data.start_date.year, config.data.end_date.year + 1)),
        "option_rows": int(sum(int(row.get("option_rows", 0)) for row in metadata_rows)),
        "underlying_rows": int(
            sum(int(row.get("underlying_rows", 0)) for row in metadata_rows)
        ),
        "zero_curve_rows": int(
            sum(int(row.get("zero_curve_rows", 0)) for row in metadata_rows)
        ),
        "interest_rate_source": "OptionMetrics zero curve (zerocd); legacy PwG rate files excluded",
        "dividend_yield_source": "OptionMetrics index dividend yield (idxdvd)",
        "partitioned": True,
        "analysis_signature": _analysis_signature(config),
    }
    empty_dataset = WRDSDataset(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metadata)
    return empty_dataset, _combine_surface_partitions(surface_partitions)


def _write_tables(
    output_root: Path,
    surface_fit: SurfaceFit,
    pca: PCAResult,
    risk: RiskResult,
    config: AppConfig,
) -> dict[str, Path]:
    tables = output_root / "tables"
    processed = output_root / "processed"
    tables.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    paths = {
        "surface_diagnostics": tables / "surface_regression_diagnostics.csv",
        "surface_method_comparison": tables / "surface_method_comparison.csv",
        "data_coverage": tables / "daily_data_coverage.csv",
        "pca_summary": tables / "pca_explained_variance.csv",
        "var_backtest": tables / "var_backtest.csv",
        "backtest_diagnostics": tables / "backtest_diagnostics.csv",
        "var_model_ranking": tables / "var_model_ranking.csv",
        "vega_exposures": tables / "vega_exposures.csv",
        "fitted_surface": processed / "fitted_surface.parquet",
        "clean_options": processed / "clean_options.parquet",
    }
    surface_fit.diagnostics.to_csv(paths["surface_diagnostics"], index=False)
    surface_fit.method_comparison.to_csv(paths["surface_method_comparison"], index=False)
    surface_fit.coverage.to_csv(paths["data_coverage"], index=False)
    pca.explained_variance.to_csv(paths["pca_summary"], index=False)
    risk.backtest.to_csv(paths["var_backtest"], index_label="quote_date")
    risk.diagnostics.to_csv(paths["backtest_diagnostics"], index=False)
    rank_var_models(risk.diagnostics).to_csv(paths["var_model_ranking"], index=False)
    risk.exposures.rename("vega_exposure").reset_index().to_csv(paths["vega_exposures"], index=False)
    surface_fit.surface.to_parquet(paths["fitted_surface"], index=False)
    if surface_fit.observations.empty:
        paths.pop("clean_options")
    else:
        surface_fit.observations.to_parquet(paths["clean_options"], index=False)

    dm_rows = []
    for confidence in config.risk.confidence_levels:
        for method_a, method_b in combinations(config.risk.methods, 2):
            dm_rows.append(
                {
                    "confidence": confidence,
                    "method_a": method_a.upper(),
                    "method_b": method_b.upper(),
                    **diebold_mariano_loss_comparison(
                        risk.backtest, confidence, method_a, method_b
                    ),
                }
            )
    dm_path = tables / "diebold_mariano_comparison.csv"
    pd.DataFrame(dm_rows).to_csv(dm_path, index=False)
    paths["diebold_mariano"] = dm_path
    return paths


def run_pipeline(
    config: AppConfig,
    *,
    dataset: WRDSDataset | None = None,
    force_download: bool = False,
) -> PipelineResult:
    if dataset is None and config.data.source == "wrds" and config.data.partition_by_year:
        dataset, surface_fit = _run_partitioned_surfaces(
            config, force_download=force_download
        )
    else:
        dataset = dataset or resolve_dataset(config, force_download=force_download)
        observations = prepare_options(
            dataset.options, dataset.underlying, dataset.zero_curve, config
        )
        surface_fit = fit_daily_surfaces(observations, config)
    pca = run_pca(surface_fit.surface, config)
    risk = run_var_backtest(pca, config)

    output_root = config.output.root
    output_root.mkdir(parents=True, exist_ok=True)
    table_paths = _write_tables(output_root, surface_fit, pca, risk, config)
    figures = tuple(create_all_plots(surface_fit, pca, risk, output_root / "figures"))

    summary = {
        **dataset.metadata,
        "analysis_start_date": config.data.start_date.isoformat(),
        "analysis_end_date": config.data.end_date.isoformat(),
        "eligible_option_rows": int(surface_fit.coverage["eligible_rows"].sum()),
        "surface_dates": int(surface_fit.surface["quote_date"].nunique()),
        "surface_grid_points": int(
            len(config.surface.maturity_days) * len(config.surface.moneyness_grid)
        ),
        "extrapolated_surface_grid_points": int(
            surface_fit.surface.get("extrapolated", pd.Series(dtype=bool)).sum()
        ),
        "extrapolated_surface_grid_share": float(
            surface_fit.surface.get("extrapolated", pd.Series(dtype=bool)).mean()
        ),
        "pca_components": int(len(pca.explained_variance)),
        "pca_cumulative_explained_variance": float(
            pca.explained_variance["cumulative_explained_variance"].iloc[-1]
        ),
        "surface_method": config.surface.method,
        "surface_comparison_methods": list(config.surface.comparison_methods),
        "risk_methods": list(config.risk.methods),
        "confidence_levels": list(config.risk.confidence_levels),
        "var_evaluation_tests": [
            "Kupiec unconditional coverage",
            "Christoffersen independence",
            "Christoffersen conditional coverage",
            "Diebold-Mariano HAC quantile-loss comparison",
            "Murphy diagrams",
        ],
        **{
            f"{method}_{int(round(confidence * 100))}_breaches": int(
                risk.backtest[
                    f"breach_{int(round(confidence * 100))}_{method}"
                ].dropna().astype(bool).sum()
            )
            for confidence in config.risk.confidence_levels
            for method in config.risk.methods
        },
        "tables": {
            name: _portable_artifact_path(path, config.project_root)
            for name, path in table_paths.items()
        },
        "figures": [
            _portable_artifact_path(path, config.project_root) for path in figures
        ],
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return PipelineResult(dataset, surface_fit, pca, risk, output_root, figures)

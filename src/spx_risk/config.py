"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    source: str
    start_date: date
    end_date: date
    underlying_ticker: str
    option_types: tuple[str, ...]
    wrds_schema_candidates: tuple[str, ...]
    cache_dir: Path
    partition_by_year: bool
    server_side_filters: bool


@dataclass(frozen=True)
class FilterConfig:
    min_option_price: float
    min_days_to_expiry: int
    max_days_to_expiry: int
    min_forward_moneyness: float
    max_forward_moneyness: float
    max_relative_spread: float
    min_open_interest: int
    min_daily_observations: int
    impute_missing_iv: bool


@dataclass(frozen=True)
class SurfaceConfig:
    method: str
    comparison_methods: tuple[str, ...]
    comparison_frequency: int
    maturity_days: tuple[int, ...]
    moneyness_grid: tuple[float, ...]
    iv_floor: float
    iv_cap: float
    spline_knots: int
    ridge_alpha: float


@dataclass(frozen=True)
class PCAConfig:
    components: int
    standardize: bool


@dataclass(frozen=True)
class RiskConfig:
    methods: tuple[str, ...]
    confidence_levels: tuple[float, ...]
    rolling_window: int
    minimum_history: int
    scenarios: int
    total_vega: float
    covariance_shrinkage: bool
    psp_weighting: str
    psp_decay: float


@dataclass(frozen=True)
class HestonConfig:
    enabled: bool
    recalibration_frequency: int
    integration_nodes: int
    integration_upper: float
    max_nfev: int
    simulation_steps: int
    simulation_integration_nodes: int
    variance_grid_size: int
    strike_grid_size: int
    physical_window: int


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    sample_root: Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    project: ProjectConfig
    data: DataConfig
    filters: FilterConfig
    surface: SurfaceConfig
    pca: PCAConfig
    risk: RiskConfig
    heston: HestonConfig
    output: OutputConfig


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _relative_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(
    path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent

    start = _parse_date(start_date or raw["data"]["start_date"])
    end = _parse_date(end_date or raw["data"]["end_date"])
    if start > end:
        raise ValueError(f"start_date {start} is later than end_date {end}")

    option_types = tuple(str(value).lower() for value in raw["data"]["option_types"])
    unsupported = set(option_types) - {"call", "put"}
    if unsupported:
        raise ValueError(f"Unsupported option types: {sorted(unsupported)}")

    confidence_levels = tuple(float(value) for value in raw["risk"]["confidence_levels"])
    if any(not 0 < value < 1 for value in confidence_levels):
        raise ValueError("All confidence levels must be between zero and one")

    supported_surfaces = {"polynomial", "b_spline", "thin_plate", "linear"}
    surface_methods = {
        str(raw["surface"].get("method", "polynomial")).lower(),
        *(
            str(value).lower()
            for value in raw["surface"].get("comparison_methods", ["polynomial"])
        ),
    }
    if unsupported_surfaces := surface_methods - supported_surfaces:
        raise ValueError(f"Unsupported surface methods: {sorted(unsupported_surfaces)}")
    risk_methods = tuple(
        str(value).lower() for value in raw["risk"].get("methods", ["pca", "pwg"])
    )
    if not risk_methods:
        raise ValueError("risk.methods must contain at least one method")
    if unsupported_risk := set(risk_methods) - {"psp", "pca", "pwg"}:
        raise ValueError(f"Unsupported risk methods: {sorted(unsupported_risk)}")
    if raw["risk"].get("psp_weighting", "uniform") not in {"uniform", "exponential"}:
        raise ValueError("risk.psp_weighting must be 'uniform' or 'exponential'")
    if int(raw["surface"].get("comparison_frequency", 1)) < 1:
        raise ValueError("surface.comparison_frequency must be at least one")
    if int(raw["surface"].get("spline_knots", 5)) < 3:
        raise ValueError("surface.spline_knots must be at least three")
    if not 0 < float(raw["risk"].get("psp_decay", 0.985)) <= 1:
        raise ValueError("risk.psp_decay must be in (0, 1]")
    heston_raw = raw.get("heston", {})
    if int(heston_raw.get("recalibration_frequency", 21)) < 1:
        raise ValueError("heston.recalibration_frequency must be at least one")
    if int(heston_raw.get("integration_nodes", 96)) < 32:
        raise ValueError("heston.integration_nodes must be at least 32")
    if int(heston_raw.get("simulation_steps", 8)) < 1:
        raise ValueError("heston.simulation_steps must be positive")
    if int(heston_raw.get("simulation_integration_nodes", 64)) < 32:
        raise ValueError("heston.simulation_integration_nodes must be at least 32")
    if int(heston_raw.get("variance_grid_size", 17)) < 5:
        raise ValueError("heston.variance_grid_size must be at least five")
    if int(heston_raw.get("strike_grid_size", 41)) < 9:
        raise ValueError("heston.strike_grid_size must be at least nine")
    if int(heston_raw.get("physical_window", 500)) < 60:
        raise ValueError("heston.physical_window must be at least 60")

    return AppConfig(
        project_root=project_root,
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(
            source=str(raw["data"]["source"]).lower(),
            start_date=start,
            end_date=end,
            underlying_ticker=str(raw["data"]["underlying_ticker"]).upper(),
            option_types=option_types,
            wrds_schema_candidates=tuple(raw["data"]["wrds_schema_candidates"]),
            cache_dir=_relative_path(project_root, raw["data"]["cache_dir"]),
            partition_by_year=bool(raw["data"].get("partition_by_year", False)),
            server_side_filters=bool(raw["data"].get("server_side_filters", False)),
        ),
        filters=FilterConfig(**raw["filters"]),
        surface=SurfaceConfig(
            method=str(raw["surface"].get("method", "polynomial")).lower(),
            comparison_methods=tuple(
                str(value).lower()
                for value in raw["surface"].get(
                    "comparison_methods", ["polynomial"]
                )
            ),
            comparison_frequency=int(raw["surface"].get("comparison_frequency", 1)),
            maturity_days=tuple(int(value) for value in raw["surface"]["maturity_days"]),
            moneyness_grid=tuple(float(value) for value in raw["surface"]["moneyness_grid"]),
            iv_floor=float(raw["surface"]["iv_floor"]),
            iv_cap=float(raw["surface"]["iv_cap"]),
            spline_knots=int(raw["surface"].get("spline_knots", 5)),
            ridge_alpha=float(raw["surface"].get("ridge_alpha", 1e-4)),
        ),
        pca=PCAConfig(**raw["pca"]),
        risk=RiskConfig(
            methods=risk_methods,
            confidence_levels=confidence_levels,
            rolling_window=int(raw["risk"]["rolling_window"]),
            minimum_history=int(raw["risk"]["minimum_history"]),
            scenarios=int(raw["risk"]["scenarios"]),
            total_vega=float(raw["risk"]["total_vega"]),
            covariance_shrinkage=bool(raw["risk"]["covariance_shrinkage"]),
            psp_weighting=str(raw["risk"].get("psp_weighting", "uniform")).lower(),
            psp_decay=float(raw["risk"].get("psp_decay", 0.985)),
        ),
        heston=HestonConfig(
            enabled=bool(heston_raw.get("enabled", False)),
            recalibration_frequency=int(
                heston_raw.get("recalibration_frequency", 21)
            ),
            integration_nodes=int(heston_raw.get("integration_nodes", 96)),
            integration_upper=float(heston_raw.get("integration_upper", 160.0)),
            max_nfev=int(heston_raw.get("max_nfev", 90)),
            simulation_steps=int(heston_raw.get("simulation_steps", 8)),
            simulation_integration_nodes=int(
                heston_raw.get("simulation_integration_nodes", 64)
            ),
            variance_grid_size=int(heston_raw.get("variance_grid_size", 17)),
            strike_grid_size=int(heston_raw.get("strike_grid_size", 41)),
            physical_window=int(heston_raw.get("physical_window", 500)),
        ),
        output=OutputConfig(
            root=_relative_path(project_root, raw["output"]["root"]),
            sample_root=_relative_path(project_root, raw["output"]["sample_root"]),
        ),
    )

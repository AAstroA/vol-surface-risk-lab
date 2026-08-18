"""Command-line interface for downloading, running, and reproducing the thesis analysis."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from spx_risk.config import AppConfig, load_config
from spx_risk.data.demo import generate_demo_dataset
from spx_risk.data.wrds import WRDSOptionMetricsClient, cache_directory
from spx_risk.pipeline import download_wrds, download_wrds_partitions, run_pipeline


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file")
    parser.add_argument("--start-date", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Override end date (YYYY-MM-DD)")


def _config(args: argparse.Namespace) -> AppConfig:
    return load_config(args.config, start_date=args.start_date, end_date=args.end_date)


def _safe_generated_path(config: AppConfig, path: Path) -> Path:
    resolved = path.resolve()
    project_root = config.project_root.resolve()
    if project_root not in resolved.parents or resolved == project_root:
        raise ValueError(f"Refusing to remove path outside the project: {resolved}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spx-risk", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("download", "Download and cache the configured WRDS interval"),
        ("run", "Run the pipeline from the configured source or existing cache"),
        ("all", "Download WRDS data and immediately run the pipeline"),
        ("demo", "Generate deterministic sample data and sample outputs"),
        ("probe-wrds", "Verify WRDS and OptionMetrics schema access"),
        ("clean-generated", "Remove only the configured generated output directory"),
    ):
        command = commands.add_parser(name, help=help_text)
        _add_common(command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _config(args)

    if args.command == "download":
        if config.data.partition_by_year:
            directories = download_wrds_partitions(config)
            print(f"Cached {len(directories)} annual WRDS partitions")
        else:
            dataset = download_wrds(config)
            print(f"Cached WRDS data at {cache_directory(config)} ({len(dataset.options):,} option rows)")
        return
    if args.command == "all":
        result = run_pipeline(config, force_download=True)
        print(f"Completed WRDS analysis: {result.output_root}")
        return
    if args.command in {"run", "demo"}:
        dataset = generate_demo_dataset(config) if args.command == "demo" else None
        result = run_pipeline(config, dataset=dataset)
        print(f"Completed analysis: {result.output_root}")
        return
    if args.command == "probe-wrds":
        with WRDSOptionMetricsClient(config) as client:
            print(f"WRDS connection successful; using schema {client.schema}")
        return
    if args.command == "clean-generated":
        target = _safe_generated_path(config, config.output.root)
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed generated output directory: {target}")
        else:
            print(f"Generated output directory does not exist: {target}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")

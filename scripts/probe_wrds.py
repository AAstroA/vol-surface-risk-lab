#!/usr/bin/env python3
"""Safely verify WRDS connectivity and OptionMetrics table access."""

from __future__ import annotations

import argparse

from spx_risk.config import load_config
from spx_risk.data.wrds import WRDSOptionMetricsClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    with WRDSOptionMetricsClient(config) as client:
        tables = sorted(
            table
            for table in client.connection.list_tables(library=client.schema)
            if table.startswith(("secnmd", "opprcd", "secprd", "zerocd"))
        )
        print(f"WRDS connection successful; OptionMetrics schema: {client.schema}")
        print(f"Recognized OptionMetrics tables: {len(tables)}")
        for table in tables[:20]:
            print(f"- {table}")
        if len(tables) > 20:
            print(f"... and {len(tables) - 20} more")


if __name__ == "__main__":
    main()

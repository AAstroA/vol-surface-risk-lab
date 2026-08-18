"""WRDS OptionMetrics downloader with schema discovery and local caching."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import dotenv_values

from spx_risk.config import AppConfig


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class WRDSDataset:
    options: pd.DataFrame
    underlying: pd.DataFrame
    zero_curve: pd.DataFrame
    metadata: dict[str, object]


def _safe_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def _years(start_date: date, end_date: date) -> Iterable[int]:
    return range(start_date.year, end_date.year + 1)


def _year_bounds(start_date: date, end_date: date, year: int) -> tuple[date, date]:
    return max(start_date, date(year, 1, 1)), min(end_date, date(year, 12, 31))


class WRDSOptionMetricsClient:
    """Query OptionMetrics IvyDB US and normalize it to the project schema."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._connection = None
        self._schema: str | None = None
        self._table_cache: dict[str, set[str]] = {}
        self._column_cache: dict[str, set[str]] = {}

    def connect(self) -> None:
        try:
            import wrds
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the project dependencies before downloading WRDS data") from exc

        credentials = dotenv_values(self.config.project_root / ".env")
        username = os.getenv("WRDS_USERNAME") or credentials.get("WRDS_USERNAME")
        password = os.getenv("WRDS_PASSWORD") or credentials.get("WRDS_PASSWORD")
        if not username or not password:
            raise RuntimeError("WRDS_USERNAME and WRDS_PASSWORD are required in .env or the environment")

        previous_password = os.environ.get("PGPASSWORD")
        os.environ["PGPASSWORD"] = str(password)
        try:
            self._connection = wrds.Connection(
                wrds_username=str(username),
                wrds_hostname=str(
                    os.getenv("WRDS_HOST")
                    or credentials.get("WRDS_HOST")
                    or "wrds-pgdata.wharton.upenn.edu"
                ),
                wrds_port=int(os.getenv("WRDS_PORT") or credentials.get("WRDS_PORT") or 9737),
                wrds_dbname=str(
                    os.getenv("WRDS_DATABASE") or credentials.get("WRDS_DATABASE") or "wrds"
                ),
                verbose=False,
            )
        finally:
            if previous_password is None:
                os.environ.pop("PGPASSWORD", None)
            else:
                os.environ["PGPASSWORD"] = previous_password
        accessible = set(self._connection.list_libraries())
        self._schema = next(
            (schema for schema in self.config.data.wrds_schema_candidates if schema in accessible),
            None,
        )
        if self._schema is None:
            raise PermissionError(
                "No configured OptionMetrics schema is accessible. "
                f"Checked: {list(self.config.data.wrds_schema_candidates)}"
            )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def __enter__(self) -> "WRDSOptionMetricsClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema(self) -> str:
        if self._schema is None:
            raise RuntimeError("WRDS connection is not open")
        return _safe_identifier(self._schema)

    @property
    def connection(self):
        if self._connection is None:
            raise RuntimeError("WRDS connection is not open")
        return self._connection

    def _tables(self) -> set[str]:
        if self.schema not in self._table_cache:
            self._table_cache[self.schema] = set(self.connection.list_tables(library=self.schema))
        return self._table_cache[self.schema]

    def _resolve_table(self, prefixes: tuple[str, ...], year: int | None = None) -> str:
        tables = self._tables()
        candidates: list[str] = []
        for prefix in prefixes:
            if year is not None:
                candidates.extend((f"{prefix}{year}", f"{prefix}_{year}"))
            candidates.append(prefix)
        for candidate in candidates:
            if candidate in tables:
                return _safe_identifier(candidate)
        raise LookupError(
            f"Could not resolve any table for prefixes={prefixes}, year={year}, schema={self.schema}"
        )

    def _columns(self, table: str) -> set[str]:
        table = _safe_identifier(table)
        if table not in self._column_cache:
            description = self.connection.describe_table(library=self.schema, table=table)
            self._column_cache[table] = set(description["name"].astype(str))
        return self._column_cache[table]

    def _numeric_alias(
        self,
        table: str,
        candidates: tuple[str, ...],
        alias: str,
    ) -> str:
        columns = self._columns(table)
        for candidate in candidates:
            if candidate in columns:
                return f'"{_safe_identifier(candidate)}" AS {_safe_identifier(alias)}'
        return f"NULL::double precision AS {_safe_identifier(alias)}"

    def _spx_secids(self) -> list[int]:
        table = self._resolve_table(("secnmd", "security_name"))
        ticker = self.config.data.underlying_ticker.replace("'", "''")
        query = f"""
            SELECT DISTINCT secid
            FROM {self.schema}.{table}
            WHERE UPPER(TRIM(ticker)) = '{ticker}'
              AND effect_date <= '{self.config.data.end_date.isoformat()}'
            ORDER BY secid
        """
        frame = self.connection.raw_sql(query)
        if frame.empty:
            raise LookupError(f"No OptionMetrics secid found for ticker {ticker}")
        return [int(value) for value in frame["secid"].dropna().unique()]

    def _fetch_options_year(self, year: int, secids: list[int]) -> pd.DataFrame:
        table = self._resolve_table(("opprcd", "option_price"), year)
        secid_sql = ",".join(str(value) for value in secids)
        start, end = _year_bounds(
            self.config.data.start_date, self.config.data.end_date, year
        )
        server_filters = ""
        if self.config.data.server_side_filters:
            filters = self.config.filters
            option_flags = ",".join(
                f"'{value[0].upper()}'" for value in self.config.data.option_types
            )
            conditions = [
                f"cp_flag IN ({option_flags})",
                f"(exdate - date) BETWEEN {filters.min_days_to_expiry} AND {filters.max_days_to_expiry}",
                f"((best_bid + best_offer) / 2.0) >= {float(filters.min_option_price)}",
                "best_offer >= best_bid",
                f"COALESCE(open_interest, 0) >= {int(filters.min_open_interest)}",
                "((best_offer - best_bid) / NULLIF((best_bid + best_offer) / 2.0, 0)) "
                f"<= {float(filters.max_relative_spread)}",
            ]
            if not filters.impute_missing_iv:
                conditions.extend(("impl_volatility IS NOT NULL", "impl_volatility > 0"))
            server_filters = "\n              AND " + "\n              AND ".join(conditions)
        query = f"""
            SELECT
                secid,
                date AS quote_date,
                exdate AS expiration,
                cp_flag,
                strike_price / 1000.0 AS strike,
                best_bid AS bid,
                best_offer AS ask,
                volume,
                open_interest,
                impl_volatility AS implied_volatility,
                delta,
                gamma,
                vega,
                theta,
                forward_price,
                optionid
            FROM {self.schema}.{table}
            WHERE secid IN ({secid_sql})
              AND date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
              {server_filters}
        """
        return self.connection.raw_sql(query, date_cols=["quote_date", "expiration"])

    def _fetch_underlying_year(self, year: int, secids: list[int]) -> pd.DataFrame:
        table = self._resolve_table(("secprd", "securd", "security_price"), year)
        secid_sql = ",".join(str(value) for value in secids)
        start, end = _year_bounds(
            self.config.data.start_date, self.config.data.end_date, year
        )
        total_return = self._numeric_alias(
            table,
            ("total_return", "return", "cfret"),
            "total_return",
        )
        query = f"""
            SELECT secid, date AS quote_date, close, volume, {total_return}
            FROM {self.schema}.{table}
            WHERE secid IN ({secid_sql})
              AND date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
        """
        return self.connection.raw_sql(query, date_cols=["quote_date"])

    def _fetch_zero_curve_year(self, year: int) -> pd.DataFrame:
        table = self._resolve_table(("zerocd", "zero_curve"), year)
        start, end = _year_bounds(
            self.config.data.start_date, self.config.data.end_date, year
        )
        query = f"""
            SELECT date AS quote_date, days, rate
            FROM {self.schema}.{table}
            WHERE date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
            ORDER BY date, days
        """
        return self.connection.raw_sql(query, date_cols=["quote_date"])

    def _fetch_index_dividends(
        self, secids: list[int], start_date: date, end_date: date
    ) -> pd.DataFrame:
        table = self._resolve_table(("idxdvd", "index_dividend"))
        secid_sql = ",".join(str(value) for value in secids)
        query = f"""
            SELECT
                secid,
                date AS quote_date,
                rate / 100.0 AS dividend_yield
            FROM {self.schema}.{table}
            WHERE secid IN ({secid_sql})
              AND date BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        """
        return self.connection.raw_sql(query, date_cols=["quote_date"])

    def download_year(self, year: int, secids: list[int] | None = None) -> WRDSDataset:
        """Download one bounded year so each completed partition can be cached immediately."""
        if year not in _years(self.config.data.start_date, self.config.data.end_date):
            raise ValueError(f"Year {year} is outside the configured interval")
        secids = secids or self._spx_secids()
        start, end = _year_bounds(
            self.config.data.start_date, self.config.data.end_date, year
        )
        options = self._fetch_options_year(year, secids)
        underlying = self._fetch_underlying_year(year, secids)
        dividends = self._fetch_index_dividends(secids, start, end)
        underlying = underlying.merge(
            dividends,
            on=["secid", "quote_date"],
            how="left",
            validate="one_to_one",
        )
        zero_curve = self._fetch_zero_curve_year(year).drop_duplicates(
            ["quote_date", "days"]
        )
        options["type"] = options["cp_flag"].map({"C": "call", "P": "put"})
        options["contract"] = options["optionid"].astype("Int64").astype(str)
        options["underlying"] = self.config.data.underlying_ticker
        metadata = {
            "source": "WRDS OptionMetrics IvyDB US",
            "schema": self.schema,
            "underlying": self.config.data.underlying_ticker,
            "secids": secids,
            "year": year,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "server_side_filters": self.config.data.server_side_filters,
            "interest_rate_source": "OptionMetrics zero curve (zerocd); legacy PwG rate files excluded",
            "option_rows": int(len(options)),
            "underlying_rows": int(len(underlying)),
            "zero_curve_rows": int(len(zero_curve)),
            "index_dividend_rows": int(len(dividends)),
            "dividend_yield_source": "OptionMetrics index dividend yield (idxdvd)",
        }
        return WRDSDataset(options, underlying, zero_curve, metadata)

    def download(self) -> WRDSDataset:
        secids = self._spx_secids()
        partitions = [
            self.download_year(year, secids)
            for year in _years(self.config.data.start_date, self.config.data.end_date)
        ]
        options = pd.concat([item.options for item in partitions], ignore_index=True)
        underlying = pd.concat([item.underlying for item in partitions], ignore_index=True)
        zero_curve = pd.concat([item.zero_curve for item in partitions], ignore_index=True).drop_duplicates(
            ["quote_date", "days"]
        )
        metadata = {
            "source": "WRDS OptionMetrics IvyDB US",
            "schema": self.schema,
            "underlying": self.config.data.underlying_ticker,
            "secids": secids,
            "start_date": self.config.data.start_date.isoformat(),
            "end_date": self.config.data.end_date.isoformat(),
            "interest_rate_source": "OptionMetrics zero curve (zerocd); legacy PwG rate files excluded",
            "option_rows": int(len(options)),
            "underlying_rows": int(len(underlying)),
            "zero_curve_rows": int(len(zero_curve)),
            "index_dividend_rows": int(
                sum(int(item.metadata["index_dividend_rows"]) for item in partitions)
            ),
            "dividend_yield_source": "OptionMetrics index dividend yield (idxdvd)",
        }
        return WRDSDataset(options, underlying, zero_curve, metadata)


def save_dataset(dataset: WRDSDataset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dataset.options.to_parquet(directory / "options.parquet", index=False)
    dataset.underlying.to_parquet(directory / "underlying.parquet", index=False)
    dataset.zero_curve.to_parquet(directory / "zero_curve.parquet", index=False)
    (directory / "metadata.json").write_text(
        json.dumps(dataset.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_dataset(directory: Path) -> WRDSDataset:
    return WRDSDataset(
        options=pd.read_parquet(directory / "options.parquet"),
        underlying=pd.read_parquet(directory / "underlying.parquet"),
        zero_curve=pd.read_parquet(directory / "zero_curve.parquet"),
        metadata=json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
    )


def load_metadata(directory: Path) -> dict[str, object]:
    return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))


def cache_directory(config: AppConfig) -> Path:
    interval = f"{config.data.start_date.isoformat()}_{config.data.end_date.isoformat()}"
    return config.data.cache_dir / config.data.underlying_ticker / interval


def annual_cache_directory(config: AppConfig, year: int) -> Path:
    """Stable annual cache location shared by long-horizon configurations."""
    return config.data.cache_dir / config.data.underlying_ticker / "by_year" / str(year)


def partition_is_complete(directory: Path) -> bool:
    return all(
        (directory / name).is_file()
        for name in ("options.parquet", "underlying.parquet", "zero_curve.parquet", "metadata.json")
    )

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


class WrdsUnavailable(RuntimeError):
    pass


@dataclass
class ResolvedTable:
    logical_name: str
    library: str
    table: str

    @property
    def sql_name(self) -> str:
        return f"{self.library}.{self.table}"


def _connect():
    try:
        import wrds
    except ImportError as exc:
        raise WrdsUnavailable("The wrds package is not installed.") from exc
    try:
        return wrds.Connection()
    except Exception as exc:  # noqa: BLE001
        raise WrdsUnavailable(f"WRDS connection failed: {type(exc).__name__}") from exc


def check_pgpass() -> dict[str, str | bool]:
    path = Path.home() / ".pgpass"
    result: dict[str, str | bool] = {"exists": path.exists()}
    if path.exists():
        result["permissions"] = oct(path.stat().st_mode)[-3:]
    return result


def resolve_tables(source_config: dict, candidates_only: Iterable[str] | None = None) -> pd.DataFrame:
    db = _connect()
    rows: list[dict[str, str | bool]] = []
    wanted = set(candidates_only or [])
    logical_tables = source_config["logical_tables"]
    for logical, spec in logical_tables.items():
        if wanted and logical not in wanted:
            continue
        resolved = None
        errors: list[str] = []
        for candidate in spec["candidates"]:
            library, table = candidate.split(".", 1)
            try:
                tables = set(db.list_tables(library=library))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{library}: {type(exc).__name__}")
                continue
            lower_map = {name.lower(): name for name in tables}
            if table in tables:
                resolved = (library, table)
                break
            if table.lower() in lower_map:
                resolved = (library, lower_map[table.lower()])
                break
            if logical in {"option_raw_prefix", "option_surface", "option_security_price"}:
                matches = sorted(name for name in tables if name.lower().startswith(table.lower()))
                if matches:
                    resolved = (library, matches[0])
                    break
        rows.append(
            {
                "logical_name": logical,
                "resolved": resolved is not None,
                "library": resolved[0] if resolved else "",
                "table": resolved[1] if resolved else "",
                "errors": "; ".join(errors[:3]),
            }
        )
    db.close()
    return pd.DataFrame(rows)


def raw_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    db = _connect()
    try:
        LOGGER.info("Executing WRDS query hash only; query text is stored in manifests.")
        return db.raw_sql(query, params=params)
    finally:
        db.close()

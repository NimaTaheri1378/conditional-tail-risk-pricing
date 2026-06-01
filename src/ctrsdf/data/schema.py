from __future__ import annotations

from pathlib import Path

import pandas as pd

from ctrsdf.config import ProjectConfig, load_yaml
from ctrsdf.data.wrds_conn import check_pgpass, resolve_tables
from ctrsdf.utils.io import atomic_write_parquet
from ctrsdf.utils.manifest import Manifest


def run_schema_audit(config: ProjectConfig) -> Path:
    source_config = load_yaml(config.root / "configs" / "data_sources.yaml")
    pgpass = check_pgpass()
    resolved = resolve_tables(source_config)
    output = config.path("manifests") / "schema_audit.parquet"
    atomic_write_parquet(resolved, output)
    manifest = Manifest(
        name="schema_audit",
        status="completed" if bool(resolved["resolved"].any()) else "failed",
        diagnostics={
            "pgpass_exists": pgpass.get("exists", False),
            "pgpass_permissions": pgpass.get("permissions", ""),
            "logical_tables": int(len(resolved)),
            "resolved_tables": int(resolved["resolved"].sum()),
        },
        outputs={"resolved_tables": str(output)},
    )
    manifest.write(config.path("manifests") / "schema_audit.json")
    return output


def load_schema_audit(config: ProjectConfig) -> pd.DataFrame:
    return pd.read_parquet(config.path("manifests") / "schema_audit.parquet")

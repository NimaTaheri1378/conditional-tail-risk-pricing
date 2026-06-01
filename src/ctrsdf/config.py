from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    root: Path

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def sample_start(self) -> str:
        return str(self.raw["project"]["sample_start"])

    @property
    def sample_end(self) -> str:
        return str(self.raw["project"]["sample_end"])

    @property
    def smoke_start(self) -> str:
        return str(self.raw["project"]["smoke_start"])

    @property
    def smoke_end(self) -> str:
        return str(self.raw["project"]["smoke_end"])

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = config_path.parents[1]
    return ProjectConfig(raw=raw, root=root)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

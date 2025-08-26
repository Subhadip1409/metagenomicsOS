# cli/core/config_model.py
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ValidationError
import yaml


class AppConfig(BaseModel):
    project_name: str = Field(default="project", description="Project name")
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    database_dir: Path = Field(default=Path("./db"), description="Database directory")
    threads: int = Field(default=4, ge=1, description="Default CPU threads")


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        # If no config on disk, return defaults; commands can prompt or write later.
        return AppConfig()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Pydantic will coerce and validate fields appropriately.
    return AppConfig(**raw)


def dump_config(cfg: AppConfig) -> dict[str, Any]:
    data: dict[str, Any] = cfg.model_dump()
    # Convert Paths to strings for YAML friendliness.
    data["data_dir"] = str(data["data_dir"])
    data["database_dir"] = str(data["database_dir"])
    return data


def save_config(path: Path, cfg: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dump_config(cfg), f, sort_keys=False)


def validate_config_dict(d: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        AppConfig(**d)
        return True, None
    except ValidationError as e:
        return False, str(e)

# cli/commands/config.py
from __future__ import annotations
from pathlib import Path
import json
import os
import tempfile
import shutil
from typing import Any, Iterable
import typer
import yaml
import click
from pydantic import ValidationError, Extra

from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config, AppConfig, dump_config

app = typer.Typer(help="Configuration management")


def _resolve_config_path(path_opt: Path | None, ctx: typer.Context = None) -> Path:
    # If path provided via --path, use it
    if path_opt is not None:
        return path_opt

    # Try to get from context, fallback to default
    try:
        app_ctx = get_context(ctx)
        return app_ctx.config_path
    except Exception:
        return Path("./config.yaml")


def _atomic_write_yaml(target: Path, data: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(target.parent), encoding="utf-8"
    ) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)


def _parse_set_flags(sets: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in sets:
        if "=" not in s:
            raise typer.BadParameter(f"--set must be key=value (got: {s})")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Basic casting for common fields in AppConfig
        if k in {"threads"}:
            v_cast: Any = int(v)
        elif k in {"data_dir", "database_dir"}:
            v_cast = str(Path(v))
        else:
            v_cast = v
        out[k] = v_cast
    return out


@app.command("path")
def path_cmd(
    path: Path | None = typer.Option(None, "--path", help="Override config path"),
) -> None:
    p = _resolve_config_path(path)
    typer.echo(str(p))


@app.command("show")
def show_cmd(
    ctx: typer.Context,
    path: Path | None = typer.Option(None, "--path", help="Override config path"),
    format: str = typer.Option(
        "yaml", "--format", case_sensitive=False, help="Output format: yaml|json"
    ),
) -> None:
    p = _resolve_config_path(path, ctx)
    try:
        cfg = load_config(p)
    except ValidationError as e:
        typer.echo(f"Invalid config: {e}", err=True)
        raise typer.Exit(code=2)
    data = dump_config(cfg)
    if format.lower() == "json":
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(yaml.safe_dump(data, sort_keys=False))


@app.command("edit")
def edit_cmd(
    path: Path | None = typer.Option(None, "--path", help="Override config path"),
    set_: list[str] = typer.Option(
        None, "--set", help="Inline update key=value (repeatable)", show_default=False
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail on unknown keys when saving"
    ),
) -> None:
    """Edit the configuration interactively or apply inline updates with --set."""
    p = _resolve_config_path(path)

    # Ensure file exists with defaults so editors have content.
    try:
        current = load_config(p)
    except ValidationError as e:
        typer.echo(f"Invalid existing config: {e}", err=True)
        raise typer.Exit(code=2)
    if not p.exists():
        _atomic_write_yaml(p, dump_config(current))

    # Inline updates
    if set_:
        updates = _parse_set_flags(set_)
        data = current.model_dump()
        data.update(updates)

        # Strict validation class if requested
        Model = AppConfig

        try:
            new_cfg = Model(**data)
        except ValidationError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=2)

        _atomic_write_yaml(p, dump_config(new_cfg))
        typer.echo("UPDATED")
        return

    # Interactive edit using system editor
    original_text = p.read_text(encoding="utf-8")
    edited = click.edit(original_text, extension=".yaml")
    if edited is None:
        typer.echo("No changes.")
        return

    # Backup before overwrite
    bak = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, bak)

    try:
        parsed = yaml.safe_load(edited) or {}
        # Strict optional validation
        if strict:

            class StrictAppConfig(AppConfig):
                class Config:
                    extra = Extra.forbid

            Model = StrictAppConfig
        else:
            Model = AppConfig

        new_cfg = Model(**parsed)
        _atomic_write_yaml(p, dump_config(new_cfg))
        typer.echo("SAVED")
    except Exception as e:
        typer.echo(f"ERROR: {e}", err=True)
        typer.echo(f"Backup kept at: {bak}", err=True)
        raise typer.Exit(code=2)


@app.command("validate")
def validate_cmd(
    path: Path | None = typer.Option(None, "--path", help="Override config path"),
    strict: bool = typer.Option(False, "--strict", help="Fail on unknown keys"),
) -> None:
    p = _resolve_config_path(path)
    try:
        # Load raw YAML to optionally enforce strictness on unknown keys
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
        if strict:

            class StrictAppConfig(AppConfig):
                class Config:
                    extra = Extra.forbid

            _ = StrictAppConfig(**raw)
        else:
            _ = AppConfig(**raw if raw else AppConfig().model_dump())
        typer.echo("OK")
    except ValidationError as e:
        typer.echo("INVALID", err=True)
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2)

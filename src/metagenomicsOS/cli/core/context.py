# cli/core/context.py
from dataclasses import dataclass
from pathlib import Path
import typer
from typing import cast


@dataclass
class AppContext:
    config_path: Path
    verbose: int = 0  # 0=WARNING, 1=INFO, 2+=DEBUG
    dry_run: bool = False


def set_context(ctx: typer.Context, obj: AppContext) -> None:
    ctx.obj = obj


def get_context(ctx: typer.Context | None = None) -> AppContext:
    if ctx is None:
        # Try to get current context, but handle if method doesn't exist
        try:
            ctx = typer.get_current_context()
        except (AttributeError, RuntimeError):
            # If typer.get_current_context doesn't exist or fails, return defaults
            return AppContext(config_path=Path("./config.yaml"))

    if ctx is None or ctx.obj is None:
        # Fallback to defaults if no context
        return AppContext(config_path=Path("./config.yaml"))

    return cast(AppContext, ctx.obj)

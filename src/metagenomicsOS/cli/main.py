# main.py
from __future__ import annotations
from pathlib import Path
import importlib
import typer

from metagenomicsOS.cli.core.context import AppContext, set_context
from metagenomicsOS.cli.core.logging import configure_logging


def _register_group(
    app: typer.Typer, name: str, module_path: str, help_text: str
) -> None:
    """Try to import module_path and fetch its `app: typer.Typer`.

    If not available, attach a named stub so CLI remains navigable.
    """
    try:
        mod = importlib.import_module(module_path)
        subapp = getattr(mod, "app")
    except Exception:
        from metagenomicsOS.cli.commands import _stubs

        subapp = _stubs.get_stub_app(name, help_text)
    app.add_typer(subapp, name=name, help=help_text)


def build_app() -> typer.Typer:
    app = typer.Typer(add_completion=True, help="Metagenomics CLI")

    @app.callback()
    def _root(
        ctx: typer.Context,
        config: Path = typer.Option(
            Path("./config.yaml"), "--config", "-c", help="Path to config file"
        ),
        verbose: int = typer.Option(
            0, "--verbose", "-v", count=True, help="-v=INFO, -vv=DEBUG"
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Do not execute actions, only plan/log"
        ),
    ) -> None:
        set_context(
            ctx, AppContext(config_path=config, verbose=verbose, dry_run=dry_run)
        )
        configure_logging(verbose)

    groups: list[tuple[str, str, str]] = [
        ("run", "metagenomicsOS.cli.commands.run", "Core workflows"),
        ("pipeline", "metagenomicsOS.cli.commands.pipeline", "End-to-end pipeline"),
        ("qc", "metagenomicsOS.cli.commands.qc", "Quality control"),
        ("assembly", "metagenomicsOS.cli.commands.assembly", "Genome assembly"),
        ("taxonomy", "metagenomicsOS.cli.commands.taxonomy", "Taxonomic profiling"),
        (
            "annotation",
            "metagenomicsOS.cli.commands.annotation",
            "Functional/ARG annotation",
        ),
        ("binning", "metagenomicsOS.cli.commands.binning", "Binning contigs/MAGs"),
        (
            "annotate",
            "metagenomicsOS.cli.commands.annotate",
            "Dedicated annotation workflows",
        ),
        ("genes", "metagenomicsOS.cli.commands.genes", "Gene-level annotation"),
        (
            "proteins",
            "metagenomicsOS.cli.commands.proteins",
            "Protein functional annotation",
        ),
        ("resistance", "metagenomicsOS.cli.commands.resistance", "ARG/AMR annotation"),
        (
            "visualize",
            "metagenomicsOS.cli.commands.visualize",
            "Visualization and reports",
        ),
        ("report", "metagenomicsOS.cli.commands.report", "Report generation"),
        ("compare", "metagenomicsOS.cli.commands.compare", "Comparative metagenomics"),
        ("database", "metagenomicsOS.cli.commands.database", "Database management"),
        ("config", "metagenomicsOS.cli.commands.config", "Configuration management"),
        ("monitor", "metagenomicsOS.cli.commands.monitor", "Monitoring"),
        ("validate", "metagenomicsOS.cli.commands.validate", "Validation utilities"),
        ("optimize", "metagenomicsOS.cli.commands.optimize", "Auto-tuning resources"),
        (
            "benchmark",
            "metagenomicsOS.cli.commands.benchmark",
            "Performance benchmarking",
        ),
        ("stream", "metagenomicsOS.cli.commands.stream", "Real-time streaming"),
        ("ml", "metagenomicsOS.cli.commands.ml", "Machine learning extras"),
        ("plugins", "metagenomicsOS.cli.commands.plugin", "Extensibility and plugins"),
    ]
    for name, module_path, help_text in groups:
        _register_group(app, name, module_path, help_text)

    return app


app = build_app()

if __name__ == "__main__":
    app()

# cli/commands/_stubs.py
import typer


def get_stub_app(name: str, help_text: str = "") -> typer.Typer:
    app = typer.Typer(help=help_text or f"{name} commands")

    @app.callback(invoke_without_command=True)
    def _stub(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            typer.echo(
                f"[{name}] commands are coming soon. Use --help to see placeholders."
            )

    return app

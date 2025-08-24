import typer

app = typer.Typer(help="Manage plugins for extending metagenomicsOS.")


@app.command()
def list():
    """List installed plugins."""
    typer.echo("Listing installed plugins...")


@app.command()
def install(name: str):
    """Install a plugin by name."""
    typer.echo(f"Installing plugin {name}")


@app.command()
def develop(name: str):
    """Scaffold a new plugin."""
    typer.echo(f"Scaffolding new plugin {name}")

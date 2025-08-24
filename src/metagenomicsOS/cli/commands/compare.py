import typer

app = typer.Typer(help="Compare metagenomic samples, groups, or resistomes.")


@app.command()
def samples(metadata: str):
    """Compare multiple samples."""
    typer.echo(f"Comparing samples using metadata {metadata}")


@app.command()
def groups(metadata: str):
    """Perform group-wise differential abundance analysis."""
    typer.echo(f"Comparing groups using metadata {metadata}")


@app.command()
def resistome(samples: str):
    """Compare ARG/AMR prevalence across samples."""
    typer.echo(f"Comparing resistomes using {samples}")

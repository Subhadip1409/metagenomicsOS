import typer

app = typer.Typer(help="Generate visualizations and reports.")


@app.command()
def barplot(report: str):
    """Generate a barplot from a taxonomy or annotation report."""
    typer.echo(f"Generating barplot from {report}")


@app.command()
def heatmap(matrix: str):
    """Generate a heatmap from abundance matrix."""
    typer.echo(f"Generating heatmap from {matrix}")


@app.command()
def network(input: str):
    """Build co-occurrence networks from annotation data."""
    typer.echo(f"Generating network visualization from {input}")


@app.command()
def report(input: str, out: str = "summary_report.pdf"):
    """Generate full HTML/PDF report."""
    typer.echo(f"Creating report from {input} -> {out}")

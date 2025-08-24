import typer

app = typer.Typer(help="Report shortcut (alias for visualize.report).")


@app.command()
def generate(input: str, out: str = "summary_report.pdf"):
    """Generate summary report (shortcut to visualize.report)."""
    typer.echo(f"Creating report from {input} -> {out}")

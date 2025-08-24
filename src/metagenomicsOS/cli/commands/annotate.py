import typer

app = typer.Typer(help="Annotate genes, proteins, or resistance genes.")


@app.command()
def genes(input: str, out: str = "genes.json"):
    """Annotate genes in assemblies or contigs."""
    typer.echo(f"Annotating genes from {input} -> {out}")


@app.command()
def proteins(input: str, out: str = "proteins.json"):
    """Annotate proteins with functional databases."""
    typer.echo(f"Annotating proteins from {input} -> {out}")


@app.command()
def resistance(input: str, out: str = "resistance.json"):
    """Annotate antibiotic resistance genes (ARGs)."""
    typer.echo(f"Annotating ARGs from {input} -> {out}")

import typer

app = typer.Typer(help="Machine learning utilities for metagenomics.")


@app.command()
def train(data: str, model_out: str = "model.pkl"):
    """Train a machine learning model for binning or ARG prediction."""
    typer.echo(f"Training model on {data} -> {model_out}")


@app.command()
def predict(model: str, input: str, out: str = "predictions.json"):
    """Predict ARGs or functional categories using a trained model."""
    typer.echo(f"Using {model} to predict on {input} -> {out}")


@app.command()
def drift(model: str, data: str):
    """Monitor model drift in new datasets."""
    typer.echo(f"Checking drift for {model} on {data}")

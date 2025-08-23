# metagenomicsOS/cli/__main__.py
import click
from .commands.config import config


@click.group()
def cli():
    """Metagenomicsos Command Line Interface."""
    pass


# Add the config command to the main CLI group
cli.add_command(config)

if __name__ == "__main__":
    cli()

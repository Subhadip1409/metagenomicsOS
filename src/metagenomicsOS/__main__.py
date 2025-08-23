"""MetagenomicsOS main entry point.

This enables 'python -m metagenomicsos' functionality.
"""

from .cli.main import main_cli


def main():
    """Main entry point for the package."""
    main_cli()


if __name__ == "__main__":
    main()

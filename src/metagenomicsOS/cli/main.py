"""Main CLI application for MetagenomicsOS.

This is a skeleton implementation that will be expanded in Week 3 Day 2-3.
Currently provides basic functionality to test the configuration system.
"""


def app() -> int:
    """Minimal app function placeholder for CLI import."""
    from .. import __version__

    print(f"MetagenomicsOS v{__version__}")
    print("CLI framework coming in Week 3 Day 2-3!")
    return 0


if __name__ == "__main__":
    exit(app())

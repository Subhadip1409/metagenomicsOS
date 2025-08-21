"""Entry point for running MetagenomicsOS as a module.

Usage: python -m metagenomicsOS

This file remains minimal as per best practices, delegating all logic to CLI.
"""


def main() -> int:  # ← Add return type
    """Main entry point - delegates to CLI."""
    try:
        from .cli.main import app

        app()
    except ImportError as e:
        # Graceful fallback during development
        print(f"MetagenomicsOS v{__version__}")
        print("CLI system initializing...")
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    from . import __version__

    exit(main())

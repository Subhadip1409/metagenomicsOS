"""core/utils.py.

Utility functions for logging, file validation, and common operations.
"""

import logging
from pathlib import Path


def setup_logging(verbose: bool = False, quiet: bool = False, level: str = "INFO"):
    """Setup logging configuration based on CLI options."""
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Silence some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def validate_input_file(file_path: Path) -> bool:
    """Validate that input file exists and has correct format for metagenomics.

    Args:
        file_path: Path to the input file

    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False

    # Check file size (shouldn't be empty)
    if file_path.stat().st_size == 0:
        return False

    # Valid extensions for metagenomics files
    valid_extensions = {
        ".fastq",
        ".fq",
        ".fastq.gz",
        ".fq.gz",
        ".fasta",
        ".fa",
        ".fasta.gz",
        ".fa.gz",
    }

    # Check if file has valid extension
    if file_path.suffix in valid_extensions:
        return True

    # Check for compressed files with double extension
    if any(file_path.name.endswith(ext) for ext in valid_extensions):
        return True

    return False

"""MetagenomicsOS - A production-grade metagenomics analysis platform.

A comprehensive bioinformatics system for metagenomics analysis with AI optimization,
real-time processing, and multi-platform deployment capabilities.
"""

__version__ = "0.1.0"
__author__ = "Subhadip Jana"
__email__ = "subhadipjana1409@gmail.com"
__description__ = "Production-grade metagenomics analysis platform"

# Import core components for easy access
from .core.config_manager import ConfigManager
from .core.exceptions import (
    MetagenomicsOSError,
    ConfigurationError,
    DatabaseError,
    WorkflowError,
)

__all__ = [
    "ConfigManager",
    "MetagenomicsOSError",
    "ConfigurationError",
    "DatabaseError",
    "WorkflowError",
]

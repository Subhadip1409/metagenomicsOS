"""MetagenomicsOS - A production-grade metagenomics analysis platform.

A comprehensive bioinformatics system for metagenomics analysis with AI optimization,
real-time processing, and multi-platform deployment capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Production-grade metagenomics analysis platform"

# Import core components for easy access
from .core.config_manager import ConfigManager, MetagenomicsConfig

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "ConfigManager",
    "MetagenomicsConfig",
]

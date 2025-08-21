"""Core functionality for MetagenomicsOS.

This module contains the fundamental components including configuration management,
data models, workflow engines, and other essential system components.
"""

from .config_manager import ConfigManager, MetagenomicsConfig

__all__ = [
    "ConfigManager",
    "MetagenomicsConfig",
]

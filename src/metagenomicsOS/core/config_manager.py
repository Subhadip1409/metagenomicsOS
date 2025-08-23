"""Configuration management using Pydantic for validation and type safety.

This module provides robust configuration loading, validation, and management
for all MetagenomicsOS operations with support for YAML and JSON formats.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    kraken2_db_path: Optional[str] = None
    eggnog_db_path: Optional[str] = None
    cache_dir: str = Field(
        default=".cache", description="Local cache directory for downloads"
    )
    auto_download: bool = Field(
        default=True, description="Automatically download missing databases"
    )

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Ensure cache directory is valid."""
        if not v or v.isspace():
            raise ValueError("Cache directory cannot be empty")
        return v.strip()


class ResourceConfig(BaseModel):
    """Resource allocation settings."""

    threads: int = Field(
        default=1, ge=1, le=128, description="Number of CPU threads to use"
    )
    memory_gb: int = Field(default=4, ge=1, le=1024, description="Memory limit in GB")
    temp_dir: str = Field(
        default_factory=tempfile.gettempdir,
        description="Temporary directory for intermediate files",
    )

    @field_validator("temp_dir")
    @classmethod
    def validate_temp_dir(cls, v: str) -> str:
        """Ensure temp directory exists or can be created."""
        temp_path = Path(v)
        try:
            temp_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            raise ValueError(f"Cannot create or access temp directory: {v}")
        return str(temp_path.absolute())


class WorkflowConfig(BaseModel):
    """Workflow execution settings."""

    # Pipeline modules
    quality_control: bool = Field(default=True, description="Run quality control")
    taxonomy: bool = Field(default=True, description="Run taxonomic classification")
    functional_analysis: bool = Field(
        default=True, description="Run functional annotation"
    )
    assembly: bool = Field(default=False, description="Run genome assembly")
    binning: bool = Field(default=False, description="Run genome binning")

    # Quality control settings
    min_read_length: int = Field(default=50, ge=1, description="Minimum read length")
    min_quality: int = Field(
        default=20, ge=1, le=40, description="Minimum quality score"
    )

    # Output settings
    save_intermediate: bool = Field(
        default=False, description="Save intermediate files"
    )
    output_formats: List[str] = Field(
        default=["tsv", "json"], description="Output formats to generate"
    )

    @field_validator("output_formats")
    @classmethod
    def validate_output_formats(
        cls, v: List[str]
    ) -> List[str]:  # ← Add : List[str]) -> List[str]
        """Ensure output formats are supported."""
        supported = {"tsv", "csv", "json", "xlsx", "html"}
        for fmt in v:
            if fmt.lower() not in supported:
                raise ValueError(f"Unsupported output format: {fmt}")
        return [fmt.lower() for fmt in v]


class MetagenomicsConfig(BaseModel):
    """Main configuration model for MetagenomicsOS."""

    # Project settings
    project_name: str = Field(
        default="metagenomics_project",
        min_length=1,
        description="Project name for output organization",
    )
    output_dir: str = Field(default="results", description="Main output directory")
    input_dir: Optional[str] = Field(
        None, description="Input directory containing sequencing data"
    )

    # Sub-configurations
    databases: DatabaseConfig = Field(default_factory=lambda: DatabaseConfig())
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)

    # Advanced settings
    debug_mode: bool = Field(default=False, description="Enable debug output")
    dry_run: bool = Field(default=False, description="Show commands without executing")

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:  # ← Add : str) -> str
        """Ensure project name is filesystem-safe."""
        if not v or v.isspace():
            raise ValueError("Project name cannot be empty")
        # Remove problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Project name cannot contain '{char}'")
        return v.strip()

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:  # ← Add : str) -> str
        """Ensure output directory is valid."""
        if not v or v.isspace():
            raise ValueError("Output directory cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_workflow_dependencies(self) -> "MetagenomicsConfig":
        """Ensure workflow module dependencies are met."""
        # Binning requires assembly
        if self.workflow.binning and not self.workflow.assembly:
            raise ValueError("Binning requires assembly to be enabled")
        return self


class ConfigManager:
    """Configuration manager with loading, validation, and saving capabilities.

    Supports YAML and JSON configuration files with automatic format detection
    and comprehensive error handling.
    """

    SUPPORTED_FORMATS = {".yaml", ".yml", ".json"}

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._config: Optional[MetagenomicsConfig] = None

    def load_config(self, config_path: Optional[str] = None) -> MetagenomicsConfig:
        """Load configuration from file or create default configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            MetagenomicsConfig: Validated configuration object

        Raises:
            ValueError: If configuration file is invalid
            FileNotFoundError: If specified config file doesn't exist
        """
        if config_path:
            self.config_path = config_path

        if self.config_path and Path(self.config_path).exists():
            self._config = self._load_from_file(self.config_path)
        else:
            # Create default configuration
            self._config = MetagenomicsConfig(input_dir=None)

        return self._config

    def _load_from_file(self, file_path: str) -> MetagenomicsConfig:
        """Load configuration from YAML or JSON file.

        Args:
            file_path: Path to configuration file

        Returns:
            MetagenomicsConfig: Validated configuration

        Raises:
            ValueError: If file format is unsupported or content is invalid
        """
        file_path_obj = Path(file_path)

        if file_path_obj.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported config format: {file_path_obj.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            with file_path_obj.open("r", encoding="utf-8") as f:
                if file_path_obj.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                elif file_path_obj.suffix.lower() == ".json":
                    data = json.load(f)

            # Handle empty or null config file
            if data is None:
                data = {}

            if not isinstance(data, dict):
                raise ValueError("Configuration file must contain a dictionary/object")

            return MetagenomicsConfig(**data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {file_path}: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON syntax in {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {file_path}: {str(e)}")

    def save_config(
        self,
        output_path: str,
        config: Optional[MetagenomicsConfig] = None,
        format: Optional[str] = None,
    ) -> None:
        """Save configuration to file.

        Args:
            output_path: Path where to save configuration
            config: Configuration to save (uses current if None)
            format: Force specific format ('yaml' or 'json')

        Raises:
            ValueError: If no configuration to save or invalid format
        """
        if config is None:
            config = self._config

        if config is None:
            raise ValueError(
                "No configuration to save. Load or create a configuration first."
            )

        output_path_obj = Path(output_path)

        # Determine format from extension or parameter
        if format:
            file_format = format.lower()
        else:
            file_format = output_path_obj.suffix.lower().lstrip(".")

        if file_format not in ["yaml", "yml", "json"]:
            raise ValueError(f"Unsupported output format: {file_format}")

        # Create output directory if needed
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path_obj.open("w", encoding="utf-8") as f:
                if file_format in ["yaml", "yml"]:
                    yaml.dump(
                        config.model_dump(),
                        f,
                        default_flow_style=False,
                        indent=2,
                        sort_keys=True,
                        allow_unicode=True,
                    )
                elif file_format == "json":
                    json.dump(
                        config.model_dump(),
                        f,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
        except Exception as e:
            raise ValueError(
                f"Failed to save configuration to {output_path_obj}: {str(e)}"
            )

    def validate_config(self, config_dict: Dict[str, Any]) -> MetagenomicsConfig:
        """Validate configuration dictionary.

        Args:
            config_dict: Configuration as dictionary

        Returns:
            MetagenomicsConfig: Validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            return MetagenomicsConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")

    @property
    def config(self) -> Optional[MetagenomicsConfig]:
        """Get current configuration."""
        return self._config

    def get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return os.path.join(os.getcwd(), "metagenomics_config.yaml")

    def create_example_config(self, output_path: str) -> None:
        """Create an example configuration file with documentation.

        Args:
            output_path: Where to save the example config
        """
        config = MetagenomicsConfig(input_dir=None)

        # Add helpful comments for YAML
        if output_path.endswith(".yaml") or output_path.endswith(".yml"):
            content = """# MetagenomicsOS Configuration File
# This is an example configuration with all available options

# Project identification
project_name: "metagenomics_project"
output_dir: "results"

# Database settings
databases:
  kraken2_db_path: null  # Path to Kraken2 database
  eggnog_db_path: null   # Path to EggNOG database
  cache_dir: ".cache"    # Local cache directory
  auto_download: true    # Auto-download missing databases

# Resource allocation
resources:
  threads: 1             # Number of CPU threads
  memory_gb: 4           # Memory limit in GB
  temp_dir: "/tmp"       # Temporary directory

# Workflow modules
workflow:
  quality_control: true      # Run QC analysis
  taxonomy: true            # Run taxonomic classification
  functional_analysis: true # Run functional annotation
  assembly: false           # Run genome assembly
  binning: false           # Run genome binning (requires assembly)

  # Quality control parameters
  min_read_length: 50      # Minimum read length
  min_quality: 20          # Minimum quality score

  # Output settings
  save_intermediate: false  # Keep intermediate files
  output_formats:          # Output formats to generate
    - "tsv"
    - "json"

# System settings
debug_mode: false        # Enable debug output
dry_run: false          # Show commands without execution
"""

            with open(output_path, "w") as f:
                f.write(content)
        else:
            # For JSON, save normally
            self.save_config(output_path, config)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        if not self._config:
            self.load_config()
        return dict(self._config.model_dump()) if self._config else {}

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        if not self._config:
            self.load_config()
        if self._config:
            # This is a simplified approach. A more robust implementation
            # would handle nested keys.
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                self.save_config(self.config_path or self.get_default_config_path())
            else:
                raise KeyError(f"Invalid configuration key: {key}")

    def reset_to_defaults(self) -> None:
        """Reset the configuration to default values."""
        self._config = MetagenomicsConfig()
        self.save_config(self.config_path or self.get_default_config_path())

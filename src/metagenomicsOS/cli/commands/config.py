# cli/commands/config.py
import click
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, cast

# Add jsonschema if you want validation (optional for now)
try:
    from jsonschema import validate, ValidationError

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class ConfigManager:
    """Manages configuration file operations with validation."""

    def __init__(self, config_path: str, format: str = "json"):
        """Initialize the ConfigManager.

        Args:
            config_path: The path to the configuration file.
            format: The format of the configuration file.
        """
        self.config_path = Path(config_path)
        self.format = format.lower()
        self.backup_path = self.config_path.with_suffix(
            f"{self.config_path.suffix}.backup"
        )

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            return {}

        try:
            with open(self.config_path, "r") as f:
                if self.format == "json":
                    return cast(Dict[str, Any], json.load(f))
                elif self.format in ["yaml", "yml"]:
                    return cast(Dict[str, Any], yaml.safe_load(f) or {})
                else:
                    raise ValueError(f"Unsupported format: {self.format}")
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            return {}

    def save_config(self, config: Dict[str, Any], create_backup: bool = True) -> bool:
        """Save configuration to file with backup."""
        try:
            # Create backup if file exists
            if create_backup and self.config_path.exists():
                shutil.copy2(self.config_path, self.backup_path)

            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_path = self.config_path.with_suffix(".tmp")

            with open(temp_path, "w") as f:
                if self.format == "json":
                    json.dump(config, f, indent=2, sort_keys=True)
                elif self.format in ["yaml", "yml"]:
                    yaml.safe_dump(config, f, default_flow_style=False, indent=2)

            # Atomic move
            temp_path.replace(self.config_path)
            return True

        except Exception as e:
            click.echo(f"Error saving config: {e}", err=True)
            if temp_path.exists():
                temp_path.unlink()
            return False

    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate configuration against schema."""
        if not schema:
            # Use default schema if none provided
            schema = self._get_default_schema()

        try:
            validate(instance=config, schema=schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _get_default_schema(self) -> Dict[str, Any]:
        """Define default configuration schema."""
        return {
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "name": {"type": "string"},
                        "user": {"type": "string"},
                    },
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        },
                        "file": {"type": "string"},
                    },
                },
                "api": {
                    "type": "object",
                    "properties": {
                        "base_url": {"type": "string", "format": "uri"},
                        "timeout": {"type": "integer", "minimum": 1},
                        "retries": {"type": "integer", "minimum": 0},
                    },
                },
            },
            "additionalProperties": True,
        }


@click.group()
@click.option(
    "--config-file", "-c", default="config.json", help="Path to configuration file"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml", "yml"]),
    default="json",
    help="Configuration file format",
)
@click.pass_context
def config(ctx, config_file, format):
    """Manage application configuration."""
    # Store config manager in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config_manager"] = ConfigManager(config_file, format)
    ctx.obj["config_file"] = config_file


@config.command()
@click.option("--section", "-s", help="Show only specific configuration section")
@click.option("--key", "-k", help="Show only specific key (requires --section)")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml", "table"]),
    default="json",
    help="Output format",
)
@click.pass_context
def view(ctx, section, key, format):
    """View current configuration settings."""
    config_manager = ctx.obj["config_manager"]
    config_data = config_manager.load_config()

    if not config_data:
        click.echo("No configuration found or config file is empty.")
        return

    # Filter by section and key if specified
    display_data = config_data

    if section:
        if section not in config_data:
            click.echo(f"Section '{section}' not found in configuration.", err=True)
            return
        display_data = {section: config_data[section]}

        if key:
            if key not in config_data[section]:
                click.echo(f"Key '{key}' not found in section '{section}'.", err=True)
                return
            display_data = {section: {key: config_data[section][key]}}

    # Display in requested format
    if format == "json":
        click.echo(json.dumps(display_data, indent=2, sort_keys=True))
    elif format == "yaml":
        click.echo(yaml.dump(display_data, default_flow_style=False, indent=2))
    elif format == "table":
        _display_as_table(display_data)


def _display_as_table(config_data: Dict[str, Any], prefix: str = ""):
    """Display configuration data as a formatted table."""
    for key, value in config_data.items():
        current_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            click.echo(f"{current_key}:")
            _display_as_table(value, current_key)
        else:
            click.echo(f"  {current_key}: {value}")


@config.command()
@click.option("--section", "-s", required=True, help="Configuration section to edit")
@click.option("--key", "-k", required=True, help="Configuration key to edit")
@click.option("--value", "-v", help="New value for the key")
@click.option(
    "--type",
    "-t",
    type=click.Choice(["string", "int", "float", "bool", "json"]),
    default="string",
    help="Value type for conversion",
)
@click.option("--delete", "--remove", is_flag=True, help="Delete the specified key")
@click.option(
    "--interactive", "-i", is_flag=True, help="Edit configuration interactively"
)
@click.option(
    "--validate-only", is_flag=True, help="Only validate changes without saving"
)
@click.pass_context
def edit(ctx, section, key, value, type, delete, interactive, validate_only):
    """Edit configuration settings with validation."""
    config_manager = ctx.obj["config_manager"]
    config_data = config_manager.load_config()

    # Initialize section if it doesn't exist
    if section not in config_data:
        config_data[section] = {}

    # Handle deletion
    if delete:
        if key in config_data[section]:
            del config_data[section][key]
            click.echo(f"Deleted {section}.{key}")
        else:
            click.echo(f"Key '{key}' not found in section '{section}'.", err=True)
            return

    # Handle interactive editing
    elif interactive:
        current_value = config_data[section].get(key, "")
        if isinstance(current_value, (dict, list)):
            current_value = json.dumps(current_value, indent=2)

        new_value = click.edit(
            f"# Edit {section}.{key}\n# Current value:\n{current_value}"
        )
        if new_value is None:
            click.echo("Edit cancelled.")
            return

        # Try to parse as JSON first, fall back to string
        try:
            config_data[section][key] = json.loads(new_value.strip())
        except json.JSONDecodeError:
            config_data[section][key] = new_value.strip()

    # Handle direct value setting
    else:
        if value is None:
            value = click.prompt(f"Enter value for {section}.{key}")

        # Convert value based on type
        converted_value = _convert_value(value, type)
        if converted_value is None:
            return

        config_data[section][key] = converted_value

    # Validate configuration
    is_valid, error_msg = config_manager.validate_config(config_data)
    if not is_valid:
        click.echo(f"Configuration validation failed: {error_msg}", err=True)
        if not click.confirm("Do you want to save anyway?"):
            return

    if validate_only:
        if is_valid:
            click.echo("Configuration validation passed.")
        return

    # Save configuration
    if config_manager.save_config(config_data):
        click.echo(f"Successfully updated {section}.{key}")

        # Show the change
        if not delete:
            click.echo(f"New value: {config_data[section][key]}")
    else:
        click.echo("Failed to save configuration.", err=True)


def _convert_value(value: str, value_type: str) -> Any:
    """Convert string value to specified type."""
    try:
        if value_type == "int":
            return int(value)
        elif value_type == "float":
            return float(value)
        elif value_type == "bool":
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == "json":
            return json.loads(value)
        else:  # string
            return value
    except ValueError as e:
        click.echo(f"Error converting value to {value_type}: {e}", err=True)
        return None


@config.command(name="validate")
@click.pass_context
def validate_config_command(ctx):
    """Validate current configuration against schema."""
    config_manager = ctx.obj["config_manager"]
    config_data = config_manager.load_config()

    if not config_data:
        click.echo("No configuration found to validate.")
        return

    is_valid, error_msg = config_manager.validate_config(config_data)

    if is_valid:
        click.echo("✓ Configuration is valid.")
    else:
        click.echo(f"✗ Configuration validation failed: {error_msg}", err=True)
        ctx.exit(1)


@config.command()
@click.option("--force", "-f", is_flag=True, help="Force reset without confirmation")
@click.pass_context
def reset(ctx, force):
    """Reset configuration to defaults."""
    config_file = ctx.obj["config_file"]

    if not force:
        if not click.confirm(f"Are you sure you want to reset {config_file}?"):
            return

    config_manager = ctx.obj["config_manager"]

    # Create default configuration
    default_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp_db",
            "user": "myapp_user",
        },
        "logging": {"level": "INFO", "file": "app.log"},
        "api": {"base_url": "https://api.example.com", "timeout": 30, "retries": 3},
    }

    if config_manager.save_config(default_config):
        click.echo("Configuration reset to defaults.")
    else:
        click.echo("Failed to reset configuration.", err=True)


@config.command()
@click.pass_context
def backup(ctx):
    """Create a backup of current configuration."""
    config_manager = ctx.obj["config_manager"]

    if not config_manager.config_path.exists():
        click.echo("No configuration file to backup.")
        return

    try:
        backup_path = config_manager.config_path.with_suffix(
            f"{config_manager.config_path.suffix}.backup"
        )
        shutil.copy2(config_manager.config_path, backup_path)
        click.echo(f"Backup created: {backup_path}")
    except Exception as e:
        click.echo(f"Failed to create backup: {e}", err=True)


@config.command()
@click.pass_context
def schema(ctx):
    """Display the configuration schema."""
    config_manager = ctx.obj["config_manager"]
    schema_data = config_manager._get_default_schema()
    click.echo(json.dumps(schema_data, indent=2))

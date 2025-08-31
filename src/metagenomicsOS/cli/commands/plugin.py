# cli/commands/plugin.py
from __future__ import annotations
from pathlib import Path
from typing import Literal, Dict, Any, List
import json
import shutil
import importlib.util
from datetime import datetime
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Extensibility and plugins")

PluginType = Literal["command", "workflow", "formatter", "validator"]


@app.command("list")
def list_plugins(
    plugin_dir: Path | None = typer.Option(
        None, "--dir", help="Plugin directory to scan"
    ),
    installed_only: bool = typer.Option(
        False, "--installed", help="Show only installed plugins"
    ),
    plugin_type: PluginType | None = typer.Option(
        None, "--type", help="Filter by plugin type"
    ),
) -> None:
    """List available and installed plugins."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if plugin_dir is None:
        plugin_dir = Path(cfg.data_dir) / "plugins"

    plugins = _discover_plugins(plugin_dir, installed_only, plugin_type)

    if not plugins:
        typer.echo("No plugins found.")
        if not installed_only:
            typer.echo("Use 'plugin develop' to create your first plugin.")
        return

    typer.echo(
        f"{'Installed' if installed_only else 'Available'} Plugins ({len(plugins)} found):"
    )
    typer.echo("-" * 80)
    typer.echo(
        f"{'Name':<20} {'Type':<12} {'Version':<10} {'Status':<12} {'Description':<20}"
    )
    typer.echo("-" * 80)

    for plugin in plugins:
        status = "✅ Active" if plugin["installed"] else "⚪ Available"
        typer.echo(
            f"{plugin['name']:<20} {plugin['type']:<12} {plugin['version']:<10} "
            f"{status:<12} {plugin['description'][:20]}"
        )


def _discover_plugins(
    plugin_dir: Path, installed_only: bool, type_filter: PluginType | None
) -> list[dict]:
    """Discover plugins in directory."""
    plugins: List[Dict[str, Any]] = []

    if not plugin_dir.exists():
        return plugins

    for plugin_path in plugin_dir.iterdir():
        if not plugin_path.is_dir():
            continue

        manifest_file = plugin_path / "plugin.json"
        if not manifest_file.exists():
            continue

        try:
            with manifest_file.open() as f:
                manifest = json.load(f)

            plugin_info = {
                "name": manifest.get("name", plugin_path.name),
                "path": str(plugin_path),
                "type": manifest.get("type", "unknown"),
                "version": manifest.get("version", "0.1.0"),
                "description": manifest.get("description", "No description"),
                "author": manifest.get("author", "Unknown"),
                "installed": (plugin_path / "__pycache__").exists()
                or (plugin_path / "installed.flag").exists(),
            }

            # Apply filters
            if type_filter is not None and plugin_info["type"] != type_filter:
                continue

            if installed_only and not plugin_info["installed"]:
                continue

            plugins.append(plugin_info)

        except Exception:  # nosec
            # Skip plugins with invalid manifests
            continue

    return sorted(plugins, key=lambda x: x["name"])


@app.command("install")
def install_plugin(
    plugin_path: Path = typer.Option(
        ..., "--path", "-p", exists=True, help="Plugin directory path"
    ),
    plugins_dir: Path | None = typer.Option(
        None, "--plugins-dir", help="Installation directory"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force reinstall if already exists"
    ),
) -> None:
    """Install a plugin into the system."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if plugins_dir is None:
        plugins_dir = Path(cfg.data_dir) / "plugins"

    plugins_dir.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would install plugin: {plugin_path}")
        typer.echo(f"[DRY-RUN] Installation dir: {plugins_dir}")
        return

    # Validate plugin
    manifest_file = plugin_path / "plugin.json"
    if not manifest_file.exists():
        typer.echo("❌ Invalid plugin: missing plugin.json manifest")
        raise typer.Exit(code=1)

    with manifest_file.open() as f:
        manifest = json.load(f)

    plugin_name = manifest.get("name", plugin_path.name)
    target_dir = plugins_dir / plugin_name

    # Check if already installed
    if target_dir.exists() and not force:
        typer.echo(
            f"❌ Plugin '{plugin_name}' already installed. Use --force to reinstall."
        )
        raise typer.Exit(code=1)

    typer.echo(f"📦 Installing plugin: {plugin_name}")
    typer.echo(f"Source: {plugin_path}")
    typer.echo(f"Target: {target_dir}")

    # Install plugin
    _install_plugin_files(plugin_path, target_dir, manifest, force)

    typer.echo(f"✅ Plugin '{plugin_name}' installed successfully!")
    typer.echo(f"Version: {manifest.get('version', 'unknown')}")
    typer.echo(f"Type: {manifest.get('type', 'unknown')}")


def _install_plugin_files(
    source: Path, target: Path, manifest: dict, force: bool
) -> None:
    """Copy plugin files to target directory."""
    if target.exists() and force:
        shutil.rmtree(target)

    # Copy plugin directory
    shutil.copytree(source, target, dirs_exist_ok=force)

    # Create installation flag
    (target / "installed.flag").touch()

    # Update manifest with installation info
    manifest["installed_at"] = datetime.now().isoformat()
    manifest["installed_from"] = str(source)

    with (target / "plugin.json").open("w") as f:
        json.dump(manifest, f, indent=2)


@app.command("uninstall")
def uninstall_plugin(
    plugin_name: str = typer.Option(
        ..., "--name", "-n", help="Plugin name to uninstall"
    ),
    plugins_dir: Path | None = typer.Option(
        None, "--plugins-dir", help="Plugins directory"
    ),
    keep_data: bool = typer.Option(False, "--keep-data", help="Keep plugin data files"),
) -> None:
    """Uninstall a plugin from the system."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if plugins_dir is None:
        plugins_dir = Path(cfg.data_dir) / "plugins"

    plugin_dir = plugins_dir / plugin_name

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would uninstall plugin: {plugin_name}")
        return

    if not plugin_dir.exists():
        typer.echo(f"❌ Plugin '{plugin_name}' not found")
        raise typer.Exit(code=1)

    typer.echo(f"🗑️  Uninstalling plugin: {plugin_name}")

    # Remove plugin directory
    if keep_data:
        # Remove only code files, keep data
        for item in plugin_dir.iterdir():
            if item.name.endswith((".py", ".pyc")) or item.name == "__pycache__":
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    else:
        # Remove entire plugin directory
        shutil.rmtree(plugin_dir)

    typer.echo(f"✅ Plugin '{plugin_name}' uninstalled successfully!")


@app.command("develop")
def develop_plugin(
    name: str = typer.Option(..., "--name", "-n", help="Plugin name"),
    plugin_type: PluginType = typer.Option(
        "command", "--type", "-t", help="Plugin type"
    ),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    author: str = typer.Option("Developer", "--author", help="Plugin author"),
    description: str = typer.Option(
        "A custom plugin", "--description", help="Plugin description"
    ),
) -> None:
    """Create a new plugin scaffold."""
    ctx = get_context()

    plugin_dir = output_dir / name

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would create plugin: {name}")
        typer.echo(f"[DRY-RUN] Type: {plugin_type}")
        typer.echo(f"[DRY-RUN] Output: {plugin_dir}")
        return

    if plugin_dir.exists():
        typer.echo(f"❌ Plugin directory already exists: {plugin_dir}")
        raise typer.Exit(code=1)

    typer.echo(f"🛠️  Creating plugin scaffold: {name}")
    typer.echo(f"Type: {plugin_type}")
    typer.echo(f"Directory: {plugin_dir}")

    _create_plugin_scaffold(name, plugin_type, plugin_dir, author, description)

    typer.echo("✅ Plugin scaffold created successfully!")
    typer.echo(f"📁 Plugin directory: {plugin_dir}")
    typer.echo("🚀 Next steps:")
    typer.echo(f"   1. Edit {plugin_dir / 'main.py'} to implement your plugin")
    typer.echo(
        f"   2. Test with: python -m metagenomicsOS.cli.main plugin test --path {plugin_dir}"
    )
    typer.echo(
        f"   3. Install with: python -m metagenomicsOS.cli.main plugin install --path {plugin_dir}"
    )


def _create_plugin_scaffold(
    name: str, plugin_type: str, output_dir: Path, author: str, description: str
) -> None:
    """Create plugin directory structure and template files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plugin manifest
    manifest = {
        "name": name,
        "version": "0.1.0",
        "type": plugin_type,
        "description": description,
        "author": author,
        "created_at": datetime.now().isoformat(),
        "entry_point": "main.py",
        "dependencies": [],
        "cli_commands": [] if plugin_type == "command" else None,
        "workflow_steps": [] if plugin_type == "workflow" else None,
    }

    with (output_dir / "plugin.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Create main plugin file based on type
    if plugin_type == "command":
        main_content = _get_command_plugin_template(name)
    elif plugin_type == "workflow":
        main_content = _get_workflow_plugin_template(name)
    elif plugin_type == "formatter":
        main_content = _get_formatter_plugin_template(name)
    else:  # validator
        main_content = _get_validator_plugin_template(name)

    with (output_dir / "main.py").open("w") as f:
        f.write(main_content)

    # Create README
    readme_content = f"""# {name} Plugin

{description}

## Type
{plugin_type}

## Author
{author}

## Installation
python -m metagenomicsOS.cli.main plugin install --path

## Usage
See main.py for implementation details.

## Development
1. Modify main.py to implement your plugin functionality
2. Update plugin.json if needed
3. Test with the CLI plugin test command
4. Install and use with the main CLI

## License
Add your license information here.
"""

    with (output_dir / "README.md").open("w") as f:
        f.write(readme_content)

    # Create test file
    test_content = f'''"""Tests for {name} plugin."""

import unittest
from pathlib import Path

class Test{name.title().replace("_", "")}Plugin(unittest.TestCase):
    """Test cases for {name} plugin."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_plugin_loads(self):
        """Test that plugin loads correctly."""
        # Add your tests here
        self.assertTrue(True)

    def test_plugin_functionality(self):
        """Test main plugin functionality."""
        # Add specific functionality tests
        pass

if __name__ == "__main__":
    unittest.main()
'''

    with (output_dir / "test_plugin.py").open("w") as f:
        f.write(test_content)


def _get_command_plugin_template(name: str) -> str:
    """Generate command plugin template."""
    return f'''"""
{name} Command Plugin

This plugin adds custom commands to the MetagenomicsOS CLI.
"""

import typer
from pathlib import Path
from metagenomicsOS.cli.core.context import get_context

# Create the plugin's command app
app = typer.Typer(help="{name} plugin commands")

@app.command("hello")
def hello_command(
    name: str = typer.Option("World", "--name", "-n", help="Name to greet"),
    count: int = typer.Option(1, "--count", "-c", help="Number of greetings")
) -> None:
    """Say hello - example command from {name} plugin."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would greet {{name}} {{count}} times")
        return

    for i in range(count):
        typer.echo(f"Hello {{name}} from {name} plugin! ({{i+1}}/{{count}})")

@app.command("info")
def info_command() -> None:
    """Show plugin information."""
    typer.echo(f"Plugin: {name}")
    typer.echo(f"Type: Command Plugin")
    typer.echo(f"Status: Active")

# Plugin initialization function (called when plugin is loaded)
def initialize_plugin():
    """Initialize the plugin."""
    print(f"Initializing {name} plugin...")

# Plugin cleanup function (called when plugin is unloaded)
def cleanup_plugin():
    """Cleanup plugin resources."""
    print(f"Cleaning up {name} plugin...")

# Export the command app for CLI integration
__all__ = ["app", "initialize_plugin", "cleanup_plugin"]
'''


def _get_workflow_plugin_template(name: str) -> str:
    """Generate workflow plugin template."""
    return f'''"""
{name} Workflow Plugin

This plugin adds custom workflow steps to the MetagenomicsOS pipeline.
"""

from pathlib import Path
from typing import Dict, Any
import json

class {name.title().replace("_", "")}WorkflowStep:
    """Custom workflow step implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "{name}_step"

    def execute(self, input_data: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step."""

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Example workflow step implementation
        result = {{
            "step_name": self.name,
            "input_path": str(input_data),
            "output_path": str(output_dir),
            "status": "completed",
            "files_processed": 1,
            "custom_metric": 42.0
        }}

        # Write step results
        result_file = output_dir / f"{{self.name}}_results.json"
        with result_file.open("w") as f:
            json.dump(result, f, indent=2)

        return result

    def validate_input(self, input_data: Path) -> bool:
        """Validate input data for this step."""
        return input_data.exists()

    def get_requirements(self) -> Dict[str, Any]:
        """Get step requirements (memory, CPU, etc.)."""
        return {{
            "min_memory_gb": 2,
            "min_cpu_cores": 1,
            "estimated_runtime_minutes": 10
        }}

# Plugin workflow registration
def get_workflow_steps():
    """Return available workflow steps from this plugin."""
    return {{
        "{name}_step": {name.title().replace("_", "")}WorkflowStep
    }}

# Plugin initialization
def initialize_plugin():
    """Initialize workflow plugin."""
    print(f"Initializing {name} workflow plugin...")

def cleanup_plugin():
    """Cleanup workflow plugin."""
    print(f"Cleaning up {name} workflow plugin...")

__all__ = ["get_workflow_steps", "initialize_plugin", "cleanup_plugin"]
'''


def _get_formatter_plugin_template(name: str) -> str:
    """Generate formatter plugin template."""
    return f'''"""
{name} Formatter Plugin

This plugin adds custom output formatters to MetagenomicsOS.
"""

from pathlib import Path
from typing import Dict, Any, List
import json

class {name.title().replace("_", "")}Formatter:
    """Custom output formatter."""

    def __init__(self):
        self.name = "{name}_formatter"
        self.supported_formats = ["custom", "extended_json"]

    def format_results(self, data: Dict[str, Any], output_format: str) -> str:
        """Format results data."""

        if output_format == "custom":
            return self._format_custom(data)
        elif output_format == "extended_json":
            return self._format_extended_json(data)
        else:
            raise ValueError(f"Unsupported format: {{output_format}}")

    def _format_custom(self, data: Dict[str, Any]) -> str:
        """Custom formatting implementation."""
        lines = []
        lines.append(f"=== {name.title()} Results ===")

        for key, value in data.items():
            lines.append(f"{{key}}: {{value}}")

        return "\\n".join(lines)

    def _format_extended_json(self, data: Dict[str, Any]) -> str:
        """Extended JSON formatting with metadata."""
        extended_data = {{
            "formatter": self.name,
            "timestamp": "2025-08-26T16:00:00Z",
            "version": "1.0.0",
            "data": data
        }}
        return json.dumps(extended_data, indent=2)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return self.supported_formats

# Plugin formatter registration
def get_formatters():
    """Return available formatters from this plugin."""
    return {{
        "{name}": {name.title().replace("_", "")}Formatter()
    }}

def initialize_plugin():
    """Initialize formatter plugin."""
    print(f"Initializing {name} formatter plugin...")

def cleanup_plugin():
    """Cleanup formatter plugin."""
    print(f"Cleaning up {name} formatter plugin...")

__all__ = ["get_formatters", "initialize_plugin", "cleanup_plugin"]
'''


def _get_validator_plugin_template(name: str) -> str:
    """Generate validator plugin template."""
    return f'''"""
{name} Validator Plugin

This plugin adds custom data validation to MetagenomicsOS.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

class {name.title().replace("_", "")}Validator:
    """Custom data validator."""

    def __init__(self):
        self.name = "{name}_validator"
        self.supported_types = ["custom_format", "extended_fasta"]

    def validate_data(self, data_path: Path, data_type: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate data file.

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if not data_path.exists():
            errors.append(f"Data file does not exist: {{data_path}}")
            return False, errors, warnings

        if data_type == "custom_format":
            return self._validate_custom_format(data_path)
        elif data_type == "extended_fasta":
            return self._validate_extended_fasta(data_path)
        else:
            errors.append(f"Unsupported data type: {{data_type}}")
            return False, errors, warnings

    def _validate_custom_format(self, data_path: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate custom format."""
        errors = []
        warnings = []

        try:
            with data_path.open("r") as f:
                lines = f.readlines()

            if len(lines) == 0:
                errors.append("File is empty")
            elif len(lines) < 10:
                warnings.append(f"File has only {{len(lines)}} lines - this seems small")

            # Add your custom validation logic here

        except Exception as e:
            errors.append(f"Error reading file: {{str(e)}}")

        return len(errors) == 0, errors, warnings

    def _validate_extended_fasta(self, data_path: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate extended FASTA format."""
        errors = []
        warnings = []

        try:
            with data_path.open("r") as f:
                line_count = 0
                header_count = 0

                for line in f:
                    line_count += 1
                    if line.startswith(">"):
                        header_count += 1
                        # Add extended FASTA header validation
                        if len(line.strip()) < 5:
                            warnings.append(f"Short header at line {{line_count}}")

            if header_count == 0:
                errors.append("No FASTA headers found")

        except Exception as e:
            errors.append(f"Error reading FASTA file: {{str(e)}}")

        return len(errors) == 0, errors, warnings

    def get_supported_types(self) -> List[str]:
        """Get list of supported data types."""
        return self.supported_types

# Plugin validator registration
def get_validators():
    """Return available validators from this plugin."""
    return {{
        "{name}": {name.title().replace("_", "")}Validator()
    }}

def initialize_plugin():
    """Initialize validator plugin."""
    print(f"Initializing {name} validator plugin...")

def cleanup_plugin():
    """Cleanup validator plugin."""
    print(f"Cleaning up {name} validator plugin...")

__all__ = ["get_validators", "initialize_plugin", "cleanup_plugin"]
'''


@app.command("test")
def test_plugin(
    plugin_path: Path = typer.Option(
        ..., "--path", "-p", exists=True, help="Plugin directory to test"
    ),
    test_type: str = typer.Option(
        "basic", "--type", help="Test type: basic, full, integration"
    ),
) -> None:
    """Test plugin functionality and compatibility."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would test plugin: {plugin_path}")
        return

    typer.echo(f"🧪 Testing plugin: {plugin_path.name}")
    typer.echo(f"Test type: {test_type}")
    typer.echo("-" * 40)

    test_results = _run_plugin_tests(plugin_path, test_type)

    # Display test results
    typer.echo("📊 Test Results:")
    typer.echo(f"  Tests run: {test_results['tests_run']}")
    typer.echo(f"  Passed: {test_results['passed']}")
    typer.echo(f"  Failed: {test_results['failed']}")
    typer.echo(f"  Warnings: {test_results['warnings']}")

    if test_results["failed"] == 0:
        typer.echo("✅ All tests passed!")
    else:
        typer.echo(f"❌ {test_results['failed']} tests failed")
        for error in test_results["errors"]:
            typer.echo(f"   • {error}")

    if test_results["warnings"] > 0:
        typer.echo("⚠️  Warnings:")
        for warning in test_results["warning_messages"]:
            typer.echo(f"   • {warning}")


def _run_plugin_tests(plugin_path: Path, test_type: str) -> dict:
    """Run plugin tests and return results."""
    test_results: Dict[str, Any] = {
        "tests_run": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "errors": [],
        "warning_messages": [],
    }

    # Test 1: Plugin structure
    test_results["tests_run"] += 1
    if (plugin_path / "plugin.json").exists():
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1
        test_results["errors"].append("Missing plugin.json manifest")

    # Test 2: Main module exists
    test_results["tests_run"] += 1
    if (plugin_path / "main.py").exists():
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1
        test_results["errors"].append("Missing main.py entry point")

    # Test 3: Plugin manifest is valid JSON
    test_results["tests_run"] += 1
    try:
        with (plugin_path / "plugin.json").open() as f:
            manifest = json.load(f)
        test_results["passed"] += 1

        # Check required fields
        required_fields = ["name", "version", "type"]
        for field in required_fields:
            if field not in manifest:
                test_results["warnings"] += 1
                test_results["warning_messages"].append(
                    f"Missing recommended field: {field}"
                )

    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Invalid plugin.json: {str(e)}")

    # Test 4: Python syntax check
    test_results["tests_run"] += 1
    try:
        spec = importlib.util.spec_from_file_location(
            "plugin_test", plugin_path / "main.py"
        )
        if spec and spec.loader:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1
            test_results["errors"].append("Cannot load main.py module")
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Python syntax error in main.py: {str(e)}")

    # Additional tests for full test type
    if test_type == "full":
        # Test 5: Check for test file
        test_results["tests_run"] += 1
        if (plugin_path / "test_plugin.py").exists():
            test_results["passed"] += 1
        else:
            test_results["warnings"] += 1
            test_results["warning_messages"].append(
                "No test file found (test_plugin.py)"
            )

        # Test 6: Check for documentation
        test_results["tests_run"] += 1
        if (plugin_path / "README.md").exists():
            test_results["passed"] += 1
        else:
            test_results["warnings"] += 1
            test_results["warning_messages"].append("No README.md documentation found")

    return test_results


@app.command("info")
def plugin_info(
    plugin_name: str = typer.Option(..., "--name", "-n", help="Plugin name"),
    plugins_dir: Path | None = typer.Option(None, "--dir", help="Plugins directory"),
) -> None:
    """Show detailed information about a plugin."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if plugins_dir is None:
        plugins_dir = Path(cfg.data_dir) / "plugins"

    plugin_dir = plugins_dir / plugin_name

    if not plugin_dir.exists():
        typer.echo(f"❌ Plugin '{plugin_name}' not found")
        raise typer.Exit(code=1)

    manifest_file = plugin_dir / "plugin.json"
    if not manifest_file.exists():
        typer.echo(f"❌ Plugin '{plugin_name}' has no valid manifest")
        raise typer.Exit(code=1)

    with manifest_file.open() as f:
        manifest = json.load(f)

    # Display plugin information
    typer.echo(f"📦 Plugin Information: {plugin_name}")
    typer.echo("=" * 50)
    typer.echo(f"Name: {manifest.get('name', plugin_name)}")
    typer.echo(f"Version: {manifest.get('version', 'unknown')}")
    typer.echo(f"Type: {manifest.get('type', 'unknown')}")
    typer.echo(f"Author: {manifest.get('author', 'unknown')}")
    typer.echo(f"Description: {manifest.get('description', 'No description')}")

    if "created_at" in manifest:
        typer.echo(f"Created: {manifest['created_at'][:10]}")

    if "installed_at" in manifest:
        typer.echo(f"Installed: {manifest['installed_at'][:10]}")

    typer.echo(f"Path: {plugin_dir}")

    # Show dependencies if any
    if "dependencies" in manifest and manifest["dependencies"]:
        typer.echo("\nDependencies:")
        for dep in manifest["dependencies"]:
            typer.echo(f"  • {dep}")

    # Show CLI commands if command plugin
    if manifest.get("type") == "command" and "cli_commands" in manifest:
        if manifest["cli_commands"]:
            typer.echo("\nCLI Commands:")
            for cmd in manifest["cli_commands"]:
                typer.echo(f"  • {cmd}")

    # Show workflow steps if workflow plugin
    if manifest.get("type") == "workflow" and "workflow_steps" in manifest:
        if manifest["workflow_steps"]:
            typer.echo("\nWorkflow Steps:")
            for step in manifest["workflow_steps"]:
                typer.echo(f"  • {step}")

    # Check plugin status
    is_installed = (plugin_dir / "installed.flag").exists()
    status = (
        "✅ Installed and Active" if is_installed else "⚪ Available but not installed"
    )
    typer.echo(f"\nStatus: {status}")

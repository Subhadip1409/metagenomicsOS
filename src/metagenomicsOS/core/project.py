"""core/project.py.

Project initialization and management.
Business logic for CLI delegation.
"""

from pathlib import Path
import yaml
import logging


class ProjectInitializer:
    """Initialize new metagenomics analysis projects."""

    def __init__(self, template: str = "basic"):
        """Initialize the ProjectInitializer.

        Args:
            template: The project template to use.
        """
        self.template = template
        self.logger = logging.getLogger("ProjectInitializer")

    def create_project(self, project_name: str, directory: Path) -> Path:
        """Create a new project with directory structure and config files.

        Args:
            project_name: Name of the project
            directory: Directory where project will be created

        Returns:
            Path: Path to created project directory
        """
        self.logger.info(f"Creating project: {project_name} in {directory}")

        # Create project directory
        project_path = directory / project_name
        project_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (project_path / "data").mkdir(exist_ok=True)
        (project_path / "results").mkdir(exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)

        # Create config file
        self._create_config_file(project_path, project_name)

        # Create README
        self._create_readme(project_path, project_name)

        # Create sample analysis script if advanced template
        if self.template == "advanced":
            self._create_analysis_script(project_path)

        self.logger.info(f"Project created successfully at: {project_path}")
        return project_path

    def _create_config_file(self, project_path: Path, project_name: str):
        """Create project configuration file."""
        config_data = {
            "project_name": project_name,
            "data_dir": "./data",
            "output_dir": "./results",
            "log_dir": "./logs",
            "default_method": "kraken2",
            "threads": 4,
            "database": "",
            "quality_threshold": 20,
            "min_length": 50,
        }

        config_file = project_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

    def _create_readme(self, project_path: Path, project_name: str):
        """Create project README file."""
        readme_content = f"""# {project_name}

MetagenomicsOS Analysis Project

## Directory Structure

- `data/` - Input sequencing data files
- `results/` - Analysis outputs and reports
- `logs/` - Log files from analysis runs
- `config.yaml` - Project configuration

## Usage

1. Place your FASTQ files in the `data/` directory
2. Edit `config.yaml` to set your preferences
3. Run analysis:

"""
        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)

    def _create_analysis_script(self, project_path: Path):
        """Create a sample analysis script."""
        script_content = """
import os

print("Hello from your new analysis script!")

# You can add your analysis code here.
# For example, you can use the metagenomicsOS API
# to run an analysis:
#
# from metagenomicsOS.core.analysis import AnalysisEngine
#
# engine = AnalysisEngine()
# engine.run_analysis("path/to/your/data.fastq")

"""
        with open(project_path / "analysis.py", "w") as f:
            f.write(script_content)

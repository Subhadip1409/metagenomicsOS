#!/usr/bin/env python3
"""MetagenomicsOS CLI - Main Entry Point.

A command-line interface for metagenomics data analysis and processing.
"""

import typer
from typing import Optional
from pathlib import Path

# Import core modules for business logic delegation
from metagenomicsOS.core.analysis import AnalysisEngine
from metagenomicsOS.core.data_processing import DataProcessor
from metagenomicsOS.core.utils import setup_logging, validate_input_file

# Create the main Typer application
app = typer.Typer(
    name="metagenomics-os",
    help="MetagenomicsOS: A comprehensive toolkit for metagenomic data analysis",
    add_completion=False,  # Can be enabled later
    rich_markup_mode="rich",  # Enable rich text formatting
)


# Global options for all commands
def version_callback(value: bool):
    """Show version and exit."""
    if value:
        typer.echo("MetagenomicsOS v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
    log_level: str = typer.Option("INFO", "--log-level", help="Set logging level"),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """Metagenomicsos CLI - A toolkit for metagenomic analysis.

    This CLI provides commands for processing, analyzing, and visualizing
    metagenomic sequencing data.
    """
    # Setup logging based on options
    setup_logging(verbose=verbose, quiet=quiet, level=log_level)


# Continue in cli/main.py


@app.command()
def analyze(
    input_file: Path = typer.Argument(..., help="Input FASTQ file path", exists=True),
    output_dir: Path = typer.Option(
        Path("./output"), "--output", "-o", help="Output directory"
    ),
    database: Optional[str] = typer.Option(
        None, "--database", "-d", help="Reference database path"
    ),
    threads: int = typer.Option(4, "--threads", "-t", help="Number of threads to use"),
    method: str = typer.Option(
        "kraken2", "--method", "-m", help="Analysis method (kraken2, minimap2)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing output"
    ),
):
    """Analyze metagenomic sequences for taxonomic classification.

    This command processes FASTQ files and performs taxonomic classification
    using the specified method and reference database.
    """
    try:
        # Validate inputs (thin CLI logic)
        if not validate_input_file(input_file):
            typer.echo(f"Error: Invalid input file: {input_file}", err=True)
            raise typer.Exit(1)

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Starting analysis of {input_file}")
        typer.echo(f"Using method: {method}")
        typer.echo(f"Output directory: {output_dir}")

        # Delegate to core business logic
        analysis_engine = AnalysisEngine(
            method=method, database=database, threads=threads, output_dir=output_dir
        )

        # Run the actual analysis
        results = analysis_engine.run_analysis(input_file, force=force)

        typer.echo("✅ Analysis completed successfully!")
        typer.echo(f"Results saved to: {results.output_path}")

    except Exception as e:
        typer.echo(f"❌ Error during analysis: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input file to process", exists=True),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    quality_threshold: int = typer.Option(
        20, "--quality", "-q", help="Minimum quality score"
    ),
    min_length: int = typer.Option(50, "--min-length", help="Minimum read length"),
    trim_adapters: bool = typer.Option(
        False, "--trim-adapters", help="Remove adapter sequences"
    ),
):
    """Process and filter sequencing data.

    Perform quality control, trimming, and filtering of sequencing reads.
    """
    try:
        typer.echo(f"Processing file: {input_file}")

        # Delegate to core processing logic
        processor = DataProcessor(
            quality_threshold=quality_threshold,
            min_length=min_length,
            trim_adapters=trim_adapters,
        )

        # Run processing
        result = processor.process_file(input_file, output_file)

        typer.echo("✅ Processing completed!")
        typer.echo(f"Processed {result.total_reads} reads")
        typer.echo(f"Kept {result.kept_reads} reads ({result.keep_percentage:.1f}%)")

    except Exception as e:
        typer.echo(f"❌ Error during processing: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    input_dir: Path = typer.Argument(
        ..., help="Directory with analysis results", exists=True
    ),
    output_file: Path = typer.Option(
        Path("report.html"), "--output", "-o", help="Output report file"
    ),
    format_type: str = typer.Option(
        "html", "--format", help="Report format (html, pdf, json)"
    ),
    include_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Include visualization plots"
    ),
):
    """Generate analysis reports and visualizations.

    Create comprehensive reports from analysis results including
    taxonomic profiles, statistics, and visualizations.
    """
    try:
        typer.echo(f"Generating report from: {input_dir}")

        # Import report generator from core
        from metagenomicsOS.core.reporting import ReportGenerator

        # Delegate to core reporting logic
        report_gen = ReportGenerator(
            format_type=format_type, include_plots=include_plots
        )

        # Generate report
        report_path = report_gen.generate_report(input_dir, output_file)

        typer.echo(f"✅ Report generated: {report_path}")

    except Exception as e:
        typer.echo(f"❌ Error generating report: {str(e)}", err=True)
        raise typer.Exit(1)


# Continue in cli/main.py


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: show, set, reset"),
    key: Optional[str] = typer.Option(None, "--key", help="Configuration key"),
    value: Optional[str] = typer.Option(None, "--value", help="Configuration value"),
):
    """Manage application configuration.

    Show current settings, set new values, or reset to defaults.
    """
    try:
        from metagenomicsOS.core.config_manager import ConfigManager

        config_manager = ConfigManager()

        if action == "show":
            # Show current configuration
            config_data = config_manager.get_all_config()
            typer.echo("Current Configuration:")
            for k, v in config_data.items():
                typer.echo(f"  {k}: {v}")

        elif action == "set":
            if not key or not value:
                typer.echo(
                    "Error: Both --key and --value are required for 'set'", err=True
                )
                raise typer.Exit(1)
            config_manager.set_config(key, value)
            typer.echo(f"✅ Set {key} = {value}")

        elif action == "reset":
            config_manager.reset_to_defaults()
            typer.echo("✅ Configuration reset to defaults")

        else:
            typer.echo(f"Error: Unknown action '{action}'", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Configuration error: {str(e)}", err=True)
        raise typer.Exit(1)


# Continue in cli/main.py


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project"),
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d", help="Project directory (default: current)"
    ),
    template: str = typer.Option(
        "basic", "--template", help="Project template (basic, advanced)"
    ),
):
    """Initialize a new metagenomics analysis project.

    Create project structure with configuration files and templates.
    """
    try:
        from metagenomicsOS.core.project import ProjectInitializer

        if directory is None:
            directory = Path.cwd() / project_name

        typer.echo(f"Initializing project: {project_name}")
        typer.echo(f"Directory: {directory}")

        # Delegate to core project initialization
        initializer = ProjectInitializer(template=template)
        project_path = initializer.create_project(project_name, directory)

        typer.echo(f"✅ Project initialized at: {project_path}")
        typer.echo("Next steps:")
        typer.echo("  1. cd into the project directory")
        typer.echo("  2. Edit config.yaml to set your preferences")
        typer.echo("  3. Run 'metagenomics-os analyze --help' for usage")

    except Exception as e:
        typer.echo(f"❌ Project initialization failed: {str(e)}", err=True)
        raise typer.Exit(1)


def main_cli():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main_cli()

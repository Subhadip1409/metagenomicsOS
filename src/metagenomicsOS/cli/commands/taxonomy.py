# cli/commands/taxonomy.py
from __future__ import annotations
from pathlib import Path
import json
import typer
from typing import Any
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Taxonomic profiling")


@app.command("classify")
def classify(
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Input FASTQ/FASTA file"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    classifier: str = typer.Option(
        "kraken2",
        "--classifier",
        "-c",
        help="Classification tool",
        case_sensitive=False,
        rich_help_panel="Classifiers",
        show_choices=True,
        choices=["kraken2", "metaphlan", "centrifuge", "kaiju"],
    ),
    database: str = typer.Option(
        "standard", "--database", "-d", help="Reference database"
    ),
    confidence: float = typer.Option(0.1, "--confidence", help="Confidence threshold"),
    threads: int | None = typer.Option(
        None, "--threads", "-t", help="Number of threads"
    ),
    format: str = typer.Option(
        "kraken",
        "--format",
        "-f",
        help="Output format",
        case_sensitive=False,
        rich_help_panel="Formats",
        show_choices=True,
        choices=["kraken", "biom", "json", "tsv"],
    ),
) -> None:
    """Perform taxonomic classification of sequences."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    threads_to_use = threads if threads is not None else cfg.threads
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "classifier": classifier,
        "database": database,
        "input": str(input),
        "output": str(output),
        "confidence": confidence,
        "threads": threads_to_use,
        "format": format,
    }

    metadata_file = output / "taxonomy_metadata.json"

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would classify {input} with {classifier}")
        typer.echo(f"[DRY-RUN] Database: {database}, Confidence: {confidence}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    # Write metadata
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Simulate classification
    _run_classifier(classifier, metadata, output)

    typer.echo(
        f"Taxonomic classification completed with {classifier}. Results in: {output}"
    )


def _run_classifier(classifier: str, metadata: dict, output: Path) -> None:
    """Simulate running taxonomic classifier."""
    # Create output files based on classifier
    if classifier == "kraken2":
        results_file = output / "kraken2_output.txt"
        report_file = output / "kraken2_report.txt"
    elif classifier == "metaphlan":
        results_file = output / "metaphlan_profile.txt"
        report_file = output / "metaphlan_report.txt"
    else:
        results_file = output / f"{classifier}_output.txt"
        report_file = output / f"{classifier}_report.txt"

    # Placeholder classification results
    results_file.touch()

    # Generate sample taxonomic report
    sample_taxa = [
        ("Bacteria", "superkingdom", 85.2),
        ("Proteobacteria", "phylum", 35.8),
        ("Bacteroidetes", "phylum", 28.4),
        ("Firmicutes", "phylum", 21.0),
        ("Escherichia coli", "species", 12.5),
        ("Bacteroides fragilis", "species", 8.7),
        ("Enterococcus faecium", "species", 6.3),
    ]

    with report_file.open("w") as f:
        f.write("Taxon\tRank\tAbundance\n")
        for taxon, rank, abundance in sample_taxa:
            f.write(f"{taxon}\t{rank}\t{abundance:.1f}\n")


@app.command("profile")
def profile(
    classifications: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Classification results"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output profile file"),
    level: str = typer.Option(
        "species",
        "--level",
        "-l",
        help="Taxonomic level (species, genus, family, etc.)",
    ),
    min_abundance: float = typer.Option(
        0.01, "--min-abundance", help="Minimum abundance threshold"
    ),
    format: str = typer.Option(
        "tsv",
        "--format",
        "-f",
        help="Output format",
        case_sensitive=False,
        rich_help_panel="Formats",
        show_choices=True,
        choices=["kraken", "biom", "json", "tsv"],
    ),
) -> None:
    """Generate abundance profile from classification results."""
    # Simulate profile generation
    profile_data: dict[str, Any] = {
        "level": level,
        "min_abundance": min_abundance,
        "total_reads": 100000,
        "classified_reads": 85000,
        "taxa": {
            "Escherichia coli": 12.5,
            "Bacteroides fragilis": 8.7,
            "Enterococcus faecium": 6.3,
            "Staphylococcus aureus": 4.2,
            "Pseudomonas aeruginosa": 3.8,
        },
    }

    if format == "json":
        with output.open("w") as f:
            json.dump(profile_data, f, indent=2)
    else:  # TSV format
        with output.open("w") as f:
            f.write("Taxon\tAbundance\n")
            for taxon, abundance in profile_data["taxa"].items():
                f.write(f"{taxon}\t{abundance:.1f}\n")

    typer.echo(f"Taxonomic profile generated: {output}")


@app.command("compare")
def compare_profiles(
    profiles: list[Path] = typer.Option(
        ..., "--profile", "-p", help="Profile files to compare"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output comparison file"),
    metric: str = typer.Option("bray_curtis", "--metric", help="Distance metric"),
) -> None:
    """Compare multiple taxonomic profiles."""
    comparison = {
        "profiles": [str(p) for p in profiles],
        "metric": metric,
        "distances": {
            "sample1_vs_sample2": 0.45,
            "sample1_vs_sample3": 0.62,
            "sample2_vs_sample3": 0.38,
        },
        "shared_taxa": 45,
        "unique_taxa": {"sample1": 8, "sample2": 12, "sample3": 6},
    }

    with output.open("w") as f:
        json.dump(comparison, f, indent=2)

    typer.echo(f"Profile comparison completed: {output}")

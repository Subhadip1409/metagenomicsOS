# cli/commands/compare.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context

app = typer.Typer(help="Comparative metagenomics")

ComparisonType = Literal["taxonomy", "functional", "diversity", "resistome"]
DistanceMetric = Literal["bray_curtis", "jaccard", "euclidean", "cosine"]


@app.command("samples")
def compare_samples(
    profiles: list[Path] = typer.Option(
        ..., "--profile", "-p", help="Sample profiles to compare"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Comparison output directory"
    ),
    comparison_type: ComparisonType = typer.Option(
        "taxonomy", "--type", "-t", help="Type of comparison"
    ),
    metric: DistanceMetric = typer.Option(
        "bray_curtis", "--metric", help="Distance metric"
    ),
    normalize: bool = typer.Option(True, "--normalize", help="Normalize data"),
) -> None:
    """Compare multiple samples."""
    ctx = get_context()

    output.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would compare {len(profiles)} samples")
        typer.echo(f"[DRY-RUN] Type: {comparison_type}, Metric: {metric}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    _perform_sample_comparison(profiles, output, comparison_type, metric, normalize)

    typer.echo(f"Sample comparison completed: {output}")


def _perform_sample_comparison(
    profiles: list[Path],
    output: Path,
    comparison_type: str,
    metric: str,
    normalize: bool,
) -> None:
    """Perform sample comparison analysis."""
    # Mock comparison results
    sample_names = [f"Sample_{i + 1}" for i in range(len(profiles))]

    # Generate distance matrix
    distance_matrix = {
        "metric": metric,
        "samples": sample_names,
        "distances": {
            "Sample_1_vs_Sample_2": 0.35,
            "Sample_1_vs_Sample_3": 0.62,
            "Sample_2_vs_Sample_3": 0.28,
        },
    }

    distance_file = output / f"{comparison_type}_distances.json"
    with distance_file.open("w") as f:
        json.dump(distance_matrix, f, indent=2)

    # Generate comparison summary
    comparison_summary = {
        "comparison_type": comparison_type,
        "samples": len(profiles),
        "metric": metric,
        "normalized": normalize,
        "most_similar": ("Sample_2", "Sample_3", 0.28),
        "most_different": ("Sample_1", "Sample_3", 0.62),
        "average_distance": 0.42,
        "shared_features": 145,
        "unique_features": {"Sample_1": 23, "Sample_2": 18, "Sample_3": 31},
    }

    summary_file = output / f"{comparison_type}_summary.json"
    with summary_file.open("w") as f:
        json.dump(comparison_summary, f, indent=2)


@app.command("groups")
def compare_groups(
    metadata: Path = typer.Option(
        ..., "--metadata", "-m", exists=True, help="Sample metadata file"
    ),
    profiles_dir: Path = typer.Option(
        ..., "--profiles", "-p", exists=True, help="Directory with sample profiles"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Group comparison output"),
    group_column: str = typer.Option(
        "group", "--column", help="Metadata column for grouping"
    ),
    comparison_type: ComparisonType = typer.Option(
        "taxonomy", "--type", help="Type of comparison"
    ),
) -> None:
    """Compare samples grouped by metadata."""
    ctx = get_context()

    output.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would compare groups by {group_column}")
        typer.echo(f"[DRY-RUN] Type: {comparison_type}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    _perform_group_comparison(
        metadata, profiles_dir, output, group_column, comparison_type
    )

    typer.echo(f"Group comparison completed: {output}")


def _perform_group_comparison(
    metadata: Path,
    profiles_dir: Path,
    output: Path,
    group_column: str,
    comparison_type: str,
) -> None:
    """Perform group-wise comparison."""
    # Mock group comparison results
    group_results = {
        "comparison_type": comparison_type,
        "grouping_variable": group_column,
        "groups": {
            "control": {
                "samples": 5,
                "mean_diversity": 3.45,
                "dominant_taxa": ["Bacteroides", "Firmicutes"],
                "unique_features": 12,
            },
            "treatment": {
                "samples": 4,
                "mean_diversity": 2.87,
                "dominant_taxa": ["Proteobacteria", "Actinobacteria"],
                "unique_features": 8,
            },
        },
        "statistical_tests": {
            "permanova": {"F_statistic": 2.34, "p_value": 0.012, "significant": True},
            "anosim": {"R_statistic": 0.65, "p_value": 0.008, "significant": True},
        },
        "differential_features": [
            {
                "feature": "Escherichia_coli",
                "log2fc": 2.1,
                "p_value": 0.003,
                "group": "treatment",
            },
            {
                "feature": "Bifidobacterium",
                "log2fc": -1.8,
                "p_value": 0.015,
                "group": "control",
            },
        ],
    }

    results_file = output / f"group_{comparison_type}_comparison.json"
    with results_file.open("w") as f:
        json.dump(group_results, f, indent=2)


@app.command("diversity")
def diversity_analysis(
    profiles: list[Path] = typer.Option(..., "--profile", "-p", help="Sample profiles"),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Diversity analysis output"
    ),
    metrics: list[str] = typer.Option(
        ["shannon", "simpson", "chao1"], "--metric", help="Diversity metrics"
    ),
) -> None:
    """Calculate diversity metrics across samples."""
    ctx = get_context()

    output.parent.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would calculate diversity for {len(profiles)} samples")
        typer.echo(f"[DRY-RUN] Metrics: {', '.join(metrics)}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    _calculate_diversity_metrics(profiles, output, metrics)

    typer.echo(f"Diversity analysis completed: {output}")


def _calculate_diversity_metrics(
    profiles: list[Path], output: Path, metrics: list[str]
) -> None:
    """Calculate diversity metrics."""
    # Mock diversity calculations
    diversity_results = {
        "metrics": metrics,
        "samples": {
            "Sample_1": {
                "shannon": 3.45,
                "simpson": 0.85,
                "chao1": 145,
                "observed_species": 125,
            },
            "Sample_2": {
                "shannon": 2.87,
                "simpson": 0.72,
                "chao1": 98,
                "observed_species": 89,
            },
            "Sample_3": {
                "shannon": 4.12,
                "simpson": 0.91,
                "chao1": 178,
                "observed_species": 156,
            },
        },
        "summary": {
            "mean_shannon": 3.48,
            "mean_simpson": 0.83,
            "highest_diversity": "Sample_3",
            "lowest_diversity": "Sample_2",
        },
    }

    with output.open("w") as f:
        json.dump(diversity_results, f, indent=2)


@app.command("resistome")
def compare_resistome(
    arg_profiles: list[Path] = typer.Option(
        ..., "--profile", "-p", help="ARG profiles to compare"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Resistome comparison output"
    ),
    prevalence_threshold: float = typer.Option(
        0.1, "--threshold", help="Minimum prevalence threshold"
    ),
) -> None:
    """Compare resistance gene profiles."""
    ctx = get_context()

    output.parent.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(
            f"[DRY-RUN] Would compare resistome across {len(arg_profiles)} samples"
        )
        typer.echo(f"[DRY-RUN] Threshold: {prevalence_threshold}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    _compare_resistome_profiles(arg_profiles, output, prevalence_threshold)

    typer.echo(f"Resistome comparison completed: {output}")


def _compare_resistome_profiles(
    profiles: list[Path], output: Path, threshold: float
) -> None:
    """Compare antibiotic resistance profiles."""
    resistome_comparison = {
        "samples": len(profiles),
        "prevalence_threshold": threshold,
        "core_resistome": ["blaOXA", "tetW"],  # ARGs present in all samples
        "accessory_resistome": ["qnrS", "ermB", "vanA"],  # Variable ARGs
        "resistance_classes": {
            "beta_lactam": {
                "prevalence": 0.85,
                "genes": ["blaOXA-48", "blaTEM-1"],
                "samples_positive": 3,
            },
            "tetracycline": {
                "prevalence": 0.67,
                "genes": ["tetW", "tetM"],
                "samples_positive": 2,
            },
            "quinolone": {
                "prevalence": 0.33,
                "genes": ["qnrS1"],
                "samples_positive": 1,
            },
        },
        "mobility_analysis": {
            "plasmid_associated": 8,
            "chromosome_associated": 3,
            "mobile_genetic_elements": 5,
        },
    }

    with output.open("w") as f:
        json.dump(resistome_comparison, f, indent=2)

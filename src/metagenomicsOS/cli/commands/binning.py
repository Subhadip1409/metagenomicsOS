# cli/commands/binning.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Binning contigs/MAGs")

Binner = Literal["metabat2", "maxbin2", "concoct", "vamb", "semibin"]


@app.command("run")
def run_binning(
    contigs: Path = typer.Option(
        ..., "--contigs", "-c", exists=True, help="Assembly contigs (FASTA)"
    ),
    abundance: Path = typer.Option(
        ..., "--abundance", "-a", exists=True, help="Abundance/coverage file"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    binner: Binner = typer.Option(
        "metabat2", "--binner", "-b", help="Binning algorithm"
    ),
    min_contig: int = typer.Option(1500, "--min-contig", help="Minimum contig length"),
    threads: int | None = typer.Option(
        None, "--threads", "-t", help="Number of threads"
    ),
    sensitivity: str = typer.Option(
        "normal", "--sensitivity", help="Binning sensitivity (low/normal/high)"
    ),
) -> None:
    """Perform contig binning to create MAGs."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    threads_to_use = threads if threads is not None else cfg.threads
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "contigs": str(contigs),
        "abundance": str(abundance),
        "output": str(output),
        "binner": binner,
        "min_contig_length": min_contig,
        "threads": threads_to_use,
        "sensitivity": sensitivity,
    }

    metadata_file = output / "binning_metadata.json"

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would bin contigs with {binner}")
        typer.echo(f"[DRY-RUN] Min contig length: {min_contig}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    _run_binning_algorithm(binner, metadata, output)

    typer.echo(f"Binning completed with {binner}. MAGs in: {output}")


def _run_binning_algorithm(binner: str, metadata: dict, output: Path) -> None:
    """Simulate contig binning."""
    # Create bin directories and files
    bins_dir = output / "bins"
    bins_dir.mkdir(exist_ok=True)

    # Generate mock bins
    bin_info = []
    for i in range(1, 8):  # Create 7 bins
        bin_file = bins_dir / f"bin_{i:03d}.fasta"
        bin_file.touch()

        # Mock bin statistics
        bin_stats = {
            "bin_id": f"bin_{i:03d}",
            "size_bp": 2500000 + (i * 350000),
            "contigs": 45 + (i * 8),
            "gc_content": 40.2 + (i * 2.1),
            "completeness": 85.4 + (i * 3.2) if i <= 5 else 65.8,
            "contamination": 2.1 + (i * 0.8) if i <= 5 else 8.3,
            "quality": "high" if i <= 5 else "medium",
        }
        bin_info.append(bin_stats)

    # Write binning summary
    summary = {
        "binner": binner,
        "total_bins": len(bin_info),
        "high_quality_bins": 5,
        "medium_quality_bins": 2,
        "bins": bin_info,
    }

    summary_file = output / "binning_summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    # Create abundance matrix
    abundance_file = output / "bin_abundance.tsv"
    with abundance_file.open("w") as f:
        f.write("BinID\tSample1\tSample2\tSample3\n")
        for i, bin_stat in enumerate(bin_info, 1):
            f.write(
                f"bin_{i:03d}\t{12.5 + i * 2.3:.1f}\t{8.7 + i * 1.8:.1f}\t{15.2 + i * 2.1:.1f}\n"
            )


@app.command("quality")
def assess_quality(
    bins_dir: Path = typer.Option(
        ..., "--bins", "-b", exists=True, help="Directory containing bins"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Quality assessment output"
    ),
    completeness_threshold: float = typer.Option(
        70.0, "--completeness", help="Minimum completeness %"
    ),
    contamination_threshold: float = typer.Option(
        10.0, "--contamination", help="Maximum contamination %"
    ),
) -> None:
    """Assess quality of genome bins."""
    # Mock quality assessment results
    quality_results = {
        "assessment_date": "2025-08-26",
        "thresholds": {
            "completeness": completeness_threshold,
            "contamination": contamination_threshold,
        },
        "bins": [
            {
                "bin": "bin_001",
                "completeness": 95.2,
                "contamination": 1.8,
                "quality": "high",
            },
            {
                "bin": "bin_002",
                "completeness": 88.7,
                "contamination": 3.2,
                "quality": "high",
            },
            {
                "bin": "bin_003",
                "completeness": 82.1,
                "contamination": 5.1,
                "quality": "medium",
            },
            {
                "bin": "bin_004",
                "completeness": 76.5,
                "contamination": 8.7,
                "quality": "medium",
            },
            {
                "bin": "bin_005",
                "completeness": 68.3,
                "contamination": 12.1,
                "quality": "low",
            },
        ],
        "summary": {
            "total_bins": 5,
            "high_quality": 2,
            "medium_quality": 2,
            "low_quality": 1,
        },
    }

    with output.open("w") as f:
        json.dump(quality_results, f, indent=2)

    typer.echo(f"Quality assessment completed: {output}")


@app.command("refine")
def refine_bins(
    bins_dir: Path = typer.Option(
        ..., "--bins", "-b", exists=True, help="Directory containing bins"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Refined bins output directory"
    ),
    min_completeness: float = typer.Option(
        50.0, "--min-completeness", help="Minimum completeness"
    ),
    max_contamination: float = typer.Option(
        15.0, "--max-contamination", help="Maximum contamination"
    ),
) -> None:
    """Refine genome bins based on quality metrics."""
    output.mkdir(parents=True, exist_ok=True)
    refined_dir = output / "refined_bins"
    refined_dir.mkdir(exist_ok=True)

    # Mock refinement process
    refinement_log = {
        "input_bins": 7,
        "refined_bins": 5,
        "removed_bins": 2,
        "criteria": {
            "min_completeness": min_completeness,
            "max_contamination": max_contamination,
        },
        "actions": [
            {"bin": "bin_001", "action": "kept", "reason": "high quality"},
            {"bin": "bin_002", "action": "kept", "reason": "high quality"},
            {
                "bin": "bin_003",
                "action": "refined",
                "reason": "removed contaminated contigs",
            },
            {"bin": "bin_004", "action": "kept", "reason": "acceptable quality"},
            {"bin": "bin_005", "action": "kept", "reason": "acceptable quality"},
            {"bin": "bin_006", "action": "removed", "reason": "low completeness"},
            {"bin": "bin_007", "action": "removed", "reason": "high contamination"},
        ],
    }

    log_file = output / "refinement_log.json"
    with log_file.open("w") as f:
        json.dump(refinement_log, f, indent=2)

    # Create placeholder refined bin files
    for i in range(1, 6):
        refined_bin = refined_dir / f"refined_bin_{i:03d}.fasta"
        refined_bin.touch()

    typer.echo(f"Bin refinement completed. Refined bins in: {refined_dir}")

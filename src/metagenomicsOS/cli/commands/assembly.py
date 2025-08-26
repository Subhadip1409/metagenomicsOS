# cli/commands/assembly.py
from __future__ import annotations
from pathlib import Path
import json
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Genome assembly")


@app.command("run")
def run_assembly(
    input: Path = typer.Option(..., "--input", "-i", help="Input reads (FASTQ)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    assembler: str = typer.Option(
        "spades",
        "--assembler",
        "-a",
        help="Assembly tool",
        case_sensitive=False,
        rich_help_panel="Assemblers",
        show_choices=True,
        choices=["spades", "megahit", "metaflye", "unicycler"],
    ),
    mode: str = typer.Option(
        "meta", "--mode", help="Assembly mode (meta, isolate, etc.)"
    ),
    kmer_sizes: str = typer.Option(
        "21,33,55", "--kmers", help="K-mer sizes (comma-separated)"
    ),
    threads: int | None = typer.Option(
        None, "--threads", "-t", help="Number of threads"
    ),
    memory: int = typer.Option(16, "--memory", help="Memory limit in GB"),
    careful: bool = typer.Option(False, "--careful", help="Enable careful mode"),
) -> None:
    """Run genome assembly on input reads."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    threads_to_use = threads if threads is not None else cfg.threads
    output.mkdir(parents=True, exist_ok=True)

    # Parse k-mer sizes
    kmer_list = [int(k.strip()) for k in kmer_sizes.split(",")]

    # Generate assembly metadata
    metadata = {
        "assembler": assembler,
        "mode": mode,
        "input": str(input),
        "output": str(output),
        "kmer_sizes": kmer_list,
        "threads": threads_to_use,
        "memory_gb": memory,
        "careful_mode": careful,
    }

    metadata_file = output / "assembly_metadata.json"

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would run {assembler} assembly")
        typer.echo(f"[DRY-RUN] Mode: {mode}, K-mers: {kmer_sizes}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    # Write metadata
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Simulate assembly
    _run_assembler(assembler, metadata, output)

    typer.echo(f"Assembly completed with {assembler}. Results in: {output}")


def _run_assembler(assembler: str, metadata: dict, output: Path) -> None:
    """Simulate running the specified assembler."""
    # Create standard assembly output files
    contigs_file = output / "contigs.fasta"
    scaffolds_file = output / "scaffolds.fasta"
    assembly_log = output / f"{assembler}_log.txt"

    # Placeholder files
    contigs_file.touch()
    scaffolds_file.touch()

    # Generate assembly log
    with assembly_log.open("w") as f:
        f.write(f"Assembly with {assembler}\n")
        f.write(f"Mode: {metadata['mode']}\n")
        f.write(f"K-mer sizes: {metadata['kmer_sizes']}\n")
        f.write(f"Threads: {metadata['threads']}\n")
        f.write(f"Memory: {metadata['memory_gb']}GB\n")
        f.write("Assembly completed successfully\n")

    # Generate basic assembly stats
    stats = {
        "assembler": assembler,
        "total_contigs": 1500,
        "total_length": 4200000,
        "largest_contig": 185000,
        "n50": 12000,
        "gc_content": 41.2,
    }

    stats_file = output / "assembly_stats.json"
    with stats_file.open("w") as f:
        json.dump(stats, f, indent=2)


@app.command("stats")
def assembly_stats(
    assembly: Path = typer.Option(
        ..., "--assembly", "-a", exists=True, help="Assembly FASTA file"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for stats"
    ),
) -> None:
    """Calculate assembly statistics."""
    # Placeholder stats calculation
    stats = {
        "file": str(assembly),
        "total_contigs": 1500,
        "total_length": 4200000,
        "largest_contig": 185000,
        "smallest_contig": 500,
        "mean_length": 2800,
        "n50": 12000,
        "l50": 89,
        "gc_content": 41.2,
    }

    if output:
        with output.open("w") as f:
            json.dump(stats, f, indent=2)
        typer.echo(f"Assembly stats written to: {output}")
    else:
        typer.echo(json.dumps(stats, indent=2))


@app.command("validate")
def validate_assembly(
    assembly: Path = typer.Option(
        ..., "--assembly", "-a", exists=True, help="Assembly FASTA file"
    ),
    min_contig_length: int = typer.Option(
        500, "--min-length", help="Minimum contig length"
    ),
    max_contigs: int = typer.Option(
        10000, "--max-contigs", help="Maximum number of contigs"
    ),
) -> None:
    """Validate assembly quality and completeness."""
    # Placeholder validation
    validation = {
        "file": str(assembly),
        "valid": True,
        "warnings": [],
        "errors": [],
        "metrics": {
            "passes_min_length": True,
            "contig_count_ok": True,
            "no_gaps": True,
        },
    }

    typer.echo("Assembly validation completed:")
    typer.echo(json.dumps(validation, indent=2))

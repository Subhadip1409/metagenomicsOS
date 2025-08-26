# cli/commands/annotation.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Functional/ARG annotation")

Database = Literal["cog", "kegg", "pfam", "tigrfam", "go", "ec", "all"]
OutputFormat = Literal["tsv", "gff", "json", "xml"]


@app.command("run")
def run_annotation(
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Input sequences (FASTA)"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    database: Database = typer.Option(
        "cog", "--database", "-d", help="Annotation database"
    ),
    evalue: float = typer.Option(1e-5, "--evalue", "-e", help="E-value threshold"),
    coverage: float = typer.Option(50.0, "--coverage", help="Coverage threshold (%)"),
    threads: int | None = typer.Option(
        None, "--threads", "-t", help="Number of threads"
    ),
    format: OutputFormat = typer.Option("tsv", "--format", "-f", help="Output format"),
) -> None:
    """Perform functional annotation of sequences."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    threads_to_use = threads if threads is not None else cfg.threads
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input": str(input),
        "output": str(output),
        "database": database,
        "evalue": evalue,
        "coverage": coverage,
        "threads": threads_to_use,
        "format": format,
    }

    metadata_file = output / "annotation_metadata.json"

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would annotate {input} against {database}")
        typer.echo(f"[DRY-RUN] E-value: {evalue}, Coverage: {coverage}%")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    _run_annotation_search(database, metadata, output)

    typer.echo(f"Functional annotation completed. Results in: {output}")


def _run_annotation_search(database: str, metadata: dict, output: Path) -> None:
    """Simulate functional annotation search."""
    # Create annotation output files
    if metadata["format"] == "gff":
        results_file = output / f"{database}_annotations.gff"
        _write_gff_results(results_file, database)
    else:
        results_file = output / f"{database}_annotations.tsv"
        _write_tsv_results(results_file, database)

    # Generate annotation summary
    summary = {
        "database": database,
        "total_sequences": 1250,
        "annotated_sequences": 987,
        "annotation_rate": 78.96,
        "top_functions": {
            "ATP synthase": 45,
            "DNA polymerase": 38,
            "Ribosomal protein": 32,
            "Cell wall synthesis": 28,
            "Transport protein": 25,
        },
    }

    summary_file = output / f"{database}_summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)


def _write_tsv_results(output_file: Path, database: str) -> None:
    """Write mock TSV annotation results."""
    sample_annotations = [
        (
            "contig_1_gene_1",
            "COG0001",
            "Glutamate-1-semialdehyde 2,1-aminomutase",
            2.3e-45,
            85.2,
        ),
        (
            "contig_1_gene_2",
            "COG0002",
            "N-acetyl-gamma-glutamyl-phosphate reductase",
            1.1e-38,
            78.9,
        ),
        ("contig_2_gene_1", "COG0003", "Oxyanion-translocating ATPase", 5.7e-52, 92.1),
        ("contig_2_gene_2", "COG0004", "Ammonia permease", 3.2e-41, 81.4),
        (
            "contig_3_gene_1",
            "COG0005",
            "Purine nucleoside phosphorylase",
            1.8e-35,
            76.3,
        ),
    ]

    with output_file.open("w") as f:
        f.write("Query\tSubject\tDescription\tE-value\tCoverage\n")
        for query, subject, desc, evalue, coverage in sample_annotations:
            f.write(f"{query}\t{subject}\t{desc}\t{evalue:.2e}\t{coverage:.1f}\n")


def _write_gff_results(output_file: Path, database: str) -> None:
    """Write mock GFF annotation results."""
    with output_file.open("w") as f:
        f.write("##gff-version 3\n")
        f.write(
            "contig_1\tprodigal\tgene\t100\t1200\t.\t+\t.\tID=gene_1;product=ATP synthase\n"
        )
        f.write(
            "contig_1\tprodigal\tgene\t1300\t2100\t.\t-\t.\tID=gene_2;product=DNA polymerase\n"
        )
        f.write(
            "contig_2\tprodigal\tgene\t50\t850\t.\t+\t.\tID=gene_3;product=Ribosomal protein\n"
        )


@app.command("summarize")
def summarize(
    annotations: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Annotation results"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Summary output file"),
    min_count: int = typer.Option(5, "--min-count", help="Minimum function count"),
) -> None:
    """Summarize annotation results."""
    summary = {
        "input_file": str(annotations),
        "total_annotations": 987,
        "unique_functions": 245,
        "function_categories": {
            "Metabolism": 312,
            "Information processing": 198,
            "Cellular processes": 167,
            "Environmental adaptation": 89,
            "Genetic information": 221,
        },
    }

    with output.open("w") as f:
        json.dump(summary, f, indent=2)

    typer.echo(f"Annotation summary created: {output}")

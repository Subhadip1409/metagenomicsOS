# cli/commands/genes.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Gene-level annotation")

Predictor = Literal["prodigal", "augustus", "genemark", "glimmer"]
OutputFormat = Literal["fasta", "gff", "genbank", "json"]


@app.command("predict")
def predict_genes(
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Input sequences (FASTA)"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    predictor: Predictor = typer.Option(
        "prodigal", "--predictor", "-p", help="Gene prediction tool"
    ),
    mode: str = typer.Option(
        "meta", "--mode", help="Prediction mode (meta, single, etc.)"
    ),
    min_gene_len: int = typer.Option(90, "--min-gene-len", help="Minimum gene length"),
    format: OutputFormat = typer.Option(
        "fasta", "--format", "-f", help="Output format"
    ),
    translation_table: int = typer.Option(11, "--table", help="Translation table"),
) -> None:
    """Predict genes in genomic sequences."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    output.mkdir(parents=True, exist_ok=True)

    threads_to_use = cfg.threads  # Use cfg.threads here

    metadata = {
        "input": str(input),
        "output": str(output),
        "predictor": predictor,
        "mode": mode,
        "min_gene_length": min_gene_len,
        "format": format,
        "translation_table": translation_table,
        "threads": threads_to_use,  # Add threads to metadata
    }

    metadata_file = output / "gene_prediction_metadata.json"

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would predict genes with {predictor}")
        typer.echo(f"[DRY-RUN] Mode: {mode}, Min length: {min_gene_len}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    _run_gene_prediction(predictor, metadata, output)

    typer.echo(f"Gene prediction completed with {predictor}. Results in: {output}")


def _run_gene_prediction(predictor: str, metadata: dict, output: Path) -> None:
    """Simulate gene prediction."""
    # Create gene output files
    genes_fasta = output / "predicted_genes.fasta"
    proteins_fasta = output / "predicted_proteins.fasta"
    gff_file = output / "genes.gff"

    # Write mock gene sequences
    with genes_fasta.open("w") as f:
        for i in range(1, 26):  # 25 mock genes
            f.write(f">gene_{i:03d}\n")
            f.write("ATGAAACGTCTGCACGAATTCGGCAAGGCTTTCGACCTGAAGGGTCTGCAG\n")

    # Write mock protein sequences
    with proteins_fasta.open("w") as f:
        for i in range(1, 26):
            f.write(f">protein_{i:03d}\n")
            f.write("MKRLHEFGKAFLDLKGLQ\n")

    # Write GFF file
    with gff_file.open("w") as f:
        f.write("##gff-version 3\n")
        for i in range(1, 26):
            start = i * 1000
            end = start + 750
            f.write(
                f"contig_1\t{predictor}\tgene\t{start}\t{end}\t.\t+\t0\tID=gene_{i:03d}\n"
            )

    # Generate prediction statistics
    stats = {
        "predictor": predictor,
        "mode": metadata["mode"],
        "total_genes": 25,
        "average_gene_length": 750,
        "longest_gene": 1245,
        "shortest_gene": 270,
        "coding_density": 85.2,
        "gc_content_genes": 42.8,
    }

    stats_file = output / "prediction_stats.json"
    with stats_file.open("w") as f:
        json.dump(stats, f, indent=2)


@app.command("annotate")
def annotate_genes(
    genes: Path = typer.Option(
        ..., "--genes", "-g", exists=True, help="Predicted genes (FASTA)"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    database: str = typer.Option(
        "uniref90", "--database", "-d", help="Annotation database"
    ),
    evalue: float = typer.Option(1e-5, "--evalue", help="E-value threshold"),
    threads: int | None = typer.Option(
        None, "--threads", "-t", help="Number of threads"
    ),
) -> None:
    """Annotate predicted genes."""
    ctx = get_context()

    output.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would annotate genes against {database}")
        typer.echo(f"[DRY-RUN] E-value: {evalue}")
        return

    # Create mock annotation results
    annotations_file = output / "gene_annotations.tsv"

    sample_annotations = [
        ("gene_001", "RecA protein", "DNA repair", 2.3e-45),
        ("gene_002", "ATP synthase subunit alpha", "Energy production", 1.1e-38),
        ("gene_003", "Ribosomal protein L1", "Translation", 5.7e-52),
        ("gene_004", "DNA polymerase I", "DNA replication", 3.2e-41),
        ("gene_005", "Elongation factor Tu", "Translation", 1.8e-35),
    ]

    with annotations_file.open("w") as f:
        f.write("GeneID\tFunction\tCategory\tE-value\n")
        for gene_id, function, category, evalue in sample_annotations:
            f.write(f"{gene_id}\t{function}\t{category}\t{evalue:.2e}\n")

    typer.echo(f"Gene annotation completed. Results in: {output}")


@app.command("summary")
def gene_summary(
    annotations: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Gene annotations"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Summary output file"),
) -> None:
    """Generate gene annotation summary."""
    summary = {
        "total_genes": 25,
        "annotated_genes": 20,
        "annotation_rate": 80.0,
        "functional_categories": {
            "Energy production": 6,
            "Translation": 5,
            "DNA replication": 4,
            "Cell wall synthesis": 3,
            "Transcription": 2,
        },
        "top_functions": [
            "ATP synthase",
            "Ribosomal protein",
            "DNA polymerase",
            "Elongation factor",
            "RNA polymerase",
        ],
    }

    with output.open("w") as f:
        json.dump(summary, f, indent=2)

    typer.echo(f"Gene summary created: {output}")

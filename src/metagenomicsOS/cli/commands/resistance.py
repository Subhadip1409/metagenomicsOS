# cli/commands/resistance.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="ARG/AMR annotation")

ARGDatabase = Literal["card", "argannot", "resfinder", "ardb", "ncbi_amr"]
ResistanceClass = Literal[
    "beta_lactam", "aminoglycoside", "tetracycline", "quinolone", "macrolide", "all"
]


@app.command("detect")
def detect_args(
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Input sequences"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    database: ARGDatabase = typer.Option(
        "card", "--database", "-d", help="ARG database"
    ),
    identity: float = typer.Option(90.0, "--identity", help="Identity threshold (%)"),
    coverage: float = typer.Option(80.0, "--coverage", help="Coverage threshold (%)"),
    evalue: float = typer.Option(1e-10, "--evalue", help="E-value threshold"),
    threads: int | None = typer.Option(
        None, "--threads", "-t", help="Number of threads"
    ),
) -> None:
    """Detect antibiotic resistance genes."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    threads_to_use = threads if threads is not None else cfg.threads
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input": str(input),
        "output": str(output),
        "database": database,
        "identity_threshold": identity,
        "coverage_threshold": coverage,
        "evalue": evalue,
        "threads": threads_to_use,
    }

    metadata_file = output / "arg_detection_metadata.json"

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would detect ARGs using {database}")
        typer.echo(f"[DRY-RUN] Identity: {identity}%, Coverage: {coverage}%")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    _run_arg_detection(database, metadata, output)

    typer.echo(f"ARG detection completed with {database}. Results in: {output}")


def _run_arg_detection(database: str, metadata: dict, output: Path) -> None:
    """Simulate ARG detection."""
    # Create ARG results file
    args_file = output / f"{database}_args.tsv"

    # Mock ARG detection results
    sample_args = [
        (
            "contig_1_gene_5",
            "blaOXA-48",
            "beta_lactam",
            "Oxacillinase",
            95.2,
            88.7,
            2.3e-45,
        ),
        (
            "contig_2_gene_12",
            "aac(3)-IV",
            "aminoglycoside",
            "Acetyltransferase",
            92.8,
            85.4,
            1.1e-38,
        ),
        (
            "contig_3_gene_8",
            "tetW",
            "tetracycline",
            "Ribosomal protection",
            88.9,
            82.1,
            5.7e-35,
        ),
        (
            "contig_1_gene_18",
            "qnrS1",
            "quinolone",
            "DNA gyrase protection",
            91.5,
            87.3,
            3.2e-41,
        ),
        (
            "contig_4_gene_3",
            "ermB",
            "macrolide",
            "23S rRNA methylase",
            89.7,
            79.6,
            1.8e-33,
        ),
    ]

    with args_file.open("w") as f:
        f.write("Query\tGene\tClass\tMechanism\tIdentity\tCoverage\tE-value\n")
        for (
            query,
            gene,
            res_class,
            mechanism,
            identity,
            coverage,
            evalue,
        ) in sample_args:
            f.write(
                f"{query}\t{gene}\t{res_class}\t{mechanism}\t{identity:.1f}\t{coverage:.1f}\t{evalue:.2e}\n"
            )

    # Generate resistance summary
    summary = {
        "database": database,
        "total_args": len(sample_args),
        "resistance_classes": {
            "beta_lactam": 1,
            "aminoglycoside": 1,
            "tetracycline": 1,
            "quinolone": 1,
            "macrolide": 1,
        },
        "resistance_mechanisms": {
            "Antibiotic inactivation": 2,
            "Target protection": 2,
            "Target modification": 1,
        },
    }

    summary_file = output / f"{database}_summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)


@app.command("profile")
def resistance_profile(
    args: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="ARG detection results"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Resistance profile output"
    ),
    class_filter: ResistanceClass = typer.Option(
        "all", "--class", help="Resistance class filter"
    ),
) -> None:
    """Generate resistance profile from ARG results."""
    # Mock resistance profile
    profile = {
        "sample_id": "sample_001",
        "total_args": 5,
        "resistance_classes": {
            "beta_lactam": {"count": 1, "genes": ["blaOXA-48"], "risk": "high"},
            "aminoglycoside": {"count": 1, "genes": ["aac(3)-IV"], "risk": "medium"},
            "tetracycline": {"count": 1, "genes": ["tetW"], "risk": "medium"},
            "quinolone": {"count": 1, "genes": ["qnrS1"], "risk": "high"},
            "macrolide": {"count": 1, "genes": ["ermB"], "risk": "low"},
        },
        "overall_risk": "high",
    }

    with output.open("w") as f:
        json.dump(profile, f, indent=2)

    typer.echo(f"Resistance profile created: {output}")


@app.command("compare")
def compare_resistomes(
    profiles: list[Path] = typer.Option(
        ..., "--profile", "-p", help="Resistance profiles to compare"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Comparison output file"),
) -> None:
    """Compare resistance profiles across samples."""
    comparison = {
        "profiles": [str(p) for p in profiles],
        "shared_args": ["blaOXA-48", "tetW"],
        "unique_args": {
            "sample_1": ["aac(3)-IV"],
            "sample_2": ["qnrS1", "ermB"],
            "sample_3": ["vanA", "mecA"],
        },
        "class_distribution": {
            "beta_lactam": {"sample_1": 1, "sample_2": 2, "sample_3": 1},
            "aminoglycoside": {"sample_1": 1, "sample_2": 0, "sample_3": 2},
            "tetracycline": {"sample_1": 1, "sample_2": 1, "sample_3": 0},
        },
        "diversity_metrics": {
            "shannon_diversity": 1.85,
            "simpson_diversity": 0.72,
            "richness": 8,
        },
    }

    with output.open("w") as f:
        json.dump(comparison, f, indent=2)

    typer.echo(f"Resistome comparison completed: {output}")


@app.command("report")
def generate_report(
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="ARG detection results"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="HTML report output"),
    title: str = typer.Option("ARG Analysis Report", "--title", help="Report title"),
) -> None:
    """Generate HTML report for ARG analysis."""
    # Create basic HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .high-risk {{ color: #d32f2f; font-weight: bold; }}
            .medium-risk {{ color: #f57c00; }}
            .low-risk {{ color: #388e3c; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <h2>Summary</h2>
        <p>Total ARGs detected: 5</p>
        <p>Resistance classes: 5</p>
        <p>Overall risk level: <span class="high-risk">HIGH</span></p>

        <h2>Detected Resistance Genes</h2>
        <table>
            <tr><th>Gene</th><th>Class</th><th>Mechanism</th><th>Risk Level</th></tr>
            <tr><td>blaOXA-48</td><td>Beta-lactam</td><td>Oxacillinase</td><td class="high-risk">High</td></tr>
            <tr><td>aac(3)-IV</td><td>Aminoglycoside</td><td>Acetyltransferase</td><td class="medium-risk">Medium</td></tr>
            <tr><td>tetW</td><td>Tetracycline</td><td>Ribosomal protection</td><td class="medium-risk">Medium</td></tr>
            <tr><td>qnrS1</td><td>Quinolone</td><td>DNA gyrase protection</td><td class="high-risk">High</td></tr>
            <tr><td>ermB</td><td>Macrolide</td><td>23S rRNA methylase</td><td class="low-risk">Low</td></tr>
        </table>

        <h2>Analysis Date</h2>
        <p>Generated on: 2025-08-26</p>
    </body>
    </html>
    """

    with output.open("w") as f:
        f.write(html_content)

    typer.echo(f"ARG analysis report generated: {output}")

# cli/commands/database.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
from datetime import datetime
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Database management")

DatabaseType = Literal["kraken2", "card", "cog", "kegg", "pfam", "uniref"]


@app.command("download")
def download_database(
    db_type: DatabaseType = typer.Option(..., "--type", "-t", help="Database type"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Download directory"),
    version: str | None = typer.Option(
        None, "--version", help="Specific version to download"
    ),
    force: bool = typer.Option(False, "--force", help="Force redownload if exists"),
) -> None:
    """Download reference database."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    # Use configured database directory if output not specified
    if str(output_dir) == ".":
        output_dir = Path(cfg.database_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would download {db_type} database")
        typer.echo(f"[DRY-RUN] Version: {version or 'latest'}")
        typer.echo(f"[DRY-RUN] Output: {output_dir}")
        return

    _download_database_files(db_type, output_dir, version, force)

    typer.echo(f"Database {db_type} downloaded to: {output_dir}")


def _download_database_files(
    db_type: str, output_dir: Path, version: str | None, force: bool
) -> None:
    """Simulate database download."""
    db_dir = output_dir / db_type
    db_dir.mkdir(parents=True, exist_ok=True)

    # Create mock database files
    if db_type == "kraken2":
        files = ["hash.k2d", "opts.k2d", "taxo.k2d", "database.kraken"]
    elif db_type == "card":
        files = ["card.json", "aro.obo", "card.fasta", "card_database.dmnd"]
    elif db_type in ["cog", "kegg", "pfam"]:
        files = [f"{db_type}.hmm", f"{db_type}.dat", f"{db_type}_mapping.txt"]
    else:
        files = [f"{db_type}.fasta", f"{db_type}.dmnd", f"{db_type}_taxonomy.txt"]

    for filename in files:
        file_path = db_dir / filename
        if not file_path.exists() or force:
            file_path.touch()

    # Create database metadata
    metadata = {
        "database": db_type,
        "version": version or "latest",
        "download_date": datetime.now().isoformat(),
        "files": files,
        "size_mb": 1250,  # Mock size
        "status": "ready",
    }

    metadata_file = db_dir / "database_info.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)


@app.command("list")
def list_databases(
    db_dir: Path | None = typer.Option(
        None, "--dir", help="Database directory to scan"
    ),
) -> None:
    """List installed databases."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if db_dir is None:
        db_dir = Path(cfg.database_dir)

    if not db_dir.exists():
        typer.echo("No databases found. Use 'database download' to install databases.")
        return

    typer.echo("Installed Databases:")
    typer.echo("-" * 50)

    for item in sorted(db_dir.iterdir()):
        if item.is_dir():
            metadata_file = item / "database_info.json"
            if metadata_file.exists():
                try:
                    with metadata_file.open() as f:
                        metadata = json.load(f)
                    status = metadata.get("status", "unknown")
                    version = metadata.get("version", "unknown")
                    size = metadata.get("size_mb", 0)

                    typer.echo(
                        f"{item.name:15} | {status:8} | {version:10} | {size:6}MB"
                    )
                except Exception:
                    typer.echo(f"{item.name:15} | error    | unknown    | unknown")
            else:
                typer.echo(f"{item.name:15} | unknown  | unknown    | unknown")


@app.command("validate")
def validate_database(
    db_type: DatabaseType = typer.Option(..., "--type", "-t", help="Database type"),
    db_dir: Path | None = typer.Option(None, "--dir", help="Database directory"),
) -> None:
    """Validate database integrity."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if db_dir is None:
        db_dir = Path(cfg.database_dir) / db_type

    if not db_dir.exists():
        typer.echo(f"Database {db_type} not found at {db_dir}")
        raise typer.Exit(code=1)

    _validate_database_integrity(db_type, db_dir)


def _validate_database_integrity(db_type: str, db_dir: Path) -> None:
    """Validate database files and integrity."""
    # Expected files for each database type
    expected_files = {
        "kraken2": ["hash.k2d", "opts.k2d", "taxo.k2d"],
        "card": ["card.json", "card.fasta"],
        "cog": ["cog.hmm", "cog_mapping.txt"],
        "kegg": ["kegg.hmm", "kegg_mapping.txt"],
        "pfam": ["pfam.hmm", "pfam.dat"],
        "uniref": ["uniref.fasta", "uniref.dmnd"],
    }

    required_files = expected_files.get(db_type, [])
    missing_files = []
    present_files = []

    for filename in required_files:
        file_path = db_dir / filename
        if file_path.exists():
            present_files.append(filename)
        else:
            missing_files.append(filename)

    # Validation results
    if missing_files:
        typer.echo(f"❌ Database {db_type} validation FAILED")
        typer.echo(f"Missing files: {', '.join(missing_files)}")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"✅ Database {db_type} validation PASSED")
        typer.echo(f"All required files present: {', '.join(present_files)}")


@app.command("update")
def update_database(
    db_type: DatabaseType = typer.Option(
        ..., "--type", "-t", help="Database type to update"
    ),
    db_dir: Path | None = typer.Option(None, "--dir", help="Database directory"),
) -> None:
    """Update existing database to latest version."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if db_dir is None:
        db_dir = Path(cfg.database_dir)

    target_dir = db_dir / db_type

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would update {db_type} database")
        typer.echo(f"[DRY-RUN] Location: {target_dir}")
        return

    if not target_dir.exists():
        typer.echo(f"Database {db_type} not found. Use 'database download' first.")
        raise typer.Exit(code=1)

    # Simulate update process
    metadata_file = target_dir / "database_info.json"
    if metadata_file.exists():
        with metadata_file.open() as f:
            metadata = json.load(f)

        metadata["version"] = "updated_" + datetime.now().strftime("%Y%m%d")
        metadata["download_date"] = datetime.now().isoformat()
        metadata["status"] = "updated"

        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)

    typer.echo(f"Database {db_type} updated successfully")


@app.command("info")
def database_info(
    db_type: DatabaseType = typer.Option(..., "--type", "-t", help="Database type"),
    db_dir: Path | None = typer.Option(None, "--dir", help="Database directory"),
) -> None:
    """Show detailed database information."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if db_dir is None:
        db_dir = Path(cfg.database_dir) / db_type

    if not db_dir.exists():
        typer.echo(f"Database {db_type} not found")
        return

    metadata_file = db_dir / "database_info.json"
    if metadata_file.exists():
        with metadata_file.open() as f:
            metadata = json.load(f)

        typer.echo(f"Database: {metadata['database']}")
        typer.echo(f"Version: {metadata.get('version', 'unknown')}")
        typer.echo(f"Download Date: {metadata.get('download_date', 'unknown')}")
        typer.echo(f"Size: {metadata.get('size_mb', 0)} MB")
        typer.echo(f"Status: {metadata.get('status', 'unknown')}")
        typer.echo(f"Files: {len(metadata.get('files', []))}")
    else:
        typer.echo(f"No metadata found for {db_type}")

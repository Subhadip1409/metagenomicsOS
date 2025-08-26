# cli/commands/validate.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context

app = typer.Typer(help="Validation utilities")

DataType = Literal["fastq", "fasta", "gff", "tsv", "json", "yaml"]
ValidationLevel = Literal["basic", "standard", "strict"]


@app.command("input")
def validate_input(
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Input file to validate"
    ),
    data_type: DataType = typer.Option(..., "--type", "-t", help="Expected data type"),
    level: ValidationLevel = typer.Option(
        "standard", "--level", "-l", help="Validation level"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save validation report"
    ),
) -> None:
    """Validate input data files."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would validate {input_file} as {data_type}")
        typer.echo(f"[DRY-RUN] Level: {level}")
        if output:
            typer.echo(f"[DRY-RUN] Report: {output}")
        return

    validation_result = _validate_input_file(input_file, data_type, level)

    # Display results
    if validation_result["valid"]:
        typer.echo("✅ Input validation PASSED")
    else:
        typer.echo("❌ Input validation FAILED")

    typer.echo(f"File: {input_file}")
    typer.echo(f"Type: {data_type}")
    typer.echo(f"Size: {validation_result['file_size']} bytes")

    if validation_result["warnings"]:
        typer.echo("\nWarnings:")
        for warning in validation_result["warnings"]:
            typer.echo(f"  ⚠️  {warning}")

    if validation_result["errors"]:
        typer.echo("\nErrors:")
        for error in validation_result["errors"]:
            typer.echo(f"  ❌ {error}")
        raise typer.Exit(code=1)

    # Save report if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(validation_result, f, indent=2)
        typer.echo(f"\nValidation report saved: {output}")


def _validate_input_file(input_file: Path, data_type: str, level: str) -> dict:
    """Perform input file validation."""
    errors: list[str] = []
    warnings: list[str] = []

    # Basic file checks
    if not input_file.exists():
        errors.append(f"File does not exist: {input_file}")
        return {"valid": False, "errors": errors, "warnings": warnings, "file_size": 0}

    file_size = input_file.stat().st_size

    # File size checks
    if file_size == 0:
        errors.append("File is empty")
    elif file_size < 100:  # Very small files
        warnings.append(f"File is very small ({file_size} bytes)")

    # Type-specific validation
    if data_type == "fastq":
        errors.extend(_validate_fastq_format(input_file, level))
    elif data_type == "fasta":
        errors.extend(_validate_fasta_format(input_file, level))
    elif data_type == "gff":
        errors.extend(_validate_gff_format(input_file, level))
    elif data_type in ["tsv", "csv"]:
        errors.extend(_validate_table_format(input_file, level))
    elif data_type == "json":
        errors.extend(_validate_json_format(input_file, level))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "file_size": file_size,
        "data_type": data_type,
        "validation_level": level,
    }


def _validate_fastq_format(file_path: Path, level: str) -> list:
    """Validate FASTQ format."""
    errors = []

    try:
        with file_path.open("r") as f:
            # Check first few records
            line_count = 0
            for i, line in enumerate(f):
                if i >= 100 and level == "basic":  # Basic validation stops early
                    break
                if i >= 1000 and level == "standard":
                    break

                line_count += 1
                line_num = i + 1

                # FASTQ format: @header, sequence, +, quality
                if line_num % 4 == 1:  # Header line
                    if not line.startswith("@"):
                        errors.append(
                            f"Line {line_num}: FASTQ header must start with @"
                        )
                elif line_num % 4 == 2:  # Sequence line
                    if not line.strip():
                        errors.append(f"Line {line_num}: Empty sequence")
                    elif level == "strict":
                        # Check for valid nucleotides
                        valid_chars = set("ATCGN")
                        if not all(c in valid_chars for c in line.strip().upper()):
                            errors.append(
                                f"Line {line_num}: Invalid nucleotide characters"
                            )
                elif line_num % 4 == 3:  # Plus line
                    if not line.startswith("+"):
                        errors.append(
                            f"Line {line_num}: FASTQ separator must start with +"
                        )
                elif line_num % 4 == 0:  # Quality line
                    if not line.strip():
                        errors.append(f"Line {line_num}: Empty quality string")

        # Check if file ends properly
        if line_count % 4 != 0:
            errors.append("FASTQ file has incomplete final record")

    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")

    return errors


def _validate_fasta_format(file_path: Path, level: str) -> list:
    """Validate FASTA format."""
    errors = []

    try:
        with file_path.open("r") as f:
            has_header = False
            has_sequence = False

            for i, line in enumerate(f):
                if i >= 100 and level == "basic":
                    break

                line_num = i + 1

                if line.startswith(">"):
                    has_header = True
                    if not line.strip()[1:]:  # Empty header after >
                        errors.append(f"Line {line_num}: Empty FASTA header")
                else:
                    if not has_header:
                        errors.append(f"Line {line_num}: Sequence without header")
                    else:
                        has_sequence = True
                        if level == "strict" and line.strip():
                            # Check for valid sequence characters
                            valid_chars = set("ATCGN-")
                            if not all(c in valid_chars for c in line.strip().upper()):
                                errors.append(
                                    f"Line {line_num}: Invalid sequence characters"
                                )

        if has_header and not has_sequence:
            errors.append("FASTA file has headers but no sequences")
        elif not has_header:
            errors.append("No FASTA headers found")

    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")

    return errors


def _validate_gff_format(file_path: Path, level: str) -> list:
    """Validate GFF format."""
    errors = []

    try:
        with file_path.open("r") as f:
            for i, line in enumerate(f):
                if line.startswith("#"):
                    continue  # Skip comments

                line_num = i + 1
                fields = line.strip().split("\t")

                if len(fields) != 9:
                    errors.append(
                        f"Line {line_num}: GFF must have 9 tab-separated fields"
                    )
                else:
                    # Validate coordinate fields
                    try:
                        start = int(fields[3])
                        end = int(fields[4])
                        if start > end:
                            errors.append(
                                f"Line {line_num}: Start coordinate > end coordinate"
                            )
                    except ValueError:
                        errors.append(f"Line {line_num}: Invalid coordinates")

                if i >= 100 and level == "basic":
                    break

    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")

    return errors


def _validate_table_format(file_path: Path, level: str) -> list:
    """Validate TSV/CSV table format."""
    errors = []

    try:
        with file_path.open("r") as f:
            header_cols = None

            for i, line in enumerate(f):
                line_num = i + 1

                # Determine delimiter
                delimiter = "\t" if "\t" in line else ","
                fields = line.strip().split(delimiter)

                if i == 0:  # Header row
                    header_cols = len(fields)
                    if header_cols == 1:
                        errors.append("Table appears to have only one column")
                else:
                    if len(fields) != header_cols:
                        errors.append(
                            f"Line {line_num}: Column count mismatch "
                            f"(expected {header_cols}, got {len(fields)})"
                        )

                if i >= 100 and level == "basic":
                    break

    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")

    return errors


def _validate_json_format(file_path: Path, level: str) -> list:
    """Validate JSON format."""
    errors = []

    try:
        with file_path.open("r") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        errors.append(f"Error reading JSON file: {str(e)}")

    return errors


@app.command("workflow")
def validate_workflow(
    workflow_file: Path = typer.Option(
        ..., "--workflow", "-w", exists=True, help="Workflow definition file"
    ),
    schema_file: Path | None = typer.Option(
        None, "--schema", "-s", help="Workflow schema file"
    ),
    check_dependencies: bool = typer.Option(
        True, "--deps", help="Check tool dependencies"
    ),
) -> None:
    """Validate workflow definition and dependencies."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would validate workflow: {workflow_file}")
        if schema_file:
            typer.echo(f"[DRY-RUN] Against schema: {schema_file}")
        return

    validation_result = _validate_workflow_definition(
        workflow_file, schema_file, check_dependencies
    )

    if validation_result["valid"]:
        typer.echo("✅ Workflow validation PASSED")
    else:
        typer.echo("❌ Workflow validation FAILED")
        for error in validation_result["errors"]:
            typer.echo(f"  ❌ {error}")
        raise typer.Exit(code=1)

    if validation_result["warnings"]:
        typer.echo("\nWarnings:")
        for warning in validation_result["warnings"]:
            typer.echo(f"  ⚠️  {warning}")


def _validate_workflow_definition(
    workflow_file: Path, schema_file: Path | None, check_deps: bool
) -> dict:
    """Validate workflow definition file."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        with workflow_file.open("r") as f:
            workflow_data = json.load(f)
    except Exception as e:
        errors.append(f"Cannot parse workflow file: {str(e)}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Basic workflow structure validation
    required_fields = ["name", "steps", "version"]
    for field in required_fields:
        if field not in workflow_data:
            errors.append(f"Missing required field: {field}")

    # Validate steps
    if "steps" in workflow_data:
        steps = workflow_data["steps"]
        if not isinstance(steps, list) or len(steps) == 0:
            errors.append("Workflow must have at least one step")
        else:
            for i, step in enumerate(steps):
                if "name" not in step:
                    errors.append(f"Step {i + 1}: Missing step name")
                if "command" not in step:
                    errors.append(f"Step {i + 1}: Missing command")

    # Check tool dependencies if requested
    if check_deps and "steps" in workflow_data:
        missing_tools = _check_tool_dependencies(workflow_data["steps"])
        if missing_tools:
            warnings.extend([f"Tool not found: {tool}" for tool in missing_tools])

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def _check_tool_dependencies(steps: list) -> list:
    """Check if required tools are available (mock implementation)."""
    # Mock tool availability check
    available_tools = {"fastp", "spades", "kraken2", "diamond", "prodigal"}
    missing_tools = []

    for step in steps:
        command = step.get("command", "")
        tool = command.split()[0] if command else ""
        if tool and tool not in available_tools:
            missing_tools.append(tool)

    return list(set(missing_tools))  # Remove duplicates


@app.command("schema")
def validate_schema(
    data_file: Path = typer.Option(
        ..., "--data", "-d", exists=True, help="Data file to validate"
    ),
    schema_file: Path = typer.Option(
        ..., "--schema", "-s", exists=True, help="Schema definition file"
    ),
    strict: bool = typer.Option(False, "--strict", help="Strict validation mode"),
) -> None:
    """Validate data against schema definition."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would validate {data_file} against {schema_file}")
        typer.echo(f"[DRY-RUN] Strict mode: {strict}")
        return

    validation_result = _validate_against_schema(data_file, schema_file, strict)

    if validation_result["valid"]:
        typer.echo("✅ Schema validation PASSED")
    else:
        typer.echo("❌ Schema validation FAILED")
        for error in validation_result["errors"]:
            typer.echo(f"  ❌ {error}")
        raise typer.Exit(code=1)


def _validate_against_schema(data_file: Path, schema_file: Path, strict: bool) -> dict:
    """Validate data file against schema (mock implementation)."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        # Load schema
        with schema_file.open("r") as f:
            schema = json.load(f)

        # Load data
        with data_file.open("r") as f:
            if data_file.suffix.lower() == ".json":
                data = json.load(f)
            else:
                # For non-JSON files, create mock validation
                data = {"mock": "data"}

        # Mock schema validation
        if "required_fields" in schema:
            for field in schema["required_fields"]:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

        if strict and "allowed_fields" in schema:
            for field in data.keys():
                if field not in schema["allowed_fields"]:
                    errors.append(f"Unexpected field: {field}")

    except Exception as e:
        errors.append(f"Schema validation error: {str(e)}")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


@app.command("batch")
def batch_validate(
    input_dir: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Directory containing files to validate"
    ),
    pattern: str = typer.Option("*", "--pattern", "-p", help="File pattern to match"),
    data_type: DataType = typer.Option(..., "--type", "-t", help="Expected data type"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Summary report output"
    ),
) -> None:
    """Validate multiple files in batch."""
    ctx = get_context()

    files_to_validate = list(input_dir.glob(pattern))

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would validate {len(files_to_validate)} files")
        typer.echo(f"[DRY-RUN] Pattern: {pattern}, Type: {data_type}")
        if output:
            typer.echo(f"[DRY-RUN] Report: {output}")
        return

    if not files_to_validate:
        typer.echo(f"No files found matching pattern: {pattern}")
        return

    results = []
    passed = 0
    failed = 0

    typer.echo(f"Validating {len(files_to_validate)} files...")

    for file_path in files_to_validate:
        validation_result = _validate_input_file(file_path, data_type, "standard")
        results.append(
            {
                "file": str(file_path),
                "valid": validation_result["valid"],
                "errors": validation_result["errors"],
                "warnings": validation_result["warnings"],
            }
        )

        if validation_result["valid"]:
            passed += 1
            typer.echo(f"✅ {file_path.name}")
        else:
            failed += 1
            typer.echo(
                f"❌ {file_path.name} ({len(validation_result['errors'])} errors)"
            )

    # Summary
    typer.echo("\nBatch validation complete:")
    typer.echo(f"  Passed: {passed}")
    typer.echo(f"  Failed: {failed}")
    typer.echo(f"  Total:  {len(files_to_validate)}")

    # Save detailed report
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(
                {
                    "summary": {
                        "passed": passed,
                        "failed": failed,
                        "total": len(files_to_validate),
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        typer.echo(f"\nDetailed report saved: {output}")

    # Exit with error code if any validations failed
    if failed > 0:
        raise typer.Exit(code=1)

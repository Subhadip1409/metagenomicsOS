# cli/commands/run.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import uuid
import shutil
import typer
from typing import Literal, List

from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Core workflows orchestration entry")

WorkflowType = Literal[
    "pipeline", "qc", "assembly", "taxonomy", "annotation", "binning"
]
StageStatus = Literal["pending", "running", "completed", "failed", "skipped"]

# ----- Core workflow orchestration -----


@app.command("execute")
def execute_workflow(
    workflow: WorkflowType = typer.Option(
        ..., "--workflow", "-w", help="Workflow type to execute"
    ),
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, help="Input reads/data"
    ),
    workdir: Path = typer.Option(
        ..., "--workdir", help="Working directory for all outputs"
    ),
    threads: int | None = typer.Option(
        None, "--threads", "-t", min=1, help="Number of CPU threads"
    ),
    memory: str | None = typer.Option(
        None, "--memory", "-m", help="Memory limit (e.g., '16GB')"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show execution plan without running"
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume from checkpoint ID"
    ),
    sample_sheet: Path | None = typer.Option(
        None, "--samples", exists=True, help="Sample sheet for batch processing"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", exists=True, help="Workflow configuration file"
    ),
    database_dir: Path | None = typer.Option(
        None, "--databases", help="Database directory override"
    ),
) -> None:
    """Execute core metagenomics workflows with full orchestration."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    # Override global dry-run if local flag provided
    effective_dry_run = dry_run or ctx.dry_run

    # Setup execution environment
    execution_env = _setup_execution_environment(
        workflow,
        input,
        workdir,
        threads,
        memory,
        sample_sheet,
        config_file,
        database_dir,
        cfg,
    )

    # Validate prerequisites
    validation_result = _validate_execution_prerequisites(
        execution_env, effective_dry_run
    )
    if not validation_result["valid"]:
        typer.echo("❌ Validation failed:", err=True)
        for error in validation_result["errors"]:
            typer.echo(f"   • {error}", err=True)
        raise typer.Exit(code=1)

    # Handle resume logic
    if resume:
        execution_env = _setup_resume_execution(
            execution_env, resume, effective_dry_run
        )

    # Generate execution plan
    execution_plan = _generate_execution_plan(execution_env, workflow)

    if effective_dry_run:
        _display_execution_plan(execution_plan, validation_result["warnings"])
        return

    # Execute workflow stages
    run_id = _execute_workflow_stages(execution_plan, execution_env)

    typer.echo(f"🚀 Workflow '{workflow}' completed. Run ID: {run_id}")
    typer.echo(f"📁 Results in: {execution_env['workdir']}")
    typer.echo(f"📊 Status: metagenomicsOS run status {run_id}")


def _setup_execution_environment(
    workflow: str,
    input_path: Path,
    workdir: Path,
    threads: int | None,
    memory: str | None,
    sample_sheet: Path | None,
    config_file: Path | None,
    database_dir: Path | None,
    cfg,
) -> dict:
    """Setup comprehensive execution environment."""
    workdir.mkdir(parents=True, exist_ok=True)

    # Resolve resource settings
    effective_threads = threads if threads is not None else cfg.threads
    effective_memory = memory or "16GB"
    effective_db_dir = database_dir or Path(cfg.database_dir)

    return {
        "workflow": workflow,
        "input": input_path,
        "workdir": workdir,
        "threads": effective_threads,
        "memory": effective_memory,
        "sample_sheet": sample_sheet,
        "config_file": config_file,
        "database_dir": effective_db_dir,
        "run_id": datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8],
        "checkpoint_dir": workdir / ".checkpoints",
        "logs_dir": workdir / "logs",
    }


def _validate_execution_prerequisites(env: dict, dry_run: bool) -> dict:
    """Validate databases, inputs, and system requirements."""
    errors = []
    warnings = []

    # Input validation
    if not env["input"].exists():
        errors.append(f"Input file/directory not found: {env['input']}")
    elif env["input"].stat().st_size == 0:
        errors.append(f"Input file is empty: {env['input']}")

    # Database validation for each workflow
    required_dbs = _get_required_databases(env["workflow"])
    for db_name in required_dbs:
        db_path = env["database_dir"] / db_name
        if not db_path.exists():
            errors.append(f"Required database not found: {db_name} at {db_path}")
        elif not _validate_database_integrity(db_path, dry_run):
            warnings.append(f"Database may be corrupted: {db_name}")

    # Sample sheet validation
    if env["sample_sheet"]:
        sample_validation = _validate_sample_sheet(env["sample_sheet"])
        errors.extend(sample_validation["errors"])
        warnings.extend(sample_validation["warnings"])

    # Resource validation
    if env["threads"] > 32:
        warnings.append(
            f"High thread count ({env['threads']}) may cause resource contention"
        )

    memory_gb = int(env["memory"].replace("GB", "").replace("G", ""))
    if memory_gb > 64:
        warnings.append(
            f"High memory allocation ({env['memory']}) - ensure system availability"
        )

    # Disk space validation
    available_space = _get_available_disk_space(env["workdir"])
    estimated_usage = _estimate_workflow_disk_usage(env["workflow"], env["input"])
    if available_space < estimated_usage * 1.5:  # 50% safety margin
        warnings.append(
            f"Low disk space: {available_space / 1e9:.1f}GB available, "
            f"~{estimated_usage / 1e9:.1f}GB estimated usage"
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def _get_required_databases(workflow: str) -> list[str]:
    """Get required databases for each workflow type."""
    db_requirements = {
        "pipeline": ["kraken2", "card", "pfam"],
        "qc": [],
        "assembly": [],
        "taxonomy": ["kraken2"],
        "annotation": ["pfam", "cog", "kegg"],
        "binning": [],
    }
    return db_requirements.get(workflow, [])


def _validate_database_integrity(db_path: Path, dry_run: bool) -> bool:
    """Validate database integrity (mock implementation)."""
    if dry_run:
        return True  # Skip expensive validation in dry-run

    # Mock validation - check for key files
    if (db_path / "database_info.json").exists():
        return True
    return False


def _validate_sample_sheet(sample_sheet: Path) -> dict:
    """Validate sample sheet format and contents."""
    errors = []
    warnings: List[str] = []

    try:
        with sample_sheet.open("r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            errors.append("Sample sheet must have header and at least one sample")
            return {"errors": errors, "warnings": warnings}

        # Check header
        header = lines[0].strip().split("\t")
        required_cols = ["sample_id", "input_file"]
        for col in required_cols:
            if col not in header:
                errors.append(f"Missing required column in sample sheet: {col}")

        # Check sample entries
        for i, line in enumerate(lines[1:], 2):
            if not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) != len(header):
                errors.append(f"Column count mismatch at line {i}")

            # Check if input files exist
            try:
                input_idx = header.index("input_file")
                input_file = Path(fields[input_idx])
                if not input_file.exists():
                    warnings.append(
                        f"Input file not found for sample at line {i}: {input_file}"
                    )
            except (ValueError, IndexError):
                pass

    except Exception as e:
        errors.append(f"Error reading sample sheet: {str(e)}")

    return {"errors": errors, "warnings": warnings}


def _get_available_disk_space(path: Path) -> int:
    """Get available disk space in bytes."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free
    except Exception:
        return 100 * 1024**3  # Mock 100GB available


def _estimate_workflow_disk_usage(workflow: str, input_path: Path) -> int:
    """Estimate disk usage for workflow (mock implementation)."""
    try:
        input_size = (
            input_path.stat().st_size
            if input_path.is_file()
            else sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())
        )
    except Exception:
        input_size = 1024**3  # Default 1GB

    # Workflow-specific multipliers
    multipliers = {
        "pipeline": 15,  # Full pipeline needs lots of space
        "qc": 3,
        "assembly": 8,
        "taxonomy": 2,
        "annotation": 4,
        "binning": 5,
    }

    return input_size * multipliers.get(workflow, 5)


def _generate_execution_plan(env: dict, workflow: str) -> dict:
    """Generate deterministic execution plan."""
    # Define workflow stages
    workflow_stages = {
        "pipeline": ["qc", "assembly", "taxonomy", "annotation", "binning", "report"],
        "qc": ["quality_filter", "adapter_trim", "contamination_screen"],
        "assembly": ["read_correction", "assembly", "polishing", "assessment"],
        "taxonomy": ["classification", "profiling", "diversity_analysis"],
        "annotation": ["gene_prediction", "functional_annotation", "pathway_analysis"],
        "binning": [
            "coverage_calculation",
            "binning",
            "bin_refinement",
            "quality_assessment",
        ],
    }

    stages = workflow_stages.get(workflow, [workflow])

    execution_plan = {
        "workflow": workflow,
        "run_id": env["run_id"],
        "total_stages": len(stages),
        "stages": [],
        "estimated_runtime": _estimate_total_runtime(workflow, stages),
        "resource_requirements": {
            "peak_memory": env["memory"],
            "cpu_cores": env["threads"],
            "disk_space": f"{_estimate_workflow_disk_usage(workflow, env['input']) / 1e9:.1f}GB",
        },
    }

    for i, stage_name in enumerate(stages):
        stage_plan = {
            "stage_id": i + 1,
            "name": stage_name,
            "status": "pending",
            "input_deps": _get_stage_dependencies(stage_name, i),
            "outputs": _get_stage_outputs(stage_name, env["workdir"]),
            "command_template": _get_stage_command_template(stage_name, env),
            "estimated_runtime": _estimate_stage_runtime(stage_name),
            "checkpoint_file": env["checkpoint_dir"]
            / f"stage_{i + 1}_{stage_name}.checkpoint",
        }
        execution_plan["stages"].append(stage_plan)

    return execution_plan


def _get_stage_dependencies(stage_name: str, stage_index: int) -> list[str]:
    """Get input dependencies for each stage."""
    if stage_index == 0:
        return ["input_data"]

    # Simple sequential dependencies for now
    return [f"stage_{stage_index}"]


def _get_stage_outputs(stage_name: str, workdir: Path) -> list[str]:
    """Get expected outputs for each stage."""
    stage_outputs = {
        "qc": ["filtered_reads.fastq.gz", "quality_report.html"],
        "quality_filter": ["filtered_reads.fastq.gz"],
        "adapter_trim": ["trimmed_reads.fastq.gz"],
        "contamination_screen": ["clean_reads.fastq.gz"],
        "assembly": ["contigs.fasta", "assembly_stats.json"],
        "read_correction": ["corrected_reads.fastq.gz"],
        "polishing": ["polished_contigs.fasta"],
        "assessment": ["assembly_metrics.json"],
        "taxonomy": ["taxonomic_profile.tsv", "kraken_results.txt"],
        "classification": ["classification_results.txt"],
        "profiling": ["abundance_profile.tsv"],
        "diversity_analysis": ["diversity_metrics.json"],
        "annotation": ["annotations.gff", "functional_summary.tsv"],
        "gene_prediction": ["predicted_genes.fasta"],
        "functional_annotation": ["gene_functions.tsv"],
        "pathway_analysis": ["pathway_results.json"],
        "binning": ["bins/", "binning_results.json"],
        "coverage_calculation": ["coverage.tsv"],
        "bin_refinement": ["refined_bins/"],
        "quality_assessment": ["bin_quality.json"],
        "report": ["final_report.html"],
    }

    outputs = stage_outputs.get(stage_name, [f"{stage_name}_output.txt"])
    return [str(workdir / "results" / output) for output in outputs]


def _get_stage_command_template(stage_name: str, env: dict) -> str:
    """Get command template for each stage."""
    templates = {
        "qc": f"fastp -i {{input}} -o {{output}} --thread {env['threads']}",
        "assembly": f"spades.py -1 {{input}} -o {{output}} --threads {env['threads']} --memory {env['memory'].replace('GB', '')}",
        "taxonomy": f"kraken2 --db {env['database_dir']}/kraken2 --threads {env['threads']} {{input}} --output {{output}}",
        "annotation": f"prodigal -i {{input}} -o {{output}} && diamond blastp --query {{output}} --db {env['database_dir']}/pfam",
        "binning": f"metabat2 -i {{input}} -a {{coverage}} -o {{output}} --numThreads {env['threads']}",
    }

    if stage_name in templates:
        return templates[stage_name]
    else:
        return f"echo 'Processing {stage_name}' > {{output}}"


def _estimate_stage_runtime(stage_name: str) -> str:
    """Estimate runtime for each stage."""
    runtimes: Dict[str, str] = {
        "qc": "00:15:00",
        "quality_filter": "00:10:00",
        "adapter_trim": "00:08:00",
        "contamination_screen": "00:05:00",
        "assembly": "02:30:00",
        "read_correction": "00:45:00",
        "polishing": "01:15:00",
        "assessment": "00:10:00",
        "taxonomy": "00:30:00",
        "classification": "00:25:00",
        "profiling": "00:08:00",
        "diversity_analysis": "00:05:00",
        "annotation": "01:00:00",
        "gene_prediction": "00:20:00",
        "functional_annotation": "00:35:00",
        "pathway_analysis": "00:10:00",
        "binning": "00:45:00",
        "coverage_calculation": "00:15:00",
        "bin_refinement": "00:25:00",
        "quality_assessment": "00:08:00",
        "report": "00:05:00",
    }

    return runtimes.get(stage_name, "00:10:00")


def _estimate_total_runtime(workflow: str, stages: list[str]) -> str:
    """Estimate total workflow runtime."""
    total_minutes = 0
    for stage in stages:
        stage_time = _estimate_stage_runtime(stage)
        hours, minutes, seconds = map(int, stage_time.split(":"))
        total_minutes += hours * 60 + minutes + seconds // 60

    hours = total_minutes // 60
    minutes = total_minutes % 60

    return f"{hours:02d}:{minutes:02d}:00"


def _setup_resume_execution(env: dict, resume_id: str, dry_run: bool) -> dict:
    """Setup execution environment for resume operation."""
    checkpoint_file = env["checkpoint_dir"] / f"run_{resume_id}.json"

    if not checkpoint_file.exists():
        typer.echo(f"❌ Resume checkpoint not found: {resume_id}", err=True)
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(f"[DRY-RUN] Would resume from checkpoint: {resume_id}")
        return env

    try:
        with checkpoint_file.open("r") as f:
            checkpoint_data = json.load(f)

        # Update environment with checkpoint data
        env.update(
            {
                "resume_from": resume_id,
                "completed_stages": checkpoint_data.get("completed_stages", []),
                "last_successful_stage": checkpoint_data.get("last_successful_stage"),
                "previous_run_id": checkpoint_data.get("run_id"),
            }
        )

        typer.echo(f"📂 Resuming from checkpoint: {resume_id}")
        typer.echo(f"✅ Completed stages: {len(env['completed_stages'])}")

    except Exception as e:
        typer.echo(f"❌ Error loading checkpoint: {e}", err=True)
        raise typer.Exit(code=1)

    return env


def _display_execution_plan(plan: dict, warnings: list[str]) -> None:
    """Display comprehensive dry-run execution plan."""
    typer.echo(f"🔍 DRY-RUN: Execution plan for '{plan['workflow']}' workflow")
    typer.echo("=" * 60)

    typer.echo("📊 Overview:")
    typer.echo(f"   Run ID: {plan['run_id']}")
    typer.echo(f"   Total stages: {plan['total_stages']}")
    typer.echo(f"   Estimated runtime: {plan['estimated_runtime']}")
    typer.echo(f"   Peak memory: {plan['resource_requirements']['peak_memory']}")
    typer.echo(f"   CPU cores: {plan['resource_requirements']['cpu_cores']}")
    typer.echo(f"   Disk space: {plan['resource_requirements']['disk_space']}")

    if warnings:
        typer.echo("\n⚠️  Warnings:")
        for warning in warnings:
            typer.echo(f"   • {warning}")

    typer.echo("\n📋 Execution stages:")
    for stage in plan["stages"]:
        typer.echo(
            f"   {stage['stage_id']:2d}. {stage['name']:<20} "
            f"({stage['estimated_runtime']}) → {len(stage['outputs'])} outputs"
        )

    typer.echo("\n💡 To execute: Remove --dry-run flag")
    typer.echo("📊 To monitor: metagenomicsOS monitor jobs")
    typer.echo("🔍 To validate: metagenomicsOS validate workflow")


def _execute_workflow_stages(plan: dict, env: dict) -> str:
    """Execute workflow stages with checkpointing and monitoring."""
    run_id = plan["run_id"]

    # Create execution directories
    env["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    env["logs_dir"].mkdir(parents=True, exist_ok=True)
    (env["workdir"] / "results").mkdir(parents=True, exist_ok=True)

    # Initialize run state
    run_state = {
        "run_id": run_id,
        "workflow": plan["workflow"],
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "completed_stages": env.get("completed_stages", []),
        "current_stage": None,
        "total_stages": plan["total_stages"],
    }

    typer.echo(f"🚀 Starting workflow execution: {plan['workflow']}")
    typer.echo(f"📁 Working directory: {env['workdir']}")
    typer.echo(f"🔄 Progress: 0/{plan['total_stages']} stages")

    try:
        for stage in plan["stages"]:
            stage_id = stage["stage_id"]
            stage_name = stage["name"]

            # Skip if resuming and stage already completed
            if stage_id in run_state["completed_stages"]:
                typer.echo(f"⏭️  Skipping completed stage {stage_id}: {stage_name}")
                continue

            run_state["current_stage"] = stage_id

            typer.echo(f"▶️  Stage {stage_id}/{plan['total_stages']}: {stage_name}")

            # Execute stage (mock implementation)
            success = _execute_single_stage(stage, env)

            if success:
                run_state["completed_stages"].append(stage_id)
                _save_checkpoint(
                    env["checkpoint_dir"] / f"run_{run_id}.json", run_state
                )

                # Create stage outputs
                for output_path in stage["outputs"]:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).touch()

                typer.echo(f"✅ Stage {stage_id} completed: {stage_name}")
            else:
                run_state["status"] = "failed"
                run_state["failed_stage"] = stage_id
                _save_checkpoint(
                    env["checkpoint_dir"] / f"run_{run_id}.json", run_state
                )

                typer.echo(f"❌ Stage {stage_id} failed: {stage_name}", err=True)
                typer.echo(f"💡 Resume with: --resume {run_id}", err=True)
                raise typer.Exit(code=1)

        run_state["status"] = "completed"
        run_state["completed_at"] = datetime.utcnow().isoformat()
        _save_checkpoint(env["checkpoint_dir"] / f"run_{run_id}.json", run_state)

    except KeyboardInterrupt:
        run_state["status"] = "interrupted"
        _save_checkpoint(env["checkpoint_dir"] / f"run_{run_id}.json", run_state)
        typer.echo(f"\n⏸️  Workflow interrupted. Resume with: --resume {run_id}")
        raise typer.Exit(code=1)

    return run_id


def _execute_single_stage(stage: dict, env: dict) -> bool:
    """Execute a single workflow stage."""
    stage_name = stage["name"]

    # Mock execution with realistic behavior
    import time

    time.sleep(0.5)  # Simulate processing time

    # Create stage log
    log_file = env["logs_dir"] / f"stage_{stage['stage_id']}_{stage_name}.log"
    with log_file.open("w") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] Starting stage: {stage_name}\n")
        f.write(
            f"[{datetime.utcnow().isoformat()}] Command: {stage['command_template']}\n"
        )
        f.write(f"[{datetime.utcnow().isoformat()}] Stage completed successfully\n")

    # Mock 95% success rate
    import random

    return random.random() > 0.05  # nosec


def _save_checkpoint(checkpoint_file: Path, state: dict) -> None:
    """Save execution checkpoint."""
    state["updated_at"] = datetime.utcnow().isoformat()

    with checkpoint_file.open("w") as f:
        json.dump(state, f, indent=2)


@app.command("plan")
def show_plan(
    workflow: WorkflowType = typer.Option(
        ..., "--workflow", "-w", help="Workflow to plan"
    ),
    input: Path = typer.Option(..., "--input", "-i", exists=True, help="Input data"),
    workdir: Path = typer.Option(
        Path("./workdir"), "--workdir", help="Working directory"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", help="Configuration file"
    ),
) -> None:
    """Show detailed execution plan without running."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    env = _setup_execution_environment(
        workflow, input, workdir, None, None, None, config_file, None, cfg
    )

    plan = _generate_execution_plan(env, workflow)
    validation = _validate_execution_prerequisites(env, dry_run=True)

    _display_execution_plan(plan, validation["warnings"])


@app.command("resume")
def resume_workflow(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID to resume from"),
    workdir: Path | None = typer.Option(
        None, "--workdir", help="Override working directory"
    ),
) -> None:
    """Resume interrupted workflow from checkpoint."""
    # This is a convenience command that calls execute with --resume
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would resume checkpoint: {checkpoint_id}")
        return

    typer.echo(f"🔄 Resuming workflow from checkpoint: {checkpoint_id}")
    typer.echo("💡 Use 'run execute --resume' for more control over resume parameters")


@app.command("validate")
def validate_workflow(
    workflow: WorkflowType = typer.Option(
        ..., "--workflow", "-w", help="Workflow to validate"
    ),
    workdir: Path = typer.Option(
        ..., "--workdir", help="Working directory to validate"
    ),
    check_databases: bool = typer.Option(
        True, "--check-dbs", help="Validate required databases"
    ),
    check_outputs: bool = typer.Option(
        True, "--check-outputs", help="Validate expected outputs"
    ),
) -> None:
    """Validate workflow setup and outputs."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    typer.echo(f"🔍 Validating {workflow} workflow setup")

    # Create minimal environment for validation
    env = {
        "workflow": workflow,
        "workdir": workdir,
        "database_dir": Path(cfg.database_dir),
        "input": Path("mock_input"),  # Placeholder for validation
    }

    validation = _validate_execution_prerequisites(env, dry_run=False)

    if validation["valid"]:
        typer.echo("✅ Workflow validation passed")
    else:
        typer.echo("❌ Workflow validation failed:")
        for error in validation["errors"]:
            typer.echo(f"   • {error}")

    if validation["warnings"]:
        typer.echo("⚠️  Warnings:")
        for warning in validation["warnings"]:
            typer.echo(f"   • {warning}")

    if check_outputs and workdir.exists():
        _validate_workflow_outputs(workflow, workdir)


def _validate_workflow_outputs(workflow: str, workdir: Path) -> None:
    """Validate expected workflow outputs exist."""
    results_dir = workdir / "results"
    if not results_dir.exists():
        typer.echo("❌ Results directory not found")
        return

    # Get expected outputs based on workflow
    if workflow == "pipeline":
        expected_stages = [
            "qc",
            "assembly",
            "taxonomy",
            "annotation",
            "binning",
            "report",
        ]
    else:
        expected_stages = [workflow]

    missing_outputs = []
    for stage in expected_stages:
        stage_outputs = _get_stage_outputs(stage, workdir)
        for output_path in stage_outputs:
            if not Path(output_path).exists():
                missing_outputs.append(output_path)

    if missing_outputs:
        typer.echo(f"❌ Missing {len(missing_outputs)} expected outputs:")
        for output in missing_outputs[:5]:  # Show first 5
            typer.echo(f"   • {output}")
        if len(missing_outputs) > 5:
            typer.echo(f"   ... and {len(missing_outputs) - 5} more")
    else:
        typer.echo("✅ All expected outputs found")

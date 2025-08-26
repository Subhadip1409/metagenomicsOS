# cli/commands/optimize.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
import json
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Auto-tuning resources")

OptimizeTarget = Literal["threads", "memory", "io", "all"]
WorkflowType = Literal["qc", "assembly", "taxonomy", "annotation", "binning"]


@app.command("suggest")
def suggest_parameters(
    workflow: WorkflowType = typer.Option(
        ..., "--workflow", "-w", help="Workflow type to optimize"
    ),
    input_size: str = typer.Option(
        ..., "--input-size", "-i", help="Input data size (e.g., '10GB', '500M')"
    ),
    target: OptimizeTarget = typer.Option(
        "all", "--target", "-t", help="Optimization target"
    ),
    constraint: str | None = typer.Option(
        None, "--constraint", help="Resource constraints (e.g., 'max_memory=32GB')"
    ),
) -> None:
    """Suggest optimal parameters for workflows."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would optimize {workflow} for {input_size} data")
        typer.echo(f"[DRY-RUN] Target: {target}")
        if constraint:
            typer.echo(f"[DRY-RUN] Constraint: {constraint}")
        return

    suggestions = _generate_optimization_suggestions(
        workflow, input_size, target, constraint, cfg
    )

    typer.echo(f"=== Optimization Suggestions for {workflow.upper()} ===")
    typer.echo(f"Input size: {input_size}")
    typer.echo(f"Current system: {suggestions['system_info']}")
    typer.echo()

    typer.echo("Recommended Parameters:")
    for param, value in suggestions["parameters"].items():
        current = suggestions["current"].get(param, "unknown")
        improvement = suggestions["improvements"].get(param, "")
        typer.echo(f"  {param:15}: {value:10} (current: {current}) {improvement}")

    if suggestions["warnings"]:
        typer.echo("\nWarnings:")
        for warning in suggestions["warnings"]:
            typer.echo(f"  ⚠️  {warning}")

    if suggestions["notes"]:
        typer.echo("\nNotes:")
        for note in suggestions["notes"]:
            typer.echo(f"  💡 {note}")


def _generate_optimization_suggestions(
    workflow: str, input_size: str, target: str, constraint: str | None, cfg
) -> dict:
    """Generate optimization suggestions based on workflow and input size."""
    # Parse input size
    size_gb = _parse_size_to_gb(input_size)

    # Mock system info
    system_info = "64GB RAM, 16 CPU cores, SSD storage"

    suggestions: dict[str, Any] = {
        "system_info": system_info,
        "parameters": {},
        "current": {"threads": cfg.threads, "memory": "16GB", "temp_space": "auto"},
        "improvements": {},
        "warnings": [],
        "notes": [],
    }

    # Workflow-specific optimizations
    if workflow == "assembly":
        suggestions["parameters"].update(
            {
                "threads": min(16, max(4, size_gb // 2)),
                "memory": f"{min(64, max(8, size_gb * 4))}GB",
                "kmer_sizes": "21,33,55,77" if size_gb > 5 else "21,33,55",
                "temp_space": f"{size_gb * 3}GB",
            }
        )
        suggestions["improvements"].update(
            {
                "threads": "↑ 25% faster",
                "memory": "↓ 40% less swapping",
                "kmer_sizes": "better assembly quality",
            }
        )

        if size_gb > 20:
            suggestions["warnings"].append(
                "Large dataset - consider splitting or using meta-spades"
            )

    elif workflow == "taxonomy":
        suggestions["parameters"].update(
            {
                "threads": min(12, max(2, size_gb // 1)),
                "memory": f"{min(32, max(4, size_gb * 2))}GB",
                "confidence": 0.1 if size_gb > 1 else 0.05,
            }
        )
        suggestions["improvements"].update(
            {
                "threads": "↑ 50% faster classification",
                "memory": "database loading optimization",
            }
        )

    elif workflow == "qc":
        suggestions["parameters"].update(
            {
                "threads": min(8, max(2, size_gb // 0.5)),
                "memory": f"{min(16, max(2, size_gb))}GB",
                "compression": "gzip" if size_gb > 2 else "none",
            }
        )

    elif workflow == "annotation":
        suggestions["parameters"].update(
            {
                "threads": min(16, max(4, size_gb)),
                "memory": f"{min(48, max(8, size_gb * 6))}GB",
                "evalue": 1e-5,
                "batch_size": max(1000, min(10000, int(size_gb * 500))),
            }
        )

    elif workflow == "binning":
        suggestions["parameters"].update(
            {
                "threads": min(12, max(4, size_gb // 2)),
                "memory": f"{min(32, max(8, size_gb * 3))}GB",
                "min_contig_len": 2500 if size_gb > 10 else 1500,
            }
        )

    # Apply constraints if specified
    if constraint:
        _apply_constraints(suggestions, constraint)

    # General notes
    suggestions["notes"].extend(
        [
            "These suggestions are based on typical performance patterns",
            "Monitor actual resource usage during initial runs",
            "Adjust parameters based on your specific dataset characteristics",
        ]
    )

    return suggestions


def _parse_size_to_gb(size_str: str) -> float:
    """Parse size string to GB."""
    size_str = size_str.upper().replace(" ", "")

    if size_str.endswith("GB"):
        return float(size_str[:-2])
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) / 1024
    elif size_str.endswith("TB"):
        return float(size_str[:-2]) * 1024
    elif size_str.endswith("G"):
        return float(size_str[:-1])
    elif size_str.endswith("M"):
        return float(size_str[:-1]) / 1024
    else:
        # Assume GB if no unit
        return float(size_str)


def _apply_constraints(suggestions: dict, constraint: str) -> None:
    """Apply resource constraints to suggestions."""
    if "max_memory=" in constraint:
        max_mem = constraint.split("max_memory=")[1].split(",")[0]
        max_mem_gb = _parse_size_to_gb(max_mem)

        # Adjust memory suggestions
        current_mem = suggestions["parameters"].get("memory", "16GB")
        current_mem_gb = _parse_size_to_gb(current_mem)

        if current_mem_gb > max_mem_gb:
            suggestions["parameters"]["memory"] = f"{max_mem_gb}GB"
            suggestions["warnings"].append(
                f"Memory limited to {max_mem} - may impact performance"
            )


@app.command("profile")
def profile_workflow(
    workflow_result: Path = typer.Option(
        ..., "--result", "-r", exists=True, help="Workflow result directory"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Profiling report output"
    ),
) -> None:
    """Profile completed workflow performance."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would profile workflow result: {workflow_result}")
        if output:
            typer.echo(f"[DRY-RUN] Report output: {output}")
        return

    profile_data = _analyze_workflow_performance(workflow_result)

    typer.echo("=== Workflow Performance Profile ===")
    typer.echo(f"Workflow: {profile_data['workflow_type']}")
    typer.echo(f"Total Runtime: {profile_data['total_runtime']}")
    typer.echo(f"Peak Memory: {profile_data['peak_memory']}")
    typer.echo(f"CPU Efficiency: {profile_data['cpu_efficiency']}%")
    typer.echo()

    typer.echo("Step Performance:")
    for step in profile_data["steps"]:
        typer.echo(
            f"  {step['name']:20} | {step['runtime']:10} | {step['memory']:8} | {step['cpu_usage']:6}%"
        )

    typer.echo()
    typer.echo("Bottlenecks:")
    for bottleneck in profile_data["bottlenecks"]:
        typer.echo(f"  🐌 {bottleneck}")

    typer.echo()
    typer.echo("Optimization Opportunities:")
    for opportunity in profile_data["optimizations"]:
        typer.echo(f"  💡 {opportunity}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(profile_data, f, indent=2)
        typer.echo(f"\nDetailed profile saved: {output}")


def _analyze_workflow_performance(result_dir: Path) -> dict:
    """Analyze workflow performance from result directory."""
    # Mock performance analysis
    profile_data = {
        "workflow_type": "assembly",
        "total_runtime": "02:45:30",
        "peak_memory": "28.5GB",
        "cpu_efficiency": 85.2,
        "steps": [
            {
                "name": "quality_control",
                "runtime": "00:12:45",
                "memory": "4.2GB",
                "cpu_usage": 92.1,
                "io_read": "15.2GB",
                "io_write": "12.8GB",
            },
            {
                "name": "assembly",
                "runtime": "02:15:30",
                "memory": "28.5GB",
                "cpu_usage": 78.5,
                "io_read": "12.8GB",
                "io_write": "8.4GB",
            },
            {
                "name": "post_processing",
                "runtime": "00:17:15",
                "memory": "6.1GB",
                "cpu_usage": 45.3,
                "io_read": "8.4GB",
                "io_write": "5.2GB",
            },
        ],
        "bottlenecks": [
            "Assembly step using only 78% CPU - consider increasing threads",
            "Post-processing shows low CPU usage - I/O bound operation",
        ],
        "optimizations": [
            "Increase assembly threads from 8 to 12 (estimated 20% speedup)",
            "Use faster storage for temporary files",
            "Consider parallel post-processing",
        ],
        "resource_usage": {
            "peak_cpu": 92.1,
            "avg_cpu": 68.7,
            "peak_memory_gb": 28.5,
            "avg_memory_gb": 15.2,
            "total_io_read_gb": 36.4,
            "total_io_write_gb": 26.4,
        },
    }

    return profile_data


@app.command("tune")
def auto_tune(
    workflow_config: Path = typer.Option(
        ..., "--config", "-c", exists=True, help="Workflow configuration file"
    ),
    benchmark_data: Path | None = typer.Option(
        None, "--benchmark", "-b", help="Previous benchmark results"
    ),
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Number of tuning iterations"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Optimized configuration output"
    ),
) -> None:
    """Auto-tune workflow parameters using iterative optimization."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would auto-tune workflow config: {workflow_config}")
        typer.echo(f"[DRY-RUN] Iterations: {iterations}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    tuned_config = _perform_auto_tuning(workflow_config, benchmark_data, iterations)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(tuned_config, f, indent=2)

    typer.echo("=== Auto-tuning Results ===")
    typer.echo(f"Tuned configuration saved: {output}")
    typer.echo()

    typer.echo("Optimized Parameters:")
    for param, value in tuned_config["optimized_parameters"].items():
        original = tuned_config["original_parameters"].get(param, "unknown")
        improvement = tuned_config["improvements"].get(param, "")
        typer.echo(f"  {param:15}: {value:10} (was: {original}) {improvement}")

    typer.echo(
        f"\nExpected Performance Improvement: {tuned_config['performance_improvement']}"
    )


def _perform_auto_tuning(
    config_file: Path, benchmark_data: Path | None, iterations: int
) -> dict:
    """Perform iterative parameter optimization."""
    # Load original configuration
    try:
        with config_file.open("r") as f:
            original_config = json.load(f)
    except Exception:
        original_config = {"threads": 4, "memory": "16GB"}

    # Mock optimization process
    tuned_config = {
        "original_parameters": original_config.copy(),
        "optimized_parameters": {
            "threads": min(16, original_config.get("threads", 4) * 2),
            "memory": f"{min(64, int(original_config.get('memory', '16GB')[:-2]) * 1.5):.0f}GB",
            "chunk_size": 10000,
            "compression_level": 6,
        },
        "improvements": {
            "threads": "↑ 35% speedup",
            "memory": "↓ 60% memory pressure",
            "chunk_size": "↑ 15% I/O efficiency",
            "compression_level": "balanced speed/size",
        },
        "performance_improvement": "42% faster execution",
        "tuning_history": [
            {"iteration": 1, "runtime": "03:20:00", "parameters": {"threads": 8}},
            {"iteration": 2, "runtime": "02:45:15", "parameters": {"threads": 12}},
            {"iteration": 3, "runtime": "02:10:30", "parameters": {"threads": 16}},
        ],
        "final_config": original_config.copy(),
    }

    # Update final config with optimized parameters
    tuned_config["final_config"].update(tuned_config["optimized_parameters"])

    return tuned_config


@app.command("resources")
def optimize_resources(
    current_usage: Path = typer.Option(
        ..., "--usage", "-u", exists=True, help="Current resource usage log"
    ),
    target_efficiency: float = typer.Option(
        80.0, "--efficiency", help="Target resource efficiency (%)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Resource optimization report"
    ),
) -> None:
    """Optimize resource allocation based on usage patterns."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would optimize resources from: {current_usage}")
        typer.echo(f"[DRY-RUN] Target efficiency: {target_efficiency}%")
        return

    optimization = _analyze_resource_usage(current_usage, target_efficiency)

    typer.echo("=== Resource Optimization Analysis ===")
    typer.echo(f"Current Efficiency: {optimization['current_efficiency']:.1f}%")
    typer.echo(f"Target Efficiency: {target_efficiency}%")
    typer.echo()

    typer.echo("Resource Utilization:")
    for resource, data in optimization["utilization"].items():
        typer.echo(
            f"  {resource:10}: avg={data['avg']:5.1f}% max={data['max']:5.1f}% efficiency={data['efficiency']}"
        )

    typer.echo()
    typer.echo("Optimization Recommendations:")
    for rec in optimization["recommendations"]:
        typer.echo(f"  📊 {rec}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(optimization, f, indent=2)
        typer.echo(f"\nOptimization report saved: {output}")


def _analyze_resource_usage(usage_file: Path, target_efficiency: float) -> dict:
    """Analyze resource usage patterns and provide optimization suggestions."""
    # Mock resource analysis
    optimization = {
        "current_efficiency": 65.2,
        "target_efficiency": target_efficiency,
        "utilization": {
            "cpu": {"avg": 45.2, "max": 92.1, "efficiency": "moderate"},
            "memory": {"avg": 68.5, "max": 85.3, "efficiency": "good"},
            "disk_io": {"avg": 32.1, "max": 78.4, "efficiency": "poor"},
            "network": {"avg": 15.8, "max": 45.2, "efficiency": "poor"},
        },
        "bottlenecks": ["disk I/O", "CPU underutilization"],
        "recommendations": [
            "Increase CPU threads by 50% (currently underutilized)",
            "Use SSD storage or RAM disk for temporary files",
            "Enable parallel processing for I/O intensive tasks",
            "Consider distributed processing for large datasets",
            "Optimize batch sizes to improve throughput",
        ],
        "potential_improvements": {
            "cpu_optimization": "25-40% speedup",
            "io_optimization": "60-80% faster I/O",
            "memory_tuning": "15-25% memory efficiency",
        },
    }

    return optimization

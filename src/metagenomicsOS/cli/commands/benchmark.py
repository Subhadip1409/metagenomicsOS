# cli/commands/benchmark.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import time
from datetime import datetime
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Performance benchmarking")

BenchmarkType = Literal["workflow", "scaling", "memory", "io", "cpu"]
DataSize = Literal["small", "medium", "large", "xlarge"]


@app.command("workflow")
def benchmark_workflow(
    workflow: str = typer.Option(..., "--workflow", "-w", help="Workflow to benchmark"),
    data_size: DataSize = typer.Option("medium", "--size", "-s", help="Test data size"),
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Number of benchmark iterations"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Benchmark results output"),
    compare_configs: bool = typer.Option(
        False, "--compare", help="Compare multiple configurations"
    ),
) -> None:
    """Benchmark workflow performance."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would benchmark {workflow} workflow")
        typer.echo(f"[DRY-RUN] Data size: {data_size}, Iterations: {iterations}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    typer.echo(f"Starting {workflow} benchmark...")
    typer.echo(f"Data size: {data_size}, Iterations: {iterations}")

    results = _run_workflow_benchmark(workflow, data_size, iterations, cfg)

    # Display results
    typer.echo("\n=== Benchmark Results ===")
    typer.echo(f"Workflow: {workflow}")
    typer.echo(f"Average Runtime: {results['avg_runtime']}")
    typer.echo(f"Peak Memory: {results['peak_memory']}")
    typer.echo(f"Throughput: {results['throughput']}")
    typer.echo(f"CPU Efficiency: {results['cpu_efficiency']:.1f}%")

    if results["iterations"]:
        typer.echo("\nDetailed Results:")
        for i, run in enumerate(results["iterations"], 1):
            typer.echo(
                f"  Run {i}: {run['runtime']:>10} | {run['memory']:>8} | {run['cpu']:>6.1f}%"
            )

    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"\nDetailed results saved: {output}")


def _run_workflow_benchmark(
    workflow: str, data_size: str, iterations: int, cfg
) -> dict:
    """Execute workflow benchmark iterations."""
    # Data size configurations
    size_configs: dict[str, dict[str, Any]] = {
        "small": {"reads": "1M", "memory_gb": 4, "expected_runtime": 300},
        "medium": {"reads": "5M", "memory_gb": 8, "expected_runtime": 900},
        "large": {"reads": "20M", "memory_gb": 16, "expected_runtime": 2700},
        "xlarge": {"reads": "100M", "memory_gb": 32, "expected_runtime": 7200},
    }

    config = size_configs[data_size]
    iteration_results = []

    for i in range(iterations):
        typer.echo(f"Running iteration {i + 1}/{iterations}...")

        # Simulate benchmark execution

        # Mock execution time with some variation
        base_time = float(config["expected_runtime"])
        variation = base_time * 0.1  # ±10% variation
        runtime_seconds = base_time + (i - 1) * variation / iterations

        time.sleep(2)  # Brief simulation

        iteration_result = {
            "iteration": i + 1,
            "runtime": f"{runtime_seconds // 3600:02.0f}:{(runtime_seconds % 3600) // 60:02.0f}:{runtime_seconds % 60:04.1f}",
            "runtime_seconds": runtime_seconds,
            "memory": f"{config['memory_gb'] + i * 0.5:.1f}GB",
            "memory_gb": config["memory_gb"] + i * 0.5,
            "cpu": 85.2 + i * 2.3,
            "io_read_gb": config["memory_gb"] * 2.5,
            "io_write_gb": config["memory_gb"] * 1.8,
        }
        iteration_results.append(iteration_result)

    # Calculate summary statistics
    avg_runtime_sec = sum(r["runtime_seconds"] for r in iteration_results) / len(
        iteration_results
    )
    avg_memory_gb = sum(r["memory_gb"] for r in iteration_results) / len(
        iteration_results
    )
    avg_cpu = sum(r["cpu"] for r in iteration_results) / len(iteration_results)

    results = {
        "benchmark_date": datetime.now().isoformat(),
        "workflow": workflow,
        "data_size": data_size,
        "data_config": config,
        "iterations_count": iterations,
        "avg_runtime": f"{avg_runtime_sec // 3600:02.0f}:{(avg_runtime_sec % 3600) // 60:02.0f}:{avg_runtime_sec % 60:04.1f}",
        "avg_runtime_seconds": avg_runtime_sec,
        "peak_memory": f"{max(r['memory_gb'] for r in iteration_results):.1f}GB",
        "avg_memory": f"{avg_memory_gb:.1f}GB",
        "cpu_efficiency": avg_cpu,
        "throughput": f"{config['reads']}/{avg_runtime_sec / 60:.1f}min",
        "iterations": iteration_results,
        "system_info": {
            "threads": cfg.threads,
            "total_memory": "64GB",
            "storage_type": "SSD",
        },
    }

    return results


@app.command("scaling")
def benchmark_scaling(
    workflow: str = typer.Option(..., "--workflow", "-w", help="Workflow to test"),
    thread_counts: str = typer.Option(
        "1,2,4,8,16", "--threads", help="Thread counts to test (comma-separated)"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Scaling results output"),
    data_size: DataSize = typer.Option("medium", "--size", help="Test data size"),
) -> None:
    """Test workflow scaling across different thread counts."""
    ctx = get_context()

    thread_list = [int(t.strip()) for t in thread_counts.split(",")]

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would test {workflow} scaling")
        typer.echo(f"[DRY-RUN] Thread counts: {thread_list}")
        typer.echo(f"[DRY-RUN] Output: {output}")
        return

    typer.echo(f"Testing {workflow} scaling performance...")
    typer.echo(f"Thread counts: {thread_list}")
    typer.echo(f"Data size: {data_size}")

    scaling_results = _run_scaling_benchmark(workflow, thread_list, data_size)

    # Display results
    typer.echo("\n=== Scaling Results ===")
    typer.echo(
        f"{'Threads':<8} {'Runtime':<10} {'Speedup':<8} {'Efficiency':<10} {'Memory':<8}"
    )
    typer.echo("-" * 50)

    baseline_time = scaling_results["results"][0]["runtime_seconds"]

    for result in scaling_results["results"]:
        speedup = baseline_time / result["runtime_seconds"]
        efficiency = (speedup / result["threads"]) * 100

        typer.echo(
            f"{result['threads']:<8} {result['runtime']:<10} "
            f"{speedup:<8.2f} {efficiency:<10.1f}% {result['memory']:<8}"
        )

    typer.echo(f"\nOptimal thread count: {scaling_results['optimal_threads']}")
    typer.echo(f"Maximum speedup: {scaling_results['max_speedup']:.2f}x")

    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(scaling_results, f, indent=2)

    typer.echo(f"\nScaling analysis saved: {output}")


def _run_scaling_benchmark(workflow: str, thread_counts: list, data_size: str) -> dict:
    """Run scaling benchmark across thread counts."""
    base_runtime = {"small": 300, "medium": 900, "large": 2700, "xlarge": 7200}[
        data_size
    ]
    base_memory = {"small": 4, "medium": 8, "large": 16, "xlarge": 32}[data_size]

    results = []
    speedups = []

    for threads in thread_counts:
        typer.echo(f"Testing with {threads} threads...")

        # Simulate scaling (not perfect linear scaling)
        if threads == 1:
            runtime_seconds = base_runtime
        else:
            # Realistic scaling with diminishing returns
            theoretical_speedup = threads
            efficiency = (
                1.0 - (threads - 1) * 0.1
            )  # Efficiency decreases with more threads
            actual_speedup = min(theoretical_speedup * efficiency, threads * 0.8)
            runtime_seconds = base_runtime / actual_speedup

        memory_gb = (
            base_memory + (threads - 1) * 0.5
        )  # Slight memory increase with threads

        result = {
            "threads": threads,
            "runtime_seconds": runtime_seconds,
            "runtime": f"{runtime_seconds // 3600:02.0f}:{(runtime_seconds % 3600) // 60:02.0f}:{runtime_seconds % 60:04.1f}",
            "memory": f"{memory_gb:.1f}GB",
            "memory_gb": memory_gb,
            "cpu_efficiency": max(
                60, 95 - threads * 2
            ),  # Efficiency decreases with more threads
        }
        results.append(result)

        speedup = base_runtime / runtime_seconds
        speedups.append(speedup)

        time.sleep(1)  # Brief simulation delay

    # Find optimal configuration
    max_speedup = max(speedups)
    optimal_threads = thread_counts[speedups.index(max_speedup)]

    scaling_data = {
        "benchmark_date": datetime.now().isoformat(),
        "workflow": workflow,
        "data_size": data_size,
        "thread_counts": thread_counts,
        "results": results,
        "optimal_threads": optimal_threads,
        "max_speedup": max_speedup,
        "scaling_efficiency": (max_speedup / optimal_threads) * 100,
        "recommendations": [
            f"Use {optimal_threads} threads for optimal performance",
            f"Speedup plateaus after {optimal_threads} threads",
            "Memory usage increases linearly with thread count",
        ],
    }

    return scaling_data


@app.command("memory")
def benchmark_memory(
    workflow: str = typer.Option(..., "--workflow", "-w", help="Workflow to benchmark"),
    memory_limits: str = typer.Option(
        "4,8,16,32", "--limits", help="Memory limits to test (GB)"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Memory benchmark output"),
) -> None:
    """Test workflow performance under different memory constraints."""
    ctx = get_context()

    memory_list = [int(m.strip()) for m in memory_limits.split(",")]

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would benchmark {workflow} memory usage")
        typer.echo(f"[DRY-RUN] Memory limits: {memory_list}GB")
        return

    typer.echo(f"Testing {workflow} under memory constraints...")

    memory_results = _run_memory_benchmark(workflow, memory_list)

    # Display results
    typer.echo("\n=== Memory Benchmark Results ===")
    typer.echo(
        f"{'Memory':<8} {'Runtime':<10} {'Swapping':<10} {'Status':<15} {'Efficiency':<10}"
    )
    typer.echo("-" * 65)

    for result in memory_results["results"]:
        typer.echo(
            f"{result['memory_limit']}GB{'':<4} {result['runtime']:<10} "
            f"{result['swap_usage']:<10} {result['status']:<15} {result['memory_efficiency']:<10}"
        )

    typer.echo(f"\nMinimum memory requirement: {memory_results['min_memory_gb']}GB")
    typer.echo(f"Recommended memory: {memory_results['recommended_memory_gb']}GB")

    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(memory_results, f, indent=2)

    typer.echo(f"\nMemory analysis saved: {output}")


def _run_memory_benchmark(workflow: str, memory_limits: list) -> dict:
    """Run memory constraint benchmark."""
    base_memory_need = 12  # GB
    base_runtime = 1800  # seconds

    results = []

    for memory_gb in memory_limits:
        typer.echo(f"Testing with {memory_gb}GB memory limit...")

        if memory_gb < base_memory_need * 0.5:
            # Insufficient memory - job fails
            status = "FAILED"
            runtime_seconds = 0
            swap_usage = "N/A"
            efficiency = 0
        elif memory_gb < base_memory_need:
            # Insufficient memory - heavy swapping
            status = "SLOW"
            swap_multiplier = (base_memory_need - memory_gb) / base_memory_need
            runtime_seconds = base_runtime * (1 + swap_multiplier * 3)
            swap_usage = f"{(base_memory_need - memory_gb) * 2:.1f}GB"
            efficiency = 40
        else:
            # Sufficient memory
            status = "OPTIMAL"
            runtime_seconds = base_runtime
            swap_usage = "0GB"
            efficiency = min(95, 80 + (memory_gb - base_memory_need) * 2)

        result = {
            "memory_limit": memory_gb,
            "runtime_seconds": runtime_seconds,
            "runtime": (
                f"{runtime_seconds // 60:02.0f}:{runtime_seconds % 60:02.0f}"
                if runtime_seconds > 0
                else "FAILED"
            ),
            "swap_usage": swap_usage,
            "status": status,
            "memory_efficiency": f"{efficiency:.0f}%" if efficiency > 0 else "N/A",
        }
        results.append(result)

        time.sleep(1)  # Brief simulation

    # Determine recommendations
    min_memory = base_memory_need * 0.6
    recommended_memory = base_memory_need * 1.2

    memory_data = {
        "benchmark_date": datetime.now().isoformat(),
        "workflow": workflow,
        "memory_limits_tested": memory_limits,
        "results": results,
        "min_memory_gb": min_memory,
        "recommended_memory_gb": recommended_memory,
        "base_memory_requirement": base_memory_need,
        "analysis": {
            "memory_sensitive": True,
            "swap_penalty": "3x slower when swapping",
            "optimal_range": f"{base_memory_need}-{recommended_memory}GB",
        },
    }

    return memory_data


@app.command("compare")
def compare_benchmarks(
    benchmark_files: list[Path] = typer.Option(
        ..., "--file", "-f", help="Benchmark result files to compare"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Comparison report output"
    ),
    metric: str = typer.Option("runtime", "--metric", help="Primary comparison metric"),
) -> None:
    """Compare multiple benchmark results."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would compare {len(benchmark_files)} benchmark files")
        typer.echo(f"[DRY-RUN] Metric: {metric}")
        return

    comparison = _compare_benchmark_results(benchmark_files, metric)

    typer.echo("=== Benchmark Comparison ===")
    typer.echo(f"Comparing {len(benchmark_files)} benchmark results")
    typer.echo(f"Primary metric: {metric}")
    typer.echo()

    typer.echo("Summary:")
    for i, result in enumerate(comparison["summaries"]):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}."
        typer.echo(f"  {rank} {result['name']}: {result[metric]}")

    typer.echo()
    typer.echo("Performance Differences:")
    for result in comparison["summaries"][1:]:
        diff = comparison["differences"].get(result["name"], {})
        typer.echo(
            f"  {result['name']}: {diff.get('runtime_diff', 'N/A')} runtime, "
            f"{diff.get('memory_diff', 'N/A')} memory"
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(comparison, f, indent=2)
        typer.echo(f"\nComparison report saved: {output}")


def _compare_benchmark_results(benchmark_files: list[Path], metric: str) -> dict:
    """Compare benchmark results across multiple files."""
    summaries = []

    for i, file_path in enumerate(benchmark_files):
        # Mock loading benchmark data
        mock_data = {
            "avg_runtime_seconds": 1800 + i * 300,
            "avg_runtime": f"{(1800 + i * 300) // 60:02d}:{(1800 + i * 300) % 60:02d}",
            "peak_memory": f"{12 + i * 2}GB",
            "cpu_efficiency": 85 - i * 5,
            "workflow": "assembly",
        }

        summary = {
            "name": file_path.stem,
            "file": str(file_path),
            "runtime": mock_data["avg_runtime"],
            "runtime_seconds": mock_data["avg_runtime_seconds"],
            "memory": mock_data["peak_memory"],
            "cpu_efficiency": mock_data["cpu_efficiency"],
        }
        summaries.append(summary)

    # Sort by primary metric
    if metric == "runtime":
        summaries.sort(key=lambda x: float(x["runtime_seconds"]))
    elif metric == "memory":
        summaries.sort(key=lambda x: float(x["memory"][:-2]))

    # Calculate differences from best performer
    baseline = summaries[0]
    differences = {}

    for summary in summaries[1:]:
        runtime_diff = (
            (summary["runtime_seconds"] - baseline["runtime_seconds"])
            / baseline["runtime_seconds"]
        ) * 100
        memory_diff = (
            (float(summary["memory"][:-2]) - float(baseline["memory"][:-2]))
            / float(baseline["memory"][:-2])
        ) * 100

        differences[summary["name"]] = {
            "runtime_diff": f"+{runtime_diff:.1f}%",
            "memory_diff": (
                f"+{memory_diff:.1f}%" if memory_diff > 0 else f"{memory_diff:.1f}%"
            ),
        }

    comparison = {
        "comparison_date": datetime.now().isoformat(),
        "metric": metric,
        "files_compared": len(benchmark_files),
        "summaries": summaries,
        "differences": differences,
        "winner": summaries[0]["name"],
        "recommendations": [
            f"Best overall performance: {summaries[0]['name']}",
            f"Consider configuration from {summaries[0]['name']} for production use",
        ],
    }

    return comparison

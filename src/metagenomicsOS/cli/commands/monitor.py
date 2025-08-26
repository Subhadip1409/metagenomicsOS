# cli/commands/monitor.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import json
import time
from datetime import datetime, timedelta
import typer
from metagenomicsOS.cli.core.context import get_context
from metagenomicsOS.cli.core.config_model import load_config

app = typer.Typer(help="Monitoring")

ResourceType = Literal["cpu", "memory", "disk", "network", "all"]
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


@app.command("status")
def system_status() -> None:
    """Show current system status and resource usage."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo("[DRY-RUN] Would display system status")
        return

    status_info = _get_system_status()

    typer.echo("=== System Status ===")
    typer.echo(f"Timestamp: {status_info['timestamp']}")
    typer.echo(f"Uptime: {status_info['uptime']}")
    typer.echo()

    typer.echo("CPU Usage:")
    typer.echo(f"  Overall: {status_info['cpu']['overall']:.1f}%")
    typer.echo(f"  Per Core: {status_info['cpu']['per_core']}")
    typer.echo()

    typer.echo("Memory Usage:")
    typer.echo(
        f"  Used: {status_info['memory']['used_gb']:.1f}GB / {status_info['memory']['total_gb']:.1f}GB"
    )
    typer.echo(f"  Usage: {status_info['memory']['percent']:.1f}%")
    typer.echo()

    typer.echo("Disk Usage:")
    for mount, usage in status_info["disk"].items():
        typer.echo(
            f"  {mount}: {usage['used_gb']:.1f}GB / {usage['total_gb']:.1f}GB ({usage['percent']:.1f}%)"
        )


def _get_system_status() -> dict:
    """Get mock system status information."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime": "5 days, 12:34:56",
        "cpu": {
            "overall": 35.2,
            "per_core": [32.1, 38.5, 29.7, 40.2, 31.8, 36.9, 33.4, 39.1],
        },
        "memory": {"total_gb": 64.0, "used_gb": 28.5, "percent": 44.5},
        "disk": {
            "/": {"total_gb": 500.0, "used_gb": 125.8, "percent": 25.2},
            "/data": {"total_gb": 2000.0, "used_gb": 850.3, "percent": 42.5},
        },
        "network": {"bytes_sent": 1250000000, "bytes_recv": 2500000000},
    }


@app.command("resources")
def monitor_resources(
    resource: ResourceType = typer.Option(
        "all", "--type", "-t", help="Resource type to monitor"
    ),
    interval: int = typer.Option(
        5, "--interval", "-i", help="Update interval (seconds)"
    ),
    duration: int = typer.Option(
        60, "--duration", "-d", help="Total monitoring duration (seconds)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save monitoring data to file"
    ),
) -> None:
    """Monitor system resources continuously."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would monitor {resource} resources")
        typer.echo(f"[DRY-RUN] Interval: {interval}s, Duration: {duration}s")
        if output:
            typer.echo(f"[DRY-RUN] Output: {output}")
        return

    _monitor_resources_continuously(resource, interval, duration, output)


def _monitor_resources_continuously(
    resource: str, interval: int, duration: int, output: Path | None
) -> None:
    """Continuously monitor system resources."""
    start_time = time.time()
    monitoring_data = []

    typer.echo(
        f"Monitoring {resource} resources for {duration}s (interval: {interval}s)"
    )
    typer.echo("Press Ctrl+C to stop early")
    typer.echo("-" * 60)

    try:
        while time.time() - start_time < duration:
            timestamp = datetime.now()
            data_point = {
                "timestamp": timestamp.isoformat(),
                "cpu_percent": 35.2 + (time.time() % 10) * 2,  # Mock fluctuation
                "memory_percent": 44.5 + (time.time() % 8) * 1.5,
                "disk_io_read": int(time.time() * 1000) % 50000,
                "disk_io_write": int(time.time() * 800) % 30000,
                "network_bytes_sent": int(time.time() * 1200) % 100000,
                "network_bytes_recv": int(time.time() * 1500) % 150000,
            }

            monitoring_data.append(data_point)

            # Display current values
            if resource == "all" or resource == "cpu":
                typer.echo(f"CPU: {data_point['cpu_percent']:.1f}%", nl=False)
            if resource == "all" or resource == "memory":
                typer.echo(f" | Memory: {data_point['memory_percent']:.1f}%", nl=False)
            if resource == "all" or resource == "disk":
                typer.echo(
                    f" | Disk I/O: R:{data_point['disk_io_read']} W:{data_point['disk_io_write']}",
                    nl=False,
                )

            typer.echo(f" | {timestamp.strftime('%H:%M:%S')}")

            time.sleep(interval)

    except KeyboardInterrupt:
        typer.echo("\nMonitoring stopped by user")

    # Save data if output specified
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(monitoring_data, f, indent=2)
        typer.echo(f"Monitoring data saved to: {output}")


@app.command("jobs")
def monitor_jobs(
    status_filter: str | None = typer.Option(
        None, "--status", help="Filter by job status"
    ),
    user_filter: str | None = typer.Option(None, "--user", help="Filter by username"),
) -> None:
    """Monitor active jobs and pipelines."""
    ctx = get_context()
    cfg = load_config(ctx.config_path)

    if ctx.dry_run:
        typer.echo("[DRY-RUN] Would display job status")
        return

    jobs = _get_active_jobs(cfg, status_filter, user_filter)

    if not jobs:
        typer.echo("No active jobs found")
        return

    typer.echo("Active Jobs:")
    typer.echo("-" * 80)
    typer.echo(
        f"{'Job ID':<12} {'Status':<10} {'Type':<15} {'Progress':<10} {'Runtime':<12} {'User':<10}"
    )
    typer.echo("-" * 80)

    for job in jobs:
        typer.echo(
            f"{job['id']:<12} {job['status']:<10} {job['type']:<15} "
            f"{job['progress']:<10} {job['runtime']:<12} {job['user']:<10}"
        )


def _get_active_jobs(cfg, status_filter: str | None, user_filter: str | None) -> list:
    """Get mock active jobs."""
    all_jobs = [
        {
            "id": "job_001",
            "status": "running",
            "type": "assembly",
            "progress": "75%",
            "runtime": "02:45:30",
            "user": "analyst1",
            "started": "2025-08-26 10:30:00",
        },
        {
            "id": "job_002",
            "status": "queued",
            "type": "taxonomy",
            "progress": "0%",
            "runtime": "00:00:00",
            "user": "analyst2",
            "started": "2025-08-26 12:15:00",
        },
        {
            "id": "job_003",
            "status": "completed",
            "type": "annotation",
            "progress": "100%",
            "runtime": "01:23:45",
            "user": "analyst1",
            "started": "2025-08-26 09:00:00",
        },
    ]

    # Apply filters
    filtered_jobs = all_jobs
    if status_filter:
        filtered_jobs = [j for j in filtered_jobs if j["status"] == status_filter]
    if user_filter:
        filtered_jobs = [j for j in filtered_jobs if j["user"] == user_filter]

    return filtered_jobs


@app.command("logs")
def tail_logs(
    log_file: Path = typer.Option(..., "--file", "-f", help="Log file to monitor"),
    lines: int = typer.Option(
        50, "--lines", "-n", help="Number of lines to show initially"
    ),
    level: LogLevel = typer.Option("info", "--level", "-l", help="Minimum log level"),
    follow: bool = typer.Option(
        True, "--follow", help="Follow log file for new entries"
    ),
) -> None:
    """Monitor log files in real-time."""
    ctx = get_context()

    if ctx.dry_run:
        typer.echo(f"[DRY-RUN] Would tail log file: {log_file}")
        typer.echo(f"[DRY-RUN] Lines: {lines}, Level: {level}, Follow: {follow}")
        return

    _tail_log_file(log_file, lines, level, follow)


def _tail_log_file(log_file: Path, lines: int, level: str, follow: bool) -> None:
    """Simulate log tailing functionality."""
    # Generate mock log entries
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    typer.echo(f"Tailing log file: {log_file}")
    typer.echo(f"Showing last {lines} lines (level: {level.upper()})")
    typer.echo("-" * 80)

    # Show initial lines
    for i in range(lines):
        timestamp = (datetime.now() - timedelta(minutes=lines - i)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        log_level = log_levels[i % len(log_levels)]
        message = f"Sample log message {i + 1} from application component"

        if follow and i == lines - 1:
            typer.echo(f"[{timestamp}] {log_level:8} | {message}")
            break
        else:
            typer.echo(f"[{timestamp}] {log_level:8} | {message}")

    # Follow mode simulation
    if follow:
        typer.echo("\nFollowing log file... (Press Ctrl+C to stop)")
        try:
            counter = 0
            while True:
                time.sleep(2)  # Simulate log updates
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_level = log_levels[counter % len(log_levels)]
                message = f"New log entry {counter + 1} - system activity detected"

                typer.echo(f"[{timestamp}] {log_level:8} | {message}")
                counter += 1

        except KeyboardInterrupt:
            typer.echo("\nLog monitoring stopped")


@app.command("alerts")
def check_alerts(
    severity: str = typer.Option(
        "warning", "--severity", help="Minimum alert severity"
    ),
    last_hours: int = typer.Option(
        24, "--hours", help="Check alerts from last N hours"
    ),
) -> None:
    """Check system alerts and warnings."""
    alerts = _get_system_alerts(severity, last_hours)

    if not alerts:
        typer.echo("No alerts found")
        return

    typer.echo(f"System Alerts (last {last_hours}h, severity >= {severity}):")
    typer.echo("-" * 70)

    for alert in alerts:
        typer.echo(
            f"[{alert['timestamp']}] "
            f"{alert['severity'].upper():8} | "
            f"{alert['component']:15} | "
            f"{alert['message']}"
        )


def _get_system_alerts(severity: str, hours: int) -> list:
    """Generate mock system alerts."""
    now = datetime.now()
    alerts = [
        {
            "timestamp": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "severity": "warning",
            "component": "disk_monitor",
            "message": "Disk usage on /data exceeds 80% (currently 85%)",
        },
        {
            "timestamp": (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "severity": "error",
            "component": "job_scheduler",
            "message": "Job job_001 exceeded memory limit, killed",
        },
        {
            "timestamp": (now - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "severity": "info",
            "component": "database",
            "message": "Database kraken2 updated successfully",
        },
    ]

    # Filter by severity and time
    severity_levels = {"info": 0, "warning": 1, "error": 2, "critical": 3}
    min_severity = severity_levels.get(severity, 1)

    filtered_alerts = [
        alert
        for alert in alerts
        if severity_levels.get(alert["severity"], 0) >= min_severity
    ]

    return filtered_alerts

"""core/analysis.py.

AnalysisEngine: Taxonomic classification using Kraken2/Minimap2 or other methods.
Business logic for CLI delegation.
"""

from pathlib import Path
import subprocess  # nosec
import logging
from typing import Optional


class AnalysisResult:
    """A class to hold the results of an analysis."""

    def __init__(self, output_path: Path, summary: dict = None):
        """Initialize the AnalysisResult.

        Args:
            output_path: The path to the output file.
            summary: A summary of the analysis.
        """
        self.output_path = output_path
        self.summary = summary or {}


class AnalysisEngine:
    """A class to run metagenomic analyses."""

    def __init__(
        self,
        method: str = "kraken2",
        database: Optional[str] = None,
        threads: int = 4,
        output_dir: Path = Path("./output"),
    ):
        """Initialize the AnalysisEngine.

        Args:
            method: The analysis method to use.
            database: The path to the reference database.
            threads: The number of threads to use.
            output_dir: The path to the output directory.
        """
        self.method = method.lower()
        self.database = database
        self.threads = threads
        self.output_dir = output_dir
        self.logger = logging.getLogger("AnalysisEngine")

    def run_analysis(self, input_file: Path, force: bool = False) -> AnalysisResult:
        self.logger.info(f"Starting analysis: {input_file}, method={self.method}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        output_path = self.output_dir / f"{input_file.stem}.{self.method}.report.txt"
        if output_path.exists() and not force:
            self.logger.warning(
                f"Output {output_path} already exists and force is False."
            )
            return AnalysisResult(output_path, summary={"cached": True})

        if self.method == "kraken2":
            self._run_kraken2(input_file, output_path)
        elif self.method == "minimap2":
            self._run_minimap2(input_file, output_path)
        else:
            raise ValueError(f"Unknown analysis method: {self.method}")
        self.logger.info(f"Analysis completed. Output: {output_path}")
        return AnalysisResult(output_path)

    def _run_kraken2(self, input_file: Path, output_path: Path):
        # Replace with your local Kraken2 executable if needed
        cmd = [
            "kraken2",
            "--db",
            self.database or "path_to_default_db",
            "--threads",
            str(self.threads),
            "--report",
            str(output_path),
            str(input_file),
        ]
        self.logger.debug(f"Running Kraken2: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)  # nosec

    def _run_minimap2(self, input_file: Path, output_path: Path):
        # Replace with your Minimap2 workflow as needed
        cmd = ["minimap2", "-a", self.database or "path_to_default_db", str(input_file)]
        with open(output_path, "w") as f:
            self.logger.debug(f"Running Minimap2: {' '.join(cmd)} > {output_path}")
            subprocess.run(cmd, stdout=f, check=True)  # nosec

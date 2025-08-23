"""core/reporting.py.

Report generation for analysis results.
Business logic for CLI delegation.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any


class ReportGenerator:
    """Generate reports from analysis results."""

    def __init__(self, format_type: str = "html", include_plots: bool = True):
        """Initialize the ReportGenerator.

        Args:
            format_type: The format of the report.
            include_plots: Whether to include plots in the report.
        """
        self.format_type = format_type.lower()
        self.include_plots = include_plots
        self.logger = logging.getLogger("ReportGenerator")

    def generate_report(self, input_dir: Path, output_file: Path) -> Path:
        """Generate analysis report from results directory.

        Args:
            input_dir: Directory containing analysis results
            output_file: Output file path for the report

        Returns:
            Path: Path to generated report file
        """
        self.logger.info(f"Generating {self.format_type} report from {input_dir}")

        # Collect analysis results
        results_data = self._collect_results(input_dir)

        # Generate report based on format
        if self.format_type == "html":
            self._generate_html_report(results_data, output_file)
        elif self.format_type == "json":
            self._generate_json_report(results_data, output_file)
        elif self.format_type == "pdf":
            self._generate_pdf_report(results_data, output_file)
        else:
            raise ValueError(f"Unsupported report format: {self.format_type}")

        self.logger.info(f"Report generated: {output_file}")
        return output_file

    def _collect_results(self, input_dir: Path) -> Dict[str, Any]:
        """Collect and parse analysis results from directory."""
        results: Dict[str, Any] = {
            "input_directory": str(input_dir),
            "analysis_files": [],
            "summary": {},
        }

        # Find analysis result files
        for file_path in input_dir.glob("*.report.txt"):
            results["analysis_files"].append(str(file_path))

        # Add basic statistics
        results["summary"]["total_files"] = len(results["analysis_files"])

        return results

    def _generate_html_report(self, data: Dict[str, Any], output_file: Path):
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MetagenomicsOS Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>🧬 MetagenomicsOS Analysis Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Input Directory:</strong> {data['input_directory']}</p>
                <p><strong>Total Analysis Files:</strong> {data['summary']['total_files']}</p>
            </div>
            <h2>Analysis Files</h2>
            <ul>
        """

        for file_path in data["analysis_files"]:
            html_content += f"<li>{Path(file_path).name}</li>\n"

        html_content += """
            </ul>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html_content)

    def _generate_json_report(self, data: Dict[str, Any], output_file: Path):
        """Generate JSON report."""
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_pdf_report(self, data: Dict[str, Any], output_file: Path):
        """Generate PDF report (placeholder - requires additional libraries)."""
        # This would require libraries like reportlab or weasyprint
        # For now, generate a text version
        text_output = output_file.with_suffix(".txt")
        with open(text_output, "w") as f:
            f.write("MetagenomicsOS Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Input Directory: {data['input_directory']}\n")
            f.write(f"Total Files: {data['summary']['total_files']}\n\n")
            f.write("Analysis Files:\n")
            for file_path in data["analysis_files"]:
                f.write(f"- {Path(file_path).name}\n")

        self.logger.info(
            f"PDF generation not implemented, created text report: {text_output}"
        )
        return text_output

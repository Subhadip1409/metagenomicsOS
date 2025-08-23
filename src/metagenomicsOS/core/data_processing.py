"""core/data_processing.py.

DataProcessor: Quality control, filtering, and adapter trimming for sequencing data.
Business logic for CLI delegation. Uses BioPython for FASTQ handling.
"""

from pathlib import Path
from typing import Optional
from Bio import SeqIO
import logging


class ProcessingResult:
    """A class to hold the results of data processing."""

    def __init__(self, total_reads: int, kept_reads: int, output_path: Optional[Path]):
        """Initialize the ProcessingResult.

        Args:
            total_reads: The total number of reads processed.
            kept_reads: The number of reads kept after filtering.
            output_path: The path to the output file.
        """
        self.total_reads = total_reads
        self.kept_reads = kept_reads
        self.keep_percentage = (
            (kept_reads / total_reads * 100) if total_reads > 0 else 0.0
        )
        self.output_path = output_path


class DataProcessor:
    """A class to process sequencing data."""

    def __init__(
        self,
        quality_threshold: int = 20,
        min_length: int = 50,
        trim_adapters: bool = False,
    ):
        """Initialize the DataProcessor.

        Args:
            quality_threshold: The minimum quality score.
            min_length: The minimum read length.
            trim_adapters: Whether to trim adapter sequences.
        """
        self.quality_threshold = quality_threshold
        self.min_length = min_length
        self.trim_adapters = trim_adapters
        self.logger = logging.getLogger("DataProcessor")

    def process_file(self, input_file: Path, output_file: Optional[Path]):
        self.logger.info(f"Processing file: {input_file}")
        total = 0
        kept = 0
        good_reads = []

        for record in SeqIO.parse(str(input_file), "fastq"):
            total += 1
            if self._passes_filters(record):
                good_reads.append(record)
                kept += 1
        if output_file:
            SeqIO.write(good_reads, str(output_file), "fastq")
            self.logger.info(f"Written {kept} reads to {output_file}")
        return ProcessingResult(
            total_reads=total, kept_reads=kept, output_path=output_file
        )

    def _passes_filters(self, record):
        # Filter by minimum quality threshold
        qualities = record.letter_annotations["phred_quality"]
        if min(qualities) < self.quality_threshold:
            return False
        if len(record) < self.min_length:
            return False
        # Adapter trimming: placeholder (could use cutadapt or biopython function)
        # Could implement advanced trimming logic here
        return True

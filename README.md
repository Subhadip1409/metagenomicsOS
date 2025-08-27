# MetagenomicsOS

![PyPI Version](https://img.shields.io/pypi/v/metagenomicsos.svg)
![Build Status](https://img.shields.io/github/actions/workflow/status/Subhadip1409/metagenomicsOS/ci.yml)
![Code Coverage](https://img.shields.io/codecov/c/github/Subhadip1409/metagenomicsOS.svg)
![License](https://img.shields.io/github/license/Subhadip1409/metagenomicsOS.svg)

> An integrated, AI-driven platform for end-to-end metagenomics analysis.

MetagenomicsOS is a comprehensive, scalable, and extensible platform designed to streamline metagenomics analysis from raw sequencing data to actionable insights. It integrates state-of-the-art bioinformatics tools, AI-powered analytics, and robust workflow orchestration to support research in complex microbial communities.

## Key Features

- **AI-Powered Analysis:** Leverages machine learning models for advanced tasks like taxonomic classification, functional prediction, and anomaly detection.
- **End-to-End Workflow Orchestration:** Uses Snakemake to manage complex, multi-step analysis pipelines, ensuring reproducibility and scalability.
- **Real-time Processing:** Ingests and processes data from streaming sources for continuous analysis and monitoring.
- **Multi-Cloud & HPC Support:** Provides flexible deployment options across local, HPC (DRMAA), and cloud (AWS, GCP, Azure) environments.
- **Extensible Plugin Architecture:** Easily extend the platform's capabilities by developing custom plugins for new tools and analyses.
- **Interactive Reporting:** Generates detailed, interactive reports and visualizations using Plotly, Matplotlib, and Bokeh.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Subhadip1409/metagenomicsOS.git
    cd metagenomicsOS
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the package:**

    ```bash
    # For a standard installation
    pip install .

    # For developers, install with all optional dependencies
    pip install -e ".[all]"
    ```

## Quick Start

The primary entry point is the `metagenomicsos` (or `mgos`) command-line interface.

```bash
# Display help and available commands
metagenomicsos --help
```

# MetagenomicsOS CLI Documentation

A comprehensive command-line interface for metagenomics analysis workflows, from raw data to publication-ready results.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Global Options](#global-options)
- [Command Reference](#command-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [Advanced Features](#advanced-features)

## Overview

MetagenomicsOS CLI provides a unified interface for:

- **Quality Control**: Read filtering, trimming, and quality assessment
- **Assembly**: Genome assembly with multiple assemblers
- **Taxonomic Analysis**: Classification and profiling
- **Functional Annotation**: Gene prediction and functional analysis
- **Binning**: Contig binning and MAG creation
- **Resistance Analysis**: ARG/AMR detection and profiling
- **Visualization**: Interactive plots and comprehensive reports
- **System Management**: Monitoring, optimization, and validation
- **Advanced Features**: Real-time streaming, ML models, and plugins

## Installation

```bash
# Install dependencies
pip install "typer[all]" pydantic PyYAML

# Clone and setup
git clone <repository>
cd metagenomicsOS
pip install -e .

# Verify installation
python -m metagenomicsOS.cli.main --help
```

## Quick Start

```bash
# Initialize configuration
python -m metagenomicsOS.cli.main config init

# Run end-to-end pipeline
python -m metagenomicsOS.cli.main pipeline run \
  --input reads.fastq.gz \
  --outdir results/

# Check status
python -m metagenomicsOS.cli.main run list
```

## Global Options

All commands support these global options:

- `--config, -c`: Path to configuration file (default: `./config.yaml`)
- `--verbose, -v`: Increase verbosity (`-v` = INFO, `-vv` = DEBUG)
- `--dry-run`: Show execution plan without running
- `--help`: Show command help

## Command Reference

### Core Workflow Orchestration

#### `run` - Core Workflows Orchestration

Execute and manage metagenomics workflows with full lifecycle support.

**Subcommands:**

##### `run execute`

```bash
python -m metagenomicsOS.cli.main run execute \
  --workflow pipeline \
  --input reads.fastq.gz \
  --workdir /path/to/work \
  --threads 16 \
  --memory 32GB
```

**Options:**

- `--workflow, -w`: Workflow type (`pipeline`, `qc`, `assembly`, `taxonomy`, `annotation`, `binning`)
- `--input, -i`: Input reads or data file
- `--workdir`: Working directory for all outputs
- `--threads, -t`: Number of CPU threads
- `--memory, -m`: Memory limit (e.g., '16GB')
- `--resume`: Resume from checkpoint ID
- `--samples`: Sample sheet for batch processing
- `--config`: Workflow configuration file
- `--databases`: Database directory override

##### `run list`

```bash
# List all runs
python -m metagenomicsOS.cli.main run list

# Filter by status
python -m metagenomicsOS.cli.main run list --status completed

# Filter by workflow type
python -m metagenomicsOS.cli.main run list --workflow assembly
```

##### `run status`

```bash
# Basic status
python -m metagenomicsOS.cli.main run status <run-id>

# Detailed information
python -m metagenomicsOS.cli.main run status <run-id> --verbose
```

##### Other `run` commands:

- `run cancel <run-id>`: Cancel running workflow
- `run rerun <run-id>`: Rerun with same/updated parameters
- `run logs <run-id>`: Show workflow logs
- `run clean`: Clean old workflow runs
- `run plan`: Show execution plan without running
- `run validate`: Validate workflow setup

#### `pipeline` - End-to-End Pipeline

Quick access to full metagenomics pipeline.

```bash
python -m metagenomicsOS.cli.main pipeline run \
  --input reads.fastq.gz \
  --outdir results/ \
  --threads 16
```

### Core Analysis Workflows

#### `qc` - Quality Control

Read quality assessment, filtering, and trimming.

**Subcommands:**

##### `qc run`

```bash
python -m metagenomicsOS.cli.main qc run \
  --input reads.fastq.gz \
  --output qc_results/ \
  --action trim \
  --quality 20 \
  --min-length 50
```

**Options:**

- `--action, -a`: QC action (`trim`, `filter`, `stats`, `report`)
- `--quality, -q`: Quality score threshold
- `--min-length`: Minimum read length
- `--adapters`: Adapter sequences file

##### `qc list`

Show recent QC runs and their status.

#### `assembly` - Genome Assembly

Assemble reads into contigs using various assemblers.

**Subcommands:**

##### `assembly run`

```bash
python -m metagenomicsOS.cli.main assembly run \
  --input trimmed_reads.fastq.gz \
  --output assembly_out/ \
  --assembler spades \
  --mode meta \
  --threads 16
```

**Options:**

- `--assembler, -a`: Assembly tool (`spades`, `megahit`, `metaflye`, `unicycler`)
- `--mode`: Assembly mode (`meta`, `isolate`)
- `--kmers`: K-mer sizes (comma-separated)
- `--memory`: Memory limit
- `--careful`: Enable careful mode

##### `assembly stats`

```bash
python -m metagenomicsOS.cli.main assembly stats \
  --assembly contigs.fasta \
  --output stats.json
```

##### `assembly validate`

Validate assembly quality and completeness.

#### `taxonomy` - Taxonomic Profiling

Classify sequences and generate taxonomic profiles.

**Subcommands:**

##### `taxonomy classify`

```bash
python -m metagenomicsOS.cli.main taxonomy classify \
  --input reads.fastq.gz \
  --output taxonomy_out/ \
  --classifier kraken2 \
  --database standard \
  --confidence 0.1
```

**Options:**

- `--classifier, -c`: Classification tool (`kraken2`, `metaphlan`, `centrifuge`, `kaiju`)
- `--database, -d`: Reference database
- `--confidence`: Confidence threshold
- `--format, -f`: Output format (`kraken`, `biom`, `json`, `tsv`)

##### `taxonomy profile`

```bash
python -m metagenomicsOS.cli.main taxonomy profile \
  --input classification_results.txt \
  --output profile.tsv \
  --level species
```

##### `taxonomy compare`

Compare taxonomic profiles across samples.

### Annotation Workflows

#### `annotation` - Functional Annotation

Annotate sequences against functional databases.

**Subcommands:**

##### `annotation run`

```bash
python -m metagenomicsOS.cli.main annotation run \
  --input contigs.fasta \
  --output annotation_out/ \
  --database cog \
  --evalue 1e-5
```

**Options:**

- `--database, -d`: Annotation database (`cog`, `kegg`, `pfam`, `tigrfam`, `go`, `ec`)
- `--evalue, -e`: E-value threshold
- `--coverage`: Coverage threshold
- `--format, -f`: Output format (`tsv`, `gff`, `json`, `xml`)

##### `annotation summarize`

Generate annotation summary statistics.

#### `binning` - Contig Binning

Group contigs into metagenome-assembled genomes (MAGs).

**Subcommands:**

##### `binning run`

```bash
python -m metagenomicsOS.cli.main binning run \
  --contigs assembly.fasta \
  --abundance coverage.tsv \
  --output binning_out/ \
  --binner metabat2
```

**Options:**

- `--binner, -b`: Binning algorithm (`metabat2`, `maxbin2`, `concoct`, `vamb`, `semibin`)
- `--min-contig`: Minimum contig length
- `--sensitivity`: Binning sensitivity

##### `binning quality`

```bash
python -m metagenomicsOS.cli.main binning quality \
  --bins bins/ \
  --output quality_report.json \
  --completeness 70 \
  --contamination 10
```

##### `binning refine`

Refine genome bins based on quality metrics.

#### `genes` - Gene-Level Annotation

Predict and annotate genes in genomic sequences.

**Subcommands:**

##### `genes predict`

```bash
python -m metagenomicsOS.cli.main genes predict \
  --input contigs.fasta \
  --output genes_out/ \
  --predictor prodigal \
  --mode meta
```

**Options:**

- `--predictor, -p`: Gene prediction tool (`prodigal`, `augustus`, `genemark`)
- `--mode`: Prediction mode
- `--min-gene-len`: Minimum gene length
- `--format, -f`: Output format (`fasta`, `gff`, `genbank`)

##### `genes annotate`

```bash
python -m metagenomicsOS.cli.main genes annotate \
  --genes predicted_genes.fasta \
  --output gene_annotations/ \
  --database uniref90
```

##### `genes summary`

Generate gene annotation summary.

#### `resistance` - ARG/AMR Analysis

Detect and analyze antibiotic resistance genes.

**Subcommands:**

##### `resistance detect`

```bash
python -m metagenomicsOS.cli.main resistance detect \
  --input sequences.fasta \
  --output arg_results/ \
  --database card \
  --identity 90 \
  --coverage 80
```

**Options:**

- `--database, -d`: ARG database (`card`, `argannot`, `resfinder`, `ardb`)
- `--identity`: Identity threshold (%)
- `--coverage`: Coverage threshold (%)

##### `resistance profile`

```bash
python -m metagenomicsOS.cli.main resistance profile \
  --input arg_results.tsv \
  --output resistance_profile.json \
  --class beta_lactam
```

##### `resistance compare`

Compare resistance profiles across samples.

##### `resistance report`

Generate HTML report for ARG analysis.

### Visualization and Reporting

#### `visualize` - Visualization and Reports

Create interactive plots and visualizations.

**Subcommands:**

##### `visualize barplot`

```bash
python -m metagenomicsOS.cli.main visualize barplot \
  --input taxonomy_profile.tsv \
  --output barplot.html \
  --type taxonomy \
  --top 20
```

**Options:**

- `--type, -t`: Data type (`taxonomy`, `functional`, `abundance`, `args`)
- `--top`: Show top N features
- `--width`: Plot width
- `--height`: Plot height

##### `visualize heatmap`

```bash
python -m metagenomicsOS.cli.main visualize heatmap \
  --input abundance_matrix.tsv \
  --output heatmap.html \
  --cluster-rows \
  --cluster-cols
```

##### `visualize network`

Create co-occurrence network visualization.

#### `report` - Report Generation

Generate comprehensive analysis reports.

**Subcommands:**

##### `report generate`

```bash
python -m metagenomicsOS.cli.main report generate \
  --input results_directory/ \
  --output final_report.html \
  --format html \
  --title "Metagenomics Analysis Report"
```

**Options:**

- `--format, -f`: Report format (`html`, `pdf`, `markdown`)
- `--plots`: Include visualizations
- `--template`: Custom template file

##### `report summary`

```bash
python -m metagenomicsOS.cli.main report summary \
  --input results/ \
  --format json
```

#### `compare` - Comparative Analysis

Compare samples and perform group-wise analysis.

**Subcommands:**

##### `compare samples`

```bash
python -m metagenomicsOS.cli.main compare samples \
  --profile sample1.tsv sample2.tsv sample3.tsv \
  --output comparison/ \
  --type taxonomy \
  --metric bray_curtis
```

##### `compare groups`

```bash
python -m metagenomicsOS.cli.main compare groups \
  --metadata sample_metadata.tsv \
  --profiles profiles_dir/ \
  --output group_comparison/ \
  --column treatment
```

##### `compare diversity`

```bash
python -m metagenomicsOS.cli.main compare diversity \
  --profile sample*.tsv \
  --output diversity.json \
  --metric shannon simpson chao1
```

##### `compare resistome`

Compare resistance gene profiles across samples.

### Database Management

#### `database` - Database Management

Download, validate, and manage reference databases.

**Subcommands:**

##### `database download`

```bash
python -m metagenomicsOS.cli.main database download \
  --type kraken2 \
  --output databases/ \
  --version latest
```

**Options:**

- `--type, -t`: Database type (`kraken2`, `card`, `cog`, `kegg`, `pfam`, `uniref`)
- `--version`: Specific version
- `--force`: Force redownload

##### `database list`

```bash
python -m metagenomicsOS.cli.main database list
```

##### `database validate`

```bash
python -m metagenomicsOS.cli.main database validate \
  --type kraken2 \
  --dir databases/
```

##### Other database commands:

- `database update`: Update existing database
- `database info`: Show database information

### System Management

#### `monitor` - System Monitoring

Monitor system resources and job status.

**Subcommands:**

##### `monitor status`

```bash
python -m metagenomicsOS.cli.main monitor status
```

##### `monitor resources`

```bash
python -m metagenomicsOS.cli.main monitor resources \
  --type cpu \
  --interval 5 \
  --duration 300 \
  --output monitoring.json
```

##### `monitor jobs`

```bash
python -m metagenomicsOS.cli.main monitor jobs \
  --status running \
  --user analyst1
```

##### `monitor logs`

```bash
python -m metagenomicsOS.cli.main monitor logs \
  --file /path/to/logfile \
  --lines 100 \
  --follow
```

##### `monitor alerts`

Check system alerts and warnings.

#### `validate` - Validation Utilities

Validate input data and workflow integrity.

**Subcommands:**

##### `validate input`

```bash
python -m metagenomicsOS.cli.main validate input \
  --input data.fastq \
  --type fastq \
  --level standard \
  --output validation_report.json
```

**Options:**

- `--type, -t`: Data type (`fastq`, `fasta`, `gff`, `tsv`, `json`, `yaml`)
- `--level, -l`: Validation level (`basic`, `standard`, `strict`)

##### `validate workflow`

```bash
python -m metagenomicsOS.cli.main validate workflow \
  --workflow pipeline_config.json \
  --schema workflow_schema.json \
  --deps
```

##### `validate schema`

Validate data against schema definitions.

##### `validate batch`

```bash
python -m metagenomicsOS.cli.main validate batch \
  --input data_directory/ \
  --pattern "*.fastq" \
  --type fastq \
  --output batch_validation.json
```

#### `optimize` - Performance Optimization

Auto-tune resources and optimize performance.

**Subcommands:**

##### `optimize suggest`

```bash
python -m metagenomicsOS.cli.main optimize suggest \
  --workflow assembly \
  --input-size 10GB \
  --target threads \
  --constraint max_memory=32GB
```

##### `optimize profile`

```bash
python -m metagenomicsOS.cli.main optimize profile \
  --result workflow_results/ \
  --output performance_profile.json
```

##### `optimize tune`

Auto-tune workflow parameters using iterative optimization.

##### `optimize resources`

Optimize resource allocation based on usage patterns.

#### `benchmark` - Performance Benchmarking

Test and compare workflow performance.

**Subcommands:**

##### `benchmark workflow`

```bash
python -m metagenomicsOS.cli.main benchmark workflow \
  --workflow assembly \
  --size medium \
  --iterations 3 \
  --output benchmark.json
```

##### `benchmark scaling`

```bash
python -m metagenomicsOS.cli.main benchmark scaling \
  --workflow qc \
  --threads "1,2,4,8,16" \
  --output scaling_results.json
```

##### `benchmark memory`

Test performance under different memory constraints.

##### `benchmark compare`

Compare multiple benchmark results.

### Advanced Features

#### `stream` - Real-Time Streaming

Process data streams in real-time.

**Subcommands:**

##### `stream classify`

```bash
python -m metagenomicsOS.cli.main stream classify \
  --source data_stream.fastq \
  --output streaming_results/ \
  --classifier kraken2 \
  --batch-size 1000
```

##### `stream dashboard`

```bash
python -m metagenomicsOS.cli.main stream dashboard \
  --data dashboard_data.json \
  --port 8080 \
  --refresh 5
```

##### `stream monitor`

Monitor streaming classification performance.

##### `stream export`

Export streaming results to standard formats.

#### `ml` - Machine Learning

Train and use ML models for metagenomics.

**Subcommands:**

##### `ml train`

```bash
python -m metagenomicsOS.cli.main ml train \
  --type binning \
  --data training_data.csv \
  --output models/binning_v1/ \
  --framework sklearn \
  --epochs 100
```

**Options:**

- `--type, -t`: Model type (`binning`, `args`, `taxonomy`, `quality`)
- `--framework, -f`: ML framework (`sklearn`, `tensorflow`, `pytorch`, `xgboost`)

##### `ml predict`

```bash
python -m metagenomicsOS.cli.main ml predict \
  --model models/binning_v1/ \
  --input contigs.fasta \
  --output predictions.json \
  --threshold 0.8
```

##### `ml evaluate`

```bash
python -m metagenomicsOS.cli.main ml evaluate \
  --model models/binning_v1/ \
  --test test_data.csv \
  --output evaluation.json \
  --metric accuracy precision recall
```

##### `ml drift`

Monitor model drift and data distribution changes.

##### `ml list`

List available trained models.

#### `plugin` - Plugin System

Extend functionality with custom plugins.

**Subcommands:**

##### `plugin list`

```bash
python -m metagenomicsOS.cli.main plugin list \
  --installed \
  --type command
```

##### `plugin develop`

```bash
python -m metagenomicsOS.cli.main plugin develop \
  --name my_custom_tool \
  --type command \
  --output plugins/ \
  --author "Your Name"
```

##### `plugin install`

```bash
python -m metagenomicsOS.cli.main plugin install \
  --path plugins/my_custom_tool/ \
  --plugins-dir ~/.metagenomics/plugins/
```

##### `plugin test`

```bash
python -m metagenomicsOS.cli.main plugin test \
  --path plugins/my_custom_tool/ \
  --type full
```

##### Other plugin commands:

- `plugin uninstall`: Remove installed plugin
- `plugin info`: Show plugin information

### Configuration Management

#### `config` - Configuration Management

Manage CLI configuration settings.

**Subcommands:**

##### `config show`

```bash
# Show as YAML (default)
python -m metagenomicsOS.cli.main config show

# Show as JSON
python -m metagenomicsOS.cli.main config show --format json
```

##### `config edit`

```bash
# Interactive editing
python -m metagenomicsOS.cli.main config edit

# Inline updates
python -m metagenomicsOS.cli.main config edit \
  --set threads=16 \
  --set project_name=my_project
```

##### `config validate`

```bash
python -m metagenomicsOS.cli.main config validate --strict
```

##### Other config commands:

- `config path`: Show config file path
- `config init`: Create default configuration

## Configuration

The CLI uses a YAML configuration file for default settings:

```yaml
# config.yaml
project_name: "my_metagenomics_project"
data_dir: "./data"
database_dir: "./databases"
threads: 8
```

### Configuration Options

- `project_name`: Default project name
- `data_dir`: Directory for analysis data
- `database_dir`: Directory for reference databases
- `threads`: Default number of CPU threads

## Examples

### Basic Quality Control and Assembly

```bash
# 1. Quality control
python -m metagenomicsOS.cli.main qc run \
  --input raw_reads.fastq.gz \
  --output qc_results/ \
  --action trim \
  --quality 25

# 2. Assembly
python -m metagenomicsOS.cli.main assembly run \
  --input qc_results/trimmed_reads.fastq.gz \
  --output assembly_results/ \
  --assembler spades \
  --threads 16
```

### Full Pipeline with Custom Configuration

```bash
# 1. Setup configuration
python -m metagenomicsOS.cli.main config edit \
  --set threads=32 \
  --set database_dir=/shared/databases

# 2. Run full pipeline
python -m metagenomicsOS.cli.main run execute \
  --workflow pipeline \
  --input sample_reads.fastq.gz \
  --workdir pipeline_results/ \
  --threads 32 \
  --memory 64GB

# 3. Generate report
python -m metagenomicsOS.cli.main report generate \
  --input pipeline_results/ \
  --output final_report.html
```

### Batch Processing Multiple Samples

```bash
# Create sample sheet
cat > samples.tsv << EOF
sample_id	input_file	condition
sample1	data/sample1.fastq.gz	control
sample2	data/sample2.fastq.gz	treatment
sample3	data/sample3.fastq.gz	treatment
EOF

# Run batch analysis
python -m metagenomicsOS.cli.main run execute \
  --workflow pipeline \
  --samples samples.tsv \
  --workdir batch_results/ \
  --threads 16
```

### Comparative Analysis

```bash
# 1. Generate profiles for each sample
for sample in sample1 sample2 sample3; do
  python -m metagenomicsOS.cli.main taxonomy classify \
    --input ${sample}_reads.fastq.gz \
    --output ${sample}_taxonomy/
done

# 2. Compare samples
python -m metagenomicsOS.cli.main compare samples \
  --profile *_taxonomy/profile.tsv \
  --output comparative_analysis/ \
  --type taxonomy

# 3. Group comparison
python -m metagenomicsOS.cli.main compare groups \
  --metadata sample_metadata.tsv \
  --profiles taxonomy_profiles/ \
  --output group_analysis/ \
  --column condition
```

### Performance Optimization

```bash
# 1. Get optimization suggestions
python -m metagenomicsOS.cli.main optimize suggest \
  --workflow assembly \
  --input-size 50GB \
  --target all

# 2. Benchmark different configurations
python -m metagenomicsOS.cli.main benchmark scaling \
  --workflow assembly \
  --threads "8,16,32" \
  --output scaling_test.json

# 3. Monitor resource usage
python -m metagenomicsOS.cli.main monitor resources \
  --type all \
  --duration 3600 \
  --output resource_usage.json
```

## Advanced Features

### Resume Capability

All workflows support resuming from checkpoints:

```bash
# If workflow is interrupted
python -m metagenomicsOS.cli.main run execute \
  --workflow pipeline \
  --resume 20250826-143022-a1b2c3d4 \
  --workdir pipeline_results/
```

### Real-time Streaming

Process data as it arrives:

```bash
# Start streaming classification
python -m metagenomicsOS.cli.main stream classify \
  --source tcp://sequencer:9999 \
  --output streaming_results/ \
  --classifier kraken2

# Launch live dashboard
python -m metagenomicsOS.cli.main stream dashboard \
  --data streaming_results/dashboard_data.json \
  --port 8080
```

### Machine Learning Integration

Train custom models for your data:

```bash
# Train binning model
python -m metagenomicsOS.cli.main ml train \
  --type binning \
  --data training_features.csv \
  --output models/custom_binner/ \
  --epochs 200

# Use for prediction
python -m metagenomicsOS.cli.main ml predict \
  --model models/custom_binner/ \
  --input new_contigs.fasta \
  --output bin_predictions.json
```

### Plugin Development

Extend functionality with custom plugins:

```bash
# Create plugin scaffold
python -m metagenomicsOS.cli.main plugin develop \
  --name contamination_filter \
  --type command \
  --output plugins/

# Test plugin
python -m metagenomicsOS.cli.main plugin test \
  --path plugins/contamination_filter/ \
  --type full

# Install plugin
python -m metagenomicsOS.cli.main plugin install \
  --path plugins/contamination_filter/
```

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure proper PYTHONPATH or use module execution
2. **Database errors**: Download required databases with `database download`
3. **Memory issues**: Use `optimize suggest` for resource recommendations
4. **Validation failures**: Check input formats with `validate input`

### Getting Help

- Use `--help` with any command for detailed options
- Check logs with `run logs <run-id>`
- Monitor system resources with `monitor status`
- Validate setup with `validate workflow`

### Performance Tips

- Use `benchmark workflow` to test performance
- Optimize resources with `optimize suggest`
- Monitor jobs with `monitor jobs`
- Use `--dry-run` to preview execution plans

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) to learn how you can get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

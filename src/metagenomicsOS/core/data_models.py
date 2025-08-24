# src/metagenomicsOS/core/data_models.py

from __future__ import annotations

from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator, SecretStr


# -------------------------
# Project / metadata
# -------------------------
class ProjectConfig(BaseModel):
    name: str = Field(default="metagenomicsos", description="Project identifier")
    version: str = Field(default="0.1.0", description="Application version")
    config_version: int = Field(
        default=1, ge=1, description="Config schema version for migrations"
    )

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("project name cannot be empty")
        return v


# -------------------------
# Paths
# -------------------------
class PathsConfig(BaseModel):
    data_dir: str = Field(default="data", description="Root input data directory")
    raw_reads_dir: str = Field(default="data/raw", description="Raw reads location")
    intermediate_dir: str = Field(
        default="data/intermediate", description="Intermediate outputs"
    )
    results_dir: str = Field(default="results", description="Final results directory")
    logs_dir: str = Field(default="logs", description="Logs directory")
    tmp_dir: str = Field(default="tmp", description="Temporary working directory")


# -------------------------
# Logging
# -------------------------
class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    to_file: bool = Field(default=True)
    file_path: Optional[str] = Field(default="logs/app.log")
    json_format: bool = Field(default=False, description="Enable JSON log formatting")


# -------------------------
# Cache
# -------------------------
class CacheConfig(BaseModel):
    enabled: bool = Field(default=True)
    dir: str = Field(default=".cache")
    max_size_gb: int = Field(default=50, ge=1)
    ttl_days: int = Field(default=30, ge=1)


# -------------------------
# Retry policy for jobs/requests
# -------------------------
class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=2.0, gt=0)
    backoff_multiplier: float = Field(default=2.0, gt=1.0)
    max_backoff_seconds: float = Field(default=60.0, gt=0)


# -------------------------
# Containers / environment
# -------------------------
class ContainerConfig(BaseModel):
    runtime: Literal["none", "docker", "singularity"] = Field(default="none")
    image: Optional[str] = Field(default=None, description="Container image reference")
    # Optional environment manager for tools (useful if not using containers everywhere)
    use_conda_env: bool = Field(default=False)
    conda_env_name: Optional[str] = Field(default=None)


# -------------------------
# Databases
# -------------------------
class DatabaseEntry(BaseModel):
    name: str
    version: Optional[str] = None
    path: Optional[str] = Field(default=None, description="Filesystem path to database")


class DatabasesConfig(BaseModel):
    kraken2: DatabaseEntry = Field(
        default_factory=lambda: DatabaseEntry(name="kraken2")
    )
    eggnog: DatabaseEntry = Field(default_factory=lambda: DatabaseEntry(name="eggnog"))


# -------------------------
# Execution backends and resources
# -------------------------
class LocalSettings(BaseModel):
    parallel_jobs: int = Field(default=1, ge=1)


class SlurmSettings(BaseModel):
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time_limit: str = Field(default="01:00:00", description="HH:MM:SS")
    extra_args: Dict[str, str] = Field(default_factory=dict)


class AWSBatchSettings(BaseModel):
    region: Optional[str] = None
    job_queue: Optional[str] = None
    job_definition: Optional[str] = None
    retry_strategy_attempts: int = Field(default=1, ge=1)
    parameters: Dict[str, str] = Field(default_factory=dict)
    # Optional credentials (prefer env/role in production)
    access_key_id: Optional[SecretStr] = None
    secret_access_key: Optional[SecretStr] = None


class GCPSettings(BaseModel):
    project: Optional[str] = None
    location: Optional[str] = None
    machine_type: Optional[str] = None
    extra: Dict[str, str] = Field(default_factory=dict)


class ExecutionProfile(BaseModel):
    backend: Literal["local", "slurm", "aws", "gcp"] = Field(default="local")
    max_threads: int = Field(default=4, ge=1)
    max_memory_gb: int = Field(default=8, ge=1)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    local: Optional[LocalSettings] = Field(default_factory=LocalSettings)
    slurm: Optional[SlurmSettings] = None
    aws: Optional[AWSBatchSettings] = None
    gcp: Optional[GCPSettings] = None
    extra: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _backend_requires_settings(self) -> "ExecutionProfile":
        if self.backend == "slurm" and self.slurm is None:
            raise ValueError("SLURM backend selected but 'slurm' settings not provided")
        if self.backend == "aws" and self.aws is None:
            raise ValueError("AWS backend selected but 'aws' settings not provided")
        if self.backend == "gcp" and self.gcp is None:
            raise ValueError("GCP backend selected but 'gcp' settings not provided")
        return self


# -------------------------
# Samples
# -------------------------
class SamplesConfig(BaseModel):
    samples_file: Optional[str] = Field(
        default=None, description="Path to samples manifest (CSV/TSV/YAML)"
    )
    schema_name: Optional[str] = Field(
        default="samples.schema.yaml", description="Schema to validate samples manifest"
    )
    id_column: Optional[str] = Field(default="sample_id")
    # Optional column mapping hooks for later schema validation
    columns: Dict[str, str] = Field(default_factory=dict)


# -------------------------
# Workflow parameters per module
# -------------------------
class QCParams(BaseModel):
    fastqc_enabled: bool = Field(default=True)
    trimming_enabled: bool = Field(default=True)
    trimmomatic_adapters: Optional[str] = None
    minlen: int = Field(default=50, ge=20)
    leading: int = Field(default=3, ge=0)
    trailing: int = Field(default=3, ge=0)
    slidingwindow_size: int = Field(default=4, ge=1)
    slidingwindow_quality: int = Field(default=20, ge=1)
    threads: int = Field(default=2, ge=1)


class TaxonomyParams(BaseModel):
    tool: Literal["kraken2"] = Field(default="kraken2")
    confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    report_format: Literal["kreport", "kraken"] = Field(default="kreport")
    use_bracken: bool = Field(default=False)
    bracken_db_path: Optional[str] = None
    threads: int = Field(default=4, ge=1)


class FunctionParams(BaseModel):
    tool: Literal["eggnog-mapper"] = Field(default="eggnog-mapper")
    database: Optional[str] = None
    diamond_sensitivity: Literal["fast", "sensitive", "more-sensitive"] = Field(
        default="sensitive"
    )
    go_annotations: bool = Field(default=True)
    kegg_annotations: bool = Field(default=True)
    threads: int = Field(default=4, ge=1)


class AssemblyParams(BaseModel):
    tool: Literal["megahit", "spades"] = Field(default="megahit")
    k_list: Optional[List[int]] = None
    min_contig_len: int = Field(default=1000, ge=200)
    threads: int = Field(default=8, ge=1)


class BinningParams(BaseModel):
    tool: Literal["metabat2"] = Field(default="metabat2")
    min_contig_len: int = Field(default=1500, ge=1000)
    threads: int = Field(default=8, ge=1)


# -------------------------
# Workflow toggles + params
# -------------------------
class WorkflowToggles(BaseModel):
    quality_control: bool = Field(
        default=True, description="Enable QC (FastQC/Trimming)"
    )
    taxonomy: bool = Field(default=True, description="Enable taxonomic classification")
    function: bool = Field(default=False, description="Enable functional annotation")
    assembly: bool = Field(default=False, description="Enable assembly")
    binning: bool = Field(default=False, description="Enable binning")


class WorkflowsConfig(BaseModel):
    toggles: WorkflowToggles = Field(default_factory=WorkflowToggles)
    qc: QCParams = Field(default_factory=QCParams)
    taxonomy: TaxonomyParams = Field(default_factory=TaxonomyParams)
    function: FunctionParams = Field(default_factory=FunctionParams)
    assembly: AssemblyParams = Field(default_factory=AssemblyParams)
    binning: BinningParams = Field(default_factory=BinningParams)


# -------------------------
# Top-level Config with cross-field validation
# -------------------------
class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    databases: DatabasesConfig = Field(default_factory=DatabasesConfig)
    profile: ExecutionProfile = Field(default_factory=ExecutionProfile)
    samples: SamplesConfig = Field(default_factory=SamplesConfig)
    workflows: WorkflowsConfig = Field(default_factory=WorkflowsConfig)

    @model_validator(mode="after")
    def _require_databases_when_enabled(self) -> "Config":
        # If taxonomy is enabled, require Kraken2 path
        if self.workflows.toggles.taxonomy:
            if (
                self.databases.kraken2.path is None
                or not str(self.databases.kraken2.path).strip()
            ):
                raise ValueError(
                    "taxonomy enabled but databases.kraken2.path is not set"
                )
        # If functional annotation is enabled, require EggNOG path
        if self.workflows.toggles.function:
            if (
                self.databases.eggnog.path is None
                or not str(self.databases.eggnog.path).strip()
            ):
                raise ValueError(
                    "function enabled but databases.eggnog.path is not set"
                )
        # If QC trimming is enabled, suggest adapters path if trimming is True (soft check)
        # We won't raise here, but you can enforce if desired.
        return self

    # Convenience helpers for downstream code
    def enabled_workflows(self) -> List[str]:
        enabled = []
        t = self.workflows.toggles
        if t.quality_control:
            enabled.append("quality_control")
        if t.taxonomy:
            enabled.append("taxonomy")
        if t.function:
            enabled.append("function")
        if t.assembly:
            enabled.append("assembly")
        if t.binning:
            enabled.append("binning")
        return enabled

    def active_backend(self) -> str:
        return self.profile.backend

    def backend_settings(
        self,
    ) -> Union[LocalSettings, SlurmSettings, AWSBatchSettings, GCPSettings, None]:
        if self.profile.backend == "local":
            return self.profile.local
        if self.profile.backend == "slurm":
            return self.profile.slurm
        if self.profile.backend == "aws":
            return self.profile.aws
        if self.profile.backend == "gcp":
            return self.profile.gcp
        return None

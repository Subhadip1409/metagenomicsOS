# tests/unit/test_core_data_models.py

import pytest
from metagenomicsOS.core.data_models import Config, ExecutionProfile, SlurmSettings


def test_defaults_basic():
    cfg = Config()
    # Defaults
    assert cfg.project.name == "metagenomicsos"
    assert "quality_control" in cfg.enabled_workflows()
    assert "taxonomy" in cfg.enabled_workflows()
    assert cfg.profile.backend == "local"
    assert cfg.profile.max_threads >= 1
    assert cfg.paths.results_dir == "results"


def test_backend_requires_settings():
    # SLURM without settings -> should fail
    with pytest.raises(ValueError):
        _ = Config(profile=ExecutionProfile(backend="slurm", slurm=None))

    # SLURM with settings -> ok
    cfg = Config(profile=ExecutionProfile(backend="slurm", slurm=SlurmSettings()))
    assert cfg.active_backend() == "slurm"


def test_require_kraken2_path_when_taxonomy_enabled():
    with pytest.raises(ValueError):
        # taxonomy toggle is True by default, but kraken2.path is None
        _ = Config()


def test_disable_taxonomy_avoids_db_requirement():
    cfg = Config(workflows={"toggles": {"taxonomy": False}})
    assert "taxonomy" not in cfg.enabled_workflows()

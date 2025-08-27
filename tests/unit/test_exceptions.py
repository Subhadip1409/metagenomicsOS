# Create tests/unit/test_exceptions.py
"""Tests for custom exception classes."""

import pytest
from metagenomicsOS.core.exceptions import (
    MetagenomicsOSError,
    ConfigurationError,
    DatabaseError,
    ValidationError,
    raise_config_error,
)


def test_base_exception():
    """Test the base MetagenomicsOSError class."""
    error = MetagenomicsOSError(
        "Test error", details={"key": "value"}, error_code="TEST_001"
    )

    assert str(error).startswith("[TEST_001] MetagenomicsOS Error: Test error")
    assert "key=value" in str(error)
    assert error.message == "Test error"
    assert error.details == {"key": "value"}
    assert error.error_code == "TEST_001"


def test_configuration_error():
    """Test ConfigurationError with config file context."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise_config_error("Invalid config format", config_file="config.yaml")

    error = exc_info.value
    assert "Invalid config format" in str(error)
    assert error.details["config_file"] == "config.yaml"
    assert error.error_code == "CONFIG_ERROR"


def test_database_error():
    """Test DatabaseError with database context."""
    error = DatabaseError(
        "Database download failed",
        database_name="kraken2",
        details={"url": "https://example.com/db"},
    )

    assert "Database download failed" in str(error)
    assert error.details["database"] == "kraken2"
    assert error.details["url"] == "https://example.com/db"


def test_validation_error_with_field_info():
    """Test ValidationError with field validation context."""
    error = ValidationError(
        "Invalid field type",
        field="sample_count",
        expected_type="int",
        actual_value="not_a_number",
    )

    assert "Invalid field type" in str(error)
    assert error.details["field"] == "sample_count"
    assert error.details["expected_type"] == "int"
    assert error.details["actual_value"] == "not_a_number"

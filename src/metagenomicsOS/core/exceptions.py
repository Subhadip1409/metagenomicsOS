import sys
import os
from typing import Optional, Dict, Any

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

"""Custom exception classes for MetagenomicsOS.

This module defines a hierarchy of custom exceptions to enable precise
error handling throughout the application. Each exception class is designed
for specific error scenarios with meaningful error messages.
"""


class MetagenomicsOSError(Exception):
    """Base exception class for all MetagenomicsOS-specific errors.

    All custom exceptions in MetagenomicsOS should inherit from this class.
    This allows for catching all application-specific errors with a single
    except clause if needed.

    Args:
        message: Human-readable error description
        details: Optional dictionary with additional error context
        error_code: Optional error code for programmatic handling
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize the MetagenomicsOSError."""
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a formatted error message."""
        result = f"MetagenomicsOS Error: {self.message}"
        if self.error_code:
            result = f"[{self.error_code}] {result}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            result += f" (Details: {details_str})"
        return result


class ConfigurationError(MetagenomicsOSError):
    """Raised when configuration-related errors occur.

    Examples:
    - Invalid configuration file format
    - Missing required configuration parameters
    - Configuration validation failures
    """

    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        """Initialize the ConfigurationError."""
        details = kwargs.get("details", {})
        if config_file:
            details["config_file"] = config_file
        super().__init__(message, details, kwargs.get("error_code", "CONFIG_ERROR"))


class DatabaseError(MetagenomicsOSError):
    """Raised when database-related errors occur.

    Examples:
    - Database download failures
    - Database corruption or integrity issues
    - Version mismatch problems
    - Missing database files
    """

    def __init__(self, message: str, database_name: Optional[str] = None, **kwargs):
        """Initialize the DatabaseError."""
        details = kwargs.get("details", {})
        if database_name:
            details["database"] = database_name
        super().__init__(message, details, kwargs.get("error_code", "DB_ERROR"))


class WorkflowError(MetagenomicsOSError):
    """Raised when workflow execution errors occur.

    Examples:
    - Snakemake rule failures
    - Invalid workflow specifications
    - Pipeline execution errors
    - Dependency resolution failures
    """

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the WorkflowError."""
        details = kwargs.get("details", {})
        if workflow_name:
            details["workflow"] = workflow_name
        if step:
            details["step"] = step
        super().__init__(message, details, kwargs.get("error_code", "WORKFLOW_ERROR"))


class ValidationError(MetagenomicsOSError):
    """Raised when data validation errors occur.

    Examples:
    - Invalid input file formats
    - Schema validation failures
    - Data type mismatches
    - Required field missing
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize the ValidationError."""
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)
        super().__init__(message, details, kwargs.get("error_code", "VALIDATION_ERROR"))


class ResourceError(MetagenomicsOSError):
    """Raised when system resource errors occur.

    Examples:
    - Insufficient memory
    - Disk space shortage
    - CPU availability issues
    - Network connectivity problems
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required: Optional[str] = None,
        available: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the ResourceError."""
        details = kwargs.get("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if required:
            details["required"] = required
        if available:
            details["available"] = available
        super().__init__(message, details, kwargs.get("error_code", "RESOURCE_ERROR"))


class ExternalToolError(MetagenomicsOSError):
    """Raised when external bioinformatics tools fail.

    Examples:
    - FastQC execution failures
    - Kraken2 classification errors
    - Assembly tool crashes
    - Tool not found or not installed
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        exit_code: Optional[int] = None,
        command: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the ExternalToolError."""
        details = kwargs.get("details", {})
        if tool_name:
            details["tool"] = tool_name
        if exit_code is not None:
            details["exit_code"] = exit_code
        if command:
            details["command"] = command
        super().__init__(message, details, kwargs.get("error_code", "TOOL_ERROR"))


class FileOperationError(MetagenomicsOSError):
    """Raised when file operation errors occur.

    Examples:
    - File not found
    - Permission denied
    - Corrupt file formats
    - I/O errors during read/write
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the FileOperationError."""
        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details, kwargs.get("error_code", "FILE_ERROR"))


class NetworkError(MetagenomicsOSError):
    """Raised when network-related errors occur.

    Examples:
    - API connection failures
    - Database download timeouts
    - Authentication failures
    - Rate limiting issues
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the NetworkError."""
        details = kwargs.get("details", {})
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details, kwargs.get("error_code", "NETWORK_ERROR"))


# Convenience functions for common error scenarios
def raise_config_error(message: str, config_file: Optional[str] = None) -> None:
    """Helper function to raise configuration errors with consistent formatting."""
    raise ConfigurationError(message, config_file=config_file)


def raise_database_error(message: str, database_name: Optional[str] = None) -> None:
    """Helper function to raise database errors with consistent formatting."""
    raise DatabaseError(message, database_name=database_name)


def raise_workflow_error(
    message: str, workflow_name: Optional[str] = None, step: Optional[str] = None
) -> None:
    """Helper function to raise workflow errors with consistent formatting."""
    raise WorkflowError(message, workflow_name=workflow_name, step=step)


def raise_validation_error(
    message: str,
    field: Optional[str] = None,
    expected_type: Optional[str] = None,
    actual_value: Optional[Any] = None,
) -> None:
    """Helper function to raise validation errors with consistent formatting."""
    raise ValidationError(
        message, field=field, expected_type=expected_type, actual_value=actual_value
    )

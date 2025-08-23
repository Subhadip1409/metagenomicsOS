"""Custom Exception Types for MetagenomicsOS.

This module defines a hierarchy of custom exception classes to provide
more specific and informative error handling throughout the application.
"""


class MetagenomicsOSError(Exception):
    """Base class for all application-specific errors."""

    pass


class ConfigurationError(MetagenomicsOSError):
    """Raised for errors related to application configuration."""

    pass


class DataProcessingError(MetagenomicsOSError):
    """Raised for errors during data processing and QC."""

    pass


class AnalysisError(MetagenomicsOSError):
    """Raised for errors during analysis, e.g., with external tools."""

    pass


class DatabaseError(MetagenomicsOSError):
    """Raised for errors related to reference databases."""

    pass


class WorkflowError(MetagenomicsOSError):
    """Raised for errors within the workflow execution engine."""

    pass


class APIError(MetagenomicsOSError):
    """Raised for errors related to the web API."""

    pass


class ValidationError(MetagenomicsOSError):
    """Raised for data validation failures."""

    pass

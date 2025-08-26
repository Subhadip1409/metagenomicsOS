# cli/core/logging.py
from __future__ import annotations
import logging
import sys

_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        logger = record.name
        msg = super().format(record)
        return f"[{level.lower()}] {logger}: {msg}"


def configure_logging(verbosity: int = 0) -> None:
    level = _LEVELS.get(verbosity, logging.DEBUG if verbosity >= 2 else logging.WARNING)
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplication in tests and repeated runs.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_ConsoleFormatter("%(message)s"))
    root.addHandler(handler)

    # Quiet noisy third-party loggers by default; commands can override as needed.
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))

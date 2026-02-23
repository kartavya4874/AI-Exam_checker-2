"""
Centralized logging setup for the Exam Checker system.

Provides a ``get_logger`` factory that returns module-specific loggers
all writing to both a shared rotating log file and the console.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from exam_checker.config import LOG_LEVEL, TEMP_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_DIR = TEMP_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "exam_checker.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
_BACKUP_COUNT = 5
_FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Shared formatter & handlers (created once)
# ---------------------------------------------------------------------------
_formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

_file_handler = RotatingFileHandler(
    str(_LOG_FILE), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
)
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.DEBUG)  # file always captures everything

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Optional: a list of handlers that the GUI log viewer can tap into
_gui_handlers: list[logging.Handler] = []


def add_gui_handler(handler: logging.Handler) -> None:
    """Register an additional handler (used by the GUI log viewer)."""
    handler.setFormatter(_formatter)
    _gui_handlers.append(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger identified by *name*.

    All loggers share the same file and console handlers so output is
    consistent across the application.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)
        for h in _gui_handlers:
            logger.addHandler(h)
        logger.propagate = False
    return logger

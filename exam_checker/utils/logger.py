"""
Centralized logging setup for the Hybrid AI Exam Checking System.

Provides file and console handlers with configurable log levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If None, logs only to console.
    """
    global _initialized
    if _initialized:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # ── File handler (optional) ──
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in [
        "urllib3",
        "httpx",
        "httpcore",
        "openai",
        "transformers",
        "PIL",
        "matplotlib",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)

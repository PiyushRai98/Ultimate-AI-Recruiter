"""Logging configuration using Loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_dir: Path | None = None,
    level: str = "INFO",
    rotation: str = "10 MB",
) -> None:
    """Configure loguru logger for the application.

    Args:
        log_dir: Directory for log files. Defaults to project_root/logs.
        level: Minimum log level.
        rotation: Log file rotation size.
    """
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "recruiter_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=rotation,
        retention="7 days",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )

"""Timing utilities for performance monitoring."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from loguru import logger


@contextmanager
def timer(operation: str) -> Generator[None, None, None]:
    """Context manager to time operations.

    Args:
        operation: Description of the operation being timed.

    Yields:
        None. Logs elapsed time on exit.
    """
    start = time.perf_counter()
    logger.info(f"Starting: {operation}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"Completed: {operation} in {elapsed:.2f}s")

"""Efficient streaming data loader for candidate JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import orjson
from loguru import logger

from src.models.candidate import Candidate
from src.utils.timing import timer


class CandidateLoader:
    """Streaming loader for candidate JSONL data.

    Designed for memory efficiency: streams candidates one at a time
    or in configurable batches to handle 100K+ candidate files.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the loader.

        Args:
            file_path: Path to candidates.jsonl file.
        """
        self._file_path = file_path
        if not self._file_path.exists():
            raise FileNotFoundError(f"Candidates file not found: {self._file_path}")

    def stream(self) -> Generator[Candidate, None, None]:
        """Stream candidates one at a time for memory efficiency.

        Yields:
            Candidate objects parsed from JSONL.
        """
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                    yield Candidate.from_dict(data)
                except (orjson.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Line {line_num}: Failed to parse candidate: {e}")

    def load_all(self) -> list[Candidate]:
        """Load all candidates into memory.

        Returns:
            List of all valid Candidate objects.
        """
        with timer("Loading all candidates"):
            candidates = list(self.stream())
            logger.info(f"Loaded {len(candidates)} candidates")
            return candidates

    def load_raw_dicts(self) -> list[dict]:
        """Load raw dictionaries without parsing into Candidate objects.

        Faster when only specific fields are needed.

        Returns:
            List of raw candidate dictionaries.
        """
        results = []
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(orjson.loads(line))
        return results

    def stream_raw(self) -> Generator[dict, None, None]:
        """Stream raw dictionaries for maximum performance.

        Yields:
            Raw candidate dictionaries.
        """
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield orjson.loads(line)

    def count(self) -> int:
        """Count total candidates without loading into memory.

        Returns:
            Number of candidates in the file.
        """
        count = 0
        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

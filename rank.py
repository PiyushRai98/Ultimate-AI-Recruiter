#!/usr/bin/env python3
"""Main ranking script - produces submission.csv from candidates.jsonl.

Usage:
    python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv
    python rank.py  (uses defaults from config)
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

from src.config.loader import get_config
from src.pipeline.ranking_pipeline import RankingPipeline
from src.utils.logger import setup_logger

app = typer.Typer(help="AI Recruiter Ranking System")


@app.command()
def main(
    candidates: Path = typer.Option(
        None,
        "--candidates",
        "-c",
        help="Path to candidates.jsonl file",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Output submission CSV path",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable embedding cache (recompute everything)",
    ),
) -> None:
    """Run the full ranking pipeline to produce submission.csv."""
    setup_logger()

    config = get_config()

    # Resolve paths
    candidates_path = candidates or config.get_path("data.candidates_file")
    output_path = out or Path("submission.csv")

    if not candidates_path.exists():
        logger.error(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    logger.info(f"Candidates: {candidates_path}")
    logger.info(f"Output: {output_path}")

    # Run pipeline
    pipeline = RankingPipeline()
    result_path = pipeline.run(
        candidates_path=candidates_path,
        output_path=output_path,
        use_cache=not no_cache,
    )

    logger.info(f"Done! Submission at: {result_path}")


if __name__ == "__main__":
    app()

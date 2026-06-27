#!/usr/bin/env python3
"""Preprocessing script - parses JD and validates data.

Usage:
    python preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from src.config.loader import get_config
from src.preprocessing.data_loader import CandidateLoader
from src.preprocessing.jd_parser import JDParser
from src.utils.logger import setup_logger
from src.utils.timing import timer

app = typer.Typer(help="Data preprocessing utilities")


@app.command()
def main() -> None:
    """Run data preprocessing and validation."""
    setup_logger()
    config = get_config()

    # Parse JD
    with timer("Parsing Job Description"):
        jd_path = config.get_path("data.job_description_file")
        parser = JDParser()
        jd = parser.parse_from_file(jd_path)

        cache_path = Path("cache/jd_parsed.yaml")
        parser.save_parsed(jd, cache_path)
        logger.info(f"JD parsed: {jd.parsed.title} at {jd.parsed.company}")

    # Validate candidate data
    with timer("Validating candidate data"):
        candidates_path = config.get_path("data.candidates_file")
        loader = CandidateLoader(candidates_path)
        count = loader.count()
        logger.info(f"Total candidates: {count}")

        # Load first 10 to verify parsing
        sample = []
        for i, c in enumerate(loader.stream()):
            sample.append(c)
            if i >= 9:
                break

        logger.info(f"Sample validation: {len(sample)} candidates parsed successfully")
        for c in sample[:3]:
            logger.info(
                f"  {c.candidate_id}: {c.current_title} | "
                f"{c.years_of_experience} yrs | "
                f"{len(c.skills)} skills"
            )

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    app()

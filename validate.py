#!/usr/bin/env python3
"""Validate submission CSV against competition rules.

Usage:
    python validate.py submission.csv
    python validate.py --check-candidates  # Also validates candidate IDs exist
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config.loader import get_config
from src.utils.logger import setup_logger

app = typer.Typer(help="Submission validation")
console = Console()

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


@app.command()
def main(
    csv_path: Path = typer.Argument(
        ..., help="Path to submission CSV file"
    ),
    check_candidates: bool = typer.Option(
        False,
        "--check-candidates",
        help="Verify candidate IDs exist in candidates.jsonl",
    ),
) -> None:
    """Validate a submission CSV file."""
    setup_logger(level="WARNING")

    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        sys.exit(1)

    errors: list[str] = []
    warnings: list[str] = []

    # Read CSV
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            if header != REQUIRED_HEADER:
                errors.append(
                    f"Header mismatch. Expected: {REQUIRED_HEADER}, Got: {header}"
                )

            rows = [row for row in reader if any(cell.strip() for cell in row)]
    except Exception as e:
        console.print(f"[red]Cannot read file: {e}[/red]")
        sys.exit(1)

    # Check row count
    if len(rows) != 100:
        errors.append(f"Expected 100 data rows, found {len(rows)}")

    # Validate each row
    seen_ids = set()
    seen_ranks = set()
    prev_score = float("inf")

    for i, row in enumerate(rows):
        row_num = i + 2

        if len(row) != 4:
            errors.append(f"Row {row_num}: Expected 4 columns, got {len(row)}")
            continue

        cid, rank_s, score_s, reasoning = row

        # Candidate ID
        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: Invalid candidate_id '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: Duplicate candidate_id '{cid}'")
        seen_ids.add(cid)

        # Rank
        try:
            rank = int(rank_s)
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: Rank {rank} out of range [1,100]")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: Duplicate rank {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: Invalid rank '{rank_s}'")

        # Score
        try:
            score = float(score_s)
            if score > prev_score + 0.0001:
                errors.append(
                    f"Row {row_num}: Score {score} > previous {prev_score} "
                    f"(must be non-increasing)"
                )
            prev_score = score
        except ValueError:
            errors.append(f"Row {row_num}: Invalid score '{score_s}'")

        # Reasoning
        if not reasoning.strip():
            warnings.append(f"Row {row_num}: Empty reasoning")

    # Check all ranks present
    missing_ranks = set(range(1, 101)) - seen_ranks
    if missing_ranks:
        errors.append(f"Missing ranks: {sorted(missing_ranks)}")

    # Optional: check candidate IDs exist
    if check_candidates and not errors:
        config = get_config()
        candidates_path = config.get_path("data.candidates_file")
        if candidates_path.exists():
            valid_ids = set()
            import orjson
            with open(candidates_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = orjson.loads(line)
                        valid_ids.add(data["candidate_id"])

            invalid_ids = seen_ids - valid_ids
            if invalid_ids:
                errors.append(
                    f"Candidate IDs not found in dataset: {sorted(invalid_ids)}"
                )

    # Report
    if errors:
        console.print(f"\n[red bold]VALIDATION FAILED ({len(errors)} errors):[/red bold]")
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
        if warnings:
            console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
            for w in warnings:
                console.print(f"  [yellow]⚠[/yellow] {w}")
        sys.exit(1)
    else:
        console.print("[green bold]✓ Submission is valid![/green bold]")
        if warnings:
            console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
            for w in warnings[:5]:
                console.print(f"  [yellow]⚠[/yellow] {w}")

        # Show summary table
        table = Table(title="Submission Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total rows", str(len(rows)))
        table.add_row("Unique candidates", str(len(seen_ids)))
        table.add_row("Score range", f"{min(float(r[2]) for r in rows):.4f} - {max(float(r[2]) for r in rows):.4f}")
        table.add_row("Has reasoning", str(sum(1 for r in rows if r[3].strip())))
        console.print(table)


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""Build embeddings and FAISS index for candidate retrieval.

This is the pre-computation step that can run outside the 5-minute window.

Usage:
    python build_index.py
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from src.config.loader import get_config
from src.preprocessing.data_loader import CandidateLoader
from src.retrieval.embedding_builder import EmbeddingBuilder
from src.utils.logger import setup_logger
from src.utils.timing import timer

app = typer.Typer(help="Build embeddings and search index")


@app.command()
def main(
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild even if cache exists"),
) -> None:
    """Build candidate embeddings and FAISS index."""
    setup_logger()
    config = get_config()

    embeddings_path = Path("cache/embeddings.npy")
    index_path = Path("artifacts/index/faiss_index.bin")

    if embeddings_path.exists() and not force:
        logger.info("Embeddings cache exists. Use --force to rebuild.")
        return

    # Load candidates
    with timer("Loading candidates"):
        candidates_path = config.get_path("data.candidates_file")
        loader = CandidateLoader(candidates_path)
        candidates = loader.load_all()

    # Generate text representations
    with timer("Generating text representations"):
        texts = [c.get_text_representation() for c in candidates]
        logger.info(f"Generated {len(texts)} text representations")

    # Build embeddings
    builder = EmbeddingBuilder()
    embeddings = builder.encode_texts(texts)

    # Save embeddings
    builder.save_embeddings(embeddings, embeddings_path)

    # Build and save FAISS index
    builder.build_index(embeddings)
    builder.save_index(index_path)

    logger.info("Index building complete!")
    logger.info(f"  Embeddings: {embeddings_path} ({embeddings.shape})")
    logger.info(f"  Index: {index_path}")


if __name__ == "__main__":
    app()

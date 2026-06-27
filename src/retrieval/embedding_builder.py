"""Embedding generation and FAISS index building."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config.loader import get_config
from src.models.candidate import Candidate
from src.utils.timing import timer


class EmbeddingBuilder:
    """Builds and manages dense embeddings for semantic search.

    Uses Sentence Transformers to generate embeddings and FAISS
    for efficient approximate nearest neighbor search.
    """

    def __init__(self) -> None:
        """Initialize embedding builder with configuration."""
        config = get_config()
        embed_config = config.get("ranking", "embedding", {})
        self._model_name = embed_config.get("model_name", "all-MiniLM-L6-v2")
        self._batch_size = embed_config.get("batch_size", 256)
        self._max_seq_length = embed_config.get("max_seq_length", 256)
        self._normalize = embed_config.get("normalize", True)
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._dimension: int = 0

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self._model is None:
            with timer(f"Loading embedding model: {self._model_name}"):
                self._model = SentenceTransformer(self._model_name)
                self._model.max_seq_length = self._max_seq_length
                # Use new API if available, fallback to old
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Model loaded: dim={self._dimension}, "
                    f"max_seq={self._max_seq_length}"
                )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
        """
        self._load_model()
        assert self._model is not None

        with timer(f"Encoding {len(texts)} texts"):
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=True,
                normalize_embeddings=self._normalize,
            )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query text.

        Args:
            query: Query text string.

        Returns:
            NumPy array of shape (1, embedding_dim).
        """
        self._load_model()
        assert self._model is not None

        embedding = self._model.encode(
            [query],
            normalize_embeddings=self._normalize,
        )
        return embedding

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build a FAISS index from embeddings.

        Uses Inner Product index (equivalent to cosine similarity
        when embeddings are normalized).

        Args:
            embeddings: Array of shape (n, dim).

        Returns:
            FAISS index ready for search.
        """
        dim = embeddings.shape[1]
        logger.info(f"Building FAISS index: {embeddings.shape[0]} vectors, dim={dim}")

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        self._index = index
        self._dimension = dim
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index

    def search(
        self, query_embedding: np.ndarray, top_k: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index for nearest neighbors.

        Args:
            query_embedding: Query vector of shape (1, dim).
            top_k: Number of results to return.

        Returns:
            Tuple of (scores, indices) arrays.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index first.")

        scores, indices = self._index.search(
            query_embedding.astype(np.float32), top_k
        )
        return scores[0], indices[0]

    def save_embeddings(self, embeddings: np.ndarray, path: Path) -> None:
        """Save embeddings to disk.

        Args:
            embeddings: Embeddings array.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        logger.info(f"Embeddings saved: {path} ({embeddings.shape})")

    def load_embeddings(self, path: Path) -> np.ndarray:
        """Load embeddings from disk.

        Args:
            path: Path to saved embeddings file.

        Returns:
            Loaded embeddings array.
        """
        embeddings = np.load(path)
        logger.info(f"Embeddings loaded: {path} ({embeddings.shape})")
        return embeddings

    def save_index(self, path: Path) -> None:
        """Save FAISS index to disk.

        Args:
            path: Output file path.
        """
        if self._index is None:
            raise RuntimeError("No index to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.info(f"FAISS index saved: {path}")

    def load_index(self, path: Path) -> None:
        """Load FAISS index from disk.

        Args:
            path: Path to saved index file.
        """
        self._index = faiss.read_index(str(path))
        logger.info(f"FAISS index loaded: {path} ({self._index.ntotal} vectors)")

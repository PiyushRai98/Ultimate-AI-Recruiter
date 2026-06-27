"""Hybrid retrieval combining BM25 sparse + dense embeddings with RRF fusion.

Upgrade from simple weighted combination to Reciprocal Rank Fusion (RRF),
which is more robust to score distribution differences between retrievers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.config.loader import get_config
from src.models.candidate import Candidate
from src.retrieval.embedding_builder import EmbeddingBuilder
from src.utils.timing import timer


class BM25:
    """Optimized BM25 implementation for sparse retrieval.

    Uses inverted index for efficient scoring instead of
    iterating over all documents per term.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: np.ndarray = np.array([])
        self._avg_dl: float = 0.0
        self._n_docs: int = 0
        # Inverted index: term -> list of (doc_id, tf)
        self._inverted_index: dict[str, list[tuple[int, int]]] = {}

    def fit(self, documents: list[list[str]]) -> None:
        """Fit BM25 on tokenized documents using inverted index.

        Args:
            documents: List of tokenized documents (list of word lists).
        """
        self._n_docs = len(documents)
        doc_lens = []

        for doc_id, doc in enumerate(documents):
            doc_lens.append(len(doc))
            tf = Counter(doc)
            for term, count in tf.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = []
                    self._doc_freqs[term] = 0
                self._inverted_index[term].append((doc_id, count))
                self._doc_freqs[term] += 1

        self._doc_lens = np.array(doc_lens, dtype=np.float32)
        self._avg_dl = float(np.mean(self._doc_lens)) if self._n_docs > 0 else 1.0
        logger.info(
            f"BM25 fitted: {self._n_docs} docs, "
            f"{len(self._inverted_index)} unique terms"
        )

    def score_query(
        self, query_tokens: list[str], top_k: int = 2000
    ) -> list[tuple[int, float]]:
        """Score documents against a query using inverted index.

        Args:
            query_tokens: Tokenized query.
            top_k: Number of top results to return.

        Returns:
            List of (doc_index, score) tuples sorted by score descending.
        """
        scores = np.zeros(self._n_docs, dtype=np.float32)

        for term in set(query_tokens):
            if term not in self._inverted_index:
                continue

            df = self._doc_freqs[term]
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self._inverted_index[term]:
                dl = self._doc_lens[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self._avg_dl
                )
                scores[doc_id] += idf * (numerator / denominator)

        # Get top-k using partial sort
        if self._n_docs <= top_k:
            top_indices = np.argsort(scores)[::-1]
            top_indices = top_indices[scores[top_indices] > 0]
        else:
            # Find candidates with non-zero scores
            nonzero_mask = scores > 0
            nonzero_count = int(np.sum(nonzero_mask))

            if nonzero_count == 0:
                return []

            if nonzero_count <= top_k:
                nonzero_indices = np.where(nonzero_mask)[0]
                top_indices = nonzero_indices[
                    np.argsort(scores[nonzero_indices])[::-1]
                ]
            else:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


class HybridRetriever:
    """Hybrid retrieval with Reciprocal Rank Fusion (RRF).

    Combines BM25 sparse retrieval with dense embedding retrieval
    using RRF for robust score combination. Retrieves a larger
    candidate pool (top-2000) for improved recall.

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each retriever i
    where k is a constant (typically 60) that controls the impact of rank.
    """

    def __init__(self) -> None:
        """Initialize hybrid retriever with configuration."""
        config = get_config()
        retrieval_config = config.get("ranking", "retrieval", {})
        self._top_k = retrieval_config.get("top_k_retrieval", 2000)
        self._rrf_k = retrieval_config.get("rrf_k", 60)
        self._bm25_weight = retrieval_config.get("bm25_weight", 0.4)
        self._dense_weight = retrieval_config.get("dense_weight", 0.6)
        self._bm25_top_k = retrieval_config.get("bm25_top_k", 3000)
        self._dense_top_k = retrieval_config.get("dense_top_k", 2000)

        self._embedding_builder = EmbeddingBuilder()
        self._bm25 = BM25()
        self._candidate_ids: list[str] = []
        self._is_fitted: bool = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 with basic stop-word removal.

        Args:
            text: Input text.

        Returns:
            List of lowercase tokens.
        """
        # Basic stop words that hurt BM25 precision
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "and", "or", "but", "if", "while", "of", "at", "by", "for",
            "with", "about", "between", "through", "during", "before",
            "after", "to", "from", "in", "on", "it", "its", "this",
            "that", "these", "those", "i", "we", "you", "they", "he",
            "she", "my", "our", "your", "their", "his", "her",
        }
        text = text.lower()
        text = re.sub(r"[^\w\s\-]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1 and t not in stop_words]

    def build_index(
        self,
        candidates: list[Candidate],
        embeddings: np.ndarray | None = None,
        embeddings_path: Path | None = None,
    ) -> None:
        """Build both BM25 and dense indices.

        Args:
            candidates: List of all candidates.
            embeddings: Pre-loaded embeddings array.
            embeddings_path: Optional path to cached embeddings.
        """
        self._candidate_ids = [c.candidate_id for c in candidates]
        candidate_texts = [c.get_text_representation() for c in candidates]

        # Build BM25 index
        with timer("Building BM25 index"):
            tokenized_docs = [self.tokenize(text) for text in candidate_texts]
            self._bm25.fit(tokenized_docs)

        # Build dense index
        if embeddings is not None:
            self._embedding_builder.build_index(embeddings)
        else:
            cache_path = embeddings_path or Path("cache/embeddings.npy")
            if cache_path.exists():
                logger.info("Loading cached embeddings for retriever")
                emb = self._embedding_builder.load_embeddings(cache_path)
                self._embedding_builder.build_index(emb)
            else:
                emb = self._embedding_builder.encode_texts(candidate_texts)
                self._embedding_builder.save_embeddings(emb, cache_path)
                self._embedding_builder.build_index(emb)

        self._is_fitted = True
        logger.info("Hybrid retriever index built successfully")

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Retrieve top candidates using RRF-fused hybrid scoring.

        Args:
            query: Job description query text.
            top_k: Number of candidates to retrieve.

        Returns:
            List of (candidate_id, rrf_score, component_scores) tuples.
        """
        if not self._is_fitted:
            raise RuntimeError("Index not built. Call build_index first.")

        k = top_k or self._top_k

        # BM25 retrieval
        with timer("BM25 retrieval"):
            query_tokens = self.tokenize(query)
            bm25_results = self._bm25.score_query(
                query_tokens, top_k=self._bm25_top_k
            )

        # Dense retrieval
        with timer("Dense retrieval"):
            query_embedding = self._embedding_builder.encode_query(query)
            dense_scores, dense_indices = self._embedding_builder.search(
                query_embedding, top_k=self._dense_top_k
            )

        # Compute RRF scores
        with timer("RRF fusion"):
            rrf_scores = self._reciprocal_rank_fusion(
                bm25_results, dense_scores, dense_indices
            )

        # Sort by RRF score and take top-k
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )[:k]

        results = [
            (
                self._candidate_ids[idx],
                scores["rrf_score"],
                scores,
            )
            for idx, scores in sorted_results
            if idx < len(self._candidate_ids)
        ]

        logger.info(
            f"Hybrid retrieval: {len(bm25_results)} BM25 + "
            f"{len(dense_scores)} dense -> {len(results)} fused (top-{k})"
        )
        return results

    def retrieve_multi_query(
        self, queries: list[str], top_k: int | None = None
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Multi-query retrieval with RRF fusion across all queries.

        Retrieves candidates for multiple JD facets independently,
        then merges results using RRF. This improves recall for
        candidates who match specific JD requirements strongly
        but don't match the general embedding well.

        Args:
            queries: List of query strings (different JD facets).
            top_k: Final number of candidates to return.

        Returns:
            List of (candidate_id, rrf_score, component_scores) tuples.
        """
        if not self._is_fitted:
            raise RuntimeError("Index not built. Call build_index first.")

        k = top_k or self._top_k

        # Collect all unique candidate scores across queries
        all_scores: dict[int, dict[str, float]] = {}

        for qi, query in enumerate(queries):
            # BM25 for this query facet
            query_tokens = self.tokenize(query)
            bm25_results = self._bm25.score_query(
                query_tokens, top_k=self._bm25_top_k
            )

            # Dense for this query facet
            query_embedding = self._embedding_builder.encode_query(query)
            dense_scores, dense_indices = self._embedding_builder.search(
                query_embedding, top_k=self._dense_top_k
            )

            # RRF for this query
            query_rrf = self._reciprocal_rank_fusion(
                bm25_results, dense_scores, dense_indices
            )

            # Merge into global scores (max RRF across queries)
            for doc_id, scores in query_rrf.items():
                if doc_id not in all_scores:
                    all_scores[doc_id] = {
                        "rrf_score": 0.0,
                        "bm25_rank": 0.0,
                        "dense_rank": 0.0,
                        "bm25_score": 0.0,
                        "dense_score": 0.0,
                    }
                # Accumulate RRF scores (candidates matching multiple facets get boosted)
                all_scores[doc_id]["rrf_score"] += scores["rrf_score"]
                all_scores[doc_id]["bm25_score"] = max(
                    all_scores[doc_id]["bm25_score"], scores["bm25_score"]
                )
                all_scores[doc_id]["dense_score"] = max(
                    all_scores[doc_id]["dense_score"], scores["dense_score"]
                )

        # Sort by accumulated RRF score
        sorted_results = sorted(
            all_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )[:k]

        results = [
            (self._candidate_ids[idx], scores["rrf_score"], scores)
            for idx, scores in sorted_results
            if idx < len(self._candidate_ids)
        ]

        logger.info(
            f"Multi-query retrieval: {len(queries)} queries -> "
            f"{len(all_scores)} unique candidates -> {len(results)} returned"
        )
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        dense_scores: np.ndarray,
        dense_indices: np.ndarray,
    ) -> dict[int, dict[str, float]]:
        """Compute Reciprocal Rank Fusion scores.

        RRF(d) = bm25_weight / (k + rank_bm25(d)) + dense_weight / (k + rank_dense(d))

        Args:
            bm25_results: BM25 (doc_id, score) tuples sorted by score desc.
            dense_scores: Dense similarity scores from FAISS.
            dense_indices: Dense document indices from FAISS.

        Returns:
            Dict mapping doc_id -> {rrf_score, bm25_rank, dense_rank, bm25_score, dense_score}.
        """
        rrf_k = self._rrf_k
        fused: dict[int, dict[str, float]] = {}

        # BM25 component
        max_bm25 = bm25_results[0][1] if bm25_results else 1.0
        for rank, (doc_id, score) in enumerate(bm25_results, 1):
            if doc_id not in fused:
                fused[doc_id] = {
                    "rrf_score": 0.0,
                    "bm25_rank": 0.0,
                    "dense_rank": 0.0,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                }
            fused[doc_id]["bm25_rank"] = float(rank)
            fused[doc_id]["bm25_score"] = score / max_bm25 if max_bm25 > 0 else 0.0
            fused[doc_id]["rrf_score"] += self._bm25_weight / (rrf_k + rank)

        # Dense component
        max_dense = float(dense_scores[0]) if len(dense_scores) > 0 else 1.0
        for rank, (idx, score) in enumerate(
            zip(dense_indices, dense_scores), 1
        ):
            doc_id = int(idx)
            if doc_id < 0:
                continue
            if doc_id not in fused:
                fused[doc_id] = {
                    "rrf_score": 0.0,
                    "bm25_rank": 0.0,
                    "dense_rank": 0.0,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                }
            fused[doc_id]["dense_rank"] = float(rank)
            fused[doc_id]["dense_score"] = (
                float(score) / max_dense if max_dense > 0 else 0.0
            )
            fused[doc_id]["rrf_score"] += self._dense_weight / (rrf_k + rank)

        return fused

    def get_component_scores(
        self, results: list[tuple[str, float, dict[str, float]]]
    ) -> dict[str, dict[str, float]]:
        """Extract component scores for downstream use.

        Args:
            results: Retrieved results with component scores.

        Returns:
            Dict mapping candidate_id -> component scores.
        """
        return {cid: scores for cid, _, scores in results}

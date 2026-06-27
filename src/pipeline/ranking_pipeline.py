"""End-to-end ranking pipeline orchestration.

Upgraded pipeline:
1. Parse JD
2. Load candidates
3. Build/load embeddings + FAISS index
4. Hybrid retrieval (BM25 + Dense + RRF) -> top-2000
5. Full feature engineering (150+ features)
6. LTR ranking (LightGBM LambdaMART with weak supervision)
7. Fallback ensemble if LTR unavailable
8. Generate evidence-based reasoning
9. Write validated submission CSV
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.config.loader import get_config
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.preprocessing.data_loader import CandidateLoader
from src.preprocessing.jd_parser import JDParser
from src.ranking.ltr_ranker import LTRRanker
from src.reasoning.reason_generator import ReasonGenerator
from src.retrieval.embedding_builder import EmbeddingBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.timing import timer


class RankingPipeline:
    """Orchestrates the full candidate ranking pipeline.

    Pipeline stages:
    1. Parse JD with NLP extraction
    2. Load candidates (streaming, 100K)
    3. Build/load embeddings and FAISS index
    4. Hybrid retrieval (BM25 + Dense + RRF) -> top-2000
    5. Full feature scoring (150+ features per candidate)
    6. LTR ranking with weak supervision OR ensemble fallback
    7. Generate evidence-based reasoning
    8. Write submission CSV
    """

    def __init__(self) -> None:
        """Initialize pipeline components."""
        self._config = get_config()
        self._jd_parser = JDParser()
        self._embedding_builder = EmbeddingBuilder()
        self._feature_builder = FeatureBuilder()
        self._ltr_ranker = LTRRanker()
        self._reason_generator = ReasonGenerator()

    def run(
        self,
        candidates_path: Path | None = None,
        jd_path: Path | None = None,
        output_path: Path | None = None,
        use_cache: bool = True,
    ) -> Path:
        """Execute the full ranking pipeline.

        Args:
            candidates_path: Path to candidates.jsonl.
            jd_path: Path to job_description.docx.
            output_path: Path for output submission.csv.
            use_cache: Whether to use cached embeddings/indices.

        Returns:
            Path to the generated submission CSV.
        """
        # Resolve paths
        if candidates_path is None:
            candidates_path = self._config.get_path("data.candidates_file")
        if jd_path is None:
            jd_path = self._config.get_path("data.job_description_file")
        if output_path is None:
            output_path = self._config.get_path("output.submission_file")

        logger.info("=" * 60)
        logger.info("STARTING RANKING PIPELINE V2")
        logger.info("=" * 60)

        # Stage 1: Parse JD
        with timer("Stage 1: Parse Job Description"):
            jd = self._jd_parser.parse_from_file(jd_path)
            parsed_jd = jd.parsed
            assert parsed_jd is not None
            logger.info(f"JD: {parsed_jd.title} @ {parsed_jd.company}")

        # Stage 2: Load candidates
        with timer("Stage 2: Load Candidates"):
            loader = CandidateLoader(candidates_path)
            candidates = loader.load_all()
            candidate_map = {c.candidate_id: c for c in candidates}

        # Stage 3: Build/load embeddings and FAISS index
        with timer("Stage 3: Build Retrieval Index"):
            embeddings_cache = Path("cache/embeddings.npy")

            if use_cache and embeddings_cache.exists():
                logger.info("Using cached embeddings")
                embeddings = self._embedding_builder.load_embeddings(embeddings_cache)
            else:
                candidate_texts = [c.get_text_representation() for c in candidates]
                embeddings = self._embedding_builder.encode_texts(candidate_texts)
                self._embedding_builder.save_embeddings(embeddings, embeddings_cache)

            self._embedding_builder.build_index(embeddings)

        # Stage 4: Hybrid Retrieval (BM25 + Dense + RRF)
        with timer("Stage 4: Hybrid Retrieval"):
            retriever = HybridRetriever()
            retriever.build_index(candidates, embeddings=embeddings)

            retrieval_k = self._config.get("ranking", "retrieval.top_k_retrieval", 2000)

            # Multi-query retrieval: different JD facets for better recall
            jd_queries = [
                parsed_jd.text_for_embedding,
                "Senior AI Engineer embeddings retrieval ranking search systems production deployment vector databases FAISS Pinecone",
                "Machine Learning Engineer NLP recommendation systems learning-to-rank evaluation NDCG hybrid search",
                "Search ranking engineer information retrieval BM25 dense retrieval reranking Python production",
                "ML Engineer LLM fine-tuning LoRA RAG embeddings sentence-transformers MLOps model serving",
            ]

            retrieval_results = retriever.retrieve_multi_query(
                jd_queries, top_k=retrieval_k
            )

            # Extract semantic scores and retrieved candidate IDs
            semantic_scores: dict[str, float] = {}
            retrieved_ids: list[str] = []
            for cid, rrf_score, component_scores in retrieval_results:
                retrieved_ids.append(cid)
                semantic_scores[cid] = component_scores.get("dense_score", 0.0)

            retrieved_candidates = [
                candidate_map[cid] for cid in retrieved_ids
                if cid in candidate_map
            ]
            logger.info(
                f"Retrieved {len(retrieved_candidates)} candidates via hybrid RRF"
            )

        # Stage 5: Full Feature Engineering
        with timer("Stage 5: Feature Engineering (150+ features)"):
            features_list = self._feature_builder.build_batch_features(
                retrieved_candidates, parsed_jd
            )

            # Add semantic/retrieval scores to features
            for i, cid in enumerate(retrieved_ids):
                if i < len(features_list):
                    features_list[i]["semantic_similarity"] = semantic_scores.get(cid, 0.0)
                    # Add RRF score as a feature
                    _, rrf_score, comp = retrieval_results[i] if i < len(retrieval_results) else ("", 0, {})
                    features_list[i]["retrieval_rrf_score"] = rrf_score
                    features_list[i]["retrieval_bm25_score"] = comp.get("bm25_score", 0.0)
                    features_list[i]["retrieval_dense_score"] = comp.get("dense_score", 0.0)

        # Stage 6: LTR Ranking
        with timer("Stage 6: Learning-to-Rank"):
            # Generate pseudo-labels from weak supervision
            labels = self._ltr_ranker.generate_pseudo_labels(
                features_list, semantic_scores, retrieved_ids
            )

            # Train LTR model
            feature_names = self._feature_builder.get_feature_names() + [
                "semantic_similarity",
                "retrieval_rrf_score",
                "retrieval_bm25_score",
                "retrieval_dense_score",
            ]

            self._ltr_ranker.train(features_list, labels, feature_names)

            # Predict final scores
            final_scores = self._ltr_ranker.predict(features_list)

        # Stage 7: Sort and select top 100
        with timer("Stage 7: Final Ranking"):
            # Combine indices, IDs, scores, features
            ranked_results: list[tuple[str, float, dict[str, float]]] = []
            for i, (cid, score) in enumerate(zip(retrieved_ids, final_scores)):
                if i < len(features_list):
                    ranked_results.append((cid, float(score), features_list[i]))

            # Sort by score descending
            ranked_results.sort(key=lambda x: x[1], reverse=True)

            # Take top 100
            top_100 = ranked_results[:100]

            # Normalize scores for submission
            top_100 = self._normalize_scores(top_100)

            logger.info(
                f"Top-100 selected. Score range: "
                f"{top_100[0][1]:.4f} - {top_100[-1][1]:.4f}"
            )

        # Stage 8: Generate Reasoning
        with timer("Stage 8: Generate Reasoning"):
            submission_rows: list[dict[str, Any]] = []
            for rank_idx, (cid, score, features) in enumerate(top_100, 1):
                candidate = candidate_map[cid]
                reasoning = self._reason_generator.generate(
                    candidate, features, rank_idx, parsed_jd
                )
                submission_rows.append({
                    "candidate_id": cid,
                    "rank": rank_idx,
                    "score": score,
                    "reasoning": reasoning,
                })

        # Stage 9: Write Submission CSV
        with timer("Stage 9: Write Submission CSV"):
            self._write_submission(submission_rows, output_path)

        # Log feature importance if available
        importance = self._ltr_ranker.get_feature_importance()
        if importance:
            top_features = list(importance.items())[:15]
            logger.info("Top 15 features by importance:")
            for name, score in top_features:
                logger.info(f"  {name}: {score:.1f}")

        logger.info("=" * 60)
        logger.info(f"PIPELINE V2 COMPLETE: {output_path}")
        logger.info("=" * 60)

        return output_path

    def _normalize_scores(
        self, results: list[tuple[str, float, dict[str, float]]]
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Normalize scores using rank-based sigmoid scaling.

        Uses a sigmoid function applied to raw score percentiles to produce
        well-distributed scores that preserve the model's ranking order
        while avoiding saturation at the top.

        The key insight: the competition evaluates rank ORDER (NDCG), so we
        need scores that differentiate between ranks, especially at the top.
        Min-max normalization destroys this by saturating the top-k.

        Args:
            results: Sorted results (descending by score).

        Returns:
            Results with normalized scores that are strictly non-increasing.
        """
        if not results:
            return []

        n = len(results)
        normalized = []

        for i, (cid, raw_score, features) in enumerate(results):
            # Rank-based score: higher rank = higher score
            # Uses a curve that spreads top candidates more than bottom
            rank_fraction = i / max(n - 1, 1)  # 0.0 for rank 1, 1.0 for rank N

            # Sigmoid-inspired curve: more spread at top, compressed at bottom
            # score = 0.99 * (1 - rank_fraction^0.6) + 0.01
            norm_score = 0.99 * (1.0 - rank_fraction ** 0.6) + 0.01

            # Round carefully to maintain monotonicity
            norm_score = round(norm_score, 4)

            # Enforce strict non-increasing (handles rounding edge cases)
            if normalized and norm_score >= normalized[-1][1]:
                norm_score = normalized[-1][1] - 0.0001

            # Clamp to valid range
            norm_score = max(0.01, norm_score)

            normalized.append((cid, norm_score, features))

        return normalized

    def _write_submission(
        self, rows: list[dict[str, Any]], output_path: Path
    ) -> None:
        """Write submission CSV with proper formatting.

        Ensures:
        - Scores are non-increasing
        - Ties broken by candidate_id ascending
        - Valid CSV format

        Args:
            rows: List of submission row dictionaries.
            output_path: Output CSV file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Enforce non-increasing scores
        for i in range(1, len(rows)):
            if rows[i]["score"] > rows[i - 1]["score"]:
                rows[i]["score"] = rows[i - 1]["score"]

        # Handle ties: break by candidate_id ascending
        i = 0
        while i < len(rows) - 1:
            j = i
            while j < len(rows) - 1 and rows[j + 1]["score"] == rows[j]["score"]:
                j += 1
            # rows[i:j+1] all have the same score
            if j > i:
                tied = rows[i : j + 1]
                tied.sort(key=lambda r: r["candidate_id"])
                for k, row in enumerate(tied):
                    row["rank"] = i + k + 1
                    rows[i + k] = row
            i = j + 1

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["candidate_id", "rank", "score", "reasoning"],
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": f"{row['score']:.4f}",
                    "reasoning": row["reasoning"],
                })

        logger.info(f"Submission written: {output_path} ({len(rows)} rows)")

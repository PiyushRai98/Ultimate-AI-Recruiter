"""Main candidate ranking engine."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from src.config.loader import get_config
from src.feature_engineering.feature_builder import FeatureBuilder
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.utils.timing import timer


class CandidateRanker:
    """Hybrid ranking engine combining feature engineering with weighted ensemble.

    The ranking strategy:
    1. Feature-based scoring (skill, career, experience, behavioral)
    2. Honeypot penalty application
    3. Weighted ensemble combination
    4. Final sorting and score normalization
    """

    def __init__(self) -> None:
        """Initialize ranker with configuration."""
        config = get_config()
        self._ensemble_weights = config.get("weights", "ensemble", {})
        self._feature_builder = FeatureBuilder()

    def rank(
        self,
        candidates: list[Candidate],
        jd: ParsedJD,
        semantic_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Rank candidates against a job description.

        Args:
            candidates: List of candidates to rank.
            jd: Parsed job description.
            semantic_scores: Optional pre-computed semantic similarity scores
                            mapping candidate_id -> score.

        Returns:
            List of (candidate_id, final_score, features) tuples
            sorted by score descending.
        """
        with timer(f"Ranking {len(candidates)} candidates"):
            results: list[tuple[str, float, dict[str, float]]] = []

            for candidate in candidates:
                # Build features
                features = self._feature_builder.build_features(candidate, jd)

                # Add semantic score if available
                if semantic_scores and candidate.candidate_id in semantic_scores:
                    features["semantic_similarity"] = semantic_scores[
                        candidate.candidate_id
                    ]
                else:
                    features["semantic_similarity"] = 0.0

                # Compute final score
                final_score = self._compute_ensemble_score(features)
                results.append((candidate.candidate_id, final_score, features))

            # Sort by final score descending
            results.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                f"Ranking complete. Top score: {results[0][1]:.4f}, "
                f"Bottom score: {results[-1][1]:.4f}"
            )

        return results

    def _compute_ensemble_score(self, features: dict[str, float]) -> float:
        """Compute weighted ensemble score from features.

        Args:
            features: Dictionary of computed features.

        Returns:
            Final score between 0.0 and 1.0.
        """
        weights = self._ensemble_weights

        # Component scores
        semantic = features.get("semantic_similarity", 0.0)
        skill = features.get("skill_combined_score", 0.0)
        career = features.get("career_combined_score", 0.0)
        behavioral = features.get("behavioral_combined_score", 0.0)
        experience = features.get("experience_combined_score", 0.0)

        # Honeypot penalty (multiplicative)
        honeypot_multiplier = features.get("honeypot_penalty_multiplier", 1.0)

        # Weighted sum
        raw_score = (
            semantic * weights.get("semantic_score", 0.25)
            + skill * weights.get("skill_match_score", 0.20)
            + career * weights.get("career_fit_score", 0.20)
            + behavioral * weights.get("behavioral_score", 0.15)
            + experience * weights.get("experience_score", 0.10)
        )

        # Apply honeypot penalty as multiplicative factor
        penalized_score = raw_score * honeypot_multiplier

        # Clamp to [0, 1]
        return max(0.0, min(1.0, penalized_score))

    def normalize_scores(
        self, results: list[tuple[str, float, dict[str, float]]], top_k: int = 100
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Normalize scores to a nice range for submission.

        Maps top_k scores linearly from highest to lowest.

        Args:
            results: Ranked results from rank().
            top_k: Number of results to output.

        Returns:
            Top-K results with normalized scores.
        """
        top_results = results[:top_k]

        if not top_results:
            return []

        # Get raw score range
        scores = [r[1] for r in top_results]
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        # Normalize to [0.2, 0.99] range (non-increasing with rank)
        normalized = []
        for i, (cid, raw_score, features) in enumerate(top_results):
            if score_range > 0:
                norm_score = 0.2 + 0.79 * (raw_score - min_score) / score_range
            else:
                norm_score = 0.5

            # Ensure non-increasing by rank
            norm_score = round(norm_score, 4)
            normalized.append((cid, norm_score, features))

        # Enforce non-increasing constraint
        for i in range(1, len(normalized)):
            cid, score, features = normalized[i]
            prev_score = normalized[i - 1][1]
            if score > prev_score:
                normalized[i] = (cid, prev_score, features)

        return normalized

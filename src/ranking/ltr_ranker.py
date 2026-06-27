"""Learning-to-Rank engine with weak supervision.

Uses LightGBM LambdaMART with pseudo-labels generated from
multi-signal agreement (semantic + skill + career + behavioral).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.config.loader import get_config
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.utils.timing import timer


class LTRRanker:
    """LightGBM-based Learning-to-Rank model.

    Since no ground-truth labels exist, we use weak supervision:
    generate pseudo-labels from multiple signal agreement, then
    train a LambdaMART model to optimize NDCG.
    """

    def __init__(self) -> None:
        """Initialize LTR ranker."""
        config = get_config()
        self._model = None
        self._model_path = Path("artifacts/models/ltr_model.txt")
        self._feature_names: list[str] = []
        self._is_trained = False

    def generate_pseudo_labels(
        self,
        features_list: list[dict[str, float]],
        semantic_scores: dict[str, float],
        candidate_ids: list[str],
    ) -> np.ndarray:
        """Generate pseudo-labels using weak supervision.

        Combines multiple signals to create relevance labels:
        - Semantic similarity (dense retrieval score)
        - Skill match quality
        - Career fitness
        - Behavioral signals
        - Honeypot detection

        Args:
            features_list: List of feature dicts for each candidate.
            semantic_scores: Semantic similarity scores.
            candidate_ids: Candidate IDs in order.

        Returns:
            Array of pseudo-labels (0-4 relevance scale).
        """
        n = len(features_list)
        raw_scores = np.zeros(n, dtype=np.float32)

        for i, (features, cid) in enumerate(zip(features_list, candidate_ids)):
            # Weighted combination of key signals
            semantic = semantic_scores.get(cid, 0.0)
            skill = features.get("skill_combined_score", 0.0)
            ontology = features.get("ontology_coverage_ratio", 0.0)
            career = features.get("career_ml_production_combined", 0.0)
            company = features.get("company_avg_quality", 0.0)
            behavioral = features.get("bi_combined_score", 0.0)
            experience = features.get("experience_fit_score", 0.0)
            honeypot = features.get("honeypot_penalty_multiplier", 1.0)

            # Multi-signal agreement score
            raw_scores[i] = (
                semantic * 0.20
                + skill * 0.20
                + ontology * 0.10
                + career * 0.20
                + company * 0.10
                + behavioral * 0.10
                + experience * 0.10
            ) * honeypot

        # Convert to relevance grades (0-4)
        # Use percentile-based binning
        labels = np.zeros(n, dtype=np.float32)
        if n > 0:
            p95 = np.percentile(raw_scores, 95)
            p80 = np.percentile(raw_scores, 80)
            p60 = np.percentile(raw_scores, 60)
            p40 = np.percentile(raw_scores, 40)

            for i, score in enumerate(raw_scores):
                if score >= p95:
                    labels[i] = 4.0
                elif score >= p80:
                    labels[i] = 3.0
                elif score >= p60:
                    labels[i] = 2.0
                elif score >= p40:
                    labels[i] = 1.0
                else:
                    labels[i] = 0.0

        return labels

    def train(
        self,
        features_list: list[dict[str, float]],
        labels: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Train the LTR model using LightGBM LambdaMART.

        Args:
            features_list: List of feature dictionaries.
            labels: Pseudo-labels for training.
            feature_names: Optional feature name list.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not available. Using fallback scoring.")
            return

        if feature_names:
            self._feature_names = feature_names
        else:
            self._feature_names = sorted(features_list[0].keys()) if features_list else []

        # Build feature matrix
        X = self._build_feature_matrix(features_list)
        n_samples = X.shape[0]

        logger.info(
            f"Training LTR: {n_samples} samples, "
            f"{X.shape[1]} features, "
            f"label distribution: {np.bincount(labels.astype(int))}"
        )

        # Single group (all candidates are one query)
        group = [n_samples]

        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X, label=labels, group=group, feature_name=self._feature_names
        )

        # LambdaMART parameters optimized for ranking
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 50, 100],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "max_depth": 7,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        with timer("Training LightGBM LambdaMART"):
            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(period=50)],
            )

        self._is_trained = True
        logger.info("LTR model trained successfully")

        # Save model
        self._save_model()

    def predict(self, features_list: list[dict[str, float]]) -> np.ndarray:
        """Predict relevance scores for candidates.

        Args:
            features_list: List of feature dictionaries.

        Returns:
            Array of predicted relevance scores.
        """
        if not self._is_trained and not self._load_model():
            logger.warning("No LTR model available. Using fallback.")
            return self._fallback_scores(features_list)

        X = self._build_feature_matrix(features_list)
        scores = self._model.predict(X)
        return scores

    def _build_feature_matrix(
        self, features_list: list[dict[str, float]]
    ) -> np.ndarray:
        """Convert feature dicts to numpy matrix.

        Args:
            features_list: List of feature dictionaries.

        Returns:
            Feature matrix of shape (n_candidates, n_features).
        """
        if not self._feature_names:
            self._feature_names = sorted(features_list[0].keys()) if features_list else []

        n = len(features_list)
        m = len(self._feature_names)
        X = np.zeros((n, m), dtype=np.float32)

        for i, features in enumerate(features_list):
            for j, name in enumerate(self._feature_names):
                X[i, j] = features.get(name, 0.0)

        return X

    def _fallback_scores(self, features_list: list[dict[str, float]]) -> np.ndarray:
        """Fallback scoring when LTR model is not available.

        Uses a weighted combination of key features as the score.
        """
        scores = np.zeros(len(features_list), dtype=np.float32)

        for i, features in enumerate(features_list):
            scores[i] = (
                features.get("skill_combined_score", 0) * 0.15
                + features.get("ontology_coverage_ratio", 0) * 0.08
                + features.get("career_ml_production_combined", 0) * 0.15
                + features.get("company_avg_quality", 0) * 0.08
                + features.get("experience_fit_score", 0) * 0.10
                + features.get("bi_combined_score", 0) * 0.08
                + features.get("ix_ml_production", 0) * 0.08
                + features.get("ix_product_ml", 0) * 0.05
                + features.get("career_search_maturity", 0) * 0.05
                + features.get("evidence_jd_alignment", 0) * 0.10
                + features.get("evidence_search_retrieval", 0) * 0.05
                + features.get("ix_evidence_product", 0) * 0.03
            ) * features.get("honeypot_penalty_multiplier", 1.0)

        return scores

    def _save_model(self) -> None:
        """Save trained model to disk."""
        if self._model is None:
            return

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self._model_path))
        logger.info(f"LTR model saved: {self._model_path}")

    def _load_model(self) -> bool:
        """Load model from disk.

        Returns:
            True if model loaded successfully.
        """
        if not self._model_path.exists():
            return False

        try:
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(self._model_path))
            self._is_trained = True
            logger.info(f"LTR model loaded: {self._model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LTR model: {e}")
            return False

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model.

        Returns:
            Dict mapping feature names to importance scores.
        """
        if not self._is_trained or self._model is None:
            return {}

        importance = self._model.feature_importance(importance_type="gain")
        names = self._feature_names or [
            f"f{i}" for i in range(len(importance))
        ]

        return dict(sorted(
            zip(names, importance),
            key=lambda x: x[1],
            reverse=True,
        ))

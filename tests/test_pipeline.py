"""Integration and unit tests for the ranking system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.loader import get_config
from src.evaluation.metrics import (
    compute_competition_score,
    dcg_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
)
from src.feature_engineering.skill_ontology import SkillOntology
from src.models.candidate import Candidate
from src.preprocessing.data_loader import CandidateLoader


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self) -> None:
        """Test that all config files load successfully."""
        config = get_config()
        assert config.weights is not None
        assert config.ranking is not None
        assert config.features is not None
        assert config.paths is not None

    def test_weights_sum(self) -> None:
        """Test that ensemble weights are reasonable."""
        config = get_config()
        ensemble = config.get("weights", "ensemble", {})
        total = sum(ensemble.values())
        assert abs(total - 1.0) < 0.01, f"Ensemble weights sum to {total}"

    def test_retrieval_config(self) -> None:
        """Test retrieval configuration is valid."""
        config = get_config()
        top_k = config.get("ranking", "retrieval.top_k_retrieval")
        assert top_k is not None
        assert top_k >= 100


class TestDataLoader:
    """Test candidate data loading."""

    def test_load_stream(self) -> None:
        """Test streaming candidate loading."""
        config = get_config()
        path = config.get_path("data.candidates_file")
        if not path.exists():
            pytest.skip("Candidates file not available")

        loader = CandidateLoader(path)
        count = 0
        for candidate in loader.stream():
            assert candidate.candidate_id.startswith("CAND_")
            assert candidate.years_of_experience >= 0
            assert len(candidate.skills) >= 0
            count += 1
            if count >= 10:
                break

        assert count == 10

    def test_candidate_text_representation(self) -> None:
        """Test that text representation is generated."""
        config = get_config()
        path = config.get_path("data.candidates_file")
        if not path.exists():
            pytest.skip("Candidates file not available")

        loader = CandidateLoader(path)
        for candidate in loader.stream():
            text = candidate.get_text_representation()
            assert len(text) > 0
            assert candidate.headline in text
            break


class TestSkillOntology:
    """Test the skill ontology system."""

    def test_ontology_loads(self) -> None:
        """Test ontology loads successfully."""
        ontology = SkillOntology()
        groups = ontology.get_all_groups()
        assert len(groups) > 10

    def test_same_group_similarity(self) -> None:
        """Test that skills in the same group have high similarity."""
        ontology = SkillOntology()
        # FAISS and Pinecone are both vector databases
        score = ontology.compute_similarity("FAISS", "Pinecone")
        assert score >= 0.6

    def test_cross_group_similarity(self) -> None:
        """Test that skills in different groups have low similarity."""
        ontology = SkillOntology()
        score = ontology.compute_similarity("Python", "Kubernetes")
        assert score < 0.3

    def test_coverage_computation(self) -> None:
        """Test coverage computation."""
        ontology = SkillOntology()
        result = ontology.compute_coverage(
            ["FAISS", "PyTorch", "BM25", "Elasticsearch", "NLP"],
            ["vector database", "deep learning", "retrieval", "NLP"],
        )
        assert result["ontology_coverage_ratio"] > 0.3
        assert result["ontology_group_coverage"] > 0.3
        # All result keys should exist
        assert "ontology_avg_match_score" in result
        assert "ontology_max_match_score" in result


class TestEvaluationMetrics:
    """Test evaluation metric implementations."""

    def test_perfect_ndcg(self) -> None:
        """Test NDCG=1.0 for perfect ranking."""
        relevance = np.array([4, 3, 2, 1, 0])
        assert ndcg_at_k(relevance, 5) == pytest.approx(1.0)

    def test_reversed_ndcg(self) -> None:
        """Test NDCG<1.0 for reversed ranking."""
        relevance = np.array([0, 1, 2, 3, 4])
        assert ndcg_at_k(relevance, 5) < 1.0

    def test_precision_at_k(self) -> None:
        """Test precision at K."""
        relevance = np.array([1, 1, 0, 1, 0])
        assert precision_at_k(relevance, 5) == pytest.approx(0.6)
        assert precision_at_k(relevance, 2) == pytest.approx(1.0)

    def test_map(self) -> None:
        """Test mean average precision."""
        relevance = np.array([1, 0, 1, 0, 1])
        score = mean_average_precision(relevance)
        assert 0.0 < score < 1.0

    def test_mrr(self) -> None:
        """Test MRR."""
        # First result is relevant
        assert mrr(np.array([1, 0, 0])) == pytest.approx(1.0)
        # Second result is relevant
        assert mrr(np.array([0, 1, 0])) == pytest.approx(0.5)
        # No relevant results
        assert mrr(np.array([0, 0, 0])) == pytest.approx(0.0)

    def test_composite_score(self) -> None:
        """Test competition composite score computation."""
        relevance = np.array([4, 3, 3, 2, 2, 2, 1, 1, 1, 1] + [0] * 90)
        metrics = compute_competition_score(relevance)
        assert "composite" in metrics
        assert "ndcg@10" in metrics
        assert "ndcg@50" in metrics
        assert "map" in metrics
        assert "p@10" in metrics
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

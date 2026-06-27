"""Career evidence extraction engine.

Mines career descriptions for hidden technical signals that
candidates may not list in their skills section. The JD explicitly
says: 'A candidate who built a recommendation system at a product
company is a fit even without listing RAG or Pinecone as skills.'

This module extracts production-system evidence, scale signals,
ownership indicators, and domain-specific depth from free text.
"""

from __future__ import annotations

from src.models.candidate import Candidate


# Domain evidence patterns: keyword -> (domain, weight)
_EVIDENCE_PATTERNS: dict[str, list[tuple[str, float]]] = {
    # Retrieval & Search
    "retrieval": [("search_retrieval", 1.0)],
    "search engine": [("search_retrieval", 1.0)],
    "search system": [("search_retrieval", 1.0)],
    "search relevance": [("search_retrieval", 1.0)],
    "query understanding": [("search_retrieval", 0.8)],
    "search ranking": [("search_retrieval", 1.0), ("ranking", 0.8)],
    "bm25": [("search_retrieval", 1.0)],
    "inverted index": [("search_retrieval", 0.9)],
    "elasticsearch": [("search_retrieval", 0.9)],
    "opensearch": [("search_retrieval", 0.9)],
    "solr": [("search_retrieval", 0.8)],
    "lucene": [("search_retrieval", 0.8)],
    # Ranking
    "ranking": [("ranking", 1.0)],
    "learning to rank": [("ranking", 1.0)],
    "ltr": [("ranking", 1.0)],
    "ndcg": [("ranking", 1.0)],
    "relevance scoring": [("ranking", 0.9)],
    "re-ranking": [("ranking", 0.9)],
    "candidate ranking": [("ranking", 0.9)],
    # Recommendation
    "recommendation": [("recommendation", 1.0)],
    "recommender": [("recommendation", 1.0)],
    "collaborative filtering": [("recommendation", 1.0)],
    "personalization": [("recommendation", 0.8)],
    "user modeling": [("recommendation", 0.8)],
    "item embedding": [("recommendation", 0.8), ("embeddings", 0.6)],
    # Embeddings & Vectors
    "embedding": [("embeddings", 1.0)],
    "vector search": [("embeddings", 1.0), ("search_retrieval", 0.7)],
    "vector database": [("embeddings", 0.9)],
    "faiss": [("embeddings", 1.0)],
    "pinecone": [("embeddings", 1.0)],
    "qdrant": [("embeddings", 1.0)],
    "milvus": [("embeddings", 1.0)],
    "similarity search": [("embeddings", 0.8)],
    "nearest neighbor": [("embeddings", 0.8)],
    "semantic search": [("embeddings", 0.9), ("search_retrieval", 0.8)],
    # NLP
    "nlp": [("nlp", 1.0)],
    "natural language": [("nlp", 1.0)],
    "text classification": [("nlp", 0.8)],
    "named entity": [("nlp", 0.8)],
    "sentiment": [("nlp", 0.7)],
    "tokeniz": [("nlp", 0.7)],
    "transformer": [("nlp", 0.9), ("deep_learning", 0.7)],
    "bert": [("nlp", 0.9)],
    "language model": [("nlp", 0.9), ("llm", 0.7)],
    # LLM
    "llm": [("llm", 1.0)],
    "large language model": [("llm", 1.0)],
    "fine-tun": [("llm", 0.9)],
    "lora": [("llm", 0.9)],
    "qlora": [("llm", 0.9)],
    "peft": [("llm", 0.9)],
    "rag": [("llm", 0.9), ("search_retrieval", 0.6)],
    "prompt engineer": [("llm", 0.7)],
    "instruction tuning": [("llm", 0.8)],
    # Production ML
    "production": [("production_ml", 0.7)],
    "deployed": [("production_ml", 0.8)],
    "serving": [("production_ml", 0.8)],
    "inference": [("production_ml", 0.8)],
    "real-time": [("production_ml", 0.7)],
    "latency": [("production_ml", 0.8)],
    "throughput": [("production_ml", 0.7)],
    "mlops": [("production_ml", 0.9)],
    "model monitoring": [("production_ml", 0.8)],
    "a/b test": [("production_ml", 0.9)],
    "feature store": [("production_ml", 0.8)],
    "model pipeline": [("production_ml", 0.8)],
    # Scale indicators
    "million": [("scale", 0.8)],
    "billion": [("scale", 1.0)],
    "at scale": [("scale", 0.9)],
    "high traffic": [("scale", 0.8)],
    "distributed": [("scale", 0.7)],
    "microservice": [("production_ml", 0.6), ("scale", 0.5)],
}


class CareerEvidenceExtractor:
    """Extracts technical domain signals from career descriptions.

    Analyzes free-text career descriptions to find evidence of
    domain expertise that may not appear in the skills section.
    """

    def __init__(self) -> None:
        """Initialize with pattern index."""
        self._patterns = _EVIDENCE_PATTERNS

    def extract_features(self, candidate: Candidate) -> dict[str, float]:
        """Extract career evidence features from descriptions.

        Args:
            candidate: Candidate to analyze.

        Returns:
            Dict of evidence-based features.
        """
        # Aggregate all career descriptions
        all_text = " ".join(
            entry.description.lower() for entry in candidate.career_history
        )

        # Also include summary
        all_text += " " + candidate.summary.lower()

        # Score each domain
        domain_scores: dict[str, float] = {
            "search_retrieval": 0.0,
            "ranking": 0.0,
            "recommendation": 0.0,
            "embeddings": 0.0,
            "nlp": 0.0,
            "llm": 0.0,
            "production_ml": 0.0,
            "deep_learning": 0.0,
            "scale": 0.0,
        }

        # Count evidence matches
        evidence_count = 0
        for pattern, domain_weights in self._patterns.items():
            if pattern in all_text:
                evidence_count += 1
                for domain, weight in domain_weights:
                    domain_scores[domain] += weight

        # Normalize each domain score (cap at 1.0)
        for domain in domain_scores:
            domain_scores[domain] = min(domain_scores[domain] / 3.0, 1.0)

        # Compute aggregate features
        total_evidence = min(evidence_count / 15.0, 1.0)

        # Core JD alignment: search + ranking + embeddings + production
        jd_alignment = (
            domain_scores["search_retrieval"] * 0.25
            + domain_scores["ranking"] * 0.25
            + domain_scores["embeddings"] * 0.20
            + domain_scores["production_ml"] * 0.15
            + domain_scores["nlp"] * 0.10
            + domain_scores["recommendation"] * 0.05
        )

        return {
            "evidence_search_retrieval": domain_scores["search_retrieval"],
            "evidence_ranking": domain_scores["ranking"],
            "evidence_recommendation": domain_scores["recommendation"],
            "evidence_embeddings": domain_scores["embeddings"],
            "evidence_nlp": domain_scores["nlp"],
            "evidence_llm": domain_scores["llm"],
            "evidence_production_ml": domain_scores["production_ml"],
            "evidence_scale": domain_scores["scale"],
            "evidence_total_count": total_evidence,
            "evidence_jd_alignment": jd_alignment,
        }

"""Advanced career progression feature engineering.

Extracts deep career trajectory signals including ML maturity,
production deployment experience, and domain-specific depth.
"""

from __future__ import annotations

from rapidfuzz import fuzz
from loguru import logger

from src.models.candidate import Candidate


class CareerProgressionEngine:
    """Computes advanced career progression features.

    Goes beyond simple title matching to understand:
    - ML/AI career maturity
    - Production vs research orientation
    - Domain depth in search/ranking/retrieval
    - Leadership trajectory
    """

    # Seniority level mapping
    SENIORITY_LEVELS: dict[str, int] = {
        "intern": 0,
        "trainee": 0,
        "junior": 1,
        "associate": 2,
        "mid": 3,
        "senior": 4,
        "staff": 5,
        "lead": 5,
        "principal": 6,
        "director": 7,
        "vp": 8,
        "head": 7,
        "chief": 9,
    }

    # Domain keywords for depth analysis
    ML_KEYWORDS = {
        "machine learning", "ml", "deep learning", "neural",
        "model", "training", "inference", "prediction",
        "classification", "regression", "clustering",
    }
    SEARCH_KEYWORDS = {
        "search", "retrieval", "ranking", "relevance",
        "index", "query", "elasticsearch", "solr", "lucene",
    }
    NLP_KEYWORDS = {
        "nlp", "natural language", "text", "language model",
        "tokeniz", "embedding", "transformer", "bert", "gpt",
    }
    RECOMMENDATION_KEYWORDS = {
        "recommendation", "personalization", "collaborative",
        "content-based", "user modeling", "click-through",
    }
    PRODUCTION_KEYWORDS = {
        "production", "deploy", "scale", "serving", "latency",
        "throughput", "api", "microservice", "pipeline", "monitor",
        "real-time", "batch", "distributed", "kubernetes", "docker",
    }

    def compute_features(self, candidate: Candidate) -> dict[str, float]:
        """Compute comprehensive career progression features.

        Args:
            candidate: Candidate to analyze.

        Returns:
            Dictionary of career progression features.
        """
        career = candidate.career_history
        if not career:
            return self._empty_features()

        features: dict[str, float] = {}

        # Promotion velocity
        features["career_promotion_velocity"] = self._compute_promotion_velocity(career)

        # Average tenure
        durations = [e.duration_months for e in career]
        features["career_avg_tenure_months"] = (
            sum(durations) / len(durations) if durations else 0.0
        )

        # Career stability (penalize excessive job-hopping)
        features["career_stability_v2"] = self._compute_stability(durations)

        # ML maturity
        features["career_ml_maturity"] = self._compute_domain_depth(
            career, self.ML_KEYWORDS
        )

        # Search/Ranking maturity
        features["career_search_maturity"] = self._compute_domain_depth(
            career, self.SEARCH_KEYWORDS
        )

        # NLP maturity
        features["career_nlp_maturity"] = self._compute_domain_depth(
            career, self.NLP_KEYWORDS
        )

        # Recommendation systems maturity
        features["career_recsys_maturity"] = self._compute_domain_depth(
            career, self.RECOMMENDATION_KEYWORDS
        )

        # Production deployment experience
        features["career_production_depth"] = self._compute_domain_depth(
            career, self.PRODUCTION_KEYWORDS
        )

        # Leadership indicators
        features["career_leadership_score"] = self._compute_leadership(career)

        # Role consistency (staying in relevant field)
        features["career_role_consistency"] = self._compute_role_consistency(career)

        # Technical depth (seniority level achieved)
        features["career_technical_depth"] = self._compute_technical_depth(career)

        # Combined ML+Production score (what JD really wants)
        features["career_ml_production_combined"] = (
            features["career_ml_maturity"] * 0.3
            + features["career_search_maturity"] * 0.25
            + features["career_production_depth"] * 0.25
            + features["career_nlp_maturity"] * 0.1
            + features["career_recsys_maturity"] * 0.1
        )

        return features

    def _compute_promotion_velocity(self, career: list) -> float:
        """Compute how fast the candidate progresses in seniority."""
        if len(career) < 2:
            return 0.5

        levels = []
        for entry in career:
            level = self._get_seniority_level(entry.title)
            levels.append(level)

        # Career is ordered newest-first, so reverse for chronological
        levels_chrono = list(reversed(levels))

        if len(levels_chrono) < 2:
            return 0.5

        # Compute level gain per year
        total_months = sum(e.duration_months for e in career)
        total_years = total_months / 12.0 if total_months > 0 else 1.0

        level_gain = levels_chrono[-1] - levels_chrono[0]
        velocity = level_gain / total_years

        # Normalize: 0.5 levels/year is good, 1.0 is excellent
        return min(max(velocity / 1.0, 0.0), 1.0)

    def _get_seniority_level(self, title: str) -> int:
        """Extract seniority level from job title."""
        title_lower = title.lower()
        best_level = 3  # Default mid-level

        for keyword, level in self.SENIORITY_LEVELS.items():
            if keyword in title_lower:
                best_level = max(best_level, level)

        return best_level

    def _compute_stability(self, durations: list[int]) -> float:
        """Compute career stability score."""
        if not durations:
            return 0.5

        avg_tenure = sum(durations) / len(durations)

        # JD explicitly penalizes <18 month average tenure
        if avg_tenure < 12:
            return 0.1
        elif avg_tenure < 18:
            return 0.25
        elif avg_tenure < 24:
            return 0.5
        elif avg_tenure < 36:
            return 0.75
        elif avg_tenure < 60:
            return 0.9
        else:
            return 1.0

    def _compute_domain_depth(
        self, career: list, keywords: set[str]
    ) -> float:
        """Compute depth of experience in a specific domain.

        Looks at career descriptions and weights by recency.
        """
        total_signal = 0.0
        total_weight = 0.0

        for i, entry in enumerate(career):
            desc_lower = entry.description.lower()
            title_lower = entry.title.lower()
            recency_weight = 1.0 / (1 + i * 0.4)
            duration_weight = min(entry.duration_months / 24.0, 1.5)

            # Count keyword matches in description
            matches = sum(1 for kw in keywords if kw in desc_lower)
            # Also check title
            title_matches = sum(1 for kw in keywords if kw in title_lower)

            if matches > 0 or title_matches > 0:
                match_density = (matches + title_matches * 2) / (len(keywords) * 0.5)
                signal = min(match_density, 1.0) * recency_weight * duration_weight
                total_signal += signal

            total_weight += recency_weight

        if total_weight == 0:
            return 0.0

        return min(total_signal / total_weight, 1.0)

    def _compute_leadership(self, career: list) -> float:
        """Compute leadership trajectory score."""
        leadership_keywords = {
            "lead", "manage", "mentor", "team", "hire",
            "direct", "architect", "own", "drove", "built team",
        }
        leadership_titles = {"lead", "manager", "director", "head", "vp", "chief"}

        score = 0.0

        for i, entry in enumerate(career):
            desc_lower = entry.description.lower()
            title_lower = entry.title.lower()
            recency = 1.0 / (1 + i * 0.3)

            title_leadership = any(lt in title_lower for lt in leadership_titles)
            desc_leadership = sum(
                1 for kw in leadership_keywords if kw in desc_lower
            )

            if title_leadership:
                score += 0.4 * recency
            if desc_leadership > 0:
                score += min(desc_leadership * 0.1, 0.3) * recency

        return min(score, 1.0)

    def _compute_role_consistency(self, career: list) -> float:
        """Compute how consistent the career path is within tech/ML.

        JD penalizes scattered careers that jump between domains.
        """
        tech_roles = 0
        total_roles = len(career)

        tech_keywords = {
            "engineer", "developer", "scientist", "architect",
            "ml", "ai", "data", "software", "platform",
            "backend", "research", "nlp", "search",
        }

        for entry in career:
            title_lower = entry.title.lower()
            if any(kw in title_lower for kw in tech_keywords):
                tech_roles += 1

        if total_roles == 0:
            return 0.0

        return tech_roles / total_roles

    def _compute_technical_depth(self, career: list) -> float:
        """Compute technical depth based on highest seniority achieved."""
        if not career:
            return 0.0

        max_level = max(self._get_seniority_level(e.title) for e in career)
        # Normalize: level 5 (staff/lead) = 0.8, level 6+ = 1.0
        return min(max_level / 6.0, 1.0)

    def _empty_features(self) -> dict[str, float]:
        """Return empty feature dict."""
        return {
            "career_promotion_velocity": 0.0,
            "career_avg_tenure_months": 0.0,
            "career_stability_v2": 0.0,
            "career_ml_maturity": 0.0,
            "career_search_maturity": 0.0,
            "career_nlp_maturity": 0.0,
            "career_recsys_maturity": 0.0,
            "career_production_depth": 0.0,
            "career_leadership_score": 0.0,
            "career_role_consistency": 0.0,
            "career_technical_depth": 0.0,
            "career_ml_production_combined": 0.0,
        }

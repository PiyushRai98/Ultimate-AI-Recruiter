"""Company intelligence and classification engine.

Classifies companies into product/services/startup/research categories
and computes company-quality signals for career scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rapidfuzz import fuzz, process

from src.config.loader import get_config
from src.models.candidate import Candidate, CareerEntry


class CompanyClassifier:
    """Classifies companies and computes company-quality features.

    Uses a knowledge base of known companies plus heuristics
    based on industry, size, and career descriptions.
    """

    def __init__(self) -> None:
        """Initialize with company knowledge base."""
        config_path = Path(__file__).parent.parent.parent / "config" / "companies.yaml"
        self._load_knowledge_base(config_path)

    def _load_knowledge_base(self, path: Path) -> None:
        """Load company classification data."""
        if not path.exists():
            logger.warning(f"Companies config not found: {path}")
            self._product_t1 = []
            self._product_t2 = []
            self._product_t3 = []
            self._consulting = []
            self._research = []
            self._product_industries = []
            self._services_industries = []
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        product = data.get("product_companies", {})
        self._product_t1 = [c.lower() for c in product.get("tier_1", [])]
        self._product_t2 = [c.lower() for c in product.get("tier_2", [])]
        self._product_t3 = [c.lower() for c in product.get("tier_3", [])]
        self._all_product = self._product_t1 + self._product_t2 + self._product_t3

        self._consulting = [
            c.lower() for c in data.get("consulting_companies", [])
        ]
        self._research = [
            c.lower() for c in data.get("research_companies", [])
        ]
        self._product_industries = [
            i.lower() for i in data.get("product_industries", [])
        ]
        self._services_industries = [
            i.lower() for i in data.get("services_industries", [])
        ]

        logger.debug(
            f"Company KB loaded: {len(self._all_product)} product, "
            f"{len(self._consulting)} consulting"
        )

    def classify_company(
        self, company: str, industry: str, size: str, description: str = ""
    ) -> dict[str, Any]:
        """Classify a single company.

        Args:
            company: Company name.
            industry: Industry string.
            size: Company size string.
            description: Role description for context.

        Returns:
            Classification dict with type, tier, and scores.
        """
        company_lower = company.lower()
        industry_lower = industry.lower()

        # Check known lists
        if self._match_in_list(company_lower, self._product_t1):
            return {"type": "product", "tier": 1, "quality_score": 1.0}
        if self._match_in_list(company_lower, self._product_t2):
            return {"type": "product", "tier": 2, "quality_score": 0.85}
        if self._match_in_list(company_lower, self._product_t3):
            return {"type": "product", "tier": 3, "quality_score": 0.75}
        if self._match_in_list(company_lower, self._consulting):
            return {"type": "consulting", "tier": 0, "quality_score": 0.2}
        if self._match_in_list(company_lower, self._research):
            return {"type": "research", "tier": 2, "quality_score": 0.6}

        # Heuristic classification
        if any(ind in industry_lower for ind in self._product_industries):
            return {"type": "product", "tier": 3, "quality_score": 0.65}
        if any(ind in industry_lower for ind in self._services_industries):
            return {"type": "services", "tier": 0, "quality_score": 0.25}

        # Size-based heuristic
        if size in ("1-10", "11-50", "51-200"):
            return {"type": "startup", "tier": 3, "quality_score": 0.6}
        if size in ("201-500", "501-1000"):
            return {"type": "product_or_enterprise", "tier": 3, "quality_score": 0.55}

        return {"type": "unknown", "tier": 3, "quality_score": 0.4}

    def _match_in_list(self, company: str, company_list: list[str]) -> bool:
        """Check if company matches any entry in list (fuzzy)."""
        if company in company_list:
            return True
        # Fuzzy match for variations
        if company_list:
            match = process.extractOne(
                company, company_list, scorer=fuzz.ratio
            )
            if match and match[1] >= 88:
                return True
        return False

    def score_candidate(self, candidate: Candidate) -> dict[str, float]:
        """Compute company-intelligence features for a candidate.

        Args:
            candidate: Candidate to evaluate.

        Returns:
            Dictionary of company-related features.
        """
        career = candidate.career_history
        if not career:
            return self._empty_features()

        # Classify each company in career history
        classifications = []
        for entry in career:
            cls = self.classify_company(
                entry.company, entry.industry, entry.company_size, entry.description
            )
            classifications.append((cls, entry.duration_months))

        # Compute aggregate features
        total_months = sum(dur for _, dur in classifications)
        if total_months == 0:
            return self._empty_features()

        # Product experience ratio
        product_months = sum(
            dur for cls, dur in classifications
            if cls["type"] in ("product", "startup")
        )
        product_ratio = product_months / total_months

        # Consulting ratio
        consulting_months = sum(
            dur for cls, dur in classifications
            if cls["type"] in ("consulting", "services")
        )
        consulting_ratio = consulting_months / total_months

        # Weighted quality score (recent experience matters more)
        quality_scores = []
        for i, (cls, dur) in enumerate(classifications):
            recency_weight = 1.0 / (1 + i * 0.3)
            quality_scores.append(cls["quality_score"] * recency_weight * dur)

        total_weighted = sum(dur * (1.0 / (1 + i * 0.3)) for i, (_, dur) in enumerate(classifications))
        avg_quality = sum(quality_scores) / total_weighted if total_weighted > 0 else 0.0

        # Best company tier achieved
        best_tier = min(
            (cls["tier"] for cls, _ in classifications), default=4
        )

        # Current company classification
        current_cls = classifications[0][0] if classifications else {"type": "unknown", "quality_score": 0.4}

        # Startup experience bonus
        startup_months = sum(
            dur for cls, dur in classifications
            if cls["type"] == "startup"
        )
        startup_bonus = min(startup_months / 24.0, 1.0) * 0.3

        # FAANG/Tier-1 experience
        tier1_months = sum(
            dur for cls, dur in classifications
            if cls.get("tier") == 1
        )
        tier1_score = min(tier1_months / 36.0, 1.0)

        # Consulting penalty (entire career in consulting = heavy penalty)
        consulting_penalty = 0.0
        if consulting_ratio >= 1.0:
            consulting_penalty = 1.0
        elif consulting_ratio >= 0.7:
            consulting_penalty = 0.7
        elif consulting_ratio >= 0.5:
            consulting_penalty = 0.4

        return {
            "company_product_ratio": product_ratio,
            "company_consulting_ratio": consulting_ratio,
            "company_avg_quality": avg_quality,
            "company_best_tier": 1.0 - (best_tier / 4.0),  # Normalize: tier 1 = 1.0
            "company_current_quality": current_cls["quality_score"],
            "company_startup_bonus": startup_bonus,
            "company_tier1_score": tier1_score,
            "company_consulting_penalty": consulting_penalty,
        }

    def _empty_features(self) -> dict[str, float]:
        """Return empty feature dict."""
        return {
            "company_product_ratio": 0.0,
            "company_consulting_ratio": 0.0,
            "company_avg_quality": 0.0,
            "company_best_tier": 0.0,
            "company_current_quality": 0.0,
            "company_startup_bonus": 0.0,
            "company_tier1_score": 0.0,
            "company_consulting_penalty": 0.0,
        }

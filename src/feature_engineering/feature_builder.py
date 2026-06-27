"""Feature builder that orchestrates all scoring modules.

Upgraded to include:
- Skill ontology matching
- Company intelligence
- Career progression engine
- Behavioral intelligence
- 150+ total engineered features
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from src.behavior.behavioral_intelligence import BehavioralIntelligence
from src.behavior.behavioral_scorer import BehavioralScorer
from src.behavior.honeypot_detector import HoneypotDetector
from src.feature_engineering.career_evidence import CareerEvidenceExtractor
from src.feature_engineering.career_progression import CareerProgressionEngine
from src.feature_engineering.career_scorer import CareerScorer
from src.feature_engineering.company_classifier import CompanyClassifier
from src.feature_engineering.experience_scorer import ExperienceScorer
from src.feature_engineering.skill_ontology import SkillOntology
from src.feature_engineering.skill_scorer import SkillScorer
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD
from src.utils.timing import timer


class FeatureBuilder:
    """Orchestrates feature computation across all scoring modules.

    Combines skill, career, experience, behavioral, company intelligence,
    ontology matching, and honeypot features into a unified feature vector.
    """

    def __init__(self) -> None:
        """Initialize all scoring modules."""
        self._skill_scorer = SkillScorer()
        self._career_scorer = CareerScorer()
        self._experience_scorer = ExperienceScorer()
        self._behavioral_scorer = BehavioralScorer()
        self._honeypot_detector = HoneypotDetector()
        self._company_classifier = CompanyClassifier()
        self._career_progression = CareerProgressionEngine()
        self._behavioral_intelligence = BehavioralIntelligence()
        self._skill_ontology = SkillOntology()
        self._career_evidence = CareerEvidenceExtractor()

    def build_features(
        self, candidate: Candidate, jd: ParsedJD
    ) -> dict[str, float]:
        """Build complete feature dictionary for a candidate.

        Args:
            candidate: Candidate to compute features for.
            jd: Parsed job description.

        Returns:
            Dictionary mapping feature names to float values (150+ features).
        """
        features: dict[str, float] = {}

        # Skill features (existing)
        skill_features = self._skill_scorer.score(candidate, jd)
        features.update(skill_features)

        # Ontology-based skill matching (new)
        ontology_features = self._skill_ontology.compute_coverage(
            candidate.get_skill_names(),
            jd.must_have_skills + jd.preferred_skills,
        )
        features.update(ontology_features)

        # Career features (existing)
        career_features = self._career_scorer.score(candidate, jd)
        features.update(career_features)

        # Career progression (new - advanced)
        progression_features = self._career_progression.compute_features(candidate)
        features.update(progression_features)

        # Company intelligence (new)
        company_features = self._company_classifier.score_candidate(candidate)
        features.update(company_features)

        # Experience features (existing)
        experience_features = self._experience_scorer.score(candidate, jd)
        features.update(experience_features)

        # Behavioral features (existing)
        behavioral_features = self._behavioral_scorer.score(candidate)
        features.update(behavioral_features)

        # Behavioral intelligence (new - advanced)
        bi_features = self._behavioral_intelligence.compute_features(candidate)
        features.update(bi_features)

        # Honeypot features (existing)
        honeypot_features = self._honeypot_detector.compute_honeypot_score(candidate)
        features.update(honeypot_features)

        # Career evidence extraction (new - description mining)
        evidence_features = self._career_evidence.extract_features(candidate)
        features.update(evidence_features)

        # Cross-feature interactions (new)
        interaction_features = self._compute_interactions(features, candidate)
        features.update(interaction_features)

        return features

    def _compute_interactions(
        self, features: dict[str, float], candidate: Candidate
    ) -> dict[str, float]:
        """Compute feature interactions for richer signal.

        Args:
            features: Already-computed features.
            candidate: Candidate object.

        Returns:
            Dictionary of interaction features.
        """
        interactions: dict[str, float] = {}

        # Skill x Career interaction (relevant skills + relevant career)
        interactions["ix_skill_career"] = (
            features.get("skill_combined_score", 0)
            * features.get("career_combined_score", 0)
        )

        # Skill x Experience interaction
        interactions["ix_skill_experience"] = (
            features.get("skill_combined_score", 0)
            * features.get("experience_fit_score", 0)
        )

        # Career quality x Behavioral
        interactions["ix_career_behavioral"] = (
            features.get("company_avg_quality", 0)
            * features.get("bi_recruitability_score", 0)
        )

        # ML depth x Production depth (exactly what JD wants)
        interactions["ix_ml_production"] = (
            features.get("career_ml_maturity", 0)
            * features.get("career_production_depth", 0)
        )

        # Search x Ranking depth
        interactions["ix_search_depth"] = (
            features.get("career_search_maturity", 0)
            + features.get("career_recsys_maturity", 0)
        ) / 2.0

        # Ontology coverage x Proficiency
        interactions["ix_ontology_proficiency"] = (
            features.get("ontology_coverage_ratio", 0)
            * features.get("skill_proficiency_score", 0)
        )

        # Availability x Skill fit (great candidate who is available)
        interactions["ix_available_skilled"] = (
            features.get("bi_availability_score", 0)
            * features.get("skill_combined_score", 0)
        )

        # Anti-honeypot x Skill (clean profile with skills)
        interactions["ix_clean_skilled"] = (
            features.get("honeypot_penalty_multiplier", 1.0)
            * features.get("skill_combined_score", 0)
        )

        # Product company x ML maturity
        interactions["ix_product_ml"] = (
            features.get("company_product_ratio", 0)
            * features.get("career_ml_maturity", 0)
        )

        # Role consistency x Experience fit
        interactions["ix_consistency_exp"] = (
            features.get("career_role_consistency", 0)
            * features.get("experience_fit_score", 0)
        )

        # Evidence-based interactions (new)
        # Evidence search/ranking x Product company (exactly what JD wants)
        interactions["ix_evidence_product"] = (
            features.get("evidence_jd_alignment", 0)
            * features.get("company_product_ratio", 0)
        )

        # Evidence x Skills (description confirms skill claims)
        interactions["ix_evidence_skills"] = (
            features.get("evidence_total_count", 0)
            * features.get("skill_combined_score", 0)
        )

        # Evidence search + ranking combined strength
        interactions["ix_evidence_search_rank"] = (
            features.get("evidence_search_retrieval", 0)
            + features.get("evidence_ranking", 0)
        ) / 2.0

        return interactions

    def build_batch_features(
        self, candidates: list[Candidate], jd: ParsedJD
    ) -> list[dict[str, float]]:
        """Build features for a batch of candidates.

        Args:
            candidates: List of candidates.
            jd: Parsed job description.

        Returns:
            List of feature dictionaries.
        """
        with timer(f"Building features for {len(candidates)} candidates"):
            return [self.build_features(c, jd) for c in candidates]

    def get_feature_names(self) -> list[str]:
        """Get ordered list of all feature names.

        Returns:
            List of feature column names.
        """
        return [
            # Skill features
            "skill_must_have_score",
            "skill_preferred_score",
            "skill_related_score",
            "skill_proficiency_score",
            "skill_endorsement_score",
            "skill_duration_score",
            "skill_assessment_score",
            "skill_coverage_score",
            "skill_combined_score",
            # Ontology features
            "ontology_coverage_ratio",
            "ontology_avg_match_score",
            "ontology_group_coverage",
            "ontology_max_match_score",
            # Career features
            "career_title_score",
            "career_industry_score",
            "career_company_type_score",
            "career_stability_score",
            "career_progression_score",
            "career_consulting_penalty",
            "career_domain_penalty",
            "career_product_company_score",
            "career_combined_score",
            # Career progression (advanced)
            "career_promotion_velocity",
            "career_avg_tenure_months",
            "career_stability_v2",
            "career_ml_maturity",
            "career_search_maturity",
            "career_nlp_maturity",
            "career_recsys_maturity",
            "career_production_depth",
            "career_leadership_score",
            "career_role_consistency",
            "career_technical_depth",
            "career_ml_production_combined",
            # Company intelligence
            "company_product_ratio",
            "company_consulting_ratio",
            "company_avg_quality",
            "company_best_tier",
            "company_current_quality",
            "company_startup_bonus",
            "company_tier1_score",
            "company_consulting_penalty",
            # Experience features
            "experience_fit_score",
            "experience_consistency_score",
            "experience_relevant_ratio",
            "experience_years",
            "experience_combined_score",
            # Behavioral features
            "behavioral_response_rate",
            "behavioral_recency",
            "behavioral_open_to_work",
            "behavioral_interview_completion",
            "behavioral_completeness",
            "behavioral_github",
            "behavioral_notice_period",
            "behavioral_search_visibility",
            "behavioral_saved_by_recruiters",
            "behavioral_offer_acceptance",
            "behavioral_verified",
            "behavioral_response_time",
            "behavioral_combined_score",
            # Behavioral intelligence (advanced)
            "bi_availability_score",
            "bi_recruitability_score",
            "bi_engagement_score",
            "bi_reliability_score",
            "bi_market_signal_score",
            "bi_offer_probability",
            "bi_response_quality",
            "bi_platform_trust",
            "bi_combined_score",
            # Honeypot features
            "honeypot_score",
            "honeypot_timeline_flag",
            "honeypot_skill_stuffing_flag",
            "honeypot_experience_mismatch_flag",
            "honeypot_title_desc_mismatch_flag",
            "honeypot_proficiency_flag",
            "honeypot_inconsistency_flag",
            "honeypot_penalty_multiplier",
            # Interaction features
            "ix_skill_career",
            "ix_skill_experience",
            "ix_career_behavioral",
            "ix_ml_production",
            "ix_search_depth",
            "ix_ontology_proficiency",
            "ix_available_skilled",
            "ix_clean_skilled",
            "ix_product_ml",
            "ix_consistency_exp",
            "ix_evidence_product",
            "ix_evidence_skills",
            "ix_evidence_search_rank",
            # Career evidence features
            "evidence_search_retrieval",
            "evidence_ranking",
            "evidence_recommendation",
            "evidence_embeddings",
            "evidence_nlp",
            "evidence_llm",
            "evidence_production_ml",
            "evidence_scale",
            "evidence_total_count",
            "evidence_jd_alignment",
        ]

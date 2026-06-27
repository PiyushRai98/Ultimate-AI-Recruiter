"""Honeypot candidate detection engine.

Detects candidates with subtly impossible profiles such as:
- 8 years experience at a company founded 3 years ago
- Expert proficiency in 10 skills with 0 duration
- Skill stuffing with no endorsements or assessments
- Career timeline inconsistencies
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from src.config.loader import get_config
from src.models.candidate import Candidate


class HoneypotDetector:
    """Detects anomalous/honeypot candidate profiles.

    Honeypots are forced to relevance tier 0 in ground truth.
    Submissions with honeypot rate > 10% in top 100 are disqualified.
    """

    def __init__(self) -> None:
        """Initialize with detection thresholds from config."""
        config = get_config()
        hp_config = config.get("weights", "honeypot", {})
        self._max_skills = hp_config.get("max_skill_count_threshold", 15)
        self._zero_endorsement_threshold = hp_config.get(
            "zero_endorsement_expert_threshold", 3
        )
        self._keyword_stuffing_ratio = hp_config.get(
            "keyword_stuffing_ratio_threshold", 0.8
        )

    def compute_honeypot_score(self, candidate: Candidate) -> dict[str, float]:
        """Compute honeypot detection features.

        Args:
            candidate: Candidate to analyze.

        Returns:
            Dictionary with honeypot-related features.
            Higher scores indicate more likely honeypot.
        """
        flags: list[float] = []

        # 1. Timeline impossibility
        timeline_flag = self._check_timeline_impossibility(candidate)
        flags.append(timeline_flag)

        # 2. Skill stuffing detection
        skill_stuffing_flag = self._check_skill_stuffing(candidate)
        flags.append(skill_stuffing_flag)

        # 3. Experience vs career mismatch
        mismatch_flag = self._check_experience_mismatch(candidate)
        flags.append(mismatch_flag)

        # 4. Title-description mismatch (career entries)
        title_desc_flag = self._check_title_description_mismatch(candidate)
        flags.append(title_desc_flag)

        # 5. Impossible proficiency claims
        proficiency_flag = self._check_impossible_proficiency(candidate)
        flags.append(proficiency_flag)

        # 6. Profile inconsistency
        inconsistency_flag = self._check_profile_inconsistency(candidate)
        flags.append(inconsistency_flag)

        # Overall honeypot score (0 = clean, 1 = definite honeypot)
        honeypot_score = sum(flags) / len(flags) if flags else 0.0

        # Penalty multiplier (1.0 = no penalty, 0.0 = full penalty)
        if honeypot_score >= 0.6:
            penalty = 0.0  # Strong honeypot signal - zero out
        elif honeypot_score >= 0.4:
            penalty = 0.3
        elif honeypot_score >= 0.25:
            penalty = 0.6
        else:
            penalty = 1.0  # Clean profile

        return {
            "honeypot_score": honeypot_score,
            "honeypot_timeline_flag": timeline_flag,
            "honeypot_skill_stuffing_flag": skill_stuffing_flag,
            "honeypot_experience_mismatch_flag": mismatch_flag,
            "honeypot_title_desc_mismatch_flag": title_desc_flag,
            "honeypot_proficiency_flag": proficiency_flag,
            "honeypot_inconsistency_flag": inconsistency_flag,
            "honeypot_penalty_multiplier": penalty,
        }

    def _check_timeline_impossibility(self, candidate: Candidate) -> float:
        """Check for impossible timeline in career history."""
        flags = 0
        checks = 0

        for entry in candidate.career_history:
            checks += 1
            try:
                start = datetime.strptime(entry.start_date, "%Y-%m-%d")
                if entry.end_date:
                    end = datetime.strptime(entry.end_date, "%Y-%m-%d")
                    actual_months = (end - start).days / 30.44

                    # Duration claims much larger than actual timespan
                    if entry.duration_months > actual_months * 1.5 + 6:
                        flags += 1
                else:
                    # Current role - check stated duration vs start date
                    now = datetime(2026, 6, 15)
                    actual_months = (now - start).days / 30.44
                    if entry.duration_months > actual_months * 1.5 + 6:
                        flags += 1
            except (ValueError, TypeError):
                pass

        return flags / max(checks, 1)

    def _check_skill_stuffing(self, candidate: Candidate) -> float:
        """Detect skill keyword stuffing."""
        skills = candidate.skills
        if not skills:
            return 0.0

        # AI/ML keywords that could be stuffed
        ai_keywords = {
            "nlp", "machine learning", "deep learning", "pytorch",
            "tensorflow", "transformers", "embeddings", "llm",
            "rag", "fine-tuning", "gpt", "bert", "ranking",
            "retrieval", "recommendation", "search", "faiss",
            "vector", "neural", "reinforcement learning",
            "computer vision", "generative ai", "langchain",
        }

        skill_names_lower = [s.name.lower() for s in skills]
        ai_skill_count = sum(
            1 for name in skill_names_lower
            if any(kw in name for kw in ai_keywords)
        )

        # If most skills are AI keywords AND most have low endorsements
        total_skills = len(skills)
        if total_skills > 0:
            ai_ratio = ai_skill_count / total_skills

            # Check endorsement credibility
            zero_endorsement_ai = sum(
                1 for s in skills
                if any(kw in s.name.lower() for kw in ai_keywords)
                and s.endorsements == 0
            )

            # Many AI skills with no endorsements = stuffing
            if ai_skill_count > 8 and zero_endorsement_ai > ai_skill_count * 0.7:
                return 0.9

            if ai_ratio > self._keyword_stuffing_ratio and total_skills > 10:
                return 0.7

        return 0.0

    def _check_experience_mismatch(self, candidate: Candidate) -> float:
        """Check if stated years doesn't match career history."""
        total_career_months = sum(
            entry.duration_months for entry in candidate.career_history
        )
        stated_months = candidate.years_of_experience * 12

        if stated_months > 0:
            ratio = total_career_months / stated_months
            # Major mismatch (career says way less than stated)
            if ratio < 0.3:
                return 0.8
            elif ratio < 0.5:
                return 0.4
            # Or way more career months than stated years
            elif ratio > 2.5:
                return 0.6

        return 0.0

    def _check_title_description_mismatch(self, candidate: Candidate) -> float:
        """Check if job descriptions don't match job titles."""
        mismatches = 0
        checks = 0

        title_domain_map = {
            "marketing": ["marketing", "brand", "campaign", "seo", "content"],
            "engineer": ["code", "system", "build", "develop", "implement", "pipeline"],
            "accountant": ["accounting", "financial", "audit", "tax", "ledger"],
            "hr": ["recruit", "hire", "employee", "talent", "people"],
            "sales": ["revenue", "client", "deal", "quota", "pipeline"],
        }

        for entry in candidate.career_history:
            checks += 1
            title_lower = entry.title.lower()
            desc_lower = entry.description.lower()

            for title_keyword, desc_keywords in title_domain_map.items():
                if title_keyword in title_lower:
                    # Title is in this domain - check if description matches
                    desc_matches = sum(
                        1 for kw in desc_keywords if kw in desc_lower
                    )
                    if desc_matches == 0:
                        # Description doesn't mention anything related to title domain
                        mismatches += 1
                    break

        return mismatches / max(checks, 1)

    def _check_impossible_proficiency(self, candidate: Candidate) -> float:
        """Check for impossible proficiency claims."""
        flags = 0
        checks = 0

        for skill in candidate.skills:
            if skill.proficiency in ("expert", "advanced"):
                checks += 1
                # Expert/advanced with 0 duration months
                if skill.duration_months == 0:
                    flags += 1
                # Expert with very low duration
                elif skill.proficiency == "expert" and skill.duration_months < 6:
                    flags += 1
                # Advanced/Expert with 0 endorsements
                elif skill.endorsements == 0 and skill.duration_months < 12:
                    flags += 0.5

        if checks == 0:
            return 0.0
        return min(flags / checks, 1.0)

    def _check_profile_inconsistency(self, candidate: Candidate) -> float:
        """Check for overall profile inconsistencies."""
        flags = 0.0

        # Non-tech title but heavy AI skills
        non_tech_titles = [
            "marketing", "accountant", "hr manager", "sales",
            "operations manager", "content writer", "graphic designer",
            "customer support", "civil engineer", "mechanical engineer",
        ]

        current_title_lower = candidate.current_title.lower()
        is_non_tech = any(t in current_title_lower for t in non_tech_titles)

        if is_non_tech:
            # Count AI/ML skills
            ai_keywords = {"ml", "nlp", "deep learning", "pytorch", "tensorflow",
                           "embeddings", "llm", "transformers", "faiss"}
            ai_skill_count = sum(
                1 for s in candidate.skills
                if any(kw in s.name.lower() for kw in ai_keywords)
            )
            if ai_skill_count > 5:
                flags += 0.7  # Non-tech title with many AI skills = suspicious

        # Headline doesn't match current title
        headline_lower = candidate.headline.lower()
        if candidate.current_title.lower() not in headline_lower:
            # Not necessarily a flag, but combined with other signals...
            pass

        return min(flags, 1.0)

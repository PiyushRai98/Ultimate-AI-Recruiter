"""Experience level scoring relative to JD requirements."""

from __future__ import annotations

from src.config.loader import get_config
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD


class ExperienceScorer:
    """Scores candidates based on years of experience fit.

    The JD specifies 5-9 years as ideal range but acknowledges
    strong candidates outside the band.
    """

    def __init__(self) -> None:
        """Initialize with experience configuration."""
        config = get_config()
        exp_config = config.get("weights", "experience", {})
        self._ideal_min = exp_config.get("ideal_min_years", 5)
        self._ideal_max = exp_config.get("ideal_max_years", 9)
        self._extended_max = exp_config.get("extended_max_years", 12)
        self._penalty_under = exp_config.get("penalty_per_year_under", 0.08)
        self._penalty_over = exp_config.get("penalty_per_year_over", 0.04)

    def score(self, candidate: Candidate, jd: ParsedJD) -> dict[str, float]:
        """Compute experience-fit scores.

        Args:
            candidate: Candidate to evaluate.
            jd: Parsed job description.

        Returns:
            Dictionary of experience-related scores.
        """
        yoe = candidate.years_of_experience

        # Core experience fit score
        if self._ideal_min <= yoe <= self._ideal_max:
            experience_fit = 1.0
        elif yoe < self._ideal_min:
            deficit = self._ideal_min - yoe
            experience_fit = max(0.0, 1.0 - deficit * self._penalty_under)
        elif yoe <= self._extended_max:
            # Slightly over is okay
            excess = yoe - self._ideal_max
            experience_fit = max(0.5, 1.0 - excess * self._penalty_over)
        else:
            # Way over - probably too senior
            excess = yoe - self._ideal_max
            experience_fit = max(0.2, 1.0 - excess * self._penalty_over * 1.5)

        # Career history validates stated experience
        total_career_months = sum(
            entry.duration_months for entry in candidate.career_history
        )
        stated_months = yoe * 12

        # Check for consistency (honeypot signal if large mismatch)
        if stated_months > 0:
            consistency = min(total_career_months / stated_months, 1.5)
            # Penalize if career months significantly less than stated
            if consistency < 0.5:
                experience_consistency = 0.3
            elif consistency < 0.7:
                experience_consistency = 0.6
            else:
                experience_consistency = 1.0
        else:
            experience_consistency = 0.5

        # Relevant experience (in tech/AI roles specifically)
        relevant_months = self._compute_relevant_months(candidate)
        relevant_ratio = relevant_months / max(total_career_months, 1)

        return {
            "experience_fit_score": experience_fit,
            "experience_consistency_score": experience_consistency,
            "experience_relevant_ratio": relevant_ratio,
            "experience_years": yoe,
            "experience_combined_score": (
                experience_fit * 0.6
                + experience_consistency * 0.2
                + relevant_ratio * 0.2
            ),
        }

    def _compute_relevant_months(self, candidate: Candidate) -> int:
        """Compute months in relevant roles (tech/AI/ML/software)."""
        relevant_keywords = [
            "engineer", "developer", "scientist", "ml", "ai",
            "machine learning", "data", "software", "backend",
            "platform", "search", "nlp", "research",
        ]

        relevant_months = 0
        for entry in candidate.career_history:
            title_lower = entry.title.lower()
            desc_lower = entry.description.lower()

            is_relevant = any(kw in title_lower for kw in relevant_keywords)
            if not is_relevant:
                # Check description for ML/AI work
                desc_keywords = ["model", "ml", "machine learning", "embeddings",
                                 "ranking", "retrieval", "nlp", "pipeline"]
                is_relevant = any(kw in desc_lower for kw in desc_keywords)

            if is_relevant:
                relevant_months += entry.duration_months

        return relevant_months

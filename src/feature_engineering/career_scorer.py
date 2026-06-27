"""Career history scoring engine."""

from __future__ import annotations

from rapidfuzz import fuzz, process
from loguru import logger

from src.config.loader import get_config
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD


class CareerScorer:
    """Scores candidates based on career trajectory and fitness.

    Evaluates title relevance, industry fit, company quality,
    career stability, and progression patterns.
    """

    def __init__(self) -> None:
        """Initialize career scorer with configuration."""
        config = get_config()
        self._ranking_config = config.ranking
        self._features_config = config.features

        # Load title relevance categories
        title_config = self._features_config.get("relevant_titles", {})
        self._high_titles = [t.lower() for t in title_config.get("high_relevance", [])]
        self._medium_titles = [t.lower() for t in title_config.get("medium_relevance", [])]
        self._low_titles = [t.lower() for t in title_config.get("low_relevance", [])]

        # Negative signals
        neg_signals = self._ranking_config.get("negative_signals", {})
        self._consulting_companies = [
            c.lower() for c in neg_signals.get("consulting_only_companies", [])
        ]
        self._non_relevant_domains = [
            d.lower() for d in neg_signals.get("non_relevant_domains", [])
        ]

        # Industry relevance
        industry_config = self._features_config.get("relevant_industries", {})
        self._high_industries = [i.lower() for i in industry_config.get("high", [])]
        self._medium_industries = [i.lower() for i in industry_config.get("medium", [])]

    def score(self, candidate: Candidate, jd: ParsedJD) -> dict[str, float]:
        """Compute career-fit feature scores.

        Args:
            candidate: Candidate to evaluate.
            jd: Parsed job description.

        Returns:
            Dictionary of career-related feature scores.
        """
        title_score = self._score_title_relevance(candidate)
        industry_score = self._score_industry_fit(candidate)
        company_type_score = self._score_company_type(candidate)
        stability_score = self._score_career_stability(candidate)
        progression_score = self._score_career_progression(candidate)
        consulting_penalty = self._compute_consulting_penalty(candidate)
        domain_penalty = self._compute_domain_penalty(candidate)
        product_company_score = self._score_product_company_experience(candidate)

        # Combined career score
        combined = (
            title_score * 0.30
            + product_company_score * 0.20
            + industry_score * 0.15
            + stability_score * 0.10
            + progression_score * 0.10
            + company_type_score * 0.05
            - consulting_penalty * 0.05
            - domain_penalty * 0.05
        )
        combined = max(0.0, min(1.0, combined))

        return {
            "career_title_score": title_score,
            "career_industry_score": industry_score,
            "career_company_type_score": company_type_score,
            "career_stability_score": stability_score,
            "career_progression_score": progression_score,
            "career_consulting_penalty": consulting_penalty,
            "career_domain_penalty": domain_penalty,
            "career_product_company_score": product_company_score,
            "career_combined_score": combined,
        }

    def _score_title_relevance(self, candidate: Candidate) -> float:
        """Score based on how relevant the candidate's titles are."""
        titles = [t.lower() for t in candidate.get_all_titles()]
        current_title = candidate.current_title.lower()

        best_score = 0.0

        # Score current title (highest weight)
        current_score = self._match_title(current_title)
        best_score = max(best_score, current_score)

        # Score historical titles with decay
        for i, title in enumerate(titles):
            title_score = self._match_title(title)
            # Recent roles matter more
            decay = 1.0 / (1 + i * 0.3)
            best_score = max(best_score, title_score * decay)

        return best_score

    def _match_title(self, title: str) -> float:
        """Match a single title against relevance categories."""
        # High relevance check
        for ref_title in self._high_titles:
            if fuzz.token_sort_ratio(title, ref_title) >= 80:
                return 1.0
            if ref_title in title or title in ref_title:
                return 0.95

        # Medium relevance check
        for ref_title in self._medium_titles:
            if fuzz.token_sort_ratio(title, ref_title) >= 80:
                return 0.6
            if ref_title in title or title in ref_title:
                return 0.55

        # Low relevance check
        for ref_title in self._low_titles:
            if fuzz.token_sort_ratio(title, ref_title) >= 80:
                return 0.3

        # Check for AI/ML keywords in title
        ai_keywords = ["ai", "ml", "machine learning", "data scien", "nlp", "search", "ranking"]
        for kw in ai_keywords:
            if kw in title:
                return 0.7

        return 0.0

    def _score_industry_fit(self, candidate: Candidate) -> float:
        """Score based on industry relevance."""
        industries = set()
        industries.add(candidate.current_industry.lower())
        for entry in candidate.career_history:
            industries.add(entry.industry.lower())

        best_score = 0.0
        for ind in industries:
            for high_ind in self._high_industries:
                if high_ind in ind or ind in high_ind:
                    best_score = max(best_score, 1.0)
            for med_ind in self._medium_industries:
                if med_ind in ind or ind in med_ind:
                    best_score = max(best_score, 0.6)
            if "software" in ind or "technology" in ind or "internet" in ind:
                best_score = max(best_score, 0.8)

        return best_score

    def _score_company_type(self, candidate: Candidate) -> float:
        """Score based on company size and type fit.

        The JD prefers product company experience at meaningful scale.
        """
        preferred_sizes = {"201-500", "501-1000", "1001-5000", "5001-10000"}
        large_sizes = {"10001+"}

        has_preferred = False
        has_large = False

        for entry in candidate.career_history:
            if entry.company_size in preferred_sizes:
                has_preferred = True
            if entry.company_size in large_sizes:
                has_large = True

        if has_preferred:
            return 0.8
        if has_large:
            return 0.5  # Large = likely services company
        return 0.3

    def _score_career_stability(self, candidate: Candidate) -> float:
        """Score career stability based on tenure patterns.

        The JD explicitly penalizes title-chasers who switch every 1.5 years.
        """
        if len(candidate.career_history) <= 1:
            return 0.7  # Neutral for single-job candidates

        durations = [entry.duration_months for entry in candidate.career_history]
        avg_tenure = sum(durations) / len(durations)

        # Short average tenure is penalized
        if avg_tenure < 18:
            return 0.2  # Title-chaser signal
        elif avg_tenure < 24:
            return 0.4
        elif avg_tenure < 36:
            return 0.7
        elif avg_tenure < 60:
            return 0.9
        else:
            return 1.0

    def _score_career_progression(self, candidate: Candidate) -> float:
        """Score career progression and growth patterns."""
        if len(candidate.career_history) < 2:
            return 0.5

        # Check for title progression (seniority increase)
        seniority_keywords = {
            "intern": 0, "junior": 1, "associate": 2,
            "": 3, "senior": 4, "lead": 5,
            "staff": 6, "principal": 7, "director": 8,
        }

        levels = []
        for entry in candidate.career_history:
            title_lower = entry.title.lower()
            level = 3  # default mid-level
            for kw, lv in seniority_keywords.items():
                if kw and kw in title_lower:
                    level = lv
                    break
            levels.append(level)

        # Progression: levels should generally increase (reversed since career_history is newest-first)
        if len(levels) >= 2:
            # Count upward moves
            upward = sum(1 for i in range(len(levels) - 1) if levels[i] > levels[i + 1])
            total_moves = len(levels) - 1
            return min(upward / total_moves + 0.3, 1.0) if total_moves > 0 else 0.5

        return 0.5

    def _compute_consulting_penalty(self, candidate: Candidate) -> float:
        """Compute penalty for consulting-only career paths.

        The JD explicitly says consulting-only careers are a bad fit.
        """
        companies = [entry.company.lower() for entry in candidate.career_history]
        total = len(companies)
        if total == 0:
            return 0.0

        consulting_count = sum(
            1 for c in companies
            if any(cons in c for cons in self._consulting_companies)
        )

        ratio = consulting_count / total
        if ratio >= 1.0:
            return 1.0  # All consulting = heavy penalty
        elif ratio >= 0.7:
            return 0.7
        elif ratio >= 0.5:
            return 0.4
        return 0.0

    def _compute_domain_penalty(self, candidate: Candidate) -> float:
        """Compute penalty for non-relevant domain experience."""
        current_title = candidate.current_title.lower()

        for domain in self._non_relevant_domains:
            if domain in current_title:
                return 0.8

        # Check if career descriptions mention relevant work
        return 0.0

    def _score_product_company_experience(self, candidate: Candidate) -> float:
        """Score based on product company vs services company experience.

        The JD strongly prefers product company experience.
        """
        product_months = 0
        total_months = 0

        for entry in candidate.career_history:
            total_months += entry.duration_months
            company_lower = entry.company.lower()

            # If not a known consulting company, assume product
            is_consulting = any(
                cons in company_lower for cons in self._consulting_companies
            )
            if not is_consulting:
                # Check if industry suggests product company
                industry_lower = entry.industry.lower()
                if industry_lower not in ["it services", "consulting"]:
                    product_months += entry.duration_months
                else:
                    product_months += entry.duration_months * 0.3

        if total_months == 0:
            return 0.0

        return min(product_months / total_months, 1.0)

"""Behavioral signals scoring engine."""

from __future__ import annotations

from datetime import date, datetime

from loguru import logger

from src.config.loader import get_config
from src.models.candidate import Candidate


class BehavioralScorer:
    """Scores candidates based on Redrob platform behavioral signals.

    Behavioral signals indicate availability, engagement, and likelihood
    of being hireable. A perfect-on-paper candidate who hasn't logged in
    for 6 months with 5% response rate is not actually available.
    """

    # Reference date for recency calculations
    REFERENCE_DATE = date(2026, 6, 15)

    def __init__(self) -> None:
        """Initialize behavioral scorer with configuration."""
        config = get_config()
        self._weights = config.get("weights", "behavioral", {})

    def score(self, candidate: Candidate) -> dict[str, float]:
        """Compute behavioral signal scores.

        Args:
            candidate: Candidate to evaluate.

        Returns:
            Dictionary of behavioral feature scores.
        """
        signals = candidate.signals

        # Response rate (most critical - can they be reached?)
        response_score = self._score_response_rate(signals.recruiter_response_rate)

        # Recency (are they active on platform?)
        recency_score = self._score_recency(signals.last_active_date)

        # Open to work (explicit signal)
        open_to_work_score = 1.0 if signals.open_to_work_flag else 0.3

        # Interview completion (do they follow through?)
        interview_score = self._score_interview_completion(
            signals.interview_completion_rate
        )

        # Profile completeness (how invested are they?)
        completeness_score = signals.profile_completeness_score / 100.0

        # GitHub activity
        github_score = self._score_github(signals.github_activity_score)

        # Notice period
        notice_score = self._score_notice_period(signals.notice_period_days)

        # Search visibility (demand signal)
        search_score = min(signals.search_appearance_30d / 200.0, 1.0)

        # Saved by recruiters (quality signal)
        saved_score = min(signals.saved_by_recruiters_30d / 10.0, 1.0)

        # Offer acceptance
        offer_score = self._score_offer_acceptance(signals.offer_acceptance_rate)

        # Verification signals
        verified_count = sum([
            signals.verified_email,
            signals.verified_phone,
            signals.linkedin_connected,
        ])
        verified_score = verified_count / 3.0

        # Response time score
        response_time_score = self._score_response_time(
            signals.avg_response_time_hours
        )

        # Weighted combination
        combined = (
            response_score * self._weights.get("recruiter_response_rate", 0.20)
            + recency_score * self._weights.get("last_active_recency", 0.15)
            + open_to_work_score * self._weights.get("open_to_work", 0.12)
            + interview_score * self._weights.get("interview_completion_rate", 0.10)
            + completeness_score * self._weights.get("profile_completeness", 0.08)
            + github_score * self._weights.get("github_activity", 0.08)
            + notice_score * self._weights.get("notice_period", 0.07)
            + search_score * self._weights.get("search_appearance", 0.05)
            + saved_score * self._weights.get("saved_by_recruiters", 0.05)
            + offer_score * self._weights.get("offer_acceptance_rate", 0.05)
            + verified_score * self._weights.get("verified_signals", 0.03)
            + response_time_score * self._weights.get("avg_response_time", 0.02)
        )

        return {
            "behavioral_response_rate": response_score,
            "behavioral_recency": recency_score,
            "behavioral_open_to_work": open_to_work_score,
            "behavioral_interview_completion": interview_score,
            "behavioral_completeness": completeness_score,
            "behavioral_github": github_score,
            "behavioral_notice_period": notice_score,
            "behavioral_search_visibility": search_score,
            "behavioral_saved_by_recruiters": saved_score,
            "behavioral_offer_acceptance": offer_score,
            "behavioral_verified": verified_score,
            "behavioral_response_time": response_time_score,
            "behavioral_combined_score": combined,
        }

    def _score_response_rate(self, rate: float) -> float:
        """Score recruiter response rate."""
        # Non-linear: low response rate is a strong negative
        if rate >= 0.7:
            return 1.0
        elif rate >= 0.5:
            return 0.8
        elif rate >= 0.3:
            return 0.5
        elif rate >= 0.15:
            return 0.3
        else:
            return 0.1

    def _score_recency(self, last_active: str) -> float:
        """Score based on how recently candidate was active."""
        try:
            active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (self.REFERENCE_DATE - active_date).days

            if days_inactive < 0:
                return 1.0  # Future date = very active
            elif days_inactive <= 7:
                return 1.0
            elif days_inactive <= 30:
                return 0.9
            elif days_inactive <= 90:
                return 0.7
            elif days_inactive <= 180:
                return 0.4
            else:
                return 0.1
        except (ValueError, TypeError):
            return 0.3

    def _score_interview_completion(self, rate: float) -> float:
        """Score interview completion rate."""
        if rate >= 0.9:
            return 1.0
        elif rate >= 0.7:
            return 0.8
        elif rate >= 0.5:
            return 0.5
        else:
            return 0.2

    def _score_github(self, score: float) -> float:
        """Score GitHub activity. -1 means no GitHub linked."""
        if score < 0:
            return 0.3  # No GitHub is neutral, not penalizing
        return score / 100.0

    def _score_notice_period(self, days: int) -> float:
        """Score notice period preference.

        JD prefers sub-30 days. Can buy out up to 30 days.
        30+ day candidates are in scope but bar is higher.
        """
        if days <= 30:
            return 1.0
        elif days <= 60:
            return 0.7
        elif days <= 90:
            return 0.4
        else:
            return 0.2

    def _score_offer_acceptance(self, rate: float) -> float:
        """Score offer acceptance rate. -1 means no history."""
        if rate < 0:
            return 0.5  # No history is neutral
        if rate >= 0.8:
            return 1.0
        elif rate >= 0.5:
            return 0.7
        else:
            return 0.3

    def _score_response_time(self, hours: float) -> float:
        """Score average response time."""
        if hours <= 4:
            return 1.0
        elif hours <= 24:
            return 0.8
        elif hours <= 72:
            return 0.6
        elif hours <= 168:
            return 0.3
        else:
            return 0.1

"""Advanced Behavioral Intelligence Engine.

Computes composite behavioral scores that go beyond individual signals
to model higher-level concepts like recruitability and reliability.
"""

from __future__ import annotations

from datetime import date, datetime

from loguru import logger

from src.models.candidate import Candidate


class BehavioralIntelligence:
    """Computes advanced behavioral intelligence features.

    Models:
    - Availability: Can this person actually be hired right now?
    - Recruitability: Will they respond to outreach?
    - Engagement: Are they actively participating in the job market?
    - Reliability: Will they show up and follow through?
    - Market Signal: Does the market validate this candidate?
    """

    REFERENCE_DATE = date(2026, 6, 15)

    def compute_features(self, candidate: Candidate) -> dict[str, float]:
        """Compute behavioral intelligence features.

        Args:
            candidate: Candidate to analyze.

        Returns:
            Dictionary of advanced behavioral features.
        """
        signals = candidate.signals

        features: dict[str, float] = {}

        # Availability Score: Can they actually be hired?
        features["bi_availability_score"] = self._compute_availability(signals)

        # Recruitability Score: Will they respond?
        features["bi_recruitability_score"] = self._compute_recruitability(signals)

        # Engagement Score: Are they active in the market?
        features["bi_engagement_score"] = self._compute_engagement(signals)

        # Reliability Score: Will they follow through?
        features["bi_reliability_score"] = self._compute_reliability(signals)

        # Market Signal Score: Does the market want them?
        features["bi_market_signal_score"] = self._compute_market_signal(signals)

        # Offer Probability: Likelihood of accepting an offer
        features["bi_offer_probability"] = self._compute_offer_probability(signals)

        # Response Quality: Quality of engagement
        features["bi_response_quality"] = self._compute_response_quality(signals)

        # Platform Trust: How much can we trust their profile?
        features["bi_platform_trust"] = self._compute_platform_trust(signals)

        # Combined behavioral intelligence score
        features["bi_combined_score"] = (
            features["bi_availability_score"] * 0.25
            + features["bi_recruitability_score"] * 0.25
            + features["bi_engagement_score"] * 0.15
            + features["bi_reliability_score"] * 0.15
            + features["bi_market_signal_score"] * 0.10
            + features["bi_offer_probability"] * 0.05
            + features["bi_platform_trust"] * 0.05
        )

        return features

    def _compute_availability(self, signals) -> float:
        """Compute availability score.

        Combines: open_to_work, notice_period, recency, willingness to relocate.
        """
        score = 0.0

        # Open to work is the strongest signal
        if signals.open_to_work_flag:
            score += 0.35
        else:
            score += 0.1

        # Notice period
        if signals.notice_period_days <= 30:
            score += 0.30
        elif signals.notice_period_days <= 60:
            score += 0.20
        elif signals.notice_period_days <= 90:
            score += 0.10
        else:
            score += 0.0

        # Recency
        try:
            active_date = datetime.strptime(signals.last_active_date, "%Y-%m-%d").date()
            days_inactive = (self.REFERENCE_DATE - active_date).days
            if days_inactive <= 14:
                score += 0.20
            elif days_inactive <= 30:
                score += 0.15
            elif days_inactive <= 90:
                score += 0.10
            elif days_inactive <= 180:
                score += 0.03
        except (ValueError, TypeError):
            score += 0.05

        # Willingness to relocate (JD is Pune/Noida)
        if signals.willing_to_relocate:
            score += 0.15
        else:
            score += 0.05

        return min(score, 1.0)

    def _compute_recruitability(self, signals) -> float:
        """Compute recruitability score.

        Combines: response_rate, response_time, verified contact.
        """
        # Response rate is the strongest signal
        rr = signals.recruiter_response_rate
        if rr >= 0.7:
            rate_score = 1.0
        elif rr >= 0.5:
            rate_score = 0.75
        elif rr >= 0.3:
            rate_score = 0.5
        elif rr >= 0.15:
            rate_score = 0.25
        else:
            rate_score = 0.05

        # Response time
        rt = signals.avg_response_time_hours
        if rt <= 4:
            time_score = 1.0
        elif rt <= 12:
            time_score = 0.85
        elif rt <= 24:
            time_score = 0.7
        elif rt <= 72:
            time_score = 0.4
        elif rt <= 168:
            time_score = 0.2
        else:
            time_score = 0.05

        # Verification (reachable?)
        verified_count = sum([
            signals.verified_email,
            signals.verified_phone,
            signals.linkedin_connected,
        ])
        verify_score = verified_count / 3.0

        return rate_score * 0.5 + time_score * 0.3 + verify_score * 0.2

    def _compute_engagement(self, signals) -> float:
        """Compute engagement score.

        Combines: applications, profile completeness, assessments, connections.
        """
        # Profile completeness shows investment
        completeness = signals.profile_completeness_score / 100.0

        # Applications show active job seeking
        apps = signals.applications_submitted_30d
        app_score = min(apps / 5.0, 1.0) if apps > 0 else 0.2

        # Skill assessments show proactive behavior
        assessments = signals.skill_assessment_scores
        assessment_score = min(len(assessments) / 4.0, 1.0) if assessments else 0.0

        # Connections indicate networking
        conn_score = min(signals.connection_count / 500.0, 1.0)

        return (
            completeness * 0.30
            + app_score * 0.25
            + assessment_score * 0.25
            + conn_score * 0.20
        )

    def _compute_reliability(self, signals) -> float:
        """Compute reliability score.

        Combines: interview_completion, offer_acceptance, recency consistency.
        """
        # Interview completion is key
        interview = signals.interview_completion_rate
        if interview >= 0.9:
            interview_score = 1.0
        elif interview >= 0.7:
            interview_score = 0.7
        elif interview >= 0.5:
            interview_score = 0.4
        else:
            interview_score = 0.15

        # Offer acceptance history
        offer_rate = signals.offer_acceptance_rate
        if offer_rate < 0:
            offer_score = 0.5  # No history
        elif offer_rate >= 0.8:
            offer_score = 1.0
        elif offer_rate >= 0.5:
            offer_score = 0.6
        else:
            offer_score = 0.2

        return interview_score * 0.6 + offer_score * 0.4

    def _compute_market_signal(self, signals) -> float:
        """Compute market signal score.

        Combines: search_appearances, saved_by_recruiters, profile_views.
        """
        # Search appearances (demand indicator)
        search = min(signals.search_appearance_30d / 200.0, 1.0)

        # Saved by recruiters (quality indicator)
        saved = min(signals.saved_by_recruiters_30d / 10.0, 1.0)

        # Profile views (interest indicator)
        views = min(signals.profile_views_received_30d / 30.0, 1.0)

        return search * 0.35 + saved * 0.40 + views * 0.25

    def _compute_offer_probability(self, signals) -> float:
        """Estimate probability of accepting an offer if extended."""
        availability = 0.5

        if signals.open_to_work_flag:
            availability += 0.2
        if signals.notice_period_days <= 30:
            availability += 0.15
        if signals.willing_to_relocate:
            availability += 0.1

        offer_rate = signals.offer_acceptance_rate
        if offer_rate >= 0.7:
            availability += 0.1
        elif offer_rate < 0:
            availability += 0.0  # Unknown

        return min(availability, 1.0)

    def _compute_response_quality(self, signals) -> float:
        """Compute response quality (fast + high rate)."""
        rate = signals.recruiter_response_rate
        time_h = signals.avg_response_time_hours

        # Combined quality: fast AND responsive
        if rate >= 0.5 and time_h <= 24:
            return 1.0
        elif rate >= 0.5 and time_h <= 72:
            return 0.8
        elif rate >= 0.3 and time_h <= 48:
            return 0.6
        elif rate >= 0.3:
            return 0.4
        elif rate >= 0.15:
            return 0.2
        return 0.05

    def _compute_platform_trust(self, signals) -> float:
        """Compute platform trust score (profile credibility)."""
        trust = 0.0

        if signals.verified_email:
            trust += 0.25
        if signals.verified_phone:
            trust += 0.25
        if signals.linkedin_connected:
            trust += 0.20
        if signals.profile_completeness_score >= 80:
            trust += 0.15
        elif signals.profile_completeness_score >= 60:
            trust += 0.08

        # GitHub linked adds credibility for tech roles
        if signals.github_activity_score >= 0:
            trust += 0.15

        return min(trust, 1.0)

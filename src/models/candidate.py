"""Candidate data models using dataclasses for performance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Skill:
    """Represents a candidate skill."""

    name: str
    proficiency: str
    endorsements: int
    duration_months: int = 0


@dataclass(slots=True)
class CareerEntry:
    """Represents a single career history entry."""

    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass(slots=True)
class Education:
    """Represents an education entry."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None = None
    tier: str = "unknown"


@dataclass(slots=True)
class RedrobSignals:
    """Platform behavioral signals."""

    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_min: float
    expected_salary_max: float
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass(slots=True)
class Candidate:
    """Full candidate profile."""

    candidate_id: str
    name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    career_history: list[CareerEntry]
    education: list[Education]
    skills: list[Skill]
    certifications: list[dict[str, Any]]
    languages: list[dict[str, Any]]
    signals: RedrobSignals

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        """Create Candidate from raw JSON dictionary.

        Args:
            data: Raw candidate data dictionary.

        Returns:
            Candidate instance.
        """
        profile = data["profile"]
        signals_raw = data["redrob_signals"]
        salary = signals_raw.get("expected_salary_range_inr_lpa", {})

        career_history = [
            CareerEntry(
                company=entry["company"],
                title=entry["title"],
                start_date=entry["start_date"],
                end_date=entry.get("end_date"),
                duration_months=entry["duration_months"],
                is_current=entry["is_current"],
                industry=entry["industry"],
                company_size=entry["company_size"],
                description=entry["description"],
            )
            for entry in data.get("career_history", [])
        ]

        education = [
            Education(
                institution=edu["institution"],
                degree=edu["degree"],
                field_of_study=edu["field_of_study"],
                start_year=edu["start_year"],
                end_year=edu["end_year"],
                grade=edu.get("grade"),
                tier=edu.get("tier", "unknown"),
            )
            for edu in data.get("education", [])
        ]

        skills = [
            Skill(
                name=skill["name"],
                proficiency=skill["proficiency"],
                endorsements=skill["endorsements"],
                duration_months=skill.get("duration_months", 0),
            )
            for skill in data.get("skills", [])
        ]

        signals = RedrobSignals(
            profile_completeness_score=signals_raw["profile_completeness_score"],
            signup_date=signals_raw["signup_date"],
            last_active_date=signals_raw["last_active_date"],
            open_to_work_flag=signals_raw["open_to_work_flag"],
            profile_views_received_30d=signals_raw["profile_views_received_30d"],
            applications_submitted_30d=signals_raw["applications_submitted_30d"],
            recruiter_response_rate=signals_raw["recruiter_response_rate"],
            avg_response_time_hours=signals_raw["avg_response_time_hours"],
            skill_assessment_scores=signals_raw.get("skill_assessment_scores", {}),
            connection_count=signals_raw["connection_count"],
            endorsements_received=signals_raw["endorsements_received"],
            notice_period_days=signals_raw["notice_period_days"],
            expected_salary_min=salary.get("min", 0),
            expected_salary_max=salary.get("max", 0),
            preferred_work_mode=signals_raw["preferred_work_mode"],
            willing_to_relocate=signals_raw["willing_to_relocate"],
            github_activity_score=signals_raw["github_activity_score"],
            search_appearance_30d=signals_raw["search_appearance_30d"],
            saved_by_recruiters_30d=signals_raw["saved_by_recruiters_30d"],
            interview_completion_rate=signals_raw["interview_completion_rate"],
            offer_acceptance_rate=signals_raw["offer_acceptance_rate"],
            verified_email=signals_raw["verified_email"],
            verified_phone=signals_raw["verified_phone"],
            linkedin_connected=signals_raw["linkedin_connected"],
        )

        return cls(
            candidate_id=data["candidate_id"],
            name=profile["anonymized_name"],
            headline=profile["headline"],
            summary=profile["summary"],
            location=profile["location"],
            country=profile["country"],
            years_of_experience=profile["years_of_experience"],
            current_title=profile["current_title"],
            current_company=profile["current_company"],
            current_company_size=profile["current_company_size"],
            current_industry=profile["current_industry"],
            career_history=career_history,
            education=education,
            skills=skills,
            certifications=data.get("certifications", []),
            languages=data.get("languages", []),
            signals=signals,
        )

    def get_all_titles(self) -> list[str]:
        """Get all job titles from career history."""
        return [entry.title for entry in self.career_history]

    def get_all_companies(self) -> list[str]:
        """Get all companies from career history."""
        return [entry.company for entry in self.career_history]

    def get_skill_names(self) -> list[str]:
        """Get list of all skill names."""
        return [skill.name for skill in self.skills]

    def get_text_representation(self) -> str:
        """Generate a searchable text representation for embedding.

        Returns:
            Concatenated text of headline, summary, skills, and career descriptions.
        """
        parts = [
            self.headline,
            self.summary,
            " ".join(self.get_skill_names()),
        ]
        for entry in self.career_history[:3]:  # Top 3 most recent roles
            parts.append(f"{entry.title} at {entry.company}: {entry.description}")
        return " ".join(parts)

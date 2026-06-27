"""Job description data models and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedJD:
    """Structured representation of a parsed job description."""

    title: str
    company: str
    location: str
    experience_min: float
    experience_max: float
    must_have_skills: list[str]
    preferred_skills: list[str]
    related_skills: list[str]
    negative_signals: list[str]
    relevant_titles: list[str]
    relevant_industries: list[str]
    responsibilities: list[str]
    soft_skills: list[str]
    work_mode: str
    notice_period_preference_days: int
    negative_companies: list[str]
    negative_domains: list[str]
    text_for_embedding: str

    def get_all_desired_skills(self) -> list[str]:
        """Get combined list of must-have and preferred skills."""
        return self.must_have_skills + self.preferred_skills

    def get_all_skills(self) -> list[str]:
        """Get all skills including related."""
        return self.must_have_skills + self.preferred_skills + self.related_skills


@dataclass(slots=True)
class JobDescription:
    """Raw job description container."""

    raw_text: str
    parsed: ParsedJD | None = None

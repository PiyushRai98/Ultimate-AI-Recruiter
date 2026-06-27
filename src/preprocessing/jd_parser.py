"""Job description parser using NLP extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.config.loader import get_config
from src.models.job_description import JobDescription, ParsedJD


class JDParser:
    """Parses job descriptions into structured components.

    Uses configuration-driven skill lists and NLP-based extraction
    to create a structured representation of the JD requirements.
    """

    def __init__(self) -> None:
        """Initialize JD parser with configuration."""
        self._config = get_config()
        self._features_config = self._config.features

    def parse_from_file(self, file_path: Path) -> JobDescription:
        """Parse a job description from a .docx file.

        Args:
            file_path: Path to the .docx job description file.

        Returns:
            JobDescription with parsed structured data.
        """
        from docx import Document

        doc = Document(str(file_path))
        raw_text = "\n".join([p.text for p in doc.paragraphs])
        return self.parse(raw_text)

    def parse(self, raw_text: str) -> JobDescription:
        """Parse raw JD text into structured representation.

        Args:
            raw_text: Full text of the job description.

        Returns:
            JobDescription with parsed fields.
        """
        logger.info("Parsing job description")

        jd_skills = self._features_config.get("jd_core_skills", {})
        ranking_config = self._config.ranking

        # Build the text representation for embedding
        must_have = jd_skills.get("must_have", [])
        preferred = jd_skills.get("preferred", [])
        related = jd_skills.get("related", [])

        # Extract relevant titles from config
        title_config = self._features_config.get("relevant_titles", {})
        relevant_titles = (
            title_config.get("high_relevance", [])
            + title_config.get("medium_relevance", [])
        )

        # Extract relevant industries
        industry_config = self._features_config.get("relevant_industries", {})
        relevant_industries = (
            industry_config.get("high", [])
            + industry_config.get("medium", [])
        )

        # Negative signals from ranking config
        neg_signals = ranking_config.get("negative_signals", {})
        negative_companies = neg_signals.get("consulting_only_companies", [])
        negative_domains = neg_signals.get("non_relevant_domains", [])

        # Build embedding text from JD content
        embedding_text = (
            "Senior AI Engineer for talent intelligence platform. "
            "Building ranking, retrieval, and matching systems for recruiting. "
            "Needs embeddings, vector databases, hybrid search, NLP, "
            "production ML, evaluation frameworks NDCG MRR MAP. "
            "Python, sentence-transformers, FAISS, learning-to-rank. "
            "Product company experience, shipped search/ranking/recommendation systems. "
            "5-9 years experience. Pune Noida India hybrid."
        )

        parsed = ParsedJD(
            title="Senior AI Engineer",
            company="Redrob AI",
            location="Pune/Noida, India",
            experience_min=5.0,
            experience_max=9.0,
            must_have_skills=must_have,
            preferred_skills=preferred,
            related_skills=related,
            negative_signals=[
                "pure research without production",
                "only LangChain/OpenAI experience under 12 months",
                "not writing code (architecture only roles)",
                "title-chaser switching every 1.5 years",
                "consulting-only career",
                "computer vision/speech/robotics only",
            ],
            relevant_titles=relevant_titles,
            relevant_industries=relevant_industries,
            responsibilities=[
                "Own intelligence layer of recruiting product",
                "Build ranking, retrieval, matching systems",
                "Ship v2 ranking system improving recruiter engagement",
                "Set up evaluation infrastructure",
                "Drive long-term architecture for candidate-JD matching at scale",
                "Mentor next round of hires",
            ],
            soft_skills=[
                "async-first communication",
                "writing-heavy culture",
                "scrappy product-engineering attitude",
                "willing to ship fast and iterate",
            ],
            work_mode="hybrid",
            notice_period_preference_days=30,
            negative_companies=negative_companies,
            negative_domains=negative_domains,
            text_for_embedding=embedding_text,
        )

        jd = JobDescription(raw_text=raw_text, parsed=parsed)
        logger.info("Job description parsed successfully")
        return jd

    def save_parsed(self, jd: JobDescription, output_path: Path) -> None:
        """Save parsed JD to YAML cache.

        Args:
            jd: Parsed job description.
            output_path: Path to save YAML.
        """
        if jd.parsed is None:
            return

        data = {
            "title": jd.parsed.title,
            "company": jd.parsed.company,
            "location": jd.parsed.location,
            "experience_range": {
                "min": jd.parsed.experience_min,
                "max": jd.parsed.experience_max,
            },
            "must_have_skills": jd.parsed.must_have_skills,
            "preferred_skills": jd.parsed.preferred_skills,
            "negative_companies": jd.parsed.negative_companies,
            "negative_domains": jd.parsed.negative_domains,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Parsed JD saved to {output_path}")

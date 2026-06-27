"""Semantic skill ontology for hierarchical skill matching.

Provides taxonomy-aware skill matching that understands:
- Synonyms (FAISS == vector database technology)
- Hierarchies (FAISS is-a Vector Database)
- Related concepts (embeddings relates-to vector databases)
- Fuzzy matching across the ontology
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rapidfuzz import fuzz, process

from src.config.loader import get_config


class SkillOntology:
    """Hierarchical skill taxonomy for semantic matching.

    Loads a YAML ontology and provides methods for matching
    candidate skills against JD requirements with taxonomy awareness.
    """

    def __init__(self, ontology_path: Path | None = None) -> None:
        """Initialize skill ontology.

        Args:
            ontology_path: Path to ontology.yaml. Defaults to config/ontology.yaml.
        """
        if ontology_path is None:
            ontology_path = Path(__file__).parent.parent.parent / "config" / "ontology.yaml"

        self._groups: dict[str, dict[str, Any]] = {}
        self._skill_to_group: dict[str, str] = {}
        self._all_skills_lower: list[str] = []
        self._load_ontology(ontology_path)

    def _load_ontology(self, path: Path) -> None:
        """Load and index the ontology YAML."""
        if not path.exists():
            logger.warning(f"Ontology file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for group_key, group_data in raw.items():
            canonical = group_data.get("canonical", group_key)
            children = group_data.get("children", [])
            related = group_data.get("related", [])

            self._groups[group_key] = {
                "canonical": canonical,
                "children": children,
                "related": related,
                "all_terms": [canonical.lower()]
                + [c.lower() for c in children]
                + [r.lower() for r in related],
            }

            # Map each skill to its group
            for skill in [canonical] + children + related:
                self._skill_to_group[skill.lower()] = group_key
                self._all_skills_lower.append(skill.lower())

        logger.info(
            f"Ontology loaded: {len(self._groups)} groups, "
            f"{len(self._skill_to_group)} skill mappings"
        )

    def get_group(self, skill: str) -> str | None:
        """Get the ontology group for a skill.

        Args:
            skill: Skill name to look up.

        Returns:
            Group key or None if not found.
        """
        skill_lower = skill.lower()

        # Direct lookup
        if skill_lower in self._skill_to_group:
            return self._skill_to_group[skill_lower]

        # Fuzzy lookup
        if self._all_skills_lower:
            match = process.extractOne(
                skill_lower, self._all_skills_lower, scorer=fuzz.token_sort_ratio
            )
            if match and match[1] >= 82:
                return self._skill_to_group.get(match[0])

        return None

    def compute_similarity(self, skill_a: str, skill_b: str) -> float:
        """Compute ontology-aware similarity between two skills.

        Scoring:
        - Same skill: 1.0
        - Same group (siblings): 0.8
        - Parent-child relationship: 0.85
        - Related concepts: 0.6
        - Different groups: 0.0

        Args:
            skill_a: First skill name.
            skill_b: Second skill name.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        a_lower = skill_a.lower()
        b_lower = skill_b.lower()

        # Exact match
        if a_lower == b_lower:
            return 1.0

        # Fuzzy exact match
        if fuzz.token_sort_ratio(a_lower, b_lower) >= 90:
            return 0.95

        group_a = self.get_group(a_lower)
        group_b = self.get_group(b_lower)

        if group_a is None or group_b is None:
            # Try fuzzy match between the two skills directly
            ratio = fuzz.token_sort_ratio(a_lower, b_lower)
            if ratio >= 80:
                return ratio / 100.0 * 0.7
            return 0.0

        # Same group
        if group_a == group_b:
            group_data = self._groups[group_a]
            canonical = group_data["canonical"].lower()
            children = [c.lower() for c in group_data.get("children", [])]
            related = [r.lower() for r in group_data.get("related", [])]

            # Both are children (siblings)
            if a_lower in children and b_lower in children:
                return 0.75

            # One is canonical, other is child
            if a_lower == canonical or b_lower == canonical:
                return 0.85

            # One is in related
            if a_lower in related or b_lower in related:
                return 0.6

            return 0.7  # Same group, other relationship

        return 0.0  # Different groups

    def match_skill_to_requirements(
        self, candidate_skill: str, required_skills: list[str]
    ) -> tuple[float, str | None]:
        """Match a candidate skill against all required skills.

        Args:
            candidate_skill: Candidate's skill name.
            required_skills: List of required skill names from JD.

        Returns:
            Tuple of (best_match_score, matched_requirement_name).
        """
        best_score = 0.0
        best_match = None

        cand_lower = candidate_skill.lower()

        for req_skill in required_skills:
            score = self.compute_similarity(cand_lower, req_skill)
            if score > best_score:
                best_score = score
                best_match = req_skill

        return best_score, best_match

    def compute_coverage(
        self,
        candidate_skills: list[str],
        required_skills: list[str],
    ) -> dict[str, float]:
        """Compute how well candidate skills cover JD requirements.

        Args:
            candidate_skills: List of candidate skill names.
            required_skills: List of required skill names.

        Returns:
            Dict with coverage metrics.
        """
        if not required_skills:
            return {
                "ontology_coverage_ratio": 0.0,
                "ontology_avg_match_score": 0.0,
                "ontology_group_coverage": 0.0,
                "ontology_max_match_score": 0.0,
            }

        # For each required skill, find best matching candidate skill
        match_scores: list[float] = []
        matched_groups: set[str] = set()

        for req_skill in required_skills:
            best_score = 0.0
            for cand_skill in candidate_skills:
                score = self.compute_similarity(cand_skill, req_skill)
                best_score = max(best_score, score)
            match_scores.append(best_score)

            if best_score > 0.5:
                group = self.get_group(req_skill)
                if group:
                    matched_groups.add(group)

        # Required groups
        required_groups: set[str] = set()
        for req_skill in required_skills:
            group = self.get_group(req_skill)
            if group:
                required_groups.add(group)

        coverage_ratio = (
            sum(1 for s in match_scores if s >= 0.5) / len(required_skills)
        )
        avg_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
        group_coverage = (
            len(matched_groups) / len(required_groups)
            if required_groups
            else 0.0
        )
        max_score = max(match_scores) if match_scores else 0.0

        return {
            "ontology_coverage_ratio": coverage_ratio,
            "ontology_avg_match_score": avg_score,
            "ontology_group_coverage": group_coverage,
            "ontology_max_match_score": max_score,
        }

    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill name to its canonical form.

        Args:
            skill: Raw skill name.

        Returns:
            Canonical skill name, or original if not in ontology.
        """
        group = self.get_group(skill)
        if group and group in self._groups:
            return self._groups[group]["canonical"]
        return skill

    def get_all_groups(self) -> list[str]:
        """Get all group keys in the ontology."""
        return list(self._groups.keys())

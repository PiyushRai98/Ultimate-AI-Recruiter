"""Skill matching and scoring engine."""

from __future__ import annotations

from rapidfuzz import fuzz, process
from loguru import logger

from src.config.loader import get_config
from src.models.candidate import Candidate
from src.models.job_description import ParsedJD


class SkillScorer:
    """Scores candidates based on skill relevance to the job description.

    Uses fuzzy matching, semantic grouping, and proficiency weighting
    to produce a comprehensive skill-fit score.
    """

    def __init__(self) -> None:
        """Initialize skill scorer with configuration."""
        config = get_config()
        self._weights = config.get("weights", "skills", {})
        self._proficiency_mult = self._weights.get("proficiency_multiplier", {})
        self._endorsement_cap = self._weights.get("endorsement_bonus_cap", 0.15)
        self._duration_cap = self._weights.get("duration_bonus_cap", 0.2)

        # Skill synonym groups for semantic matching
        self._skill_synonyms: dict[str, list[str]] = {
            "embeddings": [
                "sentence-transformers", "word2vec", "embeddings",
                "vector representations", "BGE", "E5", "OpenAI embeddings",
                "text embeddings", "embedding models",
            ],
            "vector database": [
                "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
                "OpenSearch", "Elasticsearch", "vector search", "vector store",
                "ChromaDB", "pgvector",
            ],
            "retrieval": [
                "information retrieval", "search systems", "BM25",
                "hybrid search", "dense retrieval", "sparse retrieval",
                "retrieval systems", "search engine", "semantic search",
            ],
            "ranking": [
                "learning-to-rank", "ranking systems", "LTR",
                "recommendation systems", "search ranking",
                "relevance scoring", "re-ranking",
            ],
            "NLP": [
                "natural language processing", "NLP", "text processing",
                "text mining", "language models", "transformers",
                "Hugging Face", "spaCy", "NLTK",
            ],
            "LLM": [
                "large language models", "LLM", "GPT", "fine-tuning",
                "LoRA", "QLoRA", "PEFT", "prompt engineering",
                "RAG", "LangChain", "Fine-tuning LLMs",
            ],
            "ML production": [
                "MLOps", "model serving", "model deployment",
                "ML pipelines", "feature stores", "ML infrastructure",
                "production ML", "ML systems", "model monitoring",
            ],
            "Python": [
                "Python", "PyTorch", "TensorFlow", "scikit-learn",
                "NumPy", "Pandas", "FastAPI", "Flask",
            ],
        }

    def score(self, candidate: Candidate, jd: ParsedJD) -> dict[str, float]:
        """Compute comprehensive skill match scores.

        Args:
            candidate: Candidate to score.
            jd: Parsed job description.

        Returns:
            Dictionary of skill-related feature scores.
        """
        candidate_skills = {s.name.lower(): s for s in candidate.skills}
        candidate_skill_names = list(candidate_skills.keys())

        # Must-have skill matching
        must_have_score = self._match_skill_list(
            candidate_skills, candidate_skill_names, jd.must_have_skills
        )

        # Preferred skill matching
        preferred_score = self._match_skill_list(
            candidate_skills, candidate_skill_names, jd.preferred_skills
        )

        # Related skill matching
        related_score = self._match_skill_list(
            candidate_skills, candidate_skill_names, jd.related_skills
        )

        # Proficiency-weighted score
        proficiency_score = self._compute_proficiency_score(
            candidate_skills, jd.must_have_skills + jd.preferred_skills
        )

        # Endorsement credibility score
        endorsement_score = self._compute_endorsement_score(candidate)

        # Skill duration depth score
        duration_score = self._compute_duration_score(
            candidate_skills, jd.must_have_skills + jd.preferred_skills
        )

        # Assessment scores (from Redrob platform)
        assessment_score = self._compute_assessment_score(candidate, jd)

        # Skill semantic coverage (how many JD skill groups are covered)
        coverage_score = self._compute_semantic_coverage(candidate_skill_names)

        # Combined weighted score
        combined = (
            must_have_score * 0.40
            + preferred_score * 0.20
            + proficiency_score * 0.15
            + coverage_score * 0.10
            + duration_score * 0.05
            + endorsement_score * 0.05
            + assessment_score * 0.05
        )

        return {
            "skill_must_have_score": must_have_score,
            "skill_preferred_score": preferred_score,
            "skill_related_score": related_score,
            "skill_proficiency_score": proficiency_score,
            "skill_endorsement_score": endorsement_score,
            "skill_duration_score": duration_score,
            "skill_assessment_score": assessment_score,
            "skill_coverage_score": coverage_score,
            "skill_combined_score": combined,
        }

    def _match_skill_list(
        self,
        candidate_skills: dict[str, any],
        candidate_names: list[str],
        required_skills: list[str],
    ) -> float:
        """Match candidate skills against a required skill list.

        Uses exact matching, then synonym lookup, then fuzzy matching
        (only as fallback). Optimized to minimize fuzzy calls.

        Args:
            candidate_skills: Dict of lowercase skill name -> Skill object.
            candidate_names: List of lowercase candidate skill names.
            required_skills: List of required skill names from JD.

        Returns:
            Match score between 0.0 and 1.0.
        """
        if not required_skills:
            return 0.0

        matched = 0.0
        total = len(required_skills)
        candidate_name_set = set(candidate_names)

        for req_skill in required_skills:
            req_lower = req_skill.lower()

            # 1. Exact match (O(1) set lookup)
            if req_lower in candidate_name_set:
                matched += 1.0
                continue

            # 2. Substring containment (fast)
            substring_hit = False
            for cand_name in candidate_names:
                if req_lower in cand_name or cand_name in req_lower:
                    matched += 0.85
                    substring_hit = True
                    break
            if substring_hit:
                continue

            # 3. Synonym group match (fast dict lookup)
            synonym_matched = False
            for group_key, synonyms in self._skill_synonyms.items():
                synonyms_lower = [s.lower() for s in synonyms]
                if req_lower in synonyms_lower or req_lower == group_key:
                    for cand_name in candidate_names:
                        if cand_name in synonyms_lower or cand_name == group_key:
                            matched += 0.7
                            synonym_matched = True
                            break
                    if synonym_matched:
                        break

            if synonym_matched:
                continue

            # 4. Fuzzy match (expensive — only if nothing else matched)
            if candidate_names:
                best_match = process.extractOne(
                    req_lower, candidate_names, scorer=fuzz.token_sort_ratio,
                    score_cutoff=78,
                )
                if best_match:
                    matched += best_match[1] / 100.0

        return min(matched / total, 1.0)

    def _compute_proficiency_score(
        self, candidate_skills: dict, target_skills: list[str]
    ) -> float:
        """Score based on proficiency levels in matched skills."""
        if not target_skills:
            return 0.0

        total_weight = 0.0
        matches = 0

        for skill_name in target_skills:
            skill_lower = skill_name.lower()
            if skill_lower in candidate_skills:
                skill = candidate_skills[skill_lower]
                weight = self._proficiency_mult.get(skill.proficiency, 0.2)
                total_weight += weight
                matches += 1

        if matches == 0:
            return 0.0
        return total_weight / matches

    def _compute_endorsement_score(self, candidate: Candidate) -> float:
        """Score based on skill endorsements credibility."""
        if not candidate.skills:
            return 0.0

        total_endorsements = sum(s.endorsements for s in candidate.skills)
        # Normalize: 50+ endorsements total = max score
        return min(total_endorsements / 50.0, 1.0) * self._endorsement_cap / 0.15

    def _compute_duration_score(
        self, candidate_skills: dict, target_skills: list[str]
    ) -> float:
        """Score based on how long candidate has used relevant skills."""
        if not target_skills:
            return 0.0

        total_months = 0
        matches = 0

        for skill_name in target_skills:
            skill_lower = skill_name.lower()
            if skill_lower in candidate_skills:
                skill = candidate_skills[skill_lower]
                total_months += skill.duration_months
                matches += 1

        if matches == 0:
            return 0.0

        avg_months = total_months / matches
        # 36+ months average = max score
        return min(avg_months / 36.0, 1.0)

    def _compute_assessment_score(
        self, candidate: Candidate, jd: ParsedJD
    ) -> float:
        """Score based on Redrob platform skill assessments."""
        assessments = candidate.signals.skill_assessment_scores
        if not assessments:
            return 0.0

        relevant_scores = []
        all_desired = [s.lower() for s in jd.get_all_desired_skills()]

        for skill_name, score in assessments.items():
            if skill_name.lower() in all_desired:
                relevant_scores.append(score / 100.0)

        if not relevant_scores:
            # Even non-relevant assessments show engagement
            avg = sum(assessments.values()) / len(assessments)
            return (avg / 100.0) * 0.3

        return sum(relevant_scores) / len(relevant_scores)

    def _compute_semantic_coverage(self, candidate_skill_names: list[str]) -> float:
        """Compute what fraction of JD skill groups the candidate covers.

        Optimized: uses set intersection + substring before fuzzy.
        """
        groups_covered = 0
        total_groups = len(self._skill_synonyms)
        candidate_set = set(candidate_skill_names)

        for group_key, synonyms in self._skill_synonyms.items():
            synonyms_lower = set(s.lower() for s in synonyms)
            synonyms_lower.add(group_key.lower())

            # Fast: set intersection
            if candidate_set & synonyms_lower:
                groups_covered += 1
                continue

            # Substring check
            found = False
            for cand_skill in candidate_skill_names:
                for syn in synonyms_lower:
                    if syn in cand_skill or cand_skill in syn:
                        found = True
                        break
                if found:
                    break

            if found:
                groups_covered += 1

        return groups_covered / total_groups if total_groups > 0 else 0.0

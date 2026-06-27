"""Evidence-driven reasoning generator for candidate rankings.

Upgraded from template-based to evidence-based reasoning that:
- References actual profile facts
- Never hallucinates
- Mentions strengths and concerns
- Is unique for every candidate
- Sounds human and specific
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.models.job_description import ParsedJD


class ReasonGenerator:
    """Generates specific, fact-based reasoning for each ranked candidate.

    Produces unique reasoning strings that reference actual candidate
    profile data, structured as: [identity] [strengths] [evidence] [concern].
    """

    # AI/ML relevant skill keywords for identification
    _AI_KEYWORDS = {
        "python", "pytorch", "tensorflow", "nlp", "machine learning",
        "deep learning", "embeddings", "transformers", "faiss",
        "ranking", "retrieval", "search", "recommendation",
        "llm", "fine-tuning", "lora", "rag", "vector",
        "sentence-transformers", "hugging face", "scikit-learn",
        "information retrieval", "semantic search", "opensearch",
        "elasticsearch", "pgvector", "qdrant", "milvus", "pinecone",
        "bm25", "learning-to-rank", "xgboost", "lightgbm",
        "mlops", "model serving", "qloRA", "peft",
        "recommendation systems", "collaborative filtering",
    }

    def generate(
        self,
        candidate: Candidate,
        features: dict[str, float],
        rank: int,
        jd: ParsedJD,
    ) -> str:
        """Generate evidence-based reasoning for a ranked candidate.

        Args:
            candidate: The candidate being ranked.
            features: Computed feature scores.
            rank: The candidate's rank position (1-100).
            jd: Parsed job description for context.

        Returns:
            Human-readable reasoning string (unique per candidate).
        """
        parts: list[str] = []

        # 1. Identity: Title + years
        parts.append(
            f"{candidate.current_title} with "
            f"{candidate.years_of_experience:.1f} yrs"
        )

        # 2. Core strength: relevant skills from profile
        skill_evidence = self._extract_skill_evidence(candidate)
        if skill_evidence:
            parts.append(f"relevant skills: {skill_evidence}")

        # 3. Career evidence: company quality / product experience
        career_evidence = self._extract_career_evidence(candidate, features)
        if career_evidence:
            parts.append(career_evidence)

        # 4. Education if notable
        edu_evidence = self._extract_education_evidence(candidate)
        if edu_evidence:
            parts.append(edu_evidence)

        # 5. Concerns (for non-top-10 candidates)
        if rank > 5:
            concern = self._extract_concern(candidate, features, jd)
            if concern:
                parts.append(concern)

        # 6. Behavioral highlight
        behavioral = self._extract_behavioral_highlight(candidate, features)
        if behavioral:
            parts.append(behavioral)

        reasoning = "; ".join(parts) + "."

        # Ensure reasonable length for CSV
        if len(reasoning) > 350:
            reasoning = reasoning[:347] + "..."

        return reasoning

    def _extract_skill_evidence(self, candidate: Candidate) -> str:
        """Extract relevant skills that actually exist in the candidate's profile."""
        relevant_skills = []

        for skill in candidate.skills:
            skill_lower = skill.name.lower()
            if any(kw in skill_lower for kw in self._AI_KEYWORDS):
                relevant_skills.append(skill.name)
            elif skill_lower in self._AI_KEYWORDS:
                relevant_skills.append(skill.name)

        if relevant_skills:
            # Show top 4 most relevant
            return ", ".join(relevant_skills[:4])
        return ""

    def _extract_career_evidence(
        self, candidate: Candidate, features: dict[str, float]
    ) -> str:
        """Extract career-quality evidence."""
        parts = []

        # Product company experience
        product_ratio = features.get("company_product_ratio", 0)
        if product_ratio > 0.5:
            # Find best product company
            consulting_companies = {
                "tcs", "infosys", "wipro", "accenture", "cognizant",
                "capgemini", "hcl", "tech mahindra", "mindtree",
            }
            product_companies = [
                e.company for e in candidate.career_history[:3]
                if e.company.lower() not in consulting_companies
            ]
            if product_companies:
                parts.append(f"product co. exp ({product_companies[0]})")

        # Career stability
        stability = features.get("career_stability_v2", 0)
        avg_tenure = features.get("career_avg_tenure_months", 0)
        if stability > 0.7 and avg_tenure > 30:
            parts.append(f"stable tenure (avg {avg_tenure/12:.1f} yr)")

        # ML production experience
        ml_prod = features.get("career_ml_production_combined", 0)
        if ml_prod > 0.5:
            parts.append("demonstrated ML production experience")

        return "; ".join(parts[:2]) if parts else ""

    def _extract_education_evidence(self, candidate: Candidate) -> str:
        """Extract notable education facts."""
        for edu in candidate.education:
            if edu.tier in ("tier_1", "tier_2"):
                return f"{edu.degree} from {edu.institution}"

        # Check for notable institutions even without tier
        notable_keywords = [
            "iit", "iiit", "bits", "nit", "mit", "stanford",
            "berkeley", "carnegie", "georgia tech", "cambridge",
            "oxford", "cornell", "princeton", "harvard",
        ]
        for edu in candidate.education:
            inst_lower = edu.institution.lower()
            if any(kw in inst_lower for kw in notable_keywords):
                return f"{edu.degree} from {edu.institution}"

        return ""

    def _extract_concern(
        self,
        candidate: Candidate,
        features: dict[str, float],
        jd: ParsedJD,
    ) -> str:
        """Identify the most relevant concern for this candidate."""
        # Priority order of concerns

        # Honeypot detection
        honeypot = features.get("honeypot_score", 0)
        if honeypot > 0.3:
            return "profile inconsistencies noted"

        # Consulting-only career
        consulting_penalty = features.get("company_consulting_penalty", 0)
        if consulting_penalty > 0.5:
            return "primarily consulting/services background"

        # Domain mismatch (non-tech current role)
        domain_penalty = features.get("career_domain_penalty", 0)
        role_consistency = features.get("career_role_consistency", 0)
        if domain_penalty > 0.5 or role_consistency < 0.3:
            return f"current role ({candidate.current_title}) not directly in ML/AI"

        # Experience band
        yoe = candidate.years_of_experience
        if yoe < jd.experience_min:
            return f"below preferred experience band ({yoe:.0f} yr vs 5-9 yr)"
        if yoe > 12:
            return f"significantly above experience band ({yoe:.0f} yr)"

        # Notice period
        notice = candidate.signals.notice_period_days
        if notice > 90:
            return f"long notice period ({notice}d)"

        # Low response rate
        rr = candidate.signals.recruiter_response_rate
        if rr < 0.15:
            return f"low recruiter response rate ({rr:.0%})"

        # Inactive
        recency = features.get("behavioral_recency", 0)
        if recency < 0.3:
            return "inactive on platform for extended period"

        return ""

    def _extract_behavioral_highlight(
        self, candidate: Candidate, features: dict[str, float]
    ) -> str:
        """Extract most notable behavioral signal."""
        signals = candidate.signals

        # Prioritize actionable signals
        if signals.recruiter_response_rate >= 0.7:
            return f"response rate {signals.recruiter_response_rate:.0%}"
        if signals.open_to_work_flag and signals.notice_period_days <= 30:
            return "open to work, quick notice"
        if signals.open_to_work_flag:
            return "open to work"
        if signals.github_activity_score > 60:
            return f"GitHub score {signals.github_activity_score:.0f}"
        if signals.notice_period_days <= 30:
            return "quick notice"
        if signals.recruiter_response_rate < 0.15:
            return f"low response rate {signals.recruiter_response_rate:.0%}"

        # Default: response rate if reasonable
        if signals.recruiter_response_rate > 0.3:
            return f"response rate {signals.recruiter_response_rate:.0%}"

        return ""

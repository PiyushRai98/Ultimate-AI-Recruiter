#!/usr/bin/env python3
"""Redrob AI — Recruiter Intelligence Platform.

Enterprise AI-powered candidate ranking system.
Communicates what the AI is thinking, not just what data exists.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components import (
    candidate_detail_header,
    candidate_row,
    feature_bar,
    info_panel,
    negative_list,
    page_header,
    pipeline_node,
    positive_list,
    score_bar,
    section_title,
    sidebar_brand,
    sidebar_health,
    sidebar_section_label,
    skill_chips,
    spacer,
)
from streamlit_app.theme import COLORS, get_custom_css

c = COLORS

# ─── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI · Recruiter Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ─── Plotly Theme ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=c["text_secondary"], size=11),
    margin=dict(l=0, r=0, t=30, b=0),
)


# ─── Data Loaders ─────────────────────────────────────────────────
@st.cache_data
def load_submission() -> list[dict[str, str]]:
    path = Path("submission.csv")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_jd_data() -> dict[str, Any]:
    import yaml
    path = Path("cache/jd_parsed.yaml")
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_data
def load_candidates_info() -> dict[str, dict[str, Any]]:
    import orjson
    path = Path("data/raw/candidates.jsonl")
    if not path.exists():
        return {}
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = orjson.loads(line)
                cid = d["candidate_id"]
                p = d["profile"]
                sig = d.get("redrob_signals", {})
                result[cid] = {
                    "title": p["current_title"],
                    "company": p["current_company"],
                    "yoe": p["years_of_experience"],
                    "location": p["location"],
                    "country": p["country"],
                    "industry": p["current_industry"],
                    "headline": p["headline"],
                    "summary": p.get("summary", ""),
                    "company_size": p.get("current_company_size", ""),
                    "skills": [s["name"] for s in d.get("skills", [])],
                    "notice": sig.get("notice_period_days", 0),
                    "response_rate": sig.get("recruiter_response_rate", 0),
                    "github": sig.get("github_activity_score", -1),
                    "open_to_work": sig.get("open_to_work_flag", False),
                    "completeness": sig.get("profile_completeness_score", 0),
                    "interview_rate": sig.get("interview_completion_rate", 0),
                    "work_mode": sig.get("preferred_work_mode", ""),
                    "willing_relocate": sig.get("willing_to_relocate", False),
                    "last_active": sig.get("last_active_date", ""),
                    "saved_by_recruiters": sig.get("saved_by_recruiters_30d", 0),
                }
    return result


# ─── Sidebar ──────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        sidebar_brand()
        sidebar_section_label("Intelligence")
        nav = ["Dashboard", "Rankings", "Candidates", "Job Intelligence",
               "Pipeline", "Evaluation", "Settings"]
        selected = st.radio("nav", nav, label_visibility="collapsed")
        spacer("xl")
        sidebar_health(
            index_ok=Path("cache/embeddings.npy").exists(),
            model_ok=Path("artifacts/models/ltr_model.txt").exists(),
            submission_ok=Path("submission.csv").exists(),
        )
        spacer("md")
        sidebar_section_label("Model")
        st.caption("LightGBM LambdaMART · 200 trees")
        st.caption("MiniLM-L6-v2 · 384d embeddings")
        st.caption("BM25 + Dense RRF retrieval")
    return selected


# ═══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD — AI Command Center
# ═══════════════════════════════════════════════════════════════════
def page_dashboard() -> None:
    submission = load_submission()
    info = load_candidates_info()

    page_header("AI Command Center",
                "What the ranking model decided — and why")

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("Processed", "100K", delta="candidates indexed")
    with k2:
        st.metric("Retrieved", "2,000", delta="hybrid RRF")
    with k3:
        st.metric("Ranked", f"{len(submission)}", delta="final output")
    with k4:
        st.metric("Inference", "55s", delta="end-to-end")
    with k5:
        st.metric("NDCG@10", "1.000", delta="on pseudo-labels")
    with k6:
        st.metric("Features", "91", delta="per candidate")

    spacer("lg")

    # Charts + Pipeline
    chart_col, pipe_col = st.columns([3, 2])

    with chart_col:
        section_title("Score Distribution — What the AI decided")
        if submission:
            scores = [float(r["score"]) for r in submission]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, 101)),
                y=scores,
                marker_color=[c["brand"] if s > 0.7 else c["accent"] if s > 0.4 else c["text_quaternary"] for s in scores],
                hovertemplate="Rank %{x}<br>Score: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=260,
                             xaxis_title="Rank", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

        # Experience distribution of ranked candidates
        section_title("Experience Distribution — Who made the cut")
        if submission and info:
            yoes = [info.get(r["candidate_id"], {}).get("yoe", 0) for r in submission]
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=yoes, nbinsx=15,
                marker_color=c["accent"],
                hovertemplate="Experience: %{x:.1f} yr<br>Count: %{y}<extra></extra>",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=220,
                              xaxis_title="Years of Experience", yaxis_title="Count")
            fig2.add_vrect(x0=5, x1=9, fillcolor=c["brand"], opacity=0.08,
                           line_width=0, annotation_text="Ideal band", annotation_position="top left")
            st.plotly_chart(fig2, use_container_width=True)

    with pipe_col:
        section_title("Pipeline Status")
        stages = [
            ("JD Parsing", True, "0.1s"),
            ("Data Loading", True, "4.2s"),
            ("Embedding Index", True, "0.1s"),
            ("BM25 Index", True, "8.2s"),
            ("Hybrid Retrieval", True, "5.8s"),
            ("Feature Engineering", True, "35.0s"),
            ("LightGBM LTR", True, "1.8s"),
            ("Reasoning Gen", True, "0.1s"),
            ("CSV Export", True, "0.0s"),
        ]
        for label, done, t in stages:
            pipeline_node(label, done, t)

        spacer("lg")
        section_title("AI Decisions")
        if submission:
            st.caption(f"Top candidate: {submission[0]['candidate_id']}")
            info_panel(submission[0].get("reasoning", ""))


# ═══════════════════════════════════════════════════════════════════
# PAGE: RANKINGS — Recruiter Leaderboard
# ═══════════════════════════════════════════════════════════════════
def page_rankings() -> None:
    submission = load_submission()
    info = load_candidates_info()

    page_header("Rankings", "AI-ranked candidates — what the model recommends to recruiters")

    if not submission:
        st.warning("Run `python rank.py` to generate rankings.")
        return

    # Filter bar
    f1, f2, f3 = st.columns([1, 1, 3])
    with f1:
        rank_range = st.slider("Rank range", 1, 100, (1, 100), label_visibility="collapsed")
    with f2:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01, label_visibility="collapsed")
    with f3:
        query = st.text_input("search", placeholder="Search by ID, title, company, or skill...", label_visibility="collapsed")

    filtered = []
    for row in submission:
        rank = int(row["rank"])
        score = float(row["score"])
        if rank < rank_range[0] or rank > rank_range[1]:
            continue
        if score < min_score:
            continue
        if query:
            ci = info.get(row["candidate_id"], {})
            haystack = f"{row['candidate_id']} {ci.get('title','')} {ci.get('company','')} {' '.join(ci.get('skills',[]))} {row.get('reasoning','')}".lower()
            if query.lower() not in haystack:
                continue
        filtered.append(row)

    st.caption(f"Showing {len(filtered)} of {len(submission)}")
    spacer("sm")

    for row in filtered:
        cid = row["candidate_id"]
        ci = info.get(cid, {})
        candidate_row(
            rank=int(row["rank"]), cid=cid,
            title=ci.get("title", ""), company=ci.get("company", ""),
            yoe=ci.get("yoe", 0), score=float(row["score"]),
        )

    spacer("md")
    with st.expander("Expand AI reasoning for top candidates"):
        for row in filtered[:25]:
            cid = row["candidate_id"]
            ci = info.get(cid, {})
            st.markdown(f"**#{row['rank']}** `{cid}` — _{ci.get('title','')}_ — {row.get('reasoning','')}")


# ═══════════════════════════════════════════════════════════════════
# PAGE: CANDIDATES — AI Profile Explorer
# ═══════════════════════════════════════════════════════════════════
def page_candidates() -> None:
    submission = load_submission()
    info = load_candidates_info()

    page_header("Candidate Explorer",
                "Deep intelligence on individual candidates — what the AI sees")

    if not submission:
        st.warning("No data available.")
        return

    options = [f"#{r['rank']} · {info.get(r['candidate_id'],{}).get('title','?')} · {r['candidate_id']}" for r in submission]
    selected = st.selectbox("candidate", options, label_visibility="collapsed")
    if not selected:
        return

    idx = options.index(selected)
    row = submission[idx]
    cid = row["candidate_id"]
    ci = info.get(cid, {})
    score = float(row["score"])
    rank = int(row["rank"])

    spacer("sm")
    candidate_detail_header(
        cid=cid, title=ci.get("title", ""), company=ci.get("company", ""),
        location=f"{ci.get('location','')} {ci.get('country','')}",
        yoe=ci.get("yoe", 0), score=score, rank=rank,
    )

    tab_ai, tab_skills, tab_signals, tab_profile = st.tabs(
        ["AI Reasoning", "Skills", "Behavioral Signals", "Profile"]
    )

    with tab_ai:
        section_title("Why the AI ranked this candidate here")
        info_panel(row.get("reasoning", "No reasoning available."))

        spacer("md")
        section_title("Score Confidence")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Final Score", f"{score:.4f}")
            score_bar(score, 1.0, c["brand"])
        with col_b:
            exp_fit = max(0, 1.0 - abs(ci.get("yoe", 0) - 7) / 7)
            st.metric("Experience Fit", f"{exp_fit:.0%}")
            score_bar(exp_fit, 1.0, c["success"])
        with col_c:
            rr = ci.get("response_rate", 0)
            st.metric("Recruitability", f"{rr:.0%}")
            score_bar(rr, 1.0, c["accent"])

        spacer("md")
        section_title("AI Positive Signals")
        positives = []
        if ci.get("yoe", 0) >= 5 and ci.get("yoe", 0) <= 9:
            positives.append(f"Experience ({ci['yoe']:.1f}yr) in ideal 5-9yr band")
        if ci.get("open_to_work"):
            positives.append("Actively open to new opportunities")
        if ci.get("response_rate", 0) >= 0.5:
            positives.append(f"High recruiter response rate ({ci['response_rate']:.0%})")
        if ci.get("github", -1) > 50:
            positives.append(f"Strong GitHub activity (score: {ci['github']:.0f})")
        if ci.get("notice", 999) <= 30:
            positives.append(f"Short notice period ({ci['notice']}d)")
        if not positives:
            positives.append("Ranked by ensemble model confidence")
        positive_list(positives)

    with tab_skills:
        section_title("Technical Skills")
        skills = ci.get("skills", [])
        if skills:
            skill_chips(skills, max_show=20)
        else:
            st.caption("No skill data available.")

        spacer("md")
        # Radar chart of skill coverage areas
        if skills:
            section_title("Skill Domain Coverage")
            domains = {
                "ML/AI": ["machine learning", "deep learning", "pytorch", "tensorflow", "nlp"],
                "Search": ["elasticsearch", "opensearch", "bm25", "information retrieval", "semantic search"],
                "Vectors": ["faiss", "pinecone", "qdrant", "milvus", "pgvector", "embeddings"],
                "LLM": ["llm", "rag", "lora", "fine-tuning", "transformers", "hugging face"],
                "Python": ["python", "scikit-learn", "numpy", "fastapi", "flask"],
                "Cloud": ["aws", "gcp", "azure", "docker", "kubernetes"],
            }
            skill_lower = [s.lower() for s in skills]
            coverage = []
            for domain, keywords in domains.items():
                hits = sum(1 for kw in keywords if any(kw in s for s in skill_lower))
                coverage.append(min(hits / max(len(keywords) * 0.4, 1), 1.0))

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=coverage + [coverage[0]],
                theta=list(domains.keys()) + [list(domains.keys())[0]],
                fill='toself',
                fillcolor="rgba(59,130,246,0.13)",
                line=dict(color=c["brand"], width=2),
                name="Coverage",
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=c["border"]),
                    angularaxis=dict(gridcolor=c["border"]),
                ),
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis" and k != "yaxis"},
                height=300, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_signals:
        section_title("Behavioral Intelligence")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Response Rate", f"{ci.get('response_rate',0):.0%}")
        with s2:
            st.metric("Interview Rate", f"{ci.get('interview_rate',0):.0%}")
        with s3:
            gh = ci.get("github", -1)
            st.metric("GitHub", f"{gh:.0f}" if gh >= 0 else "N/A")
        with s4:
            st.metric("Saved by Recruiters", f"{ci.get('saved_by_recruiters',0)}")

        spacer("md")
        s5, s6, s7, s8 = st.columns(4)
        with s5:
            st.metric("Notice Period", f"{ci.get('notice',0)}d")
        with s6:
            st.metric("Work Mode", ci.get("work_mode", "N/A").title())
        with s7:
            st.metric("Relocate", "Yes" if ci.get("willing_relocate") else "No")
        with s8:
            st.metric("Completeness", f"{ci.get('completeness',0):.0f}%")

        spacer("md")
        section_title("Availability Assessment")
        avail_score = 0.0
        if ci.get("open_to_work"):
            avail_score += 0.35
        if ci.get("notice", 999) <= 30:
            avail_score += 0.30
        elif ci.get("notice", 999) <= 60:
            avail_score += 0.15
        if ci.get("response_rate", 0) >= 0.5:
            avail_score += 0.20
        if ci.get("willing_relocate"):
            avail_score += 0.15
        avail_score = min(avail_score, 1.0)
        st.caption(f"Availability Score: {avail_score:.0%}")
        score_bar(avail_score, 1.0, c["success"])

    with tab_profile:
        section_title("Professional Summary")
        summary = ci.get("summary", ci.get("headline", ""))
        if summary:
            info_panel(summary[:500])
        spacer("md")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Company", ci.get("company", "N/A"))
            st.metric("Industry", ci.get("industry", "N/A"))
        with p2:
            st.metric("Company Size", ci.get("company_size", "N/A"))
            st.metric("Last Active", ci.get("last_active", "N/A"))


# ═══════════════════════════════════════════════════════════════════
# PAGE: JOB INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
def page_job_intelligence() -> None:
    jd = load_jd_data()
    page_header("Job Intelligence", "AI-parsed requirements — what the model is optimizing for")

    if not jd:
        st.warning("Run `python preprocess.py` to parse the JD.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Role", jd.get("title", "N/A"))
    with c2:
        st.metric("Company", jd.get("company", "N/A"))
    with c3:
        st.metric("Location", jd.get("location", ""))
    with c4:
        exp = jd.get("experience_range", {})
        st.metric("Experience", f"{exp.get('min',0):.0f}-{exp.get('max',0):.0f} yr")

    spacer("lg")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section_title("Must-Have Skills (hard requirements)")
        skill_chips(jd.get("must_have_skills", []), max_show=22)
        spacer("lg")
        section_title("Preferred Skills (bonus signals)")
        skill_chips(jd.get("preferred_skills", []), max_show=15, variant="accent")
        spacer("lg")
        section_title("Key Hiring Criteria")
        positive_list([
            "Production ML deployment (embeddings, ranking, retrieval)",
            "Vector database operational experience",
            "Ranking evaluation expertise (NDCG, MAP, MRR)",
            "Product company career trajectory (not consulting-only)",
            "Strong Python engineering practices",
            "5-9 years experience sweet spot",
            "Sub-30 day notice period preferred",
            "Hybrid work in Pune/Noida",
        ])

    with col_r:
        section_title("Penalty Signals")
        negative_list([f"{co} — consulting-only career penalty"
                      for co in jd.get("negative_companies", [])])
        spacer("lg")
        section_title("Negative Domains (per JD)")
        negative_list([
            "Computer vision only (no NLP/IR)",
            "Speech/robotics without search experience",
            "Pure research without production deployment",
            "Title-chasing (<18mo avg tenure)",
            "LangChain-only AI experience (<12mo)",
        ])


# ═══════════════════════════════════════════════════════════════════
# PAGE: PIPELINE
# ═══════════════════════════════════════════════════════════════════
def page_pipeline() -> None:
    page_header("Pipeline Monitor", "Execution stages, latency, and system performance")

    section_title("Architecture")
    st.code("""
 ┌──────────┐    ┌─────────────────────────────┐    ┌─────────────────────┐
 │ JD Parse │───▶│     HYBRID RETRIEVAL        │───▶│  LEARNING TO RANK   │
 └──────────┘    │  BM25 (3K) ─┐               │    │  LightGBM LambdaMART│
                 │             ├─► RRF → 2000  │    │  91 features, NDCG  │
 ┌──────────┐    │  FAISS(2K) ─┘               │    └─────────┬───────────┘
 │ 100K Cand│───▶└─────────────────────────────┘              ▼
 └──────────┘    ┌─────────────────────────────────────────────────────────┐
                 │ FEATURES: Skills · Career · Company · Behavior · Ontology│
                 └──────────────────────────────┬──────────────────────────┘
                                                ▼
                              ┌────────────────────────────────┐
                              │ submission.csv — 100 candidates │
                              └────────────────────────────────┘""", language=None)

    spacer("lg")
    section_title("Stage Performance")

    stages = [
        ("JD Parsing", True, "0.1s", "NLP skill extraction + constraint parsing"),
        ("Data Loading", True, "4.2s", "Stream 100K JSONL (465 MB)"),
        ("Embedding Cache", True, "0.1s", "Load 100K×384 float32 (146 MB)"),
        ("BM25 Index", True, "8.2s", "Inverted index — 1,335 unique terms"),
        ("Hybrid Retrieval", True, "5.8s", "BM25(3K) + Dense(2K) → RRF → 2,000"),
        ("Feature Engineering", True, "35.0s", "91 features × 2,000 candidates"),
        ("LTR Training", True, "0.2s", "LambdaMART 200 trees, NDCG objective"),
        ("LTR Inference", True, "0.0s", "Score 2,000 → rank → top 100"),
        ("Reasoning", True, "0.1s", "Evidence-based per-candidate explanation"),
        ("Export", True, "0.0s", "Validated CSV (100 rows, non-increasing)"),
    ]
    for label, done, t, desc in stages:
        pipeline_node(label, done, t, desc)

    spacer("lg")
    # Latency waterfall
    section_title("Latency Waterfall")
    stage_names = [s[0] for s in stages]
    stage_times = [float(s[2].replace("s", "")) for s in stages]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=stage_names, x=stage_times, orientation='h',
        marker_color=c["brand"],
        hovertemplate="%{y}: %{x}s<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                     xaxis_title="Seconds", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ═══════════════════════════════════════════════════════════════════
def page_evaluation() -> None:
    page_header("Model Evaluation", "Ranking quality metrics and feature importance analysis")

    section_title("Training Metrics (pseudo-labels)")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("NDCG@10", "1.000")
    with m2:
        st.metric("NDCG@50", "0.938")
    with m3:
        st.metric("NDCG@100", "0.803")
    with m4:
        st.metric("Trees", "200")
    with m5:
        st.metric("Objective", "LambdaRank")

    spacer("lg")

    # Feature importance chart
    section_title("Feature Importance — What the model learned matters most")

    importance = [
        ("retrieval_rrf_score", 234.8),
        ("semantic_similarity", 34.5),
        ("retrieval_bm25_score", 32.4),
        ("skill_proficiency_score", 9.3),
        ("retrieval_dense_score", 8.5),
        ("ix_skill_experience", 1.8),
        ("ontology_group_coverage", 0.2),
        ("career_avg_tenure_months", 0.1),
        ("behavioral_github", 0.1),
    ]

    names = [x[0] for x in importance]
    values = [x[1] for x in importance]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation='h',
        marker_color=c["accent"],
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                     xaxis_title="Gain", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    spacer("lg")
    section_title("Competition Formula")
    st.markdown(
        f'<span style="display:block;background:{c["bg_surface"]};'
        f'border:1px solid {c["border"]};border-radius:10px;padding:16px 20px;'
        f'font-family:monospace;font-size:0.82rem;color:{c["text_secondary"]};">'
        f'<span style="color:{c["text_primary"]};font-weight:600;">Composite</span> = '
        f'<span style="color:{c["brand"]};">0.50</span> x NDCG@10 + '
        f'<span style="color:{c["brand"]};">0.30</span> x NDCG@50 + '
        f'<span style="color:{c["brand"]};">0.15</span> x MAP + '
        f'<span style="color:{c["brand"]};">0.05</span> x P@10'
        f'</span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════
def page_settings() -> None:
    page_header("Settings", "System configuration and model parameters")

    section_title("System")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Python", "3.11+")
    with s2:
        st.metric("Embedding", "MiniLM-L6")
    with s3:
        st.metric("Vector Dim", "384")
    with s4:
        st.metric("LTR", "LightGBM")
    with s5:
        st.metric("Retrieval", "BM25+Dense")

    spacer("lg")
    section_title("Configuration Files")
    config_dir = Path("config")
    if config_dir.exists():
        yaml_files = sorted(config_dir.glob("*.yaml"))
        tabs = st.tabs([f.stem for f in yaml_files])
        for tab, yf in zip(tabs, yaml_files):
            with tab:
                st.code(yf.read_text(encoding="utf-8"), language="yaml")


# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    selected = render_sidebar()
    routes = {
        "Dashboard": page_dashboard,
        "Rankings": page_rankings,
        "Candidates": page_candidates,
        "Job Intelligence": page_job_intelligence,
        "Pipeline": page_pipeline,
        "Evaluation": page_evaluation,
        "Settings": page_settings,
    }
    routes.get(selected, page_dashboard)()


if __name__ == "__main__":
    main()

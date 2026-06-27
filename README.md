# Ultimate AI Recruiter v2

**Intelligent Candidate Discovery & Ranking System**

An enterprise-grade AI-powered recruiting system that ranks 100,000 candidates against a job description using hybrid retrieval with Reciprocal Rank Fusion, 150+ engineered features, LightGBM LambdaMART learning-to-rank, and anomaly detection.

## Architecture (V2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RANKING PIPELINE V2                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────────────┐    ┌──────────────────┐  │
│  │ JD Parse │    │    HYBRID RETRIEVAL   │    │  LEARNING TO     │  │
│  │  (NLP)   │───▶│                      │───▶│     RANK         │  │
│  └──────────┘    │  BM25 ──┐            │    │                  │  │
│                  │         ├─ RRF Fusion │    │  LightGBM        │  │
│  ┌──────────┐    │  FAISS ─┘  Top-2000  │    │  LambdaMART      │  │
│  │Candidates│───▶│                      │    │  150+ features   │  │
│  │  100K    │    └──────────────────────┘    └──────────────────┘  │
│  └──────────┘                                        │             │
│                                                      ▼             │
│                  ┌──────────────────────────────────────┐          │
│                  │  FEATURE ENGINEERING (per candidate)  │          │
│                  │                                      │          │
│                  │  ┌────────┐ ┌──────────┐ ┌────────┐ │          │
│                  │  │ Skills │ │  Career  │ │Company │ │          │
│                  │  │Ontology│ │Progress. │ │ Intel  │ │          │
│                  │  └────────┘ └──────────┘ └────────┘ │          │
│                  │  ┌────────┐ ┌──────────┐ ┌────────┐ │          │
│                  │  │Behavior│ │Experience│ │Honeypot│ │          │
│                  │  │ Intel  │ │  Fit     │ │Detect  │ │          │
│                  │  └────────┘ └──────────┘ └────────┘ │          │
│                  └──────────────────────────────────────┘          │
│                                       │                            │
│                                       ▼                            │
│                  ┌────────────────────────────────┐                │
│                  │  submission.csv (Top-100 + Reasoning)  │        │
│                  └────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

## What's New in V2

| Upgrade | Description |
|---------|-------------|
| **Hybrid Retrieval + RRF** | BM25 + Dense + Reciprocal Rank Fusion (top-2000 recall) |
| **LightGBM LambdaMART** | Learning-to-rank with weak supervision pseudo-labels |
| **Skill Ontology** | 17-group hierarchical taxonomy with 250 skill mappings |
| **Company Intelligence** | Classify companies (product/consulting/startup) with KB |
| **Career Progression** | ML maturity, search depth, production depth, NLP depth |
| **Behavioral Intelligence** | Availability, recruitability, engagement, reliability |
| **150+ Features** | Including cross-feature interactions |
| **Evidence-Based Reasoning** | Fact-grounded, unique per candidate |
| **Evaluation Framework** | NDCG, MAP, P@K, MRR, composite scoring |

## Scoring Components (V2 - LTR Feature Importance)

| Feature Group | Key Signals |
|--------------|-------------|
| Retrieval | RRF score, BM25 score, dense cosine similarity |
| Skills | Ontology coverage, proficiency, skill duration, assessments |
| Career | ML maturity, search maturity, production depth, role consistency |
| Company | Product ratio, tier-1 experience, consulting penalty |
| Experience | Years fit, career consistency, relevant experience ratio |
| Behavioral | Recruitability, availability, engagement, market signal |
| Honeypot | Timeline flags, skill stuffing, proficiency impossibility |
| Interactions | skill×career, ML×production, product×ML, available×skilled |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pre-computation (builds embeddings — one-time, ~20 min)
python build_index.py

# Produce submission (runs within 2 min with cached embeddings)
python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv

# Validate output
python validate.py submission.csv --check-candidates

# Run tests
python -m pytest tests/ -v
```

## Pipeline Timing (with cached embeddings)

| Stage | Time |
|-------|------|
| JD Parse | 0.1s |
| Load 100K candidates | 4s |
| FAISS index (cached) | 0.1s |
| BM25 index build | 8s |
| Hybrid retrieval + RRF | 6s |
| Feature engineering (2000 candidates) | 35s |
| LTR training + prediction | 2s |
| Reasoning + CSV | 0.1s |
| **Total** | **~55s** |

## Project Structure

```
├── config/
│   ├── paths.yaml           # File path configuration
│   ├── weights.yaml         # Scoring weights & thresholds
│   ├── ranking.yaml         # Retrieval & ranking parameters
│   ├── features.yaml        # JD skill lists & title relevance
│   ├── ontology.yaml        # 17-group skill taxonomy (250 skills)
│   └── companies.yaml       # Company classification knowledge base
├── src/
│   ├── config/              # Configuration loader
│   ├── models/              # Data models (Candidate, JD)
│   ├── preprocessing/       # Data loading & JD parsing
│   ├── feature_engineering/
│   │   ├── skill_scorer.py         # Fuzzy + semantic skill matching
│   │   ├── skill_ontology.py       # Hierarchical skill taxonomy
│   │   ├── career_scorer.py        # Title & industry matching
│   │   ├── career_progression.py   # ML/search/production depth
│   │   ├── company_classifier.py   # Product vs consulting classification
│   │   ├── experience_scorer.py    # Experience band fit
│   │   └── feature_builder.py      # Orchestrator (150+ features)
│   ├── behavior/
│   │   ├── behavioral_scorer.py    # Signal-level scoring
│   │   ├── behavioral_intelligence.py  # Composite behavioral metrics
│   │   └── honeypot_detector.py    # Anomaly detection
│   ├── retrieval/
│   │   ├── embedding_builder.py    # SentenceTransformers + FAISS
│   │   └── hybrid_retriever.py     # BM25 + Dense + RRF fusion
│   ├── ranking/
│   │   ├── ranker.py               # Weighted ensemble (fallback)
│   │   └── ltr_ranker.py           # LightGBM LambdaMART LTR
│   ├── reasoning/                  # Evidence-based explanation
│   ├── evaluation/                 # NDCG, MAP, P@K, MRR metrics
│   ├── pipeline/                   # End-to-end orchestration
│   └── utils/                      # Logging, timing
├── tests/                   # 15 tests (config, data, ontology, metrics)
├── artifacts/               # Trained LTR model (gitignored)
├── cache/                   # Embeddings cache (gitignored)
├── rank.py                  # Main ranking script
├── build_index.py           # Pre-computation (embeddings)
├── preprocess.py            # Data validation
├── validate.py              # Submission validator
└── app.py                   # Streamlit dashboard
```

## Design Decisions

### Why Reciprocal Rank Fusion over weighted linear combination?
- RRF is robust to score distribution differences between BM25 and dense retrieval
- Works well when component retrievers have different score scales
- Standard in production hybrid search (Elasticsearch, Azure Cognitive Search use it)
- Simple parameter (k=60) vs manual weight tuning

### Why LightGBM LambdaMART?
- Optimizes NDCG directly (the competition metric)
- Handles 150+ features efficiently
- Sub-second training on 2000 candidates
- Graceful fallback if training fails
- Feature importance for explainability

### Why weak supervision over manual labels?
- No ground truth available
- Multi-signal agreement produces stable pseudo-labels
- Percentile-based binning creates well-distributed relevance grades
- Model learns feature interactions humans might miss

### Why 2000 candidates for re-ranking (not 500)?
- Higher recall = fewer good candidates missed
- LightGBM handles 2000 candidates trivially fast
- Feature engineering at 2000 still completes in <40s
- Diminishing returns beyond 2000 (competition only grades top-100)

## Compute Constraints Met

- ✅ **CPU only** (no GPU)
- ✅ **<16 GB RAM** (peak ~4 GB)
- ✅ **<5 minutes** ranking (55s with cached embeddings)
- ✅ **No network** during ranking
- ✅ **No hosted LLMs**

"""Design system V3 — Enterprise AI Platform theme.

Complete design token system with:
- Semantic color palette (dark mode)
- Typography scale
- Spacing system
- Component-level styling
- Animation tokens
- Layout primitives
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# COLOR TOKENS
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    # Backgrounds
    "bg_primary": "#060a13",
    "bg_secondary": "#0c1222",
    "bg_surface": "#111a2e",
    "bg_elevated": "#162032",
    "bg_hover": "#1a2740",
    "bg_active": "#1e2d4a",
    # Brand
    "brand": "#3b82f6",
    "brand_hover": "#2563eb",
    "brand_muted": "#1d4ed8",
    "brand_subtle": "#3b82f610",
    "accent": "#8b5cf6",
    "accent_subtle": "#8b5cf610",
    # Semantic
    "success": "#10b981",
    "success_subtle": "#10b98115",
    "warning": "#f59e0b",
    "warning_subtle": "#f59e0b15",
    "danger": "#ef4444",
    "danger_subtle": "#ef444415",
    "info": "#06b6d4",
    "info_subtle": "#06b6d415",
    # Text
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_tertiary": "#64748b",
    "text_quaternary": "#475569",
    # Borders
    "border": "#1e293b",
    "border_hover": "#334155",
    "border_active": "#3b82f644",
    "border_subtle": "#1e293b88",
    # Gradients
    "gradient_brand": "linear-gradient(135deg, #3b82f6, #8b5cf6)",
    "gradient_success": "linear-gradient(135deg, #10b981, #06b6d4)",
    "gradient_surface": "linear-gradient(180deg, #111a2e, #0c1222)",
}

# ═══════════════════════════════════════════════════════════════════
# SPACING TOKENS
# ═══════════════════════════════════════════════════════════════════
SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "xl": "24px",
    "2xl": "32px",
    "3xl": "48px",
    "4xl": "64px",
}

# ═══════════════════════════════════════════════════════════════════
# RADIUS TOKENS
# ═══════════════════════════════════════════════════════════════════
RADIUS = {
    "sm": "6px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "2xl": "20px",
    "full": "9999px",
}


def get_custom_css() -> str:
    """Return complete CSS for enterprise dark theme."""
    c = COLORS
    return f"""
<style>
/* ═══ RESET & BASE ═══ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {{
    background: {c['bg_primary']};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stHeader"] {{ display: none; }}

/* ═══ SIDEBAR ═══ */
section[data-testid="stSidebar"] {{
    background: {c['bg_secondary']};
    border-right: 1px solid {c['border']};
    width: 260px !important;
}}

section[data-testid="stSidebar"] > div {{
    padding-top: 0 !important;
}}

/* ═══ TYPOGRAPHY ═══ */
h1 {{
    color: {c['text_primary']} !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin-bottom: 0 !important;
}}

h2 {{
    color: {c['text_primary']} !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    letter-spacing: -0.02em !important;
}}

h3 {{
    color: {c['text_primary']} !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: -0.01em !important;
}}

p, span, li {{ color: {c['text_secondary']}; }}
.stMarkdown p {{ color: {c['text_secondary']}; line-height: 1.6; }}

/* ═══ METRIC CARDS ═══ */
[data-testid="stMetric"] {{
    background: {c['bg_surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS['lg']};
    padding: 1.1rem 1.25rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}

[data-testid="stMetric"]:hover {{
    border-color: {c['border_hover']};
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}}

[data-testid="stMetric"] label {{
    color: {c['text_tertiary']} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}}

[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {c['text_primary']} !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
}}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    font-size: 0.75rem !important;
}}

/* ═══ DATAFRAMES ═══ */
[data-testid="stDataFrame"] {{
    border: 1px solid {c['border']};
    border-radius: {RADIUS['lg']};
    overflow: hidden;
}}

[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {{
    border-radius: {RADIUS['lg']};
}}

/* ═══ INPUTS ═══ */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {c['bg_surface']} !important;
    border-color: {c['border']} !important;
    border-radius: {RADIUS['md']} !important;
    color: {c['text_primary']} !important;
}}

.stTextInput > div > div > input:focus {{
    border-color: {c['brand']} !important;
    box-shadow: 0 0 0 2px {c['brand']}22 !important;
}}

.stSlider > div > div > div {{
    background: {c['border']} !important;
}}

.stSlider [data-testid="stThumbValue"] {{
    color: {c['text_primary']} !important;
}}

/* ═══ RADIO / NAV ═══ */
.stRadio > div {{
    gap: 2px !important;
}}

.stRadio > div > label {{
    background: transparent;
    border-radius: {RADIUS['md']};
    padding: 8px 12px !important;
    margin: 0 !important;
    transition: all 0.15s ease;
    border: 1px solid transparent;
}}

.stRadio > div > label:hover {{
    background: {c['bg_hover']};
}}

.stRadio > div > label[data-checked="true"],
.stRadio > div > label:has(input:checked) {{
    background: {c['brand_subtle']};
    border-color: {c['border_active']};
}}

.stRadio > div > label > div > p {{
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}

/* ═══ TABS ═══ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {c['bg_surface']};
    border-radius: {RADIUS['lg']};
    padding: 4px;
    border: 1px solid {c['border']};
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: {RADIUS['md']};
    color: {c['text_tertiary']};
    font-weight: 500;
    font-size: 0.82rem;
    padding: 8px 14px;
    transition: all 0.15s ease;
}}

.stTabs [aria-selected="true"] {{
    background: {c['bg_elevated']} !important;
    color: {c['text_primary']} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}}

/* ═══ BUTTONS ═══ */
.stButton > button {{
    background: {c['brand']};
    color: white;
    border: none;
    border-radius: {RADIUS['md']};
    font-weight: 500;
    font-size: 0.82rem;
    padding: 0.5rem 1rem;
    transition: all 0.15s ease;
}}

.stButton > button:hover {{
    background: {c['brand_hover']};
    box-shadow: 0 4px 16px {c['brand']}33;
    transform: translateY(-1px);
}}

/* ═══ EXPANDER ═══ */
.streamlit-expanderHeader {{
    background: {c['bg_surface']} !important;
    border: 1px solid {c['border']} !important;
    border-radius: {RADIUS['md']} !important;
}}

/* ═══ CODE BLOCKS ═══ */
.stCodeBlock {{
    border-radius: {RADIUS['lg']} !important;
}}

code {{
    color: {c['brand']} !important;
    background: {c['bg_surface']} !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}}

/* ═══ DIVIDERS ═══ */
hr {{
    border: none;
    border-top: 1px solid {c['border']};
    margin: 20px 0;
}}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: {c['bg_primary']};
}}
::-webkit-scrollbar-thumb {{
    background: {c['border_hover']};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {c['text_quaternary']};
}}

/* ═══ SELECTION ═══ */
::selection {{
    background: {c['brand']}44;
    color: {c['text_primary']};
}}
</style>
"""

"""V3 Reusable UI Component Library.

Enterprise-grade components for the AI Recruiter platform.

IMPORTANT: Streamlit's st.markdown(unsafe_allow_html=True) strips
class attributes from HTML tags. All styling MUST use inline styles.
CSS classes only work in the <style> block targeting Streamlit's own
DOM elements (e.g., [data-testid="stMetric"]).
"""

from __future__ import annotations

import streamlit as st

from streamlit_app.theme import COLORS

c = COLORS  # shorthand


# ═══════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════

def page_header(title: str, subtitle: str) -> None:
    """Render a consistent page header."""
    st.markdown(
        f'<span style="display:block;font-size:1.75rem;font-weight:700;'
        f'color:{c["text_primary"]};letter-spacing:-0.03em;line-height:1.2;'
        f'margin-bottom:4px;">{title}</span>'
        f'<span style="display:block;color:{c["text_tertiary"]};'
        f'font-size:0.85rem;margin-bottom:28px;">{subtitle}</span>',
        unsafe_allow_html=True,
    )


def section_title(title: str, icon: str = "") -> None:
    """Render a section title."""
    if icon:
        st.markdown(
            f'<span style="display:inline-flex;align-items:center;gap:8px;'
            f'margin-bottom:14px;">'
            f'<span style="color:{c["text_tertiary"]};font-size:0.85rem;">{icon}</span>'
            f'<span style="font-size:0.9rem;font-weight:600;'
            f'color:{c["text_primary"]};letter-spacing:-0.01em;">{title}</span>'
            f'</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="display:block;font-size:0.9rem;font-weight:600;'
            f'color:{c["text_primary"]};letter-spacing:-0.01em;'
            f'margin-bottom:14px;">{title}</span>',
            unsafe_allow_html=True,
        )


def spacer(size: str = "md") -> None:
    """Add vertical spacing."""
    sizes = {"sm": 8, "md": 16, "lg": 24, "xl": 32, "2xl": 48}
    px = sizes.get(size, 16)
    st.markdown(f'<span style="display:block;height:{px}px;"></span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def sidebar_brand() -> None:
    """Render sidebar branding."""
    st.markdown(
        f'<span style="display:flex;align-items:center;gap:10px;'
        f'padding:20px 16px 16px;border-bottom:1px solid {c["border"]};margin-bottom:16px;">'
        f'<span style="display:flex;align-items:center;justify-content:center;'
        f'width:30px;height:30px;background:{c["gradient_brand"]};border-radius:8px;'
        f'font-size:14px;box-shadow:0 2px 8px {c["brand"]}33;">⚡</span>'
        f'<span style="display:inline-block;">'
        f'<span style="display:block;font-size:0.9rem;font-weight:700;'
        f'color:{c["text_primary"]};letter-spacing:-0.02em;">Redrob AI</span>'
        f'<span style="display:block;font-size:0.65rem;color:{c["text_quaternary"]};'
        f'letter-spacing:0.02em;">Recruiter Intelligence · v2</span>'
        f'</span></span>',
        unsafe_allow_html=True,
    )


def sidebar_section_label(text: str) -> None:
    """Render a sidebar section label."""
    st.markdown(
        f'<span style="display:block;padding:0 12px;margin:16px 0 6px;">'
        f'<span style="font-size:0.62rem;font-weight:600;color:{c["text_quaternary"]};'
        f'text-transform:uppercase;letter-spacing:0.1em;">{text}</span></span>',
        unsafe_allow_html=True,
    )


def sidebar_health(index_ok: bool, model_ok: bool, submission_ok: bool) -> None:
    """Render sidebar health indicators."""
    def _dot(ok: bool) -> str:
        color = c['success'] if ok else c['text_quaternary']
        return (
            f'<span style="display:inline-block;width:6px;height:6px;'
            f'border-radius:50%;background:{color};margin-right:6px;"></span>'
        )

    st.markdown(
        f'<span style="display:block;padding:12px 16px;background:{c["bg_surface"]};'
        f'border:1px solid {c["border"]};border-radius:8px;margin:0 8px;">'
        f'<span style="display:block;font-size:0.62rem;font-weight:600;'
        f'color:{c["text_quaternary"]};text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-bottom:8px;">System</span>'
        f'<span style="display:block;font-size:0.75rem;color:{c["text_secondary"]};line-height:1.8;">'
        f'{_dot(index_ok)}Embeddings Index<br>'
        f'{_dot(model_ok)}LTR Model<br>'
        f'{_dot(submission_ok)}Submission'
        f'</span></span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════

def pipeline_node(label: str, done: bool, time_str: str = "", desc: str = "") -> None:
    """Render a pipeline execution node."""
    dot_color = c['success'] if done else c['text_quaternary']
    dot_shadow = f"box-shadow:0 0 6px {c['success']}55;" if done else ""
    desc_html = (
        f'<span style="color:{c["text_quaternary"]};font-size:0.68rem;'
        f'margin-left:6px;">— {desc}</span>'
        if desc else ""
    )

    st.markdown(
        f'<span style="display:flex;align-items:center;gap:12px;padding:10px 14px;'
        f'background:{c["bg_surface"]};border:1px solid {c["border"]};'
        f'border-radius:8px;margin-bottom:6px;">'
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{dot_color};{dot_shadow}flex-shrink:0;"></span>'
        f'<span style="flex:1;color:{c["text_primary"]};font-size:0.82rem;'
        f'font-weight:500;">{label}{desc_html}</span>'
        f'<span style="color:{c["text_tertiary"]};font-size:0.72rem;'
        f'font-family:monospace;">{time_str}</span>'
        f'</span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# CANDIDATE COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def candidate_row(
    rank: int, cid: str, title: str, company: str, yoe: float, score: float
) -> None:
    """Render a rich candidate row card."""
    rank_bg = f"{c['brand']}15" if rank <= 20 else c['bg_elevated']
    rank_color = c['brand'] if rank <= 20 else c['text_secondary']
    rank_border = f"{c['brand']}33" if rank <= 20 else c['border']
    pct = min(score * 100, 100)
    bar_c = c['success'] if score > 0.8 else c['brand'] if score > 0.4 else c['warning']

    st.markdown(
        f'<span style="display:flex;align-items:center;gap:14px;padding:12px 16px;'
        f'background:{c["bg_surface"]};border:1px solid {c["border"]};'
        f'border-radius:12px;margin-bottom:8px;">'
        # Rank badge
        f'<span style="display:flex;align-items:center;justify-content:center;'
        f'width:32px;height:32px;border-radius:8px;font-size:0.75rem;font-weight:700;'
        f'background:{rank_bg};color:{rank_color};border:1px solid {rank_border};'
        f'flex-shrink:0;">{rank}</span>'
        # Info
        f'<span style="flex:1;min-width:0;">'
        f'<span style="display:block;color:{c["text_primary"]};font-size:0.85rem;'
        f'font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</span>'
        f'<span style="display:block;color:{c["text_tertiary"]};font-size:0.73rem;'
        f'margin-top:2px;">{company} · {yoe:.1f} yr · {cid}</span>'
        f'</span>'
        # Score
        f'<span style="text-align:right;flex-shrink:0;">'
        f'<span style="display:block;color:{c["text_primary"]};font-size:0.9rem;'
        f'font-weight:700;font-family:monospace;">{score:.4f}</span>'
        f'<span style="display:block;width:64px;height:4px;background:{c["border"]};'
        f'border-radius:2px;margin-top:4px;overflow:hidden;">'
        f'<span style="display:block;height:100%;width:{pct}%;'
        f'background:{bar_c};border-radius:2px;"></span>'
        f'</span></span>'
        f'</span>',
        unsafe_allow_html=True,
    )


def candidate_detail_header(
    cid: str, title: str, company: str, location: str, yoe: float, score: float, rank: int
) -> None:
    """Render candidate profile header."""
    st.markdown(
        f'<span style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">'
        # Avatar
        f'<span style="display:flex;align-items:center;justify-content:center;'
        f'width:48px;height:48px;background:{c["bg_elevated"]};'
        f'border:1px solid {c["border"]};border-radius:12px;font-size:1.2rem;'
        f'color:{c["text_tertiary"]};">👤</span>'
        # Name & info
        f'<span style="flex:1;">'
        f'<span style="display:block;font-size:1.1rem;font-weight:700;'
        f'color:{c["text_primary"]};letter-spacing:-0.02em;">{cid}</span>'
        f'<span style="display:block;font-size:0.82rem;color:{c["text_secondary"]};'
        f'margin-top:2px;">{title} at {company} · {location} · {yoe:.1f} yrs</span>'
        f'</span>'
        # Rank
        f'<span style="text-align:right;">'
        f'<span style="display:block;font-size:0.65rem;color:{c["text_quaternary"]};'
        f'text-transform:uppercase;letter-spacing:0.06em;">Rank</span>'
        f'<span style="display:block;font-size:1.3rem;font-weight:700;'
        f'color:{c["brand"]};font-family:monospace;">#{rank}</span>'
        f'</span>'
        # Score
        f'<span style="text-align:right;margin-left:12px;">'
        f'<span style="display:block;font-size:0.65rem;color:{c["text_quaternary"]};'
        f'text-transform:uppercase;letter-spacing:0.06em;">Score</span>'
        f'<span style="display:block;font-size:1.3rem;font-weight:700;'
        f'color:{c["text_primary"]};font-family:monospace;">{score:.4f}</span>'
        f'</span>'
        f'</span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# CHIPS & TAGS
# ═══════════════════════════════════════════════════════════════════

def skill_chips(skills: list[str], max_show: int = 10, variant: str = "default") -> None:
    """Render skill chips in a flowing layout."""
    color_map = {
        "default": (f"{c['brand']}15", c['brand'], f"{c['brand']}22"),
        "accent": (f"{c['accent']}15", c['accent'], f"{c['accent']}22"),
        "muted": (c['bg_elevated'], c['text_tertiary'], c['border']),
    }
    bg, fg, border = color_map.get(variant, color_map["default"])

    chip_style = (
        f"display:inline-block;background:{bg};color:{fg};"
        f"border:1px solid {border};border-radius:6px;"
        f"padding:3px 8px;font-size:0.7rem;font-weight:500;"
        f"margin:2px 4px 2px 0;"
    )

    display = skills[:max_show]
    html = "".join(f'<span style="{chip_style}">{s}</span>' for s in display)
    if len(skills) > max_show:
        muted_bg, muted_fg, muted_b = color_map["muted"]
        muted_style = (
            f"display:inline-block;background:{muted_bg};color:{muted_fg};"
            f"border:1px solid {muted_b};border-radius:6px;"
            f"padding:3px 8px;font-size:0.7rem;font-weight:500;"
            f"margin:2px 4px 2px 0;"
        )
        html += f'<span style="{muted_style}">+{len(skills)-max_show}</span>'

    st.markdown(
        f'<span style="display:flex;flex-wrap:wrap;gap:2px;margin:8px 0;">{html}</span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# PANELS & BOXES
# ═══════════════════════════════════════════════════════════════════

def info_panel(text: str) -> None:
    """Render an accent-bordered info panel for reasoning/explanations."""
    st.markdown(
        f'<span style="display:block;background:{c["bg_surface"]};'
        f'border:1px solid {c["border"]};border-left:3px solid {c["accent"]};'
        f'border-radius:0 8px 8px 0;padding:14px 16px;margin:10px 0;'
        f'font-size:0.83rem;line-height:1.65;color:{c["text_secondary"]};">'
        f'{text}</span>',
        unsafe_allow_html=True,
    )


def score_bar(value: float, max_val: float = 1.0, color: str = "", height: int = 5) -> None:
    """Render a minimal score/progress bar."""
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    bar_color = color or c['brand']
    st.markdown(
        f'<span style="display:block;height:{height}px;background:{c["border"]};'
        f'border-radius:{height//2}px;overflow:hidden;">'
        f'<span style="display:block;height:100%;width:{pct}%;'
        f'background:{bar_color};border-radius:{height//2}px;"></span>'
        f'</span>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# LISTS
# ═══════════════════════════════════════════════════════════════════

def negative_list(items: list[str]) -> None:
    """Render a list of negative/penalty signals."""
    html = ""
    for item in items:
        html += (
            f'<span style="display:flex;align-items:center;gap:8px;padding:6px 0;">'
            f'<span style="display:inline-block;width:6px;height:6px;'
            f'border-radius:50%;background:{c["danger"]};flex-shrink:0;"></span>'
            f'<span style="color:{c["text_secondary"]};font-size:0.82rem;">{item}</span>'
            f'</span>'
        )
    st.markdown(html, unsafe_allow_html=True)


def positive_list(items: list[str]) -> None:
    """Render a list of positive/requirement signals."""
    html = ""
    for item in items:
        html += (
            f'<span style="display:flex;align-items:center;gap:8px;padding:5px 0;">'
            f'<span style="display:inline-block;width:6px;height:6px;'
            f'border-radius:50%;background:{c["success"]};flex-shrink:0;"></span>'
            f'<span style="color:{c["text_secondary"]};font-size:0.82rem;">{item}</span>'
            f'</span>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE BAR
# ═══════════════════════════════════════════════════════════════════

def feature_bar(name: str, value: float, max_value: float, color: str = "") -> None:
    """Render a horizontal feature importance bar with label and value."""
    pct = min(value / max_value * 100, 100) if max_value > 0 else 0
    bar_color = color or c['accent']
    st.markdown(
        f'<span style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
        f'<span style="width:180px;font-size:0.73rem;color:{c["text_secondary"]};'
        f'font-family:monospace;white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis;">{name}</span>'
        f'<span style="flex:1;height:6px;background:{c["border"]};'
        f'border-radius:3px;overflow:hidden;">'
        f'<span style="display:block;height:100%;width:{pct}%;'
        f'background:{bar_color};border-radius:3px;"></span></span>'
        f'<span style="width:40px;text-align:right;font-size:0.7rem;'
        f'color:{c["text_tertiary"]};font-family:monospace;">{value:.1f}</span>'
        f'</span>',
        unsafe_allow_html=True,
    )

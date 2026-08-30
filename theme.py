"""
שכבת עיצוב לדאשבורד: פלטת צבעים, CSS מותאם, ועיצוב אחיד לגרפים.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------- פלטה -----------------------------
INK = "#111827"
MUTED = "#6B7280"
SURFACE = "#FFFFFF"
CANVAS = "#F5F7F9"
BORDER = "#E6E8EB"
GRID = "#EEF1F4"
PRIMARY = "#0F766E"
ACCENT = "#D97706"

# רצף צבעים קטגוריאלי - נבדל ונגיש
CATEGORICAL = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#DB2777", "#0891B2"]
SEQUENTIAL = ["#E6F2F0", "#B7DED8", "#7FC3B9", "#46A899", "#1C8577", "#0F5F55"]
DIVERGING = "RdBu_r"

TIER_COLORS = {"עילית": "#0F766E", "גבוה": "#2563EB", "בינוני": "#D97706", "מתפתח": "#9CA3AF"}


# ----------------------------- CSS -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700;800&display=swap');

          html, body, [class*="css"], .stApp, button, input, select, textarea {
              font-family: 'Heebo', -apple-system, 'Segoe UI', sans-serif;
          }
          .stApp { direction: rtl; background: #F5F7F9; }
          section[data-testid="stSidebar"] { direction: rtl; background: #FFFFFF; border-left: 1px solid #E6E8EB; }

          /* יישור לימין (עברית) */
          .stApp, section[data-testid="stSidebar"] { text-align: right; }
          [data-testid="stMarkdownContainer"],
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li,
          [data-testid="stCaptionContainer"],
          .stAlert, .stAlert p,
          h1, h2, h3, h4, label, .stRadio, .stSelectbox label, .stSlider label {
              text-align: right;
          }
          [data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol {
              padding-right: 1.2rem; padding-left: 0;
          }
          /* קוד ומספרים נשארים LTR */
          code, pre, [data-testid="stDataFrame"] { direction: ltr; text-align: left; }

          /* פריסה */
          .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
          #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }

          /* כותרות */
          h1 { font-weight: 800 !important; color: #111827; letter-spacing: -0.02em; }
          h2, h3 {
              font-weight: 700 !important; color: #1F2937;
              border-right: 4px solid #0F766E; padding-right: 10px; margin-top: 1.4rem;
          }

          /* כרטיסי מדד (metric) */
          div[data-testid="stMetric"] {
              background: #FFFFFF; border: 1px solid #E6E8EB; border-radius: 14px;
              padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,0.05);
          }
          div[data-testid="stMetric"] > div { direction: rtl; }
          div[data-testid="stMetricLabel"] p { color: #6B7280; font-size: 0.82rem; font-weight: 500; }
          div[data-testid="stMetricValue"] { color: #0F766E; font-size: 1.7rem; font-weight: 800; direction: ltr; }

          /* טבלאות */
          div[data-testid="stDataFrame"] { border: 1px solid #E6E8EB; border-radius: 12px; overflow: hidden; }

          /* ניווט בסרגל הצד */
          section[data-testid="stSidebar"] div[role="radiogroup"] label {
              background: #F5F7F9; border: 1px solid #E6E8EB; border-radius: 10px;
              padding: 9px 12px; margin-bottom: 6px; transition: all .15s ease;
          }
          section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { border-color: #0F766E; }
          section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
              background: #0F766E; border-color: #0F766E;
          }
          section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: #FFFFFF; font-weight: 600; }

          /* באנר */
          .hero {
              background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #134E4A 100%);
              color: #FFFFFF; border-radius: 18px; padding: 24px 28px; margin-bottom: 20px;
              box-shadow: 0 6px 20px rgba(15,118,110,0.22); text-align: right; direction: rtl;
          }
          .hero h1 { color: #FFFFFF !important; margin: 0 0 4px 0; font-size: 1.7rem; }
          .hero p { color: #D1FAE5; margin: 0; font-size: 0.95rem; }

          /* תגית דרגה */
          .badge { display: inline-block; padding: 3px 12px; border-radius: 999px;
                   font-size: 0.8rem; font-weight: 600; color: #FFFFFF; }

          div[data-testid="stExpander"] { border: 1px solid #E6E8EB; border-radius: 12px; background: #FFFFFF; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


# ----------------------------- גרפים -----------------------------
_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="Heebo, sans-serif", size=13, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        colorway=CATEGORICAL,
        margin=dict(l=48, r=24, t=44, b=44),
        title=dict(font=dict(size=15, color="#1F2937"), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="outside",
                   tickcolor=BORDER, automargin=True),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="outside",
                   tickcolor=BORDER, automargin=True),
        hoverlabel=dict(font=dict(family="Heebo, sans-serif", size=12), bgcolor="#FFFFFF",
                        bordercolor=BORDER),
        colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
    )
)
pio.templates["tennis"] = _TEMPLATE


def style_fig(fig: go.Figure, height: int | None = None, showlegend: bool | None = None) -> go.Figure:
    fig.update_layout(template="tennis")
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def chart(fig: go.Figure, height: int | None = 360, **layout_kwargs) -> None:
    style_fig(fig, height=height)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

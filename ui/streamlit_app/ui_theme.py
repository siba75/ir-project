import streamlit as st


def apply_theme():
    colors = theme_colors(st.session_state.dark_mode)

    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {colors['bg']};
        color: {colors['text']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {colors['panel']};
        color: {colors['text']};
    }}

    section[data-testid="stSidebar"] * {{
        color: {colors['text']} !important;
    }}

    .main-title {{
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 0;
        color: {colors['text']};
    }}

    .subtitle {{
        font-size: 18px;
        color: {colors['muted']};
        margin-top: 0;
    }}

    .result-card {{
        padding: 22px;
        border-radius: 16px;
        border: 1px solid {colors['border']};
        margin-bottom: 18px;
        background-color: {colors['card']};
        color: {colors['text']};
    }}

    .doc-title {{
        font-size: 24px;
        font-weight: 700;
        color: {colors['text']};
    }}

    .small-muted {{
        color: {colors['muted']};
        font-size: 14px;
    }}

    .highlight {{
        background-color: #facc15;
        color: #111827;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 700;
    }}

    div[data-testid="stMetric"] {{
        background-color: {colors['card']};
        border: 1px solid {colors['border']};
        border-radius: 14px;
        padding: 14px;
    }}

    div[data-testid="stMetric"] * {{
        color: {colors['text']} !important;
    }}

    input, textarea {{
        background-color: {colors['input_bg']} !important;
        color: {colors['text']} !important;
        border: 1px solid {colors['border']} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {colors['panel']};
        color: {colors['text']};
        border-radius: 10px;
        padding: 10px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: #2563eb !important;
        color: white !important;
    }}

    .footer {{
        text-align: center;
        color: {colors['muted']};
        padding: 30px;
        font-size: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)


def theme_colors(dark_mode):
    if dark_mode:
        return {
            "bg": "#0b1120",
            "card": "#111827",
            "panel": "#1e293b",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "border": "#334155",
            "input_bg": "#0f172a",
        }

    return {
        "bg": "#ffffff",
        "card": "#ffffff",
        "panel": "#f8fafc",
        "text": "#1f2937",
        "muted": "#6b7280",
        "border": "#e5e7eb",
        "input_bg": "#ffffff",
    }

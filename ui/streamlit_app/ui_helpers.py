import json
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"


def load_evaluation_results():
    results = {}

    for report_path in REPORTS_DIR.glob("*_evaluation_results.json"):
        dataset_name = report_path.name.replace("_evaluation_results.json", "")

        with open(report_path, "r", encoding="utf-8") as file:
            results[dataset_name] = json.load(file)

    return results


def load_resource_manifest():
    manifest_path = BASE_DIR / "indexes" / "quora" / "resource_manifest.json"

    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file)


def metrics_table(report_section):
    return pd.DataFrame([
        {
            "Mode": mode,
            "Evaluated Queries": values.get("evaluated_queries", 0),
            "Precision@10": values.get("mean_precision@10", 0),
            "Recall@10": values.get("mean_recall@10", 0),
            "MRR": values.get("mrr", 0),
            "MAP": values.get("map", 0),
            "nDCG@10": values.get("ndcg@10", 0),
        }
        for mode, values in report_section.items()
    ])


def comparison_table(comparison_section):
    return pd.DataFrame([
        {
            "Mode": mode,
            "Δ Precision@10": values.get("precision@10_delta", 0),
            "Δ Recall@10": values.get("recall@10_delta", 0),
            "Δ MRR": values.get("mrr_delta", 0),
            "Δ MAP": values.get("map_delta", 0),
            "Δ nDCG@10": values.get("ndcg@10_delta", 0),
        }
        for mode, values in comparison_section.items()
    ])


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


def check_service(url):
    try:
        response = requests.get(url, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def extract_top_terms(refined_query):
    terms = re.findall(r"\w+", refined_query.lower())
    terms = [term for term in terms if len(term) > 2]
    return list(dict.fromkeys(terms))[:12]


def highlight_terms(text, query):
    terms = [
        term
        for term in re.findall(r"\w+", query.lower())
        if len(term) > 2
    ]
    highlighted = text

    for term in terms:
        highlighted = re.sub(
            f"(?i)({re.escape(term)})",
            r"<span class='highlight'>\1</span>",
            highlighted,
        )

    return highlighted


def get_score_value(item, parallel_key, serial_key):
    return item.get(parallel_key, item.get(serial_key, 0))


def results_to_dataframe(results):
    return pd.DataFrame([
        {
            "Rank": item.get("rank"),
            "Document": item.get("doc_id"),
            "Final Score": item.get("score", 0),
            "BM25 Score": get_score_value(item, "bm25_score", "bm25_candidate_score"),
            "Semantic Score": get_score_value(item, "semantic_score", "semantic_rerank_score"),
            "Text": item.get("text", ""),
        }
        for item in results
    ])


def cluster_results(results):
    texts = [item.get("text", "") for item in results]

    if len(texts) < 2:
        return pd.DataFrame()

    cluster_count = min(3, len(texts))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    return pd.DataFrame([
        {
            "Cluster": int(label) + 1,
            "Rank": item.get("rank"),
            "Document": item.get("doc_id"),
            "Score": item.get("score", 0),
        }
        for item, label in zip(results, labels)
    ])


def parameter_comparison_dataframe(previous_response, current_response):
    previous_by_doc = {
        str(item.get("doc_id")): item
        for item in previous_response.get("results", [])
    }
    rows = []

    for item in current_response.get("results", []):
        doc_id = str(item.get("doc_id"))
        previous_item = previous_by_doc.get(doc_id)

        rows.append({
            "Document": doc_id,
            "Current Rank": item.get("rank"),
            "Previous Rank": previous_item.get("rank") if previous_item else None,
            "Rank Change": (
                previous_item.get("rank") - item.get("rank")
                if previous_item else None
            ),
            "Current Score": item.get("score", 0),
            "Previous Score": previous_item.get("score", 0) if previous_item else None,
            "Score Change": (
                round(item.get("score", 0) - previous_item.get("score", 0), 6)
                if previous_item else None
            ),
        })

    return pd.DataFrame(rows)

import json
import logging
import re
from pathlib import Path

import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"


def load_evaluation_results():
    results = {}

    for report_path in REPORTS_DIR.glob("*_evaluation_results.json"):
        dataset_name = report_path.name.replace("_evaluation_results.json", "")

        try:
            with open(report_path, "r", encoding="utf-8") as file:
                results[dataset_name] = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load evaluation report %s: %s", report_path, exc)

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


def check_service(url):
    try:
        response = requests.get(url, timeout=3)
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    except requests.Timeout:
        logger.debug("Service health check timed out: %s", url)
        return False
    except requests.RequestException as exc:
        logger.warning("Service health check failed for %s: %s", url, exc)
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

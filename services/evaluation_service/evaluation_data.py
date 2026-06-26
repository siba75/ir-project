import json
import logging

from evaluation_config import EVALUATION_RUNS, SEARCH_MODES

logger = logging.getLogger(__name__)


def load_queries_and_qrels(base_dir, dataset_name):
    dataset_dir = base_dir / "datasets" / dataset_name

    with open(dataset_dir / "queries.json", "r", encoding="utf-8") as file:
        queries = json.load(file)

    with open(dataset_dir / "qrels.json", "r", encoding="utf-8") as file:
        qrels = json.load(file)

    return queries, qrels


def select_query_ids(qrels, max_queries=None):
    query_ids = list(qrels.keys())
    return query_ids if max_queries is None else query_ids[:max_queries]


def history_for_query(query_ids, queries, index, history_size):
    if history_size <= 0:
        return []

    start = max(0, index - history_size)
    previous_ids = query_ids[start:index]
    return [
        queries[query_id]
        for query_id in previous_ids
        if queries.get(query_id)
    ]


def build_pending_items(query_ids, queries, qrels, completed_set, history_size):
    items = []

    for index, query_id in enumerate(query_ids):
        if query_id in completed_set:
            continue

        query_text = queries.get(query_id)
        relevant_docs = qrels.get(query_id, [])

        if not query_text or not relevant_docs:
            logger.debug(
                "Skipping query_id=%s: missing %s",
                query_id,
                "query text" if not query_text else "relevant docs",
            )
            continue

        items.append({
            "query_id": query_id,
            "query_text": query_text,
            "relevant_docs": relevant_docs,
            "user_history": history_for_query(query_ids, queries, index, history_size),
        })

    return items


def load_before_results_from_report(report_path):
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    before_features = report.get("before_features", {})
    normalized_results = {}

    for mode in SEARCH_MODES:
        before_result = before_features.get(mode)

        if not before_result:
            raise ValueError(f"Could not find before_features/{mode} in {report_path}")

        normalized = dict(before_result)
        normalized["retrieval_mode"] = mode
        normalized["phase"] = "before_features"
        normalized["description"] = EVALUATION_RUNS[f"before_{mode}"]["description"]
        normalized_results[mode] = normalized

    return normalized_results

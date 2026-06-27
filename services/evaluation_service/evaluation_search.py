from evaluation_config import EVALUATION_RUNS, TOP_K
from metrics import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from personalization import build_personalized_query
from retrieval_core import (
    dataset_bm25_search,
    dataset_tfidf_search,
    hybrid_search,
    hybrid_serial_search,
    semantic_search,
)


def run_search(mode, query_text, dataset_name):
    if mode == "tfidf":
        return dataset_tfidf_search(query_text, TOP_K, dataset_name)

    if mode == "bm25":
        return dataset_bm25_search(query_text, TOP_K, dataset_name, 1.5, 0.75)

    if mode == "semantic":
        return semantic_search(query_text, TOP_K, dataset_name)

    if mode == "hybrid_parallel":
        return hybrid_search(query_text, TOP_K, 0.4, 0.6, dataset_name)

    if mode == "hybrid_serial":
        return hybrid_serial_search(query_text, TOP_K, 50, dataset_name)

    raise ValueError(f"Unsupported search mode: {mode}")


def query_for_phase(run_config, query_text, user_history):
    if run_config["phase"] == "before_features":
        return query_text

    personalized_query, _, _ = build_personalized_query(query_text, user_history)
    return personalized_query


def run_evaluation_item(run_name, dataset_name, query_id, query_text, relevant_docs, user_history):
    run_config = EVALUATION_RUNS[run_name]
    effective_query = query_for_phase(run_config, query_text, user_history)
    data = run_search(run_config["mode"], effective_query, dataset_name)
    retrieved_docs = [
        str(item["doc_id"])
        for item in data.get("results", [])
    ]

    return {
        "query_id": query_id,
        "precision": precision_at_k(retrieved_docs, relevant_docs, TOP_K),
        "recall": recall_at_k(retrieved_docs, relevant_docs, TOP_K),
        "mrr": mean_reciprocal_rank(retrieved_docs, relevant_docs),
        "map": average_precision(retrieved_docs, relevant_docs),
        "ndcg": ndcg_at_k(retrieved_docs, relevant_docs, TOP_K),
    }

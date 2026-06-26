from fastapi import HTTPException

from dataset_manager import load_dataset_resources, validate_dataset
from lexical_retrieval import lexical_scores, tokenize_query
from retrieval_common import (
    get_doc_text,
    normalize_scores,
    validate_bm25_parameters,
    validate_query,
)
from semantic_retrieval import get_vector_method, semantic_scores


def hybrid_search(
    query: str,
    top_k: int,
    bm25_weight: float,
    semantic_weight: float,
    dataset: str,
    k1: float = 1.5,
    b: float = 0.75,
):
    dataset = validate_dataset(dataset)
    validate_query(query)
    validate_bm25_parameters(k1, b)
    validate_hybrid_weights(bm25_weight, semantic_weight)
    resources = load_dataset_resources(dataset)
    query_tokens = tokenize_query(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    bm25_scores = normalize_scores(lexical_scores(query_tokens, resources, k1=k1, b=b))
    semantic_raw = semantic_scores(query, resources, search_size=max(top_k * 5, 50))
    semantic_normalized = normalize_scores(semantic_raw)
    final_scores = fuse_parallel_scores(bm25_scores, semantic_normalized, bm25_weight, semantic_weight)
    ranked_results = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    results = hybrid_parallel_results(ranked_results[:top_k], bm25_scores, semantic_normalized, resources)

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Hybrid Parallel Search (BM25 + Semantic FAISS Score Fusion)",
        "vector_method": get_vector_method(resources),
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "weights": {
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight,
        },
        "parameters": {"k1": k1, "b": b},
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results,
    }


def hybrid_serial_search(
    query: str,
    top_k: int,
    initial_k: int,
    dataset: str,
    k1: float = 1.5,
    b: float = 0.75,
):
    dataset = validate_dataset(dataset)
    validate_query(query)
    validate_bm25_parameters(k1, b)
    validate_serial_parameters(top_k, initial_k)
    initial_k = max(initial_k, top_k)
    resources = load_dataset_resources(dataset)
    query_tokens = tokenize_query(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    bm25_scores_raw = lexical_scores(query_tokens, resources, k1=k1, b=b)
    bm25_scores = normalize_scores(bm25_scores_raw)
    ranked_bm25 = sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)
    candidate_docs = ranked_bm25[:initial_k]

    if not candidate_docs:
        return empty_hybrid_serial_response(query, query_tokens, dataset, resources, initial_k, k1, b)

    candidate_doc_ids = {doc_id for doc_id, _ in candidate_docs}
    semantic_raw = semantic_scores(query, resources, search_size=max(initial_k * 3, top_k))
    semantic_filtered = {
        doc_id: score
        for doc_id, score in semantic_raw.items()
        if doc_id in candidate_doc_ids
    }
    semantic_normalized = normalize_scores(semantic_filtered)
    final_scores = {
        doc_id: semantic_normalized.get(doc_id, 0.0)
        for doc_id in candidate_doc_ids
    }
    ranked_results = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    results = hybrid_serial_results(ranked_results[:top_k], bm25_scores, semantic_normalized, resources)

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Hybrid Serial Search (BM25 Candidate Generation -> Semantic Re-ranking)",
        "vector_method": get_vector_method(resources),
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "parameters": {"k1": k1, "b": b},
        "initial_candidate_count": initial_k,
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results,
    }


def validate_hybrid_weights(bm25_weight: float, semantic_weight: float):
    if bm25_weight < 0 or semantic_weight < 0:
        raise HTTPException(status_code=400, detail="Weights must be non-negative")

    if bm25_weight + semantic_weight == 0:
        raise HTTPException(status_code=400, detail="At least one weight must be greater than 0")


def validate_serial_parameters(top_k: int, initial_k: int):
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    if initial_k <= 0:
        raise HTTPException(status_code=400, detail="initial_k must be greater than 0")


def fuse_parallel_scores(bm25_scores, semantic_scores_normalized, bm25_weight, semantic_weight):
    all_doc_ids = set(bm25_scores.keys()) | set(semantic_scores_normalized.keys())

    return {
        doc_id: (
            bm25_weight * bm25_scores.get(doc_id, 0.0)
            + semantic_weight * semantic_scores_normalized.get(doc_id, 0.0)
        )
        for doc_id in all_doc_ids
    }


def hybrid_parallel_results(ranked_results, bm25_scores, semantic_normalized, resources):
    return [
        {
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "bm25_score": round(float(bm25_scores.get(doc_id, 0.0)), 6),
            "semantic_score": round(float(semantic_normalized.get(doc_id, 0.0)), 6),
            "text": get_doc_text(doc_id, resources),
        }
        for rank, (doc_id, score) in enumerate(ranked_results, start=1)
    ]


def hybrid_serial_results(ranked_results, bm25_scores, semantic_normalized, resources):
    return [
        {
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "bm25_candidate_score": round(float(bm25_scores.get(doc_id, 0.0)), 6),
            "semantic_rerank_score": round(float(semantic_normalized.get(doc_id, 0.0)), 6),
            "text": get_doc_text(doc_id, resources),
        }
        for rank, (doc_id, score) in enumerate(ranked_results, start=1)
    ]


def empty_hybrid_serial_response(query, query_tokens, dataset, resources, initial_k, k1, b):
    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Hybrid Serial Search (BM25 Candidate Generation -> Semantic Re-ranking)",
        "vector_method": get_vector_method(resources),
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "parameters": {"k1": k1, "b": b},
        "initial_candidate_count": initial_k,
        "returned_results": 0,
        "results": [],
    }

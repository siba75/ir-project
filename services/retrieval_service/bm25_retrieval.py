from fastapi import HTTPException

from dataset_manager import load_dataset_resources, validate_dataset
from lexical_retrieval import lexical_scores, tokenize_query
from retrieval_common import (
    ranked_doc_results,
    validate_bm25_parameters,
    validate_query,
)


def dataset_bm25_search(query: str, top_k: int, dataset: str, k1: float, b: float):
    dataset = validate_dataset(dataset)
    validate_query(query)
    validate_bm25_parameters(k1, b)
    resources = load_dataset_resources(dataset)
    query_tokens = tokenize_query(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    scores = lexical_scores(query_tokens, resources, k1=k1, b=b)
    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results = ranked_doc_results(ranked_results[:top_k], resources)

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "rank_bm25 BM25Okapi",
        "parameters": {"k1": k1, "b": b},
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results,
    }

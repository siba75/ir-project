import numpy as np
from fastapi import HTTPException

from dataset_manager import load_dataset_resources, validate_dataset
from retrieval_common import get_doc_text, validate_query


def dataset_tfidf_search(query: str, top_k: int, dataset: str):
    dataset = validate_dataset(dataset)
    validate_query(query)
    resources = load_dataset_resources(dataset)
    tfidf_data = resources.get("tfidf", {})
    vectorizer = tfidf_data.get("vectorizer")
    matrix = tfidf_data.get("matrix")
    doc_ids = tfidf_data.get("doc_ids", resources["doc_ids"])

    if vectorizer is None or matrix is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF resource is missing. Run prepare_submission_resources.py.",
        )

    query_vector = vectorizer.transform([query])

    if query_vector.nnz == 0:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    raw_scores = (matrix @ query_vector.T).toarray().ravel()
    ranked_indices = top_tfidf_indices(raw_scores, top_k)
    results = tfidf_results(ranked_indices, raw_scores, doc_ids, resources)

    return {
        "query": query,
        "model": "sklearn TfidfVectorizer cosine similarity",
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results,
    }


def top_tfidf_indices(raw_scores, top_k: int):
    candidate_count = min(top_k, raw_scores.shape[0])

    if candidate_count == 0:
        return []

    candidate_indices = np.argpartition(raw_scores, -candidate_count)[-candidate_count:]
    return candidate_indices[np.argsort(raw_scores[candidate_indices])[::-1]]


def tfidf_results(ranked_indices, raw_scores, doc_ids, resources):
    results = []

    for rank, doc_index in enumerate(ranked_indices, start=1):
        score = raw_scores[doc_index]

        if score <= 0:
            continue

        doc_id = str(doc_ids[doc_index])
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "text": get_doc_text(doc_id, resources),
        })

    return results

from fastapi import HTTPException


VECTOR_METHOD_ALIASES = {
    "lsa": "lsa_tfidf_svd",
    "tfidf_svd": "lsa_tfidf_svd",
    "lsa_tfidf_svd": "lsa_tfidf_svd",
    "transformer": "sentence_transformer",
    "sentence_transformer": "sentence_transformer",
    "sentence-transformer": "sentence_transformer",
    "sentence_transformers": "sentence_transformer",
    "sentence-transformers": "sentence_transformer",
}


def normalize_scores(scores: dict):
    if not scores:
        return {}

    max_score = max(scores.values())

    if max_score <= 0:
        return scores

    return {
        doc_id: score / max_score
        for doc_id, score in scores.items()
    }


def normalize_vector_method(vector_method: str | None):
    if not vector_method:
        return None

    normalized_name = str(vector_method).lower().strip()
    return VECTOR_METHOD_ALIASES.get(normalized_name, normalized_name)


def get_doc_text(doc_id: str, resources):
    document_store = resources.get("document_store")

    if document_store:
        text = document_store.get(doc_id)

        if text:
            return text

    metadata = resources.get("metadata", {})
    doc_ids = [str(item) for item in metadata.get("doc_ids", [])]

    try:
        position = doc_ids.index(str(doc_id))
        return metadata["documents"][position]
    except (ValueError, KeyError, IndexError):
        return ""


def validate_query(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")


def validate_bm25_parameters(k1: float, b: float):
    if k1 <= 0:
        raise HTTPException(status_code=400, detail="k1 must be greater than 0")

    if b < 0 or b > 1:
        raise HTTPException(status_code=400, detail="b must be between 0 and 1")


def ranked_doc_results(ranked_results, resources):
    return [
        {
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "text": get_doc_text(doc_id, resources),
        }
        for rank, (doc_id, score) in enumerate(ranked_results, start=1)
    ]

from functools import lru_cache
from pathlib import Path
import os
import re
import ssl

import certifi
import numpy as np
from fastapi import HTTPException
from sklearn.preprocessing import normalize

from dataset_manager import load_dataset_resources, validate_dataset


BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_TEMP_DIR = BASE_DIR / "reports" / "runtime_cache" / "temp"

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


def configure_ssl_for_ml_imports():
    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(LOCAL_TEMP_DIR))
    os.environ.setdefault("TEMP", str(LOCAL_TEMP_DIR))
    os.environ.setdefault("TMPDIR", str(LOCAL_TEMP_DIR))

    original_create_default_context = ssl.create_default_context

    def create_certifi_context(*args, **kwargs):
        return original_create_default_context(cafile=certifi.where())

    ssl.create_default_context = create_certifi_context


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def lexical_scores(
    query_tokens: list[str],
    resources,
    k1: float | None = None,
    b: float | None = None,
):
    bm25 = resources["bm25"]
    doc_ids = resources["doc_ids"]
    original_k1 = getattr(bm25, "k1", None)
    original_b = getattr(bm25, "b", None)

    try:
        if k1 is not None:
            bm25.k1 = k1

        if b is not None:
            bm25.b = b

        raw_scores = bm25.get_scores(query_tokens)
    finally:
        if original_k1 is not None:
            bm25.k1 = original_k1

        if original_b is not None:
            bm25.b = original_b

    return {
        str(doc_ids[index]): float(score)
        for index, score in enumerate(raw_scores)
        if score > 0
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


@lru_cache(maxsize=2)
def get_sentence_transformer_model(model_name: str):
    configure_ssl_for_ml_imports()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def semantic_scores(query: str, resources, search_size: int):
    faiss_index = resources.get("faiss_index")
    metadata = resources.get("metadata", {})

    if faiss_index is None:
        raise HTTPException(
            status_code=500,
            detail="FAISS index was not loaded. Please build vector resources first.",
        )

    vector_method = normalize_vector_method(metadata.get("vector_method"))

    if vector_method == "lsa_tfidf_svd":
        query_embedding = lsa_query_embedding(query, metadata)
    elif vector_method == "sentence_transformer":
        query_embedding = transformer_query_embedding(query, metadata)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported vector method: {vector_method}. "
            "Supported methods are: lsa, lsa_tfidf_svd, transformer, sentence_transformer.",
        )

    total_documents = int(resources.get("total_documents", 0))

    if total_documents <= 0:
        raise HTTPException(status_code=500, detail="No documents found in loaded resources.")

    search_limit = min(search_size, total_documents)
    scores_raw, indices = faiss_index.search(query_embedding, search_limit)
    doc_ids = metadata.get("doc_ids", [])
    scores = {}

    for doc_index, score in zip(indices[0], scores_raw[0]):
        if doc_index == -1 or doc_index >= len(doc_ids):
            continue

        scores[str(doc_ids[doc_index])] = float(score)

    return scores


def lsa_query_embedding(query: str, metadata: dict):
    if "vectorizer" not in metadata or "svd" not in metadata:
        raise HTTPException(
            status_code=500,
            detail="LSA vectorizer or SVD model is missing from metadata.",
        )

    query_tfidf = metadata["vectorizer"].transform([query])
    query_embedding = metadata["svd"].transform(query_tfidf)
    return normalize(query_embedding).astype("float32")


def transformer_query_embedding(query: str, metadata: dict):
    model_name = metadata.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    model = get_sentence_transformer_model(model_name)

    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def get_vector_method(resources):
    metadata = resources.get("metadata", {})
    return normalize_vector_method(metadata.get("vector_method")) or "unknown"


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
    candidate_count = min(top_k, raw_scores.shape[0])

    if candidate_count == 0:
        ranked_indices = []
    else:
        candidate_indices = np.argpartition(raw_scores, -candidate_count)[-candidate_count:]
        ranked_indices = candidate_indices[np.argsort(raw_scores[candidate_indices])[::-1]]

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

    return {
        "query": query,
        "model": "sklearn TfidfVectorizer cosine similarity",
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results,
    }


def dataset_bm25_search(query: str, top_k: int, dataset: str, k1: float, b: float):
    dataset = validate_dataset(dataset)
    validate_query(query)
    validate_bm25_parameters(k1, b)
    resources = load_dataset_resources(dataset)
    query_tokens = preprocess(query)

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


def semantic_search(query: str, top_k: int, dataset: str):
    dataset = validate_dataset(dataset)
    validate_query(query)
    resources = load_dataset_resources(dataset)
    metadata = resources.get("metadata", {})
    scores = semantic_scores(query, resources, top_k)
    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results = ranked_doc_results(ranked_results[:top_k], resources)

    return {
        "query": query,
        "model": "Semantic Search using FAISS Vector Index",
        "vector_method": metadata.get("vector_method", "N/A"),
        "embedding_dimension": metadata.get("embedding_dimension", "N/A"),
        "dataset": dataset,
        "storage": resources.get("storage", {}),
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results,
    }


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

    if bm25_weight < 0 or semantic_weight < 0:
        raise HTTPException(status_code=400, detail="Weights must be non-negative")

    if bm25_weight + semantic_weight == 0:
        raise HTTPException(status_code=400, detail="At least one weight must be greater than 0")

    resources = load_dataset_resources(dataset)
    query_tokens = preprocess(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    bm25_scores = normalize_scores(lexical_scores(query_tokens, resources, k1=k1, b=b))
    semantic_raw = semantic_scores(query, resources, search_size=max(top_k * 5, 50))
    semantic_normalized = normalize_scores(semantic_raw)
    final_scores = fuse_parallel_scores(bm25_scores, semantic_normalized, bm25_weight, semantic_weight)
    ranked_results = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

    results = [
        {
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "bm25_score": round(float(bm25_scores.get(doc_id, 0.0)), 6),
            "semantic_score": round(float(semantic_normalized.get(doc_id, 0.0)), 6),
            "text": get_doc_text(doc_id, resources),
        }
        for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1)
    ]

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

    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    if initial_k <= 0:
        raise HTTPException(status_code=400, detail="initial_k must be greater than 0")

    if initial_k < top_k:
        initial_k = top_k

    resources = load_dataset_resources(dataset)
    query_tokens = preprocess(query)

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

    results = [
        {
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "bm25_candidate_score": round(float(bm25_scores.get(doc_id, 0.0)), 6),
            "semantic_rerank_score": round(float(semantic_normalized.get(doc_id, 0.0)), 6),
            "text": get_doc_text(doc_id, resources),
        }
        for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1)
    ]

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


def fuse_parallel_scores(bm25_scores, semantic_scores_normalized, bm25_weight, semantic_weight):
    all_doc_ids = set(bm25_scores.keys()) | set(semantic_scores_normalized.keys())

    return {
        doc_id: (
            bm25_weight * bm25_scores.get(doc_id, 0.0)
            + semantic_weight * semantic_scores_normalized.get(doc_id, 0.0)
        )
        for doc_id in all_doc_ids
    }


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

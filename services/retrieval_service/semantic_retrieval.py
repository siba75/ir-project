from functools import lru_cache
from pathlib import Path
import os
import ssl

import certifi
from fastapi import HTTPException
from sklearn.preprocessing import normalize

from dataset_manager import load_dataset_resources, validate_dataset
from retrieval_common import normalize_vector_method
from retrieval_common import ranked_doc_results, validate_query


BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_TEMP_DIR = BASE_DIR / "reports" / "runtime_cache" / "temp"


def configure_ssl_for_ml_imports():
    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(LOCAL_TEMP_DIR))
    os.environ.setdefault("TEMP", str(LOCAL_TEMP_DIR))
    os.environ.setdefault("TMPDIR", str(LOCAL_TEMP_DIR))

    original_create_default_context = ssl.create_default_context

    def create_certifi_context(*args, **kwargs):
        return original_create_default_context(cafile=certifi.where())

    ssl.create_default_context = create_certifi_context


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

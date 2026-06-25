from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx
import numpy as np
import re


app = FastAPI(
    title="IR Gateway Service",
    description="Central gateway for the complete IR pipeline",
    version="2.0.0"
)

REFINEMENT_SERVICE_URL = "http://127.0.0.1:8005/refine"

TFIDF_SEARCH_URL = "http://127.0.0.1:8003/search/dataset/tfidf"
BM25_SEARCH_URL = "http://127.0.0.1:8003/search/dataset/bm25"
SEMANTIC_SEARCH_URL = "http://127.0.0.1:8003/search/semantic"
HYBRID_PARALLEL_SEARCH_URL = "http://127.0.0.1:8003/search/hybrid"
HYBRID_SERIAL_SEARCH_URL = "http://127.0.0.1:8003/search/hybrid/serial"

SUPPORTED_DATASETS = ["quora"]

SUPPORTED_RETRIEVAL_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial"
]


class FullSearchRequest(BaseModel):
    query: str
    top_k: int = 5

    dataset: str = "quora"
    retrieval_mode: str = "hybrid_parallel"

    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    initial_k: int = 50

    remove_stopwords: bool = True
    use_stemming: bool = False
    use_lemmatization: bool = False
    use_expansion: bool = True

    use_personalization: bool = False
    user_history: list[str] = []


def build_tfidf_payload(request: FullSearchRequest, refined_query: str):
    return TFIDF_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "dataset": request.dataset
    }


def build_bm25_payload(request: FullSearchRequest, refined_query: str):
    return BM25_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "dataset": request.dataset,
        "k1": request.bm25_k1,
        "b": request.bm25_b
    }


def build_semantic_payload(request: FullSearchRequest, refined_query: str):
    return SEMANTIC_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "dataset": request.dataset
    }


def build_hybrid_parallel_payload(request: FullSearchRequest, refined_query: str):
    return HYBRID_PARALLEL_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "bm25_weight": request.bm25_weight,
        "semantic_weight": request.semantic_weight,
        "k1": request.bm25_k1,
        "b": request.bm25_b,
        "dataset": request.dataset
    }


def build_hybrid_serial_payload(request: FullSearchRequest, refined_query: str):
    return HYBRID_SERIAL_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "initial_k": request.initial_k,
        "k1": request.bm25_k1,
        "b": request.bm25_b,
        "dataset": request.dataset
    }


RETRIEVAL_STRATEGIES = {
    "tfidf": build_tfidf_payload,
    "bm25": build_bm25_payload,
    "semantic": build_semantic_payload,
    "hybrid_parallel": build_hybrid_parallel_payload,
    "hybrid_serial": build_hybrid_serial_payload,
}


def clean_history_queries(user_history: list[str], current_query: str):
    cleaned_queries = []
    current_normalized = current_query.lower().strip()

    for query in user_history[-20:]:
        normalized_query = re.sub(r"\s+", " ", query.lower().strip())

        if not normalized_query:
            continue

        if normalized_query == current_normalized:
            continue

        cleaned_queries.append(normalized_query)

    return cleaned_queries


def vector_top_terms(vector, feature_names, existing_terms, limit=6):
    dense_vector = np.asarray(vector).ravel()
    ranked_indices = dense_vector.argsort()[::-1]
    terms = []

    for index in ranked_indices:
        score = float(dense_vector[index])
        term = feature_names[index]

        if score <= 0:
            break

        if term in existing_terms:
            continue

        if len(term) < 3:
            continue

        terms.append({
            "term": term,
            "score": round(score, 6),
        })

        if len(terms) == limit:
            break

    return terms


def build_personalized_query(refined_query: str, user_history: list[str]):
    history_queries = clean_history_queries(user_history, refined_query)

    empty_profile = {
        "enabled": False,
        "method": "TF-IDF history vector profile",
        "history_queries_used": 0,
        "interest_terms": [],
        "combined_terms": [],
        "similar_history_queries": [],
        "query_suggestions": [],
    }

    if not history_queries:
        return refined_query, [], empty_profile

    corpus = history_queries + [refined_query]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=300,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return refined_query, [], empty_profile

    history_matrix = matrix[:-1]
    query_vector = matrix[-1]
    feature_names = vectorizer.get_feature_names_out()
    existing_terms = set(re.findall(r"[a-zA-Z0-9]+", refined_query.lower()))

    similarities = (history_matrix @ query_vector.T).toarray().ravel()
    recency_weights = np.linspace(0.6, 1.0, num=len(history_queries))
    history_weights = (similarities + 0.08) * recency_weights
    weight_sum = float(history_weights.sum())

    if weight_sum == 0:
        history_weights = recency_weights
        weight_sum = float(history_weights.sum())

    interest_vector = (history_matrix.T @ history_weights) / weight_sum
    interest_vector = np.asarray(interest_vector).ravel()

    combined_vector = (
        0.70 * query_vector.toarray().ravel()
        + 0.30 * interest_vector
    )

    interest_terms = vector_top_terms(
        interest_vector,
        feature_names,
        existing_terms,
        limit=8,
    )
    combined_terms = vector_top_terms(
        combined_vector,
        feature_names,
        existing_terms,
        limit=6,
    )

    selected_terms = []

    for item in combined_terms + interest_terms:
        if item["term"] not in selected_terms:
            selected_terms.append(item["term"])

        if len(selected_terms) == 6:
            break

    similar_indices = similarities.argsort()[::-1][:5]
    similar_history = [
        {
            "query": history_queries[index],
            "similarity": round(float(similarities[index]), 6),
        }
        for index in similar_indices
        if similarities[index] > 0
    ]

    if not similar_history and history_queries:
        similar_history = [
            {
                "query": history_queries[-1],
                "similarity": 0.0,
            }
        ]

    query_suggestions = []

    for term in selected_terms[:4]:
        query_suggestions.append(f"{refined_query} {term}")

    for item in similar_history[:2]:
        if item["query"] not in query_suggestions:
            query_suggestions.append(item["query"])

    personalized_query = refined_query

    if selected_terms:
        personalized_query = f"{refined_query} {' '.join(selected_terms)}"

    profile = {
        "enabled": True,
        "method": "TF-IDF history vector profile + query-interest vector fusion",
        "history_queries_used": len(history_queries),
        "query_vector_weight": 0.70,
        "interest_vector_weight": 0.30,
        "interest_terms": interest_terms,
        "combined_terms": combined_terms,
        "similar_history_queries": similar_history,
        "query_suggestions": query_suggestions[:6],
    }

    return personalized_query, selected_terms, profile


@app.get("/")
def home():
    return {
        "service": "Gateway Service",
        "status": "running",
        "version": "2.0.0",
        "available_datasets": SUPPORTED_DATASETS,
        "available_retrieval_modes": SUPPORTED_RETRIEVAL_MODES,
        "pipeline": [
            "query_refinement",
            "retrieval",
            "ranking"
        ],
        "dataset": {
            "name": "quora",
            "source": "beir/quora/test"
        }
    }


@app.post("/search/full")
async def full_search(request: FullSearchRequest):
    request.dataset = request.dataset.lower().strip()
    request.retrieval_mode = request.retrieval_mode.lower().strip()

    if request.dataset not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"dataset must be one of {SUPPORTED_DATASETS}"
        )

    if request.retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"retrieval_mode must be one of {SUPPORTED_RETRIEVAL_MODES}"
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0"
        )

    if request.initial_k < request.top_k:
        request.initial_k = request.top_k

    if request.bm25_weight < 0 or request.semantic_weight < 0:
        raise HTTPException(
            status_code=400,
            detail="weights must be non-negative"
        )

    if request.bm25_k1 <= 0:
        raise HTTPException(
            status_code=400,
            detail="bm25_k1 must be greater than 0"
        )

    if request.bm25_b < 0 or request.bm25_b > 1:
        raise HTTPException(
            status_code=400,
            detail="bm25_b must be between 0 and 1"
        )

    if request.retrieval_mode == "hybrid_parallel":
        if request.bm25_weight + request.semantic_weight == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one ranking weight must be greater than 0"
            )

    refinement_payload = {
        "query": request.query,
        "remove_stopwords": request.remove_stopwords,
        "use_stemming": request.use_stemming,
        "use_lemmatization": request.use_lemmatization,
        "use_expansion": request.use_expansion
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            refinement_response = await client.post(
                REFINEMENT_SERVICE_URL,
                json=refinement_payload
            )
            refinement_response.raise_for_status()

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Refinement service error: {str(error)}"
            )

        refinement_data = refinement_response.json()
        refined_query = refinement_data["refined_query"]

        personalization_terms = []
        personalization_profile = {
            "enabled": False,
            "method": "TF-IDF history vector profile",
            "history_queries_used": 0,
            "interest_terms": [],
            "combined_terms": [],
            "similar_history_queries": [],
            "query_suggestions": [],
        }

        if request.use_personalization:
            refined_query, personalization_terms, personalization_profile = build_personalized_query(
                refined_query,
                request.user_history
            )

        retrieval_strategy = RETRIEVAL_STRATEGIES[request.retrieval_mode]
        retrieval_url, retrieval_payload = retrieval_strategy(
            request,
            refined_query
        )

        try:
            retrieval_response = await client.post(
                retrieval_url,
                json=retrieval_payload
            )
            retrieval_response.raise_for_status()

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval service error: {str(error)}"
            )

        retrieval_data = retrieval_response.json()

    return {
        "original_query": request.query,
        "refined_query": refined_query,
        "dataset": request.dataset,
        "retrieval_mode": request.retrieval_mode,
        "retrieval_model": retrieval_data.get("model"),
        "vector_method": retrieval_data.get("vector_method"),
        "storage": retrieval_data.get("storage", {}),
        "pipeline": {
            "refinement_enabled": True,
            "retrieval_enabled": True,
            "ranking_enabled": True
        },
        "additional_features": {
            "vector_store_faiss": True,
            "personalization_enabled": request.use_personalization,
            "personalization_terms": personalization_terms,
            "personalization_profile": personalization_profile,
            "query_suggestions": personalization_profile.get("query_suggestions", [])
        },
        "configuration": {
            "dataset": request.dataset,
            "top_k": request.top_k,
            "bm25_weight": request.bm25_weight,
            "semantic_weight": request.semantic_weight,
            "bm25_k1": request.bm25_k1,
            "bm25_b": request.bm25_b,
            "initial_k": request.initial_k,
            "remove_stopwords": request.remove_stopwords,
            "use_stemming": request.use_stemming,
            "use_lemmatization": request.use_lemmatization,
            "use_expansion": request.use_expansion,
            "use_personalization": request.use_personalization
        },
        "returned_results": retrieval_data.get("returned_results", 0),
        "results": retrieval_data.get("results", [])
    }

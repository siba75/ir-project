from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
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


def build_personalized_query(refined_query: str, user_history: list[str]):
    history_terms = []

    for history_query in user_history[-5:]:
        terms = re.findall(r"[a-zA-Z0-9]+", history_query.lower())
        history_terms.extend(term for term in terms if len(term) > 2)

    unique_terms = list(dict.fromkeys(history_terms))[:8]

    if not unique_terms:
        return refined_query, []

    personalized_query = f"{refined_query} {' '.join(unique_terms)}"

    return personalized_query, unique_terms


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

        if request.use_personalization:
            refined_query, personalization_terms = build_personalized_query(
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
            "personalization_terms": personalization_terms
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

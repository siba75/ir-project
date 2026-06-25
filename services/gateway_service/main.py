from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

from personalization import build_personalized_query, empty_profile
from retrieval_strategies import RETRIEVAL_STRATEGIES


REFINEMENT_SERVICE_URL = "http://127.0.0.1:8005/refine"
SUPPORTED_DATASETS = ["quora"]
SUPPORTED_RETRIEVAL_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial",
]

app = FastAPI(
    title="IR Gateway Service",
    description="Central gateway for the complete IR pipeline",
    version="2.0.0",
)


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


def validate_request(request: FullSearchRequest):
    request.dataset = request.dataset.lower().strip()
    request.retrieval_mode = request.retrieval_mode.lower().strip()

    if request.dataset not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"dataset must be one of {SUPPORTED_DATASETS}",
        )

    if request.retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"retrieval_mode must be one of {SUPPORTED_RETRIEVAL_MODES}",
        )

    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    if request.initial_k < request.top_k:
        request.initial_k = request.top_k

    if request.bm25_weight < 0 or request.semantic_weight < 0:
        raise HTTPException(status_code=400, detail="weights must be non-negative")

    if request.bm25_k1 <= 0:
        raise HTTPException(status_code=400, detail="bm25_k1 must be greater than 0")

    if request.bm25_b < 0 or request.bm25_b > 1:
        raise HTTPException(status_code=400, detail="bm25_b must be between 0 and 1")

    if (
        request.retrieval_mode == "hybrid_parallel"
        and request.bm25_weight + request.semantic_weight == 0
    ):
        raise HTTPException(
            status_code=400,
            detail="At least one ranking weight must be greater than 0",
        )


def build_refinement_payload(request: FullSearchRequest):
    return {
        "query": request.query,
        "remove_stopwords": request.remove_stopwords,
        "use_stemming": request.use_stemming,
        "use_lemmatization": request.use_lemmatization,
        "use_expansion": request.use_expansion,
    }


async def post_json(client: httpx.AsyncClient, url: str, payload: dict, label: str):
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"{label} service error: {str(error)}",
        )


def apply_personalization(request: FullSearchRequest, refined_query: str):
    if not request.use_personalization:
        return refined_query, [], empty_profile()

    return build_personalized_query(refined_query, request.user_history)


def build_full_response(
    request: FullSearchRequest,
    refined_query: str,
    retrieval_data: dict,
    personalization_terms: list[str],
    personalization_profile: dict,
):
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
            "ranking_enabled": True,
        },
        "additional_features": {
            "vector_store_faiss": True,
            "personalization_enabled": request.use_personalization,
            "personalization_terms": personalization_terms,
            "personalization_profile": personalization_profile,
            "query_suggestions": personalization_profile.get("query_suggestions", []),
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
            "use_personalization": request.use_personalization,
        },
        "returned_results": retrieval_data.get("returned_results", 0),
        "results": retrieval_data.get("results", []),
    }


@app.get("/")
def home():
    return {
        "service": "Gateway Service",
        "status": "running",
        "version": "2.0.0",
        "available_datasets": SUPPORTED_DATASETS,
        "available_retrieval_modes": SUPPORTED_RETRIEVAL_MODES,
        "pipeline": ["query_refinement", "retrieval", "ranking"],
        "dataset": {
            "name": "quora",
            "source": "beir/quora/test",
        },
    }


@app.post("/search/full")
async def full_search(request: FullSearchRequest):
    validate_request(request)

    async with httpx.AsyncClient(timeout=120.0) as client:
        refinement_data = await post_json(
            client,
            REFINEMENT_SERVICE_URL,
            build_refinement_payload(request),
            "Refinement",
        )
        refined_query = refinement_data["refined_query"]
        refined_query, personalization_terms, personalization_profile = apply_personalization(
            request,
            refined_query,
        )

        retrieval_strategy = RETRIEVAL_STRATEGIES[request.retrieval_mode]
        retrieval_url, retrieval_payload = retrieval_strategy(request, refined_query)
        retrieval_data = await post_json(
            client,
            retrieval_url,
            retrieval_payload,
            "Retrieval",
        )

    return build_full_response(
        request,
        refined_query,
        retrieval_data,
        personalization_terms,
        personalization_profile,
    )

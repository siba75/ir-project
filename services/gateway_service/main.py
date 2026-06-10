from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(
    title="IR Gateway Service",
    description="Central gateway for the complete IR pipeline",
    version="1.3.0"
)

REFINEMENT_SERVICE_URL = "http://127.0.0.1:8005/refine"
HYBRID_PARALLEL_SEARCH_URL = "http://127.0.0.1:8003/search/hybrid"
HYBRID_SERIAL_SEARCH_URL = "http://127.0.0.1:8003/search/hybrid/serial"

SUPPORTED_DATASETS = ["cranfield", "scifact"]
SUPPORTED_RETRIEVAL_MODES = ["hybrid_parallel", "hybrid_serial"]


class FullSearchRequest(BaseModel):
    query: str
    top_k: int = 5

    dataset: str = "cranfield"
    retrieval_mode: str = "hybrid_parallel"

    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    initial_k: int = 50

    remove_stopwords: bool = True
    use_stemming: bool = False
    use_expansion: bool = True


@app.get("/")
def home():
    return {
        "service": "Gateway Service",
        "status": "running",
        "version": "1.3.0",
        "available_datasets": SUPPORTED_DATASETS,
        "available_retrieval_modes": SUPPORTED_RETRIEVAL_MODES,
        "pipeline": [
            "query_refinement",
            "multi_dataset_retrieval",
            "ranking"
        ]
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

        if request.retrieval_mode == "hybrid_parallel":
            retrieval_url = HYBRID_PARALLEL_SEARCH_URL
            retrieval_payload = {
                "query": refined_query,
                "top_k": request.top_k,
                "bm25_weight": request.bm25_weight,
                "semantic_weight": request.semantic_weight,
                "dataset": request.dataset
            }

        else:
            retrieval_url = HYBRID_SERIAL_SEARCH_URL
            retrieval_payload = {
                "query": refined_query,
                "top_k": request.top_k,
                "initial_k": request.initial_k,
                "dataset": request.dataset
            }

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
        "pipeline": {
            "refinement_enabled": True,
            "retrieval_enabled": True,
            "ranking_enabled": True,
            "multi_dataset_enabled": True
        },
        "configuration": {
            "dataset": request.dataset,
            "top_k": request.top_k,
            "bm25_weight": request.bm25_weight,
            "semantic_weight": request.semantic_weight,
            "initial_k": request.initial_k,
            "remove_stopwords": request.remove_stopwords,
            "use_stemming": request.use_stemming,
            "use_expansion": request.use_expansion
        },
        "returned_results": retrieval_data.get("returned_results", 0),
        "results": retrieval_data.get("results", [])
    }
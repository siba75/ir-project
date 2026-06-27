from fastapi import FastAPI
import httpx

from pipeline import (
    REFINEMENT_SERVICE_URL,
    apply_personalization,
    build_personalized_query,
    build_full_response,
    build_refinement_payload,
    post_json,
)
from retrieval_strategies import RETRIEVAL_STRATEGIES
from schemas import (
    SUPPORTED_DATASETS,
    SUPPORTED_RETRIEVAL_MODES,
    FullSearchRequest,
    validate_request,
)


app = FastAPI(
    title="IR Gateway Service",
    description="Central gateway for the complete IR pipeline",
    version="2.0.0",
)


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

import logging

import httpx
from fastapi import HTTPException

from personalization import build_personalized_query, empty_profile
from schemas import FullSearchRequest

logger = logging.getLogger(__name__)


REFINEMENT_SERVICE_URL = "http://127.0.0.1:8005/refine"


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
    except httpx.ConnectError as exc:
        logger.error("%s service unreachable at %s: %s", label, url, exc)
        raise HTTPException(
            status_code=502,
            detail=f"{label} service is unreachable at {url}",
        )
    except httpx.TimeoutException as exc:
        logger.error("%s service timed out at %s: %s", label, url, exc)
        raise HTTPException(
            status_code=504,
            detail=f"{label} service timed out",
        )

    if response.status_code >= 400:
        logger.error(
            "%s service returned HTTP %d: %s",
            label,
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(
            status_code=response.status_code,
            detail=f"{label} service returned HTTP {response.status_code}: {response.text[:200]}",
        )

    try:
        return response.json()
    except ValueError as exc:
        logger.error("%s service returned invalid JSON: %s", label, exc)
        raise HTTPException(
            status_code=502,
            detail=f"{label} service returned invalid JSON",
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

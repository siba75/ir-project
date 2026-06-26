from fastapi import HTTPException
from pydantic import BaseModel


SUPPORTED_DATASETS = ["quora"]
SUPPORTED_RETRIEVAL_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial",
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from retrieval_core import (
    dataset_bm25_search,
    dataset_tfidf_search,
    hybrid_search,
    hybrid_serial_search,
    normalize_vector_method,
    semantic_search,
)


app = FastAPI(
    title="IR Retrieval Service",
    description="Quora retrieval service for IR search and ranking",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


MAX_QUERY_LENGTH = 2000
MAX_TOP_K = 100


def _validate_query_length(v: str) -> str:
    if len(v) > MAX_QUERY_LENGTH:
        raise ValueError(f"query exceeds maximum length of {MAX_QUERY_LENGTH} characters")
    return v


def _validate_top_k(v: int) -> int:
    if v > MAX_TOP_K:
        raise ValueError(f"top_k must not exceed {MAX_TOP_K}")
    return v


class DatasetBM25SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "quora"
    k1: float = 1.5
    b: float = 0.75

    _check_query = field_validator("query")(_validate_query_length)
    _check_top_k = field_validator("top_k")(_validate_top_k)


class IndexedSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "quora"

    _check_query = field_validator("query")(_validate_query_length)
    _check_top_k = field_validator("top_k")(_validate_top_k)


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "quora"

    _check_query = field_validator("query")(_validate_query_length)
    _check_top_k = field_validator("top_k")(_validate_top_k)


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    dataset: str = "quora"
    k1: float = 1.5
    b: float = 0.75

    _check_query = field_validator("query")(_validate_query_length)
    _check_top_k = field_validator("top_k")(_validate_top_k)


class HybridSerialSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    initial_k: int = 50
    dataset: str = "quora"
    k1: float = 1.5
    b: float = 0.75

    _check_query = field_validator("query")(_validate_query_length)
    _check_top_k = field_validator("top_k")(_validate_top_k)


@app.get("/")
def home():
    return {
        "service": "Retrieval Service",
        "status": "running",
        "version": "2.0.0",
        "dataset": "quora",
        "source": "beir/quora/test",
        "storage": {
            "documents": "SQLite document store",
            "indexes": "Compressed resource files with runtime cache",
            "tfidf": "sklearn TfidfVectorizer",
            "bm25": "rank_bm25 BM25Okapi",
        },
        "available_models": [
            "tfidf_vsm",
            "bm25",
            "semantic_search",
            "hybrid_parallel_search",
            "hybrid_serial_search",
        ],
    }


@app.post("/search/dataset/tfidf")
def search_dataset_tfidf(request: IndexedSearchRequest):
    return dataset_tfidf_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset,
    )


@app.post("/search/dataset/bm25")
def search_dataset_bm25(request: DatasetBM25SearchRequest):
    return dataset_bm25_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset,
        k1=request.k1,
        b=request.b,
    )


@app.post("/search/semantic")
def search_semantic(request: SemanticSearchRequest):
    return semantic_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset,
    )


@app.post("/search/hybrid")
def search_hybrid(request: HybridSearchRequest):
    return hybrid_search(
        query=request.query,
        top_k=request.top_k,
        bm25_weight=request.bm25_weight,
        semantic_weight=request.semantic_weight,
        dataset=request.dataset,
        k1=request.k1,
        b=request.b,
    )


@app.post("/search/hybrid/serial")
def search_hybrid_serial(request: HybridSerialSearchRequest):
    return hybrid_serial_search(
        query=request.query,
        top_k=request.top_k,
        initial_k=request.initial_k,
        dataset=request.dataset,
        k1=request.k1,
        b=request.b,
    )

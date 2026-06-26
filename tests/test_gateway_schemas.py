import sys
from pathlib import Path

import pytest

GATEWAY_DIR = Path(__file__).resolve().parents[1] / "services" / "gateway_service"
sys.path.insert(0, str(GATEWAY_DIR))

from fastapi import HTTPException

from schemas import (
    SUPPORTED_DATASETS,
    SUPPORTED_RETRIEVAL_MODES,
    FullSearchRequest,
    validate_request,
)


def test_supported_datasets():
    assert "quora" in SUPPORTED_DATASETS


def test_supported_retrieval_modes():
    expected = {"tfidf", "bm25", "semantic", "hybrid_parallel", "hybrid_serial"}
    assert set(SUPPORTED_RETRIEVAL_MODES) == expected


def test_full_search_request_defaults():
    req = FullSearchRequest(query="test")
    assert req.query == "test"
    assert req.top_k == 5
    assert req.dataset == "quora"
    assert req.retrieval_mode == "hybrid_parallel"
    assert req.bm25_weight == 0.4
    assert req.semantic_weight == 0.6
    assert req.bm25_k1 == 1.5
    assert req.bm25_b == 0.75
    assert req.initial_k == 50
    assert req.remove_stopwords is True
    assert req.use_stemming is False
    assert req.use_lemmatization is False
    assert req.use_expansion is True
    assert req.use_personalization is False
    assert req.user_history == []


def test_validate_request_valid():
    req = FullSearchRequest(query="test", dataset="quora", retrieval_mode="bm25")
    validate_request(req)


def test_validate_request_normalizes_dataset():
    req = FullSearchRequest(query="test", dataset="  QUORA  ")
    validate_request(req)
    assert req.dataset == "quora"


def test_validate_request_normalizes_retrieval_mode():
    req = FullSearchRequest(query="test", retrieval_mode="  BM25  ")
    validate_request(req)
    assert req.retrieval_mode == "bm25"


def test_validate_request_unsupported_dataset():
    req = FullSearchRequest(query="test", dataset="unknown")
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400
    assert "dataset" in exc_info.value.detail


def test_validate_request_unsupported_retrieval_mode():
    req = FullSearchRequest(query="test", retrieval_mode="unknown")
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400
    assert "retrieval_mode" in exc_info.value.detail


def test_validate_request_top_k_zero():
    req = FullSearchRequest(query="test", top_k=0)
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400
    assert "top_k" in exc_info.value.detail


def test_validate_request_negative_top_k():
    req = FullSearchRequest(query="test", top_k=-5)
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400


def test_validate_request_adjusts_initial_k():
    req = FullSearchRequest(query="test", top_k=100, initial_k=10)
    validate_request(req)
    assert req.initial_k >= req.top_k


def test_validate_request_negative_weights():
    req = FullSearchRequest(query="test", bm25_weight=-1.0)
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400


def test_validate_request_negative_semantic_weight():
    req = FullSearchRequest(query="test", semantic_weight=-0.5)
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400


def test_validate_request_invalid_bm25_k1():
    req = FullSearchRequest(query="test", bm25_k1=0)
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400


def test_validate_request_invalid_bm25_b():
    req = FullSearchRequest(query="test", bm25_b=1.5)
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400


def test_validate_request_zero_weights_hybrid():
    req = FullSearchRequest(
        query="test",
        retrieval_mode="hybrid_parallel",
        bm25_weight=0.0,
        semantic_weight=0.0,
    )
    with pytest.raises(HTTPException) as exc_info:
        validate_request(req)
    assert exc_info.value.status_code == 400
    assert "weight" in exc_info.value.detail.lower()


def test_validate_request_zero_weights_non_hybrid():
    req = FullSearchRequest(
        query="test",
        retrieval_mode="bm25",
        bm25_weight=0.0,
        semantic_weight=0.0,
    )
    validate_request(req)

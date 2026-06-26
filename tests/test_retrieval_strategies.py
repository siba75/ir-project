import sys
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parents[1] / "services" / "gateway_service"
sys.path.insert(0, str(GATEWAY_DIR))

from retrieval_strategies import (
    RETRIEVAL_STRATEGIES,
    build_bm25_payload,
    build_hybrid_parallel_payload,
    build_hybrid_serial_payload,
    build_semantic_payload,
    build_tfidf_payload,
)
from schemas import FullSearchRequest


def _make_request(**kwargs):
    defaults = {
        "query": "test query",
        "top_k": 10,
        "dataset": "quora",
        "bm25_weight": 0.4,
        "semantic_weight": 0.6,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "initial_k": 50,
    }
    defaults.update(kwargs)
    return FullSearchRequest(**defaults)


def test_strategies_has_all_modes():
    expected = {"tfidf", "bm25", "semantic", "hybrid_parallel", "hybrid_serial"}
    assert set(RETRIEVAL_STRATEGIES) == expected


def test_build_tfidf_payload():
    req = _make_request(top_k=5, dataset="quora")
    url, payload = build_tfidf_payload(req, "refined query")

    assert url.endswith("/search/dataset/tfidf")
    assert payload == {
        "query": "refined query",
        "top_k": 5,
        "dataset": "quora",
    }


def test_build_bm25_payload():
    req = _make_request(top_k=7, bm25_k1=2.0, bm25_b=0.5)
    url, payload = build_bm25_payload(req, "refined")

    assert url.endswith("/search/dataset/bm25")
    assert payload == {
        "query": "refined",
        "top_k": 7,
        "dataset": "quora",
        "k1": 2.0,
        "b": 0.5,
    }


def test_build_semantic_payload():
    req = _make_request(top_k=3)
    url, payload = build_semantic_payload(req, "semantic query")

    assert url.endswith("/search/semantic")
    assert payload == {
        "query": "semantic query",
        "top_k": 3,
        "dataset": "quora",
    }


def test_build_hybrid_parallel_payload():
    req = _make_request(
        top_k=7,
        bm25_weight=0.3,
        semantic_weight=0.7,
        bm25_k1=2.0,
        bm25_b=0.5,
    )
    url, payload = build_hybrid_parallel_payload(req, "hybrid query")

    assert url.endswith("/search/hybrid")
    assert payload == {
        "query": "hybrid query",
        "top_k": 7,
        "bm25_weight": 0.3,
        "semantic_weight": 0.7,
        "k1": 2.0,
        "b": 0.5,
        "dataset": "quora",
    }


def test_build_hybrid_serial_payload():
    req = _make_request(top_k=5, initial_k=100, bm25_k1=1.2, bm25_b=0.8)
    url, payload = build_hybrid_serial_payload(req, "serial query")

    assert url.endswith("/search/hybrid/serial")
    assert payload == {
        "query": "serial query",
        "top_k": 5,
        "initial_k": 100,
        "k1": 1.2,
        "b": 0.8,
        "dataset": "quora",
    }


def test_strategies_are_callable():
    req = _make_request()
    for mode, strategy_fn in RETRIEVAL_STRATEGIES.items():
        url, payload = strategy_fn(req, "test")
        assert isinstance(url, str)
        assert isinstance(payload, dict)
        assert "query" in payload
        assert payload["query"] == "test"

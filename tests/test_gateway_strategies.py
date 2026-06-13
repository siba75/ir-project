import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
GATEWAY_DIR = BASE_DIR / "services" / "gateway_service"
sys.path.insert(0, str(GATEWAY_DIR))

from main import FullSearchRequest, RETRIEVAL_STRATEGIES  # noqa: E402


def test_gateway_has_strategy_for_each_supported_mode():
    expected_modes = {
        "tfidf",
        "bm25",
        "semantic",
        "hybrid_parallel",
        "hybrid_serial",
    }

    assert set(RETRIEVAL_STRATEGIES) == expected_modes


def test_hybrid_parallel_strategy_builds_expected_payload():
    request = FullSearchRequest(
        query="how to learn programming",
        retrieval_mode="hybrid_parallel",
        top_k=7,
        dataset="quora",
        bm25_weight=0.3,
        semantic_weight=0.7,
        bm25_k1=2.0,
        bm25_b=0.5,
    )

    url, payload = RETRIEVAL_STRATEGIES["hybrid_parallel"](
        request,
        "learn programming"
    )

    assert url.endswith("/search/hybrid")
    assert payload == {
        "query": "learn programming",
        "top_k": 7,
        "bm25_weight": 0.3,
        "semantic_weight": 0.7,
        "k1": 2.0,
        "b": 0.5,
        "dataset": "quora",
    }

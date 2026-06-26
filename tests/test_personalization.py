import sys
from pathlib import Path

import numpy as np

GATEWAY_DIR = Path(__file__).resolve().parents[1] / "services" / "gateway_service"
sys.path.insert(0, str(GATEWAY_DIR))

from personalization import (
    build_query_suggestions,
    clean_history_queries,
    empty_profile,
    find_similar_history,
    select_personalization_terms,
    vector_top_terms,
)


def test_clean_history_queries_removes_current():
    result = clean_history_queries(["hello", "world", "hello"], "hello")
    assert "hello" not in result
    assert "world" in result


def test_clean_history_queries_removes_empty():
    result = clean_history_queries(["", "  ", "valid"], "test")
    assert len(result) == 1
    assert result[0] == "valid"


def test_clean_history_queries_normalizes():
    result = clean_history_queries(["  HELLO  World  "], "test")
    assert result[0] == "hello world"


def test_clean_history_queries_limits_to_20():
    history = [f"query{i}" for i in range(30)]
    result = clean_history_queries(history, "other")
    assert len(result) <= 20


def test_clean_history_queries_empty_history():
    result = clean_history_queries([], "test")
    assert result == []


def test_clean_history_queries_case_insensitive_current():
    result = clean_history_queries(["Hello", "HELLO", "world"], "hello")
    assert "world" in result
    assert len(result) == 1


def test_empty_profile_returns_default():
    profile = empty_profile()
    assert profile["enabled"] is False
    assert profile["history_queries_used"] == 0
    assert profile["interest_terms"] == []
    assert profile["combined_terms"] == []
    assert profile["similar_history_queries"] == []
    assert profile["query_suggestions"] == []


def test_empty_profile_returns_copy():
    p1 = empty_profile()
    p2 = empty_profile()
    p1["enabled"] = True
    assert p2["enabled"] is False


def test_vector_top_terms_basic():
    vector = np.array([0.5, 0.8, 0.1, 0.0])
    feature_names = np.array(["alpha", "beta", "gamma", "delta"])
    existing = set()
    terms = vector_top_terms(vector, feature_names, existing, limit=2)
    assert len(terms) == 2
    assert terms[0]["term"] == "beta"
    assert terms[0]["score"] == 0.8
    assert terms[1]["term"] == "alpha"


def test_vector_top_terms_skips_existing():
    vector = np.array([0.9, 0.5])
    feature_names = np.array(["existing_term", "new_term"])
    existing = {"existing_term"}
    terms = vector_top_terms(vector, feature_names, existing, limit=5)
    assert len(terms) == 1
    assert terms[0]["term"] == "new_term"


def test_vector_top_terms_skips_short():
    vector = np.array([0.9, 0.5])
    feature_names = np.array(["ab", "long_term"])
    existing = set()
    terms = vector_top_terms(vector, feature_names, existing, limit=5)
    assert len(terms) == 1
    assert terms[0]["term"] == "long_term"


def test_vector_top_terms_skips_zero_scores():
    vector = np.array([0.0, 0.0])
    feature_names = np.array(["alpha", "beta"])
    existing = set()
    terms = vector_top_terms(vector, feature_names, existing, limit=5)
    assert terms == []


def test_vector_top_terms_respects_limit():
    vector = np.array([0.9, 0.8, 0.7, 0.6])
    feature_names = np.array(["aaa", "bbb", "ccc", "ddd"])
    terms = vector_top_terms(vector, feature_names, set(), limit=2)
    assert len(terms) == 2


def test_select_personalization_terms_deduplicates():
    combined = [{"term": "alpha", "score": 0.9}, {"term": "beta", "score": 0.8}]
    interest = [{"term": "beta", "score": 0.7}, {"term": "gamma", "score": 0.6}]
    result = select_personalization_terms(combined, interest, limit=6)
    assert result == ["alpha", "beta", "gamma"]


def test_select_personalization_terms_respects_limit():
    combined = [{"term": f"t{i}", "score": 0.5} for i in range(10)]
    result = select_personalization_terms(combined, [], limit=3)
    assert len(result) == 3


def test_select_personalization_terms_empty():
    result = select_personalization_terms([], [], limit=6)
    assert result == []


def test_find_similar_history_returns_top():
    queries = ["query a", "query b", "query c"]
    similarities = np.array([0.1, 0.9, 0.5])
    result = find_similar_history(queries, similarities, limit=2)
    assert len(result) == 2
    assert result[0]["query"] == "query b"
    assert result[0]["similarity"] == 0.9


def test_find_similar_history_excludes_zero():
    queries = ["a", "b"]
    similarities = np.array([0.5, 0.0])
    result = find_similar_history(queries, similarities, limit=5)
    assert len(result) == 1
    assert result[0]["query"] == "a"


def test_find_similar_history_all_zero_fallback():
    queries = ["last query"]
    similarities = np.array([0.0])
    result = find_similar_history(queries, similarities, limit=5)
    assert len(result) == 1
    assert result[0]["query"] == "last query"
    assert result[0]["similarity"] == 0.0


def test_find_similar_history_empty():
    result = find_similar_history([], np.array([]), limit=5)
    assert result == []


def test_build_query_suggestions_basic():
    suggestions = build_query_suggestions(
        "test query",
        ["term1", "term2", "term3", "term4", "term5"],
        [{"query": "history1", "similarity": 0.5}],
    )
    assert suggestions[0] == "test query term1"
    assert len(suggestions) <= 6


def test_build_query_suggestions_includes_history():
    suggestions = build_query_suggestions(
        "test",
        [],
        [{"query": "past query", "similarity": 0.5}],
    )
    assert "past query" in suggestions


def test_build_query_suggestions_empty():
    suggestions = build_query_suggestions("test", [], [])
    assert suggestions == []


def test_build_query_suggestions_max_six():
    suggestions = build_query_suggestions(
        "q",
        ["a", "b", "c", "d"],
        [
            {"query": "h1", "similarity": 0.5},
            {"query": "h2", "similarity": 0.4},
            {"query": "h3", "similarity": 0.3},
        ],
    )
    assert len(suggestions) <= 6


def test_build_query_suggestions_no_duplicate_history():
    suggestions = build_query_suggestions(
        "test",
        ["term1"],
        [{"query": "test term1", "similarity": 0.9}],
    )
    assert suggestions.count("test term1") <= 1

import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "retrieval_service"
sys.path.insert(0, str(SERVICE_DIR))

from fastapi import HTTPException

from retrieval_common import (
    get_doc_text,
    normalize_scores,
    normalize_vector_method,
    ranked_doc_results,
    validate_bm25_parameters,
    validate_query,
)


def test_normalize_scores_basic():
    scores = {"d1": 2.0, "d2": 4.0, "d3": 1.0}
    result = normalize_scores(scores)
    assert result["d2"] == 1.0
    assert result["d1"] == 0.5
    assert result["d3"] == 0.25


def test_normalize_scores_empty():
    assert normalize_scores({}) == {}


def test_normalize_scores_all_zeros():
    scores = {"d1": 0.0, "d2": 0.0}
    result = normalize_scores(scores)
    assert result == scores


def test_normalize_scores_negative_max():
    scores = {"d1": -2.0, "d2": -1.0}
    result = normalize_scores(scores)
    assert result == scores


def test_normalize_scores_single_entry():
    result = normalize_scores({"d1": 5.0})
    assert result["d1"] == 1.0


def test_normalize_vector_method_aliases():
    assert normalize_vector_method("lsa") == "lsa_tfidf_svd"
    assert normalize_vector_method("tfidf_svd") == "lsa_tfidf_svd"
    assert normalize_vector_method("lsa_tfidf_svd") == "lsa_tfidf_svd"
    assert normalize_vector_method("transformer") == "sentence_transformer"
    assert normalize_vector_method("sentence_transformer") == "sentence_transformer"
    assert normalize_vector_method("sentence-transformer") == "sentence_transformer"
    assert normalize_vector_method("sentence_transformers") == "sentence_transformer"
    assert normalize_vector_method("sentence-transformers") == "sentence_transformer"


def test_normalize_vector_method_none():
    assert normalize_vector_method(None) is None


def test_normalize_vector_method_empty():
    assert normalize_vector_method("") is None


def test_normalize_vector_method_unknown():
    assert normalize_vector_method("unknown_method") == "unknown_method"


def test_normalize_vector_method_case_insensitive():
    assert normalize_vector_method("LSA") == "lsa_tfidf_svd"
    assert normalize_vector_method("Transformer") == "sentence_transformer"


def test_normalize_vector_method_strips_whitespace():
    assert normalize_vector_method("  lsa  ") == "lsa_tfidf_svd"


def test_validate_query_valid():
    validate_query("hello world")


def test_validate_query_empty():
    with pytest.raises(HTTPException) as exc_info:
        validate_query("")
    assert exc_info.value.status_code == 400


def test_validate_query_whitespace_only():
    with pytest.raises(HTTPException) as exc_info:
        validate_query("   ")
    assert exc_info.value.status_code == 400


def test_validate_bm25_parameters_valid():
    validate_bm25_parameters(k1=1.5, b=0.75)


def test_validate_bm25_parameters_k1_zero():
    with pytest.raises(HTTPException) as exc_info:
        validate_bm25_parameters(k1=0, b=0.75)
    assert exc_info.value.status_code == 400


def test_validate_bm25_parameters_k1_negative():
    with pytest.raises(HTTPException) as exc_info:
        validate_bm25_parameters(k1=-1.0, b=0.75)
    assert exc_info.value.status_code == 400


def test_validate_bm25_parameters_b_negative():
    with pytest.raises(HTTPException) as exc_info:
        validate_bm25_parameters(k1=1.5, b=-0.1)
    assert exc_info.value.status_code == 400


def test_validate_bm25_parameters_b_above_one():
    with pytest.raises(HTTPException) as exc_info:
        validate_bm25_parameters(k1=1.5, b=1.5)
    assert exc_info.value.status_code == 400


def test_validate_bm25_parameters_boundary_values():
    validate_bm25_parameters(k1=0.01, b=0.0)
    validate_bm25_parameters(k1=0.01, b=1.0)


class FakeDocumentStore:
    def __init__(self, docs):
        self._docs = docs

    def get(self, doc_id):
        return self._docs.get(doc_id, "")


def test_get_doc_text_from_document_store():
    store = FakeDocumentStore({"d1": "doc one text"})
    resources = {"document_store": store}
    assert get_doc_text("d1", resources) == "doc one text"


def test_get_doc_text_missing_from_store():
    store = FakeDocumentStore({})
    resources = {
        "document_store": store,
        "metadata": {
            "doc_ids": ["d1", "d2"],
            "documents": ["doc one", "doc two"],
        },
    }
    assert get_doc_text("d2", resources) == "doc two"


def test_get_doc_text_no_store():
    resources = {
        "metadata": {
            "doc_ids": ["d1"],
            "documents": ["fallback text"],
        },
    }
    assert get_doc_text("d1", resources) == "fallback text"


def test_get_doc_text_not_found_anywhere():
    resources = {"metadata": {"doc_ids": [], "documents": []}}
    assert get_doc_text("missing", resources) == ""


def test_ranked_doc_results_format():
    store = FakeDocumentStore({"d1": "text one", "d2": "text two"})
    resources = {"document_store": store}
    ranked = [("d1", 0.95), ("d2", 0.85)]
    results = ranked_doc_results(ranked, resources)

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["doc_id"] == "d1"
    assert results[0]["score"] == 0.95
    assert results[0]["text"] == "text one"
    assert results[1]["rank"] == 2


def test_ranked_doc_results_empty():
    resources = {"document_store": FakeDocumentStore({})}
    assert ranked_doc_results([], resources) == []

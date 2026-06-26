import sys
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parents[1] / "services" / "gateway_service"
sys.path.insert(0, str(GATEWAY_DIR))

from pipeline import build_full_response, build_refinement_payload
from schemas import FullSearchRequest


def test_build_refinement_payload():
    req = FullSearchRequest(
        query="test query",
        remove_stopwords=True,
        use_stemming=False,
        use_lemmatization=True,
        use_expansion=False,
    )
    payload = build_refinement_payload(req)

    assert payload == {
        "query": "test query",
        "remove_stopwords": True,
        "use_stemming": False,
        "use_lemmatization": True,
        "use_expansion": False,
    }


def test_build_refinement_payload_defaults():
    req = FullSearchRequest(query="hello")
    payload = build_refinement_payload(req)

    assert payload["query"] == "hello"
    assert payload["remove_stopwords"] is True
    assert payload["use_stemming"] is False
    assert payload["use_lemmatization"] is False
    assert payload["use_expansion"] is True


def test_build_full_response_structure():
    req = FullSearchRequest(
        query="original query",
        top_k=5,
        dataset="quora",
        retrieval_mode="bm25",
        bm25_weight=0.4,
        semantic_weight=0.6,
        bm25_k1=1.5,
        bm25_b=0.75,
        initial_k=50,
        use_personalization=False,
    )
    retrieval_data = {
        "model": "BM25",
        "vector_method": None,
        "storage": {"compressed": True},
        "returned_results": 2,
        "results": [
            {"rank": 1, "doc_id": "d1", "score": 0.9, "text": "doc one"},
            {"rank": 2, "doc_id": "d2", "score": 0.8, "text": "doc two"},
        ],
    }

    response = build_full_response(
        req,
        refined_query="refined query",
        retrieval_data=retrieval_data,
        personalization_terms=["term1"],
        personalization_profile={"enabled": False, "query_suggestions": []},
    )

    assert response["original_query"] == "original query"
    assert response["refined_query"] == "refined query"
    assert response["dataset"] == "quora"
    assert response["retrieval_mode"] == "bm25"
    assert response["retrieval_model"] == "BM25"
    assert response["returned_results"] == 2
    assert len(response["results"]) == 2

    assert response["pipeline"]["refinement_enabled"] is True
    assert response["pipeline"]["retrieval_enabled"] is True
    assert response["pipeline"]["ranking_enabled"] is True

    features = response["additional_features"]
    assert features["vector_store_faiss"] is True
    assert features["personalization_enabled"] is False
    assert features["personalization_terms"] == ["term1"]

    config = response["configuration"]
    assert config["dataset"] == "quora"
    assert config["top_k"] == 5
    assert config["bm25_weight"] == 0.4
    assert config["semantic_weight"] == 0.6
    assert config["bm25_k1"] == 1.5
    assert config["bm25_b"] == 0.75
    assert config["initial_k"] == 50


def test_build_full_response_empty_retrieval():
    req = FullSearchRequest(query="test")
    response = build_full_response(
        req,
        refined_query="test",
        retrieval_data={},
        personalization_terms=[],
        personalization_profile={"enabled": False, "query_suggestions": []},
    )
    assert response["returned_results"] == 0
    assert response["results"] == []
    assert response["retrieval_model"] is None

import sys
import importlib.util
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INDEXING_DIR = BASE_DIR / "services" / "indexing_service"
REFINEMENT_DIR = BASE_DIR / "services" / "refinement_service"
RETRIEVAL_DIR = BASE_DIR / "services" / "retrieval_service"


def import_service_module(service_dir, module_name, alias):
    module_path = service_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(alias, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(service_dir))

    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    return module


indexing_main = import_service_module(INDEXING_DIR, "main", "indexing_main")
refinement_main = import_service_module(REFINEMENT_DIR, "main", "refinement_main")
retrieval_main = import_service_module(RETRIEVAL_DIR, "main", "retrieval_main")


def test_indexing_preprocessing_matches_query_terms():
    assert indexing_main.preprocess_for_indexing(
        "Programming, PROGRAMMING!!! https://example.com"
    ) == ["programming", "programming"]


def test_query_refinement_expands_quora_terms():
    result = refinement_main.refine_query(
        "Best programming job",
        remove_stopwords=True,
        use_stemming=False,
        use_lemmatization=False,
        use_expansion=True,
    )

    assert "programming" in result["tokens"]
    assert "coding" in result["expanded_terms"]
    assert "career" in result["expanded_terms"]


def test_query_refinement_supports_lemmatization():
    result = refinement_main.refine_query(
        "phones",
        remove_stopwords=False,
        use_stemming=False,
        use_lemmatization=True,
        use_expansion=False,
    )

    assert result["refined_query"] in ["phone", "phones"]


def test_ranking_orders_documents_by_score():
    response = retrieval_main.dataset_bm25_search(
        query="programming job",
        top_k=5,
        dataset="quora",
        k1=1.5,
        b=0.75,
    )

    results = response["results"]
    scores = [item["score"] for item in results]

    assert response["returned_results"] > 0
    assert results[0]["rank"] == 1
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    test_indexing_preprocessing_matches_query_terms()
    test_query_refinement_expands_quora_terms()
    test_query_refinement_supports_lemmatization()
    test_ranking_orders_documents_by_score()

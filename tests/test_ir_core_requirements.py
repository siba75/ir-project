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
    documents = [
        retrieval_main.Document(doc_id="d1", text="programming job career"),
        retrieval_main.Document(doc_id="d2", text="cooking recipe"),
    ]
    inverted_index, documents_store = retrieval_main.build_inverted_index(documents)
    results = retrieval_main.score_documents(
        ["programming", "job"],
        inverted_index,
        documents_store,
    )

    assert results[0]["doc_id"] == "d1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > 0


if __name__ == "__main__":
    test_indexing_preprocessing_matches_query_terms()
    test_query_refinement_expands_quora_terms()
    test_query_refinement_supports_lemmatization()
    test_ranking_orders_documents_by_score()

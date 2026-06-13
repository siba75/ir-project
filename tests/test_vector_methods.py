import sys
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "retrieval_service"
sys.path.insert(0, str(SERVICE_DIR))

from main import normalize_vector_method


def test_vector_method_aliases():
    assert normalize_vector_method("lsa") == "lsa_tfidf_svd"
    assert normalize_vector_method("lsa_tfidf_svd") == "lsa_tfidf_svd"
    assert normalize_vector_method("transformer") == "sentence_transformer"
    assert normalize_vector_method("sentence_transformer") == "sentence_transformer"
    assert normalize_vector_method("sentence-transformers") == "sentence_transformer"

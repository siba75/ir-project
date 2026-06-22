import gzip
import json
import pickle
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "indexes" / "quora"


def test_submission_resources_exist():
    expected_files = [
        "documents.sqlite",
        "bm25.pkl.gz",
        "tfidf.pkl.gz",
        "faiss.index.gz",
        "metadata.pkl.gz",
        "inverted_index.json.gz",
        "resource_manifest.json",
    ]

    for file_name in expected_files:
        assert (INDEX_DIR / file_name).exists()


def test_documents_are_available_from_sqlite():
    with sqlite3.connect(INDEX_DIR / "documents.sqlite") as connection:
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        sample = connection.execute(
            "SELECT doc_id, content FROM documents LIMIT 1"
        ).fetchone()

    assert count == 522931
    assert sample[0]
    assert sample[1]


def test_tfidf_uses_sklearn_vectorizer():
    with gzip.open(INDEX_DIR / "tfidf.pkl.gz", "rb") as file:
        payload = pickle.load(file)

    assert payload["model"] == "sklearn.feature_extraction.text.TfidfVectorizer"
    assert payload["total_documents"] == 522931


def test_resource_manifest_documents_storage_choices():
    with open(INDEX_DIR / "resource_manifest.json", "r", encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["document_store"] == "indexes\\quora\\documents.sqlite" or manifest["document_store"] == "indexes/quora/documents.sqlite"
    assert manifest["compressed_resources"]["bm25"] == "indexes/quora/bm25.pkl.gz"

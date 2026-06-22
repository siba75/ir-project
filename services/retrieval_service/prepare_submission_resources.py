import gzip
import json
import pickle
import shutil
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET = "quora"
INDEX_DIR = BASE_DIR / "indexes" / DATASET
DB_PATH = INDEX_DIR / "documents.sqlite"
TFIDF_PATH = INDEX_DIR / "tfidf.pkl.gz"


def gzip_copy(source_path: Path, target_path: Path):
    if target_path.exists() and target_path.stat().st_mtime >= source_path.stat().st_mtime:
        print(f"Up to date: {target_path}")
        return

    print(f"Compressing {source_path.name} -> {target_path.name}")
    with open(source_path, "rb") as source, gzip.open(target_path, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def load_bm25_data():
    compressed_path = INDEX_DIR / "bm25.pkl.gz"

    if compressed_path.exists():
        with gzip.open(compressed_path, "rb") as file:
            return pickle.load(file)

    with open(INDEX_DIR / "bm25.pkl", "rb") as file:
        return pickle.load(file)


def build_documents_database(doc_ids, documents):
    print("Building SQLite document store")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM documents")
        connection.executemany(
            "INSERT INTO documents (doc_id, content) VALUES (?, ?)",
            ((str(doc_id), text) for doc_id, text in zip(doc_ids, documents)),
        )
        connection.commit()

    print(f"Saved {len(doc_ids)} documents to {DB_PATH}")


def build_tfidf_resource(documents, doc_ids):
    if TFIDF_PATH.exists():
        print(f"TF-IDF resource already exists: {TFIDF_PATH}")
        return

    print("Building sklearn TF-IDF resource")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        lowercase=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(documents)

    payload = {
        "dataset": DATASET,
        "source_dataset": "beir/quora/test",
        "model": "sklearn.feature_extraction.text.TfidfVectorizer",
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_ids": [str(doc_id) for doc_id in doc_ids],
        "total_documents": len(doc_ids),
    }

    with gzip.open(TFIDF_PATH, "wb", compresslevel=6) as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved TF-IDF resource to {TFIDF_PATH}")


def write_manifest():
    manifest = {
        "dataset": DATASET,
        "source_dataset": "beir/quora/test",
        "document_store": str(DB_PATH.relative_to(BASE_DIR)),
        "compressed_resources": {
            "bm25": "indexes/quora/bm25.pkl.gz",
            "tfidf": "indexes/quora/tfidf.pkl.gz",
            "faiss": "indexes/quora/faiss.index.gz",
            "metadata": "indexes/quora/metadata.pkl.gz",
            "inverted_index": "indexes/quora/inverted_index.json.gz",
        },
        "notes": [
            "Documents are stored in SQLite and fetched by doc_id for display.",
            "BM25 uses rank_bm25.BM25Okapi.",
            "TF-IDF uses sklearn TfidfVectorizer.",
            "FAISS is loaded from a compressed index through a runtime cache file.",
        ],
    }

    with open(INDEX_DIR / "resource_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["bm25.pkl", "metadata.pkl", "faiss.index", "inverted_index.json"]:
        source_path = INDEX_DIR / name
        if source_path.exists():
            gzip_copy(source_path, INDEX_DIR / f"{name}.gz")
        else:
            print(f"Missing optional source: {source_path}")

    bm25_data = load_bm25_data()
    doc_ids = bm25_data["doc_ids"]
    documents = bm25_data["documents"]

    build_documents_database(doc_ids, documents)
    build_tfidf_resource(documents, doc_ids)
    write_manifest()

    print("Submission resources are ready.")


if __name__ == "__main__":
    main()

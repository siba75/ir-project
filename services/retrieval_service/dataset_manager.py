import faiss
import gzip
import pickle
import shutil
import sqlite3
from pathlib import Path

from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parents[2]
INDEXES_DIR = BASE_DIR / "indexes"
RUNTIME_CACHE_DIR = BASE_DIR / "reports" / "runtime_cache"

SUPPORTED_DATASETS = {
    "quora": {
        "type": "generic",
        "bm25_path": INDEXES_DIR / "quora" / "bm25.pkl.gz",
        "bm25_fallback_path": INDEXES_DIR / "quora" / "bm25.pkl",
        "tfidf_path": INDEXES_DIR / "quora" / "tfidf.pkl.gz",
        "faiss_index_path": INDEXES_DIR / "quora" / "faiss.index",
        "faiss_compressed_path": INDEXES_DIR / "quora" / "faiss.index.gz",
        "vector_metadata_path": INDEXES_DIR / "quora" / "metadata.pkl.gz",
        "vector_metadata_fallback_path": INDEXES_DIR / "quora" / "metadata.pkl",
        "document_db_path": INDEXES_DIR / "quora" / "documents.sqlite",
    }
}

_resource_cache = {}


def validate_dataset(dataset_name: str):
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset '{dataset_name}'. Supported datasets: {list(SUPPORTED_DATASETS.keys())}"
        )

    return dataset_name


def preferred_existing_path(config, resource_name):
    path = config[resource_name]

    if path.exists():
        return path

    fallback = config.get(resource_name.replace("_path", "_fallback_path"))

    if fallback and fallback.exists():
        return fallback

    raise HTTPException(
        status_code=404,
        detail=f"{resource_name} not found. Run prepare_submission_resources.py."
    )


def compressed_path_exists(config, resource_name):
    compressed_path = config.get(resource_name)
    return bool(compressed_path and compressed_path.exists())


def load_pickle(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            return pickle.load(file)

    with open(path, "rb") as file:
        return pickle.load(file)


def materialize_gzip(path: Path, dataset_name: str):
    if path.suffix != ".gz":
        return path

    RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RUNTIME_CACHE_DIR / f"{dataset_name}_{path.name.removesuffix('.gz')}"

    if target_path.exists() and target_path.stat().st_mtime >= path.stat().st_mtime:
        return target_path

    with gzip.open(path, "rb") as source, open(target_path, "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)

    return target_path


class DocumentStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)

    def get(self, doc_id: str):
        row = self.connection.execute(
            "SELECT content FROM documents WHERE doc_id = ?",
            (str(doc_id),),
        ).fetchone()

        return row[0] if row else ""


def load_generic_resources(dataset_name: str, include_vector: bool = False):
    cache_key = (dataset_name, include_vector)

    if cache_key in _resource_cache:
        return _resource_cache[cache_key]

    config = SUPPORTED_DATASETS[dataset_name]
    document_db_path = config["document_db_path"]

    if not document_db_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document database not found at {document_db_path}. Run prepare_submission_resources.py."
        )

    bm25_path = preferred_existing_path(config, "bm25_path")
    tfidf_path = config["tfidf_path"]

    if not tfidf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"TF-IDF resource not found at {tfidf_path}. Run prepare_submission_resources.py."
        )

    bm25_data = load_pickle(bm25_path)
    tfidf_data = load_pickle(tfidf_path)

    doc_ids = bm25_data["doc_ids"]
    document_store = DocumentStore(document_db_path)

    resources = {
        "dataset": dataset_name,
        "type": "generic",
        "document_store": document_store,
        "bm25": bm25_data["bm25"],
        "doc_ids": doc_ids,
        "tfidf": tfidf_data,
        "resource_paths": {
            "document_db": str(document_db_path),
            "bm25": str(bm25_path),
            "tfidf": str(tfidf_path),
        },
        "storage": {
            "documents_in_sqlite": True,
            "compressed_indexes": bm25_path.suffix == ".gz" and tfidf_path.suffix == ".gz",
            "resource_cache": True,
        },
        "total_documents": len(doc_ids),
    }

    if include_vector:
        add_vector_resources(resources, config, dataset_name)

    _resource_cache[cache_key] = resources
    return resources


def add_vector_resources(resources: dict, config: dict, dataset_name: str):
    faiss_path = preferred_existing_path(config, "faiss_index_path")
    faiss_compressed_path = config["faiss_compressed_path"]
    metadata_path = preferred_existing_path(config, "vector_metadata_path")

    if not faiss_compressed_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Compressed FAISS resource not found at {faiss_compressed_path}. Run prepare_submission_resources.py."
        )

    resources["faiss_index"] = faiss.read_index(str(materialize_gzip(faiss_path, dataset_name)))
    resources["metadata"] = load_pickle(metadata_path)
    resources["resource_paths"].update({
        "faiss": str(faiss_path),
        "faiss_compressed": str(faiss_compressed_path),
        "metadata": str(metadata_path),
    })
    resources["storage"]["compressed_indexes"] = (
        resources["storage"]["compressed_indexes"]
        and faiss_compressed_path.suffix == ".gz"
        and faiss_compressed_path.exists()
        and metadata_path.suffix == ".gz"
    )


def load_dataset_resources(dataset_name: str, include_vector: bool = False):
    dataset_name = validate_dataset(dataset_name)
    return load_generic_resources(dataset_name, include_vector=include_vector)

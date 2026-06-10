from fastapi import HTTPException
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

BASE_DIR = Path(__file__).resolve().parents[2]
INDEXES_DIR = BASE_DIR / "indexes"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_DATASETS = {
    "cranfield": {
        "type": "cranfield",
        "inverted_index_path": INDEXES_DIR / "cranfield_inverted_index.json",
        "faiss_index_path": INDEXES_DIR / "cranfield_faiss.index",
        "vector_metadata_path": INDEXES_DIR / "cranfield_vector_metadata.pkl",
    },
    "scifact": {
        "type": "scifact",
        "bm25_path": INDEXES_DIR / "scifact" / "scifact_bm25.pkl",
        "faiss_index_path": INDEXES_DIR / "scifact" / "scifact_faiss.index",
        "vector_metadata_path": INDEXES_DIR / "scifact" / "scifact_metadata.pkl",
    }
}

_embedding_model = None
_resource_cache = {}


def validate_dataset(dataset_name: str):
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset '{dataset_name}'. Supported datasets: {list(SUPPORTED_DATASETS.keys())}"
        )

    return dataset_name


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def load_cranfield_resources():
    dataset_name = "cranfield"

    if dataset_name in _resource_cache:
        return _resource_cache[dataset_name]

    config = SUPPORTED_DATASETS[dataset_name]

    if not config["inverted_index_path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"Cranfield inverted index not found at {config['inverted_index_path']}"
        )

    if not config["faiss_index_path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"Cranfield FAISS index not found at {config['faiss_index_path']}"
        )

    if not config["vector_metadata_path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"Cranfield vector metadata not found at {config['vector_metadata_path']}"
        )

    with open(config["inverted_index_path"], "r", encoding="utf-8") as file:
        index_data = json.load(file)

    faiss_index = faiss.read_index(str(config["faiss_index_path"]))

    with open(config["vector_metadata_path"], "rb") as file:
        metadata = pickle.load(file)

    resources = {
        "dataset": "cranfield",
        "type": "cranfield",
        "index_data": index_data,
        "documents_store": index_data["documents_store"],
        "inverted_index": index_data["inverted_index"],
        "faiss_index": faiss_index,
        "metadata": metadata,
        "total_documents": index_data["total_documents"]
    }

    _resource_cache[dataset_name] = resources
    return resources


def load_scifact_resources():
    dataset_name = "scifact"

    if dataset_name in _resource_cache:
        return _resource_cache[dataset_name]

    config = SUPPORTED_DATASETS[dataset_name]

    if not config["bm25_path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"SciFact BM25 index not found at {config['bm25_path']}"
        )

    if not config["faiss_index_path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"SciFact FAISS index not found at {config['faiss_index_path']}"
        )

    if not config["vector_metadata_path"].exists():
        raise HTTPException(
            status_code=404,
            detail=f"SciFact vector metadata not found at {config['vector_metadata_path']}"
        )

    with open(config["bm25_path"], "rb") as file:
        bm25_data = pickle.load(file)

    faiss_index = faiss.read_index(str(config["faiss_index_path"]))

    with open(config["vector_metadata_path"], "rb") as file:
        metadata = pickle.load(file)

    documents = bm25_data["documents"]
    doc_ids = bm25_data["doc_ids"]

    documents_store = {
        str(doc_id): text
        for doc_id, text in zip(doc_ids, documents)
    }

    resources = {
        "dataset": "scifact",
        "type": "scifact",
        "bm25": bm25_data["bm25"],
        "documents": documents,
        "doc_ids": doc_ids,
        "documents_store": documents_store,
        "faiss_index": faiss_index,
        "metadata": metadata,
        "total_documents": len(documents)
    }

    _resource_cache[dataset_name] = resources
    return resources


def load_dataset_resources(dataset_name: str):
    dataset_name = validate_dataset(dataset_name)

    if dataset_name == "cranfield":
        return load_cranfield_resources()

    if dataset_name == "scifact":
        return load_scifact_resources()

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported dataset '{dataset_name}'"
    )
from fastapi import HTTPException
from pathlib import Path
import faiss
import pickle

BASE_DIR = Path(__file__).resolve().parents[2]
INDEXES_DIR = BASE_DIR / "indexes"

SUPPORTED_DATASETS = {
    "quora": {
        "type": "generic",
        "bm25_path": INDEXES_DIR / "quora" / "bm25.pkl",
        "faiss_index_path": INDEXES_DIR / "quora" / "faiss.index",
        "vector_metadata_path": INDEXES_DIR / "quora" / "metadata.pkl",
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


def load_generic_resources(dataset_name: str):
    if dataset_name in _resource_cache:
        return _resource_cache[dataset_name]

    config = SUPPORTED_DATASETS[dataset_name]

    for resource_name in ["bm25_path", "faiss_index_path", "vector_metadata_path"]:
        if not config[resource_name].exists():
            raise HTTPException(
                status_code=404,
                detail=f"{dataset_name} resource not found at {config[resource_name]}"
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
        "dataset": dataset_name,
        "type": "generic",
        "documents_store": documents_store,
        "bm25": bm25_data["bm25"],
        "documents": documents,
        "doc_ids": doc_ids,
        "faiss_index": faiss_index,
        "metadata": metadata,
        "total_documents": len(documents),
    }

    _resource_cache[dataset_name] = resources
    return resources


def load_dataset_resources(dataset_name: str):
    dataset_name = validate_dataset(dataset_name)
    return load_generic_resources(dataset_name)
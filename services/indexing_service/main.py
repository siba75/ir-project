import sys
from functools import lru_cache
from pathlib import Path
import gzip
import json

from fastapi import FastAPI, HTTPException

_SHARED_DIR = str(Path(__file__).resolve().parent.parent / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.append(_SHARED_DIR)

from text_cleaning import clean_text  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[2]
QUORA_INDEX_PATH = BASE_DIR / "indexes" / "quora" / "inverted_index.json.gz"
QUORA_INDEX_FALLBACK_PATH = BASE_DIR / "indexes" / "quora" / "inverted_index.json"

app = FastAPI(
    title="IR Indexing Service",
    description="Service for inspecting the prebuilt Quora inverted index",
    version="2.0.0"
)


def preprocess_for_indexing(text: str) -> list[str]:
    return clean_text(text).split()


@lru_cache(maxsize=1)
def load_quora_index():
    index_path = QUORA_INDEX_PATH if QUORA_INDEX_PATH.exists() else QUORA_INDEX_FALLBACK_PATH

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Quora inverted index not found at {QUORA_INDEX_PATH}"
        )

    opener = gzip.open if index_path.suffix == ".gz" else open

    with opener(index_path, "rt", encoding="utf-8") as file:
        return json.load(file)


@app.get("/")
def home():
    return {
        "service": "Indexing Service",
        "status": "running",
        "dataset": "quora",
        "source": "beir/quora/test",
        "index_type": "compressed_prebuilt_inverted_index",
        "cache": "lru_cache(maxsize=1)"
    }


@app.get("/index/stats")
def index_stats():
    index_data = load_quora_index()

    return {
        "dataset": index_data.get("dataset", "quora"),
        "source_dataset": index_data.get("source_dataset", "beir/quora/test"),
        "total_documents": index_data.get("total_documents", 0),
        "unique_terms": index_data.get("unique_terms", 0),
        "index_path": str(QUORA_INDEX_PATH if QUORA_INDEX_PATH.exists() else QUORA_INDEX_FALLBACK_PATH),
        "compressed": QUORA_INDEX_PATH.exists(),
    }


@app.get("/index/term/{term}")
def get_term_postings(term: str):
    index_data = load_quora_index()
    normalized_terms = preprocess_for_indexing(term)

    if not normalized_terms:
        raise HTTPException(
            status_code=400,
            detail="Term has no valid searchable tokens"
        )

    normalized_term = normalized_terms[0]
    inverted_index = index_data.get("inverted_index", {})
    postings = inverted_index.get(normalized_term, {})

    return {
        "dataset": "quora",
        "term": normalized_term,
        "document_frequency": len(postings),
        "postings": postings
    }

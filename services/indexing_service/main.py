from functools import lru_cache
from pathlib import Path
import json
import re

from fastapi import FastAPI, HTTPException


BASE_DIR = Path(__file__).resolve().parents[2]
QUORA_INDEX_PATH = BASE_DIR / "indexes" / "quora" / "inverted_index.json"

app = FastAPI(
    title="IR Indexing Service",
    description="Service for inspecting the prebuilt Quora inverted index",
    version="2.0.0"
)


def preprocess_for_indexing(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


@lru_cache(maxsize=1)
def load_quora_index():
    if not QUORA_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Quora inverted index not found at {QUORA_INDEX_PATH}"
        )

    with open(QUORA_INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/")
def home():
    return {
        "service": "Indexing Service",
        "status": "running",
        "dataset": "quora",
        "source": "beir/quora/test",
        "index_type": "prebuilt_inverted_index"
    }


@app.get("/index/stats")
def index_stats():
    index_data = load_quora_index()

    return {
        "dataset": index_data.get("dataset", "quora"),
        "source_dataset": index_data.get("source_dataset", "beir/quora/test"),
        "total_documents": index_data.get("total_documents", 0),
        "unique_terms": index_data.get("unique_terms", 0),
        "index_path": str(QUORA_INDEX_PATH),
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

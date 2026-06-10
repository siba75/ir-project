from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import defaultdict, Counter
import re

app = FastAPI(
    title="IR Indexing Service",
    description="Service for building and managing inverted indexes",
    version="1.0.0"
)

inverted_index = defaultdict(dict)
documents_store = {}
document_lengths = {}
total_documents = 0


class Document(BaseModel):
    doc_id: str
    text: str


class IndexRequest(BaseModel):
    documents: list[Document]


def preprocess_for_indexing(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


@app.get("/")
def home():
    return {
        "service": "Indexing Service",
        "status": "running"
    }


@app.post("/index/build")
def build_index(request: IndexRequest):
    global inverted_index, documents_store, document_lengths, total_documents

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    inverted_index = defaultdict(dict)
    documents_store = {}
    document_lengths = {}

    for document in request.documents:
        tokens = preprocess_for_indexing(document.text)

        documents_store[document.doc_id] = document.text
        document_lengths[document.doc_id] = len(tokens)

        term_frequencies = Counter(tokens)

        for term, frequency in term_frequencies.items():
            inverted_index[term][document.doc_id] = frequency

    total_documents = len(request.documents)

    return {
        "message": "Index built successfully",
        "total_documents": total_documents,
        "unique_terms": len(inverted_index),
        "document_lengths": document_lengths
    }


@app.get("/index/stats")
def index_stats():
    return {
        "total_documents": total_documents,
        "unique_terms": len(inverted_index),
        "indexed_documents": list(documents_store.keys())
    }


@app.get("/index/term/{term}")
def get_term_postings(term: str):
    term = term.lower()

    if term not in inverted_index:
        return {
            "term": term,
            "document_frequency": 0,
            "postings": {}
        }

    return {
        "term": term,
        "document_frequency": len(inverted_index[term]),
        "postings": inverted_index[term]
    }
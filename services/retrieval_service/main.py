from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import math
import re

from dataset_manager import (
    load_dataset_resources,
    get_embedding_model,
    validate_dataset
)

app = FastAPI(
    title="IR Retrieval Service",
    description="Multi-dataset retrieval service for IR search and ranking",
    version="2.0.0"
)


class Document(BaseModel):
    doc_id: str
    text: str


class SearchRequest(BaseModel):
    query: str
    documents: list[Document]
    top_k: int = 10


class BM25SearchRequest(SearchRequest):
    k1: float = 1.5
    b: float = 0.75


class DatasetBM25SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "cranfield"
    k1: float = 1.5
    b: float = 0.75


class IndexedSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "cranfield"


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "cranfield"


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    bm25_weight: float = 0.4
    semantic_weight: float = 0.6
    dataset: str = "cranfield"


class HybridSerialSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    initial_k: int = 50
    dataset: str = "cranfield"


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def preprocess_to_text(text: str) -> str:
    return " ".join(preprocess(text))


def build_inverted_index(documents: list[Document]):
    inverted_index = defaultdict(dict)
    documents_store = {}

    for document in documents:
        tokens = preprocess(document.text)
        documents_store[document.doc_id] = document.text
        term_frequencies = Counter(tokens)

        for term, frequency in term_frequencies.items():
            inverted_index[term][document.doc_id] = frequency

    return inverted_index, documents_store


def score_documents(query_tokens: list[str], inverted_index, documents_store):
    scores = defaultdict(float)

    for term in query_tokens:
        if term in inverted_index:
            for doc_id, frequency in inverted_index[term].items():
                scores[doc_id] += frequency

    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return [
        {
            "rank": index + 1,
            "doc_id": doc_id,
            "score": score,
            "text": documents_store[doc_id]
        }
        for index, (doc_id, score) in enumerate(ranked_results)
    ]


def lexical_scores(query_tokens, resources):
    dataset_type = resources["type"]

    if dataset_type == "cranfield":
        inverted_index = resources["inverted_index"]
        scores = defaultdict(float)

        for term in query_tokens:
            if term in inverted_index:
                for doc_id, frequency in inverted_index[term].items():
                    scores[doc_id] += frequency

        return dict(scores)

    if dataset_type in ["scifact", "generic"]:
        bm25 = resources["bm25"]
        doc_ids = resources["doc_ids"]

        raw_scores = bm25.get_scores(query_tokens)

        return {
            str(doc_ids[index]): float(score)
            for index, score in enumerate(raw_scores)
            if score > 0
        }

    return {}


def normalize_scores(scores: dict):
    if not scores:
        return {}

    max_score = max(scores.values())

    if max_score <= 0:
        return scores

    return {
        doc_id: score / max_score
        for doc_id, score in scores.items()
    }


def semantic_scores(query: str, resources, search_size: int):
    faiss_index = resources["faiss_index"]
    metadata = resources["metadata"]

    if metadata.get("vector_method") == "lsa_tfidf_svd":
        query_tfidf = metadata["vectorizer"].transform([query])
        query_embedding = metadata["svd"].transform(query_tfidf)
        query_embedding = normalize(query_embedding).astype("float32")
    else:
        model = get_embedding_model()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

    scores_raw, indices = faiss_index.search(
        query_embedding,
        min(search_size, resources["total_documents"])
    )

    scores = {}

    for doc_index, score in zip(indices[0], scores_raw[0]):
        if doc_index == -1:
            continue

        doc_id = str(metadata["doc_ids"][doc_index])

        if resources["type"] == "scifact" and metadata.get("vector_method") != "lsa_tfidf_svd":
            converted_score = 1 / (1 + float(score))
        else:
            converted_score = float(score)

        scores[doc_id] = converted_score

    return scores


def get_doc_text(doc_id: str, resources):
    documents_store = resources.get("documents_store", {})

    if doc_id in documents_store:
        return documents_store[doc_id]

    metadata = resources.get("metadata", {})
    doc_ids = [str(item) for item in metadata.get("doc_ids", [])]

    try:
        position = doc_ids.index(str(doc_id))
        return metadata["documents"][position]
    except ValueError:
        return ""


def get_dataset_documents(resources):
    metadata = resources.get("metadata", {})

    if metadata.get("documents") and metadata.get("doc_ids"):
        return [str(item) for item in metadata["doc_ids"]], metadata["documents"]

    documents_store = resources.get("documents_store", {})
    doc_ids = list(documents_store.keys())
    documents = [documents_store[doc_id] for doc_id in doc_ids]

    return doc_ids, documents


def indexed_search(query: str, top_k: int, dataset: str):
    dataset = validate_dataset(dataset)
    resources = load_dataset_resources(dataset)

    query_tokens = preprocess(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    scores = lexical_scores(query_tokens, resources)
    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results = []

    for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1):
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "text": get_doc_text(doc_id, resources)
        })

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Dataset Lexical Index Search",
        "dataset": dataset,
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results
    }


def dataset_tfidf_search(query: str, top_k: int, dataset: str):
    dataset = validate_dataset(dataset)

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    resources = load_dataset_resources(dataset)
    query_tokens = preprocess(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    inverted_index = resources.get("inverted_index", {})
    total_documents = resources["total_documents"]
    scores = defaultdict(float)

    for term in query_tokens:
        postings = inverted_index.get(term, {})

        if not postings:
            continue

        idf = math.log((total_documents + 1) / (len(postings) + 1)) + 1

        for doc_id, term_frequency in postings.items():
            scores[doc_id] += float(term_frequency) * idf

    if not scores and "bm25" in resources:
        bm25 = resources["bm25"]
        doc_ids = resources["doc_ids"]

        for doc_index, term_frequencies in enumerate(bm25.doc_freqs):
            score = 0.0

            for term in query_tokens:
                term_frequency = term_frequencies.get(term, 0)

                if term_frequency:
                    score += float(term_frequency) * float(bm25.idf.get(term, 0.0))

            if score > 0:
                scores[str(doc_ids[doc_index])] = score

    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results = []

    for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1):
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "text": get_doc_text(doc_id, resources)
        })

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Precomputed TF-IDF over Inverted Index",
        "dataset": dataset,
        "total_documents": total_documents,
        "returned_results": len(results),
        "results": results
    }


def dataset_bm25_search(query: str, top_k: int, dataset: str, k1: float, b: float):
    dataset = validate_dataset(dataset)

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    resources = load_dataset_resources(dataset)
    doc_ids, documents = get_dataset_documents(resources)
    wrapped_documents = [
        Document(doc_id=doc_id, text=text)
        for doc_id, text in zip(doc_ids, documents)
    ]

    result = bm25_search(query, wrapped_documents, top_k, k1, b)
    result["dataset"] = dataset
    return result


def semantic_search(query: str, top_k: int, dataset: str):
    dataset = validate_dataset(dataset)

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    resources = load_dataset_resources(dataset)
    scores = semantic_scores(query, resources, top_k)

    ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results = []

    for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1):
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "text": get_doc_text(doc_id, resources)
        })

    return {
        "query": query,
        "model": "Semantic Search using Sentence Transformers + FAISS",
        "dataset": dataset,
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results
    }


def hybrid_search(query: str, top_k: int, bm25_weight: float, semantic_weight: float, dataset: str):
    dataset = validate_dataset(dataset)

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if bm25_weight < 0 or semantic_weight < 0:
        raise HTTPException(status_code=400, detail="Weights must be non-negative")

    if bm25_weight + semantic_weight == 0:
        raise HTTPException(status_code=400, detail="At least one weight must be greater than 0")

    resources = load_dataset_resources(dataset)
    query_tokens = preprocess(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    bm25_scores = normalize_scores(lexical_scores(query_tokens, resources))
    semantic_raw = semantic_scores(query, resources, top_k * 5)
    semantic_normalized = normalize_scores(semantic_raw)

    all_doc_ids = set(bm25_scores.keys()) | set(semantic_normalized.keys())

    final_scores = {}

    for doc_id in all_doc_ids:
        bm25_score = bm25_scores.get(doc_id, 0.0)
        semantic_score = semantic_normalized.get(doc_id, 0.0)

        final_scores[doc_id] = (
            bm25_weight * bm25_score +
            semantic_weight * semantic_score
        )

    ranked_results = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

    results = []

    for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1):
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "bm25_score": round(float(bm25_scores.get(doc_id, 0.0)), 6),
            "semantic_score": round(float(semantic_normalized.get(doc_id, 0.0)), 6),
            "text": get_doc_text(doc_id, resources)
        })

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Hybrid Parallel Search (BM25 + Semantic FAISS Score Fusion)",
        "dataset": dataset,
        "weights": {
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight
        },
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results
    }


def hybrid_serial_search(query: str, top_k: int, initial_k: int, dataset: str):
    dataset = validate_dataset(dataset)

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    if initial_k <= 0:
        raise HTTPException(status_code=400, detail="initial_k must be greater than 0")

    if initial_k < top_k:
        initial_k = top_k

    resources = load_dataset_resources(dataset)
    query_tokens = preprocess(query)

    if not query_tokens:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    bm25_scores_raw = lexical_scores(query_tokens, resources)
    bm25_scores = normalize_scores(bm25_scores_raw)

    ranked_bm25 = sorted(
        bm25_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    candidate_docs = ranked_bm25[:initial_k]

    if not candidate_docs:
        return {
            "query": query,
            "processed_query": query_tokens,
            "model": "Hybrid Serial Search (BM25 Candidate Generation → Semantic Re-ranking)",
            "dataset": dataset,
            "initial_candidate_count": initial_k,
            "returned_results": 0,
            "results": []
        }

    candidate_doc_ids = {doc_id for doc_id, _ in candidate_docs}

    semantic_raw = semantic_scores(
        query=query,
        resources=resources,
        search_size=max(initial_k * 3, top_k)
    )

    semantic_filtered = {
        doc_id: score
        for doc_id, score in semantic_raw.items()
        if doc_id in candidate_doc_ids
    }

    semantic_normalized = normalize_scores(semantic_filtered)

    final_scores = {}

    for doc_id in candidate_doc_ids:
        final_scores[doc_id] = semantic_normalized.get(doc_id, 0.0)

    ranked_results = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

    results = []

    for rank, (doc_id, score) in enumerate(ranked_results[:top_k], start=1):
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 6),
            "bm25_candidate_score": round(float(bm25_scores.get(doc_id, 0.0)), 6),
            "semantic_rerank_score": round(float(semantic_normalized.get(doc_id, 0.0)), 6),
            "text": get_doc_text(doc_id, resources)
        })

    return {
        "query": query,
        "processed_query": query_tokens,
        "model": "Hybrid Serial Search (BM25 Candidate Generation → Semantic Re-ranking)",
        "dataset": dataset,
        "initial_candidate_count": initial_k,
        "total_documents": resources["total_documents"],
        "returned_results": len(results),
        "results": results
    }


def tfidf_search(query: str, documents: list[Document], top_k: int):
    doc_ids = [doc.doc_id for doc in documents]
    original_texts = [doc.text for doc in documents]
    processed_documents = [preprocess_to_text(doc.text) for doc in documents]
    processed_query = preprocess_to_text(query)

    if not processed_query:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    vectorizer = TfidfVectorizer()
    document_matrix = vectorizer.fit_transform(processed_documents)
    query_vector = vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(query_vector, document_matrix).flatten()
    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for rank, doc_index in enumerate(ranked_indices[:top_k], start=1):
        score = float(similarity_scores[doc_index])

        if score == 0:
            continue

        results.append({
            "rank": rank,
            "doc_id": doc_ids[doc_index],
            "score": round(score, 6),
            "text": original_texts[doc_index]
        })

    return {
        "query": query,
        "processed_query": processed_query,
        "model": "TF-IDF Vector Space Model",
        "total_documents": len(documents),
        "returned_results": len(results),
        "results": results
    }


def bm25_search(query: str, documents: list[Document], top_k: int, k1: float, b: float):
    if k1 <= 0:
        raise HTTPException(status_code=400, detail="k1 must be greater than 0")

    if b < 0 or b > 1:
        raise HTTPException(status_code=400, detail="b must be between 0 and 1")

    doc_ids = [doc.doc_id for doc in documents]
    original_texts = [doc.text for doc in documents]
    tokenized_documents = [preprocess(doc.text) for doc in documents]
    tokenized_query = preprocess(query)

    if not tokenized_query:
        raise HTTPException(status_code=400, detail="Query has no valid searchable terms")

    bm25 = BM25Okapi(tokenized_documents, k1=k1, b=b)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = scores.argsort()[::-1]

    results = []

    for rank, doc_index in enumerate(ranked_indices[:top_k], start=1):
        score = float(scores[doc_index])

        if score == 0:
            continue

        results.append({
            "rank": rank,
            "doc_id": doc_ids[doc_index],
            "score": round(score, 6),
            "text": original_texts[doc_index]
        })

    return {
        "query": query,
        "processed_query": tokenized_query,
        "model": "BM25",
        "parameters": {"k1": k1, "b": b},
        "total_documents": len(documents),
        "returned_results": len(results),
        "results": results
    }


@app.get("/")
def home():
    return {
        "service": "Retrieval Service",
        "status": "running",
        "version": "2.0.0",
        "supported_datasets": [
            "quora",
        ],
        "available_models": [
            "term_frequency_baseline",
            "tfidf_vsm",
            "bm25",
            "indexed_search",
            "semantic_search",
            "hybrid_parallel_search",
            "hybrid_serial_search"
        ]
    }


@app.post("/search")
def search(request: SearchRequest):
    query_tokens = preprocess(request.query)
    inverted_index, documents_store = build_inverted_index(request.documents)
    results = score_documents(query_tokens, inverted_index, documents_store)

    return {
        "query": request.query,
        "processed_query": query_tokens,
        "model": "Term Frequency Baseline",
        "total_documents": len(request.documents),
        "returned_results": len(results[:request.top_k]),
        "results": results[:request.top_k]
    }


@app.post("/search/tfidf")
def search_tfidf(request: SearchRequest):
    return tfidf_search(request.query, request.documents, request.top_k)


@app.post("/search/bm25")
def search_bm25(request: BM25SearchRequest):
    return bm25_search(request.query, request.documents, request.top_k, request.k1, request.b)


@app.post("/search/dataset/tfidf")
def search_dataset_tfidf(request: IndexedSearchRequest):
    return dataset_tfidf_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset
    )


@app.post("/search/dataset/bm25")
def search_dataset_bm25(request: DatasetBM25SearchRequest):
    return dataset_bm25_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset,
        k1=request.k1,
        b=request.b
    )


@app.post("/search/indexed")
def search_indexed(request: IndexedSearchRequest):
    return indexed_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset
    )


@app.post("/search/semantic")
def search_semantic(request: SemanticSearchRequest):
    return semantic_search(
        query=request.query,
        top_k=request.top_k,
        dataset=request.dataset
    )


@app.post("/search/hybrid")
def search_hybrid(request: HybridSearchRequest):
    return hybrid_search(
        query=request.query,
        top_k=request.top_k,
        bm25_weight=request.bm25_weight,
        semantic_weight=request.semantic_weight,
        dataset=request.dataset
    )


@app.post("/search/hybrid/serial")
def search_hybrid_serial(request: HybridSerialSearchRequest):
    return hybrid_serial_search(
        query=request.query,
        top_k=request.top_k,
        initial_k=request.initial_k,
        dataset=request.dataset
    )

TFIDF_SEARCH_URL = "http://127.0.0.1:8003/search/dataset/tfidf"
BM25_SEARCH_URL = "http://127.0.0.1:8003/search/dataset/bm25"
SEMANTIC_SEARCH_URL = "http://127.0.0.1:8003/search/semantic"
HYBRID_PARALLEL_SEARCH_URL = "http://127.0.0.1:8003/search/hybrid"
HYBRID_SERIAL_SEARCH_URL = "http://127.0.0.1:8003/search/hybrid/serial"


def build_tfidf_payload(request, refined_query: str):
    return TFIDF_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "dataset": request.dataset,
    }


def build_bm25_payload(request, refined_query: str):
    return BM25_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "dataset": request.dataset,
        "k1": request.bm25_k1,
        "b": request.bm25_b,
    }


def build_semantic_payload(request, refined_query: str):
    return SEMANTIC_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "dataset": request.dataset,
    }


def build_hybrid_parallel_payload(request, refined_query: str):
    return HYBRID_PARALLEL_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "bm25_weight": request.bm25_weight,
        "semantic_weight": request.semantic_weight,
        "k1": request.bm25_k1,
        "b": request.bm25_b,
        "dataset": request.dataset,
    }


def build_hybrid_serial_payload(request, refined_query: str):
    return HYBRID_SERIAL_SEARCH_URL, {
        "query": refined_query,
        "top_k": request.top_k,
        "initial_k": request.initial_k,
        "k1": request.bm25_k1,
        "b": request.bm25_b,
        "dataset": request.dataset,
    }


RETRIEVAL_STRATEGIES = {
    "tfidf": build_tfidf_payload,
    "bm25": build_bm25_payload,
    "semantic": build_semantic_payload,
    "hybrid_parallel": build_hybrid_parallel_payload,
    "hybrid_serial": build_hybrid_serial_payload,
}

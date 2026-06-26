SERVICE_URLS = {
    "Preprocessing Service": "http://127.0.0.1:8001/",
    "Indexing Service": "http://127.0.0.1:8002/",
    "Gateway Service": "http://127.0.0.1:8006/",
    "Retrieval Service": "http://127.0.0.1:8003/",
    "Evaluation Service": "http://127.0.0.1:8004/",
    "Refinement Service": "http://127.0.0.1:8005/",
}

GATEWAY_URL = "http://127.0.0.1:8006/search/full"

RETRIEVAL_MODE_OPTIONS = {
    "TF-IDF": "tfidf",
    "BM25": "bm25",
    "Semantic": "semantic",
    "Hybrid Parallel": "hybrid_parallel",
    "Hybrid Serial": "hybrid_serial",
}

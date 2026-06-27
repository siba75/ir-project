TOP_K = 10
DEFAULT_WORKERS = 4

SEARCH_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial",
]

EVALUATION_RUNS = {
    **{
        f"before_{mode}": {
            "phase": "before_features",
            "mode": mode,
            "description": f"Core {mode} retrieval before additional features.",
        }
        for mode in SEARCH_MODES
    },
    **{
        f"after_{mode}": {
            "phase": "after_features",
            "mode": mode,
            "description": (
                f"{mode} retrieval after enabling personalization. "
                "Semantic and hybrid modes also use the FAISS vector store."
            ),
        }
        for mode in SEARCH_MODES
    },
}

VECTOR_MODES = {"semantic", "hybrid_parallel", "hybrid_serial"}

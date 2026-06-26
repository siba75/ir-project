import streamlit as st

from ui_config import RETRIEVAL_MODE_OPTIONS, SERVICE_URLS
from ui_helpers import check_service


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Search Settings")
        st.session_state.dark_mode = st.toggle(
            "🌙 Dark Mode",
            value=st.session_state.dark_mode,
        )

        st.metric("Dataset", "Quora")
        st.caption("Fixed dataset: beir/quora/test")
        st.caption("Documents: SQLite DB · Indexes: compressed cache files")

        retrieval_mode_label = st.selectbox(
            "Retrieval Mode",
            list(RETRIEVAL_MODE_OPTIONS.keys()),
        )
        retrieval_mode = RETRIEVAL_MODE_OPTIONS[retrieval_mode_label]

        top_k = st.slider("Number of Results", 1, 20, 10)
        bm25_k1, bm25_b = render_bm25_controls(retrieval_mode)
        bm25_weight, semantic_weight, initial_k = render_hybrid_controls(retrieval_mode, top_k)

        st.divider()
        feature_flags = render_feature_flags()
        render_system_status()
        render_search_history()

    return {
        "dataset": "quora",
        "dataset_label": "Quora",
        "retrieval_mode": retrieval_mode,
        "retrieval_mode_label": retrieval_mode_label,
        "top_k": top_k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_weight": bm25_weight,
        "semantic_weight": semantic_weight,
        "initial_k": initial_k,
        **feature_flags,
    }


def render_bm25_controls(retrieval_mode):
    if retrieval_mode not in ["bm25", "hybrid_parallel", "hybrid_serial"]:
        return 1.5, 0.75

    bm25_k1 = st.slider("BM25 k1", 0.1, 3.0, 1.5, 0.1)
    bm25_b = st.slider("BM25 b", 0.0, 1.0, 0.75, 0.05)
    st.caption(
        "BM25 parameters are applied at query time. Change k1 or b, run "
        "the same query again, and compare the ranked results and scores."
    )

    with st.expander("How BM25 parameters affect results"):
        st.write(
            "k1 controls term-frequency saturation. Higher values let "
            "repeated query terms influence the score more. b controls "
            "document-length normalization. Higher values penalize long "
            "documents more strongly."
        )

    return bm25_k1, bm25_b


def render_hybrid_controls(retrieval_mode, top_k):
    if retrieval_mode == "hybrid_parallel":
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.4, 0.05)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.6, 0.05)
        st.caption(
            "Parallel hybrid: BM25 and Semantic FAISS run together, then their "
            "normalized scores are fused using the selected weights."
        )
        return bm25_weight, semantic_weight, 50

    if retrieval_mode == "hybrid_serial":
        initial_k = st.slider(
            "Initial Candidates",
            min_value=top_k,
            max_value=200,
            value=max(50, top_k),
            step=5,
        )
        st.caption(
            "Serial mode: BM25 retrieves initial candidates, then Semantic Search re-ranks them."
        )
        return 0.4, 0.6, initial_k

    return 0.4, 0.6, 50


def render_feature_flags():
    return {
        "remove_stopwords": st.checkbox("Remove Stopwords", value=True),
        "use_lemmatization": st.checkbox("Lemmatization", value=False),
        "use_expansion": st.checkbox("Query Expansion", value=True),
        "use_stemming": st.checkbox("Stemming", value=False),
        "use_personalization": st.checkbox("Personalization", value=False),
        "use_clustering": st.checkbox("Result Clustering", value=True),
    }


def render_system_status():
    st.divider()
    st.header("🟢 System Status")

    for service_name, service_url in SERVICE_URLS.items():
        status = check_service(service_url)
        st.write(f"{'✅' if status else '❌'} {service_name}")


def render_search_history():
    st.divider()
    st.header("🕘 Search History")

    if st.session_state.search_history:
        for item in reversed(st.session_state.search_history[-8:]):
            st.caption(
                f"{item['time']} — {item['query']} | {item['dataset']} | "
                f"{item['mode']} ({item['results']} results)"
            )
    else:
        st.caption("No searches yet.")

    if st.button("Clear History"):
        st.session_state.search_history = []
        st.session_state.last_results = []
        st.session_state.last_response = None
        st.session_state.last_search_time = None
        st.rerun()

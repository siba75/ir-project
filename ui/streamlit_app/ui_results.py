import streamlit as st

from ui_helpers import extract_top_terms, get_score_value, highlight_terms


def render_results_tab(data, results, config):
    if not data:
        st.info("Run a search to view ranked results.")
        return

    render_query_metrics(data)
    render_storage_metrics(data)
    render_matching_terms(data)
    render_ranked_results(data, results, config)


def render_query_metrics(data):
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Query", data.get("original_query", ""))
    col2.metric("Refined Query", data.get("refined_query", ""))
    col3.metric(
        "Search Time",
        f"{st.session_state.last_search_time:.3f} sec"
        if st.session_state.last_search_time is not None
        else "N/A",
    )

    mode_col1, mode_col2, mode_col3, mode_col4 = st.columns(4)
    mode_col1.metric("Dataset", "Quora")
    mode_col2.metric("Retrieval Mode", data.get("retrieval_mode", "N/A"))
    mode_col3.metric("Retrieval Model", data.get("retrieval_model", "N/A"))
    mode_col4.metric("Vector Method", data.get("vector_method", "N/A"))


def render_storage_metrics(data):
    storage = data.get("storage", {})
    st.subheader("Storage and Cache")
    col1, col2, col3 = st.columns(3)
    col1.metric("Document Store", "SQLite" if storage.get("documents_in_sqlite") else "Unavailable")
    col2.metric("Compressed Indexes", "Enabled" if storage.get("compressed_indexes") else "Fallback")
    col3.metric("Resource Cache", "Enabled" if storage.get("resource_cache") else "N/A")


def render_matching_terms(data):
    st.subheader("Top Matching Terms")
    terms = extract_top_terms(data.get("refined_query", ""))
    st.write(" ".join([f"`{term}`" for term in terms]))


def render_ranked_results(data, results, config):
    st.subheader("Ranked Results")

    if not results:
        st.warning("No results found.")
        return

    for result in results:
        final_score, bm25_score, semantic_score = result_scores(result, config)
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='doc-title'>Rank {result['rank']} — Document {result['doc_id']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='small-muted'>Final Score: {final_score} | "
            f"BM25: {bm25_score} | Semantic: {semantic_score}</div>",
            unsafe_allow_html=True,
        )
        highlighted_text = highlight_terms(
            result.get("text", ""),
            data.get("refined_query", ""),
        )
        st.markdown(highlighted_text, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Score", final_score)
        c2.metric("BM25 Score", bm25_score)
        c3.metric("Semantic Score", semantic_score)
        st.markdown("</div>", unsafe_allow_html=True)


def result_scores(result, config):
    final_score = result.get("score", 0)
    bm25_score = get_score_value(result, "bm25_score", "bm25_candidate_score")
    semantic_score = get_score_value(result, "semantic_score", "semantic_rerank_score")
    mode = config["retrieval_mode"]

    if mode == "semantic":
        return final_score, 0, final_score

    if mode == "bm25":
        return final_score, final_score, 0

    if mode == "tfidf":
        return final_score, 0, 0

    return final_score, bm25_score, semantic_score

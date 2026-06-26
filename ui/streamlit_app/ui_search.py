import time
from datetime import datetime

import requests
import streamlit as st

from ui_config import GATEWAY_URL


def render_header():
    st.markdown(
        "<div class='main-title'>🔎 Information Retrieval Search Engine</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Quora IR System: Query Refinement + BM25 + Semantic FAISS Search</div>",
        unsafe_allow_html=True,
    )
    st.divider()


def render_search_form(config):
    query = st.text_input(
        "Enter your search query",
        placeholder="Example: how can I learn programming?",
    )
    search_clicked = st.button("Search", type="primary")

    if search_clicked:
        run_search(query, config)


def run_search(query, config):
    if not query.strip():
        st.warning("Please enter a query.")
        return

    if (
        config["retrieval_mode"] == "hybrid_parallel"
        and config["bm25_weight"] + config["semantic_weight"] == 0
    ):
        st.warning("At least one ranking weight must be greater than 0.")
        return

    payload = build_search_payload(query, config)

    with st.spinner("Running full IR pipeline..."):
        try:
            start_time = time.perf_counter()
            response = requests.post(GATEWAY_URL, json=payload, timeout=120)
            elapsed_time = time.perf_counter() - start_time

            if response.status_code != 200:
                st.error(
                    "Search failed. Make sure Gateway, Retrieval, and Refinement services are running."
                )
                st.code(response.text)
                return

            store_successful_search(query, response.json(), elapsed_time, config)
            st.success("Search completed successfully")
        except Exception as error:
            st.error(f"Connection error: {error}")


def build_search_payload(query, config):
    return {
        "query": query,
        "top_k": config["top_k"],
        "dataset": config["dataset"],
        "retrieval_mode": config["retrieval_mode"],
        "bm25_weight": config["bm25_weight"],
        "semantic_weight": config["semantic_weight"],
        "bm25_k1": config["bm25_k1"],
        "bm25_b": config["bm25_b"],
        "initial_k": config["initial_k"],
        "remove_stopwords": config["remove_stopwords"],
        "use_lemmatization": config["use_lemmatization"],
        "use_stemming": config["use_stemming"],
        "use_expansion": config["use_expansion"],
        "use_personalization": config["use_personalization"],
        "user_history": [
            item["query"]
            for item in st.session_state.search_history[-20:]
        ],
    }


def store_successful_search(query, data, elapsed_time, config):
    results = data.get("results", [])
    st.session_state.previous_response = st.session_state.last_response
    st.session_state.last_response = data
    st.session_state.last_results = results
    st.session_state.last_search_time = elapsed_time
    st.session_state.search_history.append({
        "query": query,
        "time": datetime.now().strftime("%H:%M:%S"),
        "results": len(results),
        "mode": config["retrieval_mode_label"],
        "dataset": config["dataset_label"],
        "bm25_k1": config["bm25_k1"],
        "bm25_b": config["bm25_b"],
    })

import json

import streamlit as st

from ui_helpers import results_to_dataframe


def render_pipeline_tab(data, config, resource_manifest):
    st.subheader("Pipeline Details")
    render_pipeline_diagram(data)

    if data:
        st.json({
            "original_query": data.get("original_query"),
            "refined_query": data.get("refined_query"),
            "dataset": "quora",
            "retrieval_mode": data.get("retrieval_mode"),
            "retrieval_model": data.get("retrieval_model"),
            "vector_method": data.get("vector_method"),
            "storage": data.get("storage", {}),
            "additional_features": data.get("additional_features", {}),
            "configuration": {
                **{key: config[key] for key in [
                    "dataset",
                    "top_k",
                    "bm25_weight",
                    "semantic_weight",
                    "bm25_k1",
                    "bm25_b",
                    "initial_k",
                    "remove_stopwords",
                    "use_lemmatization",
                    "use_expansion",
                    "use_stemming",
                    "use_personalization",
                    "use_clustering",
                ]},
                "search_time_seconds": st.session_state.last_search_time,
            },
        })
    else:
        st.info("Run a search to view the latest pipeline configuration.")

    st.subheader("Submission Resources")
    st.json(resource_manifest) if resource_manifest else st.info(
        "Resource manifest not found. Run prepare_submission_resources.py."
    )


def render_pipeline_diagram(data):
    if data and data.get("retrieval_mode") == "hybrid_serial":
        st.code("""
User Query
   ↓
Query Refinement
   ↓
Fixed Quora Dataset
   ↓
Hybrid Serial Retrieval
   ├── BM25 Candidate Generation
   └── Semantic FAISS Re-ranking
   ↓
Ranked Results
        """)
        return

    st.code("""
User Query
   ↓
Query Refinement
   ↓
Fixed Quora Dataset
   ↓
Hybrid Parallel Retrieval
   ├── BM25 Lexical Search
   └── Semantic FAISS Search
   ↓
Score Fusion
   ↓
Ranked Results
        """)


def render_export_tab(data, results):
    st.subheader("Export Search Results")

    if not results:
        st.info("Run a search first to export results.")
        return

    df = results_to_dataframe(results)
    csv_data = df.to_csv(index=False).encode("utf-8")
    json_data = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    col_csv, col_json = st.columns(2)

    with col_csv:
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="ir_search_results.csv",
            mime="text/csv",
        )

    with col_json:
        st.download_button(
            label="Download Full Response as JSON",
            data=json_data,
            file_name="ir_search_response.json",
            mime="application/json",
        )

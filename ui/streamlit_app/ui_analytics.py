import pandas as pd
import streamlit as st

from ui_helpers import (
    cluster_results,
    parameter_comparison_dataframe,
    results_to_dataframe,
)


def render_analytics_tab(data, results, config):
    st.subheader("Search Analytics")

    if not results:
        st.info("Run a search to view analytics.")
        return

    df = results_to_dataframe(results)
    st.dataframe(df, use_container_width=True)
    st.subheader("Score Comparison")
    st.bar_chart(df.set_index("Document")[["Final Score", "BM25 Score", "Semantic Score"]])
    st.subheader("Ranking Curve")
    st.line_chart(df[["Rank", "Final Score"]].set_index("Rank"))
    render_parameter_comparison(data)
    render_clustering(results, config["use_clustering"])
    render_score_summary(df)
    render_personalization_profile(data)


def render_parameter_comparison(data):
    previous_response = st.session_state.previous_response

    if not (
        previous_response
        and data
        and previous_response.get("original_query") == data.get("original_query")
        and previous_response.get("retrieval_mode") == data.get("retrieval_mode")
        and data.get("retrieval_mode") in ["bm25", "hybrid_parallel", "hybrid_serial"]
    ):
        return

    previous_config = previous_response.get("configuration", {})
    current_config = data.get("configuration", {})
    st.subheader("BM25 Parameter Difference")
    c_prev, c_curr = st.columns(2)
    c_prev.metric(
        "Previous k1 / b",
        f"{previous_config.get('bm25_k1', 1.5)} / {previous_config.get('bm25_b', 0.75)}",
    )
    c_curr.metric(
        "Current k1 / b",
        f"{current_config.get('bm25_k1', 1.5)} / {current_config.get('bm25_b', 0.75)}",
    )
    st.dataframe(parameter_comparison_dataframe(previous_response, data), use_container_width=True)


def render_clustering(results, use_clustering):
    st.subheader("Result Clustering")

    if not use_clustering:
        st.info(
            "Result clustering is disabled. Enable it from the sidebar to compare search with and without clustering."
        )
        return

    try:
        cluster_df = cluster_results(results)

        if cluster_df.empty:
            st.info("At least two results are needed for clustering.")
            return

        st.dataframe(cluster_df, use_container_width=True)
        st.bar_chart(cluster_df.groupby("Cluster")["Document"].count())
    except Exception as error:
        st.warning(f"Clustering unavailable for these results: {error}")


def render_score_summary(df):
    avg_score = df["Final Score"].mean()
    max_score = df["Final Score"].max()
    min_score = df["Final Score"].min()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Returned Results", len(df))
    m2.metric("Average Score", round(avg_score, 4))
    m3.metric("Best Score", round(max_score, 4))
    m4.metric("Lowest Score", round(min_score, 4))


def render_personalization_profile(data):
    profile = data.get("additional_features", {}).get("personalization_profile", {})

    if not profile.get("enabled"):
        return

    st.subheader("Personalization IR Profile")
    p1, p2, p3 = st.columns(3)
    p1.metric("History Queries Used", profile.get("history_queries_used", 0))
    p2.metric("Query Vector Weight", profile.get("query_vector_weight", 0))
    p3.metric("Interest Vector Weight", profile.get("interest_vector_weight", 0))
    render_profile_table("User Interest Terms from Search History", profile.get("interest_terms", []), "term")
    render_profile_table("Terms Selected After Query-Interest Fusion", profile.get("combined_terms", []), "term")
    render_profile_table("Most Similar History Queries", profile.get("similar_history_queries", []), "query")

    suggestions = profile.get("query_suggestions", [])

    if suggestions:
        st.subheader("IR-based Query Suggestions")
        st.write(" ".join([f"`{suggestion}`" for suggestion in suggestions]))


def render_profile_table(title, rows, index_column):
    if not rows:
        return

    df = pd.DataFrame(rows)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    value_column = "score" if "score" in df.columns else "similarity"
    st.bar_chart(df.set_index(index_column)[value_column])

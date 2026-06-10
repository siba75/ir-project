import streamlit as st
import requests
import pandas as pd
import re
import json
import time
from datetime import datetime

GATEWAY_URL = "http://127.0.0.1:8006/search/full"

SERVICE_URLS = {
    "Gateway Service": "http://127.0.0.1:8006/",
    "Retrieval Service": "http://127.0.0.1:8003/",
    "Refinement Service": "http://127.0.0.1:8005/"
}

EVALUATION_RESULTS = {
    "Evaluated Queries": 50,
    "Mean Precision@10": 0.1940,
    "Mean Recall@10": 0.3364,
    "MRR": 0.4740,
    "MAP": 0.1869
}

st.set_page_config(
    page_title="IR Search Engine",
    page_icon="🔎",
    layout="wide"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "last_search_time" not in st.session_state:
    st.session_state.last_search_time = None


def apply_theme():
    if st.session_state.dark_mode:
        bg = "#0b1120"
        card = "#111827"
        panel = "#1e293b"
        text = "#f8fafc"
        muted = "#cbd5e1"
        border = "#334155"
        input_bg = "#0f172a"
    else:
        bg = "#ffffff"
        card = "#ffffff"
        panel = "#f8fafc"
        text = "#1f2937"
        muted = "#6b7280"
        border = "#e5e7eb"
        input_bg = "#ffffff"

    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {panel};
        color: {text};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    .main-title {{
        font-size: 52px;
        font-weight: 800;
        margin-bottom: 0;
        color: {text};
    }}

    .subtitle {{
        font-size: 18px;
        color: {muted};
        margin-top: 0;
    }}

    .result-card {{
        padding: 22px;
        border-radius: 16px;
        border: 1px solid {border};
        margin-bottom: 18px;
        background-color: {card};
        color: {text};
    }}

    .doc-title {{
        font-size: 24px;
        font-weight: 700;
        color: {text};
    }}

    .small-muted {{
        color: {muted};
        font-size: 14px;
    }}

    .highlight {{
        background-color: #facc15;
        color: #111827;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 700;
    }}

    div[data-testid="stMetric"] {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 14px;
    }}

    div[data-testid="stMetric"] * {{
        color: {text} !important;
    }}

    input, textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {panel};
        color: {text};
        border-radius: 10px;
        padding: 10px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: #2563eb !important;
        color: white !important;
    }}

    .footer {{
        text-align: center;
        color: {muted};
        padding: 30px;
        font-size: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)


def check_service(url):
    try:
        response = requests.get(url, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def extract_top_terms(refined_query):
    terms = re.findall(r"\w+", refined_query.lower())
    terms = [term for term in terms if len(term) > 2]
    return list(dict.fromkeys(terms))[:12]


def highlight_terms(text, query):
    terms = [
        term
        for term in re.findall(r"\w+", query.lower())
        if len(term) > 2
    ]

    highlighted = text

    for term in terms:
        highlighted = re.sub(
            f"(?i)({re.escape(term)})",
            r"<span class='highlight'>\1</span>",
            highlighted
        )

    return highlighted


def truncate_text(text, max_chars=900):
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def get_score_value(item, parallel_key, serial_key):
    return item.get(parallel_key, item.get(serial_key, 0))


def results_to_dataframe(results):
    return pd.DataFrame([
        {
            "Rank": item.get("rank"),
            "Document": item.get("doc_id"),
            "Final Score": item.get("score", 0),
            "BM25 Score": get_score_value(item, "bm25_score", "bm25_candidate_score"),
            "Semantic Score": get_score_value(item, "semantic_score", "semantic_rerank_score"),
            "Text": item.get("text", "")
        }
        for item in results
    ])


with st.sidebar:
    st.header("⚙️ Search Settings")

    st.session_state.dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

    dataset_label = st.selectbox(
        "Dataset",
        [
            "Cranfield",
            "SciFact"
        ]
    )

    dataset = dataset_label.lower()

    retrieval_mode_label = st.selectbox(
        "Retrieval Mode",
        [
            "Hybrid Parallel",
            "Hybrid Serial"
        ]
    )

    retrieval_mode = (
        "hybrid_parallel"
        if retrieval_mode_label == "Hybrid Parallel"
        else "hybrid_serial"
    )

    top_k = st.slider("Number of Results", 1, 20, 5)

    if retrieval_mode == "hybrid_parallel":
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.4, 0.05)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.6, 0.05)
        initial_k = 50
    else:
        bm25_weight = 0.4
        semantic_weight = 0.6

        initial_k = st.slider(
            "Initial Candidates",
            min_value=top_k,
            max_value=200,
            value=max(50, top_k),
            step=5
        )

        st.caption(
            "Serial mode: BM25 retrieves initial candidates, then Semantic Search re-ranks them."
        )

    st.divider()

    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
    use_expansion = st.checkbox("Query Expansion", value=True)
    use_stemming = st.checkbox("Stemming", value=False)

    st.divider()

    st.header("🟢 System Status")

    for service_name, service_url in SERVICE_URLS.items():
        status = check_service(service_url)
        st.write(f"{'✅' if status else '❌'} {service_name}")

    st.divider()

    st.header("🕘 Search History")

    if st.session_state.search_history:
        for item in reversed(st.session_state.search_history[-8:]):
            st.caption(
                f"{item['time']} — {item['query']} | {item['dataset']} | {item['mode']} ({item['results']} results)"
            )
    else:
        st.caption("No searches yet.")

    if st.button("Clear History"):
        st.session_state.search_history = []
        st.session_state.last_results = []
        st.session_state.last_response = None
        st.session_state.last_search_time = None
        st.rerun()


apply_theme()

st.markdown(
    "<div class='main-title'>🔎 Information Retrieval Search Engine</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Professional Multi-Dataset IR System: Query Refinement + BM25 + Semantic FAISS Search</div>",
    unsafe_allow_html=True
)

st.divider()

query = st.text_input(
    "Enter your search query",
    placeholder="Example: aircraft wing flow / cancer treatment"
)

search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Please enter a query.")

    elif retrieval_mode == "hybrid_parallel" and bm25_weight + semantic_weight == 0:
        st.warning("At least one ranking weight must be greater than 0.")

    else:
        payload = {
            "query": query,
            "top_k": top_k,
            "dataset": dataset,
            "retrieval_mode": retrieval_mode,
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight,
            "initial_k": initial_k,
            "remove_stopwords": remove_stopwords,
            "use_stemming": use_stemming,
            "use_expansion": use_expansion
        }

        with st.spinner("Running full IR pipeline..."):
            try:
                start_time = time.perf_counter()

                response = requests.post(
                    GATEWAY_URL,
                    json=payload,
                    timeout=120
                )

                elapsed_time = time.perf_counter() - start_time

                if response.status_code != 200:
                    st.error(
                        "Search failed. Make sure Gateway, Retrieval, and Refinement services are running."
                    )
                    st.code(response.text)
                else:
                    data = response.json()
                    results = data.get("results", [])

                    st.session_state.last_response = data
                    st.session_state.last_results = results
                    st.session_state.last_search_time = elapsed_time

                    st.session_state.search_history.append({
                        "query": query,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "results": len(results),
                        "mode": retrieval_mode_label,
                        "dataset": dataset_label
                    })

                    st.success("Search completed successfully")

            except Exception as error:
                st.error(f"Connection error: {error}")


data = st.session_state.last_response
results = st.session_state.last_results

tab_results, tab_analytics, tab_evaluation, tab_pipeline, tab_export = st.tabs(
    ["📄 Results", "📊 Analytics", "📈 Evaluation", "🧠 Pipeline", "⬇️ Export"]
)

with tab_results:
    if not data:
        st.info("Run a search to view ranked results.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Original Query", data.get("original_query", ""))
        col2.metric("Refined Query", data.get("refined_query", ""))

        if st.session_state.last_search_time is not None:
            col3.metric("Search Time", f"{st.session_state.last_search_time:.3f} sec")
        else:
            col3.metric("Search Time", "N/A")

        mode_col1, mode_col2, mode_col3 = st.columns(3)
        mode_col1.metric("Dataset", data.get("dataset", "N/A"))
        mode_col2.metric("Retrieval Mode", data.get("retrieval_mode", "N/A"))
        mode_col3.metric("Retrieval Model", data.get("retrieval_model", "N/A"))

        st.subheader("Top Matching Terms")

        terms = extract_top_terms(data.get("refined_query", ""))
        st.write(" ".join([f"`{term}`" for term in terms]))

        st.subheader("Ranked Results")

        if not results:
            st.warning("No results found.")
        else:
            for result in results:
                final_score = result.get("score", 0)

                bm25_score = get_score_value(
                    result,
                    "bm25_score",
                    "bm25_candidate_score"
                )

                semantic_score = get_score_value(
                    result,
                    "semantic_score",
                    "semantic_rerank_score"
                )

                st.markdown("<div class='result-card'>", unsafe_allow_html=True)

                st.markdown(
                    f"<div class='doc-title'>Rank {result['rank']} — Document {result['doc_id']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='small-muted'>Final Score: {final_score} | BM25: {bm25_score} | Semantic: {semantic_score}</div>",
                    unsafe_allow_html=True
                )

                highlighted_text = highlight_terms(
                    truncate_text(result.get("text", "")),
                    data.get("refined_query", query)
                )

                st.markdown(highlighted_text, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Final Score", final_score)
                c2.metric("BM25 Score", bm25_score)
                c3.metric("Semantic Score", semantic_score)

                st.markdown("</div>", unsafe_allow_html=True)


with tab_analytics:
    st.subheader("Search Analytics")

    if results:
        df = results_to_dataframe(results)

        st.dataframe(df, use_container_width=True)

        st.subheader("Score Comparison")

        st.bar_chart(
            df.set_index("Document")[
                ["Final Score", "BM25 Score", "Semantic Score"]
            ]
        )

        st.subheader("Ranking Curve")

        ranking_df = df[["Rank", "Final Score"]].set_index("Rank")
        st.line_chart(ranking_df)

        avg_score = df["Final Score"].mean()
        max_score = df["Final Score"].max()
        min_score = df["Final Score"].min()

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Returned Results", len(results))
        m2.metric("Average Score", round(avg_score, 4))
        m3.metric("Best Score", round(max_score, 4))
        m4.metric("Lowest Score", round(min_score, 4))
    else:
        st.info("Run a search to view analytics.")


with tab_evaluation:
    st.subheader("Benchmark Evaluation Dashboard")

    st.caption(
        "Current benchmark summary is based on Cranfield. SciFact retrieval is now supported and ready for separate evaluation."
    )

    e1, e2, e3, e4, e5 = st.columns(5)

    e1.metric("Queries", EVALUATION_RESULTS["Evaluated Queries"])
    e2.metric("Precision@10", EVALUATION_RESULTS["Mean Precision@10"])
    e3.metric("Recall@10", EVALUATION_RESULTS["Mean Recall@10"])
    e4.metric("MRR", EVALUATION_RESULTS["MRR"])
    e5.metric("MAP", EVALUATION_RESULTS["MAP"])

    evaluation_df = pd.DataFrame([
        {"Metric": "Precision@10", "Score": EVALUATION_RESULTS["Mean Precision@10"]},
        {"Metric": "Recall@10", "Score": EVALUATION_RESULTS["Mean Recall@10"]},
        {"Metric": "MRR", "Score": EVALUATION_RESULTS["MRR"]},
        {"Metric": "MAP", "Score": EVALUATION_RESULTS["MAP"]}
    ])

    st.dataframe(evaluation_df, use_container_width=True)
    st.bar_chart(evaluation_df.set_index("Metric"))

    st.markdown("""
    - **Precision@10**: نسبة الوثائق الصحيحة ضمن أول 10 نتائج.
    - **Recall@10**: نسبة الوثائق الصحيحة التي استطاع النظام استرجاعها.
    - **MRR**: يقيس ترتيب أول وثيقة صحيحة.
    - **MAP**: يقيس جودة الترتيب عبر جميع الاستعلامات.
    """)


with tab_pipeline:
    st.subheader("Pipeline Details")

    if data and data.get("retrieval_mode") == "hybrid_serial":
        st.code("""
User Query
   ↓
Query Refinement
   ↓
Dataset Selection
   ↓
Hybrid Serial Retrieval
   ├── BM25 Candidate Generation
   └── Semantic FAISS Re-ranking
   ↓
Ranked Results
        """)
    else:
        st.code("""
User Query
   ↓
Query Refinement
   ↓
Dataset Selection
   ↓
Hybrid Parallel Retrieval
   ├── BM25 Lexical Search
   └── Semantic FAISS Search
   ↓
Score Fusion
   ↓
Ranked Results
        """)

    if data:
        st.json({
            "original_query": data.get("original_query"),
            "refined_query": data.get("refined_query"),
            "dataset": data.get("dataset"),
            "retrieval_mode": data.get("retrieval_mode"),
            "retrieval_model": data.get("retrieval_model"),
            "configuration": {
                "dataset": dataset,
                "top_k": top_k,
                "bm25_weight": bm25_weight,
                "semantic_weight": semantic_weight,
                "initial_k": initial_k,
                "remove_stopwords": remove_stopwords,
                "use_expansion": use_expansion,
                "use_stemming": use_stemming,
                "search_time_seconds": st.session_state.last_search_time
            }
        })
    else:
        st.info("Run a search to view the latest pipeline configuration.")


with tab_export:
    st.subheader("Export Search Results")

    if results:
        df = results_to_dataframe(results)

        csv_data = df.to_csv(index=False).encode("utf-8")

        json_data = json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        col_csv, col_json = st.columns(2)

        with col_csv:
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="ir_search_results.csv",
                mime="text/csv"
            )

        with col_json:
            st.download_button(
                label="Download Full Response as JSON",
                data=json_data,
                file_name="ir_search_response.json",
                mime="application/json"
            )
    else:
        st.info("Run a search first to export results.")


st.markdown(
    "<div class='footer'>Built with FastAPI · FAISS · BM25 · Sentence Transformers · Streamlit · Multi-Dataset Retrieval</div>",
    unsafe_allow_html=True
)
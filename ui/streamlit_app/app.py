import streamlit as st
import requests
import pandas as pd
import re
import json
import time
from datetime import datetime
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

GATEWAY_URL = "http://127.0.0.1:8006/search/full"

SERVICE_URLS = {
    "Preprocessing Service": "http://127.0.0.1:8001/",
    "Indexing Service": "http://127.0.0.1:8002/",
    "Gateway Service": "http://127.0.0.1:8006/",
    "Retrieval Service": "http://127.0.0.1:8003/",
    "Evaluation Service": "http://127.0.0.1:8004/",
    "Refinement Service": "http://127.0.0.1:8005/"
}

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"


def load_evaluation_results():
    results = {}

    for report_path in REPORTS_DIR.glob("*_evaluation_results.json"):
        dataset_name = report_path.name.replace("_evaluation_results.json", "")

        with open(report_path, "r", encoding="utf-8") as file:
            report_data = json.load(file)

        results[dataset_name] = report_data

    return results


def metrics_table(report_section):
    return pd.DataFrame([
        {
            "Mode": mode,
            "Evaluated Queries": values.get("evaluated_queries", 0),
            "Precision@10": values.get("mean_precision@10", 0),
            "Recall@10": values.get("mean_recall@10", 0),
            "MRR": values.get("mrr", 0),
            "MAP": values.get("map", 0),
            "nDCG@10": values.get("ndcg@10", 0)
        }
        for mode, values in report_section.items()
    ])


def comparison_table(comparison_section):
    return pd.DataFrame([
        {
            "Mode": mode,
            "Δ Precision@10": values.get("precision@10_delta", 0),
            "Δ Recall@10": values.get("recall@10_delta", 0),
            "Δ MRR": values.get("mrr_delta", 0),
            "Δ MAP": values.get("map_delta", 0),
            "Δ nDCG@10": values.get("ndcg@10_delta", 0)
        }
        for mode, values in comparison_section.items()
    ])


EVALUATION_RESULTS = load_evaluation_results()
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


def cluster_results(results):
    texts = [item.get("text", "") for item in results]

    if len(texts) < 2:
        return pd.DataFrame()

    cluster_count = min(3, len(texts))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    matrix = vectorizer.fit_transform(texts)

    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(matrix)

    return pd.DataFrame([
        {
            "Cluster": int(label) + 1,
            "Rank": item.get("rank"),
            "Document": item.get("doc_id"),
            "Score": item.get("score", 0)
        }
        for item, label in zip(results, labels)
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
            "Quora",
        ]
    )

    dataset_mapping = {
        "Quora": "quora",
    }

    dataset = dataset_mapping[dataset_label]

    retrieval_mode_label = st.selectbox(
        "Retrieval Mode",
        [
            "TF-IDF",
            "BM25",
            "Semantic",
            "Hybrid Parallel",
            "Hybrid Serial"
        ]
    )

    retrieval_mode_options = {
        "TF-IDF": "tfidf",
        "BM25": "bm25",
        "Semantic": "semantic",
        "Hybrid Parallel": "hybrid_parallel",
        "Hybrid Serial": "hybrid_serial"
    }

    retrieval_mode = retrieval_mode_options[retrieval_mode_label]

    top_k = st.slider("Number of Results", 1, 20, 5)

    if retrieval_mode in ["bm25", "hybrid_parallel", "hybrid_serial"]:
        bm25_k1 = st.slider("BM25 k1", 0.1, 3.0, 1.5, 0.1, disabled=True)
        bm25_b = st.slider("BM25 b", 0.0, 1.0, 0.75, 0.05, disabled=True)
        st.caption("Quora BM25 index is prebuilt with k1=1.5 and b=0.75 for fast full-dataset retrieval.")
    else:
        bm25_k1 = 1.5
        bm25_b = 0.75

    if retrieval_mode == "hybrid_parallel":
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.4, 0.05)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.6, 0.05)
        initial_k = 50
    elif retrieval_mode == "hybrid_serial":
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
    else:
        bm25_weight = 0.4
        semantic_weight = 0.6
        initial_k = 50

    st.divider()

    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
    use_expansion = st.checkbox("Query Expansion", value=True)
    use_stemming = st.checkbox("Stemming", value=False)
    use_personalization = st.checkbox("Personalization", value=False)

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
    "<div class='subtitle'>Quora IR System: Query Refinement + BM25 + Semantic FAISS Search</div>",
    unsafe_allow_html=True
)

st.divider()

query = st.text_input(
    "Enter your search query",
    placeholder="Example: how can I learn programming?"
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
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "initial_k": initial_k,
            "remove_stopwords": remove_stopwords,
            "use_stemming": use_stemming,
            "use_expansion": use_expansion,
            "use_personalization": use_personalization,
            "user_history": [
                item["query"]
                for item in st.session_state.search_history[-5:]
            ]
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

        mode_col1, mode_col2, mode_col3, mode_col4 = st.columns(4)
        mode_col1.metric("Dataset", data.get("dataset", "N/A"))
        mode_col2.metric("Retrieval Mode", data.get("retrieval_mode", "N/A"))
        mode_col3.metric("Retrieval Model", data.get("retrieval_model", "N/A"))
        mode_col4.metric("Vector Method", data.get("vector_method", "N/A"))

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

        st.subheader("Result Clustering")

        try:
            cluster_df = cluster_results(results)

            if cluster_df.empty:
                st.info("At least two results are needed for clustering.")
            else:
                st.dataframe(cluster_df, use_container_width=True)
                st.bar_chart(cluster_df.groupby("Cluster")["Document"].count())
        except Exception as error:
            st.warning(f"Clustering unavailable for these results: {error}")

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

    if not EVALUATION_RESULTS:
        st.info("No evaluation reports found. Run the evaluation scripts first.")
    else:
        selected_eval_dataset = st.selectbox(
            "Evaluation Dataset",
            list(EVALUATION_RESULTS.keys())
        )

        report = EVALUATION_RESULTS[selected_eval_dataset]

        st.caption(
            "Evaluation is computed using benchmark queries and relevance judgments."
        )

        if "before_features" in report and "after_features" in report:
            st.info(
                f"Showing before/after feature evaluation for: {selected_eval_dataset}"
            )

            before_df = metrics_table(report["before_features"])
            after_df = metrics_table(report["after_features"])
            delta_df = comparison_table(report.get("comparison", {}))

            q1, q2 = st.columns(2)
            q1.metric(
                "Evaluated Queries",
                report.get("evaluated_queries", before_df["Evaluated Queries"].max())
            )
            q2.metric("Top K", report.get("top_k", 10))

            st.subheader("Before Additional Features")
            st.dataframe(before_df, use_container_width=True)
            st.bar_chart(before_df.set_index("Mode")[["Precision@10", "Recall@10", "MAP", "nDCG@10"]])

            st.subheader("After Additional Features")
            st.dataframe(after_df, use_container_width=True)
            st.bar_chart(after_df.set_index("Mode")[["Precision@10", "Recall@10", "MAP", "nDCG@10"]])

            st.subheader("Before vs After Delta")
            st.dataframe(delta_df, use_container_width=True)
            st.bar_chart(delta_df.set_index("Mode"))

        else:
            selected_eval_mode = st.selectbox(
                "Evaluation Retrieval Mode",
                list(report.keys())
            )

            current_eval = metrics_table({
                selected_eval_mode: report[selected_eval_mode]
            }).iloc[0]

            e1, e2, e3 = st.columns(3)
            e4, e5, e6 = st.columns(3)
            st.info(f"Showing results for: {selected_eval_dataset} / {selected_eval_mode}")
            e1.metric("Queries", current_eval["Evaluated Queries"])
            e2.metric("Precision@10", f"{current_eval['Precision@10']:.4f}")
            e3.metric("Recall@10", f"{current_eval['Recall@10']:.4f}")
            e4.metric("MRR", f"{current_eval['MRR']:.4f}")
            e5.metric("MAP", f"{current_eval['MAP']:.4f}")
            e6.metric("nDCG@10", f"{current_eval['nDCG@10']:.4f}")

            comparison_df = metrics_table(report)
            st.subheader(f"{selected_eval_dataset.capitalize()} Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            st.bar_chart(comparison_df.set_index("Mode"))

        st.markdown("""
        - **Precision@10**: نسبة الوثائق الصحيحة ضمن أول 10 نتائج.
        - **Recall@10**: نسبة الوثائق الصحيحة التي استطاع النظام استرجاعها.
        - **MRR**: يقيس ترتيب أول وثيقة صحيحة.
        - **MAP**: يقيس جودة الترتيب عبر جميع الاستعلامات.
        - **nDCG@10**: يقيس جودة ترتيب النتائج مع مراعاة موقع الوثائق الصحيحة.
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
            "vector_method": data.get("vector_method"),
            "configuration": {
                "dataset": dataset,
                "top_k": top_k,
                "bm25_weight": bm25_weight,
                "semantic_weight": semantic_weight,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b,
                "initial_k": initial_k,
                "remove_stopwords": remove_stopwords,
                "use_expansion": use_expansion,
                "use_stemming": use_stemming,
                "use_personalization": use_personalization,
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
    "<div class='footer'>Built with FastAPI · FAISS · BM25 · Streamlit · Quora BEIR Retrieval</div>",
    unsafe_allow_html=True
)


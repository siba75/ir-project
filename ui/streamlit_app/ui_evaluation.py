import pandas as pd
import streamlit as st

from ui_helpers import comparison_table, metrics_table


def render_evaluation_tab(evaluation_results):
    st.subheader("Benchmark Evaluation Dashboard")

    if not evaluation_results:
        st.info("No evaluation reports found. Run the evaluation scripts first.")
        return

    report = evaluation_results.get("quora")

    if not report:
        st.info("Quora evaluation report was not found. Run the evaluation script first.")
        st.stop()

    st.metric("Evaluation Dataset", "Quora")
    st.caption("Fixed benchmark: beir/quora/test")
    st.caption("Evaluation is computed using benchmark queries and relevance judgments.")

    if "before_features" in report and "after_features" in report:
        render_before_after_evaluation(report)
    else:
        render_legacy_evaluation(report)

    st.markdown("""
    - **Precision@10**: نسبة الوثائق الصحيحة ضمن أول 10 نتائج.
    - **Recall@10**: نسبة الوثائق الصحيحة التي استطاع النظام استرجاعها.
    - **MRR**: يقيس ترتيب أول وثيقة صحيحة.
    - **MAP**: يقيس جودة الترتيب عبر جميع الاستعلامات.
    - **nDCG@10**: يقيس جودة ترتيب النتائج مع مراعاة موقع الوثائق الصحيحة.
    """)


def render_before_after_evaluation(report):
    st.info("Showing before/after feature evaluation for Quora")
    before_df = metrics_table(report["before_features"])
    after_df = metrics_table(report["after_features"])
    delta_df = comparison_table(report.get("comparison", {}))

    q1, q2 = st.columns(2)
    q1.metric("Evaluated Queries", report.get("evaluated_queries", before_df["Evaluated Queries"].max()))
    q2.metric("Top K", report.get("top_k", 10))
    q3, q4 = st.columns(2)
    q3.metric("Total qrels Queries", report.get("total_qrels_queries", 0))
    q4.metric("Evaluation Scope", report.get("evaluation_scope", "N/A"))

    render_metric_section("Before Additional Features", before_df)
    render_metric_section("After Additional Features", after_df)
    st.subheader("Before vs After Delta")
    st.dataframe(delta_df, use_container_width=True)
    st.bar_chart(delta_df.set_index("Mode"))
    render_map_ndcg_focus(before_df, after_df)


def render_metric_section(title, df):
    st.subheader(title)
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Mode")[["Precision@10", "Recall@10", "MAP", "nDCG@10"]])


def render_map_ndcg_focus(before_df, after_df):
    st.subheader("MAP and nDCG Focus")
    focus_df = pd.concat(
        [before_df.assign(Phase="Before"), after_df.assign(Phase="After")],
        ignore_index=True,
    )
    st.dataframe(focus_df[["Phase", "Mode", "Evaluated Queries", "MAP", "nDCG@10"]], use_container_width=True)
    st.bar_chart(focus_df.pivot(index="Mode", columns="Phase", values="MAP"))
    st.line_chart(focus_df.pivot(index="Mode", columns="Phase", values="nDCG@10"))


def render_legacy_evaluation(report):
    selected_mode = st.selectbox("Evaluation Retrieval Mode", list(report.keys()))
    current_eval = metrics_table({selected_mode: report[selected_mode]}).iloc[0]
    e1, e2, e3 = st.columns(3)
    e4, e5, e6 = st.columns(3)
    st.info(f"Showing results for: Quora / {selected_mode}")
    e1.metric("Queries", current_eval["Evaluated Queries"])
    e2.metric("Precision@10", f"{current_eval['Precision@10']:.4f}")
    e3.metric("Recall@10", f"{current_eval['Recall@10']:.4f}")
    e4.metric("MRR", f"{current_eval['MRR']:.4f}")
    e5.metric("MAP", f"{current_eval['MAP']:.4f}")
    e6.metric("nDCG@10", f"{current_eval['nDCG@10']:.4f}")
    comparison_df = metrics_table(report)
    st.subheader("Quora Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    st.bar_chart(comparison_df.set_index("Mode"))

import streamlit as st

from ui_analytics import render_analytics_tab
from ui_evaluation import render_evaluation_tab
from ui_pipeline_export import render_export_tab, render_pipeline_tab
from ui_results import render_results_tab
from ui_search import render_header, render_search_form
from ui_sidebar import render_sidebar


def render_tabs(config, evaluation_results, resource_manifest):
    data = st.session_state.last_response
    results = st.session_state.last_results
    tabs = st.tabs(["📄 Results", "📊 Analytics", "📈 Evaluation", "🧠 Pipeline", "⬇️ Export"])

    with tabs[0]:
        render_results_tab(data, results, config)

    with tabs[1]:
        render_analytics_tab(data, results, config)

    with tabs[2]:
        render_evaluation_tab(evaluation_results)

    with tabs[3]:
        render_pipeline_tab(data, config, resource_manifest)

    with tabs[4]:
        render_export_tab(data, results)

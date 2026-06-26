import streamlit as st

from ui_helpers import load_evaluation_results, load_resource_manifest
from ui_sections import render_header, render_search_form, render_sidebar, render_tabs
from ui_theme import apply_theme


def initialize_state():
    defaults = {
        "dark_mode": False,
        "search_history": [],
        "last_results": [],
        "last_response": None,
        "previous_response": None,
        "last_search_time": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="IR Search Engine",
        page_icon="🔎",
        layout="wide",
    )
    initialize_state()
    config = render_sidebar()
    apply_theme()
    render_header()
    render_search_form(config)
    render_tabs(
        config=config,
        evaluation_results=load_evaluation_results(),
        resource_manifest=load_resource_manifest(),
    )
    st.markdown(
        "<div class='footer'>Built with FastAPI · FAISS · BM25 · Streamlit · Quora BEIR Retrieval</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

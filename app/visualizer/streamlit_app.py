"""Minimal Streamlit visualizer for the Eilaf volunteer chatbot (Hebrew, RTL).

Run from the repo root::

    streamlit run app/visualizer/streamlit_app.py
"""

import sys
from pathlib import Path

# Make the repo root importable when launched via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from chatbot.rag.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="צ'אטבוט מתנדבי אילאף", page_icon="🤝")

# Right-to-left layout for Hebrew.
st.markdown(
    "<style>.stApp, .stMarkdown, .stTextInput { direction: rtl; text-align: right; }</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_static_dir()


pipeline = get_pipeline()

st.title("צ'אטבוט מתנדבי אילאף")

if not pipeline.documents:
    st.warning("לא נמצאו מסמכים בתיקיית data/static. הוסיפו קובץ ‎.pdf‏ או ‎.txt‏.")

query = st.text_input("מה תרצו לשאול?")
if query:
    contexts = pipeline.retrieve(query)
    answer = pipeline.generator.generate(query, contexts)

    st.subheader("תשובה")
    st.write(answer or "אין תשובה.")

    st.subheader("מקורות")
    for doc in contexts:
        st.markdown(f"**{doc.source}**")
        st.write(doc.text)

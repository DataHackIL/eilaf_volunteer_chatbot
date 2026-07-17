"""Minimal Streamlit visualizer for the Eilaf volunteer chatbot.

Supports Hebrew, Levantine Arabic and English via a language switch; UI
strings live in ``i18n.py`` (one Enum per language).

Run from the repo root::

    streamlit run app/visualizer/streamlit_app.py
"""

import streamlit as st

from app.localities import load_localities
from app.visualizer.i18n import TEXTS, Language
from chatbot.rag.pipeline import RAGPipeline

# `set_page_config` must be the first Streamlit call, so read the language the
# user previously picked from session state (defaulting to Hebrew) before the
# selector widget is created below.
lang: Language = st.session_state.get("language", Language.HEBREW)
T = TEXTS[lang]

st.set_page_config(page_title=T.TITLE.value, page_icon="🤝")

# Language switch. The widget writes back to `st.session_state["language"]`,
# so the next rerun picks up the choice at the top of the script.
lang = st.sidebar.selectbox(
    "🌐",
    options=list(Language),
    format_func=lambda language: language.native_name,
    key="language",
)
T = TEXTS[lang]

# Match text direction to the chosen language.
direction = "rtl" if lang.is_rtl else "ltr"
align = "right" if lang.is_rtl else "left"
st.markdown(
    f"<style>.stApp, .stMarkdown, .stTextInput {{ direction: {direction}; text-align: {align}; }}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_static_dir()


pipeline = get_pipeline()

st.title(T.TITLE.value)

if not pipeline.documents:
    st.warning(T.NO_DOCUMENTS.value)

query = st.text_input(T.QUERY_PROMPT.value)

# Optional closed-form filters. Blank answers stay out of `facts`, so they
# don't constrain retrieval. Tag *values* on the corpus come in a later
# scraping pass; until then these are wired but inert.
with st.expander(T.FILTER_EXPANDER.value):
    specify_age = st.checkbox(T.SPECIFY_AGE.value)
    age = st.number_input(T.AGE.value, min_value=0, max_value=120, value=30, step=1) if specify_age else None
    gender_options = {
        T.GENDER_UNSPECIFIED.value: None,
        T.GENDER_FEMALE.value: "f",
        T.GENDER_MALE.value: "m",
    }
    gender_label = st.selectbox(T.GENDER.value, list(gender_options))
    # Categorical locality (CBS list). ``None`` is the "unspecified" option;
    # options are labelled in the UI language (English shows the Latin
    # transliteration), while the value stored in `facts` is the Hebrew name.
    locality_choices = [None, *sorted(load_localities(), key=lambda loc: loc.label(lang))]
    locality = st.selectbox(
        T.LOCALITY.value,
        options=locality_choices,
        format_func=lambda loc: T.LOCALITY_UNSPECIFIED.value if loc is None else loc.label(lang),
    )

facts: dict = {}
if age is not None:
    facts["age"] = int(age)
gender_code = gender_options[gender_label]
if gender_code is not None:
    facts["gender"] = gender_code
if locality is not None:
    facts["locality"] = locality.hebrew

if query:
    contexts = pipeline.retrieve(query, facts or None)
    answer = pipeline.generator.generate(query, contexts)

    st.subheader(T.ANSWER.value)
    st.write(answer or T.NO_ANSWER.value)

    st.subheader(T.SOURCES.value)
    for doc in contexts:
        st.markdown(f"**{doc.source}**")
        st.write(doc.text)

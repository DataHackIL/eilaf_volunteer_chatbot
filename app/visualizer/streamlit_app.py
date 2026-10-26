"""Minimal Streamlit visualizer for the Eilaf volunteer chatbot.

Supports Hebrew, Levantine Arabic and English via a language switch; UI
strings live in ``i18n.py`` (one Enum per language).

Run from the repo root::

    streamlit run app/visualizer/streamlit_app.py
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.localities import load_localities
from app.visualizer.i18n import TEXTS, Language
from chatbot.rag.generator import (
    BedrockGenerator,
    ClaudeGenerator,
    ContextEchoGenerator,
    Generator,
    OpenAICompatibleGenerator,
    gemini_rotation,
)
from chatbot.rag.pipeline import RAGPipeline
from chatbot.rag.pipeline.pipeline import DEFAULT_CONTEXT_ANCHOR

# Load GEMINI_API_KEY / ANTHROPIC_API_KEY from the repo-root .env so the LLM
# generators can authenticate. Explicit path (not a bare load_dotenv(), which
# searches up from this file) so it resolves however `streamlit run` is invoked.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# `set_page_config` must be the first Streamlit call, so read the language the
# user previously picked from session state (defaulting to Hebrew) before the
# selector widget is created below.
lang: Language = st.session_state.get("language", Language.HEBREW)
T = TEXTS[lang]

st.set_page_config(page_title=T.TITLE.value)

# Language switch. The widget writes back to `st.session_state["language"]`,
# so the next rerun picks up the choice at the top of the script.
lang = st.sidebar.selectbox(
    T.LANGUAGE_LABEL.value,
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


@st.cache_resource
def get_generators() -> dict[str, Generator]:
    """Selectable answer engines, built once.

    Each LLM head resolves its SDK client and credentials lazily on first
    generation, so constructing them all is cheap and needs nothing up front —
    a key (or, for Bedrock, a region and AWS credentials) is only required once
    the user actually picks that engine and asks.
    """
    return {
        "echo": ContextEchoGenerator(),
        "claude": ClaudeGenerator(),
        # Same model and prompt as "claude", reached through Bedrock on the
        # project's own AWS account: no API key, and no free-tier quota
        # ceiling. Needs EILAF_BEDROCK_REGION; inert until it is set.
        "bedrock": BedrockGenerator(),
        "gemini": gemini_rotation(),  # rotates Gemini models to dodge per-model quotas
        "cerebras": OpenAICompatibleGenerator(),
    }


pipeline = get_pipeline()

st.title(T.TITLE.value)

if not pipeline.documents:
    st.warning(T.NO_DOCUMENTS.value)

query = st.text_input(T.QUERY_PROMPT.value)

# Optional second query line describing the event. Prefilled with the domain
# anchor, which steers retrieval toward violence-victim material; the user can
# refine it or clear it. Blended into the query embedding by `context_weight`.
context = st.text_input(T.CONTEXT_PROMPT.value, value=DEFAULT_CONTEXT_ANCHOR)

# How hard the event description pulls the query vector, 0 (ignore) .. 1 (only
# the description). A live tuning knob: it re-encodes the query per query, so it
# needs no rebuild of the cached pipeline. Defaults to the pipeline's own value.
context_weight = st.sidebar.slider(
    T.CONTEXT_WEIGHT.value,
    min_value=0.0,
    max_value=1.0,
    value=float(pipeline.context_weight),
    step=0.05,
)

# How many nearest peer clauses to append per match for lateral context. Like
# the weight above, it's a per-query override, so no rebuild of the pipeline.
sibling_k = st.sidebar.slider(
    T.SIBLING_K.value,
    min_value=0,
    max_value=5,
    value=int(pipeline.sibling_k),
    step=1,
)

# Which head phrases the answer from the retrieved passages: "echo" returns them
# verbatim (no key needed); the rest generate a grounded Hebrew answer. "claude",
# "gemini" and "cerebras" need the matching key in .env; "bedrock" runs the same
# model as "claude" on the project's AWS account, so it needs no key — only
# EILAF_BEDROCK_REGION. Retrieval is identical across all of them.
gen_labels = {
    "echo": T.GEN_ECHO.value,
    "claude": T.GEN_CLAUDE.value,
    "bedrock": T.GEN_BEDROCK.value,
    "gemini": T.GEN_GEMINI.value,
    "cerebras": T.GEN_CEREBRAS.value,
}
gen_key = st.sidebar.selectbox(
    T.GENERATOR.value, options=list(gen_labels), format_func=lambda k: gen_labels[k]
)
generator = get_generators()[gen_key]

# Optional closed-form filters. Blank answers stay out of `facts`, so they
# don't constrain retrieval. Tag *values* on the corpus come in a later
# scraping pass; until then these are wired but inert.
with st.expander(T.FILTER_EXPANDER.value):
    specify_age = st.checkbox(T.SPECIFY_AGE.value)
    age = (
        st.number_input(T.AGE.value, min_value=0, max_value=120, value=30, step=1)
        if specify_age
        else None
    )
    gender_options = {
        T.GENDER_UNSPECIFIED.value: None,
        T.GENDER_FEMALE.value: "f",
        T.GENDER_MALE.value: "m",
    }
    gender_label = st.selectbox(T.GENDER.value, list(gender_options))
    # Categorical locality (CBS list). ``None`` is the "unspecified" option;
    # options are labelled in the UI language (English shows the Latin
    # transliteration), while the value stored in `facts` is the Hebrew name.
    locality_choices = [
        None,
        *sorted(load_localities(), key=lambda loc: loc.label(lang)),
    ]
    locality = st.selectbox(
        T.LOCALITY.value,
        options=locality_choices,
        format_func=lambda loc: (
            T.LOCALITY_UNSPECIFIED.value if loc is None else loc.label(lang)
        ),
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
    contexts = pipeline.retrieve(
        query,
        facts or None,
        context=context,
        context_weight=context_weight,
        sibling_k=sibling_k,
    )
    # A missing key or an exhausted free-tier quota surfaces here as an SDK
    # error; show it instead of crashing the app (retrieval already succeeded,
    # so the sources below still render).
    try:
        answer = generator.generate(query, contexts)
    except Exception as exc:
        st.error(T.GENERATOR_ERROR.value.format(error=exc))
        answer = ""

    st.subheader(T.ANSWER.value)
    st.write(answer or T.NO_ANSWER.value)

    st.subheader(T.SOURCES.value)
    for doc in contexts:
        # Each passage under its citations, linked where a URL is known.
        for citation in doc.citations:
            text, url = citation.get("text") or "", citation.get("url") or ""
            st.markdown(f"**[{text or url}]({url})**" if url else f"**{text}**")
        st.write(doc.text)

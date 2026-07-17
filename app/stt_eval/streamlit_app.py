"""Streamlit app for evaluating speech-to-text models on spoken prompts.

Record a prompt from the microphone, eyeball its waveform for a sanity check,
and read back the transcription. The language switch selects the backend model:
Hebrew uses ivrit.ai's Whisper large-v3-turbo, Arabic uses Whisper large-v3.

Run from the repo root (``PYTHONPATH=.`` makes the project packages importable
under ``streamlit run``, which otherwise only puts this file's directory on the
path)::

    PYTHONPATH=. streamlit run app/stt_eval/streamlit_app.py
"""

import io

import numpy as np
import streamlit as st

from app.visualizer.i18n import Language
from transcription import (
    IvritHebrewTranscriber,
    Transcriber,
    WhisperArabicTranscriber,
)

SAMPLE_RATE = 16000

# Only the languages we have a transcriber for (English is out of scope for now).
TRANSCRIBERS = {
    Language.HEBREW: IvritHebrewTranscriber,
    Language.ARABIC: WhisperArabicTranscriber,
}

# `set_page_config` must be the first Streamlit call, so read the language the
# user previously picked from session state (defaulting to Hebrew) before the
# selector widget is created below.
lang: Language = st.session_state.get("language", Language.HEBREW)

st.set_page_config(page_title="STT eval", page_icon="🎙️")

# Language switch. The widget writes back to `st.session_state["language"]`,
# so the next rerun picks up the choice at the top of the script.
lang = st.sidebar.selectbox(
    "🌐",
    options=list(TRANSCRIBERS),
    format_func=lambda language: language.native_name,
    key="language",
)

# Match text direction to the chosen language.
direction = "rtl" if lang.is_rtl else "ltr"
align = "right" if lang.is_rtl else "left"
st.markdown(
    f"<style>.stApp, .stMarkdown {{ direction: {direction}; text-align: {align}; }}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_transcriber(language: Language) -> Transcriber:
    """Cold-start (and cache) the transcription model for ``language``."""
    return TRANSCRIBERS[language]()


st.title("🎙️ Speech-to-text evaluation")

audio = st.audio_input("Record a spoken prompt")

if audio is not None:
    # Decode + resample to mono 16 kHz float32 via faster-whisper's own helper
    # (backed by `av`, already pulled in as a faster-whisper dependency).
    from faster_whisper import decode_audio

    waveform = decode_audio(io.BytesIO(audio.getvalue()), sampling_rate=SAMPLE_RATE)

    # Sanity-check the capture: downsample to keep the chart light.
    st.subheader("Waveform")
    step = max(1, len(waveform) // 2000)
    st.line_chart(waveform[::step], height=180)

    with st.spinner(f"Transcribing with {get_transcriber(lang).name}…"):
        text = get_transcriber(lang).transcribe(np.asarray(waveform), SAMPLE_RATE)

    st.subheader("Transcription")
    st.write(text or "_(no speech detected)_")

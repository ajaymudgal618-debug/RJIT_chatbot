import uuid
import requests
import streamlit as st
from bot import GROQ_MODELS

API_URL = "https://rjit-chatbot.onrender.com"

st.set_page_config(page_title="RJIT Assistant", page_icon="🎓")
st.title("🎓 RJIT College Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Groq model", GROQ_MODELS)
    st.caption("Memory is kept even if you switch models.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        try:
            requests.delete(
                f"{API_URL}/chat/{st.session_state.thread_id}",
                timeout=10,
            )
        except requests.exceptions.RequestException:
            # Even if the API call fails (e.g. server briefly down), we still want the UI to reset and start a brand-new thread — a fresh thread_id means the old memory can never be reused.
            pass

        st.session_state.history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

query = st.chat_input("RJIT ke baare mein kuch bhi pucho...")
if query:
    st.chat_message("user").write(query)
    st.session_state.history.append(("user", query))

    with st.spinner("Soch raha hoon..."):
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": query,
                "model_name": model_name,
                "thread_id": st.session_state.thread_id,
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["answer"]

    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("assistant", answer))

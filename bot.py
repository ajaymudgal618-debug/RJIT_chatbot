from dotenv import load_dotenv

load_dotenv()  # reads .env: sets GROQ_API_KEY, HUGGINGFACEHUB_API_TOKEN, etc.

import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from chunk_simple import build_retriever

# Resolved relative to this file's own folder (not the current working
# directory) so it works the same whether run locally on Windows or on
# Render's Linux containers. The markdown file must live in the repo,
# alongside bot.py, and be committed to Git (not .gitignore'd) — Render
# only has access to what's actually pushed to GitHub.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "new_clean_rjit.md")

# Groq models currently active (llama-3.3-70b-versatile and llama-3.1-8b-instant
# were deprecated by Groq on 17-Jun-2026 — these are their recommended replacements).
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "gemma2-9b-it",
]

_retriever = None


def get_retriever():
    """Built once per process (embeddings + FAISS index are expensive),
    then reused across every chat turn and every model switch.

    Plain module-level cache so this also works outside Streamlit
    (e.g. from the FastAPI process), where st.cache_resource isn't available.
    """
    global _retriever
    if _retriever is None:
        _retriever = build_retriever(FILE_PATH)
    return _retriever


@tool
def rjit_tool(query: str) -> str:
    """Search RJIT (Rustamji Institute of Technology) website content and
    return the most relevant context. Covers courses, fees, scholarships,
    faculty, departments, hostel, library, placements, labs, and more."""
    docs = get_retriever().invoke(query)
    return "\n\n".join(d.page_content for d in docs)


SYSTEM_PROMPT = """You are RJIT Assistant, a helpful chatbot for Rustamji Institute of \
Technology (RJIT).

Rules you must always follow:
- Always use the rjit_tool to search for relevant information before answering \
any question about RJIT (courses, fees, faculty, departments, hostel, library, \
placements, admissions, etc.).
- Answer ONLY using the context returned by rjit_tool. Do not use your own \
general knowledge, training data, or assumptions about RJIT or any college.
- If the tool's context does not contain the answer, say clearly that this \
information is not available on the RJIT website, instead of guessing or \
making something up.
- If the question is unrelated to RJIT (general chit-chat, unrelated topics), \
politely say you can only answer questions about RJIT.
"""


def build_agent(model_name, checkpointer):
    """Builds a fresh agent for the chosen Groq model, sharing the same
    checkpointer so conversation memory survives a model switch."""
    model_id = f"groq:{model_name}"
    # Summarization is a background bookkeeping step, not something the user
    # is waiting on for quality — so it always runs on a small/fast model,
    # regardless of which (possibly huge) model the user picked for answers.
    summarizer_model_id = "groq:gemma2-9b-it"
    return create_agent(
        model=model_id,
        tools=[rjit_tool],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=summarizer_model_id,
                trigger=("tokens", 3000),
                keep=("messages", 5),
            ),
        ],
        checkpointer=checkpointer,
    )

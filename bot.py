from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from chunk_simple import build_retriever

load_dotenv()  # reads .env in the current working directory, sets GROQ_API_KEY etc.
FILE_PATH = r"C:\Users\ajaym\OneDrive\Desktop\project_langchain\webscrapping\new_clean_rjit.md"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "gemma2-9b-it"]

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever(FILE_PATH)
    return _retriever

@tool
def rjit_tool(query: str) -> str:
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
    model_id = f"groq:{model_name}"
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
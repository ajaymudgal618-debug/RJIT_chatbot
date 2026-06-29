from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from bot import build_agent, GROQ_MODELS

checkpointer = InMemorySaver()

_agent_cache: dict[str, object] = {}


def get_agent(model_name: str):
    if model_name not in GROQ_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose one of: {GROQ_MODELS}",
        )
    if model_name not in _agent_cache:
        _agent_cache[model_name] = build_agent(model_name, checkpointer)
    return _agent_cache[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the retriever (embeddings + FAISS index) once at startup instead
    # of on the first request, so the first user isn't stuck waiting.
    from bot import get_retriever
    get_retriever()
    yield


app = FastAPI(title="RJIT Assistant API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    model_name: str = GROQ_MODELS[0]
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    model_name: str
    thread_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": GROQ_MODELS}


@app.delete("/chat/{thread_id}")
def clear_chat(thread_id: str):
    """Wipes all stored memory (checkpoints) for this thread.
    Call this when the user hits 'Clear chat' so the agent forgets
    everything from this conversation, not just the visible UI history."""
    checkpointer.delete_thread(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    agent = get_agent(req.model_name)
    config = {"configurable": {"thread_id": req.thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )
    answer = result["messages"][-1].content

    return ChatResponse(
        answer=answer,
        model_name=req.model_name,
        thread_id=req.thread_id,
    )
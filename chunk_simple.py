import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def get_chunks(file_path):
    """Reads the scraped markdown file and returns one chunk per === URL === section.
    Each chunk = {"url": ..., "text": ...}. Empty / failed sections are skipped.
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n")

    sections = re.split(r"\n\n=== (.+?) ===\n\n", content)[1:]  # drop empty preamble
    urls = sections[0::2]
    bodies = sections[1::2]

    chunks = []
    for url, body in zip(urls, bodies):
        text = body.strip()
        if not text or text.startswith("[SKIPPED"):
            continue
        chunks.append({"url": url, "text": text})

    return chunks


def split_chunks(chunks, chunk_size=1250, chunk_overlap=100):
    """Splits each URL-level chunk further using RecursiveCharacterTextSplitter.
    Custom separators make it prefer breaking at markdown heading boundaries
    (##### down to #) before falling back to paragraphs/lines/words — this
    stops it from cutting a chunk in the middle of one person's info and
    bleeding into the next heading's content.
    Sections shorter than chunk_size pass through unchanged.
    """
    separators = [
        "\n##### ", "\n#### ", "\n### ", "\n## ", "\n# ",
        "\n", " ", "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator="start",  # heading stays attached to the chunk it starts
    )
    result = []
    for c in chunks:
        for piece in splitter.split_text(c["text"]):
            result.append({"url": c["url"], "text": piece})
    return result


def build_retriever(file_path, chunk_size=1250, chunk_overlap=100, k=3,
                     bm25_weight=0.5, vector_weight=0.5):
    """Runs the full pipeline (read -> URL-level chunks -> sub-chunks ->
    embeddings -> FAISS + BM25) and returns a ready-to-use hybrid retriever.

    This is the function bot.py imports and wraps in @st.cache_resource,
    so it is built exactly once per process.
    """
    chunks = get_chunks(file_path)
    print(f"URL-level chunks: {len(chunks)}")

    chunks = split_chunks(chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Final chunks: {len(chunks)}")

    # 1. HuggingFace Embeddings via Inference API (no local model load —
    # keeps RAM usage low enough for Render's free 512MB tier). Needs
    # HUGGINGFACEHUB_API_TOKEN set in the environment (free HF account).
    embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    # Convert to LangChain Documents (url kept as metadata, not lost)
    docs = [
        Document(page_content=c["text"], metadata={"url": c["url"]})
        for c in chunks
    ]

    # Vector retriever (FAISS)
    db = FAISS.from_documents(docs, embeddings)
    vector_retriever = db.as_retriever(search_kwargs={"k": k})

    # Keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # Hybrid retriever = keyword (BM25) + semantic (FAISS) combined
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
    return hybrid_retriever

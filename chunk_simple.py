import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    """Splits each URL-level chunk further, preferring markdown heading
    boundaries (##### down to #) before paragraphs/lines/words.
    """
    separators = [
        "\n##### ", "\n#### ", "\n### ", "\n## ", "\n# ",
        "\n", " ", "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator="start",
    )
    result = []
    for c in chunks:
        for piece in splitter.split_text(c["text"]):
            result.append({"url": c["url"], "text": piece})
    return result


def build_retriever(file_path, chunk_size=1250, chunk_overlap=100, k=3,
                    bm25_weight=0.5, vector_weight=0.5):
    """read -> chunks -> sub-chunks -> BM25 (+ FAISS with Gemini embeddings
    if available) -> returns a ready-to-use retriever.
    """
    chunks = get_chunks(file_path)
    print(f"URL-level chunks: {len(chunks)}")

    chunks = split_chunks(chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Final chunks: {len(chunks)}")

    docs = [
        Document(page_content=c["text"], metadata={"url": c["url"]})
        for c in chunks
    ]

    # Keyword retriever (BM25) - no API, no model, always works
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # Semantic retriever (FAISS + Gemini embeddings via API, low RAM).
    # If key is missing or the API fails, fall back to BM25 only.
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            db = FAISS.from_documents(docs, embeddings)
            vector_retriever = db.as_retriever(search_kwargs={"k": k})

            print("Using hybrid retriever (BM25 + FAISS/Gemini)")
            return EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[bm25_weight, vector_weight],
            )
        except Exception as e:
            print(f"Embeddings failed, falling back to BM25 only: {e}")
    else:
        print("GOOGLE_API_KEY not set, using BM25 only")

    return bm25_retriever

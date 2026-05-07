"""
vector_store.py
---------------
Manages the ChromaDB vector store: building it from documents and querying it.

Supports two embedding backends:
  - Google Generative AI (gemini-embedding-001) — requires GOOGLE_API_KEY
  - Ollama local embeddings (nomic-embed-text, etc.) — fully offline
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)


def _get_embeddings(model_name: str) -> GoogleGenerativeAIEmbeddings:
    # Reads GOOGLE_API_KEY from the environment (set via main.py from GEMINI_API_KEY)
    return GoogleGenerativeAIEmbeddings(model=model_name)


def _get_local_embeddings(base_url: str, model: str):
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(base_url=base_url, model=model)


def build_vector_store(
    documents: list[Document],
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    ollama_embed_cfg: dict | None = None,
) -> Chroma:
    """
    Embed `documents` and persist them to ChromaDB at `persist_dir`.
    If the collection already exists it will be overwritten.

    Pass `ollama_embed_cfg` (keys: base_url, model) to use local Ollama embeddings
    instead of the Google Generative AI backend.
    """
    os.makedirs(persist_dir, exist_ok=True)
    if ollama_embed_cfg:
        embeddings = _get_local_embeddings(
            base_url=ollama_embed_cfg["embed_url"],
            model=ollama_embed_cfg["embedding_model"],
        )
        logger.info(f"Building vector store with Ollama embeddings ({ollama_embed_cfg['embedding_model']})")
    else:
        embeddings = _get_embeddings(embedding_model)
        logger.info(f"Building vector store with Gemini embeddings ({embedding_model})")

    logger.info(f"Embedding {len(documents)} chunks → {persist_dir}")
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    logger.info("Vector store built and persisted.")
    return store


def load_vector_store(
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    ollama_embed_cfg: dict | None = None,
) -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Pass `ollama_embed_cfg` (keys: embed_url, embedding_model) to use local Ollama
    embeddings — must match the model used when the store was built.
    """
    if ollama_embed_cfg:
        embeddings = _get_local_embeddings(
            base_url=ollama_embed_cfg["embed_url"],
            model=ollama_embed_cfg["embedding_model"],
        )
    else:
        embeddings = _get_embeddings(embedding_model)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    logger.info(f"Vector store loaded from {persist_dir} (collection: {collection_name})")
    return store


def retrieve(store: Chroma, query: str, top_k: int = 5) -> list[Document]:
    """Return the top-k most relevant documents for `query`."""
    return store.similarity_search(query, k=top_k)

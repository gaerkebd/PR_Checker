"""
vector_store.py
---------------
Manages the ChromaDB vector store: building it from documents and querying it.

Supports two embedding backends:
  - Google Generative AI (gemini-embedding-001) — requires GOOGLE_API_KEY
  - Ollama local embeddings (nomic-embed-text, etc.) — fully offline

Retrieval supports both plain similarity search and Maximal Marginal Relevance
(MMR), which trades a small amount of relevance for diversity across results.
With 500 BigVul records in the store, MMR prevents five nearly-identical CWE
examples from dominating every retrieval call.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)


def _get_local_embeddings(base_url: str, model: str):
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(base_url=base_url, model=model)


def build_vector_store(
    documents: list[Document],
    persist_dir: str,
    collection_name: str,
    ollama_cfg: dict,
) -> Chroma:
    """
    Embed `documents` and persist them to ChromaDB at `persist_dir`.
    If the collection already exists it will be overwritten.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = _get_local_embeddings(
        base_url=ollama_cfg["embed_url"],
        model=ollama_cfg["embedding_model"],
    )
    logger.info(f"Building vector store with Ollama embeddings ({ollama_cfg['embedding_model']})")

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
    ollama_cfg: dict,
) -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    embeddings = _get_local_embeddings(
        base_url=ollama_cfg["embed_url"],
        model=ollama_cfg["embedding_model"],
    )

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    logger.info(f"Vector store loaded from {persist_dir} (collection: {collection_name})")
    return store


def retrieve(
    store: Chroma,
    query: str,
    top_k: int = 8,
    use_mmr: bool = True,
    mmr_lambda: float = 0.6,
) -> list[Document]:
    """
    Return the top-k most relevant documents for `query`.

    use_mmr=True (default): Maximal Marginal Relevance — fetches top_k * 4
    candidates then selects top_k that balance relevance and diversity.
    mmr_lambda: 1.0 = pure relevance, 0.0 = pure diversity. 0.6 is a good
    default that prevents redundant BigVul examples from flooding the context.

    use_mmr=False: plain cosine similarity search.
    """
    if use_mmr:
        return store.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=top_k * 4,
            lambda_mult=mmr_lambda,
        )
    return store.similarity_search(query, k=top_k)

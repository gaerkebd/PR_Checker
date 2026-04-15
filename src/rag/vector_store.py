"""
vector_store.py
---------------
Manages the ChromaDB vector store: building it from documents and querying it.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)


def _get_embeddings(model_name: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=model_name)


def build_vector_store(
    documents: list[Document],
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
) -> Chroma:
    """
    Embed `documents` and persist them to ChromaDB at `persist_dir`.
    If the collection already exists it will be overwritten.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = _get_embeddings(embedding_model)

    logger.info(f"Building vector store with {len(documents)} chunks → {persist_dir}")
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
) -> Chroma:
    """Load an existing ChromaDB collection from disk."""
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

# =============================================================================
# embed_index.py — Embedding Generation & Vector Store Management
# =============================================================================
# WHAT THIS FILE DOES:
#   1. EMBED   — Convert every text chunk into a numeric vector (embedding)
#   2. PERSIST — Save those vectors + metadata into ChromaDB on disk
#   3. LOAD    — Reload the saved vector store when the app restarts
#
# WHY EMBEDDINGS?
#   Computers can't compare meaning between two sentences directly.
#   An embedding model converts text into a list of numbers (a "vector")
#   that encodes semantic meaning. Two sentences about similar topics will
#   have vectors that point in the same direction in vector space.
#   This lets us do "semantic search" — find chunks by MEANING, not keywords.
#
# ANALOGY:
#   If "dog" → [0.9, 0.1, 0.3] and "puppy" → [0.88, 0.12, 0.31],
#   their vectors are close. "airplane" → [0.1, 0.9, 0.2] is far away.
#   Cosine similarity measures the angle between vectors (0=opposite, 1=same).
# =============================================================================

from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    CHROMA_COLLECTION_NAME,
)
from src.utils import get_logger, timer, ensure_dir

logger = get_logger(__name__)


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-transformer model.

    The first time this runs it downloads the model weights (~80MB for
    all-MiniLM-L6-v2) and caches them locally. Subsequent runs are instant.

    The model converts a string → a list of floats (a vector).
    Example:
        model.embed_query("What is machine learning?")
        # → [0.023, -0.14, 0.87, ... ]  (384 numbers for MiniLM)

    Returns:
        HuggingFaceEmbeddings instance ready to call .embed_documents()
        or .embed_query().
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    # model_kwargs controls the device: "cpu" works everywhere;
    # change to "cuda" if you have a GPU for faster embedding.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        # normalize_embeddings=True ensures cosine similarity works correctly
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded ✓")
    return embeddings


# =============================================================================
# VECTOR STORE — BUILDING & PERSISTING
# =============================================================================

@timer
def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed all chunks and store them in a persistent ChromaDB database.

    HOW IT WORKS:
    1. We pass each chunk's text through the embedding model → vector
    2. ChromaDB stores (vector, original text, metadata) as one record
    3. The database is saved to disk so we don't re-embed on every restart

    IMPORTANT: Run this function ONCE after downloading your dataset.
    The resulting database lives in vector_store/ and is loaded (not rebuilt)
    on every subsequent app start.

    Args:
        chunks: List of LangChain Document objects from clean_chunk.py

    Returns:
        A Chroma vector store instance.
    """
    if not chunks:
        raise ValueError("No chunks to embed. Run the ingestion pipeline first.")

    db_path = ensure_dir(VECTOR_STORE_DIR)
    logger.info(f"Building vector store with {len(chunks)} chunks...")
    logger.info(f"Persisting to: {db_path}")

    embeddings = load_embedding_model()

    # Chroma.from_documents() does three things at once:
    #   1. Calls embeddings.embed_documents([chunk.page_content for chunk in chunks])
    #   2. Stores each (text, vector, metadata) tuple in the database
    #   3. Saves everything to persist_directory on disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": "cosine"}
    )

    count = vector_store._collection.count()
    logger.info(f"Vector store built and persisted ✓ ({count} vectors stored)")
    return vector_store


# =============================================================================
# VECTOR STORE — LOADING FROM DISK
# =============================================================================

def load_vector_store() -> Optional[Chroma]:
    """
    Load an already-built ChromaDB vector store from disk.

    Call this at app startup (instead of rebuilding from scratch).
    Returns None if the vector store doesn't exist yet — the caller
    should then prompt the user to run the ingestion pipeline first.

    Returns:
        Chroma instance if the database exists, otherwise None.
    """
    db_path = VECTOR_STORE_DIR

    if not db_path.exists() or not any(db_path.iterdir()):
        logger.warning(
            "Vector store not found. "
            "Run: python run.py --ingest  to build it first."
        )
        return None

    logger.info(f"Loading existing vector store from {db_path} ...")
    embeddings = load_embedding_model()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": "cosine"}
    )

    count = vector_store._collection.count()
    logger.info(f"Vector store loaded ✓ ({count} vectors)")
    return vector_store


# =============================================================================
# UTILITY — INSPECT THE STORE
# =============================================================================

def get_store_stats(vector_store: Chroma) -> dict:
    """
    Return basic statistics about the loaded vector store.
    Useful for the UI's status panel.

    Args:
        vector_store: A loaded Chroma instance.

    Returns:
        Dict with 'total_chunks', 'collection_name', 'model_name'.
    """
    return {
        "total_chunks": vector_store._collection.count(),
        "collection_name": CHROMA_COLLECTION_NAME,
        "model_name": EMBEDDING_MODEL_NAME,
    }

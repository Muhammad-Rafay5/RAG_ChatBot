# =============================================================================
# retrieve.py — Semantic Retrieval
# =============================================================================
# WHAT THIS FILE DOES:
#   Given a user's question, find the most relevant text chunks from the
#   vector store and return them with their source metadata.
#
# HOW SEMANTIC RETRIEVAL WORKS:
#   1. Embed the user's query using the SAME model used to embed the documents.
#      This puts the query and the documents into the same vector space.
#   2. Compute cosine similarity between the query vector and every stored vector.
#      Cosine similarity = cos(angle between two vectors).
#        → 1.0 means identical direction (very similar meaning)
#        → 0.0 means perpendicular (unrelated)
#        → -1.0 means opposite (rare with normalized embeddings)
#   3. Return the TOP_K chunks with the highest similarity scores.
#   4. Filter out chunks below SIMILARITY_THRESHOLD (low confidence).
#
# WHY NOT KEYWORD SEARCH?
#   "What is a neural network?" and "how does deep learning work?" share
#   very few keywords but are semantically similar. Embedding-based search
#   catches this; BM25/keyword search would miss it.
# =============================================================================

from dataclasses import dataclass
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from src.utils import get_logger, truncate

logger = get_logger(__name__)


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class RetrievedChunk:
    """
    One search result: the text chunk + its relevance score + source metadata.

    This object is what gets passed to the prompt builder in generate.py.
    The UI also uses the metadata to display verified source references.

    Attributes:
        text:       The raw chunk text that will be injected into the prompt.
        score:      Cosine similarity score (0.0 – 1.0). Higher = more relevant.
        source:     File path or URL the chunk came from.
        chunk_index: Position of this chunk within its source document.
        metadata:   Full metadata dict from ChromaDB.
    """
    text:        str
    score:       float
    source:      str
    chunk_index: int
    metadata:    dict


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve(query: str, vector_store: Chroma) -> List[RetrievedChunk]:
    """
    Semantic search: embed the query and return the top relevant chunks.

    Steps:
    1. ChromaDB embeds the query using the stored embedding function.
    2. It performs an approximate nearest-neighbour (ANN) search over all
       stored vectors using HNSW indexing (fast even with millions of vectors).
    3. Returns the top TOP_K results with their distance scores.
    4. We convert distances to similarities and apply the threshold filter.

    Args:
        query:        The user's natural language question.
        vector_store: A loaded Chroma instance (from embed_index.py).

    Returns:
        List of RetrievedChunk objects, sorted by relevance (best first).
        May be empty if no chunk clears the similarity threshold.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    logger.info(f"Retrieving top-{TOP_K_RESULTS} chunks for: '{truncate(query, 80)}'")

    # similarity_search_with_relevance_scores returns:
    #   List of (Document, score) tuples
    # where score is a value in [0, 1] (higher = more similar).
    results: List[tuple[Document, float]] = (
        vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=TOP_K_RESULTS,
        )
    )

    # Convert raw results into RetrievedChunk objects and apply the threshold
    chunks: List[RetrievedChunk] = []
    for doc, score in results:
        if score < SIMILARITY_THRESHOLD:
            logger.debug(
                f"Skipping chunk (score {score:.3f} < threshold {SIMILARITY_THRESHOLD}): "
                f"{truncate(doc.page_content, 60)}"
            )
            continue

        chunk = RetrievedChunk(
            text=doc.page_content,
            score=round(score, 4),
            source=doc.metadata.get("source", "unknown"),
            chunk_index=doc.metadata.get("chunk_index", -1),
            metadata=doc.metadata,
        )
        chunks.append(chunk)

    if not chunks:
        logger.warning(
            "No chunks passed the similarity threshold. "
            "Try rephrasing the question or lowering SIMILARITY_THRESHOLD in config.py."
        )
    else:
        logger.info(
            f"Retrieved {len(chunks)} chunk(s). "
            f"Top score: {chunks[0].score:.4f} | "
            f"Source: {chunks[0].source}"
        )

    return chunks


# =============================================================================
# FORMAT FOR PROMPT INJECTION
# =============================================================================

def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Combine retrieved chunks into a single formatted context string
    that will be injected into the LLM prompt.

    Each chunk is numbered and its source is labelled so the model can
    reference or differentiate between multiple sources.

    Args:
        chunks: List of RetrievedChunk objects from retrieve().

    Returns:
        A formatted multi-chunk context string.

    Example output:
        [Source 1 | score: 0.87 | file: data/wiki.txt | chunk #3]
        The mitochondria is the powerhouse of the cell...

        [Source 2 | score: 0.81 | file: data/wiki.txt | chunk #4]
        ATP synthesis occurs across the inner membrane...
    """
    if not chunks:
        return "No relevant context was found in the knowledge base."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i} | score: {chunk.score:.4f} | "
            f"file: {chunk.source} | chunk #{chunk.chunk_index}]"
        )
        parts.append(f"{header}\n{chunk.text.strip()}")

    return "\n\n".join(parts)

# =============================================================================
# clean_chunk.py — Data Ingestion, Cleaning, and Chunking
# =============================================================================
# WHAT THIS FILE DOES (in order):
#   1. LOAD   — Read raw source files (TXT, PDF, HTML) from the data/ folder
#   2. CLEAN  — Strip junk: HTML tags, bad characters, boilerplate whitespace
#   3. CHUNK  — Split cleaned text into overlapping windows of ~500 tokens
#
# WHY CHUNKING MATTERS:
#   LLMs have a limited context window. You can't feed an entire book into a
#   prompt. Chunking breaks text into bite-sized pieces so you can retrieve
#   only the RELEVANT pieces at query time.
#
# WHY OVERLAP MATTERS:
#   If a sentence spans a chunk boundary, neither chunk alone makes sense.
#   A 10% overlap repeats the tail of the previous chunk at the start of the
#   next one, so no answer ever gets cut off.
# =============================================================================

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import get_logger, timer, collect_source_files

logger = get_logger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SourceDocument:
    """
    A container that holds the cleaned text of one source file together
    with metadata about where it came from.

    Metadata is preserved all the way into the vector store and shown
    in the UI so users can verify which source answered their question.
    """
    content: str                         # cleaned full text of the file
    source:  str                         # file path string (for attribution)
    metadata: dict = field(default_factory=dict)  # any extra fields


# =============================================================================
# STEP 1 — LOADING
# =============================================================================

def load_file(file_path: Path) -> SourceDocument:
    """
    Load a single file into a SourceDocument.

    Dispatches to the right LangChain loader based on the file extension:
    - .txt  → TextLoader
    - .pdf  → PyPDFLoader   (requires: pip install pypdf)
    - .html → UnstructuredHTMLLoader  (requires: pip install unstructured)

    Args:
        file_path: Path object pointing at the file.

    Returns:
        SourceDocument with the raw text content and source metadata.

    Raises:
        ValueError:  If the file extension is not supported.
        RuntimeError: If the loader fails.
    """
    ext = file_path.suffix.lower()
    logger.info(f"Loading [{ext}] → {file_path.name}")

    try:
        if ext == ".txt":
            # TextLoader reads the file as plain text.
            # encoding="utf-8" + errors="ignore" silently skips bad bytes
            # instead of crashing on special characters.
            loader = TextLoader(str(file_path), encoding="utf-8",
                                autodetect_encoding=True)

        elif ext == ".pdf":
            # PyPDFLoader extracts text page-by-page from PDF files.
            loader = PyPDFLoader(str(file_path))

        elif ext in (".html", ".htm"):
            # UnstructuredHTMLLoader strips tags and extracts visible text.
            loader = UnstructuredHTMLLoader(str(file_path))

        else:
            raise ValueError(
                f"Unsupported file type '{ext}' for file: {file_path.name}\n"
                "Supported: .txt, .pdf, .html"
            )

        # Each loader returns a list of LangChain Document objects.
        # We join multiple pages/sections into one continuous string.
        docs = loader.load()
        combined_text = "\n\n".join(doc.page_content for doc in docs)

        return SourceDocument(
            content=combined_text,
            source=str(file_path),
            metadata={"filename": file_path.name, "extension": ext},
        )

    except Exception as exc:
        raise RuntimeError(f"Failed to load {file_path}: {exc}") from exc


@timer
def load_all_files(data_dir: Path = DATA_DIR) -> List[SourceDocument]:
    """
    Load every supported file from data_dir.

    Args:
        data_dir: Directory containing source files.

    Returns:
        List of SourceDocument objects, one per file.
    """
    supported = [".txt", ".pdf", ".html", ".htm"]
    files = collect_source_files(data_dir, supported)
    logger.info(f"Found {len(files)} source file(s) in {data_dir}")

    documents = []
    for f in files:
        try:
            documents.append(load_file(f))
        except Exception as exc:
            # Log the error but continue loading other files.
            logger.warning(f"Skipping {f.name}: {exc}")

    logger.info(f"Successfully loaded {len(documents)} document(s)")
    return documents


# =============================================================================
# STEP 2 — CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Sanitize raw extracted text before chunking.

    Operations performed (in order):
    1. Strip HTML/XML tags         — removes leftover <p>, <div>, etc.
    2. Remove null bytes           — some PDFs embed \x00 characters
    3. Normalize unicode dashes    — replace — and – with a plain hyphen
    4. Remove control characters   — tabs become spaces; other ctrl chars removed
    5. Collapse whitespace         — multiple spaces/newlines → single space
    6. Strip leading/trailing space

    Args:
        text: Raw text string.

    Returns:
        Cleaned string.
    """
    # 1. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Remove null bytes
    text = text.replace("\x00", "")

    # 3. Normalize unicode dashes to ASCII hyphen
    text = text.replace("\u2014", "-").replace("\u2013", "-")

    # 4. Replace tabs with spaces; strip other non-printable control characters
    text = text.replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Collapse runs of whitespace (keep single newlines for readability)
    text = re.sub(r" {2,}", " ", text)          # multiple spaces → one space
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ newlines → double newline

    # 6. Strip edges
    return text.strip()


def clean_document(doc: SourceDocument) -> SourceDocument:
    """
    Apply clean_text() to a SourceDocument's content field.
    Returns a new SourceDocument (original is not mutated).
    """
    cleaned = clean_text(doc.content)
    logger.debug(f"Cleaned '{doc.metadata.get('filename')}' "
                 f"({len(doc.content)} → {len(cleaned)} chars)")
    return SourceDocument(content=cleaned, source=doc.source, metadata=doc.metadata)


# =============================================================================
# STEP 3 — CHUNKING
# =============================================================================

def chunk_document(doc: SourceDocument) -> List[Document]:
    """
    Split one SourceDocument into smaller, overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split
    on paragraph breaks first (\\n\\n), then single newlines, then sentences,
    then words — this hierarchy preserves natural language boundaries whenever
    possible.

    Sliding window recap:
        |--------- chunk 1 ---------|
                         |----overlap----|--------- chunk 2 ---------|
                                                    |----overlap----|--- chunk 3 ...

    Each returned Document carries metadata so we can trace any chunk back
    to its source file.

    Args:
        doc: A cleaned SourceDocument.

    Returns:
        List of LangChain Document objects, each representing one chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,      # max characters per chunk (≈ 500 tokens)
        chunk_overlap=CHUNK_OVERLAP,  # characters repeated between adjacent chunks
        # Try to split at these separators, in priority order:
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,        # measure length in characters
    )

    # Create a LangChain Document so the splitter can attach metadata
    base_doc = Document(page_content=doc.content, metadata=doc.metadata)
    chunks = splitter.split_documents([base_doc])

    # Add the source path and chunk index to each chunk's metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = doc.source
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    logger.info(
        f"'{doc.metadata.get('filename')}' → {len(chunks)} chunks "
        f"(~{CHUNK_SIZE} chars each, {CHUNK_OVERLAP} overlap)"
    )
    return chunks


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

@timer
def run_ingestion_pipeline(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Full ingestion pipeline: Load → Clean → Chunk.

    This is the function called by embed_index.py and run.py.

    Args:
        data_dir: Path to the folder containing source files.

    Returns:
        Flat list of all Document chunks from all source files,
        ready to be embedded.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1/3 — Loading source files")
    raw_docs = load_all_files(data_dir)

    logger.info("STAGE 2/3 — Cleaning text")
    cleaned_docs = [clean_document(d) for d in raw_docs]

    logger.info("STAGE 3/3 — Chunking")
    all_chunks: List[Document] = []
    for doc in cleaned_docs:
        all_chunks.extend(chunk_document(doc))

    logger.info(f"Ingestion complete: {len(all_chunks)} total chunks from "
                f"{len(cleaned_docs)} document(s)")
    logger.info("=" * 60)
    return all_chunks

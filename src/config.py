# =============================================================================
# config.py — Centralized Configuration
# =============================================================================
# This is the SINGLE place where every setting lives.
# Every other file imports from here — no magic strings scattered around.
# To change the model, chunk size, or paths, you only touch this one file.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# The root of the project (one level up from src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Where raw downloaded/copied source files live (txt, pdf, html, etc.)
DATA_DIR = ROOT_DIR / "data"

# Where ChromaDB will persist the vector index on disk
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
# These control how your raw text is split into smaller pieces.
# The assignment specifies a sliding window strategy.

# Maximum number of characters per chunk.
# ~500 tokens ≈ ~2000 characters (1 token ≈ 4 chars for English text).
CHUNK_SIZE = 2000

# How much of the previous chunk to repeat at the start of the next one.
# 10% of 2000 = 200 characters of overlap.
# This prevents answers from being cut off at a chunk boundary.
CHUNK_OVERLAP = 200

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
# We use a free, open-source model from HuggingFace.
# all-MiniLM-L6-v2  → fast, small (80MB), 384-dim vectors — great for learning
# BAAI/bge-large-en → slower, larger (1.3GB), 1024-dim vectors — higher quality
#
# Change EMBEDDING_MODEL_NAME below to switch between them.

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"   # default: fast & lightweight
)

# =============================================================================
# VECTOR STORE (ChromaDB) CONFIGURATION
# =============================================================================
# A "collection" in ChromaDB is like a table in a database.
# All your document chunks go into this one collection.

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_knowledge_base")

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
# When the user asks a question, we search the vector store and pull back
# the TOP_K most relevant chunks to inject into the prompt.
# The assignment specifies 3 to 5. We default to 4.

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Minimum similarity score (0.0 – 1.0) for a chunk to be considered relevant.
# Chunks scoring below this are discarded even if they are in the top-k.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))

# =============================================================================
# LLM (LANGUAGE MODEL) CONFIGURATION
# =============================================================================
# We use Ollama to run a model locally — no API costs, no internet needed.
# Make sure Ollama is installed and the model is pulled before running.
#
#   Install Ollama:  https://ollama.com
#   Pull a model:    ollama pull llama3.2
#
# Supported model strings: "llama3.2", "mistral", "phi3", "gemma2", etc.

LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "ollama")       # "ollama" or "openai"
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")      # model tag for Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# If you switch to OpenAI (set LLM_PROVIDER="openai" in .env):
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# If you switch to Groq (set LLM_PROVIDER="groq" in .env):
# Get your free API key at: https://console.groq.com
# Fast inference — Llama 3, Mixtral, and Gemma models available for free.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")
# Other good Groq model options:
#   "llama3-70b-8192"      → larger, smarter, still free
#   "mixtral-8x7b-32768"   → 32k context window, great for long documents
#   "gemma2-9b-it"         → Google's Gemma 2

# Controls randomness. 0.0 = deterministic, 1.0 = creative.
# For a factual RAG chatbot, keep this low.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_TITLE       = "RAG Knowledge-Base Chatbot"
APP_DESCRIPTION = "Ask questions grounded in your indexed documents."

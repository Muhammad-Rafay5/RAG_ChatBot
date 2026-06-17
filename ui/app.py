# =============================================================================
# ui/app.py — Streamlit Chat Interface
# =============================================================================
# WHAT THIS FILE DOES:
#   Renders the web UI. Users can:
#     - Type natural language questions
#     - See AI answers grounded in the indexed documents
#     - Inspect the exact source chunks that backed each answer
#     - View the full conversation history in the same session
#
# HOW STREAMLIT WORKS (quick primer):
#   Streamlit re-runs this entire script top-to-bottom on every user
#   interaction. State that should persist between runs (like chat history)
#   must be stored in st.session_state — a dict that survives re-runs.
#
# ARCHITECTURE:
#   app.py is the only file a user interacts with. It imports the RAGChain
#   from generate.py and the vector store loader from embed_index.py.
#   All business logic stays in src/ — the UI is purely presentation.
#
# RUN WITH:
#   streamlit run ui/app.py
# =============================================================================

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
from pathlib import Path

# Make sure Python can find the src/ package when running from ui/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import APP_TITLE, APP_DESCRIPTION, DATA_DIR, VECTOR_STORE_DIR
from src.embed_index import load_vector_store, get_store_stats, build_vector_store
from src.clean_chunk import run_ingestion_pipeline
from src.generate import RAGChain, RAGResponse
from src.utils import get_logger


logger = get_logger(__name__)


# =============================================================================
# PAGE SETUP
# =============================================================================
# st.set_page_config MUST be the first Streamlit call in the script.

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-ingest if vector store doesn't exist (e.g. on Streamlit Cloud)
if not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir()):
    with st.spinner("Building knowledge base for first time (this takes ~1 min)..."):
        chunks = run_ingestion_pipeline(DATA_DIR)
        if chunks:
            build_vector_store(chunks)
        else:
            st.error("Failed to extract data chunks for vector store.")
            st.stop()



# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* ---- Global font & background ---- */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ---- Main container ---- */
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* ---- Chat bubbles ---- */
    .user-bubble {
        background: #1a1a2e;
        color: #e0e0e0;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }

    .assistant-bubble {
        background: #0f3460;
        color: #e8f4f8;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 85%;
        font-size: 0.95rem;
        border-left: 3px solid #e94560;
    }

    /* ---- Source card ---- */
    .source-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #a0c4d8;
    }

    .source-score {
        color: #e94560;
        font-weight: 600;
    }

    /* ---- Status badge ---- */
    .status-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================
# These keys persist across Streamlit re-runs within the same browser session.

def init_session_state():
    """Set up session state keys on first load."""
    if "chat_history" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "sources": List[RetrievedChunk] | None}
        st.session_state.chat_history = []

    if "store_stats" not in st.session_state:
        st.session_state.store_stats = None


init_session_state()


# =============================================================================
# RESOURCE LOADING (cached)
# =============================================================================
# @st.cache_resource tells Streamlit to run this function ONCE and reuse
# the result across all re-runs and all users. Without this, the embedding
# model and vector store would reload on every interaction — very slow.
# NOTE: We only cache the embeddings. Caching the Chroma vector store causes
# SQLite threading errors because Streamlit runs each interaction in a new thread.

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    """Load the HuggingFace embeddings model (cached)."""
    from src.embed_index import load_embedding_model
    return load_embedding_model()

def get_rag_chain():
    """Create a fresh RAGChain and vector store on every run."""
    from src.config import CHROMA_COLLECTION_NAME, VECTOR_STORE_DIR
    from src.embed_index import get_store_stats
    from langchain_chroma import Chroma
    
    db_path = VECTOR_STORE_DIR
    if not db_path.exists() or not any(db_path.iterdir()):
        return None, None
        
    embeddings = load_embeddings()
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    stats = get_store_stats(vector_store)
    chain = RAGChain(vector_store)
    return chain, stats


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the left sidebar with status info and controls."""
    with st.sidebar:
        st.markdown("## 📚 RAG Chatbot")
        st.markdown(APP_DESCRIPTION)
        st.divider()

        # Knowledge base status
        st.markdown("### Knowledge Base")
        if st.session_state.store_stats:
            stats = st.session_state.store_stats
            st.success("Vector store loaded")
            st.metric("Indexed Chunks", stats["total_chunks"])
            st.caption(f"Model: `{stats['model_name']}`")
            st.caption(f"Collection: `{stats['collection_name']}`")
        else:
            st.error("Vector store not found")
            st.info(
                "Run the ingestion pipeline first:\n\n"
                "```bash\npython run.py --ingest\n```"
            )

        st.divider()

        # Clear chat button
        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption("Built for: Implementation of Intelligent Systems")
        st.caption("Stack: LangChain · ChromaDB · HuggingFace · Streamlit")


# =============================================================================
# CHAT HISTORY RENDERER
# =============================================================================

def render_message(role: str, content: str, sources=None):
    """
    Render a single chat message as a styled HTML bubble.

    For assistant messages that have sources, also renders expandable
    source cards so the user can verify where the answer came from.

    Args:
        role:    "user" or "assistant"
        content: The message text.
        sources: List of RetrievedChunk objects (only for assistant messages).
    """
    if role == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {content}</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f'<div class="assistant-bubble">🤖 {content}</div>',
            unsafe_allow_html=True,
        )

        # Source cards — shown under every assistant response
        if sources:
            with st.expander(f"📎 View {len(sources)} source(s) used", expanded=False):
                for i, chunk in enumerate(sources, start=1):
                    filename = Path(chunk.source).name
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>Source {i}</strong> — {filename} '
                        f'| chunk #{chunk.chunk_index} '
                        f'| <span class="source-score">score: {chunk.score:.4f}</span>'
                        f'<br><br>{chunk.text[:400]}{"…" if len(chunk.text) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def render_chat_history():
    """Render all messages in the current session's chat history."""
    for msg in st.session_state.chat_history:
        render_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
        )


# =============================================================================
# QUERY HANDLER
# =============================================================================

def handle_query(question: str):
    """
    Process a new user question through the RAG pipeline.

    1. Add the user message to history immediately (gives instant feedback)
    2. Call rag_chain.ask() to retrieve + generate
    3. Add the assistant response + sources to history
    4. Trigger a re-run so the new messages appear

    Args:
        question: The user's input string.
    """
    # Append user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "sources": None,
    })

    # Call the RAG chain with a spinner so the user knows it's working
    with st.spinner("Searching knowledge base and generating answer..."):
        try:
            rag_chain, _ = get_rag_chain()
            response: RAGResponse = rag_chain.ask(question)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            })

        except Exception as exc:
            logger.error(f"RAG pipeline error: {exc}", exc_info=True)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ An error occurred: {exc}",
                "sources": None,
            })

    # Rerun so the new messages render immediately
    st.rerun()


# =============================================================================
# MAIN LAYOUT
# =============================================================================

def main():
    render_sidebar()

    # --- Load resources on first run to update stats ---
    chain, stats = get_rag_chain()
    if stats:
        st.session_state.store_stats = stats

    # --- Page header ---
    st.markdown(f"# 📚 {APP_TITLE}")
    st.markdown(
        "Ask any question. Answers are generated **strictly from indexed documents** "
        "— no hallucination."
    )

    # Guard: if the vector store isn't loaded, block the chat
    if chain is None:
        st.warning(
            "**Knowledge base not found.** "
            "Please run `python run.py --ingest` first to index your documents."
        )
        st.stop()

    st.divider()

    # --- Chat history ---
    if not st.session_state.chat_history:
        st.markdown(
            "<p style='color:#666; text-align:center; margin-top:40px;'>"
            "💬 Ask your first question below to get started."
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        render_chat_history()

    # --- Input box (always at the bottom) ---
    st.divider()
    question = st.chat_input(
        placeholder="Ask a question about your indexed documents...",
    )

    if question:
        handle_query(question)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

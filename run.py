# =============================================================================
# run.py — Command-Line Entry Point
# =============================================================================
# WHAT THIS FILE DOES:
#   Acts as the project's main CLI. Two modes:
#
#   --ingest   Build (or rebuild) the vector store from scratch.
#              Run this ONCE after placing your source files in data/.
#              What happens: load → clean → chunk → embed → persist to disk.
#
#   --chat     Launch the Streamlit web UI in your browser.
#              The vector store must already exist (run --ingest first).
#
#   --stats    Print information about the current vector store.
#
# USAGE EXAMPLES:
#   python run.py --ingest            # index your documents
#   python run.py --chat              # start the web UI
#   python run.py --stats             # inspect the vector store
#   python run.py --ingest --chat     # index then immediately launch UI
# =============================================================================

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import DATA_DIR, VECTOR_STORE_DIR, APP_TITLE
from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# INGEST COMMAND
# =============================================================================

def run_ingest():
    """
    Execute the full ingestion pipeline:
    Load raw files → Clean → Chunk → Embed → Persist to ChromaDB.

    This is deliberately kept thin — all logic lives in the src/ modules.
    run.py just orchestrates them in the right order.
    """
    logger.info("=" * 60)
    logger.info(f"Starting ingestion pipeline for: {DATA_DIR}")
    logger.info("=" * 60)

    # Import here so errors in config/imports show clearly
    from src.clean_chunk import run_ingestion_pipeline
    from src.embed_index import build_vector_store

    # Sanity check: make sure the data directory has files
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        logger.error(
            f"\n  Data directory is empty or missing: {DATA_DIR}\n"
            "  Add your open-source source files (.txt, .pdf, .html) to data/\n"
            "  then re-run:  python run.py --ingest"
        )
        sys.exit(1)

    # Step 1+2+3: Load, clean, and chunk all source files
    chunks = run_ingestion_pipeline(DATA_DIR)

    if not chunks:
        logger.error("No chunks were produced. Check your source files and try again.")
        sys.exit(1)

    # Step 4+5: Embed chunks and persist to ChromaDB
    vector_store = build_vector_store(chunks)

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"Vector store saved to: {VECTOR_STORE_DIR}")
    logger.info("Next step → run:  python run.py --chat")
    logger.info("=" * 60)


# =============================================================================
# CHAT COMMAND
# =============================================================================

def run_chat():
    """
    Launch the Streamlit web UI.

    We use subprocess to call `streamlit run ui/app.py` so the user
    doesn't need to remember the streamlit command.
    The browser opens automatically.
    """
    ui_path = Path(__file__).parent / "ui" / "app.py"

    if not ui_path.exists():
        logger.error(f"UI file not found: {ui_path}")
        sys.exit(1)

    if not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir()):
        logger.warning(
            "Vector store not found. The UI will show a warning.\n"
            "Run python run.py --ingest first to index your documents."
        )

    logger.info(f"Launching {APP_TITLE} UI...")
    logger.info("Open your browser at: http://localhost:8501")

    # subprocess.run blocks until the Streamlit server is killed (Ctrl+C)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ui_path),
             "--server.headless", "false", "--server.fileWatcherType", "none"],
            check=True,
        )
    except KeyboardInterrupt:
        logger.info("Streamlit server stopped.")
    except subprocess.CalledProcessError as exc:
        logger.error(f"Failed to start Streamlit: {exc}")
        sys.exit(1)


# =============================================================================
# STATS COMMAND
# =============================================================================

def run_stats():
    """Print statistics about the current vector store to the terminal."""
    from src.embed_index import load_vector_store, get_store_stats

    vector_store = load_vector_store()
    if vector_store is None:
        logger.error(
            "Vector store not found. Run python run.py --ingest first."
        )
        sys.exit(1)

    stats = get_store_stats(vector_store)
    print("\n" + "=" * 50)
    print("  Vector Store Statistics")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key:<20}: {value}")
    print("=" * 50 + "\n")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=f"{APP_TITLE} — CLI entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --ingest           # index documents in data/
  python run.py --chat             # launch Streamlit UI
  python run.py --stats            # inspect the vector store
  python run.py --ingest --chat    # index then launch UI
        """
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Load, clean, chunk, embed and persist documents from data/",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch the Streamlit web UI",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print vector store statistics",
    )
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    # If no flags given, show help
    if not any([args.ingest, args.chat, args.stats]):
        print(
            "\nUsage:\n"
            "  python run.py --ingest      # build the vector store\n"
            "  python run.py --chat        # launch the web UI\n"
            "  python run.py --stats       # inspect the vector store\n"
            "\nRun python run.py --help for full options.\n"
        )
        sys.exit(0)

    # Flags can be combined: --ingest --chat runs both in sequence
    if args.ingest:
        run_ingest()

    if args.stats:
        run_stats()

    if args.chat:
        run_chat()

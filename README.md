# 📚 RAG Knowledge-Base Chatbot

> **Implementation of Intelligent Systems** — Coursework Project  
> Engineering a Knowledge-Base RAG (Retrieval-Augmented Generation) System

---

## What is RAG?

A standard LLM answers from its training data — it can hallucinate and can't access your private documents.

**RAG fixes this:**
1. Your documents are chunked and stored as semantic vectors in a database.
2. When you ask a question, the system finds the most relevant chunks.
3. Those chunks are injected into the LLM prompt as verified context.
4. The LLM answers **strictly from that context** — no hallucination.

---

## Project Structure

```
rag-chatbot/
├── data/                    ← Place your source files here (.txt, .pdf, .html)
├── src/
│   ├── config.py            ← All settings (models, paths, thresholds)
│   ├── utils.py             ← Shared helpers (logger, timer, file tools)
│   ├── clean_chunk.py       ← Load → Clean → Chunk pipeline
│   ├── embed_index.py       ← Embed chunks → Persist to ChromaDB
│   ├── retrieve.py          ← Semantic search (cosine similarity)
│   └── generate.py          ← Prompt template + LLM + RAG chain
├── ui/
│   └── app.py               ← Streamlit web interface
├── reports/
│   └── system_report.md     ← Engineering whitepaper (coursework deliverable)
├── architecture_diagram/
│   └── pipeline.png         ← Visual pipeline diagram
├── .env.example             ← Environment variable template
├── .gitignore
├── requirements.txt
├── README.md
└── run.py                   ← CLI entry point (ingest / chat / stats)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB (local disk) |
| LLM (local) | Ollama — Llama 3.2 / Mistral |
| LLM (cloud) | OpenAI GPT-4o-mini (optional) |
| UI | Streamlit |

---

## Setup & Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER, model names, etc.
```

### 5. Install and start Ollama (for local LLM)

```bash
# Download from https://ollama.com, then:
ollama pull llama3.2
ollama serve                    # runs in a separate terminal
```

### 6. Add your open-source dataset

Place `.txt`, `.pdf`, or `.html` files inside `data/`.

Example datasets (all open-source):
- **Project Gutenberg** — https://www.gutenberg.org (public domain books)
- **Wikipedia dumps** — https://dumps.wikimedia.org
- **Kaggle open-license** — https://www.kaggle.com/datasets (filter: license = open)

### 7. Build the vector store (run once)

```bash
python run.py --ingest
```

This loads, cleans, chunks, embeds, and persists all documents. Takes 1–5 minutes depending on dataset size.

### 8. Launch the web UI

```bash
python run.py --chat
# Or run both steps at once:
python run.py --ingest --chat
```

Open **http://localhost:8501** in your browser.

---

## Data Pipeline (Flow)

```
data/*.txt / *.pdf / *.html
        │
        ▼  src/clean_chunk.py
   Load → Clean → Chunk (500 chars, 10% overlap)
        │
        ▼  src/embed_index.py
   HuggingFace Embeddings → ChromaDB (persisted to vector_store/)
        │
        ▼  [At query time]
   User Question → src/retrieve.py
   Embed query → Cosine similarity search → Top-4 chunks
        │
        ▼  src/generate.py
   Strict system prompt + context injection → Ollama LLM
        │
        ▼  ui/app.py
   Answer + verified source cards displayed in Streamlit
```

---

## Dataset Attribution

All source documents used in this project are open-source and publicly available.  
See `data/sources.md` for full attribution details including licenses and URLs.

---

## Key Design Decisions

**Sliding window chunking (500 chars, 10% overlap)**  
Prevents answers from being cut off at chunk boundaries. The 200-character overlap repeats the tail of each chunk at the start of the next.

**Cosine similarity with threshold filtering**  
Chunks with a relevance score below `SIMILARITY_THRESHOLD` (default: 0.3) are discarded — this prevents low-confidence context from confusing the LLM.

**Strict prompt engineering**  
The system prompt explicitly forbids extrapolation. If the answer isn't in the retrieved context, the model responds with a standard fallback message rather than hallucinating.

**Single config.py**  
All tunable parameters live in one file. Switching embedding models, LLMs, or chunk sizes requires editing exactly one place.

---

## CLI Reference

```bash
python run.py --ingest    # build vector store from data/
python run.py --chat      # launch Streamlit UI
python run.py --stats     # inspect vector store statistics
python run.py --help      # full help
```

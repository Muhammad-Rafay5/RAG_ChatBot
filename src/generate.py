# =============================================================================
# generate.py — Prompt Engineering & LLM Response Generation
# =============================================================================
# WHAT THIS FILE DOES:
#   1. PROMPT  — Build a strict, context-only prompt from retrieved chunks
#   2. LLM     — Load the language model (Ollama local or OpenAI API)
#   3. CHAIN   — Wire retrieval + prompt + LLM into one callable pipeline
#   4. ANSWER  — Return the final answer + the sources that backed it
#
# THE CORE IDEA OF RAG (all comes together here):
#   Standard LLM prompt:  "User: What is X?  →  Model answers from memory"
#   RAG prompt:           "Here are verified facts [CONTEXT]. Answer ONLY
#                          from these facts. User: What is X?  →  Model
#                          answers from the injected context, not memory."
#
# WHY A STRICT PROMPT?
#   Without constraints the LLM "hallucinates" — it fills gaps with plausible
#   but wrong information from its training data. The strict prompt template
#   forces it to say "I don't know" if the answer isn't in the context.
# =============================================================================

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
)
from src.retrieve import RetrievedChunk, retrieve, format_context
from src.utils import get_logger, timer

logger = get_logger(__name__)


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class RAGResponse:
    """
    The complete output of one RAG query.

    Attributes:
        answer:  The LLM's text answer (grounded in the retrieved context).
        sources: The chunks that were injected into the prompt — the user
                 can inspect these to verify where the answer came from.
        context: The formatted context string that was sent to the LLM.
    """
    answer:  str
    sources: List[RetrievedChunk]
    context: str


# =============================================================================
# STEP 1 — PROMPT TEMPLATE
# =============================================================================
# This is the most important part of a RAG system.
# The prompt template has two runtime variables:
#   {context}    → filled with the retrieved chunks (from retrieve.py)
#   {question}   → filled with the user's query
#
# The system message is deliberately strict:
#   "Use ONLY the context" + "If you can't find it, say so explicitly"
#   This directly follows the assignment's example system prompt.

SYSTEM_TEMPLATE = """You are an objective, factual AI assistant \
built on a verified knowledge base.

Your task is to answer the user's question using ONLY the source \
context fragments provided below. Follow these rules strictly:

1. Base every part of your answer exclusively on the context provided.
2. Do NOT use any prior knowledge, assumptions, or external information.
3. If the answer cannot be confidently found in the context, respond \
with exactly:
   "The requested information is not available in the provided knowledge base."
4. Do not speculate, infer beyond the text, or make things up.
5. Keep your answer concise and direct.

[CONTEXT]
{context}
"""

HUMAN_TEMPLATE = "[USER QUERY]\n{question}"


def build_prompt_template() -> ChatPromptTemplate:
    """
    Construct the LangChain ChatPromptTemplate from the two message templates.

    ChatPromptTemplate takes a list of (role, content) tuples.
    - "system" → sets the LLM's behaviour rules (sent once per conversation)
    - "human"  → the user's message

    Returns:
        A ChatPromptTemplate instance ready to be used in a chain.
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human",  HUMAN_TEMPLATE),
    ])


# =============================================================================
# STEP 2 — LLM LOADER
# =============================================================================

def load_llm():
    """
    Load and return the language model based on LLM_PROVIDER in config.py.

    Supported providers:
    ┌──────────┬──────────────────────────────────────────────────────────┐
    │ groq     │ Groq cloud API. Very fast, generous free tier.           │
    │          │ Get key at: https://console.groq.com                     │
    ├──────────┼──────────────────────────────────────────────────────────┤
    │ ollama   │ Runs a model locally via Ollama. Free, private, offline. │
    │          │ Requires: ollama pull llama3.2 (run once in terminal)    │
    ├──────────┼──────────────────────────────────────────────────────────┤
    │ openai   │ Calls OpenAI's API. Requires OPENAI_API_KEY in .env.     │
    │          │ Costs money per token but is faster and more capable.    │
    └──────────┴──────────────────────────────────────────────────────────┘

    Returns:
        A LangChain-compatible chat model instance.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "groq":
        # langchain-groq wraps the Groq REST API.
        # Groq uses purpose-built LPU (Language Processing Unit) hardware
        # which makes it significantly faster than GPU-based providers.
        # Free tier is very generous — good for coursework projects.
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "1. Get a free key at https://console.groq.com\n"
                "2. Add  GROQ_API_KEY=gsk_...  to your .env file"
            )

        logger.info(f"Loading Groq model: {GROQ_MODEL}")
        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

    elif provider == "ollama":
        # Lazy import — only load if actually using Ollama
        from langchain_ollama import ChatOllama

        logger.info(f"Loading Ollama model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        logger.info(f"Loading OpenAI model: {OPENAI_MODEL}")
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            "Set LLM_PROVIDER to 'groq', 'ollama', or 'openai' in your .env file."
        )

    logger.info("LLM loaded ✓")
    return llm


# =============================================================================
# STEP 3 — RAG CHAIN
# =============================================================================

class RAGChain:
    """
    The end-to-end RAG pipeline: retrieve → format → prompt → LLM → parse.

    Usage:
        chain = RAGChain(vector_store)
        response = chain.ask("What is the boiling point of water?")
        print(response.answer)
        print(response.sources)

    Internals (LangChain LCEL notation):
        The | operator chains runnables left-to-right.
        Each step's output becomes the next step's input.

        prompt_template | llm | output_parser
             ↑                       ↑
        fills {context}        parses AIMessage
        and {question}         → plain string
    """

    def __init__(self, vector_store):
        """
        Initialize the chain.

        Args:
            vector_store: A loaded Chroma instance from embed_index.py.
        """
        self.vector_store = vector_store
        self.prompt       = build_prompt_template()
        self.llm          = load_llm()
        # StrOutputParser converts the LLM's AIMessage object → plain string
        self.parser       = StrOutputParser()

        # Build the LangChain Expression Language (LCEL) chain.
        # This is just: prompt → llm → parse output
        self._chain = self.prompt | self.llm | self.parser

        logger.info("RAGChain initialized ✓")

    @timer
    def ask(self, question: str) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Full flow:
          1. retrieve()      — find top-k relevant chunks from the vector store
          2. format_context() — combine chunks into one context string
          3. _chain.invoke() — fill the prompt template and call the LLM
          4. Return RAGResponse with answer + source metadata

        Args:
            question: The user's natural language question.

        Returns:
            RAGResponse with .answer, .sources, and .context.
        """
        logger.info(f"Question: {question}")

        # --- Step 1: Semantic Retrieval ---
        sources = retrieve(question, self.vector_store)

        # --- Step 2: Format context string for prompt injection ---
        context = format_context(sources)

        # --- Step 3: Run the LLM chain ---
        # .invoke() fills the template variables and calls the model.
        answer = self._chain.invoke({
            "context":  context,
            "question": question,
        })

        logger.info(f"Answer generated ({len(answer)} chars)")

        return RAGResponse(
            answer=answer.strip(),
            sources=sources,
            context=context,
        )

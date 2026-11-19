"""FastAPI application serving RAG chatbot queries over DarGlobal data.

Endpoints:
  POST /chat   – Accept a question (+ optional history), return grounded answer.
  GET  /health – Liveness / readiness probe.
  POST /ingest – Rebuild the vector-search index from data/raw/darglobal.json.
  POST /scrape – Run the scraping pipeline (disabled by default; set
                 ENABLE_SCRAPE_ENDPOINT=true in .env to enable).

Run:
  uvicorn api.main:app --reload --port 8000
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    RetrievedDoc,
    ScrapeResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DarGlobal RAG Chatbot",
    description=(
        "Retrieval-Augmented Generation chatbot grounded exclusively on "
        "public data scraped from darglobal.co.uk."
    ),
    version="1.0.0",
)

# CORS – allow Streamlit frontend (default port 8501) and common dev origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers – lazy imports so the module loads instantly (no ChromaDB / fastembed
# at import time), which keeps tests fast and avoids sandbox hangs.
# ---------------------------------------------------------------------------
def _answer_question(question: str, history: list[dict[str, str]]) -> dict:
    """Lazy-import wrapper around rag.graph.answer_question."""
    from rag.graph import answer_question  # noqa: C0415
    return answer_question(question=question, history=history)


def _run_ingest() -> int:
    """Lazy-import wrapper around rag.ingest.ingest; returns chunk count."""
    from rag.ingest import ingest  # noqa: C0415
    return ingest()


def _run_scrape() -> str:
    """Lazy-import wrapper around scraper.run_scrape.run_pipeline."""
    from scraper.run_scrape import run_pipeline  # noqa: C0415
    records, summary = run_pipeline(target="darglobal")
    return f"Scrape completed ({len(records)} records). Run POST /ingest to rebuild the index."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Return service health and configured model name."""
    from rag.llm import get_llm_client  # noqa: C0415

    model = get_llm_client().model
    return HealthResponse(status="ok", model=model)


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Accept a user question and return a grounded answer with citations.

    The endpoint delegates to the LangGraph RAG graph which:
    1. Classifies the query (rag / chitchat / out_of_scope).
    2. Rewrites follow-ups into standalone queries.
    3. Retrieves relevant chunks via hybrid search.
    4. Generates an answer strictly from the retrieved context.
    5. Verifies groundedness before responding.
    """
    question = request.question

    try:
        history = [msg.model_dump() for msg in request.history]
        result = _answer_question(question=question, history=history)
    except RuntimeError as err:
        # LLM rate-limit or config errors bubble up as RuntimeError
        logger.error("RAG pipeline error: %s", err)
        raise HTTPException(
            status_code=503,
            detail=(
                "The LLM service is temporarily unavailable (likely rate-limited). "
                f"Details: {err}"
            ),
        ) from err
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your question.",
        ) from exc

    retrieved_docs = [
        RetrievedDoc(
            text=doc.get("text", ""),
            metadata=doc.get("metadata", {}),
            score=doc.get("score", 0.0),
        )
        for doc in result.get("retrieved_docs", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        route=result.get("route", "rag"),
        retrieved_docs=retrieved_docs,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["system"])
async def ingest() -> IngestResponse:
    """Rebuild the ChromaDB vector index from data/raw/darglobal.json."""
    try:
        chunk_count = _run_ingest()
        return IngestResponse(status="ok", chunks=chunk_count)
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {fnf}. Run the scraper first.",
        ) from fnf
    except Exception as exc:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/scrape", response_model=ScrapeResponse, tags=["system"])
async def scrape() -> ScrapeResponse:
    """Run the DarGlobal scraping pipeline.

    Disabled by default. Set ENABLE_SCRAPE_ENDPOINT=true in .env to enable.
    """
    enabled = os.getenv("ENABLE_SCRAPE_ENDPOINT", "false").lower() in ("true", "1", "yes")
    if not enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "The /scrape endpoint is disabled. "
                "Set ENABLE_SCRAPE_ENDPOINT=true in .env to enable it."
            ),
        )
    try:
        message = _run_scrape()
        return ScrapeResponse(status="ok", message=message)
    except Exception as exc:
        logger.exception("Scrape error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single message in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'.")
    content: str = Field(..., max_length=2000, description="Message text content.")


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question text (max 1 000 characters).",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Prior conversation turns (max 20).",
    )
    source_filter: str | None = Field(
        default=None,
        description="Optional source filter.",
    )


class RetrievedDoc(BaseModel):
    """Single retrieved document returned to the client."""

    text: str = Field("", description="Chunk text shown to the user.")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata.")
    score: float = Field(0.0, description="Retrieval relevance score.")


class ChatResponse(BaseModel):
    """Chat endpoint response payload."""

    answer: str = Field(..., description="Generated answer grounded in scraped data.")
    citations: list[str] = Field(
        default_factory=list, description="Source URLs cited in the answer."
    )
    route: str = Field(
        "rag", description="Classification route: rag | chitchat | out_of_scope."
    )
    retrieved_docs: list[RetrievedDoc] = Field(
        default_factory=list,
        description="Retrieved context documents (for transparency).",
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    model: str = ""


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""

    status: str = "ok"
    chunks: int = Field(0, description="Number of chunks ingested.")


class ScrapeResponse(BaseModel):
    """Response from the /scrape endpoint."""

    status: str = "ok"
    message: str = ""

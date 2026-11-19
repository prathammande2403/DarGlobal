"""Tests for the FastAPI chat API (api/main.py & api/schemas.py).

Uses FastAPI's TestClient with mocked RAG graph so tests run without
network, LLM, or ChromaDB dependencies.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_RAG_RESULT = {
    "answer": "Dar Global has the W Residences project in Dubai Marina.",
    "citations": ["https://darglobal.co.uk/projects/w-residences"],
    "route": "rag",
    "retrieved_docs": [
        {
            "text": "[DarGlobal | project | Dubai] W Residences\nLuxury residences in Dubai Marina.",
            "metadata": {
                "source": "darglobal",
                "url": "https://darglobal.co.uk/projects/w-residences",
                "page_type": "project",
            },
            "score": 0.85,
        }
    ],
}

MOCK_CHITCHAT_RESULT = {
    "answer": "Hello! I am your real estate research assistant for DarGlobal.",
    "citations": [],
    "route": "chitchat",
    "retrieved_docs": [],
}

MOCK_OOS_RESULT = {
    "answer": "I specialize exclusively in real estate properties.",
    "citations": [],
    "route": "out_of_scope",
    "retrieved_docs": [],
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model" in body


# ---------------------------------------------------------------------------
# Chat endpoint – unit tests
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    """Tests for POST /chat."""

    @patch("api.main._answer_question", return_value=MOCK_RAG_RESULT)
    def test_chat_rag_response(self, mock_aq) -> None:
        payload = {"question": "What projects does Dar Global have in Dubai?"}
        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["route"] == "rag"
        assert "W Residences" in body["answer"]
        assert len(body["citations"]) == 1
        assert body["citations"][0].startswith("https://")
        assert len(body["retrieved_docs"]) == 1
        assert body["retrieved_docs"][0]["metadata"]["source"] == "darglobal"
        mock_aq.assert_called_once()

    @patch("api.main._answer_question", return_value=MOCK_CHITCHAT_RESULT)
    def test_chat_chitchat_response(self, mock_aq) -> None:
        payload = {"question": "Hello!"}
        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["route"] == "chitchat"
        assert body["citations"] == []
        assert body["retrieved_docs"] == []

    @patch("api.main._answer_question", return_value=MOCK_RAG_RESULT)
    def test_chat_with_history(self, mock_aq) -> None:
        payload = {
            "question": "What about its price?",
            "history": [
                {"role": "user", "content": "Tell me about W Residences"},
                {"role": "assistant", "content": "W Residences is a luxury project."},
            ],
        }
        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        _, kwargs = mock_aq.call_args
        assert len(kwargs["history"]) == 2

    @patch("api.main._answer_question", return_value=MOCK_RAG_RESULT)
    def test_chat_delegates_to_rag(self, mock_aq) -> None:
        payload = {"question": "List luxury projects"}
        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        _, kwargs = mock_aq.call_args
        assert kwargs["question"] == "List luxury projects"

    def test_chat_empty_question_rejected(self) -> None:
        payload = {"question": ""}
        response = client.post("/chat", json=payload)
        assert response.status_code == 422

    def test_chat_missing_question_rejected(self) -> None:
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_question_too_long_rejected(self) -> None:
        payload = {"question": "x" * 1001}
        response = client.post("/chat", json=payload)
        assert response.status_code == 422

    @patch("api.main._answer_question", side_effect=RuntimeError("Rate limited"))
    def test_chat_runtime_error_returns_503(self, mock_aq) -> None:
        payload = {"question": "Some question"}
        response = client.post("/chat", json=payload)

        assert response.status_code == 503
        assert "Rate limited" in response.json()["detail"]

    @patch("api.main._answer_question", side_effect=ValueError("unexpected"))
    def test_chat_unexpected_error_returns_500(self, mock_aq) -> None:
        payload = {"question": "Some question"}
        response = client.post("/chat", json=payload)

        assert response.status_code == 500
        assert "internal error" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Ingest & scrape endpoints
# ---------------------------------------------------------------------------


class TestSystemEndpoints:
    """Tests for POST /ingest and POST /scrape."""

    @patch("api.main._run_ingest", return_value=42)
    def test_ingest_returns_chunk_count(self, mock_ingest) -> None:
        response = client.post("/ingest")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["chunks"] == 42

    @patch("api.main._run_ingest", side_effect=FileNotFoundError("combined.json"))
    def test_ingest_missing_data_returns_404(self, mock_ingest) -> None:
        response = client.post("/ingest")
        assert response.status_code == 404

    def test_scrape_disabled_by_default(self) -> None:
        response = client.post("/scrape")
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()

    @patch.dict("os.environ", {"ENABLE_SCRAPE_ENDPOINT": "true"})
    @patch("api.main._run_scrape", return_value="Done")
    def test_scrape_enabled_runs_pipeline(self, mock_scrape) -> None:
        response = client.post("/scrape")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for Pydantic schemas."""

    def test_chat_response_schema_fields(self) -> None:
        from api.schemas import ChatResponse

        resp = ChatResponse(answer="test", citations=[], route="rag", retrieved_docs=[])
        assert resp.answer == "test"
        assert resp.route == "rag"

    def test_chat_request_defaults(self) -> None:
        from api.schemas import ChatRequest

        req = ChatRequest(question="hello")
        assert req.history == []
        assert req.source_filter is None


# ---------------------------------------------------------------------------
# End-to-end smoke tests (3 sample questions through /chat with mocked RAG)
# ---------------------------------------------------------------------------


class TestE2ESmokeTests:
    """End-to-end smoke tests hitting /chat with 3 representative questions."""

    @patch("api.main._answer_question", return_value=MOCK_RAG_RESULT)
    def test_smoke_project_query(self, mock_aq) -> None:
        """Q1: Real estate project question → expects rag route and answer."""
        resp = client.post("/chat", json={"question": "What projects does Dar Global have in Dubai?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "rag"
        assert len(body["answer"]) > 0
        assert isinstance(body["citations"], list)
        assert isinstance(body["retrieved_docs"], list)

    @patch("api.main._answer_question", return_value=MOCK_CHITCHAT_RESULT)
    def test_smoke_greeting(self, mock_aq) -> None:
        """Q2: Simple greeting → expects chitchat route."""
        resp = client.post("/chat", json={"question": "Hello, what can you do?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "chitchat"
        assert body["citations"] == []

    @patch("api.main._answer_question", return_value=MOCK_OOS_RESULT)
    def test_smoke_out_of_scope(self, mock_aq) -> None:
        """Q3: Off-topic question → expects out_of_scope route."""
        resp = client.post("/chat", json={"question": "What is the weather in Paris today?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "out_of_scope"
        assert body["citations"] == []

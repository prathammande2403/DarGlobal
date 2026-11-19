"""Streamlit chat interface for the DarGlobal RAG chatbot.

Calls the FastAPI /chat endpoint (configurable via API_URL env var).

Run:
  streamlit run ui/app.py
"""

import os
import re

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CONTENT_CHARS = 2000


def format_assistant_answer(text: str) -> str:
    """Put inline numbered headings on separate Markdown paragraphs."""
    return re.sub(
        r"(?<!\n)[ \t]+(?=(?:\*\*)?\d{1,2}\.\s+)",
        "\n\n",
        text,
    )


SAMPLE_QUESTIONS = [
    "What projects does Dar Global have in Dubai?",
    "What amenities does the W Residences project offer?",
    "Tell me about Tierra Viva",
    "Where are Dar Global's offices located?",
    "What is the starting price for Trump Tower Dubai?",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DarGlobal Real Estate Chatbot",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 DarGlobal Real Estate Chatbot")
st.caption(
    "Ask about luxury properties, developments, prices, amenities, and company info "
    "from [darglobal.co.uk](https://darglobal.co.uk). "
    "Answers are grounded exclusively in publicly scraped data."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "history" not in st.session_state:
    st.session_state.history: list[dict[str, str]] = []

# ---------------------------------------------------------------------------
# Sidebar: sample questions, info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💡 Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state["_pending_question"] = q
            st.rerun()

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        "This chatbot answers questions **only** from publicly scraped data from **DarGlobal** (`darglobal.co.uk`). "
        "It never invents prices, locations, or specifications."
    )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: call the FastAPI /chat endpoint
# ---------------------------------------------------------------------------
def call_chat_api(question: str, history: list[dict[str, str]]) -> dict:
    """Send a question to the FastAPI backend and return the parsed response."""
    bounded_history = [
        {"role": item.get("role", "user"), "content": item.get("content", "")[:MAX_HISTORY_CONTENT_CHARS]}
        for item in history[-MAX_HISTORY_MESSAGES:]
    ]
    payload = {"question": question, "history": bounded_history}
    try:
        resp = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {
            "answer": (
                f"⚠️ Could not connect to the API at `{API_URL}`. "
                "Make sure the FastAPI server is running:\n\n"
                "```bash\nuvicorn api.main:app --port 8000\n```"
            ),
            "citations": [],
            "route": "error",
            "retrieved_docs": [],
        }
    except requests.HTTPError as err:
        status_code = err.response.status_code
        if status_code == 503:
            message = "The AI model is unavailable right now. Please try again shortly."
        elif status_code == 422:
            message = "I couldn't process this request. Please shorten your message or clear the chat and try again."
        else:
            message = "The chat service hit a temporary error. Please try again shortly."
        return {
            "answer": f"⚠️ {message}",
            "citations": [],
            "route": "error",
            "retrieved_docs": [],
        }
    except requests.Timeout:
        return {
            "answer": "⚠️ Request timed out. The LLM may be slow — please try again.",
            "citations": [],
            "route": "error",
            "retrieved_docs": [],
        }


# ---------------------------------------------------------------------------
# Render prior messages
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(format_assistant_answer(msg["content"]))
        if msg.get("citations"):
            with st.expander("📎 Sources"):
                for url in msg["citations"]:
                    st.markdown(f"- [{url}]({url})")
        if msg.get("route") and msg["role"] == "assistant":
            route_label = msg["route"]
            colour = {"rag": "🟢", "chitchat": "🔵", "out_of_scope": "🟠", "error": "🔴"}.get(
                route_label, "⚪"
            )
            st.caption(f"{colour} Route: **{route_label}**")

# ---------------------------------------------------------------------------
# Chat input (from text box or sample-question button)
# ---------------------------------------------------------------------------
pending = st.session_state.pop("_pending_question", None)
user_input = pending or st.chat_input("Ask about DarGlobal properties…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching DarGlobal data…"):
            result = call_chat_api(user_input, st.session_state.history)

        answer = format_assistant_answer(result.get("answer", ""))
        citations = result.get("citations", [])
        route = result.get("route", "rag")

        st.markdown(answer)

        if citations:
            with st.expander("📎 Sources"):
                for url in citations:
                    st.markdown(f"- [{url}]({url})")

        route_colour = {"rag": "🟢", "chitchat": "🔵", "out_of_scope": "🟠", "error": "🔴"}.get(
            route, "⚪"
        )
        st.caption(f"{route_colour} Route: **{route}**")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "route": route,
    })

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.history = st.session_state.history[-6:]

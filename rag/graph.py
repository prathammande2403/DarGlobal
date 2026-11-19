"""LangGraph workflow for grounded real estate retrieval-augmented generation.

Graph Nodes:
1. classify: route to "rag" | "chitchat" | "out_of_scope"
2. rewrite: turn follow-up questions into standalone queries; detect document_type intent
3. retrieve: execute hybrid search with document_type filtering (auto-retry if < 3 docs)
4. grade: verify doc relevance; retry once with broadened query if irrelevant
5. generate: generate grounded response citing URLs and original currencies
6. verify: lightweight groundedness check; regenerate once or return safe fallback

Exposes:
- answer_question(question, history) -> {answer, citations, route, retrieved_docs}
- CLI entry: python -m rag.graph "query"
"""

import logging
import re
import sys
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from rag.llm import call_llm
from rag.guards import PRICE_PATTERN, is_on_topic, is_price_question, is_prompt_injection, strip_price_sentences
from rag.prompts import (
    CHITCHAT_RESPONSE,
    CLASSIFY_PROMPT,
    GENERATE_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    REWRITE_PROMPT,
    SAFE_FALLBACK_ANSWER,
    SYSTEM_PROMPT,
)
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

class GraphState(TypedDict, total=False):
    """Typed state for the LangGraph Q&A graph."""

    question: str
    history: list[dict[str, str]]
    route: str  # "rag" | "chitchat" | "out_of_scope"
    rewritten_query: str
    doc_type_filter: Optional[str]  # "property_listing" | "corporate" | "news" | None
    docs: list[dict[str, Any]]
    answer: str
    citations: list[str]
    grounded: bool
    retry_count: int  # For retrieval retry loop (max 1)
    verify_retry_count: int  # For verification retry loop (max 1)
    needs_retrieval_retry: bool
    retrieval_failed: bool


TRANSIENT_ERROR_ANSWER = "I'm having trouble answering that right now. Please try again in a moment."
MODEL_UNAVAILABLE_ANSWER = "The AI model is unavailable right now. Please try again shortly."
TOPIC_REFUSAL = "I can only help with Dar Global properties and company information. Try asking about projects, locations or amenities."
PRICE_REFUSAL = "I can't share pricing here. Please check the official Dar Global website or contact their sales team for current prices. I can help with location, type, amenities or status instead."
BRACKETED_TAG_PATTERN = re.compile(r"\[[^\]]*\|[^\]]*\]")
UNAVAILABLE_ANSWER_PATTERN = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't)\s+have\s+that\s+information|"
    r"(?:the\s+)?(?:collected\s+)?data\s+(?:does\s+not|doesn't|do\s+not)\s+include|"
    r"not\s+(?:available|included|covered|found)\s+in\s+the\s+(?:collected\s+)?data|"
    r"no\s+(?:relevant\s+)?information\s+(?:is\s+)?available)\b",
    re.IGNORECASE,
)


def _guard_output(text: str) -> str:
    """Apply the final deterministic output filter."""
    if text in (TRANSIENT_ERROR_ANSWER, SAFE_FALLBACK_ANSWER, TOPIC_REFUSAL, PRICE_REFUSAL, CHITCHAT_RESPONSE, OUT_OF_SCOPE_RESPONSE):
        return text
    if UNAVAILABLE_ANSWER_PATTERN.search(text):
        return SAFE_FALLBACK_ANSWER
    try:
        answer = strip_price_sentences(text)
        answer = BRACKETED_TAG_PATTERN.sub("", answer).strip()
        # LLMs sometimes place ordered-list headings after prose on the same
        # line. Separate each inline numbered heading for readable rendering.
        answer = re.sub(
            r"([^\n])[ \t]+(?=(?:\*\*)?\d{1,2}\.\s+(?:\*\*)?[\w])",
            r"\1\n\n",
            answer,
        )
        if PRICE_PATTERN.search(answer) or BRACKETED_TAG_PATTERN.search(answer):
            answer = strip_price_sentences(answer)
            answer = BRACKETED_TAG_PATTERN.sub("", answer).strip()
        if not answer:
            try:
                raise ValueError("Generated answer was empty after output safety filtering.")
            except ValueError:
                logger.exception("Output guard removed the entire generated answer")
            return TRANSIENT_ERROR_ANSWER
        return answer
    except Exception:
        logger.exception("Answer output guard failed; preserving unfiltered LLM answer")
        return text


# ---------------------------------------------------------------------------
# Node 1: Classify
# ---------------------------------------------------------------------------

CHITCHAT_PATTERNS = re.compile(
    r"^(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you|who\s+are\s+you|thanks|thank\s+you|bye|goodbye)[!.?]*$",
    re.IGNORECASE,
)

REAL_ESTATE_KEYWORDS = re.compile(
    r"\b(darglobal|villa|villas|apartment|apartments|penthouse|penthouses|mansion|mansions|"
    r"real\s+estate|property|properties|project|projects|tower|towers|residence|residences|"
    r"developer|price|prices|pricing|cost|aed|gbp|eur|sqm|bedroom|bedrooms|bed|beds|"
    r"bathroom|bathrooms|location|locations|dubai|london|spain|benahavis|oman|muscat|qatar|"
    r"maldives|amenities|amenity|track\s+record|listing|listings|contact|office|offices)\b",
    re.IGNORECASE,
)


def classify_node(state: GraphState) -> dict[str, Any]:
    """Classify user query into 'rag', 'chitchat', or 'out_of_scope'."""
    question = (state.get("question") or "").strip()

    if is_prompt_injection(question):
        return {"route": "direct_reply", "answer": TOPIC_REFUSAL, "citations": [], "docs": []}
    if is_price_question(question):
        return {"route": "direct_reply", "answer": PRICE_REFUSAL, "citations": [], "docs": []}
    if not is_on_topic(question):
        return {"route": "out_of_scope"}

    # Fast heuristic check for greetings
    if CHITCHAT_PATTERNS.match(question):
        return {"route": "chitchat"}

    # Fast heuristic check for explicit real estate terms
    if REAL_ESTATE_KEYWORDS.search(question):
        return {"route": "rag"}

    return {"route": "rag"}


def direct_reply_node(state: GraphState) -> dict[str, Any]:
    return {"answer": state.get("answer", TOPIC_REFUSAL), "citations": [], "docs": [], "grounded": True}


def handle_chitchat_node(state: GraphState) -> dict[str, Any]:
    """Provide helpful greeting explaining chatbot capabilities."""
    return {
        "answer": CHITCHAT_RESPONSE,
        "citations": [],
        "docs": [],
        "grounded": True,
    }


def handle_out_of_scope_node(state: GraphState) -> dict[str, Any]:
    """Politely refuse queries unrelated to DarGlobal real estate."""
    return {
        "answer": OUT_OF_SCOPE_RESPONSE,
        "citations": [],
        "docs": [],
        "grounded": True,
    }


# ---------------------------------------------------------------------------
# Node 2: Rewrite & Document Type Intent Detection
# ---------------------------------------------------------------------------

PROPERTY_INTENT_KEYWORDS = re.compile(
    # Listing verbs alone do not imply a property query (e.g. "list the
    # countries we operate in"). The query must name a property-related term.
    r"\b(available|projects?|apartments?|villas?|prices?|pricing|bedrooms?|beds?|"
    r"bathrooms?|locations?|in\s+dubai|properties|property|tower|residences?|penthouses?|"
    r"mansions?|units?|sqm|cost|starting\s+price)\b",
    re.IGNORECASE,
)

CORPORATE_INTENT_KEYWORDS = re.compile(
    # Do not classify the conversational phrase "tell me about <project>" as
    # corporate intent. It otherwise filters project records out of retrieval.
    r"\b(company|corporate|board|leadership|management|founder|ceo|offices?|headquarters?|"
    r"track\s+record|history|investor\s+relations|who\s+is|team)\b",
    re.IGNORECASE,
)

NEWS_INTENT_KEYWORDS = re.compile(
    r"\b(news|press|announcements?|articles?|updates?|blog|media)\b",
    re.IGNORECASE,
)


def detect_document_type(query: str) -> Optional[str]:
    """Infer document_type filter: 'property_listing', 'corporate', 'news', or None."""
    q_lower = query.lower()
    if PROPERTY_INTENT_KEYWORDS.search(q_lower):
        return "property_listing"
    if CORPORATE_INTENT_KEYWORDS.search(q_lower):
        return "corporate"
    if NEWS_INTENT_KEYWORDS.search(q_lower):
        return "news"
    return None


def rewrite_node(state: GraphState) -> dict[str, Any]:
    """Turn follow-up questions into standalone queries and detect document_type filter."""
    question = state.get("question", "")
    history = state.get("history") or []

    # If no history, classify directly from question
    if not history:
        doc_type = detect_document_type(question)
        return {
            "rewritten_query": question,
            "doc_type_filter": doc_type,
            "retry_count": 0,
            "verify_retry_count": 0,
        }

    # Format history into readable transcript
    history_lines = []
    for msg in history[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {msg.get('content', '')}")
    history_str = "\n".join(history_lines)

    prompt = REWRITE_PROMPT.format(history=history_str, question=question)
    standalone_query = question
    follow_up = bool(
        re.match(
            r"\s*(?:but\b|and\b|also\b|however\b|i\s+saw\b|there\s+(?:are|were)\b|"
            r"you\s+(?:said|mentioned|listed)\b|what\s+about\b)",
            question,
            re.IGNORECASE,
        )
    )
    previous_user_question = next(
        (msg.get("content", "") for msg in reversed(history) if msg.get("role") == "user"),
        "",
    )

    try:
        resp = call_llm([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
        for line in resp.splitlines():
            if "STANDALONE_QUERY:" in line:
                standalone_query = line.split("STANDALONE_QUERY:", 1)[1].strip() or question
    except Exception as err:
        logger.warning("Query rewrite LLM call failed (%s); using original question", err)

    # If a conversational follow-up could not be rewritten, retain its prior
    # topic so retrieval does not search using only fragments like "7 properties".
    if follow_up and standalone_query == question and previous_user_question:
        standalone_query = f"{previous_user_question} {question}"

    doc_type = detect_document_type(standalone_query)
    return {
        "rewritten_query": standalone_query,
        "doc_type_filter": doc_type,
        "retry_count": 0,
        "verify_retry_count": 0,
    }


# ---------------------------------------------------------------------------
# Node 3: Retrieve (with document_type filter & auto-retry)
# ---------------------------------------------------------------------------


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Execute hybrid vector + BM25 search with document_type filtering.

    If a property query returns fewer than 3 docs, retries with no filter.
    """
    query = state.get("rewritten_query") or state.get("question") or ""
    doc_type = state.get("doc_type_filter")

    filters = {"document_type": doc_type} if doc_type else None
    try:
        # A broader result set improves recall for questions needing several
        # relevant records, such as comparisons and portfolio summaries.
        docs = retrieve(query=query, k=20, filters=filters)

    # If property query returns fewer than 3 docs, retry with no filter
        if doc_type == "property_listing" and len(docs) < 3:
            logger.info(
                "Property query returned %d docs (< 3); retrying with no filter",
                len(docs),
            )
            docs = retrieve(query=query, k=20, filters=None)
    except Exception:
        logger.exception("Document retrieval failed")
        return {"docs": [], "retrieval_failed": True}

    return {"docs": docs}


# ---------------------------------------------------------------------------
# Node 4: Grade Documents
# ---------------------------------------------------------------------------


def grade_node(state: GraphState) -> dict[str, Any]:
    """Check if retrieved docs are relevant. Broadens query and retries once if not."""
    docs = state.get("docs") or []
    query = state.get("rewritten_query") or state.get("question") or ""
    retry_count = state.get("retry_count", 0)

    if not docs:
        if state.get("retrieval_failed"):
            return {"needs_retrieval_retry": False}
        if retry_count < 1:
            tokens = re.findall(r"\w+", query)
            broadened = " ".join(tokens[:3]) + " property projects" if tokens else "luxury real estate"
            logger.info("No docs retrieved. Broadening query from '%s' to '%s' (retry 1)", query, broadened)
            return {
                "rewritten_query": broadened,
                "doc_type_filter": None,
                "retry_count": 1,
                "needs_retrieval_retry": True,
            }
        return {"needs_retrieval_retry": False}

    # Keyword check: do query terms appear in retrieved documents?
    query_words = set(re.findall(r"\w{3,}", query.lower()))
    doc_text_combined = " ".join(d.get("text", "") for d in docs).lower()
    matches = sum(1 for w in query_words if w in doc_text_combined)

    is_relevant = bool(matches >= min(1, len(query_words)))

    if not is_relevant and retry_count < 1:
        tokens = re.findall(r"\w+", query)
        broadened = " ".join(tokens[:3]) if tokens else "real estate"
        logger.info("Docs graded irrelevant. Broadening query to '%s' (retry 1)", broadened)
        return {
            "rewritten_query": broadened,
            "doc_type_filter": None,
            "retry_count": 1,
            "needs_retrieval_retry": True,
        }

    return {"needs_retrieval_retry": False}


# ---------------------------------------------------------------------------
# Node 5: Generate
# ---------------------------------------------------------------------------


def generate_node(state: GraphState) -> dict[str, Any]:
    """Generate answer strictly from retrieved docs with citations and original currencies."""
    docs = state.get("docs") or []
    question = state.get("question") or ""

    if state.get("retrieval_failed"):
        return {"answer": TRANSIENT_ERROR_ANSWER, "citations": [], "grounded": True}

    if not docs:
        return {
            "answer": SAFE_FALLBACK_ANSWER,
            "citations": [],
            "grounded": True,
        }

    # Format context blocks and collect unique citations
    context_blocks = []
    citations: list[str] = []

    for i, d in enumerate(docs, start=1):
        meta = d.get("metadata", {})
        url = meta.get("url", "")
        if url and url not in citations:
            citations.append(url)
        context_blocks.append(f"[Document {i} | Source URL: {url}]\n{d.get('text', '')}")

    context_str = "\n\n".join(context_blocks)
    prompt = GENERATE_PROMPT.format(system_prompt=SYSTEM_PROMPT, context=context_str, question=question)
    prompt += (
        "\n\nDo not invent requested facts or ranking criteria that are absent from the context. "
        "If part of the question cannot be answered but relevant property information is present, "
        "briefly state the limitation and answer the supported part. Use this fallback only when "
        f'no relevant part can be answered: "{SAFE_FALLBACK_ANSWER}". '
        "When the source gives a number of markets in response to a question about countries, report the sourced number as international markets; do not call it a number of countries."
    )

    try:
        answer = call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
        )
        if not answer:
            raise RuntimeError("OpenRouter returned an empty response.")
    except Exception as err:
        status_code = getattr(err, "status_code", None)
        response = getattr(err, "response", None)
        response_body = getattr(response, "text", None) if response is not None else None
        logger.exception(
            "Generation failed: status_code=%s response_body=%s",
            status_code,
            response_body,
        )
        answer = (
            "The free AI model is busy. Please try again in a few seconds."
            if status_code == 429
            else MODEL_UNAVAILABLE_ANSWER
        )

    if answer.strip() == "OFF_TOPIC":
        answer = TOPIC_REFUSAL

    return {
        "answer": _guard_output(answer),
        "citations": citations,
        "grounded": False,
    }


# ---------------------------------------------------------------------------
# Node 6: Verify Groundedness
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> set[str]:
    """Extract and normalize numeric values (removing commas)."""
    raw_nums = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)
    return {n.replace(",", "").strip() for n in raw_nums if n.strip()}


def verify_node(state: GraphState) -> dict[str, Any]:
    """Lightweight check ensuring answer is supported by the context."""
    answer = state.get("answer") or ""
    docs = state.get("docs") or []
    verify_retries = state.get("verify_retry_count", 0)

    if answer in (SAFE_FALLBACK_ANSWER, TRANSIENT_ERROR_ANSWER, MODEL_UNAVAILABLE_ANSWER,
                  CHITCHAT_RESPONSE, OUT_OF_SCOPE_RESPONSE, TOPIC_REFUSAL, PRICE_REFUSAL):
        return {"grounded": True, "answer": _guard_output(answer)}

    doc_text_combined = " ".join(d.get("text", "") for d in docs).lower()

    numbers_in_answer = _extract_numbers(answer)
    numbers_in_docs = _extract_numbers(doc_text_combined)

    unsupported_numbers = numbers_in_answer - numbers_in_docs
    unsupported_significant = []
    for n in unsupported_numbers:
        try:
            val = float(n)
            if val > 20:
                unsupported_significant.append(n)
        except ValueError:
            pass

    if not unsupported_significant:
        return {"grounded": True, "answer": _guard_output(answer)}

    logger.warning("Groundedness warning: unsupported numbers in answer: %s", unsupported_significant)
    if verify_retries < 1:
        try:
            raise RuntimeError(f"Groundedness check rejected unsupported values: {unsupported_significant}")
        except RuntimeError:
            logger.exception("Generated answer failed groundedness verification")
        return {
            "grounded": False,
            "answer": TRANSIENT_ERROR_ANSWER,
            "verify_retry_count": 1,
        }

    return {"grounded": True, "answer": _guard_output(answer)}


# ---------------------------------------------------------------------------
# Graph Construction & Conditional Routing
# ---------------------------------------------------------------------------


def route_classify(state: GraphState) -> str:
    """Route from classify node based on category."""
    route = state.get("route", "rag")
    if route == "chitchat":
        return "handle_chitchat"
    if route == "out_of_scope":
        return "handle_out_of_scope"
    if route == "direct_reply":
        return "direct_reply"
    return "rewrite"


def route_grade(state: GraphState) -> str:
    """Route from grade node: retry retrieval or proceed to generate."""
    if state.get("needs_retrieval_retry", False):
        return "retrieve"
    return "generate"


def create_rag_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("classify", classify_node)
    workflow.add_node("handle_chitchat", handle_chitchat_node)
    workflow.add_node("handle_out_of_scope", handle_out_of_scope_node)
    workflow.add_node("direct_reply", direct_reply_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges(
        "classify",
        route_classify,
        {
            "handle_chitchat": "handle_chitchat",
            "handle_out_of_scope": "handle_out_of_scope",
            "direct_reply": "direct_reply",
            "rewrite": "rewrite",
        },
    )

    workflow.add_edge("handle_chitchat", END)
    workflow.add_edge("handle_out_of_scope", END)
    workflow.add_edge("direct_reply", END)

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        route_grade,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)

    return workflow


app = create_rag_graph().compile()


def answer_question(
    question: str,
    history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Execute the compiled LangGraph workflow to answer a user question."""
    initial_state: GraphState = {
        "question": question,
        "history": history or [],
        "retry_count": 0,
        "verify_retry_count": 0,
    }
    final_state = app.invoke(initial_state)

    return {
        "answer": final_state.get("answer", SAFE_FALLBACK_ANSWER),
        "citations": final_state.get("citations", []),
        "route": final_state.get("route", "rag"),
        "retrieved_docs": [],
}


def main() -> None:
    """CLI entry point: python -m rag.graph 'What projects does Dar Global have in Dubai?'."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "What projects does Dar Global have in Dubai?"
    print(f"\nUser Query: {query}\n" + "-" * 50)

    result = answer_question(query)

    print(f"Route: {result['route']}")
    print(f"\nAnswer:\n{result['answer']}")
    if result["citations"]:
        print("\nCitations:")
        for url in result["citations"]:
            print(f"  - {url}")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    main()

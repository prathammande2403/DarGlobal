"""Unit tests for LangGraph RAG workflow, nodes, and conditional edges."""

import unittest
from unittest.mock import MagicMock, patch

from rag.graph import (
    GraphState,
    answer_question,
    classify_node,
    detect_document_type,
    generate_node,
    grade_node,
    handle_chitchat_node,
    handle_out_of_scope_node,
    rewrite_node,
    verify_node,
)
from rag.prompts import (
    CHITCHAT_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    SAFE_FALLBACK_ANSWER,
)


class TestRAGGraphNodes(unittest.TestCase):
    """Test suite for graph nodes, routing logic, and grounded answer synthesis."""

    def test_classify_node_heuristics(self) -> None:
        """Verify routing for greetings, out of scope questions, and real estate queries."""
        # Chitchat
        state_hi: GraphState = {"question": "Hello!"}
        self.assertEqual(classify_node(state_hi)["route"], "chitchat")

        state_how: GraphState = {"question": "how are you?"}
        self.assertEqual(classify_node(state_how)["route"], "chitchat")

        # RAG queries
        state_dg: GraphState = {"question": "What villas does DarGlobal offer in Dubai?"}
        self.assertEqual(classify_node(state_dg)["route"], "rag")

        state_ws: GraphState = {"question": "Find apartments in Dubai with 2 bedrooms"}
        self.assertEqual(classify_node(state_ws)["route"], "rag")

        # A pricing follow-up must stay in the DarGlobal RAG flow even when
        # it does not repeat a project or company name.
        state_pricing: GraphState = {"question": "what are their pricing"}
        self.assertEqual(classify_node(state_pricing)["route"], "rag")

        state_blog: GraphState = {
            "question": "what was that blog about"
        }
        self.assertEqual(classify_node(state_blog)["route"], "rag")
        self.assertEqual(detect_document_type(state_blog["question"]), "news")

        state_offices: GraphState = {"question": "Where are DarGlobal's offices located?"}
        self.assertEqual(classify_node(state_offices)["route"], "rag")
        self.assertEqual(detect_document_type(state_offices["question"]), "corporate")

    @patch("rag.graph.call_llm", return_value="out_of_scope")
    def test_classify_out_of_scope_with_llm(self, mock_llm: MagicMock) -> None:
        """Verify unrelated questions route to out_of_scope via LLM."""
        state: GraphState = {"question": "How do I reverse a binary tree in Python?"}
        res = classify_node(state)
        self.assertEqual(res["route"], "out_of_scope")

    def test_handle_chitchat_and_out_of_scope_nodes(self) -> None:
        """Verify static node handlers return correct polite messages and empty citations."""
        chitchat_res = handle_chitchat_node({})
        self.assertEqual(chitchat_res["answer"], CHITCHAT_RESPONSE)
        self.assertEqual(chitchat_res["citations"], [])
        self.assertTrue(chitchat_res["grounded"])

        out_res = handle_out_of_scope_node({})
        self.assertEqual(out_res["answer"], OUT_OF_SCOPE_RESPONSE)
        self.assertEqual(out_res["citations"], [])
        self.assertTrue(out_res["grounded"])

    def test_document_type_detection(self) -> None:
        """Verify intent keywords map to 'property_listing', 'corporate', or 'news'."""
        self.assertEqual(detect_document_type("list projects in Dubai"), "property_listing")
        self.assertEqual(detect_document_type("available apartments and villas"), "property_listing")
        self.assertEqual(detect_document_type("who is on the corporate leadership board"), "corporate")
        self.assertEqual(detect_document_type("latest news and press updates"), "news")
        self.assertIsNone(detect_document_type("hello there"))

    @patch("rag.graph.call_llm")
    def test_rewrite_with_history(self, mock_llm: MagicMock) -> None:
        """Verify conversational query rewriting uses history."""
        mock_llm.return_value = (
            "STANDALONE_QUERY: What are the starting prices for DaVinci Tower by DarGlobal in Dubai?"
        )
        history = [
            {"role": "user", "content": "Tell me about DaVinci Tower."},
            {"role": "assistant", "content": "DaVinci Tower is a luxury residential development in Dubai."},
        ]
        state: GraphState = {
            "question": "What are the starting prices?",
            "history": history,
        }
        res = rewrite_node(state)

        self.assertIn("DaVinci Tower", res["rewritten_query"])
        self.assertEqual(res["doc_type_filter"], "property_listing")

    def test_grade_node_retry_loop(self) -> None:
        """Verify irrelevant or empty retrieval triggers a single query broadening retry."""
        empty_state: GraphState = {
            "question": "Ultra rare castle in Riyadh",
            "docs": [],
            "retry_count": 0,
        }
        grade_res_1 = grade_node(empty_state)
        self.assertTrue(grade_res_1["needs_retrieval_retry"])
        self.assertEqual(grade_res_1["retry_count"], 1)

        # Second time (retry_count == 1), should not retry again
        exhausted_state: GraphState = {
            "question": "Ultra rare castle in Riyadh",
            "docs": [],
            "retry_count": 1,
        }
        grade_res_2 = grade_node(exhausted_state)
        self.assertFalse(grade_res_2["needs_retrieval_retry"])

    @patch("rag.graph.call_llm")
    def test_generate_node_with_citations(self, mock_llm: MagicMock) -> None:
        """Verify generation extracts URLs and queries LLM."""
        mock_llm.return_value = "Urban Oasis by Missoni offers apartments starting from 2,500,000 AED."
        sample_docs = [
            {
                "text": "[DarGlobal | project | Dubai] Urban Oasis\nPrice: 2,500,000 AED",
                "metadata": {"url": "https://darglobal.co.uk/urban-oasis"},
            }
        ]
        state: GraphState = {
            "question": "What is the price of Urban Oasis?",
            "docs": sample_docs,
        }
        gen_res = generate_node(state)

        self.assertIn("2,500,000 AED", gen_res["answer"])
        self.assertEqual(gen_res["citations"], ["https://darglobal.co.uk/urban-oasis"])

    def test_verify_node_groundedness(self) -> None:
        """Verify hallucinated numbers trigger fallback to safe answer."""
        docs = [
            {
                "text": "Villa in Riyadh priced at 3,500,000 SAR.",
                "metadata": {"url": "https://darglobal.co.uk/villa-1"},
            }
        ]
        # Grounded answer: matches number in docs
        state_ok: GraphState = {
            "answer": "The villa costs 3,500,000 SAR.",
            "docs": docs,
            "verify_retry_count": 0,
        }
        res_ok = verify_node(state_ok)
        self.assertTrue(res_ok["grounded"])

        # Ungrounded answer: contains invented number 99,000,000
        state_hallucinated: GraphState = {
            "answer": "The penthouse costs 99,000,000 SAR.",
            "docs": docs,
            "verify_retry_count": 0,
        }
        res_fail = verify_node(state_hallucinated)
        self.assertFalse(res_fail["grounded"])
        self.assertEqual(res_fail["answer"], SAFE_FALLBACK_ANSWER)

    @patch("rag.graph.retrieve")
    @patch("rag.graph.call_llm")
    def test_answer_question_end_to_end(
        self,
        mock_llm: MagicMock,
        mock_retrieve: MagicMock,
    ) -> None:
        """Verify answer_question returns the specified dict schema."""
        mock_retrieve.return_value = [
            {
                "text": "[DarGlobal | project | Dubai] Urban Oasis by Missoni\nLocation: Dubai, UAE\nPrice: 2,500,000 AED",
                "metadata": {
                    "source": "darglobal",
                    "url": "https://darglobal.co.uk/urban-oasis",
                    "location": "Dubai",
                },
                "score": 0.032,
            }
        ]
        mock_llm.return_value = "DarGlobal offers Urban Oasis in Dubai with prices from 2,500,000 AED."

        result = answer_question("What projects does Dar Global have in Dubai?")

        self.assertIn("answer", result)
        self.assertIn("citations", result)
        self.assertIn("route", result)
        self.assertIn("retrieved_docs", result)

        self.assertEqual(result["route"], "rag")
        self.assertIn("Urban Oasis", result["answer"])
        self.assertIn("https://darglobal.co.uk/urban-oasis", result["citations"])
        self.assertEqual(len(result["retrieved_docs"]), 1)


if __name__ == "__main__":
    unittest.main()

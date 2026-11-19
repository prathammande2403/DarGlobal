"""Unit tests for RAG data ingestion, chunking, and hybrid retrieval."""

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import chromadb
from fastembed import TextEmbedding

from rag.ingest import (
    build_chunks,
    chunk_record,
    format_record_body,
    format_record_header,
    ingest,
)
from rag.retriever import HybridRetriever, build_chroma_where, tokenize

# In-memory test fixtures representing DarGlobal properties
TINY_RECORD_FIXTURES: list[dict[str, Any]] = [
    {
        "source": "darglobal",
        "url": "https://darglobal.co.uk/projects/davinci-tower",
        "page_type": "project",
        "title": "DaVinci Tower London",
        "location": "London",
        "country": "United Kingdom",
        "property_type": "Apartment",
        "starting_price": "1,200,000",
        "currency": "GBP",
        "bedrooms": "1, 2 & 3",
        "amenities": ["Concierge", "Gym", "Private Cinema", "Rooftop Garden"],
        "description": "Ultra luxury residential tower in Vauxhall London designed with Pagani interiors.",
        "key_facts": {"Units": "160", "Completion": "2025"},
    },
    {
        "source": "darglobal",
        "url": "https://darglobal.co.uk/projects/urban-oasis",
        "page_type": "project",
        "title": "Urban Oasis by Missoni Dubai",
        "location": "Dubai",
        "country": "UAE",
        "property_type": "Apartment",
        "price": "2,500,000",
        "currency": "AED",
        "bedrooms": "2",
        "amenities": ["Infinity Pool", "Waterfront Promenade", "Spa"],
        "description": "Waterfront residential luxury tower located along Dubai Water Canal.",
        "key_facts": {"Floors": "34", "Developer": "DarGlobal"},
    },
    {
        "source": "darglobal",
        "url": "https://darglobal.co.uk/projects/tierra-viva",
        "page_type": "project",
        "title": "Tierra Viva Lamborghini Villas",
        "location": "Benahavis",
        "country": "Spain",
        "property_type": "Villa",
        "price": "4,500,000",
        "currency": "EUR",
        "bedrooms": "5",
        "amenities": ["Private Pool", "Panoramic Sea Views", "Gated Community"],
        "description": "Exclusive luxury villas in Costa del Sol inspired by Automobili Lamborghini design.",
        "key_facts": {"Units": "53"},
    },
    {
        "source": "darglobal",
        "url": "https://darglobal.co.uk/about",
        "page_type": "about",
        "title": "About DarGlobal",
        "location": "London",
        "country": "United Kingdom",
        "description": "Dar Global PLC is an international luxury real estate developer listed on the London Stock Exchange.",
        "key_facts": {"Headquarters": "London, UK", "LSE Ticker": "DAR"},
    },
]


class TestIngestAndChunking(unittest.TestCase):
    """Test chunk formatting, metadata preservation, and ingestion idempotence."""

    def test_header_line_formatting(self) -> None:
        """Verify chunk header format: '[Source | page_type | Location] Title'."""
        rec_darglobal = TINY_RECORD_FIXTURES[0]
        header_dg = format_record_header(rec_darglobal)
        self.assertEqual(header_dg, "[DarGlobal | project | London] DaVinci Tower London")

        rec_tv = TINY_RECORD_FIXTURES[2]
        header_tv = format_record_header(rec_tv)
        self.assertEqual(header_tv, "[DarGlobal | project | Benahavis] Tierra Viva Lamborghini Villas")

    def test_chunking_metadata_retention(self) -> None:
        """Verify chunking preserves source, url, page_type, document_type, location, property_type, and price."""
        rec = TINY_RECORD_FIXTURES[2]
        chunks = chunk_record(rec)

        self.assertGreaterEqual(len(chunks), 1)
        first_chunk = chunks[0]
        meta = first_chunk["metadata"]

        self.assertEqual(meta["source"], "darglobal")
        self.assertEqual(meta["url"], "https://darglobal.co.uk/projects/tierra-viva")
        self.assertEqual(meta["page_type"], "project")
        self.assertEqual(meta["document_type"], "property_listing")
        self.assertEqual(meta["location"], "Benahavis")
        self.assertEqual(meta["property_type"], "Villa")
        self.assertEqual(meta["price"], "4,500,000")
        self.assertIn("Tierra Viva Lamborghini Villas", first_chunk["text"])

    def test_long_document_multi_chunking(self) -> None:
        """Verify documents exceeding word limits are partitioned into multiple chunks."""
        long_rec = dict(TINY_RECORD_FIXTURES[0])
        long_rec["description"] = " ".join(["word" + str(i) for i in range(800)])
        chunks = chunk_record(long_rec, max_words_per_chunk=300, overlap_words=50)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c["text"].startswith("[DarGlobal | project | London]"))

    def test_idempotent_ingestion(self) -> None:
        """Verify repeated ingestion produces identical chunk counts without duplicates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_file = tmp_path / "test_data.json"
            chroma_dir = tmp_path / "chroma"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(TINY_RECORD_FIXTURES, f)

            client = chromadb.EphemeralClient()
            embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

            count1 = ingest(
                input_file=json_file,
                chroma_dir=chroma_dir,
                client=client,
                embedding_model=embedder,
            )
            self.assertEqual(count1, 4)

            count2 = ingest(
                input_file=json_file,
                chroma_dir=chroma_dir,
                client=client,
                embedding_model=embedder,
            )
            self.assertEqual(count2, 4)


class TestHybridRetriever(unittest.TestCase):
    """Test suite for HybridRetriever vector, BM25, and Reciprocal Rank Fusion."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build in-memory Chroma collection and shared embedder once for test suite."""
        cls.embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
        cls.client = chromadb.EphemeralClient()
        cls.collection_name = "test_hybrid_retriever_collection"

        # Ingest test fixtures
        chunks = build_chunks(TINY_RECORD_FIXTURES)
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        raw_emb = list(cls.embedder.embed(texts))
        embs = [v.tolist() for v in raw_emb]

        cls.collection = cls.client.create_collection(
            name=cls.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        cls.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embs,
            metadatas=metas,
        )

        cls.retriever = HybridRetriever(
            collection_name=cls.collection_name,
            client=cls.client,
            embedder=cls.embedder,
        )

    def test_tokenization(self) -> None:
        """Verify BM25 tokenization extracts lowercase alphanumeric tokens."""
        tokens = tokenize("Luxury Villa, 5-Bedrooms @ Dubai Water Canal!")
        self.assertIn("luxury", tokens)
        self.assertIn("villa", tokens)
        self.assertIn("5", tokens)
        self.assertIn("bedrooms", tokens)
        self.assertIn("canal", tokens)

    def test_hybrid_search_accuracy(self) -> None:
        """Verify relevant documents are returned for a specific property query."""
        results = self.retriever.retrieve(query="Pagani tower in London", k=3)

        self.assertGreaterEqual(len(results), 1)
        top_doc = results[0]
        self.assertIn("DaVinci Tower", top_doc["text"])
        self.assertEqual(top_doc["metadata"]["source"], "darglobal")

    def test_hybrid_search_waterfront_dubai(self) -> None:
        """Verify Dubai waterfront query retrieves Urban Oasis."""
        results = self.retriever.retrieve(query="Dubai Water Canal luxury apartment", k=3)

        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertIn("Urban Oasis", top_match["text"])

    def test_filter_by_source_darglobal(self) -> None:
        """Verify filtering by source='darglobal' returns only DarGlobal chunks."""
        results = self.retriever.retrieve(query="apartment", k=5, source="darglobal")

        self.assertGreaterEqual(len(results), 1)
        for item in results:
            self.assertEqual(item["metadata"]["source"], "darglobal")

    def test_filter_by_document_type(self) -> None:
        """Verify filtering by filters={'document_type': ...} returns matching chunks."""
        results = self.retriever.retrieve(
            query="luxury",
            k=5,
            filters={"document_type": "property_listing"},
        )

        self.assertGreaterEqual(len(results), 1)
        for item in results:
            self.assertEqual(item["metadata"]["document_type"], "property_listing")

    def test_custom_metadata_filters(self) -> None:
        """Verify custom filters (e.g. property_type='Villa')."""
        results = self.retriever.retrieve(
            query="luxury residential property",
            k=5,
            filters={"property_type": "villa"},
        )

        self.assertGreaterEqual(len(results), 1)
        for item in results:
            self.assertEqual(item["metadata"]["property_type"].lower(), "villa")

    def test_chroma_where_builder(self) -> None:
        """Verify Chroma where clause generation."""
        where_src = build_chroma_where(source="darglobal")
        self.assertEqual(where_src, {"source": "darglobal"})

        where_multi = build_chroma_where(source="darglobal", filters={"document_type": "property_listing"})
        self.assertIn("$and", where_multi)
        self.assertEqual(len(where_multi["$and"]), 2)

        self.assertIsNone(build_chroma_where())

    def test_empty_query_or_corpus_handling(self) -> None:
        """Verify empty query returns empty list without error."""
        results = self.retriever.retrieve(query="")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()

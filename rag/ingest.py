"""Data ingestion and chunking pipeline for RAG.

Loads data/processed/combined.json, formats and chunks records into structured
text passages with metadata headers (e.g. "[DarGlobal | project | London] Title"),
generates local dense embeddings (fastembed BAAI/bge-small-en-v1.5), and stores
them idempotently into ChromaDB at data/chroma/.
"""

from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from fastembed import TextEmbedding
logger = logging.getLogger(__name__)

DEFAULT_INPUT_FILE = Path("data/raw/darglobal.json")
DEFAULT_CHROMA_DIR = Path("data/chroma_updated")
DEFAULT_COLLECTION_NAME = "real_estate_collection"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
def format_record_header(record: dict[str, Any]) -> str:
    """Construct standard chunk header: '[Source | page_type | Location] Title'."""
    raw_source = record.get("source", "DarGlobal")
    source_title = "DarGlobal" if raw_source.lower() == "darglobal" else raw_source.capitalize()

    page_type = record.get("page_type", "other")

    # Resolve location
    location = (
        record.get("location")
        or record.get("city")
        or record.get("district")
        or record.get("country")
        or "Global"
    )

    title = record.get("title") or "Real Estate Document"
    return f"[{source_title} | {page_type} | {location}] {title}"


def format_record_body(record: dict[str, Any]) -> str:
    """Format property details, amenities, specifications, and descriptions into text."""
    parts: list[str] = []

    # Location specs
    loc_parts = []
    if record.get("location"):
        loc_parts.append(str(record["location"]))
    if record.get("district"):
        loc_parts.append(f"District: {record['district']}")
    if record.get("city") and record.get("city") != record.get("location"):
        loc_parts.append(f"City: {record['city']}")
    if record.get("country"):
        loc_parts.append(f"Country: {record['country']}")
    if loc_parts:
        parts.append("Location: " + ", ".join(loc_parts))

    # Core property attributes
    if record.get("property_type"):
        parts.append(f"Property Type: {record['property_type']}")

    # Preserve explicitly extracted starting prices in retrieval text. The
    # scraper only writes this field when it sees a price cue and a plausible
    # amount; currency is kept exactly as published.
    price = record.get("starting_price") or record.get("price")
    currency = record.get("currency")
    if price:
        parts.append(f"Starting Price: {price}{f' {currency}' if currency else ''}")

    specs = []
    if record.get("bedrooms"):
        specs.append(f"Bedrooms: {record['bedrooms']}")
    if record.get("bathrooms"):
        specs.append(f"Bathrooms: {record['bathrooms']}")
    if record.get("area_sqm"):
        specs.append(f"Area: {record['area_sqm']} sqm")
    if specs:
        parts.append(" | ".join(specs))

    # Status & Dates
    if record.get("article_date"):
        parts.append(f"Publication Date: {record['article_date']}")

    timing = []
    if record.get("status"):
        timing.append(f"Status: {record['status']}")
    if record.get("completion_date"):
        timing.append(f"Completion: {record['completion_date']}")
    if record.get("developer"):
        timing.append(f"Developer: {record['developer']}")
    if timing:
        parts.append(" | ".join(timing))

    # Amenities
    amenities = record.get("amenities") or []
    if isinstance(amenities, list) and amenities:
        parts.append("Amenities: " + ", ".join(str(a) for a in amenities))

    # Key Facts
    key_facts = record.get("key_facts") or {}
    if isinstance(key_facts, dict) and key_facts:
        facts_lines = [
            f"  - {k}: {v}"
            for k, v in key_facts.items()
            if v
        ]
        if facts_lines:
            parts.append("Key Facts:\n" + "\n".join(facts_lines))

    # Description (English)
    desc = record.get("description")
    if desc:
        parts.append(f"Description:\n{desc}")

    # Description (Arabic)
    desc_ar = record.get("description_ar")
    if desc_ar:
        parts.append(f"Arabic Description:\n{desc_ar}")

    return "\n\n".join(parts)


def chunk_record(
    record: dict[str, Any],
    max_words_per_chunk: int = 350,
    overlap_words: int = 50,
) -> list[dict[str, Any]]:
    """Convert a single scraped record into one or more structured text chunks.

    Chunks start with the required header line and retain metadata.
    """
    header = format_record_header(record)
    body = format_record_body(record)
    full_text = f"{header}\n\n{body}".strip()

    page_type = str(record.get("page_type", "other")).lower()
    if page_type == "project":
        doc_type = "property_listing"
    elif page_type == "about":
        doc_type = "corporate"
    elif page_type == "news":
        doc_type = "news"
    else:
        doc_type = "other"

    # Base metadata matching constraints
    base_metadata: dict[str, Any] = {
        "source": str(record.get("source", "unknown")).lower(),
        "url": str(record.get("url", "")),
        "page_type": page_type,
        "document_type": doc_type,
        "location": str(
            record.get("location")
            or record.get("city")
            or record.get("district")
            or record.get("country")
            or ""
        ),
        "property_type": str(record.get("property_type") or ""),
        "title": str(record.get("title") or ""),
        "price": str(record.get("starting_price") or record.get("price") or ""),
        "currency": str(record.get("currency") or ""),
        "article_date": str(record.get("article_date") or ""),
    }

    words = body.split()
    # If body is within single chunk limit, return 1 chunk
    if len(words) <= max_words_per_chunk:
        chunk_id = hashlib.sha256(f"{base_metadata['url']}#0".encode("utf-8")).hexdigest()[:16]
        meta = dict(base_metadata)
        meta["chunk_id"] = chunk_id
        meta["chunk_index"] = 0
        return [{"chunk_id": chunk_id, "text": full_text, "metadata": meta}]

    # Sliding window chunking for long documents
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    step = max_words_per_chunk - overlap_words

    while start < len(words):
        end = min(start + max_words_per_chunk, len(words))
        chunk_words = words[start:end]
        sub_body = " ".join(chunk_words)
        chunk_text = f"{header}\n\n{sub_body}"

        chunk_id = hashlib.sha256(f"{base_metadata['url']}#{idx}".encode("utf-8")).hexdigest()[:16]
        meta = dict(base_metadata)
        meta["chunk_id"] = chunk_id
        meta["chunk_index"] = idx

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": meta,
        })
        idx += 1
        start += step

    return chunks


def build_chunks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform records into chunks while preserving source content."""
    all_chunks: list[dict[str, Any]] = []
    for rec in records:
        all_chunks.extend(chunk_record(rec))
    return all_chunks


def ingest(
    input_file: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_model: Optional[Any] = None,
    client: Optional[chromadb.ClientAPI] = None,
) -> int:
    """Ingest processed records, generate local embeddings, and populate ChromaDB idempotently.

    Args:
        input_file: Path to combined.json (default: data/processed/combined.json).
        chroma_dir: Path to Chroma storage directory (default: data/chroma).
        collection_name: Target collection name in Chroma.
        model_name: Fastembed local model identifier (default: BAAI/bge-small-en-v1.5).
        embedding_model: Optional pre-loaded embedding model instance.
        client: Optional pre-configured Chroma client (e.g. for testing).

    Returns:
        Number of chunks stored in Chroma.
    """
    in_path = input_file or DEFAULT_INPUT_FILE
    db_path = chroma_dir or DEFAULT_CHROMA_DIR

    # 1. Load records from input file
    records: list[dict[str, Any]] = []
    if in_path.exists():
        with open(in_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        # Fallback check raw files if combined doesn't exist yet
        raw_darglobal = Path("data/raw/darglobal.json")
        if raw_darglobal.exists():
            with open(raw_darglobal, "r", encoding="utf-8") as f:
                records.extend(json.load(f))

    if not records:
        logger.warning("No records found to ingest at %s", in_path)
        return 0

    logger.info("Loaded %d records for chunking and ingestion", len(records))

    # 2. Chunk records
    chunks = build_chunks(records)
    if not chunks:
        logger.warning("No chunks generated from %d records", len(records))
        return 0

    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    chunk_metadatas = [c["metadata"] for c in chunks]

    # 3. Generate embeddings locally
    logger.info("Embedding %d chunks with model %s...", len(chunks), model_name)
    configured_threads = os.getenv("FASTEMBED_THREADS")
    embedder = embedding_model or TextEmbedding(
        model_name=model_name,
        threads=int(configured_threads) if configured_threads else None,
    )
    raw_embeddings = list(embedder.embed(chunk_texts))
    embeddings = [v.tolist() for v in raw_embeddings]

    # 4. Store in ChromaDB idempotently (rebuild collection on rerun)
    db_path.mkdir(parents=True, exist_ok=True)
    chroma_client = client or chromadb.PersistentClient(path=str(db_path))

    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info("Deleted existing Chroma collection '%s' for clean rebuild", collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch add to collection
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        end = i + batch_size
        collection.add(
            ids=chunk_ids[i:end],
            documents=chunk_texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=chunk_metadatas[i:end],
        )

    # Also persist chunks cache for rapid BM25 index initialization
    chunks_cache_file = db_path / "chunks.json"
    with open(chunks_cache_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Count per document_type
    type_counts = Counter(c.get("metadata", {}).get("document_type", "other") for c in chunks)
    print("Chunks count per document_type:")
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count}")

    logger.info(
        "Successfully ingested %d chunks into Chroma collection '%s' at %s",
        len(chunks),
        collection_name,
        db_path,
    )
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    ingest()

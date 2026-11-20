"""Small deterministic guards for user questions and generated text."""

import re


# Currency/price references are intentionally broad: a sentence mentioning any
# of these is excluded from retrieval text and generated answers.
PRICE_PATTERN = re.compile(
    r"(?:\b(?:AED|USD|GBP|OMR|SAR|EUR)\b\s*\d[\d,.]*(?:\s*(?:million|mn))?"
    r"|\d[\d,.]*(?:\s*(?:million|mn))?\s*\b(?:AED|USD|GBP|OMR|SAR|EUR)\b"
    r"|[$£€]\s*\d[\d,.]*(?:\s*(?:million|mn))?"
    r"|\d[\d,.]*(?:\s*(?:million|mn))?\s*[$£€]"
    r"|\b\d[\d,.]*\s*(?:million|mn)\b"
    r"|\b(?:AED|USD|GBP|OMR|SAR|EUR)\b"
    r"|\b(?:prices?|pricing|costs?|starting\s+from|price\s+from|per\s+sq\.?\s*ft)\b)",
    re.IGNORECASE,
)
GDV_METRIC_PATTERN = re.compile(r"\b(?:gross\s+development\s+value|GDV)\b", re.IGNORECASE)
PROPERTY_PRICE_CONTEXT_PATTERN = re.compile(
    r"\b(?:price|pricing|cost|payment\s+plan|starting\s+from|per\s+sq\.?\s*ft)\b",
    re.IGNORECASE,
)

PRICE_QUESTION_TERMS = (
    "price", "cost", "how much", "budget", "payment plan", "installment",
    "starting from", "afford", "rate", "roi",
)
ON_TOPIC_TERMS = (
    "dar global", "darglobal", "property", "properties", "villa", "apartment",
    "residence", "project", "location", "dubai", "muscat", "oman", "maldives",
    "london", "spain", "marbella", "qatar", "saudi", "riyadh", "jeddah",
    "amenities", "pool", "bedrooms", "status", "completion", "sold out",
    "launch", "developer", "company", "news", "about", "price", "prices",
    "pricing", "cost", "payment plan", "installment", "starting price",
    "blog", "blogs", "press", "article", "articles", "announcement", "announcements",
    "office", "offices", "headquarters", "head office",
    "hello", "hi", "thanks", "how are you",
    # Company footprint questions are in scope, even when phrased without a
    # DarGlobal/property keyword (for example, "how many countries do you operate in?").
    "country", "countries", "operate", "operates", "operating", "market", "markets",
    "international markets", "global presence", "footprint",
)
INJECTION_TERMS = ("ignore previous", "system prompt", "act as", "reveal instructions")
TOPIC_SCORE_MIN = 0.01


def strip_price_sentences(text: str) -> str:
    """Remove each sentence/line containing a price cue."""
    kept: list[str] = []
    for line in text.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for sentence in sentences:
            if not sentence.strip():
                continue
            if _is_company_gdv_metric(sentence) or not PRICE_PATTERN.search(sentence):
                kept.append(sentence)
    return "\n".join(kept).strip()


def contains_removable_price_sentence(text: str) -> bool:
    """Return true when text contains a price sentence that the guard removes."""
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if PRICE_PATTERN.search(sentence) and not _is_company_gdv_metric(sentence):
                return True
    return False


def _is_company_gdv_metric(sentence: str) -> bool:
    return bool(GDV_METRIC_PATTERN.search(sentence)) and not PROPERTY_PRICE_CONTEXT_PATTERN.search(sentence)


def is_price_question(q: str) -> bool:
    lowered = q.casefold()
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lowered) for term in PRICE_QUESTION_TERMS)


def is_on_topic(q: str) -> bool:
    lowered = q.casefold()
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lowered) for term in ON_TOPIC_TERMS)


def is_prompt_injection(q: str) -> bool:
    lowered = q.casefold()
    return any(term in lowered for term in INJECTION_TERMS)

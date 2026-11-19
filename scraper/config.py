"""Configuration settings for web scrapers and polite HTTP client."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class ScraperConfig:
    """Dataclass holding scraping limits, delays, headers, and paths.

    Attributes:
        dar_global_base_url: Base URL for DarGlobal scraping.
        user_agent_email: Contact email embedded in polite User-Agent string.
        user_agent_bot_name: Bot name token.
        min_crawl_delay: Minimum mandatory crawl delay in seconds between domain requests.
        jitter_min: Minimum random jitter in seconds added to crawl delay.
        jitter_max: Maximum random jitter in seconds added to crawl delay.
        request_timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of retries for 429 or 5xx responses.
        backoff_factor: Multiplier for exponential backoff calculations.
        max_pages_per_run: Global cap on pages fetched across domains per run.
        max_pages_per_domain: Cap on pages fetched for an individual domain.
        cache_dir: Directory where raw response payloads are cached.
        openrouter_api_key: Optional API key for OpenRouter models.
        openrouter_model: Free model identifier for OpenRouter RAG.
    """

    dar_global_base_url: str = field(
        default_factory=lambda: os.getenv("DARGLOBAL_BASE_URL", "https://darglobal.co.uk")
    )

    user_agent_email: str = field(
        default_factory=lambda: os.getenv("USER_AGENT_EMAIL", "student@example.com")
    )
    user_agent_bot_name: str = field(
        default_factory=lambda: os.getenv("USER_AGENT_BOT_NAME", "AssignmentResearchBot/1.0")
    )

    min_crawl_delay: float = field(
        default_factory=lambda: float(os.getenv("MIN_CRAWL_DELAY", "2.0"))
    )
    jitter_min: float = field(
        default_factory=lambda: float(os.getenv("JITTER_MIN", "0.5"))
    )
    jitter_max: float = field(
        default_factory=lambda: float(os.getenv("JITTER_MAX", "1.5"))
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "20.0"))
    )

    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    backoff_factor: float = field(
        default_factory=lambda: float(os.getenv("BACKOFF_FACTOR", "2.0"))
    )

    max_pages_per_run: int = field(
        default_factory=lambda: int(os.getenv("MAX_PAGES_PER_RUN", "150"))
    )
    max_pages_per_domain: int = field(
        default_factory=lambda: int(os.getenv("MAX_PAGES_PER_DOMAIN", "100"))
    )

    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CACHE_DIR", "data/raw/cache"))
    )

    openrouter_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite:free")
    )

    @property
    def user_agent(self) -> str:
        """Construct full polite User-Agent string."""
        return f"{self.user_agent_bot_name} (contact: {self.user_agent_email})"


config = ScraperConfig()

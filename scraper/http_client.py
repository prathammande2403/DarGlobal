"""Polite HTTP client adhering to robots.txt, rate limits, caching, and anti-bot policies."""

from collections import defaultdict
import datetime
import email.utils
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Optional
import urllib.parse

import requests

from scraper.config import ScraperConfig, config as default_config
from scraper.robots import can_fetch, crawl_delay

logger = logging.getLogger(__name__)


class PoliteHttpClient:
    """A responsible HTTP client for ethical web scraping.

    Enforces robots.txt checks, per-domain delays with jitter, exponential backoff
    with Retry-After compliance on 429/5xx, disk response caching, page caps,
    and immediate termination on anti-bot/captcha/403 pages without attempting bypasses.
    """

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize the polite HTTP client.

        Args:
            config: ScraperConfig dataclass instance. Uses default config if None.
            session: Optional custom requests.Session (e.g. for testing).
        """
        self.config: ScraperConfig = config or default_config
        self.session: requests.Session = session or requests.Session()

        # Configure default session headers
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
            }
        )

        self.cache_dir: Path = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_time: dict[str, float] = {}
        self.pages_fetched_total: int = 0
        self.pages_fetched_per_domain: dict[str, int] = defaultdict(int)

    def _get_domain(self, url: str) -> str:
        """Extract netloc domain from a URL."""
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lower()

    def _get_cache_path(self, url: str) -> Path:
        """Compute the cache file path for a URL hash."""
        url_hash = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
        return self.cache_dir / f"{url_hash}.json"

    def _load_from_cache(self, url: str, cache_path: Path) -> Optional[requests.Response]:
        """Load and reconstruct a requests.Response from cache."""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            response = requests.Response()
            response.status_code = data["status_code"]
            response.url = data.get("url", url)
            response.headers.update(data.get("headers", {}))
            response.encoding = data.get("encoding", "utf-8")
            response._content = data.get("text", "").encode(response.encoding or "utf-8")
            response.from_cache = True  # type: ignore[attr-defined]
            return response
        except Exception as err:
            logger.warning("Failed to read cache for %s (%s); ignoring cache", url, err)
            return None

    def _save_to_cache(self, url: str, response: requests.Response) -> None:
        """Persist successful raw response to cache."""
        cache_path = self._get_cache_path(url)
        try:
            payload = {
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "encoding": response.encoding or "utf-8",
                "text": response.text,
                "cached_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as err:
            logger.warning("Failed to write cache for %s: %s", url, err)

    def _enforce_delay(self, domain: str) -> None:
        """Enforce domain-specific delay of max(crawl-delay, 2s) plus jitter."""
        declared_delay = crawl_delay(domain, user_agent=self.config.user_agent_bot_name)
        base_delay = max(declared_delay or 0.0, self.config.min_crawl_delay)
        jitter = random.uniform(self.config.jitter_min, self.config.jitter_max)
        required_delay = base_delay + jitter

        last_time = self._last_request_time.get(domain)
        if last_time is not None:
            elapsed = time.time() - last_time
            if elapsed < required_delay:
                sleep_duration = required_delay - elapsed
                logger.debug(
                    "Polite delay: sleeping %.2fs for domain %s (base: %.1fs, jitter: %.2fs)",
                    sleep_duration,
                    domain,
                    base_delay,
                    jitter,
                )
                time.sleep(sleep_duration)

        self._last_request_time[domain] = time.time()

    def _is_blocked_or_captcha(self, response: requests.Response) -> bool:
        """Detect 403 Forbidden or captcha / anti-bot challenge pages."""
        if response.status_code == 403:
            return True

        content_preview = response.text[:4096].lower()
        anti_bot_signatures = (
            "cf-mitigated",
            "cf-browser-verification",
            "cf-turnstile",
            "challenge-platform",
            "hcaptcha",
            "g-recaptcha",
            "perimeterx",
            "datadome",
            "access denied",
            "<title>just a moment...</title>",
            "attention required! | cloudflare",
            "security check to access",
            "checking your browser",
        )
        return any(sig in content_preview for sig in anti_bot_signatures)

    def _parse_retry_after(self, retry_after: Optional[str]) -> Optional[float]:
        """Parse Retry-After header as either seconds or HTTP date."""
        if not retry_after:
            return None
        clean_val = retry_after.strip()
        try:
            return float(clean_val)
        except ValueError:
            pass
        try:
            dt = email.utils.parsedate_to_datetime(clean_val)
            now = datetime.datetime.now(datetime.timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None

    def get(self, url: str, **kwargs: Any) -> Optional[requests.Response]:
        """Perform a polite HTTP GET request.

        Args:
            url: Absolute URL to fetch.
            **kwargs: Extra parameters passed to requests.Session.get.

        Returns:
            A requests.Response object, or None if skipped, blocked, or failed.
        """
        # 1. robots.txt check before EVERY request
        if not can_fetch(url, user_agent=self.config.user_agent_bot_name):
            logger.warning("Robots.txt disallows crawling: %s. Skipping URL.", url)
            return None

        # 2. Check disk cache
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            cached_resp = self._load_from_cache(url, cache_path)
            if cached_resp is not None:
                return cached_resp

        # 3. Check page caps
        domain = self._get_domain(url)
        if self.pages_fetched_total >= self.config.max_pages_per_run:
            logger.warning(
                "Global cap reached (%d pages). Skipping %s",
                self.config.max_pages_per_run,
                url,
            )
            return None

        if self.pages_fetched_per_domain[domain] >= self.config.max_pages_per_domain:
            logger.warning(
                "Domain cap reached for %s (%d pages). Skipping %s",
                domain,
                self.config.max_pages_per_domain,
                url,
            )
            return None

        # 4. Request loop with exponential backoff on 429/5xx and immediate stop on 403/captcha
        timeout = kwargs.pop("timeout", self.config.request_timeout)

        for attempt in range(self.config.max_retries + 1):
            self._enforce_delay(domain)

            try:
                response = self.session.get(url, timeout=timeout, **kwargs)
            except (requests.Timeout, requests.ConnectionError) as req_err:
                if attempt >= self.config.max_retries:
                    logger.error("Request failed after %d retries for %s: %s", attempt, url, req_err)
                    return None
                backoff_wait = (self.config.backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    "Network error fetching %s; retrying in %.2fs (attempt %d/%d)",
                    url,
                    backoff_wait,
                    attempt + 1,
                    self.config.max_retries,
                )
                time.sleep(backoff_wait)
                continue

            # Anti-bot / Captcha / 403: Stop immediately, never bypass
            if self._is_blocked_or_captcha(response):
                logger.error(
                    "Detected 403 or anti-bot challenge on %s. Stopping request without bypass.",
                    url,
                )
                return None

            # 429 or 5xx: Retry with exponential backoff or honour Retry-After
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt >= self.config.max_retries:
                    logger.warning(
                        "Max retries (%d) exceeded for %s with HTTP %d",
                        self.config.max_retries,
                        url,
                        response.status_code,
                    )
                    return None

                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                if retry_after is not None and retry_after > 0:
                    sleep_seconds = retry_after
                    logger.info(
                        "HTTP %d on %s; honoring Retry-After: %.2fs",
                        response.status_code,
                        url,
                        sleep_seconds,
                    )
                else:
                    sleep_seconds = self.config.backoff_factor ** (attempt + 1)
                    logger.info(
                        "HTTP %d on %s; exponential backoff: %.2fs (attempt %d/%d)",
                        response.status_code,
                        url,
                        sleep_seconds,
                        attempt + 1,
                        self.config.max_retries,
                    )

                time.sleep(sleep_seconds)
                continue

            # Valid response
            if response.status_code == 200:
                self.pages_fetched_total += 1
                self.pages_fetched_per_domain[domain] += 1
                self._save_to_cache(url, response)
                response.from_cache = False  # type: ignore[attr-defined]
                return response

            # Non-200, non-retryable response (e.g. 404)
            logger.info("Received HTTP %d for %s", response.status_code, url)
            response.from_cache = False  # type: ignore[attr-defined]
            return response

        return None

    def close(self) -> None:
        """Close underlying session."""
        self.session.close()

    def __enter__(self) -> "PoliteHttpClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Alias for convenience
HttpClient = PoliteHttpClient

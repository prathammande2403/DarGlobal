"""Robots.txt parser and polite crawl policy checker."""

import logging
from typing import Optional
import urllib.parse
import urllib.robotparser

import requests

from scraper.config import config

logger = logging.getLogger(__name__)

# In-memory cache for parsed robots.txt per domain (e.g., "darglobal.co.uk": RobotFileParser)
_ROBOTS_CACHE: dict[str, urllib.robotparser.RobotFileParser] = {}


def _normalize_domain(url_or_domain: str) -> tuple[str, str]:
    """Extract normalized (scheme, domain) from a URL or domain string.

    Args:
        url_or_domain: A full URL or domain name.

    Returns:
        A tuple of (scheme, domain_name).
    """
    clean_target = url_or_domain.strip()
    if not clean_target.startswith(("http://", "https://")):
        clean_target = f"https://{clean_target}"
    parsed = urllib.parse.urlparse(clean_target)
    scheme = parsed.scheme or "https"
    domain = parsed.netloc.lower()
    return scheme, domain


def get_parser(
    domain_or_url: str,
    user_agent: Optional[str] = None,
    force_refresh: bool = False,
) -> urllib.robotparser.RobotFileParser:
    """Retrieve or fetch-and-cache a RobotFileParser for a domain.

    Args:
        domain_or_url: The domain or full URL to inspect.
        user_agent: Custom User-Agent header for fetching robots.txt.
        force_refresh: If True, bypass the cache and re-fetch robots.txt.

    Returns:
        A populated urllib.robotparser.RobotFileParser instance.
    """
    scheme, domain = _normalize_domain(domain_or_url)
    if not force_refresh and domain in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[domain]

    parser = urllib.robotparser.RobotFileParser()
    robots_url = f"{scheme}://{domain}/robots.txt"
    parser.set_url(robots_url)

    active_user_agent = user_agent or config.user_agent
    headers = {"User-Agent": active_user_agent}

    try:
        response = requests.get(
            robots_url,
            headers=headers,
            timeout=config.request_timeout,
        )
        if response.status_code == 200:
            parser.parse(response.text.splitlines())
            logger.info("Successfully fetched and parsed robots.txt for %s", domain)
        elif response.status_code in (401, 403):
            # Explicitly forbidden or unauthorized means disallow all
            parser.disallow_all = True
            logger.warning(
                "Received HTTP %d for %s; disallowing all paths",
                response.status_code,
                robots_url,
            )
        elif response.status_code == 404:
            # Not found means no restrictions
            parser.allow_all = True
            logger.info("No robots.txt found for %s (HTTP 404); allowing all paths", domain)
        else:
            # Other 4xx or 5xx: assume allowed or default policy
            parser.allow_all = True
            logger.warning(
                "Unexpected HTTP status %d fetching %s; defaulting to allow_all",
                response.status_code,
                robots_url,
            )
    except requests.RequestException as err:
        logger.warning(
            "Network error fetching robots.txt for %s: %s; defaulting to allow_all",
            domain,
            err,
        )
        parser.allow_all = True

    _ROBOTS_CACHE[domain] = parser
    return parser


def can_fetch(url: str, user_agent: Optional[str] = None) -> bool:
    """Check whether a URL can be fetched by the specified or configured User-Agent.

    Args:
        url: The absolute target URL to check.
        user_agent: User-Agent name or string. Defaults to configured bot name.

    Returns:
        True if allowed to crawl, False otherwise.
    """
    if not url:
        return False

    agent = user_agent or config.user_agent_bot_name
    parser = get_parser(url, user_agent=config.user_agent)
    return parser.can_fetch(agent, url)


def crawl_delay(domain: str, user_agent: Optional[str] = None) -> Optional[float]:
    """Retrieve the declared crawl delay (in seconds) for a given domain and agent.

    Args:
        domain: Target domain or URL.
        user_agent: User-Agent name or string. Defaults to configured bot name.

    Returns:
        The crawl delay in seconds, or None if unspecified.
    """
    agent = user_agent or config.user_agent_bot_name
    parser = get_parser(domain, user_agent=config.user_agent)

    # Check for agent-specific crawl-delay first
    delay = parser.crawl_delay(agent)
    if delay is None:
        # Fall back to wildcard user-agent crawl-delay
        delay = parser.crawl_delay("*")

    return float(delay) if delay is not None else None


def clear_cache() -> None:
    """Clear cached RobotFileParser instances. Useful for testing."""
    _ROBOTS_CACHE.clear()

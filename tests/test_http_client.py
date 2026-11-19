"""Unit tests for PoliteHttpClient and scraping foundation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from scraper.config import ScraperConfig
from scraper.http_client import PoliteHttpClient
from scraper.robots import can_fetch, clear_cache, crawl_delay


class TestHttpClient(unittest.TestCase):
    """Test suite for polite HTTP client behaviors and policies."""

    def setUp(self) -> None:
        """Set up temporary cache directory and test configuration."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.config = ScraperConfig(
            min_crawl_delay=0.01,
            jitter_min=0.0,
            jitter_max=0.01,
            request_timeout=5.0,
            max_retries=2,
            backoff_factor=1.5,
            max_pages_per_run=5,
            max_pages_per_domain=3,
            cache_dir=self.cache_dir,
        )
        self.mock_session = MagicMock(spec=requests.Session())
        self.client = PoliteHttpClient(config=self.config, session=self.mock_session)
        clear_cache()
        # Mock crawl_delay to avoid network calls fetching robots.txt during client tests
        self.crawl_delay_patcher = patch("scraper.http_client.crawl_delay", return_value=0.01)
        self.mock_crawl_delay = self.crawl_delay_patcher.start()

    def tearDown(self) -> None:
        """Clean up temporary directory, patchers, and caches."""
        self.crawl_delay_patcher.stop()
        self.client.close()
        self.temp_dir.cleanup()
        clear_cache()

    @patch("scraper.http_client.can_fetch")
    def test_disallowed_url_is_skipped(self, mock_can_fetch: MagicMock) -> None:
        """Ensure that disallowed URLs are skipped without making network requests."""
        mock_can_fetch.return_value = False
        target_url = "https://darglobal.co.uk/restricted-portal"

        response = self.client.get(target_url)

        self.assertIsNone(response, "Expected None when URL is disallowed by robots.txt")
        self.mock_session.get.assert_not_called()
        mock_can_fetch.assert_called_once_with(target_url, user_agent=self.config.user_agent_bot_name)

    @patch("scraper.http_client.can_fetch")
    @patch("scraper.http_client.time.sleep")
    def test_429_triggers_backoff_with_retry_after(
        self,
        mock_sleep: MagicMock,
        mock_can_fetch: MagicMock,
    ) -> None:
        """Verify that HTTP 429 triggers backoff honoring Retry-After and retries."""
        mock_can_fetch.return_value = True
        target_url = "https://darglobal.co.uk/projects/residence"

        # First response: 429 with Retry-After header
        resp_429 = requests.Response()
        resp_429.status_code = 429
        resp_429.headers["Retry-After"] = "2.5"
        resp_429._content = b"Too Many Requests"

        # Second response: 200 OK success
        resp_200 = requests.Response()
        resp_200.status_code = 200
        resp_200.headers["Content-Type"] = "text/html"
        resp_200._content = b"<html>Success</html>"
        resp_200.url = target_url

        self.mock_session.get.side_effect = [resp_429, resp_200]

        result = self.client.get(target_url)

        self.assertIsNotNone(result)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, "<html>Success</html>")
        self.assertEqual(self.mock_session.get.call_count, 2)

        # Confirm that sleep was called with the Retry-After value
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertTrue(any(abs(arg - 2.5) < 0.01 for arg in sleep_args), f"Expected sleep(2.5), got {sleep_args}")

    @patch("scraper.http_client.can_fetch")
    @patch("scraper.http_client.time.sleep")
    def test_429_exponential_backoff_without_retry_after(
        self,
        mock_sleep: MagicMock,
        mock_can_fetch: MagicMock,
    ) -> None:
        """Verify exponential backoff when Retry-After is absent on 429 or 503."""
        mock_can_fetch.return_value = True
        target_url = "https://darglobal.co.uk/properties/villa-1"

        resp_503 = requests.Response()
        resp_503.status_code = 503
        resp_503._content = b"Service Unavailable"

        resp_200 = requests.Response()
        resp_200.status_code = 200
        resp_200._content = b"<html>Property Details</html>"
        resp_200.url = target_url

        self.mock_session.get.side_effect = [resp_503, resp_200]

        result = self.client.get(target_url)

        self.assertIsNotNone(result)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.mock_session.get.call_count, 2)
        # Expected backoff factor ** (attempt 1) = 1.5 ** 1 = 1.5
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertTrue(any(abs(arg - 1.5) < 0.01 for arg in sleep_args))

    @patch("scraper.http_client.can_fetch")
    def test_cache_hit_makes_no_network_call(self, mock_can_fetch: MagicMock) -> None:
        """Verify that a pre-existing cache file returns data without invoking the network session."""
        mock_can_fetch.return_value = True
        target_url = "https://darglobal.co.uk/projects/luxury-villas"

        # Pre-seed cache file
        cache_path = self.client._get_cache_path(target_url)
        cached_payload = {
            "url": target_url,
            "status_code": 200,
            "headers": {"Content-Type": "text/html"},
            "encoding": "utf-8",
            "text": "<html><title>Cached Page</title></html>",
            "cached_at": "2026-11-18T10:00:00Z",
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached_payload, f)

        response = self.client.get(target_url)

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Cached Page", response.text)
        self.assertTrue(getattr(response, "from_cache", False))
        self.mock_session.get.assert_not_called()

    @patch("scraper.http_client.can_fetch")
    def test_stops_on_403_and_anti_bot_pages(self, mock_can_fetch: MagicMock) -> None:
        """Ensure client stops immediately on HTTP 403 or Cloudflare captcha challenge pages."""
        mock_can_fetch.return_value = True
        target_url = "https://darglobal.co.uk/anti-bot-gate"

        resp_403 = requests.Response()
        resp_403.status_code = 403
        resp_403._content = b"<title>Attention Required! | Cloudflare</title>"

        self.mock_session.get.return_value = resp_403

        response = self.client.get(target_url)

        self.assertIsNone(response, "Should return None on 403/captcha challenge")
        self.assertEqual(self.mock_session.get.call_count, 1, "Should not retry on 403/captcha")

    @patch("scraper.http_client.can_fetch")
    def test_global_page_cap_enforced(self, mock_can_fetch: MagicMock) -> None:
        """Verify global page cap stops further requests."""
        mock_can_fetch.return_value = True
        # Set max pages per run to 1
        limited_config = ScraperConfig(
            max_pages_per_run=1,
            max_pages_per_domain=5,
            cache_dir=self.cache_dir,
            min_crawl_delay=0.0,
            jitter_min=0.0,
            jitter_max=0.0,
        )
        client = PoliteHttpClient(config=limited_config, session=self.mock_session)

        resp_200 = requests.Response()
        resp_200.status_code = 200
        resp_200._content = b"Page 1"
        self.mock_session.get.return_value = resp_200

        res1 = client.get("https://darglobal.co.uk/page-1")
        self.assertIsNotNone(res1)
        self.assertEqual(client.pages_fetched_total, 1)

        # Second request should exceed global cap
        res2 = client.get("https://darglobal.co.uk/page-2")
        self.assertIsNone(res2)
        self.assertEqual(self.mock_session.get.call_count, 1)


class TestRobotsParser(unittest.TestCase):
    """Test suite for robots.txt parsing and caching."""

    def setUp(self) -> None:
        clear_cache()

    def tearDown(self) -> None:
        clear_cache()

    @patch("scraper.robots.requests.get")
    def test_robots_can_fetch_and_crawl_delay(self, mock_get: MagicMock) -> None:
        """Verify robots.txt directives are correctly parsed and cached."""
        robots_content = (
            "User-agent: *\n"
            "Disallow: /admin/\n"
            "Disallow: /private/\n"
            "Crawl-delay: 5\n"
        )
        mock_resp = requests.Response()
        mock_resp.status_code = 200
        mock_resp._content = robots_content.encode("utf-8")
        mock_get.return_value = mock_resp

        self.assertTrue(can_fetch("https://darglobal.co.uk/projects"))
        self.assertFalse(can_fetch("https://darglobal.co.uk/admin/dashboard"))
        self.assertEqual(crawl_delay("darglobal.co.uk"), 5.0)

        # Check caching: requests.get should only be called once for darglobal.co.uk
        self.assertEqual(mock_get.call_count, 1)
        can_fetch("https://darglobal.co.uk/about")
        self.assertEqual(mock_get.call_count, 1)


if __name__ == "__main__":
    unittest.main()

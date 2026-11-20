"""Unit tests for DarGlobal scraper and schema extraction with HTML fixtures."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from scraper.config import ScraperConfig
from scraper.dar_global import (
    detect_page_type,
    discover_urls,
    parse_page,
    run,
)


# Fixture 1: JSON-LD Project Page
FIXTURE_JSON_LD_PROJECT = """<!DOCTYPE html>
<html lang="en">
<head>
  <title>DaVinci Tower by Pagani | DarGlobal</title>
  <meta property="og:title" content="DaVinci Tower by Pagani" />
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "RealEstateListing",
    "name": "DaVinci Tower by Pagani",
    "description": "Super-luxury residential apartments in Downtown Dubai overlooking the Dubai Canal.",
    "address": {
      "@type": "PostalAddress",
      "addressLocality": "Dubai",
      "addressCountry": "UAE"
    },
    "offers": {
      "@type": "Offer",
      "price": "8,000,000",
      "priceCurrency": "AED"
    },
    "numberOfBedrooms": "2 to 4 Bedrooms",
    "amenityFeature": [
      {"name": "Infinity Pool"},
      {"name": "Private Gym"},
      {"name": "Concierge Service"},
      {"name": "Sauna"}
    ],
    "image": [
      "https://darglobal.co.uk/images/davinci-exterior.jpg",
      "https://darglobal.co.uk/images/davinci-interior.jpg"
    ]
  }
  </script>
</head>
<body>
  <h1>DaVinci Tower by Pagani</h1>
</body>
</html>
"""

# Fixture 2: Framework Embedded JSON (__NEXT_DATA__) Project Page
FIXTURE_NEXT_DATA_PROJECT = """<!DOCTYPE html>
<html>
<head>
  <title>Tierra Viva by Lamborghini</title>
</head>
<body>
  <h1>Tierra Viva by Automobili Lamborghini</h1>
  <script id="__NEXT_DATA__" type="application/json">
  {
    "props": {
      "pageProps": {
        "project": {
          "title": "Tierra Viva by Automobili Lamborghini",
          "description": "Exclusive luxury villas situated in Benahavis, Costa del Sol, Spain.",
          "location": "Benahavís, Costa del Sol",
          "country": "Spain",
          "price": "15,000,000",
          "currency": "EUR",
          "bedrooms": "4 to 6 Bedrooms",
          "amenities": ["Private Swimming Pool", "SPA", "24/7 Security", "Panoramic Sea Views"],
          "images": [
            "https://darglobal.co.uk/images/tierra-viva-1.jpg"
          ]
        }
      }
    }
  }
  </script>
</body>
</html>
"""

# Fixture 3: Pure HTML & OpenGraph fallback Project Page
FIXTURE_HTML_FALLBACK_PROJECT = """<!DOCTYPE html>
<html>
<head>
  <title>Urban Oasis by Missoni | Luxury Waterfront Apartments</title>
  <meta property="og:title" content="Urban Oasis by Missoni" />
  <meta property="og:description" content="Waterfront luxury apartments situated in Business Bay, Dubai Water Canal." />
  <meta property="og:image" content="https://darglobal.co.uk/images/urban-oasis.jpg" />
</head>
<body>
  <div class="content">
    <h1>Urban Oasis by Missoni</h1>
    <p class="lead">Waterfront luxury apartments situated in Business Bay, Dubai Water Canal.</p>

    <div class="project-specs">
      <div class="price-box">Starting from AED 2,500,000</div>
      <div class="config">1, 2 & 3 Bedrooms</div>
      <div class="dates">Estimated Completion: Q4 2026</div>
      <div class="status">Status: Under Construction</div>
      <div class="type">Property Type: Luxury Apartments</div>
    </div>

    <div class="amenities-section">
      <h3>Amenities</h3>
      <ul>
        <li>Infinity Swimming Pool</li>
        <li>State-of-the-art Gym</li>
        <li>Waterfront Promenade</li>
        <li>Concierge Desk</li>
      </ul>
    </div>

    <dl class="key-facts">
      <dt>Floors</dt>
      <dd>34 Stories</dd>
      <dt>Interior Designer</dt>
      <dd>Missoni Home</dd>
      <dt>Units</dt>
      <dd>200</dd>
    </dl>
  </div>
</body>
</html>
"""

# Fixture 4: About Page (Company info, LSE listing, track record)
FIXTURE_ABOUT_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>About Us | DarGlobal</title>
  <meta property="og:title" content="About DarGlobal" />
  <meta property="og:description" content="DarGlobal is a premier international real estate developer listed on the London Stock Exchange (LSE)." />
</head>
<body>
  <h1>About DarGlobal</h1>
  <p>DarGlobal was established to innovate luxury real estate across prime international hubs.</p>
  <p>Listed on the London Stock Exchange (LSE), DarGlobal manages a $2.3 billion portfolio GDV spanning the UAE, Oman, Qatar, Spain, and the UK.</p>
  <dl class="offices">
    <dt>Headquarters</dt>
    <dd>London, United Kingdom</dd>
    <dt>Regional Offices</dt>
    <dd>Dubai, Madrid, Muscat</dd>
  </dl>
</body>
</html>
"""

# Fixture 5: Sitemap Index and Urlset XML
FIXTURE_SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://darglobal.co.uk/sitemap-projects.xml</loc>
  </sitemap>
</sitemapindex>
"""

FIXTURE_SITEMAP_URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://darglobal.co.uk/urban-oasis-by-missoni</loc>
  </url>
  <url>
    <loc>https://darglobal.co.uk/about</loc>
  </url>
  <url>
    <loc>https://darglobal.co.uk/tierra-viva</loc>
  </url>
</urlset>
"""


class TestDarGlobalScraper(unittest.TestCase):
    """Test suite for DarGlobal extraction, schema adherence, and run workflow."""

    def test_json_ld_extraction(self) -> None:
        """Verify extraction from application/ld+json scripts."""
        url = "https://darglobal.co.uk/one-of-one/project-pagani"
        record = parse_page(FIXTURE_JSON_LD_PROJECT, url)

        self.assertEqual(record["source"], "darglobal")
        self.assertEqual(record["url"], url)
        self.assertEqual(record["page_type"], "project")
        self.assertEqual(record["title"], "DaVinci Tower by Pagani")
        self.assertEqual(record["location"], "Dubai")
        self.assertEqual(record["country"], "UAE")
        self.assertEqual(record["starting_price"], "8,000,000")
        self.assertEqual(record["currency"], "AED")
        self.assertEqual(record["bedrooms"], "2 to 4 Bedrooms")
        self.assertIn("Infinity Pool", record["amenities"])
        self.assertIn("https://darglobal.co.uk/images/davinci-exterior.jpg", record["image_urls"])
        self.assertIn("Super-luxury residential apartments", record["description"])
        self.assertIn("scraped_at", record)

    def test_embedded_next_data_extraction(self) -> None:
        """Verify extraction from embedded __NEXT_DATA__ JSON script blocks."""
        url = "https://darglobal.co.uk/tierra-viva"
        record = parse_page(FIXTURE_NEXT_DATA_PROJECT, url)

        self.assertEqual(record["source"], "darglobal")
        self.assertEqual(record["page_type"], "project")
        self.assertEqual(record["title"], "Tierra Viva by Automobili Lamborghini")
        self.assertEqual(record["location"], "Benahavís, Costa del Sol")
        self.assertEqual(record["country"], "Spain")
        self.assertEqual(record["starting_price"], "15,000,000")
        self.assertEqual(record["currency"], "EUR")
        self.assertEqual(record["bedrooms"], "4 to 6 Bedrooms")
        self.assertIn("Private Swimming Pool", record["amenities"])

    def test_html_and_opengraph_fallback_extraction(self) -> None:
        """Verify extraction from standard HTML, OpenGraph tags, and semantic tables."""
        url = "https://darglobal.co.uk/urban-oasis-by-missoni"
        record = parse_page(FIXTURE_HTML_FALLBACK_PROJECT, url)

        self.assertEqual(record["source"], "darglobal")
        self.assertEqual(record["page_type"], "project")
        self.assertEqual(record["title"], "Urban Oasis by Missoni")
        self.assertEqual(record["location"], "Dubai")
        self.assertEqual(record["country"], "UAE")
        self.assertEqual(record["property_type"], "Apartments")
        self.assertEqual(record["starting_price"], "2,500,000")
        self.assertEqual(record["currency"], "AED")
        self.assertIn("1, 2 & 3 Bedrooms", record["bedrooms"])
        self.assertEqual(record["completion_date"], "Q4 2026")
        self.assertEqual(record["status"], "Under Construction")
        self.assertIn("Infinity Swimming Pool", record["amenities"])
        self.assertEqual(record["key_facts"].get("Floors"), "34 Stories")
        self.assertEqual(record["key_facts"].get("Interior Designer"), "Missoni Home")

    def test_about_page_extraction(self) -> None:
        """Verify company information extraction on 'about' pages."""
        url = "https://darglobal.co.uk/about"
        record = parse_page(FIXTURE_ABOUT_PAGE, url)

        self.assertEqual(record["source"], "darglobal")
        self.assertEqual(record["page_type"], "about")
        self.assertIn("About DarGlobal", record["title"])
        self.assertEqual(record["country"], "UK")
        self.assertIn("London Stock Exchange", record["key_facts"].get("Listing", ""))
        self.assertEqual(record["key_facts"].get("Portfolio GDV"), "$2.3 billion")
        self.assertEqual(record["key_facts"].get("Headquarters"), "London, United Kingdom")
        # Missing project-specific fields must be null (never guessed)
        self.assertIsNone(record["starting_price"])
        self.assertIsNone(record["currency"])
        self.assertIsNone(record["bedrooms"])

    def test_blog_listing_extracts_cards_without_main_wrapper(self) -> None:
        summary = "A useful blog summary about luxury property investment, buyer considerations, and global market trends. " * 5
        html = f"""<html><body><h1>Blogs</h1><article><time>Nov 13, 2026</time>
        <h2>A sample DarGlobal blog title</h2><p>{summary}</p>
        </article></body></html>"""
        record = parse_page(html, "https://darglobal.co.uk/blog")
        self.assertEqual(record["page_type"], "news")
        self.assertIn("A sample DarGlobal blog title", record["description"])
        self.assertEqual(record["article_date"], "Nov 13, 2026")

    def test_page_type_detection(self) -> None:
        """Verify URL and DOM-based page classification."""
        self.assertEqual(detect_page_type("https://darglobal.co.uk/projects/villas"), "project")
        self.assertEqual(detect_page_type("https://darglobal.co.uk/urban-oasis-by-missoni"), "project")
        self.assertEqual(detect_page_type("https://darglobal.co.uk/about"), "about")
        self.assertEqual(detect_page_type("https://darglobal.co.uk/press/annual-results"), "news")
        self.assertEqual(detect_page_type("https://darglobal.co.uk/contact-us"), "other")

    @patch("scraper.http_client.PoliteHttpClient.get")
    def test_discover_urls_sitemap_index_and_urlset(self, mock_get: MagicMock) -> None:
        """Verify sitemap traversal across sitemap index and urlset."""
        resp_robots = requests.Response()
        resp_robots.status_code = 200
        resp_robots._content = b"User-agent: *\nSitemap: https://darglobal.co.uk/sitemap_index.xml\n"

        resp_index = requests.Response()
        resp_index.status_code = 200
        resp_index._content = FIXTURE_SITEMAP_INDEX.encode("utf-8")

        resp_urlset = requests.Response()
        resp_urlset.status_code = 200
        resp_urlset._content = FIXTURE_SITEMAP_URLSET.encode("utf-8")

        # Session responses sequence: robots.txt -> sitemap_index.xml -> sitemap-projects.xml
        mock_get.side_effect = [resp_robots, resp_index, resp_urlset]

        client = MagicMock()
        client.get.side_effect = [resp_robots, resp_index, resp_urlset]

        urls = discover_urls(client, "https://darglobal.co.uk")
        self.assertEqual(len(urls), 3)
        self.assertIn("https://darglobal.co.uk/urban-oasis-by-missoni", urls)
        self.assertIn("https://darglobal.co.uk/about", urls)
        self.assertIn("https://darglobal.co.uk/tierra-viva", urls)

    @patch("scraper.dar_global.can_fetch", return_value=True)
    @patch("scraper.dar_global.discover_urls")
    @patch("scraper.http_client.PoliteHttpClient.get")
    def test_run_workflow_and_deduplication(
        self,
        mock_client_get: MagicMock,
        mock_discover: MagicMock,
        mock_can_fetch: MagicMock,
    ) -> None:
        """Verify end-to-end run execution, deduplication by URL, and output file generation."""
        test_urls = [
            "https://darglobal.co.uk/urban-oasis-by-missoni",
            "https://darglobal.co.uk/about",
            "https://darglobal.co.uk/urban-oasis-by-missoni",  # duplicate URL
        ]
        mock_discover.return_value = test_urls

        resp_project = requests.Response()
        resp_project.status_code = 200
        resp_project._content = FIXTURE_HTML_FALLBACK_PROJECT.encode("utf-8")
        resp_project.url = "https://darglobal.co.uk/urban-oasis-by-missoni"

        resp_about = requests.Response()
        resp_about.status_code = 200
        resp_about._content = FIXTURE_ABOUT_PAGE.encode("utf-8")
        resp_about.url = "https://darglobal.co.uk/about"

        mock_client_get.side_effect = [resp_project, resp_about]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "darglobal.json"
            cfg = ScraperConfig(
                cache_dir=Path(tmpdir) / "cache",
                min_crawl_delay=0.0,
                jitter_min=0.0,
                jitter_max=0.0,
            )

            records = run(config=cfg, output_path=out_file, max_pages=10)

            # Deduplication check: 3 discovered URLs, 1 duplicate -> 2 distinct records
            self.assertEqual(len(records), 2)
            self.assertTrue(out_file.exists())

            # Validate saved JSON structure
            with open(out_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            self.assertEqual(len(saved_data), 2)
            required_keys = {
                "source",
                "url",
                "page_type",
                "title",
                "location",
                "country",
                "property_type",
                "starting_price",
                "currency",
                "bedrooms",
                "completion_date",
                "status",
                "amenities",
                "description",
                "key_facts",
                "image_urls",
                "scraped_at",
            }
            for rec in saved_data:
                self.assertTrue(required_keys.issubset(rec.keys()))


if __name__ == "__main__":
    unittest.main()

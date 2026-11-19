"""DarGlobal (darglobal.co.uk) scraper implementation.

Extracts public project, company, news, and property data into a standardized schema.
Follows polite crawling policies using PoliteHttpClient, inspecting robots.txt,
sitemaps, JSON-LD / structured data, and HTML fallback.

Playwright Justification Note (Rule 3):
Raw HTML and XML sitemaps provide structured access to DarGlobal content.
When pages contain static HTML, OpenGraph metadata, or JSON-LD, Playwright
is avoided to conserve resources and avoid external browser dependencies.
If a page is purely client-side rendered with empty raw HTML body and no JSON/API
endpoints exist, a Playwright fallback can be engaged.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional
import urllib.parse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from scraper.config import ScraperConfig, config as default_config
from scraper.http_client import PoliteHttpClient
from scraper.robots import can_fetch

logger = logging.getLogger(__name__)


# Known country / location mappings for DarGlobal developments
LOCATION_KEYWORDS = [
    ("Downtown Dubai", "Dubai", "UAE"),
    ("Business Bay", "Dubai", "UAE"),
    ("Dubai", "Dubai", "UAE"),
    ("Benahavís", "Benahavís, Costa del Sol", "Spain"),
    ("Marbella", "Marbella, Costa del Sol", "Spain"),
    ("Costa del Sol", "Costa del Sol", "Spain"),
    ("Spain", "Spain", "Spain"),
    ("Muscat", "Muscat", "Oman"),
    ("AIDA", "Muscat", "Oman"),
    ("Oman", "Oman", "Oman"),
    ("Jeddah", "Jeddah", "Saudi Arabia"),
    ("Saudi Arabia", "Saudi Arabia", "Saudi Arabia"),
    ("Simaisma", "Simaisma", "Qatar"),
    ("Qatar", "Qatar", "Qatar"),
    ("London", "London", "UK"),
    ("United Kingdom", "UK", "UK"),
    ("Maldives", "Maldives", "Maldives"),
]


def detect_page_type(url: str, soup: Optional[BeautifulSoup] = None) -> str:
    """Classify the target page type based on URL path and DOM cues.

    Args:
        url: Absolute URL of the page.
        soup: Parsed BeautifulSoup DOM (optional).

    Returns:
        One of 'project', 'about', 'news', or 'other'.
    """
    url_lower = url.lower()
    path = urllib.parse.urlparse(url_lower).path

    # About / Company info
    if any(k in path for k in ("/about", "/corporate", "/company", "/investor", "/hospitality", "/insignia")):
        return "about"

    # News / Press / Insights / Blog
    if any(k in path for k in ("/press", "/news", "/blog", "/insights")):
        return "news"

    # Projects / Properties
    project_indicators = (
        "/projects",
        "/project",
        "/properties",
        "/one-of-one",
        "/villas",
        "/residences",
        "urban-oasis",
        "tierra-viva",
        "da-vinci",
        "w-residences",
        "trump",
        "aida",
        "marea",
        "dg-circle",
        "rayana",
        "/amaya",
    )
    if any(k in path for k in project_indicators):
        return "project"

    if soup is not None:
        title = (soup.title.string or "").lower() if soup.title else ""
        if "about" in title or "who we are" in title:
            return "about"
        if "news" in title or "press" in title or "blog" in title:
            return "news"

    return "other"


def extract_from_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    """Extract structured property/article attributes from ld+json script blocks."""
    extracted: dict[str, Any] = {}

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string.strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        items = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "@graph" in data:
            items = data["@graph"]

        for item in items:
            if not isinstance(item, dict):
                continue

            # Title / Name
            if not extracted.get("title") and ("name" in item or "headline" in item):
                extracted["title"] = str(item.get("name") or item.get("headline")).strip()

            # Description
            if not extracted.get("description") and "description" in item:
                extracted["description"] = str(item["description"]).strip()

            # Address / Location / Country
            address = item.get("address")
            if isinstance(address, dict):
                if not extracted.get("location") and address.get("addressLocality"):
                    extracted["location"] = str(address["addressLocality"]).strip()
                if not extracted.get("country") and address.get("addressCountry"):
                    extracted["country"] = str(address["addressCountry"]).strip()

            # Offers / Pricing
            offers = item.get("offers")
            if isinstance(offers, dict):
                if not extracted.get("starting_price") and offers.get("price"):
                    extracted["starting_price"] = str(offers["price"]).strip()
                if not extracted.get("currency") and offers.get("priceCurrency"):
                    extracted["currency"] = str(offers["priceCurrency"]).strip()

            # Bedrooms
            if not extracted.get("bedrooms") and "numberOfBedrooms" in item:
                extracted["bedrooms"] = str(item["numberOfBedrooms"]).strip()
            elif not extracted.get("bedrooms") and "numberOfRooms" in item:
                extracted["bedrooms"] = str(item["numberOfRooms"]).strip()

            # Amenities
            if "amenityFeature" in item and isinstance(item["amenityFeature"], list):
                amenities = []
                for feat in item["amenityFeature"]:
                    if isinstance(feat, dict) and "name" in feat:
                        amenities.append(str(feat["name"]).strip())
                    elif isinstance(feat, str):
                        amenities.append(feat.strip())
                if amenities:
                    extracted["amenities"] = amenities

            # Images
            image = item.get("image")
            if image:
                if isinstance(image, list):
                    extracted["image_urls"] = [str(img).strip() for img in image if isinstance(img, str)]
                elif isinstance(image, str):
                    extracted["image_urls"] = [image.strip()]

    return extracted


def extract_from_embedded_json(soup: BeautifulSoup) -> dict[str, Any]:
    """Extract page data from embedded framework JSON (e.g. Next.js __NEXT_DATA__)."""
    extracted: dict[str, Any] = {}
    next_data_script = soup.find("script", id="__NEXT_DATA__")
    if not next_data_script or not next_data_script.string:
        return extracted

    try:
        data = json.loads(next_data_script.string.strip())
        page_props = data.get("props", {}).get("pageProps", {})

        # Check for project or page dictionary in props
        project_obj = page_props.get("project") or page_props.get("data") or page_props
        if isinstance(project_obj, dict):
            if "title" in project_obj:
                extracted["title"] = str(project_obj["title"]).strip()
            if "description" in project_obj:
                extracted["description"] = str(project_obj["description"]).strip()
            if "location" in project_obj:
                extracted["location"] = str(project_obj["location"]).strip()
            if "country" in project_obj:
                extracted["country"] = str(project_obj["country"]).strip()
            if "price" in project_obj:
                extracted["starting_price"] = str(project_obj["price"]).strip()
            if "currency" in project_obj:
                extracted["currency"] = str(project_obj["currency"]).strip()
            if "bedrooms" in project_obj:
                extracted["bedrooms"] = str(project_obj["bedrooms"]).strip()
            if "amenities" in project_obj and isinstance(project_obj["amenities"], list):
                extracted["amenities"] = [str(a).strip() for a in project_obj["amenities"]]
            if "images" in project_obj and isinstance(project_obj["images"], list):
                extracted["image_urls"] = [str(i).strip() for i in project_obj["images"]]
    except Exception as err:
        logger.debug("Failed parsing __NEXT_DATA__: %s", err)

    return extracted


def extract_html_metadata(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    """Extract metadata using OpenGraph, HTML meta tags, and semantic elements."""
    data: dict[str, Any] = {}

    # OpenGraph & Meta dictionary
    metas: dict[str, str] = {}
    for meta in soup.find_all("meta"):
        key = meta.get("property") or meta.get("name")
        content = meta.get("content")
        if key and content:
            metas[key.lower()] = content.strip()

    # Title
    og_title = metas.get("og:title") or metas.get("twitter:title")
    h1_tag = soup.find("h1")
    h1_text = h1_tag.get_text(strip=True) if h1_tag else None
    title_text = soup.title.get_text(strip=True) if soup.title else None
    data["title"] = h1_text or og_title or title_text or None

    # Description
    og_desc = metas.get("og:description") or metas.get("description")
    if og_desc:
        data["description"] = og_desc
    else:
        # Fall back to first meaningful paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 40:
                data["description"] = text
                break

    # Image URLs
    image_urls: list[str] = []
    og_image = metas.get("og:image")
    if og_image:
        image_urls.append(urllib.parse.urljoin(url, og_image))

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and not src.startswith("data:"):
            full_img_url = urllib.parse.urljoin(url, src)
            if full_img_url not in image_urls and any(ext in full_img_url.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                image_urls.append(full_img_url)
    data["image_urls"] = image_urls[:10]  # Keep reasonable top images

    # Full text for regex-based extraction
    page_text = soup.get_text(separator=" ", strip=True)

    # Location & Country resolution
    matched_loc, matched_country = None, None
    for kw, loc, cntry in LOCATION_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", page_text, re.IGNORECASE) or kw.lower() in url.lower():
            matched_loc = loc
            matched_country = cntry
            break
    data["location"] = matched_loc
    data["country"] = matched_country

    # Property Type
    property_type = None
    prop_type_patterns = [
        r"\b(Villas?|Apartments?|Penthouses?|Mansions?|Townhouses?|Hotel & Residences?|Residences?)\b",
    ]
    for pattern in prop_type_patterns:
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            property_type = match.group(1).title()
            break
    data["property_type"] = property_type

    # Starting Price & Currency
    starting_price, currency = None, None
    price_match = re.search(
        r"(?:Starting\s+(?:from|at)|Prices?\s+from|From)\s*([£$€]|AED|GBP|USD|EUR)?\s*([\d,]+(?:\.\d+)?)\s*(AED|GBP|USD|EUR)?",
        page_text,
        re.IGNORECASE,
    )
    if price_match:
        curr_prefix, amount, curr_suffix = price_match.groups()
        currency = curr_suffix or curr_prefix
        starting_price = amount
    else:
        # Secondary search: e.g. "AED 2,500,000" or "GBP 1,200,000"
        standalone_price = re.search(
            r"\b(AED|GBP|USD|EUR|[£$€])\s*([\d,]{4,})\b",
            page_text,
        )
        if standalone_price:
            currency, amount = standalone_price.groups()
            starting_price = amount

    data["starting_price"] = starting_price
    data["currency"] = currency

    # Bedrooms
    bedrooms = None
    beds_match = re.search(
        r"(\d+(?:[\s,]+(?:to|-|&)?\s*\d+)*)\s*(?:Bedrooms?|Beds?|BR)\b",
        page_text,
        re.IGNORECASE,
    )
    if beds_match:
        bedrooms = beds_match.group(0).strip()
    data["bedrooms"] = bedrooms

    # Completion Date
    completion_date = None
    date_match = re.search(
        r"(?:Completion|Handover|Estimated Completion|Expected Completion)[\s:]*(Q[1-4]\s*\d{4}|\w+\s*\d{4}|\d{4})",
        page_text,
        re.IGNORECASE,
    )
    if date_match:
        completion_date = date_match.group(1).strip()
    data["completion_date"] = completion_date

    # Status
    status = None
    status_keywords = (
        "Under Construction",
        "Off-plan",
        "Completed",
        "Ready to Move",
        "Sold Out",
        "Launching Soon",
        "New Launch",
    )
    for kw in status_keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", page_text, re.IGNORECASE):
            status = kw
            break
    data["status"] = status

    # Amenities
    amenities: list[str] = []

    def _is_amenity_element(el: Any) -> bool:
        classes = " ".join(el.get("class", [])) if el.get("class") else ""
        el_id = el.get("id", "") or ""
        return "amenit" in classes.lower() or "amenit" in el_id.lower()

    amenity_container = soup.find(
        lambda el: el.name in ("div", "section", "ul") and _is_amenity_element(el)
    )
    if amenity_container:
        for li in amenity_container.find_all(["li", "span", "p"]):
            text = li.get_text(strip=True)
            if 3 < len(text) < 50 and text not in amenities:
                amenities.append(text)
    if not amenities:
        common_amenities = (
            "Swimming Pool",
            "Gym",
            "Fitness Centre",
            "Spa",
            "Golf Course",
            "Private Beach",
            "Concierge",
            "Security",
            "Parking",
            "Fine Dining",
            "Balcony",
            "Waterfront",
        )
        for am in common_amenities:
            if re.search(r"\b" + re.escape(am) + r"\b", page_text, re.IGNORECASE):
                amenities.append(am)
    data["amenities"] = amenities

    # Key Facts / Specifications
    key_facts: dict[str, str] = {}
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            k = dt.get_text(strip=True).rstrip(":")
            v = dd.get_text(strip=True)
            if k and v:
                key_facts[k] = v

    data["key_facts"] = key_facts
    return data


def extract_primary_content(soup: BeautifulSoup) -> str:
    """Extract readable page content from main/article sections, not just meta text."""
    root = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
    if root is None:
        return ""

    parts = [
        node.get_text(" ", strip=True)
        for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li"])
    ]
    parts = [part for part in parts if part]
    return "\n".join(parts).strip()


def parse_page(html_content: str, url: str) -> dict[str, Any]:
    """Parse raw HTML content into the standardized DarGlobal schema.

    Schema:
        {
          "source": "darglobal",
          "url": str,
          "page_type": "project|about|news|other",
          "title": Optional[str],
          "location": Optional[str],
          "country": Optional[str],
          "property_type": Optional[str],
          "starting_price": Optional[str],
          "currency": Optional[str],
          "bedrooms": Optional[str],
          "completion_date": Optional[str],
          "status": Optional[str],
          "amenities": list[str],
          "description": Optional[str],
          "key_facts": dict[str, Any],
          "image_urls": list[str],
          "scraped_at": str (ISO8601)
        }
    """
    soup = BeautifulSoup(html_content, "html.parser")
    page_type = detect_page_type(url, soup)

    # 1. Prefer JSON-LD
    json_ld_data = extract_from_json_ld(soup)

    # 2. Prefer embedded framework JSON (e.g. Next.js)
    embedded_data = extract_from_embedded_json(soup)

    # 3. HTML parsing fallback
    html_data = extract_html_metadata(soup, url)
    primary_content = extract_primary_content(soup)
    if len(primary_content) > 300:
        html_data["description"] = primary_content

    # Merge with precedence: JSON-LD > embedded JSON > HTML fallback
    def pick_value(key: str, default: Any = None) -> Any:
        return (
            json_ld_data.get(key)
            or embedded_data.get(key)
            or html_data.get(key)
            or default
        )

    # Special handling for company info on 'about' pages
    key_facts = pick_value("key_facts", {}) or {}
    location = pick_value("location", None)
    country = pick_value("country", None)

    if page_type == "about":
        # Extract corporate track record, offices, or founding details if present
        text = soup.get_text(separator=" ", strip=True)
        market_match = re.search(
            r"projects\s+across\s+key\s+markets\s+in\s+([^.;]+)",
            text,
            re.IGNORECASE,
        )
        if market_match:
            key_facts["Key project markets"] = market_match.group(1).strip()
        if "London Stock Exchange" in text:
            key_facts["Listing"] = "London Stock Exchange (LSE)"
        gdv_match = re.search(r"(\$[\d.]+\s*(?:billion|b))\s*(?:portfolio|gdv)", text, re.IGNORECASE)
        if gdv_match:
            key_facts["Portfolio GDV"] = gdv_match.group(1)
        # If headquarters is declared in key facts or text, company location is HQ
        hq = key_facts.get("Headquarters", "")
        if "London" in hq or "United Kingdom" in hq or "London" in text:
            location = "London"
            country = "UK"

    record: dict[str, Any] = {
        "source": "darglobal",
        "url": url,
        "page_type": page_type,
        "title": pick_value("title", None),
        "location": location,
        "country": country,
        "property_type": pick_value("property_type", None),
        "starting_price": pick_value("starting_price", None),
        "currency": pick_value("currency", None),
        "bedrooms": pick_value("bedrooms", None),
        "completion_date": pick_value("completion_date", None),
        "status": pick_value("status", None),
        "amenities": pick_value("amenities", []) or [],
        "description": pick_value("description", None),
        "key_facts": key_facts,
        "image_urls": pick_value("image_urls", []) or [],
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

    return record


def discover_urls(client: PoliteHttpClient, base_url: str) -> list[str]:
    """Discover URLs by parsing /robots.txt and /sitemap.xml (following sitemap indexes).

    Discovery order:
        1. Read /robots.txt to locate Sitemap directives.
        2. Fetch and parse /sitemap.xml (recursing into any <sitemapindex> entries).
        3. Fall back to standard known paths if sitemap retrieval fails.
    """
    discovered: list[str] = []
    seen: set[str] = set()

    # Step 1: Check robots.txt for sitemap declarations
    robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
    sitemap_targets: list[str] = []

    robots_resp = client.get(robots_url)
    if robots_resp and robots_resp.status_code == 200:
        for line in robots_resp.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sm_url = line.split(":", 1)[1].strip()
                if sm_url and sm_url not in sitemap_targets:
                    sitemap_targets.append(sm_url)

    if not sitemap_targets:
        sitemap_targets = [urllib.parse.urljoin(base_url, "/sitemap.xml")]

    # Step 2: Traverse sitemaps (handles sitemap index and urlset)
    sitemaps_to_process = list(sitemap_targets)
    processed_sitemaps: set[str] = set()

    while sitemaps_to_process:
        sitemap_url = sitemaps_to_process.pop(0)
        if sitemap_url in processed_sitemaps:
            continue
        processed_sitemaps.add(sitemap_url)

        resp = client.get(sitemap_url)
        if not resp or resp.status_code != 200:
            logger.warning("Could not fetch sitemap at %s", sitemap_url)
            continue

        try:
            root = ET.fromstring(resp.content)
            tag = root.tag.lower()

            # Sitemap Index
            if "sitemapindex" in tag:
                for sm in root.findall(".//{*}sitemap"):
                    loc = sm.find("{*}loc")
                    if loc is not None and loc.text:
                        child_sitemap = loc.text.strip()
                        if child_sitemap not in processed_sitemaps:
                            sitemaps_to_process.append(child_sitemap)

            # URL Set
            elif "urlset" in tag:
                for u in root.findall(".//{*}url"):
                    loc = u.find("{*}loc")
                    if loc is not None and loc.text:
                        page_url = loc.text.strip()
                        if page_url not in seen:
                            seen.add(page_url)
                            discovered.append(page_url)
        except Exception as err:
            logger.warning("Error parsing XML from %s: %s", sitemap_url, err)

    # Step 3: Fallback seed URLs if sitemap was empty or failed
    if not discovered:
        logger.info("No URLs found in sitemaps; using fallback seed URLs")
        seed_paths = [
            "/",
            "/about",
            "/projects",
            "/press",
            "/blog",
            "/commercial",
            "/hospitality",
        ]
        for path in seed_paths:
            full_seed = urllib.parse.urljoin(base_url, path)
            if full_seed not in seen:
                seen.add(full_seed)
                discovered.append(full_seed)

    return discovered


def run(
    config: Optional[ScraperConfig] = None,
    output_path: Optional[Path] = None,
    max_pages: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Execute DarGlobal scraping workflow.

    Args:
        config: Scraper configuration dataclass.
        output_path: Filepath where output JSON should be stored. Defaults to data/raw/darglobal.json.
        max_pages: Optional maximum number of pages to scrape (useful for test runs).

    Returns:
        List of scraped and validated records.
    """
    cfg = config or default_config
    out_file = output_path or Path("data/raw/darglobal.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    client = PoliteHttpClient(config=cfg)

    # Metrics
    pages_found = 0
    pages_scraped = 0
    pages_skipped_robots = 0
    pages_failed = 0

    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    try:
        logger.info("Discovering URLs for %s", cfg.dar_global_base_url)
        urls = discover_urls(client, cfg.dar_global_base_url)
        pages_found = len(urls)
        logger.info("Discovered %d URLs from %s", pages_found, cfg.dar_global_base_url)

        limit = max_pages or cfg.max_pages_per_domain
        target_urls = urls[:limit]

        for url in target_urls:
            if url in seen_urls:
                continue

            # Check robots.txt permission
            if not can_fetch(url, user_agent=cfg.user_agent_bot_name):
                logger.info("Robots disallowed URL: %s", url)
                pages_skipped_robots += 1
                continue

            # Fetch page via polite HTTP client
            response = client.get(url)
            if response is None or response.status_code != 200:
                logger.warning("Failed or skipped fetching %s", url)
                pages_failed += 1
                continue

            # Parse and extract
            try:
                record = parse_page(response.text, url)
                records.append(record)
                seen_urls.add(url)
                pages_scraped += 1
                logger.info("Successfully scraped [%s]: %s", record["page_type"], url)
            except Exception as err:
                logger.error("Error parsing content from %s: %s", url, err)
                pages_failed += 1

        # Write deduplicated records to data/raw/darglobal.json
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(
            "DarGlobal scrape finished: Found=%d, Scraped=%d, Skipped(robots)=%d, Failed=%d, Output=%s",
            pages_found,
            pages_scraped,
            pages_skipped_robots,
            pages_failed,
            out_file,
        )

    finally:
        client.close()

    return records


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    run()

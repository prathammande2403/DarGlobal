"""DarGlobal scraping pipeline runner.

Executes DarGlobal scraper, generates raw JSON file
(data/raw/darglobal.json), unifies records into
data/processed/combined.json, and prints summary metric tables.
"""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any, Optional

from scraper.config import ScraperConfig, config as default_config
from scraper.dar_global import run as run_darglobal

logger = logging.getLogger(__name__)


def generate_summary(records: list[dict[str, Any]]) -> str:
    """Generate human-readable summary tables for scraped data quality.

    Tables generated:
        1. Records per Source
        2. Records per Page Type
        3. Null-rate percentage per field
    """
    if not records:
        return "No records were scraped."

    lines: list[str] = []
    total_records = len(records)

    # 1. Records per Source
    source_counts = Counter(r.get("source", "unknown") for r in records)
    lines.append("\n" + "=" * 60)
    lines.append("                  SCRAPE RUN SUMMARY")
    lines.append("=" * 60)
    lines.append("\n--- Records per Source ---")
    lines.append(f"{'Source':<20} {'Count':<10} {'Percentage':<10}")
    lines.append("-" * 42)
    for src, count in source_counts.items():
        pct = (count / total_records) * 100
        lines.append(f"{src:<20} {count:<10} {pct:>6.1f}%")
    lines.append("-" * 42)
    lines.append(f"{'Total':<20} {total_records:<10} 100.0%")

    # 2. Records per Page Type
    type_counts = Counter(r.get("page_type", "unknown") for r in records)
    lines.append("\n--- Records per Page Type ---")
    lines.append(f"{'Page Type':<20} {'Count':<10} {'Percentage':<10}")
    lines.append("-" * 42)
    for ptype, count in type_counts.items():
        pct = (count / total_records) * 100
        lines.append(f"{ptype:<20} {count:<10} {pct:>6.1f}%")
    lines.append("-" * 42)

    # 3. Null-rate per field across all records
    all_keys = sorted({k for r in records for k in r.keys()})
    lines.append("\n--- Field Completeness & Null-Rate ---")
    lines.append(f"{'Field':<22} {'Total':<8} {'Nulls':<8} {'Null Rate':<10}")
    lines.append("-" * 52)
    for key in all_keys:
        null_count = sum(1 for r in records if r.get(key) is None or r.get(key) == "")
        null_pct = (null_count / total_records) * 100
        lines.append(f"{key:<22} {total_records:<8} {null_count:<8} {null_pct:>8.1f}%")
    lines.append("-" * 52 + "\n")

    return "\n".join(lines)


def run_pipeline(
    target: str = "darglobal",
    config: Optional[ScraperConfig] = None,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    max_pages: Optional[int] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Execute scraping pipeline for DarGlobal and save outputs.

    Args:
        target: 'all' or 'darglobal'.
        config: Optional ScraperConfig.
        raw_dir: Directory for raw JSON files (default: data/raw).
        processed_dir: Directory for processed combined JSON (default: data/processed).
        max_pages: Page limit per scraper run.

    Returns:
        Tuple of (combined_records, summary_table_text).
    """
    cfg = config or default_config
    raw_path = raw_dir or Path("data/raw")
    processed_path = processed_dir or Path("data/processed")
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    darglobal_file = raw_path / "darglobal.json"
    combined_file = processed_path / "combined.json"

    logger.info("Executing DarGlobal scraper...")
    darglobal_records = run_darglobal(config=cfg, output_path=darglobal_file, max_pages=max_pages)

    # Deduplicate by URL
    combined: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for rec in darglobal_records:
        u = rec.get("url")
        if u and u not in seen_urls:
            seen_urls.add(u)
            combined.append(rec)

    # Write combined processed output
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    summary_text = generate_summary(combined)
    logger.info("Pipeline complete. Output written to %s", combined_file)

    return combined, summary_text


def main() -> None:
    """CLI entry point for python -m scraper.run_scrape."""
    parser = argparse.ArgumentParser(description="DarGlobal Scraper Runner")
    parser.add_argument(
        "--target",
        choices=["all", "darglobal"],
        default="darglobal",
        help="Scraping target: 'all' or 'darglobal'",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit per target",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="data/raw",
        help="Raw output directory",
    )
    parser.add_argument(
        "--output-processed",
        type=str,
        default="data/processed",
        help="Processed output directory",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    _, summary = run_pipeline(
        target=args.target,
        raw_dir=Path(args.output_raw),
        processed_dir=Path(args.output_processed),
        max_pages=args.max_pages,
    )
    print(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""OpenSubtitles scraper for Spanish regional subtitles.

Downloads Spanish subtitles from OpenSubtitles API for slang mining
and evaluation dataset creation.

Usage:
    # First, get an API key from https://www.opensubtitles.com/en/consumers
    export OPENSUBTITLES_API_KEY="your-api-key"

    # Download subtitles for Mexican content
    python scripts/opensubtitles_scraper.py --region mexico --limit 100

    # Download subtitles for Argentine content
    python scripts/opensubtitles_scraper.py --region argentina --limit 100
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# OpenSubtitles API configuration
API_BASE_URL = "https://api.opensubtitles.com/api/v1"

# Region-specific search terms and IMDB IDs for content known to have regional Spanish
REGIONAL_CONTENT = {
    "mexico": {
        "search_terms": [
            "El Chavo del Ocho",
            "Club de Cuervos",
            "La Casa de las Flores",
            "Narcos Mexico",
            "Y tu mamá también",
            "Roma",
            "Amores Perros",
            "El Infierno",
            "Nosotros los Nobles",
            "Instructions Not Included",
        ],
        "imdb_ids": [
            "tt0229889",  # El Chavo del Ocho
            "tt4954642",  # Club de Cuervos
            "tt7670892",  # La Casa de las Flores
            "tt8714904",  # Narcos: Mexico
            "tt0245574",  # Y tu mamá también
            "tt6155172",  # Roma
            "tt0245712",  # Amores Perros
        ],
        "language_code": "es",
    },
    "argentina": {
        "search_terms": [
            "El secreto de sus ojos",
            "Nueve reinas",
            "Relatos salvajes",
            "El Marginal",
            "El Clan",
            "Okupas",
            "Argentina 1985",
            "Esperando la carroza",
        ],
        "imdb_ids": [
            "tt1305806",  # El secreto de sus ojos
            "tt0292090",  # Nueve reinas
            "tt3011894",  # Relatos salvajes
            "tt5765280",  # El Marginal
            "tt4411504",  # El Clan
            "tt0278469",  # Okupas
        ],
        "language_code": "es",
    },
    "spain": {
        "search_terms": [
            "La Casa de Papel",
            "Elite",
            "Vis a vis",
            "El Hoyo",
            "El laberinto del fauno",
            "Volver",
            "Todo sobre mi madre",
            "Abre los ojos",
            "Mar adentro",
        ],
        "imdb_ids": [
            "tt6468322",  # La Casa de Papel
            "tt7134908",  # Elite
            "tt4524056",  # Vis a vis
            "tt8228288",  # El Hoyo (The Platform)
            "tt0457430",  # Pan's Labyrinth
            "tt0441909",  # Volver
        ],
        "language_code": "es",
    },
    "chile": {
        "search_terms": [
            "Una mujer fantástica",
            "No",
            "Gloria",
            "El club",
            "Machuca",
            "Tony Manero",
        ],
        "imdb_ids": [
            "tt5639354",  # Una mujer fantástica
            "tt2059255",  # No
            "tt2425486",  # Gloria
        ],
        "language_code": "es",
    },
}


class OpenSubtitlesClient:
    """Client for OpenSubtitles REST API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "SpanishSlangSTT v1.0",
        }
        self.jwt_token: Optional[str] = None
        self.downloads_remaining = 0

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make API request with rate limiting."""
        url = f"{API_BASE_URL}{endpoint}"
        headers = self.headers.copy()

        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"

        response = requests.request(method, url, headers=headers, **kwargs)

        # Check rate limits
        remaining = response.headers.get("X-RateLimit-Remaining-Second")
        if remaining and int(remaining) < 2:
            logger.info("Rate limit approaching, sleeping 1 second...")
            time.sleep(1)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            logger.warning(f"Rate limited, sleeping {retry_after} seconds...")
            time.sleep(retry_after)
            return self._request(method, endpoint, **kwargs)

        response.raise_for_status()
        return response.json()

    def login(self, username: str, password: str) -> bool:
        """Login to get JWT token for higher download limits."""
        try:
            result = self._request(
                "POST",
                "/login",
                json={"username": username, "password": password},
            )
            self.jwt_token = result.get("token")
            user = result.get("user", {})
            self.downloads_remaining = user.get("allowed_downloads", 0)
            logger.info(f"Logged in. Downloads remaining: {self.downloads_remaining}")
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def search(
        self,
        query: Optional[str] = None,
        imdb_id: Optional[str] = None,
        languages: str = "es",
        page: int = 1,
        per_page: int = 50,
    ) -> dict:
        """Search for subtitles."""
        params = {
            "languages": languages,
            "page": page,
            "per_page": per_page,
        }

        if query:
            params["query"] = query
        if imdb_id:
            # Remove 'tt' prefix if present
            params["imdb_id"] = imdb_id.replace("tt", "")

        return self._request("GET", "/subtitles", params=params)

    def download(self, file_id: int) -> dict:
        """Request download link for a subtitle file."""
        return self._request("POST", "/download", json={"file_id": file_id})

    def get_subtitle_content(self, download_url: str) -> str:
        """Download actual subtitle content from temporary URL."""
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        return response.text


def parse_srt(content: str) -> list[dict]:
    """Parse SRT subtitle file into list of entries."""
    entries = []

    # Split by double newline (subtitle blocks)
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            # First line is index
            index = int(lines[0])

            # Second line is timestamp
            timestamp = lines[1]

            # Remaining lines are text
            text = ' '.join(lines[2:])

            # Clean HTML tags and formatting
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\{[^}]+\}', '', text)  # Remove ASS/SSA formatting
            text = text.strip()

            if text:
                entries.append({
                    "index": index,
                    "timestamp": timestamp,
                    "text": text,
                })
        except (ValueError, IndexError):
            continue

    return entries


def download_subtitles_for_region(
    client: OpenSubtitlesClient,
    region: str,
    output_dir: Path,
    limit: int = 100,
) -> list[dict]:
    """Download subtitles for a specific region."""
    config = REGIONAL_CONTENT.get(region)
    if not config:
        logger.error(f"Unknown region: {region}")
        return []

    region_dir = output_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    total_downloaded = 0

    # Search by IMDB IDs first (more reliable)
    for imdb_id in config["imdb_ids"]:
        if total_downloaded >= limit:
            break

        logger.info(f"Searching subtitles for IMDB: {imdb_id}")

        try:
            results = client.search(imdb_id=imdb_id, languages=config["language_code"])
            subtitles = results.get("data", [])

            for sub in subtitles[:3]:  # Max 3 per movie
                if total_downloaded >= limit:
                    break

                attrs = sub.get("attributes", {})
                files = attrs.get("files", [])

                if not files:
                    continue

                file_info = files[0]
                file_id = file_info.get("file_id")

                if not file_id:
                    continue

                # Download subtitle
                try:
                    dl_result = client.download(file_id)
                    download_url = dl_result.get("link")

                    if not download_url:
                        continue

                    content = client.get_subtitle_content(download_url)

                    # Save raw subtitle
                    filename = f"{imdb_id}_{file_id}.srt"
                    filepath = region_dir / filename
                    filepath.write_text(content, encoding="utf-8")

                    # Parse and save structured data
                    entries = parse_srt(content)

                    downloaded.append({
                        "file_id": file_id,
                        "imdb_id": imdb_id,
                        "filename": filename,
                        "region": region,
                        "title": attrs.get("feature_details", {}).get("title", "Unknown"),
                        "year": attrs.get("feature_details", {}).get("year"),
                        "num_entries": len(entries),
                        "entries": entries,
                    })

                    total_downloaded += 1
                    logger.info(f"Downloaded: {filename} ({len(entries)} lines)")

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Failed to download file {file_id}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Search failed for {imdb_id}: {e}")
            continue

    # Also search by title if we haven't hit the limit
    for search_term in config["search_terms"]:
        if total_downloaded >= limit:
            break

        logger.info(f"Searching: {search_term}")

        try:
            results = client.search(query=search_term, languages=config["language_code"])
            subtitles = results.get("data", [])

            for sub in subtitles[:2]:  # Max 2 per search
                if total_downloaded >= limit:
                    break

                attrs = sub.get("attributes", {})
                files = attrs.get("files", [])

                if not files:
                    continue

                file_info = files[0]
                file_id = file_info.get("file_id")

                if not file_id:
                    continue

                try:
                    dl_result = client.download(file_id)
                    download_url = dl_result.get("link")

                    if not download_url:
                        continue

                    content = client.get_subtitle_content(download_url)

                    # Create filename from search term
                    safe_name = re.sub(r'[^\w\s-]', '', search_term).replace(' ', '_')
                    filename = f"{safe_name}_{file_id}.srt"
                    filepath = region_dir / filename
                    filepath.write_text(content, encoding="utf-8")

                    entries = parse_srt(content)

                    downloaded.append({
                        "file_id": file_id,
                        "search_term": search_term,
                        "filename": filename,
                        "region": region,
                        "title": attrs.get("feature_details", {}).get("title", search_term),
                        "year": attrs.get("feature_details", {}).get("year"),
                        "num_entries": len(entries),
                        "entries": entries,
                    })

                    total_downloaded += 1
                    logger.info(f"Downloaded: {filename} ({len(entries)} lines)")

                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Failed to download: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Search failed for '{search_term}': {e}")
            continue

    # Save manifest
    manifest_path = region_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2)

    logger.info(f"Downloaded {len(downloaded)} subtitle files for {region}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download Spanish subtitles from OpenSubtitles for slang mining"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["mexico", "argentina", "spain", "chile", "all"],
        default="all",
        help="Region to download subtitles for (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/subtitles"),
        help="Output directory for subtitles (default: data/subtitles)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum subtitles to download per region (default: 50)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENSUBTITLES_API_KEY"),
        help="OpenSubtitles API key (or set OPENSUBTITLES_API_KEY env var)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("OPENSUBTITLES_USERNAME"),
        help="OpenSubtitles username for login (optional, increases download limit)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("OPENSUBTITLES_PASSWORD"),
        help="OpenSubtitles password for login (optional)",
    )

    args = parser.parse_args()

    if not args.api_key:
        logger.error("API key required. Set OPENSUBTITLES_API_KEY or use --api-key")
        logger.info("Get an API key at: https://www.opensubtitles.com/en/consumers")
        return

    # Initialize client
    client = OpenSubtitlesClient(args.api_key)

    # Login if credentials provided (increases download limit)
    if args.username and args.password:
        client.login(args.username, args.password)

    # Download for regions
    regions = list(REGIONAL_CONTENT.keys()) if args.region == "all" else [args.region]

    all_downloaded = []
    for region in regions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading subtitles for: {region.upper()}")
        logger.info(f"{'='*60}")

        downloaded = download_subtitles_for_region(
            client, region, args.output_dir, args.limit
        )
        all_downloaded.extend(downloaded)

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total files downloaded: {len(all_downloaded)}")
    for region in regions:
        count = sum(1 for d in all_downloaded if d["region"] == region)
        print(f"  {region}: {count}")
    print(f"\nSubtitles saved to: {args.output_dir}")
    print(f"\nNext step: Run slang mining")
    print(f"  python scripts/slang_miner.py --input-dir {args.output_dir}")


if __name__ == "__main__":
    main()

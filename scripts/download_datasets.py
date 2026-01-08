#!/usr/bin/env python3
"""Download and prepare speech datasets for CatalanSTT training.

Supported datasets:
1. ParlamentParla (Catalan Parliament) - OpenSLR 59
2. Google Crowdsourced Catalan - OpenSLR 69
3. Spanish Conversational Corpus - MagicHub (requires account)

Usage:
    python scripts/download_datasets.py --dataset all --output-dir data/raw
    python scripts/download_datasets.py --dataset parlament --output-dir data/raw
    python scripts/download_datasets.py --dataset google-catalan --output-dir data/raw
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Dataset configurations
DATASETS = {
    "parlament-clean": {
        "name": "ParlamentParla Clean (Catalan Parliament)",
        "url": "https://openslr.trmal.net/resources/59/parlament_v1.0_clean.tar.gz",
        "size_gb": 7.7,
        "hours": 90,
        "language": "ca",
        "license": "CC BY 4.0",
        "format": "tar.gz",
        "audio_format": "wav",
        "sample_rate": 16000,
    },
    "parlament-other": {
        "name": "ParlamentParla Other (Catalan Parliament)",
        "url": "https://openslr.trmal.net/resources/59/parlament_v1.0_other.tar.gz",
        "size_gb": 19,
        "hours": 230,
        "language": "ca",
        "license": "CC BY 4.0",
        "format": "tar.gz",
        "audio_format": "wav",
        "sample_rate": 16000,
    },
    "google-catalan-female": {
        "name": "Google Crowdsourced Catalan (Female)",
        "url": "https://openslr.trmal.net/resources/69/ca_es_female.zip",
        "index_url": "https://openslr.trmal.net/resources/69/line_index_female.tsv",
        "size_gb": 1.0,
        "language": "ca",
        "license": "CC BY-SA 4.0",
        "format": "zip",
        "audio_format": "wav",
        "sample_rate": 48000,
    },
    "google-catalan-male": {
        "name": "Google Crowdsourced Catalan (Male)",
        "url": "https://openslr.trmal.net/resources/69/ca_es_male.zip",
        "index_url": "https://openslr.trmal.net/resources/69/line_index_male.tsv",
        "size_gb": 0.8,
        "language": "ca",
        "license": "CC BY-SA 4.0",
        "format": "zip",
        "audio_format": "wav",
        "sample_rate": 48000,
    },
    "spanish-conversational": {
        "name": "Spanish Conversational Speech Corpus",
        "url": None,  # Requires MagicHub account
        "manual_download": True,
        "download_page": "https://magichub.com/datasets/spanish-conversational-speech-corpus/",
        "size_gb": 0.5,
        "hours": 5.56,
        "language": "es",
        "license": "CC BY-NC-ND 4.0",
        "audio_format": "wav",
        "sample_rate": 16000,
    },
}

# Dataset groups for convenience
DATASET_GROUPS = {
    "all": list(DATASETS.keys()),
    "catalan": ["parlament-clean", "parlament-other", "google-catalan-female", "google-catalan-male"],
    "parlament": ["parlament-clean", "parlament-other"],
    "google-catalan": ["google-catalan-female", "google-catalan-male"],
    "spanish": ["spanish-conversational"],
    "quick": ["google-catalan-female", "google-catalan-male"],  # Smaller downloads for testing
}


def get_file_size(url: str) -> Optional[int]:
    """Get file size from URL headers."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return int(response.headers.get("content-length", 0))
    except Exception:
        return None


def download_file(
    url: str,
    output_path: Path,
    chunk_size: int = 8192,
    resume: bool = True,
) -> bool:
    """Download a file with progress bar and resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing partial download
    existing_size = 0
    if output_path.exists() and resume:
        existing_size = output_path.stat().st_size

    # Get total size
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        # Handle resume
        if response.status_code == 416:  # Range not satisfiable (file complete)
            logger.info(f"File already complete: {output_path.name}")
            return True

        response.raise_for_status()

        # Get total size
        total_size = int(response.headers.get("content-length", 0)) + existing_size

        # Determine write mode
        mode = "ab" if existing_size > 0 else "wb"

        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    remove_archive: bool = False,
) -> bool:
    """Extract tar.gz or zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
            logger.info(f"Extracting tar.gz: {archive_path.name}")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(output_dir)
        elif archive_path.suffix == ".zip":
            logger.info(f"Extracting zip: {archive_path.name}")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False

        if remove_archive:
            archive_path.unlink()
            logger.info(f"Removed archive: {archive_path.name}")

        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_dataset(
    dataset_id: str,
    output_dir: Path,
    extract: bool = True,
    keep_archive: bool = True,
) -> bool:
    """Download and optionally extract a dataset."""
    if dataset_id not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_id}")
        return False

    config = DATASETS[dataset_id]
    dataset_dir = output_dir / dataset_id

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {config['name']}")
    logger.info(f"Size: {config['size_gb']} GB")
    logger.info(f"Language: {config['language']}")
    logger.info(f"License: {config['license']}")
    logger.info(f"{'='*60}")

    # Handle manual download datasets
    if config.get("manual_download"):
        logger.warning(f"This dataset requires manual download!")
        logger.warning(f"Please visit: {config['download_page']}")
        logger.warning(f"After downloading, place files in: {dataset_dir}")

        # Create directory and instruction file
        dataset_dir.mkdir(parents=True, exist_ok=True)
        instructions = dataset_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions, "w") as f:
            f.write(f"Dataset: {config['name']}\n")
            f.write(f"Download URL: {config['download_page']}\n")
            f.write(f"License: {config['license']}\n")
            f.write(f"\nInstructions:\n")
            f.write(f"1. Create an account at MagicHub\n")
            f.write(f"2. Download the dataset from the URL above\n")
            f.write(f"3. Extract the contents to this directory\n")

        return False

    # Download main archive
    url = config["url"]
    archive_name = Path(urlparse(url).path).name
    archive_path = output_dir / "archives" / archive_name

    logger.info(f"Downloading: {url}")

    if not download_file(url, archive_path):
        return False

    # Download index files if available
    if "index_url" in config:
        index_url = config["index_url"]
        index_name = Path(urlparse(index_url).path).name
        index_path = dataset_dir / index_name

        logger.info(f"Downloading index: {index_url}")
        download_file(index_url, index_path)

    # Extract
    if extract:
        if not extract_archive(archive_path, dataset_dir, remove_archive=not keep_archive):
            return False

    logger.info(f"Dataset ready: {dataset_dir}")
    return True


def list_datasets():
    """Print available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 80)

    total_size = 0
    for dataset_id, config in DATASETS.items():
        manual = " (manual download)" if config.get("manual_download") else ""
        hours = f", {config.get('hours', '?')}h" if "hours" in config else ""
        print(f"  {dataset_id:25} {config['size_gb']:>6.1f} GB  {config['language']:3}{hours}{manual}")
        total_size += config["size_gb"]

    print("-" * 80)
    print(f"  {'Total':25} {total_size:>6.1f} GB")

    print("\nDataset Groups:")
    print("-" * 80)
    for group, datasets in DATASET_GROUPS.items():
        print(f"  {group:15} -> {', '.join(datasets)}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download speech datasets for CatalanSTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="quick",
        help="Dataset ID or group name (default: quick). Use 'list' to see available options.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloads (default: data/raw)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract archives after download",
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Remove archive after extraction",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list or args.dataset == "list":
        list_datasets()
        return

    # Resolve dataset group
    if args.dataset in DATASET_GROUPS:
        datasets = DATASET_GROUPS[args.dataset]
    elif args.dataset in DATASETS:
        datasets = [args.dataset]
    else:
        logger.error(f"Unknown dataset or group: {args.dataset}")
        logger.info("Use --list to see available options")
        sys.exit(1)

    # Calculate total size
    total_size = sum(DATASETS[d]["size_gb"] for d in datasets if d in DATASETS)
    logger.info(f"Will download {len(datasets)} dataset(s), total size: {total_size:.1f} GB")

    # Confirm large downloads
    if total_size > 5:
        response = input(f"This will download {total_size:.1f} GB. Continue? [y/N] ")
        if response.lower() != "y":
            logger.info("Download cancelled")
            return

    # Download each dataset
    success = []
    failed = []
    manual = []

    for dataset_id in datasets:
        if DATASETS[dataset_id].get("manual_download"):
            manual.append(dataset_id)
            download_dataset(dataset_id, args.output_dir)
        elif download_dataset(
            dataset_id,
            args.output_dir,
            extract=not args.no_extract,
            keep_archive=not args.remove_archive,
        ):
            success.append(dataset_id)
        else:
            failed.append(dataset_id)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    if success:
        print(f"✓ Successfully downloaded: {', '.join(success)}")
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")
    if manual:
        print(f"⚠ Manual download required: {', '.join(manual)}")

    print(f"\nData location: {args.output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Run: python scripts/prepare_datasets.py --input-dir data/raw --output-dir data/processed")
    print("  2. Run: python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()

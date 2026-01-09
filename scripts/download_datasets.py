#!/usr/bin/env python3
"""Download and prepare speech datasets for SpanishSlangSTT training.

Supported datasets:
1. Spanish Conversational Corpus - MagicHub (requires account)
2. Mozilla Common Voice - Spanish (auto-download with locale filtering)
3. CIEMPIESS - Mexican Spanish (requires manual download)

Usage:
    python scripts/download_datasets.py --dataset all --output-dir data/raw
    python scripts/download_datasets.py --dataset spanish --output-dir data/raw
    python scripts/download_datasets.py --dataset common-voice --region spain --output-dir data/raw
    python scripts/download_datasets.py --dataset common-voice --region mexico --output-dir data/raw
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

# Try to import datasets library for Common Voice
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Region to locale mapping for Common Voice
REGION_LOCALES = {
    "spain": "es-ES",
    "mexico": "es-MX",
    "argentina": "es-AR",
    "colombia": "es-CO",
    "chile": "es-CL",
    "peru": "es-PE",
    "venezuela": "es-VE",
}

# Dataset configurations
DATASETS = {
    "spanish-conversational": {
        "name": "Spanish Conversational Speech Corpus",
        "url": None,  # Requires MagicHub account
        "manual_download": True,
        "download_page": "https://magichub.com/datasets/spanish-conversational-speech-corpus/",
        "size_gb": 0.5,
        "hours": 5.56,
        "language": "es",
        "region": "spain",
        "license": "CC BY-NC-ND 4.0",
        "audio_format": "wav",
        "sample_rate": 16000,
    },
    "common-voice": {
        "name": "Mozilla Common Voice - Spanish",
        "url": None,  # Uses HuggingFace datasets
        "manual_download": False,
        "auto_download": True,
        "size_gb": 25.0,  # Full Spanish dataset, filtered will be smaller
        "hours": 400,  # Approximate
        "language": "es",
        "region": "all",  # Can be filtered by locale
        "license": "CC-0",
        "audio_format": "mp3",
        "sample_rate": 48000,  # Will be resampled to 16kHz
        "hf_dataset": "mozilla-foundation/common_voice_17_0",
        "hf_config": "es",
    },
    "ciempiess": {
        "name": "CIEMPIESS Mexican Spanish Corpus",
        "url": None,
        "manual_download": True,
        "download_page": "https://www.ciempiess.org/",
        "size_gb": 3.0,
        "hours": 17,
        "language": "es",
        "region": "mexico",
        "license": "Research",
        "audio_format": "wav",
        "sample_rate": 16000,
    },
}

# Dataset groups for convenience
DATASET_GROUPS = {
    "all": list(DATASETS.keys()),
    "spanish": ["spanish-conversational"],
    "common-voice-all": ["common-voice"],
    "mexico": ["ciempiess", "common-voice"],
    "spain": ["spanish-conversational", "common-voice"],
    "argentina": ["common-voice"],
}


def get_file_size(url: str) -> Optional[int]:
    """Get file size from URL headers."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return int(response.headers.get("content-length", 0))
    except Exception:
        return None


def download_common_voice(
    output_dir: Path,
    region: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
) -> bool:
    """Download Mozilla Common Voice dataset with optional locale filtering.

    Args:
        output_dir: Directory to save processed audio files
        region: Region to filter by (spain, mexico, argentina, etc.)
        split: Dataset split to download (train, validation, test)
        max_samples: Maximum number of samples to download (for testing)
        streaming: Use streaming mode (recommended for large datasets)

    Returns:
        True if successful
    """
    if not HF_DATASETS_AVAILABLE:
        logger.error("HuggingFace datasets library not available.")
        logger.error("Install with: pip install datasets")
        return False

    config = DATASETS["common-voice"]
    locale = REGION_LOCALES.get(region) if region else None

    logger.info(f"Downloading Common Voice Spanish dataset...")
    if locale:
        logger.info(f"Filtering by locale: {locale} (region: {region})")

    # List of dataset versions to try (newest first, then fallbacks)
    # Note: Many older datasets use loading scripts which are no longer supported.
    # We try datasets in modern Parquet/Arrow format first.
    dataset_versions = [
        # Modern format datasets (Parquet/Arrow) - these should work
        ("facebook/multilingual_librispeech", "spanish"),
        ("PolyAI/minds14", "es-ES"),  # Spanish (Spain)
        # Common Voice - may require accepting terms on HuggingFace
        ("mozilla-foundation/common_voice_17_0", "es"),
        ("mozilla-foundation/common_voice_16_1", "es"),
        ("mozilla-foundation/common_voice_13_0", "es"),
    ]

    try:
        # Check if user is logged in to HuggingFace (required for Common Voice)
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami()
            logger.info(f"Logged in to HuggingFace as: {user_info.get('name', 'unknown')}")
        except Exception:
            logger.warning("Not logged in to HuggingFace. Common Voice requires authentication.")
            logger.warning("Please run: huggingface-cli login")
            logger.warning("And accept the dataset terms at: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")

        # Try loading dataset from different versions
        dataset = None
        used_dataset = None
        for hf_dataset, hf_config in dataset_versions:
            try:
                logger.info(f"Trying to load: {hf_dataset}")
                # Load dataset without automatic audio decoding (we'll decode manually)
                from datasets import Audio
                dataset = load_dataset(
                    hf_dataset,
                    hf_config,
                    split=split,
                    streaming=streaming,
                )
                # Disable automatic audio decoding - we'll handle it manually
                dataset = dataset.cast_column("audio", Audio(decode=False))
                used_dataset = hf_dataset
                logger.info(f"Successfully loaded {hf_dataset}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {hf_dataset}: {e}")
                continue

        if dataset is None:
            logger.error("Failed to load any Common Voice version.")
            logger.error("Please ensure you have:")
            logger.error("  1. Logged in: huggingface-cli login")
            logger.error("  2. Accepted the terms at: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
            return False

        logger.info(f"Loaded dataset from Common Voice {split} split (streaming={streaming})")

        # Create output directory
        region_suffix = f"-{region}" if region else ""
        dataset_dir = output_dir / f"common-voice{region_suffix}"
        audio_dir = dataset_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Import soundfile early to fail fast
        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile not available. Install with: pip install soundfile")
            return False

        # Process samples (works with both streaming and regular datasets)
        import json
        manifest = []
        idx = 0

        # Set up progress bar (unknown total for streaming)
        total = max_samples if max_samples else None
        pbar = tqdm(desc="Processing samples", total=total)

        for sample in dataset:
            # Filter by locale if specified (Common Voice specific)
            if locale and sample.get("locale") and sample.get("locale") != locale:
                continue

            # Get audio data - handle both decoded and raw formats
            audio_data = sample.get("audio", {})

            if isinstance(audio_data, dict) and "bytes" in audio_data:
                # Audio is in raw bytes format - decode manually
                import io
                audio_bytes = audio_data["bytes"]
                if audio_bytes is None:
                    # Try to get from path
                    audio_path_src = audio_data.get("path")
                    if audio_path_src:
                        logger.warning(f"Skipping sample with path-only audio: {audio_path_src}")
                        continue
                    else:
                        logger.warning("Skipping sample with no audio data")
                        continue

                try:
                    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                except Exception as e:
                    logger.warning(f"Failed to decode audio: {e}")
                    continue
            elif isinstance(audio_data, dict) and "array" in audio_data:
                # Audio is already decoded
                audio_array = audio_data["array"]
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                logger.warning(f"Unknown audio format: {type(audio_data)}")
                continue

            # Get transcript (different field names across datasets)
            transcript = (
                sample.get("sentence") or  # Common Voice
                sample.get("transcript") or  # Multilingual LibriSpeech
                sample.get("transcription") or  # FLEURS
                sample.get("raw_text") or  # VoxPopuli
                sample.get("text", "")  # Generic fallback
            )

            # Save audio file
            audio_filename = f"cv_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            sf.write(str(audio_path), audio_array, sample_rate)

            # Add to manifest
            manifest.append({
                "audio_path": str(audio_path),
                "transcript": transcript,
                "duration": len(audio_array) / sample_rate,
                "language": "es",
                "region": region or "general",
                "locale": sample.get("locale", sample.get("language", "es")),
                "speaker_id": sample.get("client_id", sample.get("speaker_id", "unknown")),
                "dataset": used_dataset.split("/")[-1] if used_dataset else "unknown",
            })

            idx += 1
            pbar.update(1)

            # Stop if we've reached max_samples
            if max_samples and idx >= max_samples:
                logger.info(f"Reached max_samples limit ({max_samples})")
                break

        pbar.close()

        # Save manifest
        manifest_path = dataset_dir / f"{split}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(manifest)} samples to {dataset_dir}")
        logger.info(f"Manifest: {manifest_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to download Common Voice: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    region: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> bool:
    """Download and optionally extract a dataset.

    Args:
        dataset_id: ID of the dataset to download
        output_dir: Directory to save the dataset
        extract: Whether to extract archives
        keep_archive: Whether to keep archives after extraction
        region: Region to filter by (for Common Voice)
        max_samples: Maximum samples to download (for testing)
    """
    if dataset_id not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_id}")
        return False

    config = DATASETS[dataset_id]
    dataset_dir = output_dir / dataset_id

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {config['name']}")
    logger.info(f"Size: {config['size_gb']} GB")
    logger.info(f"Language: {config['language']}")
    logger.info(f"Region: {config.get('region', 'general')}")
    logger.info(f"License: {config['license']}")
    logger.info(f"{'='*60}")

    # Handle Common Voice (uses HuggingFace datasets)
    if config.get("auto_download") and dataset_id == "common-voice":
        return download_common_voice(
            output_dir=output_dir,
            region=region,
            split="train",
            max_samples=max_samples,
        )

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
            f.write(f"1. Create an account at the source website\n")
            f.write(f"2. Download the dataset from the URL above\n")
            f.write(f"3. Extract the contents to this directory\n")

        return False

    # Download main archive (for URL-based datasets)
    url = config.get("url")
    if not url:
        logger.error(f"No download URL for dataset: {dataset_id}")
        return False

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
        manual = " (manual)" if config.get("manual_download") else ""
        auto = " (auto)" if config.get("auto_download") else ""
        hours = f", {config.get('hours', '?')}h" if "hours" in config else ""
        region = config.get("region", "general")
        print(f"  {dataset_id:25} {config['size_gb']:>6.1f} GB  {region:10} {config['language']:3}{hours}{manual}{auto}")
        total_size += config["size_gb"]

    print("-" * 80)
    print(f"  {'Total':25} {total_size:>6.1f} GB")

    print("\nDataset Groups:")
    print("-" * 80)
    for group, datasets in DATASET_GROUPS.items():
        print(f"  {group:15} -> {', '.join(datasets)}")

    print("\nAvailable Regions (for Common Voice filtering):")
    print("-" * 80)
    for region, locale in REGION_LOCALES.items():
        print(f"  {region:15} -> {locale}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download speech datasets for SpanishSlangSTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="spanish",
        help="Dataset ID or group name (default: spanish). Use 'list' to see available options.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloads (default: data/raw)",
    )
    parser.add_argument(
        "--region", "-r",
        type=str,
        choices=list(REGION_LOCALES.keys()),
        help="Region to filter Common Voice data by (spain, mexico, argentina, etc.)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to download (useful for testing)",
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
    if args.region:
        logger.info(f"Filtering by region: {args.region}")

    # Confirm large downloads (unless max_samples is set)
    if total_size > 5 and not args.max_samples:
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
            region=args.region,
            max_samples=args.max_samples,
        ):
            success.append(dataset_id)
        else:
            failed.append(dataset_id)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    if success:
        print(f"Successfully downloaded: {', '.join(success)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    if manual:
        print(f"Manual download required: {', '.join(manual)}")

    print(f"\nData location: {args.output_dir.absolute()}")
    print("\nNext steps:")
    region_flag = f" --region {args.region}" if args.region else ""
    print(f"  1. Run: python scripts/prepare_datasets.py --input-dir data/raw --output-dir data/processed{region_flag}")
    print(f"  2. Run: python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()

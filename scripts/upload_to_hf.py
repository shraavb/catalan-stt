#!/usr/bin/env python3
"""Upload Spanish STT dataset to Hugging Face Hub.

This script uploads the processed audio and manifests to HuggingFace
for easy transfer to RunPod or other training environments.

Usage:
    python scripts/upload_to_hf.py --repo-id shraavb/spanish-slang-stt-data
"""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def upload_dataset(
    repo_id: str,
    data_dir: Path = Path("data"),
    private: bool = True,
):
    """Upload dataset to Hugging Face Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'username/dataset-name')
        data_dir: Local data directory containing processed/ and splits/
        private: Whether to make the repo private
    """
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        logger.info(f"Repository ready: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repo: {e}")
        raise

    # Directories to upload
    dirs_to_upload = [
        ("processed", "processed"),  # (local_path, remote_path)
        ("splits", "splits"),
    ]

    for local_name, remote_name in dirs_to_upload:
        local_path = data_dir / local_name
        if not local_path.exists():
            logger.warning(f"Directory not found: {local_path}")
            continue

        logger.info(f"Uploading {local_name}...")

        # Upload entire folder
        api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {local_name} data",
        )
        logger.info(f"Uploaded {local_name} to {repo_id}/{remote_name}")

    # Create a README for the dataset
    readme_content = f"""---
license: cc-by-nc-4.0
language:
- es
tags:
- speech
- stt
- spanish
- regional-spanish
- slang
---

# Spanish Slang STT Dataset

Regional Spanish speech recognition dataset for fine-tuning Whisper.

## Regions
- **Argentina** - OpenSLR Argentine Spanish
- **Mexico** - CIEMPIESS Mexican Spanish
- **Spain** - TEDX Spanish, Common Voice
- **Chile** - Chilean Spanish corpus

## Structure
```
processed/
  argentina/
  mexico/
  spain/
  chile/
splits/
  train.json
  val.json
  test.json
  all.json
```

## Usage

```python
from huggingface_hub import snapshot_download

# Download the dataset
snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="./data"
)
```

## Training

See [spanish-slang-stt](https://github.com/shraavb/spanish-slang-stt) for training scripts.
"""

    # Upload README
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add README",
    )

    logger.info(f"\nDataset uploaded successfully!")
    logger.info(f"View at: https://huggingface.co/datasets/{repo_id}")
    logger.info(f"\nTo download on RunPod:")
    logger.info(f"  pip install huggingface_hub")
    logger.info(f"  python -c \"from huggingface_hub import snapshot_download; snapshot_download('{repo_id}', repo_type='dataset', local_dir='./data')\"")


def main():
    parser = argparse.ArgumentParser(description="Upload Spanish STT dataset to HuggingFace")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="shraavb/spanish-slang-stt-data",
        help="HuggingFace repo ID (default: shraavb/spanish-slang-stt-data)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Local data directory (default: data)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public (default: private)",
    )

    args = parser.parse_args()

    upload_dataset(
        repo_id=args.repo_id,
        data_dir=args.data_dir,
        private=not args.public,
    )


if __name__ == "__main__":
    main()

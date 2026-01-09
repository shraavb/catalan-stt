#!/usr/bin/env python3
"""Upload fine-tuned Whisper models to Hugging Face Hub.

Usage:
    # Upload unified model
    python scripts/upload_model.py --model-path ./models/spanish-slang-whisper \
        --repo-id shraavb/spanish-slang-whisper

    # Upload regional model
    python scripts/upload_model.py --model-path ./models/spanish-slang-whisper-mexico \
        --repo-id shraavb/spanish-slang-whisper-mexico --region mexico

    # Upload all regional models
    python scripts/upload_model.py --all-regions --models-dir ./models
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REGIONS = ["spain", "mexico", "argentina", "chile"]
BASE_REPO = "shraavb/spanish-slang-whisper"


def create_model_card(
    repo_id: str,
    region: Optional[str] = None,
    base_model: str = "openai/whisper-small",
    metrics: Optional[dict] = None,
) -> str:
    """Generate model card content."""
    if region:
        title = f"Spanish Slang Whisper - {region.title()}"
        description = f"Fine-tuned Whisper model for {region.title()} Spanish speech recognition with regional slang support."
        tags = ["spanish", region, "regional-spanish", "slang", "asr"]
    else:
        title = "Spanish Slang Whisper"
        description = "Fine-tuned Whisper model for Spanish speech recognition with regional slang support across Spain, Mexico, Argentina, and Chile."
        tags = ["spanish", "multilingual-spanish", "regional-spanish", "slang", "asr"]

    metrics_section = ""
    if metrics:
        metrics_section = f"""
## Performance

| Metric | Score |
|--------|-------|
| WER | {metrics.get('wer', 'N/A')} |
| CER | {metrics.get('cer', 'N/A')} |
"""

    return f"""---
license: apache-2.0
language:
- es
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
{chr(10).join(f'- {tag}' for tag in tags)}
base_model: {base_model}
---

# {title}

{description}

## Model Description

This model is fine-tuned from `{base_model}` on Spanish speech data including regional dialects and colloquial expressions.

### Supported Regions
- **Spain** (Peninsular Spanish): tío, mola, guay, flipar
- **Mexico**: chido, güey, padre, onda, neta
- **Argentina** (Rioplatense): che, boludo, pibe, laburo, copado
- **Chile**: po, weón, cachai, bacán, pololo
{metrics_section}
## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Load model
processor = WhisperProcessor.from_pretrained("{repo_id}")
model = WhisperForConditionalGeneration.from_pretrained("{repo_id}")

# Load audio
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Transcribe
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features, language="es", task="transcribe")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

## Training

- **Base Model**: {base_model}
- **Dataset**: Regional Spanish corpora (OpenSLR Argentina, CIEMPIESS Mexico, Common Voice Spain/Chile)
- **Training Framework**: Hugging Face Transformers

## Citation

```bibtex
@misc{{spanish-slang-whisper,
  author = {{Shraavasti Bhat}},
  title = {{{title}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## Limitations

- Optimized for conversational Spanish; may underperform on formal/technical content
- Regional models work best with matching dialect
- Performance may vary on heavily accented or very fast speech
"""


def upload_model(
    model_path: Path,
    repo_id: str,
    region: Optional[str] = None,
    private: bool = False,
    metrics: Optional[dict] = None,
):
    """Upload a model to Hugging Face Hub.

    Args:
        model_path: Path to the fine-tuned model directory
        repo_id: HuggingFace repo ID
        region: Optional region identifier for model card
        private: Whether to make the repo private
        metrics: Optional evaluation metrics dict
    """
    api = HfApi()

    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Create repo
    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        logger.info(f"Repository ready: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repo: {e}")
        raise

    # Upload model files
    logger.info(f"Uploading model from {model_path}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {'regional ' + region if region else 'unified'} Spanish Whisper model",
    )

    # Create and upload model card
    model_card = create_model_card(repo_id, region=region, metrics=metrics)
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    logger.info(f"Model uploaded successfully!")
    logger.info(f"View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned Whisper models to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload unified model
  python scripts/upload_model.py --model-path ./models/spanish-slang-whisper \\
      --repo-id shraavb/spanish-slang-whisper

  # Upload regional model
  python scripts/upload_model.py --model-path ./models/spanish-slang-whisper-mexico \\
      --repo-id shraavb/spanish-slang-whisper-mexico --region mexico

  # Upload all regional models
  python scripts/upload_model.py --all-regions --models-dir ./models
        """
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repo ID (e.g., shraavb/spanish-slang-whisper-mexico)",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=REGIONS,
        help="Region for model card (spain, mexico, argentina, chile)",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Upload all regional models from models directory",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing regional models (for --all-regions)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="shraavb",
        help="HuggingFace username for repo naming (default: shraavb)",
    )

    args = parser.parse_args()

    if args.all_regions:
        # Upload all regional models
        for region in REGIONS:
            model_path = args.models_dir / f"spanish-slang-whisper-{region}"
            if model_path.exists():
                repo_id = f"{args.username}/spanish-slang-whisper-{region}"
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Uploading {region} model")
                logger.info(f"{'=' * 60}")
                upload_model(
                    model_path=model_path,
                    repo_id=repo_id,
                    region=region,
                    private=args.private,
                )
            else:
                logger.warning(f"Model not found for {region}: {model_path}")

        print("\n" + "=" * 60)
        print("ALL REGIONAL MODELS UPLOADED")
        print("=" * 60)
        for region in REGIONS:
            print(f"  {region}: https://huggingface.co/{args.username}/spanish-slang-whisper-{region}")

    else:
        if not args.model_path or not args.repo_id:
            parser.error("--model-path and --repo-id are required unless using --all-regions")

        upload_model(
            model_path=args.model_path,
            repo_id=args.repo_id,
            region=args.region,
            private=args.private,
        )


if __name__ == "__main__":
    main()

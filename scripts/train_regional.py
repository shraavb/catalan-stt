#!/usr/bin/env python3
"""Regional fine-tuning script (Approach 2).

This script fine-tunes the unified Spanish model on region-specific data.
It starts from the pre-trained model on HuggingFace and adapts it to each region.

Usage:
    # Fine-tune for a single region
    python scripts/train_regional.py --region mexico --train-manifest data/splits/train.json

    # Fine-tune for all regions
    python scripts/train_regional.py --all-regions --train-manifest data/splits/train.json

    # Fine-tune with custom base model
    python scripts/train_regional.py --region spain --base-model ./models/spanish-slang-whisper-base \
        --train-manifest data/splits/train.json
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from src.training import WhisperTrainer, TrainingConfig
from src.data import load_manifest, filter_by_region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default regions
REGIONS = ["spain", "mexico", "argentina", "chile"]

# HuggingFace model for the unified Spanish model
DEFAULT_BASE_MODEL = "shraavb/spanish-slang-whisper"


def load_fine_tuning_config(region: str, base_config_path: str = "configs/base.yaml") -> dict:
    """Load fine-tuning configuration from base config."""
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    fine_tune_config = base_config.get("fine_tuning", {})
    training_config = base_config.get("training", {})

    # Merge configs with fine-tuning overrides
    config = {
        "base_model": fine_tune_config.get("base_model_hub", DEFAULT_BASE_MODEL),
        "output_dir": f"./models/spanish-slang-whisper-{region}",
        "num_train_epochs": fine_tune_config.get("num_epochs", 2),
        "learning_rate": fine_tune_config.get("learning_rate", 5e-6),
        "warmup_steps": fine_tune_config.get("warmup_steps", 100),
        "save_steps": fine_tune_config.get("save_steps", 250),
        "eval_steps": fine_tune_config.get("eval_steps", 250),
        "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 8),
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 8),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 2),
        "fp16": training_config.get("fp16", True),
        "logging_steps": training_config.get("logging_steps", 100),
    }

    return config


def fine_tune_region(
    region: str,
    train_samples: list,
    val_samples: Optional[list],
    base_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> str:
    """Fine-tune the unified model for a specific region.

    Args:
        region: Target region (spain, mexico, argentina, chile)
        train_samples: Training samples (will be filtered by region)
        val_samples: Validation samples (will be filtered by region)
        base_model: Base model path or HuggingFace ID
        output_dir: Output directory for fine-tuned model
        max_steps: Max training steps (overrides epochs)

    Returns:
        Path to fine-tuned model
    """
    # Load config
    config_dict = load_fine_tuning_config(region)

    # Override with arguments
    if base_model:
        config_dict["base_model"] = base_model
    if output_dir:
        config_dict["output_dir"] = output_dir
    if max_steps:
        config_dict["max_steps"] = max_steps

    # Filter samples for this region
    region_train = filter_by_region(train_samples, [region])
    region_val = filter_by_region(val_samples, [region]) if val_samples else None

    if not region_train:
        logger.warning(f"No training samples found for region: {region}")
        return None

    logger.info(f"Fine-tuning for {region}: {len(region_train)} train samples")
    if region_val:
        logger.info(f"Validation: {len(region_val)} samples")

    # Create training config
    config = TrainingConfig(
        base_model=config_dict["base_model"],
        output_dir=config_dict["output_dir"],
        num_train_epochs=config_dict["num_train_epochs"],
        learning_rate=config_dict["learning_rate"],
        warmup_steps=config_dict["warmup_steps"],
        save_steps=config_dict["save_steps"],
        eval_steps=config_dict["eval_steps"],
        per_device_train_batch_size=config_dict["per_device_train_batch_size"],
        per_device_eval_batch_size=config_dict["per_device_eval_batch_size"],
        gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
        fp16=config_dict["fp16"],
        logging_steps=config_dict["logging_steps"],
        max_steps=config_dict.get("max_steps", -1),
    )

    # Train
    trainer = WhisperTrainer(config)
    model_path = trainer.train(region_train, region_val)

    logger.info(f"Fine-tuned model for {region} saved to: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Spanish Whisper model for specific regions (Approach 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune for Mexico
  python scripts/train_regional.py --region mexico --train-manifest data/splits/train.json

  # Fine-tune for all regions
  python scripts/train_regional.py --all-regions --train-manifest data/splits/train.json

  # Quick test with limited steps
  python scripts/train_regional.py --region spain --train-manifest data/splits/train.json --max-steps 100
        """
    )

    parser.add_argument(
        "--region",
        type=str,
        choices=REGIONS,
        help="Region to fine-tune for",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Fine-tune for all regions sequentially",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        required=True,
        help="Path to training manifest JSON",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Path to validation manifest JSON",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=f"Base model to fine-tune from (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (for testing)",
    )

    args = parser.parse_args()

    if not args.region and not args.all_regions:
        parser.error("Either --region or --all-regions must be specified")

    # Load data
    logger.info(f"Loading training data from {args.train_manifest}")
    train_samples = load_manifest(args.train_manifest)
    logger.info(f"Loaded {len(train_samples)} training samples")

    # Log region distribution
    region_counts = {}
    for sample in train_samples:
        region = sample.region or "general"
        region_counts[region] = region_counts.get(region, 0) + 1
    logger.info(f"Region distribution: {region_counts}")

    val_samples = None
    if args.val_manifest:
        logger.info(f"Loading validation data from {args.val_manifest}")
        val_samples = load_manifest(args.val_manifest)
        logger.info(f"Loaded {len(val_samples)} validation samples")

    # Determine regions to train
    regions_to_train = REGIONS if args.all_regions else [args.region]

    results = {}
    for region in regions_to_train:
        logger.info("=" * 60)
        logger.info(f"Fine-tuning for region: {region}")
        logger.info("=" * 60)

        model_path = fine_tune_region(
            region=region,
            train_samples=train_samples,
            val_samples=val_samples,
            base_model=args.base_model,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
        )

        if model_path:
            results[region] = model_path

    # Summary
    print("\n" + "=" * 60)
    print("REGIONAL FINE-TUNING COMPLETE")
    print("=" * 60)
    for region, path in results.items():
        print(f"  {region}: {path}")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print("1. Evaluate each model: python scripts/evaluate.py --model-path <path> --test-manifest data/splits/test.json")
    print("2. Upload to HuggingFace: python scripts/upload_to_hub.py --model-path <path> --repo-name spanish-slang-whisper-<region>")


if __name__ == "__main__":
    main()

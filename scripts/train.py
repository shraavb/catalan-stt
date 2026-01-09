#!/usr/bin/env python3
"""Training script for Whisper fine-tuning.

Supports:
- Single region training
- Multi-region training (unified model)
- Base + fine-tune approach (train on all, then fine-tune per region)

Usage:
    # Train on all data (unified model)
    python scripts/train.py --config configs/default.yaml --train-manifest data/splits/train.json

    # Train region-specific model
    python scripts/train.py --config configs/regions/mexico.yaml --region mexico \\
        --train-manifest data/splits/mexico/train.json

    # Train base model then fine-tune per region
    python scripts/train.py --config configs/base.yaml --train-manifest data/splits/train.json \\
        --fine-tune-regions spain,mexico,argentina
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from src.training import WhisperTrainer, TrainingConfig
from src.data import load_manifest, filter_by_region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_single_model(
    config: TrainingConfig,
    train_samples: list,
    val_samples: Optional[list],
    region: Optional[str] = None,
) -> str:
    """Train a single model."""
    if region:
        logger.info(f"Training model for region: {region}")
        # Filter samples by region
        train_samples = filter_by_region(train_samples, [region])
        if val_samples:
            val_samples = filter_by_region(val_samples, [region])

        # Update output directory
        if region not in config.output_dir:
            config.output_dir = f"{config.output_dir}-{region}"

    logger.info(f"Training with {len(train_samples)} samples")
    if val_samples:
        logger.info(f"Validating with {len(val_samples)} samples")

    trainer = WhisperTrainer(config)
    return trainer.train(train_samples, val_samples)


def train_base_and_finetune(
    config: TrainingConfig,
    train_samples: list,
    val_samples: Optional[list],
    regions: List[str],
) -> dict:
    """Train base model on all data, then fine-tune per region.

    Returns:
        Dictionary mapping region to model path
    """
    results = {}

    # Step 1: Train base model on all data
    logger.info("=" * 60)
    logger.info("Step 1: Training base model on all regions")
    logger.info("=" * 60)

    base_output_dir = config.output_dir
    config.output_dir = f"{base_output_dir}-base"

    trainer = WhisperTrainer(config)
    base_model_path = trainer.train(train_samples, val_samples)
    results["base"] = base_model_path

    logger.info(f"Base model saved to: {base_model_path}")

    # Step 2: Fine-tune on each region
    for region in regions:
        logger.info("=" * 60)
        logger.info(f"Step 2: Fine-tuning for region: {region}")
        logger.info("=" * 60)

        # Filter samples for this region
        region_train = filter_by_region(train_samples, [region])
        region_val = filter_by_region(val_samples, [region]) if val_samples else None

        if not region_train:
            logger.warning(f"No training samples for region {region}, skipping")
            continue

        logger.info(f"Region {region}: {len(region_train)} train samples")

        # Create fine-tune config
        finetune_config = TrainingConfig.from_yaml(config._config_path)
        finetune_config.base_model = base_model_path  # Start from base model
        finetune_config.output_dir = f"{base_output_dir}-{region}"

        # Reduce epochs for fine-tuning
        finetune_config.num_train_epochs = max(1, finetune_config.num_train_epochs // 2)

        # Fine-tune
        ft_trainer = WhisperTrainer(finetune_config)
        region_model_path = ft_trainer.train(region_train, region_val)
        results[region] = region_model_path

        logger.info(f"Region {region} model saved to: {region_model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Spanish slang data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train unified model on all data
  python scripts/train.py --config configs/default.yaml --train-manifest data/splits/train.json

  # Train region-specific model
  python scripts/train.py --config configs/regions/mexico.yaml --region mexico \\
      --train-manifest data/splits/train.json

  # Train base model then fine-tune per region
  python scripts/train.py --config configs/base.yaml --train-manifest data/splits/train.json \\
      --fine-tune-regions spain,mexico,argentina
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config YAML",
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
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["spain", "mexico", "argentina", "chile", "general"],
        default=None,
        help="Train model for specific region only",
    )
    parser.add_argument(
        "--fine-tune-regions",
        type=str,
        default=None,
        help="Comma-separated list of regions to fine-tune after base training (e.g., spain,mexico,argentina)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps (useful for testing)",
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = TrainingConfig.from_yaml(args.config)
    config._config_path = args.config  # Store for fine-tuning

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.max_steps:
        config.max_steps = args.max_steps

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

    # Determine training mode
    if args.fine_tune_regions:
        # Base + fine-tune mode
        regions = [r.strip() for r in args.fine_tune_regions.split(",")]
        logger.info(f"Training mode: Base + Fine-tune for regions: {regions}")

        results = train_base_and_finetune(config, train_samples, val_samples, regions)

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        for region, path in results.items():
            logger.info(f"  {region}: {path}")

    else:
        # Single model mode (optionally filtered by region)
        if args.region:
            logger.info(f"Training mode: Single region ({args.region})")
        else:
            logger.info("Training mode: Unified model (all regions)")

        model_path = train_single_model(config, train_samples, val_samples, args.region)
        logger.info(f"\nTraining complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()

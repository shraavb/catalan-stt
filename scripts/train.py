#!/usr/bin/env python3
"""Training script for Whisper fine-tuning."""

import argparse
import logging
from pathlib import Path

from src.training import WhisperTrainer, TrainingConfig
from src.data import load_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Catalan-Spanish data")

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

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = TrainingConfig.from_yaml(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir

    # Load data
    logger.info(f"Loading training data from {args.train_manifest}")
    train_samples = load_manifest(args.train_manifest)
    logger.info(f"Loaded {len(train_samples)} training samples")

    val_samples = None
    if args.val_manifest:
        logger.info(f"Loading validation data from {args.val_manifest}")
        val_samples = load_manifest(args.val_manifest)
        logger.info(f"Loaded {len(val_samples)} validation samples")

    # Train
    trainer = WhisperTrainer(config)
    model_path = trainer.train(train_samples, val_samples)

    logger.info(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()

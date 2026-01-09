#!/usr/bin/env python3
"""Master training script for all 4 Spanish regional dialects.

This script trains:
1. A unified base model on all regional data
2. Fine-tuned models for each dialect: Spain, Mexico, Argentina, Chile

Usage:
    # Train all dialects with base + fine-tune approach (recommended)
    python scripts/train_all_dialects.py --mode base-finetune

    # Train each region independently (separate models from scratch)
    python scripts/train_all_dialects.py --mode independent

    # Train only a unified model on all data
    python scripts/train_all_dialects.py --mode unified

    # Quick test run with limited samples
    python scripts/train_all_dialects.py --mode base-finetune --max-samples 100 --max-steps 50

    # Train with custom epochs
    python scripts/train_all_dialects.py --mode base-finetune --epochs 5 --finetune-epochs 2
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import WhisperTrainer, TrainingConfig
from src.data import load_manifest, filter_by_region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)

# All supported regions
REGIONS = ["spain", "mexico", "argentina", "chile"]

# Region-specific configurations
REGION_CONFIGS = {
    "spain": "configs/regions/spain.yaml",
    "mexico": "configs/regions/mexico.yaml",
    "argentina": "configs/regions/argentina.yaml",
    "chile": "configs/regions/chile.yaml",
}


def log_region_distribution(samples: list, title: str = "Dataset"):
    """Log the distribution of samples across regions."""
    region_counts = {}
    for sample in samples:
        region = getattr(sample, 'region', 'unknown') or 'unknown'
        region_counts[region] = region_counts.get(region, 0) + 1

    logger.info(f"\n{title} Distribution:")
    logger.info("-" * 40)
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(samples)
        logger.info(f"  {region:<15} {count:>8} samples ({pct:.1f}%)")
    logger.info(f"  {'Total':<15} {len(samples):>8} samples")
    logger.info("-" * 40)


def train_unified_model(
    config: TrainingConfig,
    train_samples: list,
    val_samples: Optional[list],
    output_dir: str,
) -> str:
    """Train a single unified model on all regional data."""
    logger.info("=" * 60)
    logger.info("TRAINING UNIFIED MODEL (All Regions)")
    logger.info("=" * 60)

    config.output_dir = output_dir
    log_region_distribution(train_samples, "Training Data")

    trainer = WhisperTrainer(config)
    model_path = trainer.train(train_samples, val_samples)

    logger.info(f"Unified model saved to: {model_path}")
    return model_path


def train_base_and_finetune(
    base_config: TrainingConfig,
    train_samples: list,
    val_samples: Optional[list],
    base_output_dir: str,
    base_epochs: int = 3,
    finetune_epochs: int = 2,
) -> Dict[str, str]:
    """Train base model on all data, then fine-tune per region.

    This is the recommended approach as it:
    1. Learns general Spanish speech patterns from all data
    2. Specializes each model on regional slang and accents

    Returns:
        Dictionary mapping region name to model path
    """
    results = {}

    # Step 1: Train base model on all data
    logger.info("=" * 60)
    logger.info("STEP 1: TRAINING BASE MODEL (All Regions)")
    logger.info("=" * 60)

    base_config.output_dir = f"{base_output_dir}/base"
    base_config.num_train_epochs = base_epochs

    log_region_distribution(train_samples, "Base Training Data")

    base_trainer = WhisperTrainer(base_config)
    base_model_path = base_trainer.train(train_samples, val_samples)
    results["base"] = base_model_path

    logger.info(f"Base model saved to: {base_model_path}")

    # Step 2: Fine-tune for each region
    for region in REGIONS:
        logger.info("=" * 60)
        logger.info(f"STEP 2: FINE-TUNING FOR REGION: {region.upper()}")
        logger.info("=" * 60)

        # Filter samples for this region
        region_train = filter_by_region(train_samples, [region])
        region_val = filter_by_region(val_samples, [region]) if val_samples else None

        if not region_train:
            logger.warning(f"No training samples for region {region}, skipping")
            continue

        logger.info(f"Region {region}: {len(region_train)} train samples")
        if region_val:
            logger.info(f"Region {region}: {len(region_val)} validation samples")

        # Load region-specific config if available
        region_config_path = REGION_CONFIGS.get(region)
        if region_config_path and Path(region_config_path).exists():
            finetune_config = TrainingConfig.from_yaml(region_config_path)
            logger.info(f"Loaded region config from {region_config_path}")
        else:
            finetune_config = TrainingConfig()

        # Override with fine-tuning settings
        finetune_config.base_model = base_model_path  # Start from base model
        finetune_config.output_dir = f"{base_output_dir}/{region}"
        finetune_config.num_train_epochs = finetune_epochs
        finetune_config.learning_rate = 5e-6  # Lower LR for fine-tuning
        finetune_config.warmup_steps = 100

        # Fine-tune
        ft_trainer = WhisperTrainer(finetune_config)
        region_model_path = ft_trainer.train(region_train, region_val)
        results[region] = region_model_path

        logger.info(f"Region {region} model saved to: {region_model_path}")

    return results


def train_independent_models(
    base_config: TrainingConfig,
    train_samples: list,
    val_samples: Optional[list],
    output_dir: str,
    epochs: int = 3,
) -> Dict[str, str]:
    """Train independent models for each region from scratch.

    This approach trains each regional model independently without a shared base.
    May be useful if regional differences are very significant.

    Returns:
        Dictionary mapping region name to model path
    """
    results = {}

    for region in REGIONS:
        logger.info("=" * 60)
        logger.info(f"TRAINING INDEPENDENT MODEL: {region.upper()}")
        logger.info("=" * 60)

        # Filter samples for this region
        region_train = filter_by_region(train_samples, [region])
        region_val = filter_by_region(val_samples, [region]) if val_samples else None

        if not region_train:
            logger.warning(f"No training samples for region {region}, skipping")
            continue

        log_region_distribution(region_train, f"{region.title()} Training Data")

        # Load region-specific config if available
        region_config_path = REGION_CONFIGS.get(region)
        if region_config_path and Path(region_config_path).exists():
            config = TrainingConfig.from_yaml(region_config_path)
            logger.info(f"Loaded region config from {region_config_path}")
        else:
            config = TrainingConfig()

        config.output_dir = f"{output_dir}/{region}"
        config.num_train_epochs = epochs

        trainer = WhisperTrainer(config)
        model_path = trainer.train(region_train, region_val)
        results[region] = model_path

        logger.info(f"Region {region} model saved to: {model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Spanish STT models for all 4 regional dialects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  base-finetune : Train unified base model, then fine-tune per region (recommended)
  independent   : Train separate models for each region from scratch
  unified       : Train single unified model on all regional data

Examples:
  # Full training with base + fine-tune (recommended)
  python scripts/train_all_dialects.py --mode base-finetune

  # Quick test run
  python scripts/train_all_dialects.py --mode base-finetune --max-samples 100 --max-steps 50

  # Independent regional models
  python scripts/train_all_dialects.py --mode independent --epochs 5
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["base-finetune", "independent", "unified"],
        default="base-finetune",
        help="Training mode (default: base-finetune)",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="data/splits/train.json",
        help="Path to training manifest JSON",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="data/splits/val.json",
        help="Path to validation manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/spanish-slang-whisper",
        help="Base output directory for models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base training config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=2,
        help="Number of fine-tuning epochs (for base-finetune mode, default: 2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per region (for testing)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Comma-separated list of regions to train (default: all)",
    )

    args = parser.parse_args()

    # Update regions if specified
    global REGIONS
    if args.regions:
        REGIONS = [r.strip() for r in args.regions.split(",")]
        logger.info(f"Training limited to regions: {REGIONS}")

    # Load base config
    logger.info(f"Loading config from {args.config}")
    config = TrainingConfig.from_yaml(args.config)

    # Apply overrides
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    # Load data
    logger.info(f"Loading training data from {args.train_manifest}")
    train_samples = load_manifest(args.train_manifest)
    logger.info(f"Loaded {len(train_samples)} training samples")

    val_samples = None
    if args.val_manifest and Path(args.val_manifest).exists():
        logger.info(f"Loading validation data from {args.val_manifest}")
        val_samples = load_manifest(args.val_manifest)
        logger.info(f"Loaded {len(val_samples)} validation samples")

    # Limit samples if specified (for testing)
    if args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples per region")
        limited_train = []
        limited_val = []

        for region in REGIONS:
            region_train = filter_by_region(train_samples, [region])[:args.max_samples]
            limited_train.extend(region_train)

            if val_samples:
                region_val = filter_by_region(val_samples, [region])[:args.max_samples // 5]
                limited_val.extend(region_val)

        train_samples = limited_train
        val_samples = limited_val if limited_val else None
        logger.info(f"After limiting: {len(train_samples)} train, {len(val_samples) if val_samples else 0} val")

    log_region_distribution(train_samples, "Full Training Dataset")

    # Run training based on mode
    start_time = datetime.now()

    if args.mode == "base-finetune":
        results = train_base_and_finetune(
            base_config=config,
            train_samples=train_samples,
            val_samples=val_samples,
            base_output_dir=args.output_dir,
            base_epochs=args.epochs,
            finetune_epochs=args.finetune_epochs,
        )
    elif args.mode == "independent":
        results = train_independent_models(
            base_config=config,
            train_samples=train_samples,
            val_samples=val_samples,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
    elif args.mode == "unified":
        model_path = train_unified_model(
            config=config,
            train_samples=train_samples,
            val_samples=val_samples,
            output_dir=args.output_dir,
        )
        results = {"unified": model_path}

    # Summary
    duration = datetime.now() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total training time: {duration}")
    logger.info("\nTrained Models:")
    for name, path in results.items():
        logger.info(f"  {name}: {path}")

    # Save results summary
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": args.mode,
        "training_time": str(duration),
        "total_train_samples": len(train_samples),
        "total_val_samples": len(val_samples) if val_samples else 0,
        "regions": REGIONS,
        "models": results,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nTraining summary saved to: {summary_path}")

    # Print next steps
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("1. Evaluate models:")
    for name, path in results.items():
        if name != "base":
            region_flag = f"--region {name}" if name in REGIONS else ""
            logger.info(f"   python scripts/evaluate.py --model {path} --test-manifest data/splits/test.json {region_flag} --per-region")
    logger.info("\n2. Run the API with regional models:")
    logger.info(f"   WHISPER_MODEL={results.get('base', list(results.values())[0])} python -m src.api.app")
    logger.info("\n3. Test transcription:")
    logger.info("   curl -X POST 'http://localhost:8000/transcribe/upload' -F 'file=@audio.wav' -F 'region=mexico'")


if __name__ == "__main__":
    main()

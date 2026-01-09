#!/usr/bin/env python3
"""Evaluate a trained Whisper model on test data.

Supports:
- Overall evaluation
- Per-region metrics breakdown
- Slang marker accuracy tracking
- Cross-region evaluation matrix

Usage:
    # Basic evaluation
    python scripts/evaluate.py --model models/spanish-slang-whisper/final --test-manifest data/splits/test.json

    # Per-region evaluation
    python scripts/evaluate.py --model models/spanish-slang-whisper/final --test-manifest data/splits/test.json --per-region

    # Evaluate region-specific model
    python scripts/evaluate.py --model models/spanish-slang-whisper-mexico --test-manifest data/splits/test.json --region mexico
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_wer,
    compute_cer,
    EvaluationResults,
    format_results,
)
from src.data.loader import load_manifest, filter_by_region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WhisperEvaluator:
    """Evaluate Whisper model on test data."""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        language: str = "ca",
        task: str = "transcribe",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.task = task

        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=self.language,
                task=self.task,
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def evaluate(self, samples: list, show_examples: int = 5) -> tuple:
        """Evaluate model on a list of samples."""
        results = []
        wers = []
        cers = []
        failed = 0
        total_duration = 0.0

        # Track per-region metrics
        region_wers: Dict[str, List[float]] = defaultdict(list)
        region_cers: Dict[str, List[float]] = defaultdict(list)

        # Track slang marker accuracy
        slang_correct = 0
        slang_total = 0

        logger.info(f"Evaluating {len(samples)} samples...")

        for sample in tqdm(samples, desc="Evaluating"):
            try:
                hypothesis = self.transcribe(sample.audio_path)
                reference = sample.transcript

                wer = compute_wer(reference, hypothesis)
                cer = compute_cer(reference, hypothesis)

                wers.append(wer)
                cers.append(cer)
                total_duration += sample.duration_seconds

                # Track per-region metrics
                region = getattr(sample, 'region', 'general') or 'general'
                region_wers[region].append(wer)
                region_cers[region].append(cer)

                # Track slang marker accuracy
                dialect_markers = getattr(sample, 'dialect_markers', None)
                if dialect_markers:
                    for marker in dialect_markers:
                        slang_total += 1
                        if marker.lower() in hypothesis.lower():
                            slang_correct += 1

                results.append({
                    "audio_path": sample.audio_path,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer,
                    "cer": cer,
                    "region": region,
                    "dialect_markers": dialect_markers,
                })

            except Exception as e:
                logger.warning(f"Failed to transcribe {sample.audio_path}: {e}")
                failed += 1

        if not wers:
            raise ValueError("All transcriptions failed")

        eval_results = EvaluationResults(
            mean_wer=np.mean(wers),
            mean_cer=np.mean(cers),
            median_wer=np.median(wers),
            median_cer=np.median(cers),
            std_wer=np.std(wers),
            std_cer=np.std(cers),
            total_samples=len(wers),
            failed_samples=failed,
            total_duration_seconds=total_duration,
            results=[],  # We store separately
        )

        # Compute per-region metrics
        regional_metrics = {}
        for region, region_wer_list in region_wers.items():
            regional_metrics[region] = {
                "mean_wer": float(np.mean(region_wer_list)),
                "mean_cer": float(np.mean(region_cers[region])),
                "median_wer": float(np.median(region_wer_list)),
                "std_wer": float(np.std(region_wer_list)),
                "num_samples": len(region_wer_list),
            }

        # Compute slang accuracy
        slang_accuracy = slang_correct / slang_total if slang_total > 0 else None

        # Show examples
        if show_examples > 0:
            logger.info(f"\n{'='*60}")
            logger.info("SAMPLE TRANSCRIPTIONS")
            logger.info(f"{'='*60}")

            # Sort by WER to show range of quality
            sorted_results = sorted(results, key=lambda x: x["wer"])

            # Show best, worst, and middle examples
            indices = [0, len(sorted_results)//2, -1][:min(show_examples, len(sorted_results))]

            for i, idx in enumerate(indices):
                r = sorted_results[idx]
                logger.info(f"\nExample {i+1} (WER: {r['wer']*100:.1f}%, Region: {r['region']}):")
                logger.info(f"  REF: {r['reference']}")
                logger.info(f"  HYP: {r['hypothesis']}")

        return eval_results, results, regional_metrics, slang_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper model on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        required=True,
        help="Path to test manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for detailed results JSON",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code (default: es for Spanish)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of example transcriptions to show",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["spain", "mexico", "argentina", "chile", "general"],
        default=None,
        help="Evaluate only on samples from specific region",
    )
    parser.add_argument(
        "--per-region",
        action="store_true",
        help="Show per-region metrics breakdown",
    )

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_manifest}")
    samples = load_manifest(args.test_manifest)
    logger.info(f"Loaded {len(samples)} test samples")

    # Filter by region if specified
    if args.region:
        samples = filter_by_region(samples, [args.region])
        logger.info(f"Filtered to {len(samples)} samples for region: {args.region}")

    if not samples:
        logger.error("No samples to evaluate after filtering")
        return

    # Initialize evaluator
    evaluator = WhisperEvaluator(
        model_path=args.model,
        device=args.device,
        language=args.language,
    )

    # Run evaluation
    eval_results, detailed_results, regional_metrics, slang_accuracy = evaluator.evaluate(
        samples,
        show_examples=args.examples,
    )

    # Print results
    print("\n" + format_results(eval_results))

    # Print per-region metrics
    if args.per_region or len(regional_metrics) > 1:
        print("\n" + "=" * 60)
        print("PER-REGION METRICS")
        print("=" * 60)
        print(f"{'Region':<15} {'WER':>10} {'CER':>10} {'Samples':>10}")
        print("-" * 60)
        for region, metrics in sorted(regional_metrics.items()):
            print(f"{region:<15} {metrics['mean_wer']*100:>9.1f}% {metrics['mean_cer']*100:>9.1f}% {metrics['num_samples']:>10}")
        print("-" * 60)

    # Print slang accuracy
    if slang_accuracy is not None:
        print(f"\nSlang Marker Accuracy: {slang_accuracy*100:.1f}%")

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "model": args.model,
            "test_manifest": args.test_manifest,
            "region_filter": args.region,
            "metrics": {
                "mean_wer": float(eval_results.mean_wer),
                "mean_cer": float(eval_results.mean_cer),
                "median_wer": float(eval_results.median_wer),
                "median_cer": float(eval_results.median_cer),
                "std_wer": float(eval_results.std_wer),
                "std_cer": float(eval_results.std_cer),
                "total_samples": eval_results.total_samples,
                "failed_samples": eval_results.failed_samples,
                "slang_accuracy": slang_accuracy,
            },
            "regional_metrics": regional_metrics,
            "results": detailed_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()

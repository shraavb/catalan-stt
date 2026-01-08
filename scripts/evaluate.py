#!/usr/bin/env python3
"""Evaluate a trained Whisper model on test data.

Usage:
    python scripts/evaluate.py --model models/catalan-whisper/final --test-manifest data/splits/test.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_wer,
    compute_cer,
    EvaluationResults,
    format_results,
)
from src.data.loader import load_manifest

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

    def evaluate(self, samples: list, show_examples: int = 5) -> EvaluationResults:
        """Evaluate model on a list of samples."""
        results = []
        wers = []
        cers = []
        failed = 0
        total_duration = 0.0

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

                results.append({
                    "audio_path": sample.audio_path,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer,
                    "cer": cer,
                })

            except Exception as e:
                logger.warning(f"Failed to transcribe {sample.audio_path}: {e}")
                failed += 1

        if not wers:
            raise ValueError("All transcriptions failed")

        import numpy as np

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
                logger.info(f"\nExample {i+1} (WER: {r['wer']*100:.1f}%):")
                logger.info(f"  REF: {r['reference']}")
                logger.info(f"  HYP: {r['hypothesis']}")

        return eval_results, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on test data")

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
        default="ca",
        help="Language code (default: ca for Catalan)",
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

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_manifest}")
    samples = load_manifest(args.test_manifest)
    logger.info(f"Loaded {len(samples)} test samples")

    # Initialize evaluator
    evaluator = WhisperEvaluator(
        model_path=args.model,
        device=args.device,
        language=args.language,
    )

    # Run evaluation
    eval_results, detailed_results = evaluator.evaluate(
        samples,
        show_examples=args.examples,
    )

    # Print results
    print("\n" + format_results(eval_results))

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "model": args.model,
            "test_manifest": args.test_manifest,
            "metrics": {
                "mean_wer": eval_results.mean_wer,
                "mean_cer": eval_results.mean_cer,
                "median_wer": eval_results.median_wer,
                "median_cer": eval_results.median_cer,
                "std_wer": eval_results.std_wer,
                "std_cer": eval_results.std_cer,
                "total_samples": eval_results.total_samples,
                "failed_samples": eval_results.failed_samples,
            },
            "results": detailed_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()

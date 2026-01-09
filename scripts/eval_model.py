#!/usr/bin/env python3
"""Evaluation script for comparing STT models."""

import argparse
import logging
from pathlib import Path

from src.data import load_manifest
from src.evaluation import Benchmark, format_results
from src.api.transcriber import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate STT models")

    parser.add_argument(
        "--test-manifest",
        type=str,
        required=True,
        help="Path to test manifest JSON",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model (if not provided, uses vanilla Whisper)",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="openai/whisper-small",
        help="Baseline model to compare against",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run evaluation on baseline model for comparison",
    )

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_manifest}")
    test_samples = load_manifest(args.test_manifest)
    logger.info(f"Loaded {len(test_samples)} test samples")

    # Extract paths and references
    audio_paths = [s.audio_path for s in test_samples]
    references = [s.transcript for s in test_samples]
    durations = [s.duration_seconds for s in test_samples]

    # Create benchmark
    benchmark = Benchmark(
        audio_paths=audio_paths,
        references=references,
        durations=durations,
        output_dir=args.output_dir,
    )

    # Evaluate baseline if requested
    if args.compare_baseline:
        logger.info(f"Evaluating baseline: {args.baseline_model}")
        baseline_transcriber = Transcriber(model_name=args.baseline_model)

        benchmark.run(
            model_name=f"baseline ({args.baseline_model})",
            transcribe_fn=lambda p: baseline_transcriber.transcribe_file(p).text,
        )

    # Evaluate fine-tuned model
    if args.model_path:
        logger.info(f"Evaluating fine-tuned model: {args.model_path}")
        finetuned_transcriber = Transcriber(model_path=args.model_path)

        benchmark.run(
            model_name=f"spanish-slang-whisper ({args.model_path})",
            transcribe_fn=lambda p: finetuned_transcriber.transcribe_file(p).text,
        )
    else:
        logger.info("Evaluating default Whisper model")
        transcriber = Transcriber()

        benchmark.run(
            model_name="whisper-small",
            transcribe_fn=lambda p: transcriber.transcribe_file(p).text,
        )

    # Print comparison
    print("\n" + benchmark.compare())

    # Save results
    results_path = benchmark.save_results()
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

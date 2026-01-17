"""Benchmark utilities for comparing STT models."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .metrics import EvaluationResults, evaluate_model, format_results

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single model benchmark."""

    model_name: str
    results: EvaluationResults
    timestamp: str
    config: Optional[Dict] = None


class Benchmark:
    """Benchmark and compare STT models."""

    def __init__(
        self,
        audio_paths: List[str],
        references: List[str],
        durations: Optional[List[float]] = None,
        output_dir: str | Path = "./benchmark_results",
    ):
        self.audio_paths = audio_paths
        self.references = references
        self.durations = durations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, BenchmarkResult] = {}

    def run(
        self,
        model_name: str,
        transcribe_fn: Callable[[str], str],
        config: Optional[Dict] = None,
    ) -> EvaluationResults:
        """Run benchmark for a single model."""
        logger.info(f"Running benchmark for: {model_name}")

        results = evaluate_model(
            transcribe_fn=transcribe_fn,
            audio_paths=self.audio_paths,
            references=self.references,
            durations=self.durations,
        )

        benchmark_result = BenchmarkResult(
            model_name=model_name,
            results=results,
            timestamp=datetime.now().isoformat(),
            config=config,
        )

        self.results[model_name] = benchmark_result

        logger.info(f"Benchmark complete for {model_name}")
        logger.info(format_results(results))

        return results

    def compare(self) -> str:
        """Generate comparison report for all benchmarked models."""
        if not self.results:
            return "No benchmark results available"

        lines = [
            "=" * 70,
            "MODEL COMPARISON",
            "=" * 70,
            "",
            f"{'Model':<30} {'WER (%)':>12} {'CER (%)':>12} {'Samples':>10}",
            "-" * 70,
        ]

        # Sort by WER
        sorted_results = sorted(self.results.items(), key=lambda x: x[1].results.mean_wer)

        for model_name, result in sorted_results:
            lines.append(
                f"{model_name:<30} "
                f"{result.results.mean_wer * 100:>11.2f}% "
                f"{result.results.mean_cer * 100:>11.2f}% "
                f"{result.results.total_samples:>10}"
            )

        lines.append("-" * 70)

        # Best model
        best_model = sorted_results[0][0]
        best_wer = sorted_results[0][1].results.mean_wer * 100
        lines.append(f"\nBest model: {best_model} (WER: {best_wer:.2f}%)")

        # Improvement over baseline (if multiple models)
        if len(sorted_results) > 1:
            baseline_wer = sorted_results[-1][1].results.mean_wer * 100
            improvement = baseline_wer - best_wer
            lines.append(f"Improvement over worst: {improvement:.2f}% absolute WER reduction")

        lines.append("=" * 70)

        return "\n".join(lines)

    def save_results(self, filename: str = "benchmark_results.json") -> Path:
        """Save benchmark results to JSON."""
        output_path = self.output_dir / filename

        # Convert to serializable format
        data = {}
        for model_name, result in self.results.items():
            data[model_name] = {
                "model_name": result.model_name,
                "timestamp": result.timestamp,
                "config": result.config,
                "metrics": {
                    "mean_wer": result.results.mean_wer,
                    "mean_cer": result.results.mean_cer,
                    "median_wer": result.results.median_wer,
                    "median_cer": result.results.median_cer,
                    "std_wer": result.results.std_wer,
                    "std_cer": result.results.std_cer,
                    "total_samples": result.results.total_samples,
                    "failed_samples": result.results.failed_samples,
                    "total_duration_seconds": result.results.total_duration_seconds,
                },
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output_path


def run_comparison(
    audio_paths: List[str],
    references: List[str],
    models: Dict[str, Callable[[str], str]],
    output_dir: str = "./benchmark_results",
) -> str:
    """Quick comparison of multiple transcription functions.

    Args:
        audio_paths: List of audio file paths
        references: List of reference transcriptions
        models: Dict mapping model names to transcribe functions

    Returns:
        Comparison report string
    """
    benchmark = Benchmark(audio_paths, references, output_dir=output_dir)

    for model_name, transcribe_fn in models.items():
        benchmark.run(model_name, transcribe_fn)

    benchmark.save_results()

    return benchmark.compare()

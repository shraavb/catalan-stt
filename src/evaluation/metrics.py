"""Evaluation metrics for STT models."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

import jiwer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of a single transcription."""
    reference: str
    hypothesis: str
    wer: float
    cer: float
    audio_path: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class EvaluationResults:
    """Aggregate evaluation results."""
    mean_wer: float
    mean_cer: float
    median_wer: float
    median_cer: float
    std_wer: float
    std_cer: float
    total_samples: int
    failed_samples: int
    total_duration_seconds: float
    results: List[TranscriptionResult]


def normalize_text(text: str) -> str:
    """Normalize text for WER/CER computation."""
    # Lowercase
    text = text.lower()

    # Remove punctuation (keep letters, numbers, spaces)
    import re
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    if not reference:
        return 1.0 if hypothesis else 0.0

    return jiwer.wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    if not reference:
        return 1.0 if hypothesis else 0.0

    return jiwer.cer(reference, hypothesis)


def compute_detailed_wer(reference: str, hypothesis: str) -> Dict:
    """Compute detailed WER breakdown with substitutions, deletions, insertions."""
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    measures = jiwer.compute_measures(reference, hypothesis)

    return {
        "wer": measures["wer"],
        "mer": measures["mer"],  # Match Error Rate
        "wil": measures["wil"],  # Word Information Lost
        "wip": measures["wip"],  # Word Information Preserved
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
    }


def evaluate_model(
    transcribe_fn,
    audio_paths: List[str],
    references: List[str],
    durations: Optional[List[float]] = None,
) -> EvaluationResults:
    """Evaluate a transcription function on a dataset.

    Args:
        transcribe_fn: Function that takes audio_path and returns transcribed text
        audio_paths: List of paths to audio files
        references: List of reference transcriptions
        durations: Optional list of audio durations in seconds

    Returns:
        EvaluationResults with aggregate metrics
    """
    if len(audio_paths) != len(references):
        raise ValueError("audio_paths and references must have same length")

    if durations is None:
        durations = [None] * len(audio_paths)

    results = []
    failed = 0

    for audio_path, reference, duration in zip(audio_paths, references, durations):
        try:
            hypothesis = transcribe_fn(audio_path)

            wer = compute_wer(reference, hypothesis)
            cer = compute_cer(reference, hypothesis)

            results.append(TranscriptionResult(
                reference=reference,
                hypothesis=hypothesis,
                wer=wer,
                cer=cer,
                audio_path=audio_path,
                duration_seconds=duration,
            ))

        except Exception as e:
            logger.warning(f"Failed to transcribe {audio_path}: {e}")
            failed += 1

    if not results:
        raise ValueError("All transcriptions failed")

    # Compute aggregate metrics
    wers = [r.wer for r in results]
    cers = [r.cer for r in results]
    total_duration = sum(r.duration_seconds or 0 for r in results)

    return EvaluationResults(
        mean_wer=np.mean(wers),
        mean_cer=np.mean(cers),
        median_wer=np.median(wers),
        median_cer=np.median(cers),
        std_wer=np.std(wers),
        std_cer=np.std(cers),
        total_samples=len(results),
        failed_samples=failed,
        total_duration_seconds=total_duration,
        results=results,
    )


def evaluate_batch(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True
) -> Dict:
    """Evaluate a batch of reference-hypothesis pairs.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions
        normalize: Whether to normalize text before comparison

    Returns:
        Dictionary with aggregate metrics
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have same length")

    wers = []
    cers = []

    for ref, hyp in zip(references, hypotheses):
        wers.append(compute_wer(ref, hyp, normalize=normalize))
        cers.append(compute_cer(ref, hyp, normalize=normalize))

    return {
        "wer": np.mean(wers),
        "cer": np.mean(cers),
        "wer_median": np.median(wers),
        "cer_median": np.median(cers),
        "num_samples": len(references),
    }


def detect_slang(
    text: str,
    slang_dict: Dict[str, Dict],
    region: Optional[str] = None
) -> List[str]:
    """Detect slang terms in text.

    Args:
        text: Text to analyze
        slang_dict: Dictionary mapping regions to slang terms
        region: Optional region to limit search to

    Returns:
        List of detected slang terms
    """
    text_lower = text.lower()
    detected = []

    if region:
        regions_to_check = [region.lower()]
    else:
        regions_to_check = list(slang_dict.keys())

    for r in regions_to_check:
        if r in slang_dict:
            for term in slang_dict[r]:
                if term.lower() in text_lower:
                    detected.append(term.lower())

    return detected


def format_results(results: EvaluationResults) -> str:
    """Format evaluation results for display."""
    lines = [
        "=" * 50,
        "EVALUATION RESULTS",
        "=" * 50,
        f"Total samples: {results.total_samples}",
        f"Failed samples: {results.failed_samples}",
        f"Total duration: {results.total_duration_seconds:.1f}s",
        "",
        "Word Error Rate (WER):",
        f"  Mean:   {results.mean_wer * 100:.2f}%",
        f"  Median: {results.median_wer * 100:.2f}%",
        f"  Std:    {results.std_wer * 100:.2f}%",
        "",
        "Character Error Rate (CER):",
        f"  Mean:   {results.mean_cer * 100:.2f}%",
        f"  Median: {results.median_cer * 100:.2f}%",
        f"  Std:    {results.std_cer * 100:.2f}%",
        "=" * 50,
    ]

    return "\n".join(lines)

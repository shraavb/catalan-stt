"""Dataset loading utilities for Whisper fine-tuning."""

import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from transformers import WhisperProcessor
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Single audio sample with transcript."""
    audio_path: str
    transcript: str
    duration_seconds: float
    language: str = "es"
    speaker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AudioDataset(Dataset):
    """PyTorch Dataset for audio transcription pairs."""

    def __init__(
        self,
        samples: list[AudioSample],
        processor: WhisperProcessor,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
    ):
        self.samples = samples
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Filter out samples that are too long
        self.samples = [s for s in self.samples if s.duration_seconds <= max_duration]
        logger.info(f"Dataset initialized with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(sample.audio_path, sr=self.sample_rate)

        # Process audio for Whisper
        input_features = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features[0]

        # Process transcript
        labels = self.processor.tokenizer(
            sample.transcript,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


def load_manifest(manifest_path: str | Path) -> list[AudioSample]:
    """Load dataset from JSON manifest file.

    Expected manifest format:
    [
        {
            "audio_path": "path/to/audio.wav",
            "transcript": "transcribed text",
            "duration": 5.2,
            "language": "es",
            "speaker_id": "speaker_001"  # optional
        },
        ...
    ]
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = AudioSample(
            audio_path=item["audio_path"],
            transcript=item["transcript"],
            duration_seconds=item.get("duration", 0.0),
            language=item.get("language", "es"),
            speaker_id=item.get("speaker_id"),
            metadata=item.get("metadata"),
        )
        samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples from {manifest_path}")
    return samples


def load_from_csv(
    csv_path: str | Path,
    audio_dir: str | Path,
    audio_column: str = "audio_file",
    transcript_column: str = "transcript",
) -> list[AudioSample]:
    """Load dataset from CSV file with audio file references."""
    csv_path = Path(csv_path)
    audio_dir = Path(audio_dir)

    df = pd.read_csv(csv_path)

    samples = []
    for _, row in df.iterrows():
        audio_path = audio_dir / row[audio_column]

        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        # Get duration
        try:
            duration = librosa.get_duration(path=str(audio_path))
        except Exception:
            duration = 0.0

        sample = AudioSample(
            audio_path=str(audio_path),
            transcript=row[transcript_column],
            duration_seconds=duration,
            language=row.get("language", "es"),
            speaker_id=row.get("speaker_id"),
        )
        samples.append(sample)

    logger.info(f"Loaded {len(samples)} samples from {csv_path}")
    return samples


def load_dataset_splits(
    data_dir: str | Path,
    train_manifest: str = "train.json",
    val_manifest: str = "val.json",
    test_manifest: str = "test.json",
) -> Dict[str, list[AudioSample]]:
    """Load train/val/test splits from manifest files."""
    data_dir = Path(data_dir)

    splits = {}

    for split_name, manifest_name in [
        ("train", train_manifest),
        ("val", val_manifest),
        ("test", test_manifest),
    ]:
        manifest_path = data_dir / manifest_name
        if manifest_path.exists():
            splits[split_name] = load_manifest(manifest_path)
        else:
            logger.warning(f"Manifest not found: {manifest_path}")
            splits[split_name] = []

    return splits


def create_data_collator(processor: WhisperProcessor):
    """Create a data collator for batching Whisper inputs."""

    def collate_fn(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack input features
        input_features = torch.stack([item["input_features"] for item in batch])

        # Pad labels
        label_features = [item["labels"] for item in batch]
        max_label_length = max(len(l) for l in label_features)

        padded_labels = []
        for labels in label_features:
            padding = torch.full(
                (max_label_length - len(labels),),
                -100,  # -100 is ignored in loss computation
                dtype=labels.dtype,
            )
            padded_labels.append(torch.cat([labels, padding]))

        labels = torch.stack(padded_labels)

        return {
            "input_features": input_features,
            "labels": labels,
        }

    return collate_fn

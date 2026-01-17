"""Audio preprocessing utilities for STT pipeline."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Metadata for processed audio."""

    duration_seconds: float
    sample_rate: int
    num_channels: int
    original_path: Optional[str] = None
    was_resampled: bool = False
    was_normalized: bool = False


class AudioPreprocessor:
    """Preprocess audio files for Whisper fine-tuning."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_db: float = -20.0,
        normalize: bool = True,
        trim_silence: bool = True,
        silence_threshold_db: float = -40.0,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_db = target_db
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db

    def load_audio(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform with sample rate."""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load with librosa (handles multiple formats)
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)

        return waveform, sr

    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == self.target_sample_rate:
            return waveform

        resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.target_sample_rate)
        return resampled

    def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to target dB level."""
        # Calculate current RMS
        rms = np.sqrt(np.mean(waveform**2))

        if rms == 0:
            logger.warning("Audio has zero RMS, skipping normalization")
            return waveform

        # Calculate current dB
        current_db = 20 * np.log10(rms)

        # Calculate gain needed
        gain_db = self.target_db - current_db
        gain = 10 ** (gain_db / 20)

        # Apply gain
        normalized = waveform * gain

        # Clip to prevent clipping
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def trim_silence_from_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence."""
        # Convert threshold to amplitude
        threshold_amp = 10 ** (self.silence_threshold_db / 20)

        # Find non-silent regions
        non_silent = np.abs(waveform) > threshold_amp

        if not np.any(non_silent):
            logger.warning("Audio appears to be entirely silent")
            return waveform

        # Find first and last non-silent samples
        non_silent_indices = np.where(non_silent)[0]
        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1] + 1

        # Add small buffer (100ms)
        buffer_samples = int(0.1 * self.target_sample_rate)
        start_idx = max(0, start_idx - buffer_samples)
        end_idx = min(len(waveform), end_idx + buffer_samples)

        return waveform[start_idx:end_idx]

    def process(self, audio_path: str | Path) -> Tuple[np.ndarray, AudioMetadata]:
        """Full preprocessing pipeline for an audio file."""
        audio_path = Path(audio_path)

        # Load audio
        waveform, orig_sr = self.load_audio(audio_path)

        was_resampled = orig_sr != self.target_sample_rate

        # Resample if needed
        waveform = self.resample(waveform, orig_sr)

        # Trim silence
        if self.trim_silence:
            waveform = self.trim_silence_from_audio(waveform)

        # Normalize
        was_normalized = False
        if self.normalize:
            waveform = self.normalize_audio(waveform)
            was_normalized = True

        # Create metadata
        metadata = AudioMetadata(
            duration_seconds=len(waveform) / self.target_sample_rate,
            sample_rate=self.target_sample_rate,
            num_channels=1,
            original_path=str(audio_path),
            was_resampled=was_resampled,
            was_normalized=was_normalized,
        )

        return waveform, metadata

    def save_audio(
        self, waveform: np.ndarray, output_path: str | Path, sample_rate: Optional[int] = None
    ) -> None:
        """Save processed audio to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sr = sample_rate or self.target_sample_rate
        sf.write(str(output_path), waveform, sr)

        logger.info(f"Saved processed audio to {output_path}")

    def process_and_save(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> AudioMetadata:
        """Process audio and save to new location."""
        waveform, metadata = self.process(input_path)
        self.save_audio(waveform, output_path)
        return metadata


def batch_preprocess(
    input_dir: str | Path,
    output_dir: str | Path,
    preprocessor: Optional[AudioPreprocessor] = None,
    extensions: Tuple[str, ...] = (".wav", ".mp3", ".m4a", ".ogg", ".flac"),
) -> list[AudioMetadata]:
    """Batch preprocess all audio files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if preprocessor is None:
        preprocessor = AudioPreprocessor()

    results = []

    for ext in extensions:
        for audio_file in input_dir.glob(f"*{ext}"):
            output_file = output_dir / f"{audio_file.stem}.wav"

            try:
                metadata = preprocessor.process_and_save(audio_file, output_file)
                results.append(metadata)
                logger.info(f"Processed: {audio_file.name} -> {output_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")

    return results

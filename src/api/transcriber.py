"""Whisper transcription service."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class WordInfo:
    """Word with timing and confidence."""

    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionResult:
    """Result of transcription."""

    text: str
    language: str
    duration_seconds: float
    confidence: float
    words: Optional[List[WordInfo]] = None


class Transcriber:
    """Whisper-based transcriber with word-level timestamps."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
        use_faster_whisper: bool = True,
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_faster_whisper = use_faster_whisper

        self._model = None
        self._processor = None

        logger.info(f"Transcriber initialized (model: {self.model_path}, device: {self.device})")

    def load_model(self) -> None:
        """Load the transcription model."""
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.model_path}")
        start_time = time.time()

        if self.use_faster_whisper:
            self._load_faster_whisper()
        else:
            self._load_transformers()

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")

    def _load_faster_whisper(self) -> None:
        """Load model using faster-whisper."""
        try:
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device == "cuda" else "int8"

            self._model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=compute_type,
            )
            self._is_faster_whisper = True

        except ImportError:
            logger.warning("faster-whisper not available, falling back to transformers")
            self._load_transformers()

    def _load_transformers(self) -> None:
        """Load model using HuggingFace transformers."""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._processor = WhisperProcessor.from_pretrained(self.model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        self._model.to(self.device)
        self._is_faster_whisper = False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "es",
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio array."""
        self.load_model()

        if self._is_faster_whisper:
            return self._transcribe_faster_whisper(audio, language, word_timestamps)
        else:
            return self._transcribe_transformers(audio, sample_rate, language)

    def _transcribe_faster_whisper(
        self,
        audio: np.ndarray,
        language: str,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper."""
        segments, info = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        # Collect all text and words
        full_text = []
        all_words = []
        total_confidence = 0
        segment_count = 0

        for segment in segments:
            full_text.append(segment.text.strip())
            total_confidence += segment.avg_logprob
            segment_count += 1

            if word_timestamps and segment.words:
                for word in segment.words:
                    all_words.append(
                        WordInfo(
                            word=word.word.strip(),
                            start=word.start,
                            end=word.end,
                            confidence=word.probability,
                        )
                    )

        # Convert log probability to confidence
        avg_confidence = np.exp(total_confidence / max(segment_count, 1))

        return TranscriptionResult(
            text=" ".join(full_text),
            language=info.language,
            duration_seconds=info.duration,
            confidence=float(avg_confidence),
            words=all_words if word_timestamps else None,
        )

    def _transcribe_transformers(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> TranscriptionResult:
        """Transcribe using HuggingFace transformers."""
        # Process audio
        input_features = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device)

        # Generate
        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                language=language,
                task="transcribe",
            )

        # Decode
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        duration = len(audio) / sample_rate

        return TranscriptionResult(
            text=text,
            language=language,
            duration_seconds=duration,
            confidence=0.9,  # Transformers doesn't provide confidence easily
            words=None,
        )

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str = "es",
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio file."""
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=16000)
        return self.transcribe(audio, sr, language, word_timestamps)


# Global transcriber instance
_transcriber: Optional[Transcriber] = None


def get_transcriber(
    model_path: Optional[str] = None,
    force_reload: bool = False,
) -> Transcriber:
    """Get or create global transcriber instance."""
    global _transcriber

    if _transcriber is None or force_reload:
        _transcriber = Transcriber(model_path=model_path)
        _transcriber.load_model()

    return _transcriber

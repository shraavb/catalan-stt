"""Generate synthetic audio data using ElevenLabs TTS."""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of TTS synthesis."""
    text: str
    audio_path: str
    voice_id: str
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class SyntheticDataGenerator:
    """Generate synthetic training data using ElevenLabs TTS.

    This creates audio from text transcripts, which can be used to:
    1. Bootstrap training data when real recordings are limited
    2. Create consistent pronunciation examples
    3. Augment real data with synthetic samples

    Note: Models trained on synthetic data may overfit to TTS characteristics.
    Always validate with real human speech.
    """

    # ElevenLabs voice IDs for Spanish speakers
    SPANISH_VOICES = {
        "mateo": "bVMeCyTHy58xNoL34h3p",  # Spanish male
        "lucia": "XB0fDUnXU5powFXDhCwa",  # Spanish female (European)
        "pablo": "TX3LPaxmHKxFdv7VOQHJ",  # Spanish male (Latin American)
        "sara": "gD1IexrzCvsXPHUuT0s3",  # Spanish female (Peninsular)
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str | Path = "data/synthetic",
        voice_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key required. Set ELEVENLABS_API_KEY env var or pass api_key."
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.voice_id = voice_id or self.SPANISH_VOICES["mateo"]

        # Try to import elevenlabs
        try:
            from elevenlabs import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install elevenlabs: pip install elevenlabs")

    def synthesize(
        self,
        text: str,
        output_filename: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesize a single text to audio."""
        voice = voice_id or self.voice_id

        if output_filename is None:
            # Generate filename from text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output_filename = f"synthetic_{text_hash}.mp3"

        output_path = self.output_dir / output_filename

        try:
            audio_generator = self._client.text_to_speech.convert(
                text=text,
                voice_id=voice,
                model_id="eleven_multilingual_v2",  # Best for non-English
            )

            # Write audio to file
            with open(output_path, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)

            # Get duration
            import librosa
            duration = librosa.get_duration(path=str(output_path))

            logger.info(f"Synthesized: '{text[:50]}...' -> {output_path}")

            return SynthesisResult(
                text=text,
                audio_path=str(output_path),
                voice_id=voice,
                duration_seconds=duration,
                success=True,
            )

        except Exception as e:
            logger.error(f"Synthesis failed for '{text[:50]}...': {e}")
            return SynthesisResult(
                text=text,
                audio_path="",
                voice_id=voice,
                duration_seconds=0,
                success=False,
                error=str(e),
            )

    def synthesize_batch(
        self,
        texts: List[str],
        delay_seconds: float = 0.5,
        alternate_voices: bool = True,
    ) -> List[SynthesisResult]:
        """Synthesize multiple texts with rate limiting."""
        results = []
        voices = list(self.SPANISH_VOICES.values())

        for i, text in enumerate(texts):
            voice = voices[i % len(voices)] if alternate_voices else self.voice_id

            result = self.synthesize(
                text=text,
                output_filename=f"synthetic_{i:05d}.mp3",
                voice_id=voice,
            )
            results.append(result)

            # Rate limiting
            if delay_seconds > 0 and i < len(texts) - 1:
                time.sleep(delay_seconds)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch synthesis complete: {successful}/{len(texts)} successful")

        return results

    def create_manifest(
        self,
        results: List[SynthesisResult],
        output_path: str | Path = "data/synthetic/manifest.json",
    ) -> Path:
        """Create a training manifest from synthesis results."""
        output_path = Path(output_path)

        manifest = []
        for result in results:
            if result.success:
                manifest.append({
                    "audio_path": result.audio_path,
                    "transcript": result.text,
                    "duration": result.duration_seconds,
                    "language": "es",
                    "synthetic": True,
                    "voice_id": result.voice_id,
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Created manifest with {len(manifest)} entries at {output_path}")
        return output_path


def generate_from_dialogues(
    dialogues_path: str | Path,
    output_dir: str | Path,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Path:
    """Generate synthetic audio from a dialogue JSON file.

    Expected format:
    [
        {"text": "Hola, ¿cómo estás?", ...},
        {"text": "Muy bien, gracias", ...},
    ]
    """
    dialogues_path = Path(dialogues_path)

    with open(dialogues_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    # Extract texts
    texts = []
    for d in dialogues:
        if isinstance(d, dict) and "text" in d:
            texts.append(d["text"])
        elif isinstance(d, str):
            texts.append(d)

    if max_samples:
        texts = texts[:max_samples]

    # Generate
    generator = SyntheticDataGenerator(api_key=api_key, output_dir=output_dir)
    results = generator.synthesize_batch(texts)

    # Create manifest
    manifest_path = generator.create_manifest(results)

    return manifest_path

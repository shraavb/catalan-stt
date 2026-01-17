"""Pytest configuration and shared fixtures."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_audio_path(tmp_path):
    """Create a sample audio file for testing."""
    import soundfile as sf

    # Generate 2 seconds of silence at 16kHz
    sample_rate = 16000
    duration = 2.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sample_rate)

    return str(audio_path)


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a sample manifest file for testing."""
    manifest_data = [
        {
            "audio_path": "/path/to/audio1.wav",
            "transcript": "Hola, ¿qué tal?",
            "duration": 2.5,
            "language": "es",
            "region": "spain"
        },
        {
            "audio_path": "/path/to/audio2.wav",
            "transcript": "¡Qué chido, güey!",
            "duration": 1.8,
            "language": "es",
            "region": "mexico"
        },
        {
            "audio_path": "/path/to/audio3.wav",
            "transcript": "Che, ¿qué onda?",
            "duration": 2.0,
            "language": "es",
            "region": "argentina"
        }
    ]

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False)

    return str(manifest_path)


@pytest.fixture
def sample_slang_dict():
    """Sample slang dictionary for testing."""
    return {
        "mexico": {
            "chido": {"meaning": "cool", "formality": "informal"},
            "güey": {"meaning": "dude", "formality": "very_informal"},
            "neta": {"meaning": "truth/really", "formality": "informal"},
        },
        "spain": {
            "mola": {"meaning": "cool", "formality": "informal"},
            "tío": {"meaning": "dude/guy", "formality": "informal"},
            "guay": {"meaning": "cool", "formality": "informal"},
        },
        "argentina": {
            "che": {"meaning": "hey/dude", "formality": "informal"},
            "boludo": {"meaning": "dude/idiot", "formality": "very_informal"},
            "copado": {"meaning": "cool", "formality": "informal"},
        },
        "chile": {
            "cachai": {"meaning": "you know?", "formality": "informal"},
            "bacán": {"meaning": "cool", "formality": "informal"},
            "fome": {"meaning": "boring", "formality": "informal"},
        }
    }


@pytest.fixture
def mock_transcription_results():
    """Mock transcription results for evaluation testing."""
    return [
        {"reference": "hola qué tal", "hypothesis": "hola qué tal"},
        {"reference": "está muy chido", "hypothesis": "está muy chido"},
        {"reference": "qué onda güey", "hypothesis": "que onda wey"},
        {"reference": "me mola mucho", "hypothesis": "me mola mucho"},
    ]

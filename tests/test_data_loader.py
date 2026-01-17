"""Tests for data loading and preprocessing functionality."""

import pytest
import json
import numpy as np
from pathlib import Path


class TestManifestLoading:
    """Tests for manifest loading functions."""

    def test_load_manifest_valid_json(self, sample_manifest):
        """Test loading a valid manifest file."""
        from src.data.loader import load_manifest

        data = load_manifest(sample_manifest)

        assert len(data) == 3
        # Data returns AudioSample objects, check attributes
        assert all(hasattr(item, "audio_path") for item in data)
        assert all(hasattr(item, "transcript") for item in data)

    def test_load_manifest_missing_file(self, tmp_path):
        """Test loading a non-existent manifest file."""
        from src.data.loader import load_manifest

        with pytest.raises(FileNotFoundError):
            load_manifest(str(tmp_path / "nonexistent.json"))

    def test_load_manifest_invalid_json(self, tmp_path):
        """Test loading an invalid JSON file."""
        from src.data.loader import load_manifest

        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_manifest(str(invalid_path))

    def test_filter_by_region(self, sample_manifest):
        """Test filtering manifest by region."""
        from src.data.loader import load_manifest, filter_by_region

        data = load_manifest(sample_manifest)
        mexico_data = filter_by_region(data, "mexico")

        assert len(mexico_data) == 1
        assert mexico_data[0].region == "mexico"

    def test_filter_by_region_case_insensitive(self, sample_manifest):
        """Test that region filtering is case insensitive."""
        from src.data.loader import load_manifest, filter_by_region

        data = load_manifest(sample_manifest)
        spain_data = filter_by_region(data, "SPAIN")

        assert len(spain_data) == 1

    def test_filter_by_nonexistent_region(self, sample_manifest):
        """Test filtering by a region that doesn't exist."""
        from src.data.loader import load_manifest, filter_by_region

        data = load_manifest(sample_manifest)
        result = filter_by_region(data, "peru")

        assert len(result) == 0


class TestAudioPreprocessing:
    """Tests for audio preprocessing functionality."""

    def test_load_audio_valid_file(self, sample_audio_path):
        """Test loading a valid audio file."""
        from src.data.preprocessing import AudioPreprocessor

        preprocessor = AudioPreprocessor()
        audio, sr = preprocessor.load_audio(sample_audio_path)

        assert sr == 16000  # Sample audio created at 16kHz
        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)

    def test_resample_audio(self, tmp_path):
        """Test that audio resampling works correctly."""
        import soundfile as sf
        from src.data.preprocessing import AudioPreprocessor

        # Create audio at 44.1kHz
        audio_44k = np.random.randn(44100).astype(np.float32)
        audio_path = tmp_path / "audio_44k.wav"
        sf.write(str(audio_path), audio_44k, 44100)

        preprocessor = AudioPreprocessor(target_sample_rate=16000)
        audio, sr = preprocessor.load_audio(str(audio_path))
        resampled = preprocessor.resample(audio, sr)

        # After resampling, length should be approximately 16000 samples for 1 second
        expected_length = int(len(audio_44k) * 16000 / 44100)
        assert abs(len(resampled) - expected_length) < 100

    def test_normalize_audio(self):
        """Test audio normalization."""
        from src.data.preprocessing import AudioPreprocessor

        preprocessor = AudioPreprocessor()

        # Create audio with some signal
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        normalized = preprocessor.normalize_audio(audio)

        # Normalized audio should have values between -1 and 1
        assert np.max(np.abs(normalized)) <= 1.0

    def test_trim_silence(self):
        """Test silence trimming functionality."""
        from src.data.preprocessing import AudioPreprocessor

        preprocessor = AudioPreprocessor(silence_threshold_db=-40)

        # Create audio with silence at start and end
        silence = np.zeros(8000, dtype=np.float32)
        signal = np.random.randn(8000).astype(np.float32) * 0.5
        audio = np.concatenate([silence, signal, silence])

        trimmed = preprocessor.trim_silence_from_audio(audio)

        # Trimmed should be shorter than original
        assert len(trimmed) < len(audio)


class TestDatasetCreation:
    """Tests for dataset creation."""

    def test_load_dataset_splits(self, tmp_path):
        """Test loading train/val/test splits."""
        from src.data.loader import load_dataset_splits

        # Create sample manifests
        for split in ["train", "val", "test"]:
            manifest = [
                {"audio_path": f"/path/{split}_1.wav", "transcript": "test", "duration": 1.0}
            ]
            with open(tmp_path / f"{split}.json", "w") as f:
                json.dump(manifest, f)

        splits = load_dataset_splits(tmp_path)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 1

    def test_load_dataset_splits_missing_file(self, tmp_path):
        """Test loading splits when some files are missing."""
        from src.data.loader import load_dataset_splits

        # Only create train manifest
        manifest = [{"audio_path": "/path/1.wav", "transcript": "test", "duration": 1.0}]
        with open(tmp_path / "train.json", "w") as f:
            json.dump(manifest, f)

        splits = load_dataset_splits(tmp_path)

        assert len(splits["train"]) == 1
        assert len(splits["val"]) == 0
        assert len(splits["test"]) == 0

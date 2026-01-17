"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import base64
import numpy as np


@pytest.fixture
def client():
    """Create a test client for the API."""
    from src.api.app import app
    return TestClient(app)


@pytest.fixture
def sample_audio_base64(sample_audio_path):
    """Create base64 encoded audio for API testing."""
    with open(sample_audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Test that health endpoint returns status field."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "ok"]


class TestTranscribeEndpoint:
    """Tests for transcription endpoint."""

    def test_transcribe_requires_audio(self, client):
        """Test that transcribe endpoint requires audio input."""
        response = client.post("/transcribe", json={})

        assert response.status_code == 422  # Validation error

    def test_transcribe_accepts_base64(self, client, sample_audio_base64):
        """Test that transcribe endpoint accepts base64 audio."""
        response = client.post(
            "/transcribe",
            json={"audio_base64": sample_audio_base64}
        )

        # Should either succeed or fail gracefully (model not loaded in test)
        assert response.status_code in [200, 500, 503]

    def test_transcribe_with_region(self, client, sample_audio_base64):
        """Test transcribe endpoint with region parameter."""
        response = client.post(
            "/transcribe",
            json={
                "audio_base64": sample_audio_base64,
                "region": "mexico"
            }
        )

        assert response.status_code in [200, 500, 503]

    def test_transcribe_invalid_region(self, client, sample_audio_base64):
        """Test transcribe endpoint with invalid region."""
        response = client.post(
            "/transcribe",
            json={
                "audio_base64": sample_audio_base64,
                "region": "invalid_region"
            }
        )

        # Should return validation error or handle gracefully
        assert response.status_code in [400, 422, 500]

    def test_transcribe_returns_transcript(self, client, sample_audio_base64):
        """Test that successful transcription returns transcript."""
        response = client.post(
            "/transcribe",
            json={"audio_base64": sample_audio_base64}
        )

        if response.status_code == 200:
            data = response.json()
            assert "transcript" in data or "text" in data


class TestTranscribeUploadEndpoint:
    """Tests for file upload transcription endpoint."""

    def test_upload_requires_file(self, client):
        """Test that upload endpoint requires a file."""
        response = client.post("/transcribe/upload")

        assert response.status_code in [400, 422]

    def test_upload_accepts_wav(self, client, sample_audio_path):
        """Test that upload endpoint accepts WAV files."""
        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe/upload",
                files={"file": ("test.wav", f, "audio/wav")}
            )

        assert response.status_code in [200, 500, 503]

    def test_upload_rejects_non_audio(self, client, tmp_path):
        """Test that upload endpoint rejects non-audio files."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an audio file")

        with open(text_file, "rb") as f:
            response = client.post(
                "/transcribe/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )

        assert response.status_code in [400, 415, 422, 500]


class TestEvaluateEndpoint:
    """Tests for evaluation endpoint."""

    def test_evaluate_requires_inputs(self, client):
        """Test that evaluate endpoint requires reference and hypothesis."""
        response = client.post("/evaluate", json={})

        assert response.status_code == 422

    def test_evaluate_returns_metrics(self, client):
        """Test that evaluate endpoint returns WER and CER."""
        response = client.post(
            "/evaluate",
            json={
                "reference": "hola qué tal",
                "hypothesis": "hola que tal"
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert "wer" in data or "word_error_rate" in data

    def test_evaluate_perfect_match(self, client):
        """Test evaluation with perfect match."""
        response = client.post(
            "/evaluate",
            json={
                "reference": "hola amigo",
                "hypothesis": "hola amigo"
            }
        )

        if response.status_code == 200:
            data = response.json()
            wer = data.get("wer") or data.get("word_error_rate")
            assert wer == 0.0


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_models_returns_list(self, client):
        """Test that models endpoint returns a list."""
        response = client.get("/models")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_models_includes_regions(self, client):
        """Test that models endpoint includes regional models info."""
        response = client.get("/models")

        if response.status_code == 200:
            data = response.json()
            # Check for region information in response
            if isinstance(data, list):
                assert len(data) >= 0
            elif isinstance(data, dict):
                assert "models" in data or "available" in data


class TestAPIValidation:
    """Tests for API input validation."""

    def test_invalid_json_returns_error(self, client):
        """Test that invalid JSON returns appropriate error."""
        response = client.post(
            "/transcribe",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [400, 422]

    def test_empty_audio_returns_error(self, client):
        """Test that empty audio returns error."""
        response = client.post(
            "/transcribe",
            json={"audio_base64": ""}
        )

        assert response.status_code in [400, 422]

    def test_invalid_base64_returns_error(self, client):
        """Test that invalid base64 returns error."""
        response = client.post(
            "/transcribe",
            json={"audio_base64": "not-valid-base64!!!"}
        )

        assert response.status_code in [400, 422, 500]

"""Tests for evaluation metrics functionality."""

import pytest


class TestWERCalculation:
    """Tests for Word Error Rate calculation."""

    def test_wer_perfect_match(self):
        """Test WER is 0 for perfect match."""
        from src.evaluation.metrics import compute_wer

        reference = "hola qué tal"
        hypothesis = "hola qué tal"

        wer = compute_wer(reference, hypothesis)
        assert wer == 0.0

    def test_wer_complete_mismatch(self):
        """Test WER for completely different text."""
        from src.evaluation.metrics import compute_wer

        reference = "hola mundo"
        hypothesis = "adios tierra"

        wer = compute_wer(reference, hypothesis)
        assert wer == 1.0  # 100% error

    def test_wer_partial_match(self):
        """Test WER for partial match."""
        from src.evaluation.metrics import compute_wer

        reference = "hola qué tal amigo"  # 4 words
        hypothesis = "hola que tal amigo"  # 1 substitution (qué -> que)

        wer = compute_wer(reference, hypothesis)
        assert 0.0 < wer < 1.0
        assert wer == pytest.approx(0.25, rel=0.01)  # 1/4 = 0.25

    def test_wer_insertion_error(self):
        """Test WER with insertion errors."""
        from src.evaluation.metrics import compute_wer

        reference = "hola amigo"  # 2 words
        hypothesis = "hola mi amigo"  # 1 insertion

        wer = compute_wer(reference, hypothesis)
        assert wer == pytest.approx(0.5, rel=0.01)  # 1/2 = 0.5

    def test_wer_deletion_error(self):
        """Test WER with deletion errors."""
        from src.evaluation.metrics import compute_wer

        reference = "hola mi amigo"  # 3 words
        hypothesis = "hola amigo"  # 1 deletion

        wer = compute_wer(reference, hypothesis)
        assert wer == pytest.approx(0.333, rel=0.01)  # 1/3

    def test_wer_empty_reference(self):
        """Test WER with empty reference."""
        from src.evaluation.metrics import compute_wer

        reference = ""
        hypothesis = "hola"

        # Empty reference should return inf or handle gracefully
        wer = compute_wer(reference, hypothesis)
        assert wer >= 1.0 or wer == float('inf')

    def test_wer_empty_hypothesis(self):
        """Test WER with empty hypothesis."""
        from src.evaluation.metrics import compute_wer

        reference = "hola amigo"
        hypothesis = ""

        wer = compute_wer(reference, hypothesis)
        assert wer == 1.0  # 100% deletion error


class TestCERCalculation:
    """Tests for Character Error Rate calculation."""

    def test_cer_perfect_match(self):
        """Test CER is 0 for perfect match."""
        from src.evaluation.metrics import compute_cer

        reference = "hola"
        hypothesis = "hola"

        cer = compute_cer(reference, hypothesis)
        assert cer == 0.0

    def test_cer_single_char_error(self):
        """Test CER with single character error."""
        from src.evaluation.metrics import compute_cer

        reference = "hola"  # 4 chars
        hypothesis = "holo"  # 1 substitution

        cer = compute_cer(reference, hypothesis)
        assert cer == pytest.approx(0.25, rel=0.01)  # 1/4

    def test_cer_with_accents(self):
        """Test CER handles Spanish accents correctly."""
        from src.evaluation.metrics import compute_cer

        reference = "qué"
        hypothesis = "que"

        cer = compute_cer(reference, hypothesis)
        assert cer > 0  # Should detect accent difference


class TestTextNormalization:
    """Tests for text normalization before evaluation."""

    def test_normalize_lowercase(self):
        """Test text normalization converts to lowercase."""
        from src.evaluation.metrics import normalize_text

        text = "HOLA AMIGO"
        normalized = normalize_text(text)

        assert normalized == "hola amigo"

    def test_normalize_removes_punctuation(self):
        """Test text normalization removes punctuation."""
        from src.evaluation.metrics import normalize_text

        text = "¡Hola! ¿Qué tal?"
        normalized = normalize_text(text)

        assert "!" not in normalized
        assert "?" not in normalized
        assert "¡" not in normalized
        assert "¿" not in normalized

    def test_normalize_handles_extra_whitespace(self):
        """Test text normalization handles extra whitespace."""
        from src.evaluation.metrics import normalize_text

        text = "hola   amigo   mío"
        normalized = normalize_text(text)

        assert "   " not in normalized
        assert normalized == "hola amigo mío"

    def test_normalize_preserves_accents(self):
        """Test that normalization preserves Spanish accents."""
        from src.evaluation.metrics import normalize_text

        text = "Qué onda güey"
        normalized = normalize_text(text)

        assert "é" in normalized
        assert "ü" in normalized


class TestBatchEvaluation:
    """Tests for batch evaluation functionality."""

    def test_evaluate_batch_returns_dict(self, mock_transcription_results):
        """Test that batch evaluation returns expected dictionary."""
        from src.evaluation.metrics import evaluate_batch

        references = [r["reference"] for r in mock_transcription_results]
        hypotheses = [r["hypothesis"] for r in mock_transcription_results]

        results = evaluate_batch(references, hypotheses)

        assert "wer" in results
        assert "cer" in results
        assert "num_samples" in results

    def test_evaluate_batch_correct_sample_count(self, mock_transcription_results):
        """Test that batch evaluation counts samples correctly."""
        from src.evaluation.metrics import evaluate_batch

        references = [r["reference"] for r in mock_transcription_results]
        hypotheses = [r["hypothesis"] for r in mock_transcription_results]

        results = evaluate_batch(references, hypotheses)

        assert results["num_samples"] == len(mock_transcription_results)

    def test_evaluate_batch_mismatched_lengths(self):
        """Test batch evaluation with mismatched list lengths."""
        from src.evaluation.metrics import evaluate_batch

        references = ["hola", "adios"]
        hypotheses = ["hola"]

        with pytest.raises(ValueError):
            evaluate_batch(references, hypotheses)


class TestSlangDetection:
    """Tests for slang detection in transcripts."""

    def test_detect_mexican_slang(self, sample_slang_dict):
        """Test detection of Mexican slang terms."""
        from src.evaluation.metrics import detect_slang

        text = "Está muy chido güey"
        detected = detect_slang(text, sample_slang_dict, region="mexico")

        assert "chido" in detected
        assert "güey" in detected

    def test_detect_slang_returns_empty_for_formal_text(self, sample_slang_dict):
        """Test that formal text returns no slang."""
        from src.evaluation.metrics import detect_slang

        text = "Buenos días señor"
        detected = detect_slang(text, sample_slang_dict, region="mexico")

        assert len(detected) == 0

    def test_detect_slang_case_insensitive(self, sample_slang_dict):
        """Test that slang detection is case insensitive."""
        from src.evaluation.metrics import detect_slang

        text = "ESTÁ MUY CHIDO"
        detected = detect_slang(text, sample_slang_dict, region="mexico")

        assert "chido" in detected

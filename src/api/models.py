"""Pydantic models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class WordTiming(BaseModel):
    """Word with timing information."""
    word: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class TranscriptionRequest(BaseModel):
    """Request for transcription."""
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    language: str = Field("es", description="Language code (es, ca)")
    word_timestamps: bool = Field(False, description="Include word-level timestamps")

    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "https://example.com/audio.wav",
                "language": "es",
                "word_timestamps": True
            }
        }


class TranscriptionResponse(BaseModel):
    """Response with transcription results."""
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected or specified language")
    duration_seconds: float = Field(..., description="Audio duration")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    words: Optional[List[WordTiming]] = Field(None, description="Word-level timings")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model: str = Field(..., description="Model used for transcription")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hola, ¿cómo estás?",
                "language": "es",
                "duration_seconds": 2.5,
                "confidence": 0.95,
                "words": [
                    {"word": "Hola", "start": 0.0, "end": 0.5, "confidence": 0.98},
                    {"word": "cómo", "start": 0.6, "end": 0.9, "confidence": 0.94},
                    {"word": "estás", "start": 1.0, "end": 1.5, "confidence": 0.93}
                ],
                "processing_time_ms": 234.5,
                "model": "catalan-whisper-small"
            }
        }


class EvaluationRequest(BaseModel):
    """Request for transcription evaluation."""
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    expected_text: str = Field(..., description="Expected transcription for comparison")
    language: str = "es"


class EvaluationResponse(BaseModel):
    """Response with evaluation metrics."""
    transcription: str
    expected: str
    wer: float = Field(..., description="Word Error Rate (0-1)")
    cer: float = Field(..., description="Character Error Rate (0-1)")
    accuracy: float = Field(..., description="Accuracy percentage (0-100)")
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    timestamp: datetime
    version: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int

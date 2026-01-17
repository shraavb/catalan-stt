"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


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
    region: Optional[str] = Field(None, description="Region used for transcription (spain, mexico, argentina)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hola, ¿qué tal, tío?",
                "language": "es",
                "duration_seconds": 2.5,
                "confidence": 0.95,
                "words": [
                    {"word": "Hola", "start": 0.0, "end": 0.5, "confidence": 0.98},
                    {"word": "qué", "start": 0.6, "end": 0.8, "confidence": 0.94},
                    {"word": "tal", "start": 0.8, "end": 1.0, "confidence": 0.95},
                    {"word": "tío", "start": 1.1, "end": 1.5, "confidence": 0.93}
                ],
                "processing_time_ms": 234.5,
                "model": "spanish-slang-whisper-small",
                "region": "spain"
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
    supported_regions: Optional[List[str]] = Field(None, description="List of supported regions")
    loaded_regions: Optional[List[str]] = Field(None, description="Currently loaded region models")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int

"""FastAPI application for CatalanSTT."""

import os
import base64
import tempfile
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa

from .models import (
    TranscriptionRequest,
    TranscriptionResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    ErrorResponse,
    WordTiming,
)
from .transcriber import get_transcriber, Transcriber
from ..evaluation.metrics import compute_wer, compute_cer

logger = logging.getLogger(__name__)

# Version
VERSION = "0.1.0"


def create_app(
    model_path: Optional[str] = None,
    debug: bool = False,
) -> FastAPI:
    """Create FastAPI application."""

    app = FastAPI(
        title="CatalanSTT API",
        description="Speech-to-Text API optimized for Catalan-accented Spanish",
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store model path
    app.state.model_path = model_path

    @app.on_event("startup")
    async def startup():
        """Load model on startup."""
        logger.info("Loading transcription model...")
        try:
            get_transcriber(model_path=app.state.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API health and model status."""
        try:
            transcriber = get_transcriber()
            model_loaded = transcriber._model is not None
        except Exception:
            model_loaded = False

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_name=app.state.model_path or "openai/whisper-small",
            timestamp=datetime.now(),
            version=VERSION,
        )

    @app.post("/transcribe", response_model=TranscriptionResponse)
    async def transcribe(request: TranscriptionRequest):
        """Transcribe audio from URL or base64."""
        start_time = time.time()

        try:
            # Get audio data
            audio = await _get_audio_from_request(request)

            # Transcribe
            transcriber = get_transcriber()
            result = transcriber.transcribe(
                audio=audio,
                language=request.language,
                word_timestamps=request.word_timestamps,
            )

            processing_time = (time.time() - start_time) * 1000

            # Build response
            words = None
            if result.words:
                words = [
                    WordTiming(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.confidence,
                    )
                    for w in result.words
                ]

            return TranscriptionResponse(
                text=result.text,
                language=result.language,
                duration_seconds=result.duration_seconds,
                confidence=result.confidence,
                words=words,
                processing_time_ms=processing_time,
                model=app.state.model_path or "openai/whisper-small",
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/transcribe/upload", response_model=TranscriptionResponse)
    async def transcribe_upload(
        file: UploadFile = File(...),
        language: str = Form("es"),
        word_timestamps: bool = Form(False),
    ):
        """Transcribe uploaded audio file."""
        start_time = time.time()

        try:
            # Save uploaded file temporarily
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Load audio
            audio, _ = librosa.load(tmp_path, sr=16000)

            # Clean up
            os.unlink(tmp_path)

            # Transcribe
            transcriber = get_transcriber()
            result = transcriber.transcribe(
                audio=audio,
                language=language,
                word_timestamps=word_timestamps,
            )

            processing_time = (time.time() - start_time) * 1000

            words = None
            if result.words:
                words = [
                    WordTiming(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.confidence,
                    )
                    for w in result.words
                ]

            return TranscriptionResponse(
                text=result.text,
                language=result.language,
                duration_seconds=result.duration_seconds,
                confidence=result.confidence,
                words=words,
                processing_time_ms=processing_time,
                model=app.state.model_path or "openai/whisper-small",
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/evaluate", response_model=EvaluationResponse)
    async def evaluate(request: EvaluationRequest):
        """Transcribe and evaluate against expected text."""
        start_time = time.time()

        try:
            # Get audio
            audio = await _get_audio_from_request(request)

            # Transcribe
            transcriber = get_transcriber()
            result = transcriber.transcribe(
                audio=audio,
                language=request.language,
            )

            # Compute metrics
            wer = compute_wer(request.expected_text, result.text)
            cer = compute_cer(request.expected_text, result.text)
            accuracy = (1 - wer) * 100

            processing_time = (time.time() - start_time) * 1000

            return EvaluationResponse(
                transcription=result.text,
                expected=request.expected_text,
                wer=wer,
                cer=cer,
                accuracy=accuracy,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                status_code=exc.status_code,
            ).model_dump(),
        )

    return app


async def _get_audio_from_request(request) -> np.ndarray:
    """Extract audio from request (URL or base64)."""
    if request.audio_base64:
        # Decode base64
        audio_bytes = base64.b64decode(request.audio_base64)

        # Save to temp file and load
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio, _ = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)
        return audio

    elif request.audio_url:
        # Download and load
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(request.audio_url)
            response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        audio, _ = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)
        return audio

    else:
        raise HTTPException(
            status_code=400,
            detail="Either audio_url or audio_base64 must be provided"
        )


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

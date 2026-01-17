"""FastAPI application for SpanishSlangSTT.

Supports:
- Multi-region transcription (spain, mexico, argentina)
- Region auto-detection (experimental)
- Per-region model loading
"""

import base64
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
VERSION = "0.2.0"

# Supported regions
SUPPORTED_REGIONS = ["spain", "mexico", "argentina", "chile", "general"]

# Region to model path mapping (can be configured)
REGION_MODEL_PATHS: Dict[str, Optional[str]] = {
    "spain": None,  # Will use environment variable or default
    "mexico": None,
    "argentina": None,
    "chile": None,
    "general": None,
}


def create_app(
    model_path: Optional[str] = None,
    region_models: Optional[Dict[str, str]] = None,
    debug: bool = False,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        model_path: Default model path for general transcription
        region_models: Optional dict mapping region -> model_path for region-specific models
        debug: Enable debug mode
    """

    app = FastAPI(
        title="SpanishSlangSTT API",
        description="Speech-to-Text API optimized for regional Spanish slang and informal speech. "
                    "Supports Spain, Mexico, Argentina, and Chile regional variants.",
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

    # Store model paths
    app.state.model_path = model_path
    app.state.region_models = region_models or {}
    app.state.transcribers: Dict[str, Transcriber] = {}

    @app.on_event("startup")
    async def startup():
        """Load default model on startup."""
        logger.info("Loading default transcription model...")
        try:
            transcriber = get_transcriber(model_path=app.state.model_path)
            app.state.transcribers["general"] = transcriber
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def get_transcriber_for_region(region: str) -> Transcriber:
        """Get or load transcriber for a specific region."""
        # Check if already loaded
        if region in app.state.transcribers:
            return app.state.transcribers[region]

        # Check if we have a region-specific model path
        if region in app.state.region_models:
            model_path = app.state.region_models[region]
            logger.info(f"Loading model for region {region}: {model_path}")
            transcriber = get_transcriber(model_path=model_path)
            app.state.transcribers[region] = transcriber
            return transcriber

        # Fall back to general model
        if "general" in app.state.transcribers:
            return app.state.transcribers["general"]

        # Load general model
        return get_transcriber(model_path=app.state.model_path)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API health and model status."""
        try:
            transcriber = get_transcriber_for_region("general")
            model_loaded = transcriber._model is not None
        except Exception:
            model_loaded = False

        # Check loaded regions
        loaded_regions = list(app.state.transcribers.keys())

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_name=app.state.model_path or "openai/whisper-small",
            timestamp=datetime.now(),
            version=VERSION,
            supported_regions=SUPPORTED_REGIONS,
            loaded_regions=loaded_regions,
        )

    @app.get("/regions")
    async def list_regions():
        """List supported regions and their model status."""
        regions_status = {}
        for region in SUPPORTED_REGIONS:
            regions_status[region] = {
                "loaded": region in app.state.transcribers,
                "model_path": app.state.region_models.get(region) or app.state.model_path,
            }
        return {
            "supported_regions": SUPPORTED_REGIONS,
            "regions": regions_status,
        }

    @app.post("/transcribe", response_model=TranscriptionResponse)
    async def transcribe(
        request: TranscriptionRequest,
        region: Optional[str] = Query(
            default=None,
            description="Region for transcription (spain, mexico, argentina, general). "
                        "If not specified, uses general model.",
        ),
    ):
        """Transcribe audio from URL or base64.

        Optionally specify a region to use region-specific model for better
        accuracy with regional slang and informal speech.
        """
        start_time = time.time()

        # Validate region
        if region and region not in SUPPORTED_REGIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid region: {region}. Supported: {SUPPORTED_REGIONS}"
            )

        effective_region = region or "general"

        try:
            # Get audio data
            audio = await _get_audio_from_request(request)

            # Get transcriber for region
            transcriber = get_transcriber_for_region(effective_region)
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
                region=effective_region,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/transcribe/upload", response_model=TranscriptionResponse)
    async def transcribe_upload(
        file: UploadFile = File(...),
        language: str = Form("es"),
        word_timestamps: bool = Form(False),
        region: str = Form(None, description="Region (spain, mexico, argentina, general)"),
    ):
        """Transcribe uploaded audio file.

        Optionally specify a region to use region-specific model.
        """
        start_time = time.time()

        # Validate region
        if region and region not in SUPPORTED_REGIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid region: {region}. Supported: {SUPPORTED_REGIONS}"
            )

        effective_region = region or "general"

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

            # Get transcriber for region
            transcriber = get_transcriber_for_region(effective_region)
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
                region=effective_region,
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

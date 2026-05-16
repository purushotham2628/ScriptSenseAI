import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.router import api_router
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.ml.ocr import AncientOCRPipeline

settings = get_settings()
configure_logging()

ocr_pipeline = AncientOCRPipeline()

SUPPORTED_SOURCE_LANGUAGES = [
    {"code": "auto", "label": "Auto Detect", "ocr": ["en"], "note": "Best-effort script detection"},
    {"code": "la", "label": "Latin", "ocr": ["en"], "note": "Latin uses English-character OCR plus Latin translation"},
    {"code": "en", "label": "English", "ocr": ["en"], "note": "Native EasyOCR support"},
    {"code": "el", "label": "Ancient Greek / Greek", "ocr": ["en"], "note": "Falls back if Greek OCR model is unavailable locally"},
    {"code": "gr", "label": "Greek (alternate code)", "ocr": ["en"], "note": "Alias for Greek"},
    {"code": "ar", "label": "Arabic", "ocr": ["ar"], "note": "Uses Arabic OCR when model files exist"},
    {"code": "he", "label": "Hebrew", "ocr": ["en"], "note": "Falls back if Hebrew OCR model is unavailable locally"},
    {"code": "sa", "label": "Sanskrit / Devanagari", "ocr": ["hi", "en"], "note": "Best effort via Devanagari-compatible OCR"},
    {"code": "hi", "label": "Hindi / Devanagari", "ocr": ["hi", "en"], "note": "Best effort if Hindi OCR model exists"},
    {"code": "ta", "label": "Tamil", "ocr": ["ta", "en"], "note": "Best effort if Tamil OCR model exists"},
    {"code": "te", "label": "Telugu", "ocr": ["te", "en"], "note": "Best effort if Telugu OCR model exists"},
    {"code": "kn", "label": "Kannada", "ocr": ["kn", "en"], "note": "Best effort if Kannada OCR model exists"},
    {"code": "ml", "label": "Malayalam", "ocr": ["ml", "en"], "note": "Best effort if Malayalam OCR model exists"},
    {"code": "bn", "label": "Bengali", "ocr": ["bn", "en"], "note": "Best effort if Bengali OCR model exists"},
    {"code": "zh", "label": "Chinese", "ocr": ["ch_sim", "en"], "note": "Best effort if Chinese OCR model exists"},
    {"code": "ja", "label": "Japanese", "ocr": ["ja", "en"], "note": "Best effort if Japanese OCR model exists"},
    {"code": "ko", "label": "Korean", "ocr": ["ko", "en"], "note": "Best effort if Korean OCR model exists"},
    {"code": "unknown", "label": "Unknown / Unseen Script", "ocr": ["en"], "note": "Uses preprocessing, anomaly-aware OCR, and low-confidence review"},
]


def encode_image_to_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("utf-8")

app = FastAPI(
    title=settings.app_name,
    description="Research-grade AI backend for ancient script recognition, unseen-script adaptation, and continual learning.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=f"/api/{settings.api_version}")

static_dir = Path(__file__).resolve().parent.parent / "frontend" / "static"
templates_dir = Path(__file__).resolve().parent.parent / "frontend" / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_path = templates_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend template not found")
    return FileResponse(index_path)


@app.get("/api")
async def api_info():
    return {
        "name": settings.app_name,
        "version": "2.0.0",
        "architecture": "modular_research_grade_backend",
        "primary_api": f"/api/{settings.api_version}",
        "pipeline": "INPUT_IMAGE -> IMAGE_QUALITY_ANALYSIS -> ADAPTIVE_PREPROCESSING -> SUPER_RESOLUTION -> LINE_SEGMENTATION -> MULTI_OCR_ENSEMBLE -> CONFIDENCE_FUSION -> OCR_CORRECTION -> LANGUAGE_DETECTION -> CONFIDENCE_AWARE_TRANSLATION",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/languages")
async def get_supported_languages():
    return {
        "source_languages": SUPPORTED_SOURCE_LANGUAGES,
        "target_languages": [{"code": "en", "label": "English"}],
        "default_source": "la",
        "default_target": "en",
    }


@app.post("/process")
async def legacy_process(
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en"),
    source_lang: Optional[str] = Form(None),
):
    """Compatibility route for the existing frontend.

    New clients should use /api/v1/inference/predict with JWT auth. This legacy
    endpoint remains unauthenticated so your current frontend keeps working.
    """
    try:
        contents = await file.read()
        requested_source_language = source_lang or source_language or "la"
        requested_target_language = target_language or "en"

        result = ocr_pipeline.process_bytes(
            contents,
            source_language=requested_source_language,
            target_language=requested_target_language,
        )
        preprocessed_b64 = encode_image_to_base64(result.preprocessing_preview)
        detection_b64 = encode_image_to_base64(result.segmentation_preview)
        api_result = result.to_api_dict()
        confidence = float(result.confidence)
        unknown_or_low_confidence = confidence < 0.50 or not result.ocr_text.strip()
        translation_details = result.details.get("translation", {})
        line_details = next(
            (step.get("details", {}) for step in result.processing_steps if step.get("name") == "line_segmentation"),
            {},
        )

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            **api_result,
            "raw_text": result.ocr_text,
            "cleaned_text": result.cleaned_text,
            "translated_text": result.translated_text,
            "source_language": result.detected_language,
            "target_language": requested_target_language,
            "confidence": confidence,
            "note": "; ".join(result.warnings) or translation_details.get("note", ""),
            "pipeline": {
                "preprocessing": {
                    "status": "completed",
                    "image": preprocessed_b64,
                    "info": "Quality analysis, illumination normalization, CLAHE, edge-preserving denoise, adaptive threshold preview",
                },
                "detection": {
                    "status": "completed",
                    "image": detection_b64,
                    "regions_found": line_details.get("line_count", 0),
                    "detections": line_details.get("lines", [])[:25],
                },
                "recognition": {
                    "status": "completed",
                    "recognized_texts": [candidate.text for candidate in result.candidates[:25]],
                    "raw_text": result.ocr_text,
                    "confidence": confidence,
                    "engine": result.ocr_engine_used,
                },
                "cleaning": {
                    "status": "completed",
                    "original_length": len(result.ocr_text),
                    "cleaned_length": len(result.cleaned_text),
                    "cleaned_text": result.cleaned_text,
                },
                "translation": {
                    "status": "completed",
                    "source_language": result.detected_language,
                    "target_language": requested_target_language,
                    "original_text": result.cleaned_text,
                    "translated_text": result.translated_text,
                    "translated": translation_details.get("translated", False),
                    "note": translation_details.get("note", ""),
                },
            },
            "final_output": {
                "extracted_text": result.cleaned_text,
                "translated_text": result.translated_text,
                "confidence_score": confidence,
            },
            "requires_human_review": unknown_or_low_confidence,
            "anomaly_score": result.details.get("anomaly_score", 1.0 - confidence),
            "nearest_symbols": [],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

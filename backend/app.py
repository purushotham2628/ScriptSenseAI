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
from utils.preprocessing import ImagePreprocessor
from utils.text_cleaning import TextCleaner
from utils.text_detection import TextDetector
from utils.translation import TextTranslator

settings = get_settings()
configure_logging()

preprocessor = ImagePreprocessor()
detector = TextDetector(languages=["en"])
cleaner = TextCleaner()
translator = TextTranslator(source_lang="la", target_lang="en")

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
        "pipeline": "UPLOAD -> PREPROCESS -> SEGMENTATION -> FEATURE_EXTRACTION -> OCR -> CONTEXT_CORRECTION -> TRANSLATION -> STORAGE -> VISUALIZATION",
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
        if requested_source_language == "auto":
            requested_source_language = "la"
        requested_target_language = target_language or "en"

        original, processed = preprocessor.preprocess_complete_pipeline(image_bytes=contents)
        ocr_image = preprocessor.ocr_image if preprocessor.ocr_image is not None else processed
        preprocessed_b64 = encode_image_to_base64(processed)

        ocr_result = detector.extract_text(
            ocr_image,
            confidence_threshold=0.2,
            source_lang=requested_source_language,
        )
        detections = ocr_result["detections"]
        detection_image = detector._draw_bounding_boxes(ocr_image, detections)
        detection_b64 = encode_image_to_base64(detection_image)

        raw_text = ocr_result["raw_text"]
        recognized_texts = [item["text"] for item in ocr_result["results"]]
        cleaned_text = cleaner.clean_alpha_text(raw_text)
        cleaned_stats = cleaner.get_cleaning_stats(raw_text, cleaned_text)

        translation_result = translator.translate_text(
            cleaned_text,
            source_lang=ocr_result["source_language"],
            target_lang=requested_target_language,
        )
        notes = [
            ocr_result.get("note", ""),
            translation_result.get("note", ""),
        ]
        note = "; ".join(dict.fromkeys(item for item in notes if item))
        confidence = float(ocr_result["confidence"])
        unknown_or_low_confidence = confidence < 0.35 or not raw_text.strip()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "translated_text": translation_result["translated_text"],
            "source_language": ocr_result["source_language"],
            "target_language": requested_target_language,
            "confidence": confidence,
            "note": note,
            "pipeline": {
                "preprocessing": {
                    "status": "completed",
                    "image": preprocessed_b64,
                    "info": "Grayscale, resize, brightness normalization, CLAHE, blur, adaptive threshold",
                },
                "detection": {
                    "status": "completed",
                    "image": detection_b64,
                    "regions_found": len(detections),
                    "detections": detections[:25],
                },
                "recognition": {
                    "status": "completed",
                    "recognized_texts": recognized_texts[:25],
                    "raw_text": raw_text,
                    "confidence": confidence,
                },
                "cleaning": {
                    "status": "completed",
                    "original_length": cleaned_stats["original_length"],
                    "cleaned_length": cleaned_stats["cleaned_length"],
                    "cleaned_text": cleaned_text,
                },
                "translation": {
                    "status": "completed",
                    "source_language": ocr_result["source_language"],
                    "target_language": requested_target_language,
                    "original_text": cleaned_text,
                    "translated_text": translation_result["translated_text"],
                    "translated": translation_result["translated"],
                    "note": translation_result.get("note", ""),
                },
            },
            "final_output": {
                "extracted_text": cleaned_text,
                "translated_text": translation_result["translated_text"],
                "confidence_score": confidence,
            },
            "requires_human_review": unknown_or_low_confidence,
            "anomaly_score": 1.0 - confidence,
            "nearest_symbols": [],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

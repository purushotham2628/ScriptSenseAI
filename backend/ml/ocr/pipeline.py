from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from backend.ml.ocr.confidence import ConfidenceFusion
from backend.ml.ocr.correction import OCRCorrector
from backend.ml.ocr.engines import MultiOCREnsemble, OCRCandidate
from backend.ml.ocr.language import ScriptLanguageDetector
from backend.ml.ocr.preprocessing import AdaptivePreprocessor
from backend.ml.ocr.segmentation import LineSegmenter
from backend.ml.ocr.super_resolution import SuperResolutionService
from backend.ml.ocr.translation import ConfidenceAwareTranslator


ProgressCallback = Optional[Callable[[dict], None]]


@dataclass
class AncientOCRResult:
    detected_language: str
    ocr_text: str
    cleaned_text: str
    translated_text: str
    confidence: float
    image_quality_score: float
    ocr_engine_used: str
    processing_steps: list[dict]
    warnings: list[str]
    preprocessing_preview: np.ndarray
    segmentation_preview: np.ndarray
    candidates: list[OCRCandidate]
    timing_ms: dict[str, float]
    details: dict

    def to_api_dict(self) -> dict:
        return {
            "detected_language": self.detected_language,
            "ocr_text": self.ocr_text,
            "cleaned_text": self.cleaned_text,
            "translated_text": self.translated_text,
            "confidence": self.confidence,
            "image_quality_score": self.image_quality_score,
            "ocr_engine_used": self.ocr_engine_used,
            "processing_steps": self.processing_steps,
            "warnings": self.warnings,
            "timing_ms": self.timing_ms,
            "details": self.details,
        }


class AncientOCRPipeline:
    """INPUT -> quality -> adaptive preprocess -> SR -> lines -> ensemble -> safe translation."""

    def __init__(self) -> None:
        self.preprocessor = AdaptivePreprocessor()
        self.super_resolution = SuperResolutionService()
        self.segmenter = LineSegmenter()
        self.ocr = MultiOCREnsemble()
        self.corrector = OCRCorrector()
        self.language_detector = ScriptLanguageDetector()
        self.confidence = ConfidenceFusion()
        self.translator = ConfidenceAwareTranslator()

    def process_bytes(
        self,
        image_bytes: bytes,
        source_language: str = "auto",
        target_language: str = "en",
        progress: ProgressCallback = None,
    ) -> AncientOCRResult:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode uploaded image.")
        return self.process_image(image, source_language, target_language, progress)

    def process_image(
        self,
        image: np.ndarray,
        source_language: str = "auto",
        target_language: str = "en",
        progress: ProgressCallback = None,
    ) -> AncientOCRResult:
        steps: list[dict] = []
        timing: dict[str, float] = {}

        def timed(name: str, fn):
            self._emit(progress, name, "started")
            start = time.perf_counter()
            value = fn()
            timing[name] = round((time.perf_counter() - start) * 1000, 2)
            self._emit(progress, name, "completed", {"duration_ms": timing[name]})
            return value

        ocr_image, binary, quality, prep_steps = timed("adaptive_preprocessing", lambda: self.preprocessor.run(image))
        steps.extend(prep_steps)

        sr_image, sr_meta = timed("super_resolution", lambda: self.super_resolution.upscale(ocr_image, quality.readability))
        if sr_image.shape[:2] != binary.shape[:2]:
            binary = cv2.resize(binary, (sr_image.shape[1], sr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        steps.append({"name": "super_resolution", "status": "completed", "details": sr_meta})

        line_images, line_meta = timed("line_segmentation", lambda: self.segmenter.segment(sr_image, binary))
        steps.append({"name": "line_segmentation", "status": "completed", "details": {"line_count": len(line_images), "lines": line_meta[:50]}})

        raw_text, ocr_confidence, engine, candidates, ocr_meta = timed(
            "multi_ocr_ensemble",
            lambda: self.ocr.recognize(line_images, source_language),
        )
        steps.append({"name": "multi_ocr_ensemble", "status": "completed", "confidence": ocr_confidence, "details": ocr_meta})

        language = timed("language_detection", lambda: self.language_detector.detect(raw_text, source_language))
        steps.append({"name": "language_detection", "status": "completed", "confidence": language.confidence, "details": language.details})

        correction = timed("ocr_correction", lambda: self.corrector.correct(raw_text, language.language))
        corrected_text = correction["text"]
        steps.append({"name": "ocr_correction", "status": "completed", "details": correction})

        fused = self.confidence.fuse(
            ocr_confidence=ocr_confidence + correction.get("confidence_delta", 0.0),
            quality=quality,
            language_confidence=language.confidence,
            corrected_text=corrected_text,
            unknown_ratio=correction["unknown_symbol_ratio"],
        )
        steps.append({"name": "confidence_fusion", "status": "completed", "confidence": fused["confidence"], "details": fused["components"]})

        anomaly_score = self._anomaly_score(language.is_unknown, correction["unknown_symbol_ratio"], fused["confidence"])
        warnings = list(fused["warnings"])
        if language.is_unknown:
            warnings.append(f"Unknown script detected; nearest script similarity: {language.nearest_script}.")
        if anomaly_score > 0.50:
            warnings.append("Anomaly detection flagged this sample as an unseen or heavily degraded script.")

        translation = timed(
            "confidence_aware_translation",
            lambda: self.translator.translate(corrected_text, language.language, target_language, fused["confidence"], language.is_unknown),
        )
        steps.append({"name": "translation", "status": "completed", "details": translation})

        return AncientOCRResult(
            detected_language=language.language,
            ocr_text=raw_text,
            cleaned_text=corrected_text,
            translated_text=translation["translated_text"],
            confidence=round(float(fused["confidence"]), 4),
            image_quality_score=round(float(quality.readability), 4),
            ocr_engine_used=engine,
            processing_steps=steps,
            warnings=warnings,
            preprocessing_preview=ocr_image,
            segmentation_preview=self._draw_lines(sr_image, line_meta),
            candidates=candidates,
            timing_ms=timing,
            details={
                "quality": quality.to_dict(),
                "language": language.__dict__,
                "confidence": fused,
                "translation": translation,
                "unknown_symbol_ratio": correction["unknown_symbol_ratio"],
                "anomaly_score": anomaly_score,
            },
        )

    def _draw_lines(self, image: np.ndarray, line_meta: list[dict]) -> np.ndarray:
        preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        for line in line_meta:
            x, y, w, h = line["x"], line["y"], line["w"], line["h"]
            cv2.rectangle(preview, (x, y), (x + w, y + h), (40, 210, 80), 2)
        return preview

    def _anomaly_score(self, is_unknown: bool, unknown_ratio: float, confidence: float) -> float:
        return round(float(np.clip((0.35 if is_unknown else 0.0) + 0.35 * unknown_ratio + 0.30 * (1.0 - confidence), 0.0, 1.0)), 4)

    def _emit(self, progress: ProgressCallback, step: str, status: str, details: dict | None = None) -> None:
        if progress:
            progress({"step": step, "status": status, "details": details or {}, "timestamp": time.time()})

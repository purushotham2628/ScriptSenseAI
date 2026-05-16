from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2

from backend.core.progress import progress_hub
from backend.ml.ocr import AncientOCRPipeline
from backend.schemas.platform import PipelineStage, PredictionResponse, PredictionStatus


class ResearchGradeInferencePipeline:
    """Compatibility wrapper around the robust manuscript OCR pipeline."""

    def __init__(self) -> None:
        self.pipeline = AncientOCRPipeline()

    async def predict_image(
        self,
        image_path: Path,
        source_language: str = "auto",
        target_language: str = "en",
        dataset_id: Optional[str] = None,
    ) -> PredictionResponse:
        prediction_id = f"pred_{uuid4().hex}"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        def publish(event: dict) -> None:
            progress_hub.publish_threadsafe(prediction_id, event)

        publish({"step": "queued", "status": "completed", "details": {"dataset_id": dataset_id}})
        result = self.pipeline.process_image(image, source_language, target_language, progress=publish)
        unknown_ratio = float(result.details.get("unknown_symbol_ratio", 1.0))
        anomaly = float(result.details.get("anomaly_score", 1.0))
        stages = [
            PipelineStage(
                name=step.get("name", "unknown"),
                status=step.get("status", "completed"),
                confidence=step.get("confidence"),
                details=step.get("details", {}),
            )
            for step in result.processing_steps
        ]
        requires_review = result.confidence < 0.72 or anomaly > 0.45 or unknown_ratio > 0.12
        publish({"step": "completed", "status": "completed", "details": {"confidence": result.confidence}})
        return PredictionResponse(
            prediction_id=prediction_id,
            status=PredictionStatus.needs_review if requires_review else PredictionStatus.completed,
            raw_text=result.ocr_text,
            corrected_text=result.cleaned_text,
            translated_text=result.translated_text,
            confidence=result.confidence,
            detected_language=result.detected_language,
            image_quality_score=result.image_quality_score,
            ocr_engine_used=result.ocr_engine_used,
            processing_steps=result.processing_steps,
            warnings=result.warnings,
            timing_ms=result.timing_ms,
            unknown_symbol_ratio=unknown_ratio,
            anomaly_score=anomaly,
            nearest_symbols=[],
            stages=stages,
            requires_human_review=requires_review,
        )

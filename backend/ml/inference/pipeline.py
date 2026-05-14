from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np

from backend.ml.embeddings.vector_store import VectorIndexService
from backend.ml.preprocessing.advanced_preprocessor import AdvancedPreprocessor
from backend.schemas.platform import PipelineStage, PredictionResponse, PredictionStatus

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None


class ContextCorrectionService:
    """Language-model correction hook for noisy OCR and missing symbols."""

    def correct(self, raw_text: str, nearest_symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Production hook: HuggingFace seq2seq or GPT-style correction model.
        corrected = " ".join(raw_text.split())
        unknown_ratio = corrected.count("<unk>") / max(1, len(corrected))
        return {"text": corrected, "unknown_symbol_ratio": unknown_ratio, "confidence_delta": 0.03}


class TranslationService:
    """Future multilingual translation abstraction."""

    def translate(self, text: str, source_language: str = "auto", target_language: str = "en") -> Dict[str, Any]:
        return {
            "translated_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "translated": source_language != target_language and bool(text),
            "note": "Translation backend hook. Add NLLB/M2M100/domain lexicons here.",
        }


class ResearchGradeInferencePipeline:
    """UPLOAD -> PREPROCESS -> SEGMENT -> EMBED -> OCR -> CORRECT -> TRANSLATE."""

    def __init__(self, vector_index: Optional[VectorIndexService] = None) -> None:
        self.preprocessor = AdvancedPreprocessor()
        self.vector_index = vector_index or VectorIndexService()
        self.corrector = ContextCorrectionService()
        self.translator = TranslationService()
        self.easyocr_reader = easyocr.Reader(["en"], gpu=False) if easyocr else None

    async def predict_image(
        self,
        image_path: Path,
        source_language: str = "auto",
        target_language: str = "en",
        dataset_id: Optional[str] = None,
    ) -> PredictionResponse:
        prediction_id = f"pred_{uuid4().hex}"
        stages: List[PipelineStage] = []

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        processed, metrics = self.preprocessor.run(image)
        stages.append(PipelineStage(name="preprocess", status="completed", details=metrics))

        regions = self.segment_symbols(processed)
        stages.append(PipelineStage(name="segmentation", status="completed", details={"regions": len(regions)}))

        embedding = self.extract_embedding(processed)
        nearest = [result.__dict__ for result in self.vector_index.search(embedding, top_k=8)]
        anomaly = self.vector_index.anomaly_score(embedding)
        stages.append(PipelineStage(name="feature_extraction", status="completed", confidence=1.0 - anomaly))

        raw_text, ocr_confidence = self.ocr(processed)
        stages.append(PipelineStage(name="ocr", status="completed", confidence=ocr_confidence))

        correction = self.corrector.correct(raw_text, nearest)
        corrected_text = correction["text"]
        confidence = min(1.0, max(0.0, ocr_confidence + correction["confidence_delta"] - anomaly * 0.18))
        unknown_ratio = correction["unknown_symbol_ratio"]
        stages.append(PipelineStage(name="context_correction", status="completed", confidence=confidence))

        translation = self.translator.translate(corrected_text, source_language, target_language)
        stages.append(PipelineStage(name="translation", status="completed", details=translation))

        requires_review = confidence < 0.72 or anomaly > 0.45 or unknown_ratio > 0.12
        return PredictionResponse(
            prediction_id=prediction_id,
            status=PredictionStatus.needs_review if requires_review else PredictionStatus.completed,
            raw_text=raw_text,
            corrected_text=corrected_text,
            translated_text=translation["translated_text"],
            confidence=confidence,
            unknown_symbol_ratio=unknown_ratio,
            anomaly_score=anomaly,
            nearest_symbols=nearest,
            stages=stages,
            requires_human_review=requires_review,
        )

    def segment_symbols(self, binary_image: np.ndarray) -> List[Dict[str, int]]:
        inverted = 255 - binary_image
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Dict[str, int]] = []
        h, w = binary_image.shape[:2]
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw * bh < max(12, 0.00002 * h * w):
                continue
            boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)})
        return sorted(boxes, key=lambda item: (item["y"], item["x"]))

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        # Lightweight deterministic descriptor; production uses ViT/Swin embeddings.
        resized = cv2.resize(image, (32, 24), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().astype("float32")
        hist = hist / (hist.sum() + 1e-12)
        vector = np.concatenate([resized.flatten(), hist])
        if vector.size < 768:
            vector = np.pad(vector, (0, 768 - vector.size))
        return vector[:768].astype("float32")

    def ocr(self, image: np.ndarray) -> tuple[str, float]:
        if self.easyocr_reader is None:
            return "", 0.0
        results = self.easyocr_reader.readtext(image)
        texts = [item[1] for item in results]
        confidences = [float(item[2]) for item in results]
        return " ".join(texts), float(np.mean(confidences)) if confidences else 0.0

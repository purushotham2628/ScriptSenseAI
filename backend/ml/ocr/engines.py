from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import cv2
import numpy as np


@dataclass
class OCRCandidate:
    engine: str
    text: str
    confidence: float
    line_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiOCREnsemble:
    """Run available OCR engines and return line-aware candidates."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or ["en"]
        self._easyocr_reader = None
        self._paddle_reader = None
        self._trocr_processor = None
        self._trocr_model = None
        self._availability: dict[str, str] = {}
        self.enable_heavy_engines = os.getenv("SCRIPTSENSE_ENABLE_HEAVY_OCR", "0") == "1"

    def recognize(self, lines: list[np.ndarray], language_hint: str = "auto") -> tuple[str, float, str, list[OCRCandidate], dict]:
        all_candidates: list[OCRCandidate] = []
        selected_lines: list[OCRCandidate] = []

        for line_index, line in enumerate(lines):
            candidates = self._recognize_line(line, line_index, language_hint)
            all_candidates.extend(candidates)
            best = self._choose_best_candidate(candidates)
            if best:
                selected_lines.append(best)

        text = "\n".join(item.text for item in selected_lines if item.text.strip()).strip()
        confidence = float(np.mean([item.confidence for item in selected_lines])) if selected_lines else 0.0
        engine = self._dominant_engine(selected_lines)
        metadata = {
            "available_engines": sorted({item.engine for item in all_candidates}),
            "engine_status": self._availability,
            "line_count": len(lines),
            "candidate_count": len(all_candidates),
        }
        return text, confidence, engine, all_candidates, metadata

    def _recognize_line(self, line: np.ndarray, line_index: int, language_hint: str) -> list[OCRCandidate]:
        return [
            *self._easyocr(line, line_index, language_hint),
            *self._paddleocr(line, line_index),
            *self._tesseract(line, line_index, language_hint),
            *self._trocr(line, line_index),
        ]

    def _easyocr(self, image: np.ndarray, line_index: int, language_hint: str) -> list[OCRCandidate]:
        try:
            if self._easyocr_reader is None:
                import easyocr

                langs = self._easyocr_languages(language_hint)
                self._easyocr_reader = easyocr.Reader(langs, gpu=False, download_enabled=False)
                self._availability["easyocr"] = "available"
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._easyocr_reader.readtext(rgb, detail=1, paragraph=False)
            candidates = []
            for result in results:
                candidates.append(OCRCandidate("easyocr", str(result[1]).strip(), float(result[2]), line_index, {"bbox": result[0]}))
            if results:
                joined = " ".join(str(item[1]).strip() for item in results if str(item[1]).strip())
                conf = float(np.mean([float(item[2]) for item in results]))
                candidates.append(OCRCandidate("easyocr_line", joined, conf, line_index))
            return candidates
        except Exception as exc:
            self._availability["easyocr"] = f"unavailable: {exc}"
            return []

    def _paddleocr(self, image: np.ndarray, line_index: int) -> list[OCRCandidate]:
        if not self.enable_heavy_engines:
            self._availability["paddleocr"] = "disabled; set SCRIPTSENSE_ENABLE_HEAVY_OCR=1"
            return []
        try:
            if self._paddle_reader is None:
                from paddleocr import PaddleOCR

                self._paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self._availability["paddleocr"] = "available"
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
            output = self._paddle_reader.ocr(rgb, cls=True)
            candidates = []
            for page in output or []:
                for item in page or []:
                    text, conf = item[1][0], item[1][1]
                    candidates.append(OCRCandidate("paddleocr", str(text).strip(), float(conf), line_index, {"bbox": item[0]}))
            return candidates
        except Exception as exc:
            self._availability["paddleocr"] = f"unavailable: {exc}"
            return []

    def _tesseract(self, image: np.ndarray, line_index: int, language_hint: str) -> list[OCRCandidate]:
        try:
            import pytesseract

            lang = {"la": "lat", "el": "ell", "gr": "ell", "en": "eng"}.get(language_hint, "eng")
            data = pytesseract.image_to_data(image, lang=lang, config="--psm 7", output_type=pytesseract.Output.DICT)
            tokens, confidences = [], []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                text = str(text).strip()
                try:
                    conf_float = float(conf) / 100.0
                except ValueError:
                    conf_float = -1.0
                if text and conf_float >= 0:
                    tokens.append(text)
                    confidences.append(conf_float)
            self._availability["tesseract"] = "available"
            return [OCRCandidate("tesseract", " ".join(tokens), float(np.mean(confidences)) if confidences else 0.0, line_index)] if tokens else []
        except Exception as exc:
            self._availability["tesseract"] = f"unavailable: {exc}"
            return []

    def _trocr(self, image: np.ndarray, line_index: int) -> list[OCRCandidate]:
        if not self.enable_heavy_engines:
            self._availability["trocr"] = "disabled; set SCRIPTSENSE_ENABLE_HEAVY_OCR=1"
            return []
        try:
            if self._trocr_processor is None or self._trocr_model is None:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                self._trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                self._availability["trocr"] = "available"
            from PIL import Image
            import torch

            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pixel_values = self._trocr_processor(images=pil_image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = self._trocr_model.generate(pixel_values, num_beams=4, max_length=96)
            text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            confidence = 0.62 if text else 0.0
            return [OCRCandidate("trocr", text, confidence, line_index)] if text else []
        except Exception as exc:
            self._availability["trocr"] = f"unavailable: {exc}"
            return []

    def _choose_best_candidate(self, candidates: list[OCRCandidate]) -> OCRCandidate | None:
        valid = [item for item in candidates if item.text.strip()]
        if not valid:
            return None
        return max(valid, key=lambda item: item.confidence * 0.68 + self._coherence(item.text) * 0.32)

    def _coherence(self, text: str) -> float:
        chars = [ch for ch in text if not ch.isspace()]
        if not chars:
            return 0.0
        alpha_ratio = sum(ch.isalpha() for ch in chars) / len(chars)
        repeat_penalty = 0.0
        for ch in set(chars):
            if text.count(ch * 4):
                repeat_penalty += 0.1
        return float(np.clip(alpha_ratio - repeat_penalty, 0.0, 1.0))

    def _dominant_engine(self, candidates: list[OCRCandidate]) -> str:
        if not candidates:
            return "none"
        weighted: dict[str, float] = {}
        for item in candidates:
            engine = item.engine.replace("_line", "")
            weighted[engine] = weighted.get(engine, 0.0) + item.confidence
        return max(weighted.items(), key=lambda item: item[1])[0]

    def _easyocr_languages(self, language_hint: str) -> list[str]:
        mapping = {
            "ar": ["ar", "en"],
            "hi": ["hi", "en"],
            "sa": ["hi", "en"],
            "ta": ["ta", "en"],
            "te": ["te", "en"],
            "kn": ["kn", "en"],
            "ml": ["ml", "en"],
            "bn": ["bn", "en"],
            "zh": ["ch_sim", "en"],
            "ja": ["ja", "en"],
            "ko": ["ko", "en"],
        }
        return mapping.get(language_hint, ["en"])

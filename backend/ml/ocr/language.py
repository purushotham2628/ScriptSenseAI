from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    nearest_script: str
    is_unknown: bool
    details: dict


class ScriptLanguageDetector:
    """Language/script detection with optional fastText/langdetect backends."""

    BRAHMI_RANGE = (0x11000, 0x1107F)

    def detect(self, text: str, requested_language: str = "auto") -> LanguageDetectionResult:
        text = text or ""
        if requested_language and requested_language not in {"auto", "unknown"}:
            heuristic = self._heuristic(text)
            return LanguageDetectionResult(requested_language, max(0.45, heuristic["confidence"]), heuristic["script"], False, heuristic)

        fasttext_result = self._fasttext(text)
        if fasttext_result:
            return fasttext_result

        langdetect_result = self._langdetect(text)
        heuristic = self._heuristic(text)
        if langdetect_result and langdetect_result.confidence >= heuristic["confidence"]:
            return langdetect_result

        language = heuristic["language"]
        unknown = language == "unknown" or heuristic["confidence"] < 0.35
        return LanguageDetectionResult(language, heuristic["confidence"], heuristic["script"], unknown, heuristic)

    def _fasttext(self, text: str) -> LanguageDetectionResult | None:
        try:
            import fasttext  # type: ignore

            # Caller can inject a real lid model in production by monkey-patching this
            # attribute; we avoid a network download at runtime.
            model = getattr(self, "_fasttext_model", None)
            if model is None:
                return None
            labels, probs = model.predict(text.replace("\n", " "), k=1)
            lang = labels[0].replace("__label__", "")
            return LanguageDetectionResult(lang, float(probs[0]), lang, False, {"backend": "fastText"})
        except Exception:
            return None

    def _langdetect(self, text: str) -> LanguageDetectionResult | None:
        try:
            from langdetect import detect_langs

            predictions = detect_langs(text)
            if not predictions:
                return None
            best = predictions[0]
            language = {"el": "gr"}.get(best.lang, best.lang)
            return LanguageDetectionResult(language, float(best.prob), language, False, {"backend": "langdetect"})
        except Exception:
            return None

    def _heuristic(self, text: str) -> dict:
        if not text.strip():
            return {"language": "unknown", "confidence": 0.0, "script": "unknown", "backend": "heuristic"}
        counts = {
            "greek": len(re.findall(r"[\u0370-\u03FF]", text)),
            "devanagari": len(re.findall(r"[\u0900-\u097F]", text)),
            "arabic": len(re.findall(r"[\u0600-\u06FF]", text)),
            "hebrew": len(re.findall(r"[\u0590-\u05FF]", text)),
            "latin": len(re.findall(r"[A-Za-z]", text)),
            "brahmi": sum(self.BRAHMI_RANGE[0] <= ord(ch) <= self.BRAHMI_RANGE[1] for ch in text),
        }
        total = sum(counts.values()) or len(text)
        script, count = max(counts.items(), key=lambda item: item[1])
        confidence = float(np.clip(count / max(total, 1), 0.0, 1.0))
        mapping = {"greek": "gr", "devanagari": "sa", "arabic": "ar", "hebrew": "he", "latin": "la", "brahmi": "brahmi"}
        if confidence < 0.20:
            return {"language": "unknown", "confidence": confidence, "script": script, "backend": "heuristic", "script_counts": counts}
        return {"language": mapping[script], "confidence": confidence, "script": script, "backend": "heuristic", "script_counts": counts}

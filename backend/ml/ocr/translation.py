from __future__ import annotations

from typing import Dict
import os

from utils.translation import TextTranslator


class ConfidenceAwareTranslator:
    """Translate only after correction/language detection and confidence gates."""

    SAFE_FAILURE_TEXT = "Text too degraded for reliable translation."

    def __init__(self) -> None:
        self._latin_translator = TextTranslator(source_lang="la", target_lang="en")
        self._hf_pipelines: dict[str, object] = {}
        self.enable_translation_models = os.getenv("SCRIPTSENSE_ENABLE_TRANSLATION_MODELS", "0") == "1"

    def translate(self, text: str, source_language: str, target_language: str, confidence: float, is_unknown: bool) -> Dict:
        if not text.strip():
            return self._blocked(text, source_language, target_language, "No OCR text to translate.")
        if is_unknown or source_language in {"unknown", "brahmi"}:
            return self._blocked(text, source_language, target_language, "Unknown or unseen script; preserving OCR text.")
        if confidence < 0.50:
            return self._blocked(text, source_language, target_language, self.SAFE_FAILURE_TEXT)
        if source_language == target_language:
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_language,
                "target_lang": target_language,
                "translated": False,
                "confidence": 1.0,
                "note": "No translation needed.",
            }
        if source_language == "la" and target_language == "en":
            if not self.enable_translation_models:
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_lang": source_language,
                    "target_lang": target_language,
                    "translated": False,
                    "confidence": 0.0,
                    "note": "Translation models are disabled by default to avoid runtime downloads. Set SCRIPTSENSE_ENABLE_TRANSLATION_MODELS=1 to enable MarianMT/mBART/ByT5.",
                }
            return self._latin_translator.translate_text(text, source_lang="la", target_lang="en")
        return self._transformer_translate(text, source_language, target_language)

    def _blocked(self, text: str, source_language: str, target_language: str, note: str) -> Dict:
        return {
            "original_text": text,
            "translated_text": self.SAFE_FAILURE_TEXT if "degraded" in note else text,
            "source_lang": source_language,
            "target_lang": target_language,
            "translated": False,
            "confidence": 0.0,
            "note": note,
        }

    def _transformer_translate(self, text: str, source_language: str, target_language: str) -> Dict:
        if not self.enable_translation_models:
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_language,
                "target_lang": target_language,
                "translated": False,
                "confidence": 0.0,
                "note": "Translation models are disabled by default to avoid runtime downloads. Set SCRIPTSENSE_ENABLE_TRANSLATION_MODELS=1 to enable MarianMT/mBART/ByT5.",
            }
        for model_name in self._candidate_models(source_language, target_language):
            try:
                if model_name not in self._hf_pipelines:
                    from transformers import pipeline

                    self._hf_pipelines[model_name] = pipeline("translation", model=model_name)
                output = self._hf_pipelines[model_name](text, max_length=512, num_beams=4)
                return {
                    "original_text": text,
                    "translated_text": output[0]["translation_text"],
                    "source_lang": source_language,
                    "target_lang": target_language,
                    "translated": True,
                    "confidence": 0.62,
                    "note": f"Translated with {model_name}.",
                }
            except Exception as exc:
                last_error = str(exc)
        return {
            "original_text": text,
            "translated_text": text,
            "source_lang": source_language,
            "target_lang": target_language,
            "translated": False,
            "confidence": 0.0,
            "note": f"Translation model unavailable; preserving OCR text. {last_error if 'last_error' in locals() else ''}",
        }

    def _candidate_models(self, source_language: str, target_language: str) -> list[str]:
        if target_language != "en":
            return ["facebook/mbart-large-50-many-to-many-mmt", "google/byt5-small"]
        mapping = {
            "gr": ["Helsinki-NLP/opus-mt-el-en", "facebook/mbart-large-50-many-to-many-mmt"],
            "el": ["Helsinki-NLP/opus-mt-el-en", "facebook/mbart-large-50-many-to-many-mmt"],
            "ar": ["Helsinki-NLP/opus-mt-ar-en", "facebook/mbart-large-50-many-to-many-mmt"],
            "he": ["Helsinki-NLP/opus-mt-he-en", "facebook/mbart-large-50-many-to-many-mmt"],
            "sa": ["Helsinki-NLP/opus-mt-mul-en", "facebook/mbart-large-50-many-to-many-mmt", "google/byt5-small"],
        }
        return mapping.get(source_language, ["facebook/mbart-large-50-many-to-many-mmt", "google/byt5-small"])

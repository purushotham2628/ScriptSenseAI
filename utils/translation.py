"""
Translation Module
Provides real Latin-to-English translation with HuggingFace OPUS-MT.
"""

from typing import Dict, List

from transformers import pipeline


class TextTranslator:
    """Translates Latin OCR text to English."""

    LATIN_TO_ENGLISH_MODEL = "Helsinki-NLP/opus-mt-la-en"
    FALLBACK_LATIN_MODELS = [
        "Helsinki-NLP/opus-mt-ine-en",
        "Helsinki-NLP/opus-mt-mul-en",
    ]

    def __init__(self, source_lang: str = "la", target_lang: str = "en"):
        self.source_lang = self.normalize_language(source_lang)
        self.target_lang = self.normalize_language(target_lang)
        self.translator = None
        self.active_model = None
        self.model_load_error = None

    def normalize_language(self, language: str) -> str:
        language = (language or "la").lower()
        if language in {"latin", "lat"}:
            return "la"
        if language in {"english", "eng"}:
            return "en"
        if language in {"greek", "ell", "el"}:
            return "gr"
        if language in {"arabic", "ara"}:
            return "ar"
        if language in {"hebrew", "heb"}:
            return "he"
        return language if language in {"la", "en", "ar", "he", "gr"} else "la"

    def _load_latin_translator(self) -> bool:
        """Lazy-load the Latin-English model only when needed."""
        if self.translator is not None:
            return True

        errors = []
        for model_name in [self.LATIN_TO_ENGLISH_MODEL, *self.FALLBACK_LATIN_MODELS]:
            try:
                print(f"Loading Latin translation model: {model_name}")
                self.translator = pipeline("translation", model=model_name)
                self.active_model = model_name
                self.model_load_error = None
                print(f"Latin translation model loaded: {model_name}")
                return True
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                print(f"Latin translation model unavailable ({model_name}): {exc}")

        self.translator = None
        self.active_model = None
        self.model_load_error = " | ".join(errors)
        return False

    def translate_text(self, text: str, source_lang: str = None,
                       target_lang: str = None, max_length: int = 512) -> Dict:
        """
        Translate Latin text to English.
        Falls back to original text if translation fails or language is unsupported.
        """
        text = text or ""
        source_lang = self.normalize_language(source_lang or self.source_lang)
        target_lang = self.normalize_language(target_lang or self.target_lang)

        if not text.strip():
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "note": "No text to translate"
            }

        if source_lang == target_lang:
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "note": "No translation needed"
            }

        if source_lang != "la" or target_lang != "en":
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "note": "Only Latin to English translation is currently enabled"
            }

        if not self._load_latin_translator():
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "note": "Latin translation failed; returning original text",
                "error": self.model_load_error
            }

        try:
            translated = self.translator(text)[0]["translation_text"]
            return {
                "original_text": text,
                "translated_text": translated,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": True,
                "note": f"Translated with {self.active_model}"
            }
        except Exception as exc:
            print(f"Latin translation error: {exc}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "note": "Latin translation failed; returning original text",
                "error": str(exc)
            }

    def translate_batch(self, texts: List[str]) -> List[Dict]:
        return [self.translate_text(text) for text in texts]

    def batch_translate_with_batching(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        return self.translate_batch(texts)

    def set_language_pair(self, source_lang: str, target_lang: str) -> None:
        self.source_lang = self.normalize_language(source_lang)
        self.target_lang = self.normalize_language(target_lang)

    def get_supported_languages(self) -> List[str]:
        return ["la", "en", "ar", "he", "gr"]

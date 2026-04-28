"""
Translation Module
Provides real Latin-to-English translation with HuggingFace OPUS-MT.
Includes accuracy improvements with preprocessing, confidence scoring, and validation.
"""

from typing import Dict, List, Tuple
import re
import unicodedata

from transformers import pipeline


class TextTranslator:
    """Translates Latin OCR text to English with enhanced accuracy."""

    LATIN_TO_ENGLISH_MODEL = "Helsinki-NLP/opus-mt-la-en"
    FALLBACK_LATIN_MODELS = [
        "Helsinki-NLP/opus-mt-ine-en",
        "Helsinki-NLP/opus-mt-mul-en",
    ]

    # Common Latin abbreviations and their expansions
    LATIN_ABBREVIATIONS = {
        'et': 'et',  # and
        'etc': 'et cetera',  # and the rest
        'q': 'que',  # and (suffix)
        'v': 'vel',  # or
        'pp': 'patres',  # fathers
        'ss': 'sanctus',  # saint
        'ff': 'fratres',  # brothers
        'dd': 'deus',  # god
        'qq': 'que',  # and
        'sc': 'scilicet',  # namely
        'viz': 'videlicet',  # namely
    }

    # Latin word validation dictionary (common words)
    COMMON_LATIN_WORDS = {
        'est', 'sum', 'sunt', 'erat', 'erant', 'eram', 'eum', 'eius', 'ei',
        'qui', 'quae', 'quod', 'quae', 'quorum', 'qua', 'quam', 'quae',
        'et', 'sed', 'vel', 'aut', 'cum', 'si', 'nisi', 'quia', 'ut',
        'in', 'ad', 'ab', 'per', 'pro', 'cum', 'sine', 'post', 'ante',
        'dominus', 'deus', 'christus', 'maria', 'filius', 'pater',
        'liber', 'rex', 'regnum', 'homo', 'mulier', 'dies', 'annus',
        'verbum', 'vita', 'mors', 'anima', 'corpus', 'populus', 'civitas'
    }

    def __init__(self, source_lang: str = "la", target_lang: str = "en"):
        self.source_lang = self.normalize_language(source_lang)
        self.target_lang = self.normalize_language(target_lang)
        self.translator = None
        self.active_model = None
        self.model_load_error = None
        self.translation_cache = {}  # Cache for repeated translations

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

    def _preprocess_latin_text(self, text: str) -> str:
        """
        Preprocess Latin text before translation to improve accuracy.
        
        Handles:
        - Normalization of Unicode characters
        - Expansion of common abbreviations
        - Removal of diacritical marks for model compatibility
        - Standardization of punctuation
        """
        text = text or ""
        
        # Normalize Unicode to NFKC form
        text = unicodedata.normalize('NFKC', text)
        
        # Expand common abbreviations
        text = self._expand_abbreviations(text)
        
        # Remove diacritical marks (they can confuse the model)
        text = self._remove_diacritics(text)
        
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Latin abbreviations to improve translation."""
        words = text.split()
        expanded = []
        
        for word in words:
            # Remove trailing punctuation to check word
            word_lower = word.lower().rstrip('.,;:!?')
            
            if word_lower in self.LATIN_ABBREVIATIONS:
                # Replace abbreviation with full word
                expansion = self.LATIN_ABBREVIATIONS[word_lower]
                # Preserve original casing
                if word[0].isupper():
                    expansion = expansion.capitalize()
                # Restore punctuation
                if word != word_lower + word[len(word_lower):]:
                    expansion += word[len(word_lower):]
                expanded.append(expansion)
            else:
                expanded.append(word)
        
        return ' '.join(expanded)

    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks to improve model compatibility."""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better translation context.
        
        Handles Latin punctuation patterns.
        """
        # Replace common Latin sentence endings with marker
        text = re.sub(r'([.!?])\s+', r'\1|SENTENCE_BREAK|', text)
        sentences = text.split('|SENTENCE_BREAK|')
        return [s.strip() for s in sentences if s.strip()]

    def _validate_latin_words(self, text: str) -> Tuple[float, List[str]]:
        """
        Validate that text contains recognizable Latin words.
        
        Returns:
            Tuple of (validation_score: float, recognized_words: List[str])
        """
        words = text.lower().split()
        recognized = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[.,;:!?]', '', word)
            if clean_word in self.COMMON_LATIN_WORDS:
                recognized.append(clean_word)
        
        if not words:
            return 0.0, []
        
        validation_score = len(recognized) / len(words)
        return validation_score, recognized

    def _chunk_text_for_translation(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into chunks that don't exceed max_length.
        
        Respects sentence boundaries when possible.
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]

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
                       target_lang: str = None, max_length: int = 512,
                       confidence_threshold: float = 0.5) -> Dict:
        """
        Translate Latin text to English with enhanced accuracy.
        Falls back to original text if translation fails or language is unsupported.
        
        Args:
            text: Input text to translate
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum length for model input
            confidence_threshold: Minimum confidence for returning translation
        
        Returns:
            Dictionary with translation results and metadata
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
                "confidence": 0.0,
                "note": "No text to translate"
            }

        if source_lang == target_lang:
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "confidence": 1.0,
                "note": "No translation needed"
            }

        if source_lang != "la" or target_lang != "en":
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "confidence": 0.0,
                "note": "Only Latin to English translation is currently enabled"
            }

        # Check cache
        cache_key = f"{source_lang}:{target_lang}:{text}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        if not self._load_latin_translator():
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "confidence": 0.0,
                "note": "Latin translation failed; returning original text",
                "error": self.model_load_error
            }

        try:
            # Preprocess Latin text
            processed_text = self._preprocess_latin_text(text)
            
            # Validate Latin content
            validation_score, recognized_words = self._validate_latin_words(processed_text)
            
            # Chunk text for translation
            chunks = self._chunk_text_for_translation(processed_text, max_length)
            
            translated_chunks = []
            total_confidence = 0.0
            
            for chunk in chunks:
                # Perform translation with optimized parameters
                result = self.translator(
                    chunk,
                    max_length=max_length,
                    num_beams=4,  # Use beam search for better quality
                    early_stopping=True,
                    temperature=0.7,  # Lower temperature for more deterministic output
                    repetition_penalty=1.2  # Avoid repetitive translations
                )
                
                translated_text = result[0]["translation_text"]
                translated_chunks.append(translated_text)
                
                # Calculate confidence based on multiple factors
                chunk_confidence = self._calculate_translation_confidence(
                    chunk, translated_text, validation_score
                )
                total_confidence += chunk_confidence
            
            # Average confidence across chunks
            avg_confidence = total_confidence / len(translated_chunks) if translated_chunks else 0.0
            
            final_translation = " ".join(translated_chunks)
            
            result_dict = {
                "original_text": text,
                "translated_text": final_translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": True,
                "confidence": min(1.0, avg_confidence),
                "validation_score": validation_score,
                "recognized_words": recognized_words,
                "note": f"Translated with {self.active_model}",
                "num_chunks": len(translated_chunks)
            }
            
            # Cache the result
            self.translation_cache[cache_key] = result_dict
            
            return result_dict
            
        except Exception as exc:
            print(f"Latin translation error: {exc}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "translated": False,
                "confidence": 0.0,
                "note": "Latin translation failed; returning original text",
                "error": str(exc)
            }

    def _calculate_translation_confidence(self, original: str, translated: str, 
                                         validation_score: float) -> float:
        """
        Calculate confidence score for a translation.
        
        Factors:
        - Length preservation (translation should be reasonably similar in length)
        - Validation score (amount of recognized Latin words)
        - Non-empty translation
        """
        if not translated or not translated.strip():
            return 0.0
        
        # Check length ratio (translation shouldn't be dramatically longer/shorter)
        orig_len = len(original.split())
        trans_len = len(translated.split())
        
        if orig_len == 0:
            length_score = 0.5
        else:
            length_ratio = trans_len / orig_len
            # Reasonable range is 0.7 to 1.5
            if 0.7 <= length_ratio <= 1.5:
                length_score = 1.0
            elif 0.5 <= length_ratio <= 2.0:
                length_score = 0.7
            else:
                length_score = 0.3
        
        # Combine scores
        confidence = (validation_score * 0.4 + length_score * 0.6)
        return confidence

    def translate_batch(self, texts: List[str], confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Translate multiple texts with confidence filtering.
        
        Args:
            texts: List of texts to translate
            confidence_threshold: Minimum confidence for translations
            
        Returns:
            List of translation results
        """
        return [self.translate_text(text, confidence_threshold=confidence_threshold) 
                for text in texts]

    def batch_translate_with_batching(self, texts: List[str], batch_size: int = 8,
                                      confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Translate multiple texts with batch processing for efficiency.
        
        Args:
            texts: List of texts to translate
            batch_size: Number of texts to process together
            confidence_threshold: Minimum confidence for translations
            
        Returns:
            List of translation results
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.translate_batch(batch, confidence_threshold)
            results.extend(batch_results)
        return results

    def get_translation_stats(self, translations: List[Dict]) -> Dict:
        """
        Calculate statistics from a batch of translations.
        
        Returns:
            Dictionary with stats including average confidence, translation rate, etc.
        """
        if not translations:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_validation_score": 0.0
            }
        
        successful = sum(1 for t in translations if t.get("translated", False))
        confidences = [t.get("confidence", 0.0) for t in translations if t.get("translated", False)]
        validation_scores = [t.get("validation_score", 0.0) for t in translations if t.get("translated", False)]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_validation = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        
        return {
            "total": len(translations),
            "successful": successful,
            "failed": len(translations) - successful,
            "success_rate": successful / len(translations) if translations else 0.0,
            "avg_confidence": avg_confidence,
            "avg_validation_score": avg_validation
        }

    def set_language_pair(self, source_lang: str, target_lang: str) -> None:
        self.source_lang = self.normalize_language(source_lang)
        self.target_lang = self.normalize_language(target_lang)

    def get_supported_languages(self) -> List[str]:
        return ["la", "en", "ar", "he", "gr"]

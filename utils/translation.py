"""
Translation Module
Translates extracted text to English using HuggingFace transformers
"""

from typing import Dict, List, Optional, Tuple
from transformers import MarianMTModel, MarianTokenizer
import torch


class TextTranslator:
    """Translates text using pre-trained transformer models"""
    
    def __init__(self, source_lang: str = 'en', target_lang: str = 'en'):
        """
        Initialize translator
        
        Args:
            source_lang: Source language code (e.g., 'en', 'la', 'el')
            target_lang: Target language code (e.g., 'en')
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if translation is needed
        if source_lang != target_lang:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load translation model from HuggingFace"""
        try:
            # Model name format: Helsinki-NLP/Tatoeba-MT-models
            model_name = f"Helsinki-NLP/Tatoeba-MT-{self.source_lang}-{self.target_lang}"
            
            # Try common language codes
            if self.source_lang == 'la':  # Latin
                model_name = "Helsinki-NLP/Tatoeba-MT-lat-eng"
            elif self.source_lang == 'el':  # Greek
                model_name = "Helsinki-NLP/Tatoeba-MT-ell-eng"
            elif self.source_lang == 'he':  # Hebrew
                model_name = "Helsinki-NLP/Tatoeba-MT-heb-eng"
            elif self.source_lang == 'ar':  # Arabic
                model_name = "Helsinki-NLP/Tatoeba-MT-ara-eng"
            
            print(f"Loading translation model: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            print(f"Model loaded successfully on {self.device}")
        
        except Exception as e:
            print(f"Could not load specific language model: {e}")
            print(f"Will use English model or fallback approach")
            self.model = None
    
    def translate_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            max_length: Maximum length of input text
            
        Returns:
            Dictionary with translation and confidence
        """
        # If no translation needed or model not loaded
        if self.source_lang == self.target_lang or self.model is None:
            return {
                'original_text': text,
                'translated_text': text,
                'source_lang': self.source_lang,
                'target_lang': self.target_lang,
                'confidence': 1.0,
                'translated': False
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   max_length=max_length, truncation=True).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                translated_ids = self.model.generate(**inputs)
            
            # Decode translation
            translated_text = self.tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            
            return {
                'original_text': text,
                'translated_text': translated_text,
                'source_lang': self.source_lang,
                'target_lang': self.target_lang,
                'confidence': 0.85,  # Approximate confidence
                'translated': True
            }
        
        except Exception as e:
            print(f"Translation error: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'source_lang': self.source_lang,
                'target_lang': self.target_lang,
                'confidence': 0.0,
                'error': str(e),
                'translated': False
            }
    
    def translate_batch(self, texts: List[str]) -> List[Dict]:
        """
        Translate multiple texts
        
        Args:
            texts: List of texts to translate
            
        Returns:
            List of translation results
        """
        results = []
        for text in texts:
            results.append(self.translate_text(text))
        
        return results
    
    def batch_translate_with_batching(self, texts: List[str], 
                                     batch_size: int = 8) -> List[Dict]:
        """
        Translate multiple texts with batching for efficiency
        
        Args:
            texts: List of texts to translate
            batch_size: Batch size for processing
            
        Returns:
            List of translation results
        """
        results = []
        
        if self.model is None or self.source_lang == self.target_lang:
            return [self.translate_text(text) for text in texts]
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(batch, return_tensors="pt", 
                                       padding=True, truncation=True).to(self.device)
                
                # Generate translations
                with torch.no_grad():
                    translated_ids = self.model.generate(**inputs)
                
                # Decode batch
                translated_batch = self.tokenizer.batch_decode(
                    translated_ids, skip_special_tokens=True)
                
                # Add to results
                for original, translated in zip(batch, translated_batch):
                    results.append({
                        'original_text': original,
                        'translated_text': translated,
                        'source_lang': self.source_lang,
                        'target_lang': self.target_lang,
                        'confidence': 0.85,
                        'translated': True
                    })
        
        except Exception as e:
            print(f"Batch translation error: {e}")
            results = [self.translate_text(text) for text in texts]
        
        return results
    
    def set_language_pair(self, source_lang: str, target_lang: str) -> None:
        """
        Change language pair for translation
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        if source_lang != self.source_lang or target_lang != self.target_lang:
            self.source_lang = source_lang
            self.target_lang = target_lang
            
            if source_lang != target_lang:
                self._load_model()
            else:
                self.model = None
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of supported language codes
        """
        # Common ancient and modern languages
        supported = [
            'en',  # English
            'la',  # Latin
            'el',  # Greek (Ancient and Modern)
            'ar',  # Arabic
            'he',  # Hebrew
            'de',  # German
            'fr',  # French
            'es',  # Spanish
            'it',  # Italian
        ]
        return supported

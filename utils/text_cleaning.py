"""
Text Cleaning Module
Cleans and corrects OCR noise using NLP techniques
"""

import re
from typing import List, Tuple, Dict
import unicodedata


class TextCleaner:
    """Cleans and corrects OCR text"""
    
    def __init__(self):
        """Initialize text cleaner"""
        self.common_errors = {
            '0': 'O',  # Zero to O
            '1': 'I',  # One to I
            '5': 'S',  # Five to S
            '8': 'B',  # Eight to B
        }
        
        # Common OCR mistakes
        self.ocr_mistakes = {
            'rn': 'm',  # rn often recognized as m
            'ii': 'u',  # ii often seen as u
            'vv': 'w',  # vv to w
            'cl': 'd',  # cl to d
            '|': 'I',   # pipe to I
        }
    
    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning
        
        Args:
            text: Input text from OCR
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove non-ASCII characters that are likely OCR noise
        text = self._remove_ocr_artifacts(text)
        
        # Fix common OCR mistakes
        text = self._fix_common_mistakes(text)
        
        # Remove special characters that shouldn't be there
        text = self._remove_noise_characters(text)
        
        return text
    
    def _remove_ocr_artifacts(self, text: str) -> str:
        """
        Remove common OCR artifacts
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Remove multiple consecutive special characters
        text = re.sub(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~-]{3,}', '...', text)
        
        return text
    
    def _fix_common_mistakes(self, text: str) -> str:
        """
        Fix common OCR mistakes
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        for mistake, correction in self.ocr_mistakes.items():
            text = text.replace(mistake, correction)
        
        return text
    
    def _remove_noise_characters(self, text: str) -> str:
        """
        Remove likely noise characters
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Keep only letters, numbers, common punctuation, and spaces
        pattern = r'[a-zA-Z0-9\s\.\,\!\?\'\"-]'
        cleaned = ''.join(char if re.match(pattern, char) else ' ' for char in text)
        
        # Clean up resulting spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def segment_text_into_words(self, text: str) -> List[str]:
        """
        Segment text into words
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        words = text.split()
        return [word for word in words if len(word) > 0]
    
    def remove_duplicate_lines(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate lines from list
        
        Args:
            texts: List of text lines
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique_texts = []
        
        for text in texts:
            if text.lower() not in seen:
                seen.add(text.lower())
                unique_texts.append(text)
        
        return unique_texts
    
    def fix_broken_words(self, words: List[str]) -> List[str]:
        """
        Attempt to fix broken words separated by noise
        
        Args:
            words: List of words
            
        Returns:
            Fixed word list
        """
        fixed = []
        
        for word in words:
            # Remove trailing/leading numbers that are likely OCR noise
            cleaned_word = re.sub(r'^[0-9]+', '', word)
            cleaned_word = re.sub(r'[0-9]+$', '', cleaned_word)
            
            # Remove excessive repeated characters
            cleaned_word = re.sub(r'(.)\1{3,}', r'\1\1', cleaned_word)
            
            if cleaned_word:  # Only add non-empty words
                fixed.append(cleaned_word)
        
        return fixed
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep common punctuation
            
        Returns:
            Cleaned text
        """
        if keep_punctuation:
            pattern = r'[^a-zA-Z0-9\s\.\,\!\?\'\"-]'
        else:
            pattern = r'[^a-zA-Z0-9\s]'
        
        cleaned = re.sub(pattern, '', text)
        return re.sub(r'\s+', ' ', cleaned).strip()
    
    def auto_correct_similar_words(self, text: str) -> str:
        """
        Auto-correct similar looking words
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        # Common substitutions
        corrections = {
            r'\ba\b': 'a',  # Single letters
            r'\bi\b': 'I',  # i to I
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean multiple text samples
        
        Args:
            texts: List of text strings
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
    
    def get_cleaning_stats(self, original: str, cleaned: str) -> Dict:
        """
        Get statistics about cleaning
        
        Args:
            original: Original text
            cleaned: Cleaned text
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'characters_removed': len(original) - len(cleaned),
            'original_words': len(original.split()),
            'cleaned_words': len(cleaned.split())
        }
        return stats

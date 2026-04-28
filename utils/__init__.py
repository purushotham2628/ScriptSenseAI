"""
Utils package initialization
"""

from .preprocessing import ImagePreprocessor
from .text_detection import TextDetector
from .character_recognition import CharacterRecognizer
from .text_cleaning import TextCleaner
from .translation import TextTranslator

__version__ = "1.0.0"
__author__ = "Ancient Script Decoder Team"

__all__ = [
    'ImagePreprocessor',
    'TextDetector',
    'CharacterRecognizer',
    'TextCleaner',
    'TextTranslator',
]

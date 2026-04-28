"""
Character Recognition Module
Recognizes characters from detected regions using EasyOCR
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import easyocr


class CharacterRecognizer:
    """Recognizes characters from image regions"""
    
    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize character recognizer
        
        Args:
            languages: List of language codes
        """
        self.reader = easyocr.Reader(languages, gpu=False)
        self.recognized_characters = []
    
    def recognize_character(self, image_region: np.ndarray) -> Dict:
        """
        Recognize a single character from image region
        
        Args:
            image_region: Cropped image containing character/word
            
        Returns:
            Dictionary with recognized text and confidence
        """
        try:
            # Convert grayscale to BGR if needed
            if len(image_region.shape) == 2:
                image_rgb = cv2.cvtColor(image_region, cv2.COLOR_GRAY2BGR)
            else:
                image_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
            
            # Recognize text
            results = self.reader.readtext(image_rgb, detail=1)
            
            if results:
                # Get highest confidence result
                text = results[0][1]
                confidence = results[0][2]
                
                recognition = {
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'all_results': [(r[1].strip(), float(r[2])) for r in results]
                }
            else:
                recognition = {
                    'text': '',
                    'confidence': 0.0,
                    'all_results': []
                }
            
            self.recognized_characters.append(recognition)
            return recognition
        
        except Exception as e:
            print(f"Error recognizing character: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'all_results': [],
                'error': str(e)
            }
    
    def recognize_batch(self, image_regions: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """
        Recognize characters from multiple image regions
        
        Args:
            image_regions: Dictionary of index to image region
            
        Returns:
            Dictionary of index to recognition results
        """
        results = {}
        
        for idx, region in image_regions.items():
            results[idx] = self.recognize_character(region)
        
        return results
    
    def batch_recognize_from_detections(self, image: np.ndarray, 
                                       detections: List[Dict]) -> List[Dict]:
        """
        Recognize text from detection bounding boxes
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            
        Returns:
            Updated detections with recognition results
        """
        # Extract regions
        from .text_detection import TextDetector
        detector = TextDetector()
        regions = detector.extract_text_regions(image, detections)
        
        # Recognize each region
        updated_detections = []
        for idx, detection in enumerate(detections):
            if idx in regions:
                recognition = self.recognize_character(regions[idx])
                updated_detection = detection.copy()
                updated_detection['recognized_text'] = recognition['text']
                updated_detection['recognition_confidence'] = recognition['confidence']
                updated_detections.append(updated_detection)
            else:
                updated_detections.append(detection)
        
        return updated_detections
    
    def enhance_recognition(self, image_region: np.ndarray, 
                           scale_factor: float = 2.0) -> Dict:
        """
        Enhance image before recognition
        
        Args:
            image_region: Image region
            scale_factor: Upscaling factor for better recognition
            
        Returns:
            Recognition result
        """
        # Upscale image
        height, width = image_region.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        upscaled = cv2.resize(image_region, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Apply contrast enhancement
        if len(upscaled.shape) == 2:
            lab = cv2.cvtColor(cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        else:
            lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Recognize
        return self.recognize_character(enhanced)
    
    def get_confidence_weighted_text(self, threshold: float = 0.5) -> str:
        """
        Get recognized text above confidence threshold
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            Concatenated text of high-confidence recognitions
        """
        texts = [char['text'] for char in self.recognized_characters 
                if char['confidence'] >= threshold]
        return " ".join(texts)
    
    def reset(self) -> None:
        """Reset recognizer state"""
        self.recognized_characters = []

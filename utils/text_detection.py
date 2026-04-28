"""
Text Detection Module
Detects text regions and character bounding boxes in images
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import easyocr


class TextDetector:
    """Detects text regions in images"""
    
    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize text detector
        
        Args:
            languages: List of language codes (e.g., ['en', 'la', 'el'])
        """
        # Initialize EasyOCR reader for multiple languages
        self.reader = easyocr.Reader(languages, gpu=False)
        self.detected_texts = []
        self.bounding_boxes = []
    
    def detect_text_regions(self, image: np.ndarray, 
                           confidence_threshold: float = 0.3) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect text regions using EasyOCR
        
        Args:
            image: Input image (BGR or grayscale)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Tuple of (detections list, image with bounding boxes)
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect text using EasyOCR
        results = self.reader.readtext(image_rgb, detail=1)
        
        # Filter by confidence threshold
        detections = []
        for detection in results:
            bbox, text, confidence = detection[0], detection[1], detection[2]
            
            if confidence >= confidence_threshold:
                # Convert bbox to integer coordinates
                bbox_int = np.int32(bbox)
                
                detection_info = {
                    'bbox': bbox_int.tolist(),
                    'text': text,
                    'confidence': float(confidence),
                    'center': self._calculate_center(bbox_int)
                }
                detections.append(detection_info)
                self.bounding_boxes.append(bbox_int)
                self.detected_texts.append(text)
        
        # Draw bounding boxes on image
        image_with_boxes = self._draw_bounding_boxes(image, detections)
        
        return detections, image_with_boxes
    
    def _calculate_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        Calculate center point of bounding box
        
        Args:
            bbox: Bounding box coordinates
            
        Returns:
            Tuple of (center_x, center_y)
        """
        center_x = np.mean(bbox[:, 0])
        center_y = np.mean(bbox[:, 1])
        return (float(center_x), float(center_y))
    
    def _draw_bounding_boxes(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn bounding boxes
        """
        image_copy = image.copy()
        
        # Convert to BGR if grayscale for better visualization
        if len(image_copy.shape) == 2:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        
        for detection in detections:
            bbox = np.array(detection['bbox'], dtype=np.int32)
            
            # Draw polygon around text
            cv2.polylines(image_copy, [bbox], True, (0, 255, 0), 2)
            
            # Put text label with confidence
            label = f"{detection['text'][:20]} ({detection['confidence']:.2f})"
            top_left = tuple(bbox[0])
            cv2.putText(image_copy, label, top_left, cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
        
        return image_copy
    
    def extract_text_regions(self, image: np.ndarray, detections: List[Dict], 
                            padding: int = 5) -> Dict[int, np.ndarray]:
        """
        Extract individual text regions from image
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            padding: Padding around detected region
            
        Returns:
            Dictionary mapping detection index to cropped image
        """
        regions = {}
        
        for idx, detection in enumerate(detections):
            bbox = np.array(detection['bbox'], dtype=np.int32)
            
            # Get bounding rectangle
            x_min = max(0, np.min(bbox[:, 0]) - padding)
            y_min = max(0, np.min(bbox[:, 1]) - padding)
            x_max = min(image.shape[1], np.max(bbox[:, 0]) + padding)
            y_max = min(image.shape[0], np.max(bbox[:, 1]) + padding)
            
            # Crop region
            region = image[y_min:y_max, x_min:x_max]
            regions[idx] = region
        
        return regions
    
    def merge_nearby_detections(self, detections: List[Dict], 
                               distance_threshold: float = 50) -> List[Dict]:
        """
        Merge nearby text detections (handles word segmentation)
        
        Args:
            detections: List of detection dictionaries
            distance_threshold: Maximum distance between centers to merge
            
        Returns:
            Merged detections list
        """
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group_text = det1['text']
            group_boxes = [np.array(det1['bbox'])]
            group_confidence = [det1['confidence']]
            
            # Find nearby detections
            for j, det2 in enumerate(detections):
                if i == j or j in used:
                    continue
                
                # Calculate distance between centers
                center1 = np.array(det1['center'])
                center2 = np.array(det2['center'])
                distance = np.linalg.norm(center1 - center2)
                
                if distance < distance_threshold:
                    group_text += " " + det2['text']
                    group_boxes.append(np.array(det2['bbox']))
                    group_confidence.append(det2['confidence'])
                    used.add(j)
            
            # Create merged detection
            merged_bbox = np.vstack(group_boxes)
            merged_det = {
                'bbox': merged_bbox.tolist(),
                'text': group_text,
                'confidence': float(np.mean(group_confidence)),
                'center': self._calculate_center(merged_bbox)
            }
            merged.append(merged_det)
            used.add(i)
        
        return merged
    
    def get_detected_text(self) -> str:
        """
        Get all detected text concatenated
        
        Returns:
            Full detected text
        """
        return " ".join(self.detected_texts)
    
    def reset(self) -> None:
        """Reset detector state"""
        self.detected_texts = []
        self.bounding_boxes = []

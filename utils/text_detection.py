"""
Text Detection Module
Detects text regions and character bounding boxes in images
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

from .ocr_languages import create_easyocr_reader


class TextDetector:
    """Detects text regions in images"""
    OCR_LANGUAGE_MAP = {
        "la": ["en"],
        "en": ["en"],
        "ar": ["ar"],
        "he": ["en"],
        "gr": ["en"],
        "el": ["en"],
    }
    
    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize text detector
        
        Args:
            languages: List of language codes (e.g., ['en', 'la', 'el'])
        """
        # Initialize EasyOCR reader for compatible installed languages.
        self.reader, self.languages = create_easyocr_reader(languages)
        self.reader_cache = {tuple(self.languages): self.reader}
        self.language_reader_cache = {}
        self.detected_texts = []
        self.bounding_boxes = []
        self.ocr_options = {
            'detail': 1,
            'decoder': 'beamsearch',
            'beamWidth': 10,
            'contrast_ths': 0.1,
            'adjust_contrast': 0.5,
            'mag_ratio': 1.5
        }
    
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
        image_rgb = self._prepare_ocr_image(image)
        
        # Detect text using EasyOCR
        results = self._read_words(image_rgb)
        
        # Filter by confidence threshold
        detections = []
        for detection in results:
            bbox, text, confidence = detection[0], detection[1], detection[2]
            print(f"OCR detected text: '{text}' | confidence: {confidence:.4f}")
            
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

    def extract_text(self, image: np.ndarray, confidence_threshold: float = 0.2,
                     source_lang: str = "en") -> Dict:
        """
        Extract OCR text from an image using confidence-filtered EasyOCR output.

        EasyOCR is still called with detail=1 so confidence scores are available,
        but this method returns text-centric output for the API.
        """
        image_rgb = self._prepare_ocr_image(image)
        requested_lang = self.normalize_source_language(source_lang)
        reader, ocr_languages, note = self._get_reader_for_language(requested_lang)
        paragraph_results = self._read_paragraphs(image_rgb, reader)
        word_results = self._read_words(image_rgb, reader)
        raw_text, confidence, accepted, detections = self._parse_ocr_results(
            paragraph_results,
            word_results,
            confidence_threshold
        )

        detected_lang = self.detect_script_language(raw_text)
        print(f"OCR requested language: {requested_lang}")
        print(f"OCR detected language: {detected_lang}")

        should_auto_switch = detected_lang != "unknown" and detected_lang != requested_lang
        if requested_lang == "la" and detected_lang == "en":
            should_auto_switch = False

        if should_auto_switch:
            fallback_reader, fallback_languages, fallback_note = self._get_reader_for_language(detected_lang)
            if tuple(fallback_languages) != tuple(ocr_languages):
                print(f"OCR auto fallback: rerunning with {fallback_languages}")
                paragraph_results = self._read_paragraphs(image_rgb, fallback_reader)
                word_results = self._read_words(image_rgb, fallback_reader)
                raw_text, confidence, accepted, detections = self._parse_ocr_results(
                    paragraph_results,
                    word_results,
                    confidence_threshold
                )
                requested_lang = detected_lang
                ocr_languages = fallback_languages
                note = self._combine_notes(note, fallback_note, "Auto fallback used after script detection")

        if confidence < 0.3:
            note = self._combine_notes(note, "Low confidence due to unsupported script")

        print(f"OCR raw text: '{raw_text}'")
        print(f"OCR average confidence: {confidence:.4f}")

        self.detected_texts.extend(item['text'] for item in accepted)
        self.bounding_boxes.extend(np.array(item['bbox'], dtype=np.int32) for item in detections)

        return {
            'raw_text': raw_text,
            'confidence': confidence,
            'results': accepted,
            'detections': detections,
            'source_language': requested_lang,
            'detected_language': detected_lang,
            'ocr_languages': ocr_languages,
            'note': note
        }

    def normalize_source_language(self, source_lang: str) -> str:
        source_lang = (source_lang or "en").lower()
        if source_lang in {"el", "ell", "greek"}:
            return "gr"
        if source_lang in {"la", "latin", "lat"}:
            return "la"
        if source_lang in {"arabic", "ara"}:
            return "ar"
        if source_lang in {"hebrew", "heb"}:
            return "he"
        return source_lang if source_lang in self.OCR_LANGUAGE_MAP else "la"

    def detect_script_language(self, text: str) -> str:
        """Detect dominant script from Unicode ranges."""
        counts = {"ar": 0, "he": 0, "gr": 0, "en": 0}
        for char in text or "":
            code = ord(char)
            if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
                counts["ar"] += 1
            elif 0x0590 <= code <= 0x05FF:
                counts["he"] += 1
            elif 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
                counts["gr"] += 1
            elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
                counts["en"] += 1

        detected, count = max(counts.items(), key=lambda item: item[1])
        return detected if count > 0 else "unknown"

    def _get_reader_for_language(self, source_lang: str):
        if source_lang in self.language_reader_cache:
            return self.language_reader_cache[source_lang]

        requested_languages = self.OCR_LANGUAGE_MAP.get(source_lang, ["en"])
        reader, resolved_languages = create_easyocr_reader(requested_languages)
        cache_key = tuple(resolved_languages)
        if cache_key not in self.reader_cache:
            self.reader_cache[cache_key] = reader

        note = ""
        if source_lang == "la":
            note = "Latin OCR uses English script recognition"
        elif source_lang in {"he", "gr"}:
            note = "Fallback used for unsupported scripts"
        elif resolved_languages != requested_languages:
            note = "Fallback used for unsupported scripts"

        result = (self.reader_cache[cache_key], resolved_languages, note)
        self.language_reader_cache[source_lang] = result
        return result

    def _parse_ocr_results(self, paragraph_results: List, word_results: List[Tuple],
                           confidence_threshold: float):
        accepted = []
        detections = []

        for bbox, text, confidence in word_results:
            confidence = float(confidence)
            print(f"OCR detected text: '{text}' | confidence: {confidence:.4f}")

            if confidence < confidence_threshold:
                continue

            clean_text = text.strip()
            if not clean_text:
                continue

            bbox_int = np.int32(bbox)
            accepted.append({
                'text': clean_text,
                'confidence': confidence
            })
            detections.append({
                'bbox': bbox_int.tolist(),
                'text': clean_text,
                'confidence': confidence,
                'center': self._calculate_center(bbox_int)
            })

        raw_text = self._merge_paragraph_text(paragraph_results)
        if not raw_text:
            raw_text = self._merge_word_text(detections)

        confidence = float(np.mean([item['confidence'] for item in accepted])) if accepted else 0.0
        return raw_text, confidence, accepted, detections

    def _combine_notes(self, *notes: str) -> str:
        return "; ".join(dict.fromkeys(note for note in notes if note))

    def _read_words(self, image: np.ndarray, reader=None) -> List[Tuple]:
        """
        Run EasyOCR in word mode so each result includes a confidence score.
        """
        reader = reader or self.reader
        return reader.readtext(image, paragraph=False, **self.ocr_options)

    def _read_paragraphs(self, image: np.ndarray, reader=None) -> List:
        """
        Run EasyOCR in paragraph mode for better spacing and sentence grouping.
        """
        reader = reader or self.reader
        return reader.readtext(image, paragraph=True, **self.ocr_options)

    def _merge_paragraph_text(self, paragraph_results: List) -> str:
        """
        Extract readable text from EasyOCR paragraph output.
        """
        parts = []
        for result in paragraph_results:
            if len(result) >= 2:
                text = str(result[1]).strip()
                if text:
                    parts.append(text)

        return " ".join(parts)

    def _merge_word_text(self, detections: List[Dict]) -> str:
        """
        Merge word detections left-to-right, top-to-bottom as a fallback.
        """
        if not detections:
            return ""

        def sort_key(detection: Dict) -> Tuple[float, float]:
            center_x, center_y = detection['center']
            return (round(center_y / 20) * 20, center_x)

        sorted_detections = sorted(detections, key=sort_key)
        return " ".join(detection['text'] for detection in sorted_detections)

    def _prepare_ocr_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an OpenCV image to the RGB/BGR shape EasyOCR expects.
        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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

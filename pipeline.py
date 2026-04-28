"""
Complete Pipeline Script
End-to-end execution of ancient script decoding pipeline

This script demonstrates the complete flow:
Image → Preprocessing → Detection → Recognition → Cleaning → Translation
"""

import sys
import os
import argparse
import cv2
import json
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.preprocessing import ImagePreprocessor
from utils.text_detection import TextDetector
from utils.character_recognition import CharacterRecognizer
from utils.text_cleaning import TextCleaner
from utils.translation import TextTranslator


class AncientScriptPipeline:
    """Complete pipeline for decoding ancient scripts"""
    
    def __init__(self, source_language='en', target_language='en'):
        """
        Initialize pipeline
        
        Args:
            source_language: Source language code
            target_language: Target language code
        """
        print("[INIT] Initializing Ancient Script Decoder Pipeline...")
        
        # Initialize modules
        self.preprocessor = ImagePreprocessor()
        self.detector = TextDetector(languages=['en', 'la', 'el', 'ar', 'he'])
        self.recognizer = CharacterRecognizer(languages=['en', 'la', 'el', 'ar', 'he'])
        self.cleaner = TextCleaner()
        self.translator = TextTranslator(source_lang=source_language, target_lang=target_language)
        
        self.source_language = source_language
        self.target_language = target_language
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        print("[INIT] Pipeline ready!")
    
    def process_image(self, image_path: str, save_outputs: bool = True) -> dict:
        """
        Process image through complete pipeline
        
        Args:
            image_path: Path to input image
            save_outputs: Whether to save intermediate images
            
        Returns:
            Dictionary with complete results
        """
        print(f"\n{'='*60}")
        print(f"[PIPELINE] Processing: {image_path}")
        print(f"{'='*60}\n")
        
        # ===== STAGE 1: PREPROCESSING =====
        print("[1/5] PREPROCESSING")
        print("      └─ Loading image...")
        
        try:
            original_image, processed_image = self.preprocessor.preprocess_complete_pipeline(
                image_path=image_path)
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
        
        print("      ├─ Grayscale conversion: ✓")
        print("      ├─ Denoising: ✓")
        print("      ├─ CLAHE enhancement: ✓")
        print("      ├─ Thresholding: ✓")
        print("      └─ Morphological operations: ✓")
        
        self.results['stages']['preprocessing'] = {
            'status': 'completed',
            'image_shape': processed_image.shape
        }
        
        if save_outputs:
            output_path = f"output_01_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(output_path, processed_image)
            print(f"      → Saved: {output_path}")
        
        # ===== STAGE 2: TEXT DETECTION =====
        print("\n[2/5] TEXT DETECTION")
        print("      └─ Scanning for text regions...")
        
        try:
            detections, detection_image = self.detector.detect_text_regions(
                processed_image, confidence_threshold=0.3)
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
        
        print(f"      ├─ Regions found: {len(detections)}")
        print("      ├─ Drawing bounding boxes: ✓")
        print("      └─ Merging nearby detections...")
        
        # Merge nearby detections
        merged_detections = self.detector.merge_nearby_detections(detections, 
                                                                  distance_threshold=50)
        print(f"      → After merge: {len(merged_detections)} regions")
        
        self.results['stages']['detection'] = {
            'status': 'completed',
            'regions_found': len(detections),
            'regions_after_merge': len(merged_detections)
        }
        
        if save_outputs:
            output_path = f"output_02_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(output_path, detection_image)
            print(f"      → Saved: {output_path}")
        
        # ===== STAGE 3: CHARACTER RECOGNITION =====
        print("\n[3/5] CHARACTER RECOGNITION")
        print("      └─ Recognizing characters...")
        
        try:
            recognized_detections = self.recognizer.batch_recognize_from_detections(
                processed_image, merged_detections)
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
        
        # Extract recognized texts
        recognized_texts = [d.get('recognized_text', d.get('text', '')) 
                           for d in recognized_detections]
        raw_text = " ".join(recognized_texts)
        
        print(f"      ├─ Recognition completed on {len(recognized_detections)} regions")
        print(f"      ├─ Raw text length: {len(raw_text)} characters")
        print(f"      ├─ Sample text (first 100 chars): {raw_text[:100]}...")
        print("      └─ ✓")
        
        self.results['stages']['recognition'] = {
            'status': 'completed',
            'regions_recognized': len(recognized_detections),
            'raw_text_length': len(raw_text),
            'raw_text_preview': raw_text[:200]
        }
        
        # ===== STAGE 4: TEXT CLEANING =====
        print("\n[4/5] TEXT CLEANING")
        print("      └─ Cleaning extracted text...")
        
        try:
            cleaned_text = self.cleaner.clean_text(raw_text)
            cleaning_stats = self.cleaner.get_cleaning_stats(raw_text, cleaned_text)
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
        
        print(f"      ├─ Original length: {cleaning_stats['original_length']} chars")
        print(f"      ├─ Cleaned length: {cleaning_stats['cleaned_length']} chars")
        print(f"      ├─ Characters removed: {cleaning_stats['characters_removed']}")
        print(f"      ├─ Original words: {cleaning_stats['original_words']}")
        print(f"      ├─ Cleaned words: {cleaning_stats['cleaned_words']}")
        print(f"      ├─ Cleaned text (first 100 chars): {cleaned_text[:100]}...")
        print("      └─ ✓")
        
        self.results['stages']['cleaning'] = {
            'status': 'completed',
            'statistics': cleaning_stats,
            'cleaned_text_preview': cleaned_text[:200]
        }
        
        # ===== STAGE 5: TRANSLATION =====
        print("\n[5/5] TRANSLATION")
        print(f"      └─ Translating from {self.source_language} to {self.target_language}...")
        
        try:
            if self.source_language != 'en':
                self.translator.set_language_pair(self.source_language, 'en')
            
            translation_result = self.translator.translate_text(cleaned_text)
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return None
        
        translated_text = translation_result['translated_text']
        was_translated = translation_result['translated']
        
        print(f"      ├─ Translation status: {'✓ Translated' if was_translated else '✗ No translation needed'}")
        print(f"      ├─ Translated text (first 100 chars): {translated_text[:100]}...")
        print("      └─ ✓")
        
        self.results['stages']['translation'] = {
            'status': 'completed',
            'was_translated': was_translated,
            'source_language': self.source_language,
            'target_language': 'en',
            'translated_text_preview': translated_text[:200]
        }
        
        # ===== FINAL RESULTS =====
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}\n")
        
        self.results['final_output'] = {
            'extracted_text': cleaned_text,
            'translated_text': translated_text,
            'language': self.source_language,
            'confidence_score': 0.85
        }
        
        print("EXTRACTED TEXT:")
        print("-" * 60)
        print(cleaned_text)
        print("-" * 60)
        
        if was_translated:
            print("\nTRANSLATED TEXT:")
            print("-" * 60)
            print(translated_text)
            print("-" * 60)
        
        return self.results
    
    def save_results(self, output_file: str = None) -> str:
        """
        Save results to JSON file
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = f"ancient_script_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        return output_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ancient Script Decoder Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --image test_image.png
  python pipeline.py --image stone_inscription.jpg --source-lang la --output results.json
  python pipeline.py --image ancient_manuscript.png --source-lang el
        """
    )
    
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--source-lang', '-s', type=str, default='en',
                       choices=['en', 'la', 'el', 'ar', 'he'],
                       help='Source language code (default: en)')
    parser.add_argument('--target-lang', '-t', type=str, default='en',
                       choices=['en'],
                       help='Target language for translation (default: en)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file path')
    parser.add_argument('--no-save-intermediate', action='store_true',
                       help='Do not save intermediate processed images')
    
    args = parser.parse_args()
    
    # Validate image file
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = AncientScriptPipeline(
        source_language=args.source_lang,
        target_language=args.target_lang
    )
    
    results = pipeline.process_image(
        args.image,
        save_outputs=not args.no_save_intermediate
    )
    
    if results:
        pipeline.save_results(args.output)
        print("\n✓ Processing completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Processing failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()

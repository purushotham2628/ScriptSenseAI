"""
Test Script for Ancient Script Decoder
Demonstrates functionality and basic testing of all modules
"""

import sys
import os
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.preprocessing import ImagePreprocessor
from utils.text_detection import TextDetector
from utils.character_recognition import CharacterRecognizer
from utils.text_cleaning import TextCleaner
from utils.translation import TextTranslator


def create_test_image():
    """Create a test image with text for demonstration"""
    # Create a white image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "Ancient Text Example", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "This is a demo image", (50, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "For testing the decoder", (50, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img


def test_preprocessing():
    """Test image preprocessing module"""
    print("\n" + "="*60)
    print("TEST 1: IMAGE PREPROCESSING")
    print("="*60)
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Save test image
        test_path = "test_input_image.png"
        cv2.imwrite(test_path, test_image)
        print(f"✓ Created test image: {test_path}")
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor()
        
        # Run preprocessing
        original, processed = preprocessor.preprocess_complete_pipeline(image_path=test_path)
        
        print("✓ Preprocessing pipeline completed")
        print(f"  - Original shape: {original.shape}")
        print(f"  - Processed shape: {processed.shape}")
        
        # Get stats
        stats = preprocessor.get_image_stats(processed)
        print(f"  - Processed image stats: {stats}")
        
        # Save processed image
        output_path = "test_output_preprocessed.png"
        preprocessor.save_image(processed, output_path)
        print(f"✓ Saved processed image: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False


def test_text_detection():
    """Test text detection module"""
    print("\n" + "="*60)
    print("TEST 2: TEXT DETECTION")
    print("="*60)
    
    try:
        # Load preprocessed image or create one
        preprocessor = ImagePreprocessor()
        test_image = create_test_image()
        _, processed = preprocessor.preprocess_complete_pipeline(image_bytes=cv2.imencode('.png', test_image)[1].tobytes())
        
        # Initialize detector
        detector = TextDetector(languages=['en'])
        
        # Detect text
        print("✓ Detecting text regions...")
        detections, detection_image = detector.detect_text_regions(processed, 
                                                                   confidence_threshold=0.3)
        
        print(f"✓ Found {len(detections)} text regions")
        
        for i, det in enumerate(detections[:3]):
            print(f"  - Region {i}: '{det['text']}' (confidence: {det['confidence']:.2f})")
        
        # Save detection image
        output_path = "test_output_detection.png"
        cv2.imwrite(output_path, detection_image)
        print(f"✓ Saved detection image: {output_path}")
        
        # Test merging
        merged = detector.merge_nearby_detections(detections, distance_threshold=50)
        print(f"✓ After merging nearby detections: {len(merged)} regions")
        
        return True
    
    except Exception as e:
        print(f"✗ Text detection test failed: {e}")
        return False


def test_character_recognition():
    """Test character recognition module"""
    print("\n" + "="*60)
    print("TEST 3: CHARACTER RECOGNITION")
    print("="*60)
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Initialize recognizer
        recognizer = CharacterRecognizer(languages=['en'])
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Extract a region and recognize
        region = gray[50:150, 50:400]
        
        print("✓ Recognizing text from image region...")
        result = recognizer.recognize_character(region)
        
        print(f"  - Recognized text: '{result['text']}'")
        print(f"  - Confidence: {result['confidence']:.2f}")
        
        if result['all_results']:
            print(f"  - Alternative results: {result['all_results'][:3]}")
        
        return True
    
    except Exception as e:
        print(f"✗ Character recognition test failed: {e}")
        return False


def test_text_cleaning():
    """Test text cleaning module"""
    print("\n" + "="*60)
    print("TEST 4: TEXT CLEANING")
    print("="*60)
    
    try:
        cleaner = TextCleaner()
        
        # Test cases with OCR noise
        test_texts = [
            "Th15 15 n015y t3xt w1th 0CR 3rr0r5",
            "Thls text has rn instead of m errors",
            "   Extra    spaces    everywhere   ",
            "Text!@#$%^&*() with special chars",
        ]
        
        print("✓ Testing text cleaning:\n")
        
        for original in test_texts:
            cleaned = cleaner.clean_text(original)
            print(f"  Original: {original}")
            print(f"  Cleaned:  {cleaned}")
            print()
        
        # Test batch cleaning
        batch_cleaned = cleaner.clean_batch(test_texts)
        print(f"✓ Batch cleaning: {len(batch_cleaned)} texts cleaned")
        
        # Test word fixing
        broken_words = ['w0rd', '3rr0r', 'n015e', 'c0rrupt3d']
        fixed_words = cleaner.fix_broken_words(broken_words)
        print(f"✓ Broken word fixing: {broken_words} → {fixed_words}")
        
        return True
    
    except Exception as e:
        print(f"✗ Text cleaning test failed: {e}")
        return False


def test_translation():
    """Test translation module"""
    print("\n" + "="*60)
    print("TEST 5: TRANSLATION")
    print("="*60)
    
    try:
        translator = TextTranslator(source_lang='en', target_lang='en')
        
        print("✓ Checking supported languages:")
        supported_langs = translator.get_supported_languages()
        print(f"  - Supported: {', '.join(supported_langs)}")
        
        # Test no-translation case (English to English)
        test_text = "This is example text in English"
        
        print(f"\n✓ Testing translation (English to English):")
        print(f"  - Original: {test_text}")
        
        result = translator.translate_text(test_text)
        print(f"  - Translated: {result['translated_text']}")
        print(f"  - Was translated: {result['translated']}")
        
        # Note: Actual Latin/Greek translation requires downloading models
        # which may not be available in test environment
        print("\n✓ Translation module ready (full translation requires internet)")
        
        return True
    
    except Exception as e:
        print(f"✗ Translation test failed: {e}")
        return False


def run_integration_test():
    """Run complete pipeline integration test"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Complete Pipeline")
    print("="*60)
    
    try:
        # Create test image
        test_image = create_test_image()
        test_path = "test_integration_image.png"
        cv2.imwrite(test_path, test_image)
        
        # Initialize all modules
        preprocessor = ImagePreprocessor()
        detector = TextDetector(languages=['en'])
        recognizer = CharacterRecognizer(languages=['en'])
        cleaner = TextCleaner()
        translator = TextTranslator(source_lang='en', target_lang='en')
        
        print("\n✓ Step 1: Preprocessing")
        original, processed = preprocessor.preprocess_complete_pipeline(image_path=test_path)
        
        print("✓ Step 2: Text Detection")
        detections, detection_img = detector.detect_text_regions(processed)
        print(f"  - Found {len(detections)} regions")
        
        print("✓ Step 3: Character Recognition")
        recognized = recognizer.batch_recognize_from_detections(processed, detections)
        raw_text = " ".join([d.get('recognized_text', '') for d in recognized])
        print(f"  - Extracted: '{raw_text[:50]}...'")
        
        print("✓ Step 4: Text Cleaning")
        cleaned = cleaner.clean_text(raw_text)
        print(f"  - Cleaned: '{cleaned[:50]}...'")
        
        print("✓ Step 5: Translation")
        translation = translator.translate_text(cleaned)
        print(f"  - Translated: '{translation['translated_text'][:50]}...'")
        
        print("\n✓ Integration test completed successfully!")
        return True
    
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ANCIENT SCRIPT DECODER - TEST SUITE")
    print("="*60)
    
    results = {
        'Preprocessing': test_preprocessing(),
        'Text Detection': test_text_detection(),
        'Character Recognition': test_character_recognition(),
        'Text Cleaning': test_text_cleaning(),
        'Translation': test_translation(),
        'Integration': run_integration_test(),
    }
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready for deployment.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")


if __name__ == '__main__':
    main()

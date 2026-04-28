"""
Example usage scripts for Ancient Script Decoder
Demonstrates various ways to use the system
"""

# ============================================
# EXAMPLE 1: Simple Pipeline Usage
# ============================================

"""
from pipeline import AncientScriptPipeline

# Create pipeline
pipeline = AncientScriptPipeline(source_language='en', target_language='en')

# Process image
results = pipeline.process_image('sample_image.jpg', save_outputs=True)

# Access results
if results:
    print("Extracted text:", results['final_output']['extracted_text'])
    print("Translation:", results['final_output']['translated_text'])
    
    # Save to file
    pipeline.save_results('output.json')
"""

# ============================================
# EXAMPLE 2: Latin Inscription Decoding
# ============================================

"""
from pipeline import AncientScriptPipeline

# Initialize for Latin
pipeline = AncientScriptPipeline(source_language='la', target_language='en')

# Process Latin inscription
results = pipeline.process_image('latin_inscription.jpg')

# Extract the translated English text
translated = results['final_output']['translated_text']
print(f"English translation:\\n{translated}")
"""

# ============================================
# EXAMPLE 3: Module-by-Module Processing
# ============================================

"""
import cv2
from utils.preprocessing import ImagePreprocessor
from utils.text_detection import TextDetector
from utils.character_recognition import CharacterRecognizer
from utils.text_cleaning import TextCleaner

# Step 1: Preprocess
preprocessor = ImagePreprocessor()
original, processed = preprocessor.preprocess_complete_pipeline('image.jpg')

# Step 2: Detect
detector = TextDetector(languages=['en'])
detections, det_image = detector.detect_text_regions(processed)
print(f"Found {len(detections)} text regions")

# Step 3: Recognize
recognizer = CharacterRecognizer(languages=['en'])
recognized = recognizer.batch_recognize_from_detections(processed, detections)

# Step 4: Clean
cleaner = TextCleaner()
raw_text = " ".join([d.get('recognized_text', '') for d in recognized])
cleaned_text = cleaner.clean_text(raw_text)

print("Extracted:", cleaned_text)
"""

# ============================================
# EXAMPLE 4: Batch Processing Multiple Images
# ============================================

"""
import os
from pipeline import AncientScriptPipeline

image_dir = 'images/'
pipeline = AncientScriptPipeline(source_language='en')

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(image_dir, filename)
        print(f"\\nProcessing {filename}...")
        
        results = pipeline.process_image(image_path)
        
        # Save results
        output_file = f"results/{filename.split('.')[0]}_results.json"
        pipeline.save_results(output_file)
"""

# ============================================
# EXAMPLE 5: API Integration
# ============================================

"""
import requests
import json

API_URL = "http://localhost:8000"

# Upload and process image
with open("inscription.jpg", "rb") as f:
    files = {'file': f}
    data = {
        'source_language': 'la',
        'target_language': 'en'
    }
    
    response = requests.post(f"{API_URL}/process", files=files, data=data)
    results = response.json()

# Extract results
extracted_text = results['final_output']['extracted_text']
translated_text = results['final_output']['translated_text']

print("Extracted:", extracted_text)
print("Translated:", translated_text)

# Save results to file
with open('api_results.json', 'w') as f:
    json.dump(results, f, indent=2)
"""

# ============================================
# EXAMPLE 6: Custom Preprocessing Pipeline
# ============================================

"""
import cv2
from utils.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()

# Load image
image = preprocessor.load_image('image.jpg')

# Apply individual steps
gray = preprocessor.convert_to_grayscale(image)
denoised = preprocessor.denoise_image(gray)
enhanced = preprocessor.apply_clahe(denoised)
binary = preprocessor.apply_thresholding(enhanced, method='adaptive')
processed = preprocessor.apply_morphological_operations(binary)

# Save result
preprocessor.save_image(processed, 'custom_preprocessed.png')
"""

# ============================================
# EXAMPLE 7: Text Detection and Visualization
# ============================================

"""
import cv2
from utils.preprocessing import ImagePreprocessor
from utils.text_detection import TextDetector

# Preprocess
preprocessor = ImagePreprocessor()
original, processed = preprocessor.preprocess_complete_pipeline('image.jpg')

# Detect with visualization
detector = TextDetector(languages=['en', 'la', 'el'])
detections, detection_image = detector.detect_text_regions(processed)

# Extract text regions
regions = detector.extract_text_regions(processed, detections, padding=10)

# Save detection visualization
cv2.imwrite('detection_result.png', detection_image)

# Visualize individual regions
for idx, region in regions.items():
    cv2.imwrite(f'region_{idx}.png', region)

print(f"Detected {len(detections)} regions and saved visualizations")
"""

# ============================================
# EXAMPLE 8: Enhanced Character Recognition
# ============================================

"""
import cv2
from utils.character_recognition import CharacterRecognizer

recognizer = CharacterRecognizer(languages=['en', 'la', 'el'])

# Load image region
region = cv2.imread('character_region.jpg', cv2.IMREAD_GRAYSCALE)

# Basic recognition
result = recognizer.recognize_character(region)
print(f"Recognized: {result['text']} (confidence: {result['confidence']:.2f})")

# Enhanced recognition with upscaling
enhanced_result = recognizer.enhance_recognition(region, scale_factor=3.0)
print(f"Enhanced: {enhanced_result['text']} (confidence: {enhanced_result['confidence']:.2f})")
"""

# ============================================
# EXAMPLE 9: Text Cleaning and Correction
# ============================================

"""
from utils.text_cleaning import TextCleaner

cleaner = TextCleaner()

# Sample OCR outputs with errors
noisy_texts = [
    "Th15 15 t3xt w1th 0CR 3rr0r5",
    "   Multiple     spaces     everywhere   ",
    "M15p3ll3d w0rd5 fr0m 0CR",
]

print("Text Cleaning Examples:\\n")
for original in noisy_texts:
    cleaned = cleaner.clean_text(original)
    stats = cleaner.get_cleaning_stats(original, cleaned)
    print(f"Original: {original}")
    print(f"Cleaned:  {cleaned}")
    print(f"Stats: {stats}\\n")

# Batch cleaning
batch_results = cleaner.clean_batch(noisy_texts)
"""

# ============================================
# EXAMPLE 10: Translation Pipeline
# ============================================

"""
from utils.translation import TextTranslator

# Initialize translator (English to English - no translation needed)
translator = TextTranslator(source_lang='en', target_lang='en')

# List supported languages
supported_langs = translator.get_supported_languages()
print("Supported languages:", supported_langs)

# Try a language pair
sample_text = "This is a sample text."

# English to English (no-op)
result = translator.translate_text(sample_text)
print(f"Original: {result['original_text']}")
print(f"Translated: {result['translated_text']}")

# For actual translation, set language pair:
# translator.set_language_pair('la', 'en')  # Latin to English
# result = translator.translate_text(latin_text)
"""

# ============================================
# EXAMPLE 11: Configuration Usage
# ============================================

"""
from config import get_config, PREPROCESSING, DETECTION

# Access specific config
clahe_limit = get_config('PREPROCESSING', 'clahe_clip_limit')
print(f"CLAHE Clip Limit: {clahe_limit}")

# Access full section
detection_config = get_config('DETECTION')
print(f"Languages: {detection_config['languages']}")

# Modify configuration for processing
from utils.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()
# Now uses config settings
preprocessor.preprocess_complete_pipeline('image.jpg')
"""

# ============================================
# EXAMPLE 12: Error Handling
# ============================================

"""
import os
from pipeline import AncientScriptPipeline

def process_safely(image_path):
    try:
        # Validate file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Not a file: {image_path}")
        
        # Process
        pipeline = AncientScriptPipeline()
        results = pipeline.process_image(image_path)
        
        return results
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage
results = process_safely('my_image.jpg')
if results:
    print("Success!")
else:
    print("Processing failed")
"""


if __name__ == '__main__':
    print("Ancient Script Decoder - Example Usage Scripts")
    print("=" * 60)
    print("\nThis file contains 12 example usage patterns.")
    print("Uncomment and run the examples you want to try.")
    print("\nExamples included:")
    print("1. Simple pipeline usage")
    print("2. Latin inscription decoding")
    print("3. Module-by-module processing")
    print("4. Batch processing multiple images")
    print("5. API integration")
    print("6. Custom preprocessing pipeline")
    print("7. Text detection and visualization")
    print("8. Enhanced character recognition")
    print("9. Text cleaning and correction")
    print("10. Translation pipeline")
    print("11. Configuration usage")
    print("12. Error handling")
    print("\nSee docstrings for details on each example.")

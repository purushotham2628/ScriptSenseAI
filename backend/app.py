"""
Main Backend API for Ancient Script Decoding System
FastAPI application with endpoints for image processing pipeline
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import numpy as np
import cv2
from io import BytesIO
import base64
from datetime import datetime
import json
from typing import Optional

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.preprocessing import ImagePreprocessor
from utils.text_detection import TextDetector
from utils.character_recognition import CharacterRecognizer
from utils.text_cleaning import TextCleaner
from utils.translation import TextTranslator

# Initialize FastAPI app
app = FastAPI(
    title="Ancient Script Decoding System",
    description="AI-powered system for decoding ancient scripts",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing modules
preprocessor = ImagePreprocessor()
detector = TextDetector(languages=['en'])
recognizer = CharacterRecognizer(languages=['en', 'la', 'el'])
cleaner = TextCleaner()
translator = TextTranslator(source_lang='la', target_lang='en')

# Store recent processing results
processing_history = {}


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode image to base64 string
    
    Args:
        image: OpenCV image
        
    Returns:
        Base64 encoded string
    """
    _, encoded = cv2.imencode('.png', image)
    base64_str = base64.b64encode(encoded).decode('utf-8')
    return base64_str


def get_image_dimensions(image: np.ndarray) -> dict:
    """Get image dimensions"""
    if len(image.shape) == 3:
        height, width, channels = image.shape
        return {'width': width, 'height': height, 'channels': channels}
    else:
        height, width = image.shape
        return {'width': width, 'height': height, 'channels': 1}


@app.get("/")
async def root():
    """Serve the web frontend."""
    index_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend template not found")
    return FileResponse(index_path)


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Ancient Script Decoding System",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-image",
            "process": "/process",
            "get_results": "/get-results",
            "history": "/history",
            "status": "/status"
        }
    }


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload image for processing
    
    Returns:
        Image metadata and base64 encoded image
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Load image
        original_image = preprocessor.load_image(image_bytes=contents)
        
        # Get image info
        img_info = get_image_dimensions(original_image)
        
        # Encode to base64 for frontend display
        base64_image = encode_image_to_base64(original_image)
        
        # Generate session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store in history
        processing_history[session_id] = {
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'image_info': img_info,
            'status': 'uploaded'
        }
        
        return {
            'success': True,
            'session_id': session_id,
            'filename': file.filename,
            'image_info': img_info,
            'image_data': base64_image
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading image: {str(e)}")


@app.post("/process")
async def process_image(file: UploadFile = File(...), 
                       source_language: str = Form('la'),
                       target_language: str = Form('en'),
                       source_lang: Optional[str] = Form(None)):
    """
    Main processing pipeline: preprocessing -> detection -> recognition -> cleaning -> translation
    
    Returns:
        Complete processing results with all intermediate outputs
    """
    try:
        # Read image
        contents = await file.read()
        
        requested_source_language = source_lang or source_language or 'la'
        requested_target_language = target_language or 'en'

        # ===== STEP 1: PREPROCESSING =====
        print("[1] Starting image preprocessing...")
        original, processed = preprocessor.preprocess_complete_pipeline(image_bytes=contents)
        ocr_image = preprocessor.ocr_image if preprocessor.ocr_image is not None else processed
        
        preprocessed_b64 = encode_image_to_base64(processed)
        
        # ===== STEP 2: OCR EXTRACTION =====
        print("[2] Extracting text with EasyOCR...")
        ocr_result = detector.extract_text(
            ocr_image,
            confidence_threshold=0.2,
            source_lang=requested_source_language
        )
        detections = ocr_result['detections']
        detection_image = detector._draw_bounding_boxes(ocr_image, detections)
        
        detection_b64 = encode_image_to_base64(detection_image)
        
        print(f"   Found {len(detections)} high-confidence text regions")
        print(f"   Detected language: {ocr_result['source_language']}")
        print(f"   Raw OCR text: {ocr_result['raw_text']}")
        print(f"   Mean OCR confidence: {ocr_result['confidence']:.4f}")
        
        # ===== STEP 3: TEXT AGGREGATION =====
        print("[3] Aggregating OCR text...")
        raw_text = ocr_result['raw_text']
        recognized_texts = [item['text'] for item in ocr_result['results']]
        
        # ===== STEP 4: TEXT CLEANING =====
        print("[4] Cleaning extracted text...")
        cleaned_text = cleaner.clean_alpha_text(raw_text)
        cleaned_stats = cleaner.get_cleaning_stats(raw_text, cleaned_text)
        print("Cleaned:", cleaned_text)
        
        # ===== STEP 5: TRANSLATION =====
        print("[5] Translating text...")
        print("INPUT TO TRANSLATOR:", cleaned_text)
        translation_result = translator.translate_text(
            cleaned_text,
            source_lang=ocr_result['source_language'],
            target_lang=requested_target_language
        )
        print("OUTPUT FROM TRANSLATOR:", translation_result['translated_text'])
        notes = [
            ocr_result.get('note', ''),
            translation_result.get('note', '')
        ]
        note = "; ".join(dict.fromkeys(item for item in notes if item))
        
        # ===== PREPARE RESPONSE =====
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'translated_text': translation_result['translated_text'],
            'source_language': ocr_result['source_language'],
            'target_language': requested_target_language,
            'confidence': ocr_result['confidence'],
            'note': note,
            'pipeline': {
                'preprocessing': {
                    'status': 'completed',
                    'image': preprocessed_b64,
                    'info': 'Grayscale, 2x resize, brightness normalization, CLAHE, Gaussian blur, adaptive threshold'
                },
                'detection': {
                    'status': 'completed',
                    'image': detection_b64,
                    'regions_found': len(detections),
                    'detections': detections[:10]  # Limit to 10 for response size
                },
                'recognition': {
                    'status': 'completed',
                    'recognized_texts': recognized_texts[:10],
                    'raw_text': raw_text,
                    'confidence': ocr_result['confidence']
                },
                'cleaning': {
                    'status': 'completed',
                    'original_length': cleaned_stats['original_length'],
                    'cleaned_length': cleaned_stats['cleaned_length'],
                    'cleaned_text': cleaned_text
                },
                'translation': {
                    'status': 'completed',
                    'source_language': ocr_result['source_language'],
                    'target_language': requested_target_language,
                    'original_text': cleaned_text,
                    'translated_text': translation_result['translated_text'],
                    'translated': translation_result['translated'],
                    'note': translation_result.get('note', '')
                }
            },
            'final_output': {
                'extracted_text': cleaned_text,
                'translated_text': translation_result['translated_text'],
                'confidence_score': ocr_result['confidence']
            }
        }
        
        # Store in history
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}"
        processing_history[session_id] = result
        result['session_id'] = session_id
        
        return result
    
    except Exception as e:
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/get-results")
async def get_results(session_id: str):
    """Get results from a specific session"""
    if session_id not in processing_history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return processing_history[session_id]


@app.get("/history")
async def get_history():
    """Get processing history"""
    history_list = []
    
    for session_id, data in processing_history.items():
        history_item = {
            'session_id': session_id,
            'timestamp': data.get('timestamp', 'unknown'),
            'extracted_text': data.get('final_output', {}).get('extracted_text', '')[:100]
        }
        history_list.append(history_item)
    
    return {'history': history_list}


@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        'status': 'operational',
        'modules': {
            'preprocessing': 'ready',
            'detection': 'ready',
            'recognition': 'ready',
            'cleaning': 'ready',
            'translation': 'ready'
        },
        'device': 'cpu',
        'active_sessions': len(processing_history)
    }


@app.get("/languages")
async def get_supported_languages():
    """Get supported languages for translation"""
    return {
        'source_languages': translator.get_supported_languages(),
        'target_language': 'en'
    }


# Mount frontend static files
static_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')
templates_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    # Run with: uvicorn app:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

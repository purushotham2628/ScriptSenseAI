"""
Configuration file for Ancient Script Decoder
Customize settings here
"""

# ============================================
# IMAGE PROCESSING SETTINGS
# ============================================

PREPROCESSING = {
    # Resize settings
    'target_width': 1280,
    
    # CLAHE settings
    'clahe_clip_limit': 2.0,
    'clahe_grid_size': (8, 8),
    
    # Bilateral filter settings
    'bilateral_diameter': 9,
    'bilateral_sigma_color': 75,
    'bilateral_sigma_space': 75,
    
    # Morphological operations
    'morph_kernel_size': (3, 3),
    'morph_iterations': 1,
    
    # Thresholding method
    'threshold_method': 'adaptive',  # 'otsu', 'adaptive', 'binary'
}

# ============================================
# TEXT DETECTION SETTINGS
# ============================================

DETECTION = {
    # Confidence threshold
    'confidence_threshold': 0.3,
    
    # EasyOCR languages
    'languages': ['en', 'la', 'el', 'ar', 'he'],
    
    # Merging settings
    'merge_distance_threshold': 50,
    
    # GPU usage
    'use_gpu': False,  # Set to True if CUDA available
}

# ============================================
# RECOGNITION SETTINGS
# ============================================

RECOGNITION = {
    # Confidence threshold for recognition
    'confidence_threshold': 0.3,
    
    # Enhancement settings
    'enhance_recognition': True,
    'enhancement_scale': 2.0,
    
    # Region padding
    'region_padding': 5,
}

# ============================================
# TEXT CLEANING SETTINGS
# ============================================

CLEANING = {
    # Remove special characters
    'keep_punctuation': True,
    
    # Common OCR corrections
    'fix_common_errors': True,
    
    # Remove duplicates
    'remove_duplicates': True,
}

# ============================================
# TRANSLATION SETTINGS
# ============================================

TRANSLATION = {
    # Default source and target languages
    'default_source_lang': 'en',
    'default_target_lang': 'en',
    
    # Supported languages
    'supported_languages': {
        'en': 'English',
        'la': 'Latin',
        'el': 'Ancient Greek',
        'ar': 'Arabic',
        'he': 'Hebrew',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
    },
    
    # Batch translation settings
    'batch_size': 8,
    'max_input_length': 512,
    
    # ===== IMPROVED TRANSLATION SETTINGS =====
    # These parameters optimize translation accuracy
    
    # Beam search parameters for better quality
    'num_beams': 4,  # More beams = better but slower
    'early_stopping': True,  # Stop when high-quality translation found
    
    # Generation parameters
    'temperature': 0.7,  # Lower = more deterministic (0.7 recommended)
    'repetition_penalty': 1.2,  # Avoid repetitive output
    
    # Confidence threshold for filtering
    'confidence_threshold': 0.5,  # Min confidence (0.0-1.0)
    
    # Enable translation caching
    'enable_caching': True,
    
    # Preprocessing options
    'normalize_unicode': True,  # Normalize to NFKC form
    'remove_diacritics': True,  # Remove accents for better compatibility
    'expand_abbreviations': True,  # Expand common abbreviations
    'validate_latin': True,  # Check for recognizable Latin words
}

# ============================================
# API SETTINGS
# ============================================

API = {
    # Server settings
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,  # Set to False in production
    
    # Logging
    'log_level': 'info',
    
    # CORS settings
    'cors_origins': ['*'],
    'cors_methods': ['*'],
    'cors_headers': ['*'],
    
    # Session management
    'session_timeout': 3600,  # 1 hour in seconds
    'max_sessions': 100,
}

# ============================================
# FRONTEND SETTINGS
# ============================================

FRONTEND = {
    # Image upload limits
    'max_file_size_mb': 50,
    'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    
    # Display settings
    'image_display_max_height': 400,
}

# ============================================
# OUTPUT SETTINGS
# ============================================

OUTPUT = {
    # Save intermediate images
    'save_preprocessing': True,
    'save_detection': True,
    
    # Output directory
    'output_directory': 'outputs/',
    
    # Naming convention
    'include_timestamp': True,
    'include_language': True,
}

# ============================================
# PERFORMANCE SETTINGS
# ============================================

PERFORMANCE = {
    # Threading
    'num_workers': 4,
    
    # Caching
    'enable_cache': True,
    'cache_size_mb': 500,
    
    # Batch processing
    'enable_batch_processing': True,
}

# ============================================
# LOGGING SETTINGS
# ============================================

LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/app.log',
}

# ============================================
# FUNCTION TO GET CONFIG
# ============================================

def get_config(section: str, key: str = None):
    """
    Get configuration value
    
    Args:
        section: Configuration section (e.g., 'PREPROCESSING')
        key: Specific key within section
        
    Returns:
        Configuration value or dictionary
    """
    config_dict = {
        'PREPROCESSING': PREPROCESSING,
        'DETECTION': DETECTION,
        'RECOGNITION': RECOGNITION,
        'CLEANING': CLEANING,
        'TRANSLATION': TRANSLATION,
        'API': API,
        'FRONTEND': FRONTEND,
        'OUTPUT': OUTPUT,
        'PERFORMANCE': PERFORMANCE,
        'LOGGING': LOGGING,
    }
    
    if section not in config_dict:
        raise ValueError(f"Unknown configuration section: {section}")
    
    section_config = config_dict[section]
    
    if key is not None:
        if key not in section_config:
            raise ValueError(f"Unknown key '{key}' in section '{section}'")
        return section_config[key]
    
    return section_config

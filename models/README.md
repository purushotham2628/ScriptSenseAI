"""
Models Directory

This directory contains or will contain pre-trained models used by the system.

Pre-trained Models Used:
- EasyOCR models (downloaded automatically)
- HuggingFace Transformers for translation
- CLAHE for image enhancement

Models are automatically downloaded on first use.
Subsequent uses will load from cache.
"""

# On first run, the following models will be downloaded:

# 1. EasyOCR Models (per language)
#    Location: ~/.EasyOCR/
#    Size: ~100-200 MB per language
#    
#    Supported languages:
#    - English (en)
#    - Latin (la) - if available
#    - Greek (el)
#    - Arabic (ar)
#    - Hebrew (he)

# 2. HuggingFace Models
#    Location: ~/.cache/huggingface/
#    Models: Helsinki-NLP Tatoeba-MT for translation
#    Size: ~150-300 MB per language pair

# Download Location Management:
#
# To customize download location, set environment variables:
#
# export HF_HOME=/path/to/huggingface  # For HuggingFace
# export EASYOCR_HOME=/path/to/easyocr  # For EasyOCR (if needed)

# Typical disk usage after first run: 2-3 GB

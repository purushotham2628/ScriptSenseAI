# Test Images Directory

This directory stores test images for the Ancient Script Decoder pipeline.

## Usage

Place your sample/test images here:

```bash
cd data/test_images
# Copy your images here
```

## Supported Formats

- `.jpg` / `.jpeg` - JPEG images
- `.png` - PNG images
- `.bmp` - Bitmap images
- `.tiff` - TIFF images

## Example Images

For testing, you can create or download:

### 1. Latin Inscriptions
- Roman stone inscriptions
- Latin manuscripts
- Legal texts

Example: `latin_inscription.jpg`

### 2. Ancient Greek
- Greek manuscripts
- Stone inscriptions with Greek letters
- Classical texts

Example: `greek_manuscript.png`

### 3. Arabic Texts
- Ancient Arabic manuscripts
- Stone carvings with Arabic script

Example: `arabic_text.jpg`

### 4. Hebrew Texts
- Ancient Hebrew inscriptions
- Religious texts

Example: `hebrew_inscription.png`

## Sample Test Images

Generate test images programmatically:

```python
import cv2
import numpy as np

# Create a test image with text
img = np.ones((400, 600, 3), dtype=np.uint8) * 255

# Add text
cv2.putText(img, "Sample Ancient Text", (50, 100),
           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

# Save
cv2.imwrite('test_images/sample.jpg', img)
```

## Directory Structure

```
test_images/
├── sample_1.jpg          # Generic sample
├── latin_samples/        # Latin text samples
├── greek_samples/        # Greek text samples
├── arabic_samples/       # Arabic text samples
└── mixed_scripts/        # Mixed language samples
```

## Notes

- Recommended image size: 1200x800 or larger
- Good lighting and contrast improve accuracy
- Use high-quality scans for best results
- Text should be clear and legible
- Avoid extreme angles or distortions

## Processing Test Images

### Command Line

```bash
# Process a test image
python pipeline.py --image data/test_images/sample.jpg

# With language specification
python pipeline.py --image data/test_images/latin.jpg --source-lang la
```

### Python API

```python
from pipeline import AncientScriptPipeline

pipeline = AncientScriptPipeline()
results = pipeline.process_image('data/test_images/sample.jpg')
```

---

Create a `sample_download.py` script to download public domain ancient texts:

```python
# Download sample images from public sources
# (Add your image sources here)
```

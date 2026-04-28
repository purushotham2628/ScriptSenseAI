# Quick Start Guide - Ancient Script Decoder

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd "Ancient Script"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Run Tests (1 minute)

```bash
python test_pipeline.py
```

Expected output:
```
✓ Preprocessing.............. PASS
✓ Text Detection............. PASS
✓ Character Recognition...... PASS
✓ Text Cleaning.............. PASS
✓ Translation................ PASS
✓ Integration................ PASS
Total: 6/6 tests passed
```

### 3. Process Your First Image (1 minute)

```bash
# Process a single image (Latin example)
python pipeline.py --image your_image.jpg --source-lang la

# Output will show:
# - Extracted text
# - Translated text
# - Processing statistics
```

### 4. Start Web Interface (1 minute)

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Then open in browser:
```
http://localhost:8000
```

## 📖 Command-Line Examples

### Basic Usage
```bash
python pipeline.py --image inscription.jpg
```

### Latin Inscription
```bash
python pipeline.py --image latin_stone.jpg --source-lang la
```

### Ancient Greek
```bash
python pipeline.py --image greek_manuscript.png --source-lang el
```

### With Custom Output
```bash
python pipeline.py --image text.jpg --output my_results.json
```

### Multiple Languages
```bash
python pipeline.py --image text.jpg --source-lang ar  # Arabic
python pipeline.py --image text.jpg --source-lang he  # Hebrew
```

## 🌐 Web Interface Features

1. **Upload Section**
   - Drag and drop images
   - Browse file system
   - Real-time preview

2. **Processing Options**
   - Select source language
   - Choose target language
   - Start processing

3. **Results Display**
   - Original image
   - Preprocessed image
   - Text detection overlay
   - Extracted text
   - Cleaned text
   - Translated text
   - Statistics

4. **Actions**
   - Download results as JSON
   - Process another image

## 🔄 Pipeline Flow

```
Image Upload
    ↓
[1] Preprocessing (Grayscale, Denoise, CLAHE, Threshold)
    ↓
[2] Detection (Find text regions)
    ↓
[3] Recognition (Extract text characters)
    ↓
[4] Cleaning (Remove noise, fix errors)
    ↓
[5] Translation (Translate to English)
    ↓
Results Display
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'easyocr'"
```bash
pip install -U easyocr
```

### "CUDA out of memory"
Models use CPU by default. For GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Connection refused" on API calls
Make sure backend is running:
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### Slow processing
- First run downloads models (~500MB)
- Subsequent runs are faster
- GPU acceleration available with CUDA

## 📁 Project Layout

```
Ancient Script/
├── backend/           ← FastAPI server
├── frontend/          ← Web interface
├── utils/             ← Processing modules
├── pipeline.py        ← CLI script
├── test_pipeline.py   ← Tests
├── config.py          ← Configuration
└── requirements.txt   ← Dependencies
```

## 🚀 Production Deployment

### Docker (if available)
```bash
docker build -t ancient-script .
docker run -p 8000:8000 ancient-script
```

### Production Server
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 📚 Supported Languages

| Code | Language | Example |
|------|----------|---------|
| en   | English  | "Hello world" |
| la   | Latin    | "Lorem ipsum" |
| el   | Greek    | "Αρχαίο κείμενο" |
| ar   | Arabic   | "النص العربي" |
| he   | Hebrew   | "טקסט עברי" |

## 💡 Tips

1. **Better Results**: Use high-quality, well-lit images
2. **Multiple Languages**: Can detect mixed scripts automatically
3. **Batch Processing**: Process folders of images with Python API
4. **API Integration**: Great for creating custom applications
5. **Performance**: GPU acceleration 5-10x faster than CPU

## 🆘 Getting Help

1. Read the full README.md
2. Check module docstrings: `help(ImagePreprocessor)`
3. Review test_pipeline.py for examples
4. Check error messages carefully

## ✅ Next Steps

After the quick start:

1. Explore the API documentation at `/docs` (if running backend)
2. Try different image types and languages
3. Review the module documentation
4. Integrate into your own projects
5. Fine-tune configuration.py for your needs

---

**Happy decoding! 🏛️**

# ScriptSenseAI - Ancient Script Recognition Platform

ScriptSenseAI is an AI-powered platform for ancient script image processing, OCR, translation, dataset ingestion, and research-grade model experimentation. It includes a FastAPI backend, a polished web frontend, an extensible ML pipeline, dataset upload support, active-learning scaffolding, vector search scaffolding, Docker files, and deployment templates.

Repository:
`https://github.com/purushotham2628/ScriptSenseAI.git`

## What This Project Does

- Upload inscription/manuscript images through the browser.
- Preprocess noisy, low-light, damaged, faded, or scanned images.
- Detect text regions and draw bounding boxes.
- Extract OCR text using EasyOCR-backed processing.
- Clean OCR output.
- Translate Latin to English when the translation model is available.
- Load source language options dynamically from the backend.
- Expose production-style API routes for datasets, inference, training jobs, active learning, and WebSocket progress.
- Provide a scalable backend architecture for future unseen-dataset training and continual learning.

## Project Structure

```text
ScriptSenseAI/
├── backend/
│   ├── app.py                         # Main FastAPI app, frontend serving, /process compatibility route
│   ├── api/                           # Modular API routers
│   ├── core/                          # Settings, logging, JWT/security helpers
│   ├── db/                            # SQLAlchemy database models/session
│   ├── ml/
│   │   ├── ocr/                       # Adaptive OCR pipeline, quality analysis, ensemble OCR, confidence fusion
│   │   ├── inference/                 # API-facing prediction pipeline
│   │   ├── preprocessing/             # Advanced preprocessing utilities
│   │   ├── training/                  # Training scaffolding
│   │   └── embeddings/                # Vector search scaffolding
│   ├── schemas/                       # Pydantic API schemas
│   ├── services/                      # Dataset ingestion and validation
│   └── workers/                       # Celery worker scaffolding
├── frontend/
│   ├── templates/index.html           # Browser UI served at /
│   └── static/
│       ├── style.css                  # Frontend styling and animations
│       └── script.js                  # Upload/process/language loading logic
├── utils/                             # Practical OCR, preprocessing, cleaning, translation utilities
├── infra/k8s/                         # Kubernetes deployment templates
├── Dockerfile                         # CPU Docker image
├── Dockerfile.gpu                     # GPU Docker image template
├── docker-compose.yml                 # API + Postgres + Mongo + Redis + worker stack
├── requirements.txt                   # Local Python dependencies
├── requirements-ocr-optional.txt      # Optional PaddleOCR/Tesseract/Real-ESRGAN/SymSpell/langdetect extras
├── requirements-production.txt        # Production dependency set
├── BACKEND_ARCHITECTURE.md            # Detailed production/research architecture
└── README.md
```

## Requirements

Use these versions for the smoothest setup:

- Python `3.10` recommended
- Git
- pip
- 8 GB RAM recommended
- Internet connection for first dependency/model download
- Optional: Docker Desktop
- Optional: CUDA GPU for future acceleration

## Step-by-Step Setup From GitHub

Follow these commands from a terminal after cloning the repository.

### 1. Clone The Repository

Windows PowerShell, Command Prompt, macOS, or Linux:

```text
git clone https://github.com/purushotham2628/ScriptSenseAI.git
cd ScriptSenseAI
```

If your local folder name is still `Ancient Script`, that is also fine. The commands below should be run from the project root, the folder that contains `backend`, `frontend`, and `requirements.txt`.

### 2. Create And Activate A Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This can take several minutes because the project uses OCR, ML, image processing, and API packages.

Optional research OCR engines:

```bash
pip install -r requirements-ocr-optional.txt
```

These optional packages enable PaddleOCR, Tesseract Python bindings, SymSpell, language detection, and Real-ESRGAN integration when the matching model files or system binaries are available. The app still runs without them and safely falls back to installed engines.

Heavy OCR engines and translation models are disabled by default to avoid long first-run downloads. Enable them only after installing the required packages/models:

Windows PowerShell:

```powershell
$env:SCRIPTSENSE_ENABLE_HEAVY_OCR="1"
$env:SCRIPTSENSE_ENABLE_TRANSLATION_MODELS="1"
```

macOS/Linux:

```bash
export SCRIPTSENSE_ENABLE_HEAVY_OCR=1
export SCRIPTSENSE_ENABLE_TRANSLATION_MODELS=1
```

### 5. Run The Backend And Frontend

Run this from the project root, not from inside `backend/`:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

If port `8000` is already in use, run on another port:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```

You should see output similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

### 6. Open The Frontend

The frontend is served by the FastAPI backend. Open:

```text
http://127.0.0.1:8000/
```

You do not need to start a separate frontend server.

### 7. Verify The Backend Is Running

Open these URLs:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/languages
http://127.0.0.1:8000/api
```

If you used port `8001`, replace `8000` with `8001` in each URL.

Expected health response:

```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

## How To Use The Web App

1. Start the backend with `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload`.
2. Open `http://127.0.0.1:8000/`.
3. Upload an image using drag-and-drop or the choose button.
4. Select a source language.
5. Click `Run Decoder`.
6. Review original, preprocessed, and detection images.
7. Review raw OCR text, cleaned text, translated text, confidence, and statistics.
8. Download results as JSON if needed.

## Supported Image Inputs

The practical browser flow supports common image uploads:

- PNG
- JPG / JPEG
- TIFF / TIF
- BMP
- WEBP if the browser provides it as image data

The production dataset ingestion scaffold also supports ZIP datasets, image folders, CSV/JSON annotations, and scanned PDF hooks.

## Source Languages

The frontend loads source languages from `/languages`. Current options include:

- Auto Detect
- Latin
- English
- Ancient Greek / Greek
- Arabic
- Hebrew
- Sanskrit / Devanagari
- Hindi / Devanagari
- Tamil
- Telugu
- Kannada
- Malayalam
- Bengali
- Chinese
- Japanese
- Korean
- Unknown / Unseen Script

Important note: some languages are best-effort unless the matching EasyOCR model files are available locally. Latin currently uses English-character OCR plus Latin-to-English translation support.

## API Quick Reference

Base URL:

```text
http://127.0.0.1:8000
```

Main frontend-compatible endpoints:

```text
GET  /
GET  /health
GET  /api
GET  /languages
POST /process
```

Production-style modular API endpoints:

```text
POST /api/v1/auth/token
POST /api/v1/auth/register
POST /api/v1/datasets/upload
GET  /api/v1/datasets/{dataset_id}
POST /api/v1/inference/predict
POST /api/v1/training/jobs
GET  /api/v1/training/jobs/{job_id}
POST /api/v1/active-learning/corrections
GET  /api/v1/active-learning/buffer
WS   /api/v1/ws/progress/{job_id}
```

### Example: Process An Image With curl

```bash
curl -X POST \
  -F "file=@test_input_image.png" \
  -F "source_language=la" \
  -F "target_language=en" \
  http://127.0.0.1:8000/process
```

### Example: Process An Image With Python

```python
import requests

with open("test_input_image.png", "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8000/process",
        files={"file": ("test_input_image.png", image_file, "image/png")},
        data={"source_language": "la", "target_language": "en"},
        timeout=120,
    )

print(response.status_code)
print(response.json()["final_output"])
```

## Running With Docker

If Docker Desktop is installed, you can run the full service stack:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000/
```

The Docker stack includes:

- API service
- Background worker scaffold
- PostgreSQL
- MongoDB
- Redis

For simple local testing, the Python virtual environment method is faster and easier.

## Troubleshooting

### Port 8000 Is Already In Use

Windows PowerShell:

```powershell
netstat -ano | Select-String ':8000'
```

Then stop the shown PID if it is an old Uvicorn process:

```powershell
Stop-Process -Id <PID> -Force
```

macOS/Linux:

```bash
lsof -i :8000
kill -9 <PID>
```

### Backend Starts But Browser Shows Old Behavior

Stop the server and restart it:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then hard refresh the browser:

- Windows/Linux: `Ctrl + F5`
- macOS: `Cmd + Shift + R`

### Missing Dependency Error

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

For optional research OCR engines:

```bash
pip install -r requirements-ocr-optional.txt
```

### EasyOCR Model Missing

The helper uses `download_enabled=False` for some OCR reader paths. If a language model is missing, use English/Latin first or initialize/download the needed EasyOCR model manually in Python:

```python
import easyocr
reader = easyocr.Reader(["en"], gpu=False)
```

### Translation Is Slow On First Run

The Latin translation model may download/load on first use. This is normal. Future runs are faster after model caching.

## Development Commands

Compile-check backend Python files:

```bash
python -m compileall backend utils
```

Check frontend JavaScript syntax if Node.js is installed:

```bash
node --check frontend/static/script.js
```

Run the original pipeline tests:

```bash
python test_pipeline.py
```

## Production Architecture

For the full research-grade backend design, read:

```text
BACKEND_ARCHITECTURE.md
```

It documents:

- Dataset ingestion
- Preprocessing
- Augmentation
- Hybrid OCR architecture
- Unseen dataset generalization
- Active learning
- Vector database embeddings
- Training pipeline
- Database schema
- Docker/Kubernetes deployment
- Future SaaS scalability plan

## Future Improvements And Features

Here are strong next features you can add as the project grows:

- Add a curated benchmark suite for ancient manuscripts, inscriptions, palm-leaf scans, stone engravings, coins, seals, and damaged documents.
- Add a dataset labeling UI so users can correct OCR output line by line and feed corrections into active learning.
- Add model fine-tuning jobs for unseen scripts using user-uploaded datasets.
- Add native Tesseract language-pack setup documentation for Latin, Greek, Sanskrit, Arabic, Hebrew, and Indic scripts.
- Add downloadable Real-ESRGAN model setup scripts and GPU acceleration profiles.
- Add per-line OCR comparison views showing EasyOCR, PaddleOCR, Tesseract, and TrOCR outputs side by side.
- Add confidence heatmaps on manuscript images so users can see which regions are unreliable.
- Add script similarity search using vector embeddings for unknown or rare scripts.
- Add human-review workflows for low-confidence predictions before translation is shown.
- Add historical lexicons for Latin, Ancient Greek, Sanskrit transliteration, Brahmi-derived scripts, and epigraphic abbreviations.
- Add translation memory and glossary controls so repeated inscription phrases translate consistently.
- Add batch processing for folders, ZIP uploads, and dataset-level OCR reports.
- Add PDF manuscript ingestion with page splitting, deskewing, and page-level OCR.
- Add export formats such as ALTO XML, PAGE XML, TEI XML, JSONL, CSV, and searchable PDF.
- Add user accounts, project workspaces, private datasets, and audit logs for research teams.
- Add cloud deployment templates for GPU workers, async queues, persistent model cache, and object storage.
- Add monitoring dashboards for OCR latency, engine failure rate, confidence distribution, and translation blocking rate.
- Add automated regression tests using noisy synthetic manuscripts and real scanned samples.
- Add a mobile-friendly capture flow with live blur/skew/readability warnings before upload.
- Add a plugin system for custom OCR models trained by universities or research labs.

## Notes For Contributors

- Do not commit `venv/`, logs, downloaded models, or runtime storage.
- Keep frontend routes compatible with `/process` unless intentionally migrating the UI to `/api/v1/inference/predict`.
- The current browser UI is served by FastAPI, so backend and frontend run together from one command.

## License

This project is provided for educational, research, and prototype SaaS development use.

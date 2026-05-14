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
│   ├── ml/                            # Preprocessing, inference, training, embeddings, active learning
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

### 1. Clone The Repository

```bash
git clone https://github.com/purushotham2628/ScriptSenseAI.git
cd ScriptSenseAI
```

If your local folder name is still `Ancient Script`, that is also fine. The commands below should be run from the project root, the folder that contains `backend`, `frontend`, and `requirements.txt`.

### 2. Create A Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
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

### 5. Run The Backend

Run this from the project root, not from inside `backend/`:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

If `uvicorn` is not recognized, use:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
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

Expected health response:

```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

## How To Use The Web App

1. Start the backend with `uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload`.
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

## Notes For Contributors

- Do not commit `venv/`, logs, downloaded models, or runtime storage.
- Keep frontend routes compatible with `/process` unless intentionally migrating the UI to `/api/v1/inference/predict`.
- The current browser UI is served by FastAPI, so backend and frontend run together from one command.

## License

This project is provided for educational, research, and prototype SaaS development use.
